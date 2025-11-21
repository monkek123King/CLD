import os, random
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
from prodigyopt import Prodigy
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import FrozenDict

from models.mmdit import CustomFluxTransformer2DModel
from models.pipeline import CustomFluxPipelineCfgLayer, CustomFluxPipeline
from models.multiLayer_adapter import MultiLayerAdapter
from tools.tools import save_checkpoint, load_checkpoint, load_config, seed_everything, get_input_box, set_lora_into_transformer, build_layer_mask, encode_target_latents, get_timesteps
from tools.dataset import LayoutTrainDataset, collate_fn


def train(config_path):
    config = load_config(config_path)
    seed_everything(config.get("seed", 1234))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading pretrained Transformer...", flush=True)
    transformer_orig = FluxTransformer2DModel.from_pretrained(
        config.get('transformer_varient', config['pretrained_model_name_or_path']),
        subfolder="" if 'transformer_varient' in config else "transformer",
        revision=config.get('revision', None),
        variant=config.get('variant', None),
        torch_dtype=torch.bfloat16,
        cache_dir=config.get('cache_dir', None),
    )
    mmdit_config = dict(transformer_orig.config)
    mmdit_config["_class_name"] = "CustomSD3Transformer2DModel"
    mmdit_config["max_layer_num"] = config['max_layer_num']
    mmdit_config = FrozenDict(mmdit_config)

    transformer = CustomFluxTransformer2DModel.from_config(mmdit_config).to(dtype=torch.bfloat16)
    missing_keys, unexpected_keys = transformer.load_state_dict(transformer_orig.state_dict(), strict=False)
    if missing_keys: print(f"[WARN] Missing keys: {missing_keys}")
    if unexpected_keys: print(f"[WARN] Unexpected keys: {unexpected_keys}")

    if 'pretrained_lora_dir' in config:
        print("[INFO] Loading LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['pretrained_lora_dir'])
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()
        print("[INFO] Successfully loaded and fused LoRA weights.", flush=True)

    if 'artplus_lora_dir' in config:
        print("[INFO] Loading artplus LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['artplus_lora_dir'])
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()
        print("[INFO] Successfully loaded and fused artplus LoRA weights.", flush=True)

    # Load MultiLayer-Adapter
    print("[INFO] Loading MultiLayer-Adapter weights...", flush=True)
    multiLayer_adapter = MultiLayerAdapter.from_pretrained(config['pretrained_adapter_path']).to(torch.bfloat16).to(torch.device("cuda"))
    multiLayer_adapter.set_layerPE(transformer.layer_pe, transformer.max_layer_num)
    print("[INFO] Successfully loaded MultiLayer-Adapter weights.", flush=True)

    pipeline = CustomFluxPipelineCfgLayer.from_pretrained(
        config['pretrained_model_name_or_path'],
        transformer=transformer,
        revision=config.get('revision', None),
        variant=config.get('variant', None),
        torch_dtype=torch.bfloat16,
        cache_dir=config.get('cache_dir', None),
    ).to(device)
    pipeline.set_multiLayerAdapter(multiLayer_adapter)
    pipeline.transformer.gradient_checkpointing = True
    pipeline.multiLayerAdapter.gradient_checkpointing = True

    lora_rank = int(config.get("lora_rank", 16))
    lora_alpha = float(config.get("lora_alpha", 16))
    lora_dropout = float(config.get("lora_dropout", 0.0))
    set_lora_into_transformer(pipeline.transformer, lora_rank, lora_alpha, lora_dropout)
    set_lora_into_transformer(pipeline.multiLayerAdapter, lora_rank, lora_alpha, lora_dropout)
    pipeline.transformer.requires_grad_(False)
    pipeline.multiLayerAdapter.requires_grad_(False)
    pipeline.transformer.train()
    pipeline.multiLayerAdapter.train()
    for n, param in pipeline.transformer.named_parameters():
        if 'lora' in n or 'layer_pe' in n:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for n, param in pipeline.multiLayerAdapter.named_parameters():
        if 'lora' in n or 'layer_pe' in n:
            param.requires_grad = True
        else:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in pipeline.transformer.parameters() if p.requires_grad)
    n_trainable_adapter = sum(p.numel() for p in pipeline.multiLayerAdapter.parameters() if p.requires_grad)
    print(f"[INFO] LoRA injected. Transformer Trainable params: {n_trainable/1e6:.2f}M; MultiLayer-Adapter Trainable params: {n_trainable_adapter/1e6:.2f}M", flush=True)

    print("[INFO] Using Prodigy optimizer.", flush=True)
    params = [p for p in pipeline.transformer.parameters() if p.requires_grad]
    params_adapter = [p for p in pipeline.multiLayerAdapter.parameters() if p.requires_grad]
    optimizer = Prodigy(
        params,
        lr=1.0,
        betas=(0.9, 0.999),
        weight_decay=0.001,
        decouple=True,
        safeguard_warmup=True,
        use_bias_correction=True,
    )
    optimizer_adapter = Prodigy(
        params_adapter,
        lr=1.0,
        betas=(0.9, 0.999),
        weight_decay=0.001,
        decouple=True,
        safeguard_warmup=True,
        use_bias_correction=True,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0
    )
    scheduler_adapter = LambdaLR(
        optimizer_adapter,
        lr_lambda=lambda step: 1.0
    )

    dataset = LayoutTrainDataset(data_dir = config['data_dir'], split="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    max_steps = int(config.get("max_steps", 1000))
    log_every = int(config.get("log_every", 50))
    save_every = int(config.get("save_every", 500))
    accum_steps = int(config.get("accum_steps", 1))
    out_dir = config.get("output_dir", "./rf_lora_out")
    os.makedirs(out_dir, exist_ok=True)
    tb_writer = SummaryWriter(out_dir)

    num_inference_steps = config.get("num_inference_steps", 28)

    start_step = 0
    if "resume_from" in config and config["resume_from"] is not None:
        ckpt_dir = config["resume_from"]
        start_step = load_checkpoint(pipeline.transformer, pipeline.multiLayerAdapter, optimizer, optimizer_adapter, scheduler, scheduler_adapter, ckpt_dir, device)
    pbar = tqdm(total=max_steps, desc="train", initial=start_step)
    step = start_step

    while step < max_steps:
        for batch in loader:
            if step >= max_steps: break

            pixel_RGB = batch["pixel_RGB"].to(device=device, dtype=torch.bfloat16)
            pixel_RGB = pipeline.image_processor.preprocess(pixel_RGB[0])
            H = int(batch["height"][0])     # By default, only a single sample per batch is allowed (because later the data will be concatenated based on bounding boxes, which have varying lengths)
            W = int(batch["width"][0])
            adapter_img = batch["whole_img"][0]
            caption = batch["caption"][0]
            layer_boxes = get_input_box(batch["layout"][0])

            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                    prompt=caption,
                    prompt_2=None,
                    num_images_per_prompt=1,
                    max_sequence_length=int(config.get("max_sequence_length", 512)),
                )

                prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)   # (1, 512, 4096)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=torch.bfloat16)     # (1, 768)
                text_ids = text_ids.to(device=device, dtype=torch.bfloat16)  # (512, 3)

                adapter_image, _, _ = pipeline.prepare_image(
                    image=adapter_img,
                    width=W,
                    height=H,
                    batch_size=1,
                    num_images_per_prompt=1,
                    device=device,
                    dtype=pipeline.transformer.dtype,
                )

            x0, latent_image_ids = encode_target_latents(pipeline, pixel_RGB.unsqueeze(0), n_layers=len(layer_boxes), list_layer_box=layer_boxes)
            _, L, C_lat, H_lat, W_lat = x0.shape

            x1 = torch.randn_like(x0)
            image_seq_len = latent_image_ids.shape[0]
            timesteps = get_timesteps(pipeline, image_seq_len=image_seq_len, num_inference_steps=num_inference_steps, device=device)
            t = timesteps[random.randint(0, len(timesteps)-1)].to(device=device, dtype=torch.float32)
            t = t.expand(x0.shape[0]).to(x0.dtype)
            t_b = t.view(1, 1, 1, 1, 1).to(x0.dtype)
            t_b = t_b / 1000.0  # [0,1]
            xt = (1.0 - t_b) * x0 + t_b * x1
            v_star = x1 - x0

            mask = build_layer_mask(L, H_lat, W_lat, layer_boxes).to(device=device, dtype=x0.dtype)  # [L,1,H_lat,W_lat]
            mask = mask.unsqueeze(0)  # [1,L,1,H_lat,W_lat]

            # classifier-free guidance
            guidance_scale=config.get('cfg', 4.0)
            if pipeline.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(x0.shape[0])
            else:
                guidance = None

            pipeline.transformer.train()
            pipeline.multiLayerAdapter.train()

            (
                adapter_block_samples,
                adapter_single_block_samples,
            ) = pipeline.multiLayerAdapter(
                hidden_states=xt,
                list_layer_box=layer_boxes,
                adapter_cond=adapter_image,
                conditioning_scale=config.get("adapter_scale", 1.0),
                timestep=t / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )

            v_pred = pipeline.transformer(
                hidden_states=xt,
                adapter_block_samples=[
                    sample.to(dtype=pipeline.transformer.dtype)
                    for sample in adapter_block_samples
                ],
                adapter_single_block_samples=[
                    sample.to(dtype=pipeline.transformer.dtype)
                    for sample in adapter_single_block_samples
                ] if adapter_single_block_samples is not None else adapter_single_block_samples,
                list_layer_box=layer_boxes,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t / 1000,                 # [0,1]
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]  # [1, L, C, H_lat, W_lat]

            # MSE（masked）
            mse = (v_pred - v_star) ** 2
            mse = mse.mean(dim=2, keepdim=True)  # [1,L,1,H_lat,W_lat]
            loss = (mse * mask).sum() / (mask.sum() + 1e-8)

            loss = loss / accum_steps
            loss.float().backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(config.get("grad_clip", 1.0)))
                optimizer.step()
                optimizer_adapter.step()
                scheduler.step()
                scheduler_adapter.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_adapter.zero_grad(set_to_none=True)

                tb_writer.add_scalar("loss", loss.item(), step)

            step += 1
            if step % log_every == 0:
                pbar.set_postfix(loss=float(loss.detach().cpu()))
                pbar.update(log_every)

            if step % save_every == 0 or step == max_steps:
                save_checkpoint(pipeline.transformer, pipeline.multiLayerAdapter, optimizer, optimizer_adapter, scheduler, scheduler_adapter, step, out_dir)

    pbar.close()
    print("[DONE] Training finished.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()
    train(args.config_path)