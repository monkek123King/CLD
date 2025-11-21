import os, yaml, random
import torch
import numpy as np
from typing import Union
import pickle
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from models.mmdit import CustomFluxTransformer2DModel
from models.pipeline import CustomFluxPipeline
from models.multiLayer_adapter import MultiLayerAdapter

def save_checkpoint(transformer, multiLayer_adater, optimizer, optimizer_adapter, scheduler, scheduler_adapter, step, save_dir):
    trans_dir = os.path.join(save_dir, "transformer")
    adapter_dir = os.path.join(save_dir, "adapter")
    os.makedirs(trans_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    flux_transformer_lora_state_dict = get_peft_model_state_dict(transformer)
    flux_adapter_lora_state_dict = get_peft_model_state_dict(multiLayer_adater)

    flux_transformer_lora_state_dict = {k: v.to(torch.float32) for k, v in flux_transformer_lora_state_dict.items()}
    flux_adapter_lora_state_dict = {k: v.to(torch.float32) for k, v in flux_adapter_lora_state_dict.items()}

    CustomFluxPipeline.save_lora_weights(
        os.path.join(trans_dir),
        flux_transformer_lora_state_dict,
        safe_serialization=True,
    )
    CustomFluxPipeline.save_lora_weights(
        os.path.join(adapter_dir),
        flux_adapter_lora_state_dict,
        safe_serialization=True,
    )

    torch.save({"layer_pe": transformer.layer_pe.detach().cpu().to(torch.float32)}, os.path.join(save_dir, "layer_pe.pth"))

    torch.save(optimizer.state_dict(), os.path.join(trans_dir, "optimizer.bin"))
    torch.save(optimizer_adapter.state_dict(), os.path.join(adapter_dir, "optimizer.bin"))

    torch.save(scheduler.state_dict(), os.path.join(trans_dir, "scheduler.bin"))
    torch.save(scheduler_adapter.state_dict(), os.path.join(adapter_dir, "scheduler.bin"))

    save_path = os.path.join(save_dir, f"random_states_0.pkl")
    state = {
        "step": step,
        "random_state": random.getstate(),
        "numpy_random_seed": np.random.get_state(),
        "torch_manual_seed": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()  # list of tensors

    with open(save_path, "wb") as f:
        pickle.dump(state, f)

    print(f"[INFO] Saved RNG states + step {step} to {save_path}")
    

def load_checkpoint(transformer, multiLayer_adater, optimizer, optimizer_adapter, scheduler, scheduler_adapter, ckpt_dir, device="cuda"):
    trans_dir = os.path.join(ckpt_dir, "transformer")
    adapter_dir = os.path.join(ckpt_dir, "adapter")
    start_step = 0

    lora_path = os.path.join(trans_dir, "pytorch_lora_weights.safetensors")
    lora_path_adapter = os.path.join(adapter_dir, "pytorch_lora_weights.safetensors")
    if os.path.exists(lora_path):
        lora_state_dict = CustomFluxPipeline.lora_state_dict(lora_path)
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        print("[INFO] Loaded LoRA weights.")
    if os.path.exists(lora_path_adapter):
        lora_state_dict = CustomFluxPipeline.lora_state_dict(lora_path_adapter)
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, multiLayer_adater)
        print("[INFO] Loaded LoRA weights.")

    pe_path = os.path.join(ckpt_dir, "layer_pe.pth")
    if os.path.exists(pe_path):
        layer_pe = torch.load(pe_path)
        missing_keys, unexpected_keys = transformer.load_state_dict(layer_pe, strict=False)

    opt_path = os.path.join(trans_dir, "optimizer.bin")
    opt_path_adapter = os.path.join(adapter_dir, "optimizer.bin")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        print("[INFO] Loaded optimizer state.")
    if os.path.exists(opt_path_adapter):
        optimizer_adapter.load_state_dict(torch.load(opt_path_adapter, map_location=device))
        print("[INFO] Loaded optimizer state.")

    sch_path = os.path.join(trans_dir, "scheduler.bin")
    sch_path_adapter = os.path.join(adapter_dir, "scheduler.bin")
    if os.path.exists(sch_path):
        scheduler.load_state_dict(torch.load(sch_path, map_location=device))
        print("[INFO] Loaded scheduler state.")
    if os.path.exists(sch_path_adapter):
        scheduler_adapter.load_state_dict(torch.load(sch_path_adapter, map_location=device))
        print("[INFO] Loaded scheduler state.")

    rng_file = None
    for f in os.listdir(ckpt_dir):
        if f.startswith("random_states_") and f.endswith(".pkl"):
            rng_file = os.path.join(ckpt_dir, f)
            break

    if rng_file:
        with open(rng_file, "rb") as f:
            state = pickle.load(f)
        start_step = state.get("step", 0)

        if "random_state" in state:
            random.setstate(state["random_state"])
        if "numpy_random_seed" in state:
            np.random.set_state(state["numpy_random_seed"])
        if "torch_manual_seed" in state:
            torch.set_rng_state(state["torch_manual_seed"])
        if "torch_cuda_manual_seed" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_manual_seed"])

        print(f"[INFO] Resumed RNG states + step {start_step}")

    return start_step


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_input_box(layer_boxes):
    list_layer_box = []
    for layer_box in layer_boxes:
        min_row, max_row = layer_box[1], layer_box[3]
        min_col, max_col = layer_box[0], layer_box[2]
        quantized_min_row = (min_row // 16) * 16
        quantized_min_col = (min_col // 16) * 16
        quantized_max_row = ((max_row // 16) + 1) * 16
        quantized_max_col = ((max_col // 16) + 1) * 16

        list_layer_box.append((quantized_min_col, quantized_min_row, quantized_max_col, quantized_max_row))
    return list_layer_box


def set_lora_into_transformer(
    model: Union[CustomFluxTransformer2DModel, MultiLayerAdapter],
    lora_rank: int,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.1,
):

    target_modules = [
        "to_k", "to_q", "to_v",
        "to_out.0",
        "add_k_proj", "add_q_proj", "add_v_proj",
        "to_add_out",
    ] + [f"single_transformer_blocks.{i}.proj_out" for i in range(model.config.num_single_layers)] + [f"transformer_blocks.{i}.proj_out" for i in range(model.config.num_layers)]

    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    model.add_adapter(transformer_lora_config)
    return model


def build_layer_mask(n_layers, H_lat, W_lat, list_layer_box):
    mask = torch.zeros((n_layers, 1, H_lat, W_lat), dtype=torch.float32)
    for i, box in enumerate(list_layer_box):
        if box is None: 
            continue
        x1, y1, x2, y2 = box
        x1_t, y1_t, x2_t, y2_t = x1 // 8, y1 // 8, x2 // 8, y2 // 8
        x1_t, y1_t = max(0, x1_t), max(0, y1_t)
        x2_t, y2_t = min(W_lat, x2_t), min(H_lat, y2_t)
        if x2_t > x1_t and y2_t > y1_t:
            mask[i, :, y1_t:y2_t, x1_t:x2_t] = 1.0
    return mask


def encode_target_latents(pipeline, pixel_bchw, n_layers, list_layer_box):
    device = pixel_bchw.device
    dtype  = pixel_bchw.dtype

    vae = pipeline.vae.eval()
    bs, n_layers_in, C, H, W = pixel_bchw.shape
    assert n_layers_in == n_layers, f"The number of input layers {n_layers_in} does not match the specified number of layers {n_layers}"

    with torch.no_grad():
        dummy_lat = vae.encode(pixel_bchw[:,0]).latent_dist.sample()
    _, C_lat, H_lat, W_lat = dummy_lat.shape

    x0 = torch.zeros((bs, n_layers, C_lat, H_lat, W_lat), device=device, dtype=dtype)

    with torch.no_grad():
        for i in range(n_layers):
            pixel_i = pixel_bchw[:, i]
            lat = vae.encode(pixel_i).latent_dist.sample()  # [1,C_lat,H_lat,W_lat]
            lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
            x0[:, i] = lat

    latent_ids = pipeline._prepare_latent_image_ids(H_lat, W_lat, list_layer_box, device, dtype)

    return x0, latent_ids


def get_timesteps(pipeline, image_seq_len, num_inference_steps, device):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.base_image_seq_len,
        pipeline.scheduler.config.max_image_seq_len,
        pipeline.scheduler.config.base_shift,
        pipeline.scheduler.config.max_shift,
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler=pipeline.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        sigmas=sigmas,
        mu=mu,
    )

    return timesteps