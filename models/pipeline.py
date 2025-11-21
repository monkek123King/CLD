import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
import einops

import torch
import torch.nn as nn
import torchvision.transforms as T

from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import is_torch_xla_available, logging
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps, FluxPipeline

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from models.multiLayer_adapter import MultiLayerAdapter

from PIL import Image

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm # type: ignore
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class CustomFluxPipeline(FluxPipeline):

    @staticmethod
    def _prepare_latent_image_ids(height, width, list_layer_box, device, dtype):

        latent_image_ids_list = []
        for layer_idx in range(len(list_layer_box)):
            if list_layer_box[layer_idx] == None:
                continue
            else:
                latent_image_ids = torch.zeros(height // 2, width // 2, 3) # [h/2, w/2, 3]
                latent_image_ids[..., 0] = layer_idx # use the first dimension for layer representation
                latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
                latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

                x1, y1, x2, y2 = list_layer_box[layer_idx]
                x1, y1, x2, y2 = x1 // 16, y1 // 16, x2 // 16, y2 // 16
                latent_image_ids = latent_image_ids[y1:y2, x1:x2, :]

                latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
                latent_image_ids = latent_image_ids.reshape(
                    latent_image_id_height * latent_image_id_width, latent_image_id_channels
                )

                latent_image_ids_list.append(latent_image_ids)

        full_latent_image_ids = torch.cat(latent_image_ids_list, dim=0)

        return full_latent_image_ids.to(device=device, dtype=dtype)

    def prepare_latents(
        self,
        batch_size,
        num_layers,
        num_channels_latents,
        height,
        width,
        list_layer_box,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // self.vae_scale_factor)     # Here, the vae_scale_factor is 16, but the actual latent size is height // 8, so we need to multiply by 2.
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_layers, num_channels_latents, height, width)  # (1, 15, 16, 64, 64)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) # [bs, f, c_latent, h, w]

        latent_image_ids = self._prepare_latent_image_ids(height, width, list_layer_box, device, dtype)

        return latents, latent_image_ids
    
    def prepare_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
        ):
        # Prepare image
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt
        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)      # (1, C, H, W)

        # create blank mask
        mask = Image.new("RGB", (width, height), (0, 0, 0))     # Currently, the mask is not being used in practice.

        # Prepare mask
        if isinstance(mask, torch.Tensor):
            pass
        else:
            self.mask_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor,
                do_resize=True,
                do_convert_grayscale=True,
                do_normalize=False,
                do_binarize=True,
            )
            mask = self.mask_processor.preprocess(mask, height=height, width=width)
        mask = mask.repeat_interleave(repeat_by, dim=0)
        mask = mask.to(device=device, dtype=dtype)      # (1, 1, H, W)

        # Get masked image
        masked_image = image.clone()
        masked_image[(mask > 0.5).repeat(1, 3, 1, 1)] = -1      # (1, 3, H, W)

        # Encode to latents
        image_latents = self.vae.encode(masked_image.to(self.vae.dtype)).latent_dist.sample()
        image_latents = (
            image_latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        image_latents = image_latents.to(dtype)     # (1, 16, H/8, W/8)

        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor * 2, width // self.vae_scale_factor * 2)
        )
        mask = 1 - mask     # (1, 1, H/8, W/8)

        adapter_image = torch.cat([image_latents, mask], dim=1)

        # Pack cond latents
        packed_adapter_image = self._pack_latents(
            adapter_image,
            batch_size * num_images_per_prompt,
            adapter_image.shape[1],
            adapter_image.shape[2],
            adapter_image.shape[3],
        )
        
        if do_classifier_free_guidance:
            packed_adapter_image = torch.cat([packed_adapter_image] * 2)

        return packed_adapter_image, height, width
    
    def set_multiLayerAdapter(self, multiLayerAdapter):
        self.multiLayerAdapter = multiLayerAdapter

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        validation_box: List[tuple] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        num_layers: int = 5,
        sdxl_vae: nn.Module = None,
        transparent_decoder: nn.Module = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_layers,
            num_channels_latents,
            height,
            width,
            validation_box,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latent_image_ids.shape[0] # ???
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    list_layer_box=validation_box,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # create a grey latent
        bs, n_frames, channel_latent, height, width = latents.shape

        pixel_grey = torch.zeros(size=(bs*n_frames, 3, height*8, width*8), device=latents.device, dtype=latents.dtype)
        latent_grey = self.vae.encode(pixel_grey).latent_dist.sample()
        latent_grey = (latent_grey - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latent_grey = latent_grey.view(bs, n_frames, channel_latent, height, width)  # [bs, f, c_latent, h, w]
        
        # fill in the latents
        for layer_idx in range(latent_grey.shape[1]):
            x1, y1, x2, y2 = validation_box[layer_idx]
            x1, y1, x2, y2 = x1 // 8, y1 // 8, x2 // 8, y2 // 8
            latent_grey[:, layer_idx, :, y1:y2, x1:x2] = latents[:, layer_idx, :, y1:y2, x1:x2]
        latents = latent_grey

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.reshape(bs * n_frames, channel_latent, height, width)
            image = self.vae.decode(latents, return_dict=False)[0]
            if sdxl_vae is not None:
                sdxl_vae = sdxl_vae.to(dtype=image.dtype, device=image.device)
                sdxl_latents = sdxl_vae.encode(image).latent_dist.sample()
                transparent_decoder = transparent_decoder.to(dtype=image.dtype, device=image.device)
                result_list, vis_list = transparent_decoder(sdxl_vae, sdxl_latents)
            else:
                result_list, vis_list = None, None
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, result_list, vis_list)

        return FluxPipelineOutput(images=image), result_list, vis_list


class CustomFluxPipelineCfgLayer(CustomFluxPipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        validation_box: List[tuple] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        true_gs: float = 3.5,
        adapter_image: PipelineImageInput = None,
        adapter_mask: PipelineImageInput = None,
        adapter_conditioning_scale: Union[float, List[float]] = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        num_layers: int = 5,
        sdxl_vae: nn.Module = None,
        transparent_decoder: nn.Module = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        (
            neg_prompt_embeds,
            neg_pooled_prompt_embeds,
            neg_text_ids,
        ) = self.encode_prompt(
            prompt="",
            prompt_2=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 3. Prepare image
        num_channels_latents = self.transformer.config.in_channels // 4
        if isinstance(self.multiLayerAdapter, MultiLayerAdapter):
            adapter_image, _, _ = self.prepare_image(
                image=adapter_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.transformer.dtype,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_layers,
            num_channels_latents,
            height,
            width,
            validation_box,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latent_image_ids.shape[0]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                (
                    adapter_block_samples,
                    adapter_single_block_samples,
                ) = self.multiLayerAdapter(
                    hidden_states=latents,
                    list_layer_box=validation_box,
                    adapter_cond=adapter_image,
                    conditioning_scale=adapter_conditioning_scale,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )

                noise_pred = self.transformer(
                    hidden_states=latents,
                    list_layer_box=validation_box,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    adapter_block_samples=[
                        sample.to(dtype=self.transformer.dtype)
                        for sample in adapter_block_samples
                    ],
                    adapter_single_block_samples=[
                        sample.to(dtype=self.transformer.dtype)
                        for sample in adapter_single_block_samples
                    ] if adapter_single_block_samples is not None else adapter_single_block_samples,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    list_layer_box=validation_box,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=neg_pooled_prompt_embeds,
                    encoder_hidden_states=neg_prompt_embeds,
                    adapter_block_samples=[
                        sample.to(dtype=self.transformer.dtype)
                        for sample in adapter_block_samples
                    ],
                    adapter_single_block_samples=[
                        sample.to(dtype=self.transformer.dtype)
                        for sample in adapter_single_block_samples
                    ] if adapter_single_block_samples is not None else adapter_single_block_samples,
                    txt_ids=neg_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                noise_pred = neg_noise_pred + true_gs * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # create a grey latent
        bs, n_frames, channel_latent, height, width = latents.shape

        def encode_in_chunks(vae, images, chunk=8):
            parts = []
            for i in range(0, images.shape[0], chunk):
                chunk_img = images[i : i + chunk]
                part_latent = vae.encode(chunk_img).latent_dist.sample()
                parts.append(part_latent)
                torch.cuda.empty_cache()
            return torch.cat(parts, dim=0)

        pixel_grey = torch.zeros(size=(bs*n_frames, 3, height*8, width*8), device=latents.device, dtype=latents.dtype)
        # latent_grey = self.vae.encode(pixel_grey).latent_dist.sample()
        latent_grey = encode_in_chunks(self.vae, pixel_grey, chunk=16)
        latent_grey = (latent_grey - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latent_grey = latent_grey.view(bs, n_frames, channel_latent, height, width)  # [bs, f, c_latent, h, w]
        
        # fill in the latents
        for layer_idx in range(latent_grey.shape[1]):
            if validation_box[layer_idx] == None:
                continue
            x1, y1, x2, y2 = validation_box[layer_idx]
            x1, y1, x2, y2 = x1 // 8, y1 // 8, x2 // 8, y2 // 8
            latent_grey[:, layer_idx, :, y1:y2, x1:x2] = latents[:, layer_idx, :, y1:y2, x1:x2]
        latents = latent_grey

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            bs, num_layers, c, h, w = latents.shape
            latents = latents.reshape(bs * n_frames, channel_latent, height, width)
            latents_segs = torch.split(latents, 8, dim=0)
            image_segs = [self.vae.decode(latents_seg, return_dict=False)[0] for latents_seg in latents_segs]
            image = torch.cat(image_segs, dim=0)
            if sdxl_vae is not None:
                sdxl_vae = sdxl_vae.to(dtype=image.dtype, device=image.device)

                # Prepare input parameters
                _, c1, h1, w1 = image.shape  # Get channels and spatial dimensions from image
                x = image.view(bs, num_layers, c1, h1, w1).permute(0, 2, 1, 3, 4).to(image.device)  # Reshape to (bs, c, num_layers, h, w)
                box = [validation_box] * bs  # Create box info for each sample
                use_layers = [list(range(len(b))) for b in box]  # Use all layers
                z_2d = latents.view(bs, num_layers, -1, h, w)  # Reshape to (bs, num_layers, c, h, w)
                z_2d = einops.rearrange(z_2d, "b t c h w -> b c t h w").to(image.device)  # Reshape to (bs, c, num_layers, h, w)

                # Call transparent VAE decoder
                x_hat = sdxl_vae(x, box, use_layers, z_2d).to(x.dtype).clamp(-1, 1)
            else:
                result_list, vis_list = None, None
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return (
            x_hat,           # Final decoded result including foreground and transparency
            image,           # Final generated RGB image
            latents          # Latent variables
        )
