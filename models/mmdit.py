import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np

from accelerate.utils import set_module_tensor_to_device
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock

from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CustomFluxTransformer2DModel(FluxTransformer2DModel):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        max_layer_num: int = 52,
    ):
        super(FluxTransformer2DModel, self).__init__()
        self.out_channels = in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.max_layer_num = max_layer_num

        # the following process ensures self.layer_pe is not created as a meta tensor
        layer_pe_value = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(
                1, self.max_layer_num, 1, 1, self.inner_dim,
            )),
            mean=0.0, std=0.02, a=-2.0, b=2.0,
        ).data.detach()
        self.layer_pe = nn.Parameter(layer_pe_value)
        set_module_tensor_to_device(
            self, 
            'layer_pe', 
            device='cpu',
            value=layer_pe_value,
            dtype=layer_pe_value.dtype,
        )

    @classmethod
    def from_pretrained(cls, *args, **kwarg):
        model = super().from_pretrained(*args, **kwarg)
        for name, para in model.named_parameters():
            if name != 'layer_pe':
                device = para.device
                break
        model.layer_pe.to(device)
        return model

    def crop_each_layer(self, hidden_states, list_layer_box):
        """
            hidden_states: [1, n_layers, h, w, inner_dim]
            list_layer_box: List, length=n_layers, each element is a Tuple of 4 elements (x1, y1, x2, y2)
        """
        token_list = []
        for layer_idx in range(hidden_states.shape[1]):
            if list_layer_box[layer_idx] == None:
                continue
            else:
                x1, y1, x2, y2 = list_layer_box[layer_idx]
                x1, y1, x2, y2 = x1 // 16, y1 // 16, x2 // 16, y2 // 16
                layer_token = hidden_states[:, layer_idx, y1:y2, x1:x2, :]
                bs, h, w, c = layer_token.shape
                layer_token = layer_token.reshape(bs, -1, c)
                token_list.append(layer_token)
        result = torch.cat(token_list, dim=1)
        return result

    def fill_in_processed_tokens(self, hidden_states, full_hidden_states, list_layer_box):
        """
            hidden_states: [1, h1xw1 + h2xw2 + ... + hlxwl , inner_dim]
            full_hidden_states: [1, n_layers, h, w, inner_dim]
            list_layer_box: List, length=n_layers, each element is a Tuple of 4 elements (x1, y1, x2, y2)
        """
        used_token_len = 0
        bs = hidden_states.shape[0]
        for layer_idx in range(full_hidden_states.shape[1]):
            if list_layer_box[layer_idx] == None:
                continue
            else:
                x1, y1, x2, y2 = list_layer_box[layer_idx]
                x1, y1, x2, y2 = x1 // 16, y1 // 16, x2 // 16, y2 // 16
                full_hidden_states[:, layer_idx, y1:y2, x1:x2, :] = hidden_states[:, used_token_len: used_token_len + (y2-y1) * (x2-x1), :].reshape(bs, y2-y1, x2-x1, -1)
                used_token_len = used_token_len + (y2-y1) * (x2-x1)
        return full_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        list_layer_box: List[Tuple] = None,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        adapter_block_samples=None,
        adapter_single_block_samples=None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        bs, n_layers, channel_latent, height, width = hidden_states.shape  # [bs, n_layers, c_latent, h, w]

        hidden_states = hidden_states.view(bs, n_layers, channel_latent, height // 2, 2, width // 2, 2)  # [bs, n_layers, c_latent, h/2, 2, w/2, 2]
        hidden_states = hidden_states.permute(0, 1, 3, 5, 2, 4, 6) # [bs, n_layers, h/2, w/2, c_latent, 2, 2]
        hidden_states = hidden_states.reshape(bs, n_layers, height // 2, width // 2, channel_latent * 4) # [bs, n_layers, h/2, w/2, c_latent*4]
        hidden_states = self.x_embedder(hidden_states) # [bs, n_layers, h/2, w/2, inner_dim]

        full_hidden_states = torch.zeros_like(hidden_states) # [bs, n_layers, h/2, w/2, inner_dim]
        layer_pe = self.layer_pe.view(1, self.max_layer_num, 1, 1, self.inner_dim)  # [1, max_n_layers, 1, 1, inner_dim]
        hidden_states = hidden_states + layer_pe[:, :n_layers]    # [bs, n_layers, h/2, w/2, inner_dim] + [1, n_layers, 1, 1, inner_dim] -->  [bs, f, h/2, w/2, inner_dim]
        hidden_states = self.crop_each_layer(hidden_states, list_layer_box)  # [bs, token_len, inner_dim]

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

            # adapter residual
            if adapter_block_samples is not None:
                interval_adapter = len(self.transformer_blocks) / len(
                    adapter_block_samples
                )
                interval_adapter = int(np.ceil(interval_adapter))
                hidden_states = (
                    hidden_states
                    + adapter_block_samples[index_block // interval_adapter]
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

            # adapter residual
            if adapter_single_block_samples is not None:
                interval_adapter = len(self.single_transformer_blocks) / len(
                    adapter_single_block_samples
                )
                interval_adapter = int(np.ceil(interval_adapter))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + adapter_single_block_samples[index_block // interval_adapter]
                )


        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        
        hidden_states = self.fill_in_processed_tokens(hidden_states, full_hidden_states, list_layer_box)  # [bs, n_layers, h/2, w/2, inner_dim]
        hidden_states = hidden_states.view(bs, -1, self.inner_dim)  # [bs, n_layers * full_len, inner_dim]

        hidden_states = self.norm_out(hidden_states, temb) # [bs, n_layers * full_len, inner_dim]
        hidden_states = self.proj_out(hidden_states) # [bs, n_layers * full_len, c_latent*4]

        # unpatchify
        hidden_states = hidden_states.view(bs, n_layers, height//2, width//2, channel_latent, 2, 2) # [bs, n_layers, h/2, w/2, c_latent, 2, 2]
        hidden_states = hidden_states.permute(0, 1, 4, 2, 5, 3, 6)
        output = hidden_states.reshape(bs, n_layers, channel_latent, height, width)  # [bs, n_layers, c_latent, h, w]

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)