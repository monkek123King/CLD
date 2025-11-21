from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.controlnet import BaseOutput, zero_module
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class MultiLayerAdapterOutput(BaseOutput):
    adapter_block_samples: Tuple[torch.Tensor]
    adapter_single_block_samples: Tuple[torch.Tensor]


class MultiLayerAdapter(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

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
        axes_dims_rope: List[int] = [16, 56, 56],
        extra_condition_channels: int = 1 * 4,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            self.controlnet_blocks.append(
                zero_module(nn.Linear(self.inner_dim, self.inner_dim))
            )

        self.controlnet_single_blocks = nn.ModuleList([])
        for _ in range(len(self.single_transformer_blocks)):
            self.controlnet_single_blocks.append(
                zero_module(nn.Linear(self.inner_dim, self.inner_dim))
            )

        self.controlnet_x_embedder = zero_module(
            torch.nn.Linear(in_channels + extra_condition_channels, self.inner_dim)
        )

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self):
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = 4,
        num_single_layers: int = 10,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        load_weights_from_transformer=True,
    ):
        config = transformer.config
        config["num_layers"] = num_layers
        config["num_single_layers"] = num_single_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads

        adapter = cls(**config)

        if load_weights_from_transformer:
            adapter.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            adapter.time_text_embed.load_state_dict(
                transformer.time_text_embed.state_dict()
            )
            adapter.context_embedder.load_state_dict(
                transformer.context_embedder.state_dict()
            )
            adapter.x_embedder.load_state_dict(transformer.x_embedder.state_dict())
            adapter.transformer_blocks.load_state_dict(
                transformer.transformer_blocks.state_dict(), strict=False
            )
            adapter.single_transformer_blocks.load_state_dict(
                transformer.single_transformer_blocks.state_dict(), strict=False
            )

            adapter.controlnet_x_embedder = zero_module(
                adapter.controlnet_x_embedder
            )

        return adapter
    
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
    
    def set_layerPE(self, layerPE, max_layer_num):
        self.layer_pe = layerPE
        self.max_layer_num = max_layer_num

    def forward(
        self,
        hidden_states: torch.Tensor,
        list_layer_box: List[Tuple] = None,
        adapter_cond: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
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
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        bs, n_layers, channel_latent, height, width = hidden_states.shape  # [bs, n_layers, c_latent, h, w]

        hidden_states = hidden_states.view(bs, n_layers, channel_latent, height // 2, 2, width // 2, 2)  # [bs, n_layers, c_latent, h/2, 2, w/2, 2]
        hidden_states = hidden_states.permute(0, 1, 3, 5, 2, 4, 6) # [bs, n_layers, h/2, w/2, c_latent, 2, 2]
        hidden_states = hidden_states.reshape(bs, n_layers, height // 2, width // 2, channel_latent * 4) # [bs, n_layers, h/2, w/2, c_latent*4]
        hidden_states = self.x_embedder(hidden_states)

        adapter_cond = adapter_cond.view(1, height // 2, width // 2, channel_latent * 4 + 4)
        adapter_cond = adapter_cond.unsqueeze(1).expand(-1, n_layers, -1, -1, -1)   # [1, n_layer, 32, 32, 68]

        # add condition
        hidden_states = hidden_states + self.controlnet_x_embedder(adapter_cond)

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

        block_samples = ()
        for _, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                (
                    encoder_hidden_states,
                    hidden_states,
                ) = torch.utils.checkpoint.checkpoint(
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
            block_samples = block_samples + (hidden_states,)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        single_block_samples = ()
        for _, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
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
            single_block_samples = single_block_samples + (
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

        adapter_block_samples = ()
        for block_sample, adapter_block in zip(
            block_samples, self.controlnet_blocks
        ):
            block_sample = adapter_block(block_sample)
            adapter_block_samples = adapter_block_samples + (block_sample,)

        adapter_single_block_samples = ()
        for single_block_sample, adapter_block in zip(
            single_block_samples, self.controlnet_single_blocks
        ):
            single_block_sample = adapter_block(single_block_sample)
            adapter_single_block_samples = adapter_single_block_samples + (
                single_block_sample,
            )

        # scaling
        adapter_block_samples = [
            sample * conditioning_scale for sample in adapter_block_samples
        ]
        adapter_single_block_samples = [
            sample * conditioning_scale for sample in adapter_single_block_samples
        ]

        #
        adapter_block_samples = (
            None if len(adapter_block_samples) == 0 else adapter_block_samples
        )
        adapter_single_block_samples = (
            None
            if len(adapter_single_block_samples) == 0
            else adapter_single_block_samples
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (adapter_block_samples, adapter_single_block_samples)

        return MultiLayerAdapterOutput(
            adapter_block_samples=adapter_block_samples,
            adapter_single_block_samples=adapter_single_block_samples,
        )
