import torch
import torch.nn as nn
import torchvision
import einops
from collections import OrderedDict
from functools import partial
from typing import Callable
from torch.utils.checkpoint import checkpoint
from diffusers.models.embeddings import apply_rotary_emb, FluxPosEmbed


class MLPBlock(torchvision.ops.misc.MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, freqs_cis):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        B, L, C = input.shape
        x = self.ln_1(input)
        if freqs_cis is not None:
            query = x.view(B, L, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
            query = apply_rotary_emb(query, freqs_cis)
            query = query.transpose(1, 2).reshape(B, L, self.hidden_dim)
        else:
            query = x
        x, _ = self.self_attention(query, query, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        # self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, freqs_cis, use_checkpoint=True):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input # + self.pos_embedding
        x = self.dropout(input)
        # x = checkpoint_sequential(self.layers, len(self.layers), x)
        # x = self.layers(x)
        for l in self.layers:
            x = checkpoint(l, x, freqs_cis) if use_checkpoint else l(x, freqs_cis)
        x = self.ln(x)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, arch='vit-b/32', use_checkpoint=True):
        super().__init__()
        self.arch = arch
        self.use_checkpoint = use_checkpoint

        if self.arch == 'vit-b/32':
            ch = 768
            layers = 12
            heads = 12
        elif self.arch == 'vit-h/14':
            ch = 1280
            layers = 32
            heads = 16
        
        self.encoder = Encoder(
            seq_length=-1,
            num_layers=layers,
            num_heads=heads,
            hidden_dim=ch,
            mlp_dim=ch*4,
            dropout=0.0,
            attention_dropout=0.0,
        )
        self.fc_in = nn.Linear(16, ch)
        self.fc_out = nn.Linear(ch, 256)
        # self.act = nn.Sigmoid()

        if self.arch == 'vit-b/32':
            from torchvision.models.vision_transformer import vit_b_32, ViT_B_32_Weights
            vit = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        elif self.arch == 'vit-h/14':
            from torchvision.models.vision_transformer import vit_h_14, ViT_H_14_Weights
            vit = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)

        missing_keys, unexpected_keys = self.encoder.load_state_dict(vit.encoder.state_dict(), strict=False)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(f"ViT Encoder Missing keys: {missing_keys}")
            print(f"ViT Encoder Unexpected keys: {unexpected_keys}")
        del vit
    
    def forward(self, x, freqs_cis):
        # o = checkpoint(self.fc_in, x)
        o = self.fc_in(x)
        o = self.encoder(o, freqs_cis, self.use_checkpoint)
        o = checkpoint(self.fc_out, o) if self.use_checkpoint else self.fc_out(o)
        # o = self.fc_out(self.encoder(self.fc_in(x), freqs_cis))
        return o


def patchify(x, patch_size=8):
    if len(x.shape) == 4:
        bs, c, h, w = x.shape
        x = einops.rearrange(x, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=patch_size, p2=patch_size)
    elif len(x.shape) == 3:
        c, h, w = x.shape
        x = einops.rearrange(x, "c (h p1) (w p2) -> (c p1 p2) h w", p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, patch_size=8):
    if len(x.shape) == 4:
        bs, c, h, w = x.shape
        x = einops.rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=patch_size, p2=patch_size)
    elif len(x.shape) == 3:
        c, h, w = x.shape
        x = einops.rearrange(x, "(c p1 p2) h w -> c (h p1) (w p2)", p1=patch_size, p2=patch_size)
    return x

def crop_each_layer(hidden_states, use_layers, list_layer_box, H, W, pos_embedding=None):
    token_list = []
    cos_list, sin_list = [], []
    for layer_idx in range(hidden_states.shape[1]):
        if list_layer_box[layer_idx] is None:
            continue
        else:
            x1, y1, x2, y2 = list_layer_box[layer_idx]
            x1, y1, x2, y2 = x1 // 8, y1 // 8, x2 // 8, y2 // 8
            layer_token = hidden_states[:, layer_idx, y1:y2, x1:x2]
            c, h, w = layer_token.shape
            layer_token = layer_token.reshape(c, -1)
            token_list.append(layer_token)
            if pos_embedding is not None:
                ids = prepare_latent_image_ids(-1, H * 2, W * 2, hidden_states.device, hidden_states.dtype)
                ids[:, 0] = use_layers[layer_idx]
                image_rotary_emb = pos_embedding(ids)
                pos_cos, pos_sin = image_rotary_emb[0].reshape(H, W, -1), image_rotary_emb[1].reshape(H, W, -1)
                cos_list.append(pos_cos[y1:y2, x1:x2].reshape(-1, 64))
                sin_list.append(pos_sin[y1:y2, x1:x2].reshape(-1, 64))
    token_list = torch.cat(token_list, dim=1).permute(1, 0)
    if pos_embedding is not None:
        cos_list = torch.cat(cos_list, dim=0)
        sin_list = torch.cat(sin_list, dim=0)
    return token_list, (cos_list, sin_list)


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


class AutoencoderKLTransformerTraining(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.decoder = ViTEncoder(use_checkpoint=self.args.single_layer_decoder is None)
        self.decoder.requires_grad_(True)

        if self.args.pos_embedding == 'rope':
            self.pos_embedding = FluxPosEmbed(theta=10000, axes_dim=(8, 28, 28))
        elif self.args.pos_embedding == 'abs':
            self.pos_embedding = nn.Parameter(torch.empty(16, 1, args.resolution // 8, args.resolution // 8).normal_(std=0.02), requires_grad=True)

        if 'rel' in self.args.layer_embedding or 'abs' in self.args.layer_embedding:
            self.layer_embedding = nn.Parameter(torch.empty(16, 2 + self.args.max_layers, 1, 1).normal_(std=0.02), requires_grad=True)

    def encode(self, x, box, use_layers, z_2d):
        B, C, T, H, W = x.shape     # H W are original image size (In ART, H W are latent size) quesion: why?(It seems no difference)

        z, freqs_cis = [], []
        for b in range(B):
            _z = z_2d[b]
            if 'vit' in self.args.decoder_arch:
                _use_layers = torch.tensor(use_layers[b], device=x.device)
                if 'rel' in self.args.layer_embedding:
                    _use_layers[_use_layers > 2] = 2
                if 'rel' in self.args.layer_embedding or 'abs' in self.args.layer_embedding:
                    _z = _z + self.layer_embedding[:, _use_layers]
                if 'abs' in self.args.pos_embedding:
                    _z = _z + self.pos_embedding
            if 'rope' not in self.args.layer_embedding:
                use_layers[b] = [0] * len(use_layers[b])
            _z, cis = crop_each_layer(_z, use_layers[b], box[b], H, W, self.pos_embedding if self.args.pos_embedding == 'rope' else None)
            # _z, cis = crop_each_layer(_z, use_layers[b], box[b], H // 8, W // 8, self.pos_embedding if self.args.pos_embedding == 'rope' else None)
            z.append(_z)
            freqs_cis.append(cis)

        return z, freqs_cis

    def decode(self, z, freqs_cis, box, H, W):
        B = len(z)
        pad = torch.zeros(4, H, W, device=z[0].device, dtype=z[0].dtype)
        pad[3, :, :] = -1
        x = []
        for b in range(B):
            _x = []
            _freqs_cis = freqs_cis[b] if 'rope' in self.args.pos_embedding else None
            if self.args.single_layer_decoder is None:
                _z = self.decoder(z[b].unsqueeze(0), _freqs_cis).squeeze(0)
            else:
                _z = z[b]
            current_index = 0
            for layer_idx in range(len(box[b])):
                if box[b][layer_idx] == None:
                    _x.append(pad.clone())
                else:
                    x1, y1, x2, y2 = box[b][layer_idx]
                    x1_tok, y1_tok, x2_tok, y2_tok = x1 // 8, y1 // 8, x2 // 8, y2 // 8
                    token_length = (x2_tok - x1_tok) * (y2_tok - y1_tok)
                    tokens = _z[current_index:current_index + token_length]
                    if self.args.single_layer_decoder == 'vit':     # single layer ViT decoder
                        tokens = self.decoder(tokens.unsqueeze(0), (_freqs_cis[0][current_index:current_index + token_length], _freqs_cis[1][current_index:current_index + token_length])).squeeze(0)
                    pixels = einops.rearrange(tokens, "(h w) c -> c h w", h=y2_tok - y1_tok, w=x2_tok - x1_tok)
                    unpatched = unpatchify(pixels)
                    pixels = pad.clone()
                    pixels[:, y1:y2, x1:x2] = unpatched
                    _x.append(pixels)
                    current_index += token_length
            _x = torch.stack(_x, dim=1)
            x.append(_x)
        x = torch.stack(x, dim=0)

        return x
    
    def forward(self, x, box, use_layers, z_2d):
        B, C, T, H, W = x.shape     # H W are original image size (In ART, H W are latent size)
        z, freqs_cis = self.encode(x, box, use_layers, z_2d)
        x_hat = self.decode(z, freqs_cis, box, H, W)
        return x_hat
