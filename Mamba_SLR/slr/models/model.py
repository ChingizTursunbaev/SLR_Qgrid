# slr/models/model.py (top of file)
try:
    from torchaudio.models.decoder import CUCTCDecoder  # optional, only if you use it
except Exception:
    CUCTCDecoder = None


import copy
import math
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, lecun_normal_

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models.decoder import CUCTCDecoder

from mamba_ssm.modules.mha import MHA
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

from .temporal_layer import TemporalLayer
from .spatial_layer import SpatialLayer
from .ffn import FeedForwardNN
from .stem import Stem


class Block(nn.Module):
    def __init__(
        self, dim, spatial_mixer_cls, temporal_mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if spatial_mixer_cls is not nn.Identity:
            self.spatial_mixer = spatial_mixer_cls(dim)
            self.spatial_norm = norm_cls(dim)
        else:
            self.spatial_mixer = None
        if temporal_mixer_cls is not nn.Identity:
            self.temporal_mixer = temporal_mixer_cls(dim)
            self.temporal_norm = norm_cls(dim)
        else:
            self.temporal_mixer = None
        if mlp_cls is not nn.Identity:
            self.mlp = mlp_cls(dim)
            self.mlp_norm = norm_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.spatial_norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, seq_idx: Tensor | None, residual: Tensor | None = None
    ):
        BT, L, D = hidden_states.shape

        if self.spatial_mixer is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual if residual is not None else hidden_states
                hidden_states = self.spatial_norm(residual.to(dtype=self.spatial_norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.spatial_norm.weight,
                    self.spatial_norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.spatial_norm.eps,
                    is_rms_norm=isinstance(self.spatial_norm, RMSNorm)
                )
            hidden_states = self.spatial_mixer(hidden_states)
            hidden_states = self.drop_path(hidden_states)
            
            if self.temporal_mixer is not None:
                hidden_states = hidden_states.transpose(0, 1).contiguous()
                residual = residual.transpose(0, 1).contiguous() if residual is not None else None
                if seq_idx is not None:
                    seq_idx = seq_idx.unsqueeze(0).expand(L, -1)

        if self.temporal_mixer is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual if residual is not None else hidden_states
                hidden_states = self.temporal_norm(residual.to(dtype=self.temporal_norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.temporal_norm.weight,
                    self.temporal_norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.temporal_norm.eps,
                    is_rms_norm=isinstance(self.temporal_norm, RMSNorm)
                )
            hidden_states = self.temporal_mixer(hidden_states, seq_idx)
            hidden_states = self.drop_path(hidden_states)

            hidden_states = hidden_states.transpose(0, 1).contiguous()
            residual = residual.transpose(0, 1).contiguous() if residual is not None else None

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.mlp_norm(residual.to(dtype=self.mlp_norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.mlp_norm.weight,
                    self.mlp_norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.mlp_norm.eps,
                    is_rms_norm=isinstance(self.mlp_norm, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.drop_path(hidden_states)

        return hidden_states, residual

def create_block(
    d_model, d_intermediate, ssm_cfg=None, attn_layer_idx=None, attn_cfg=None, drop_path=0., norm_epsilon=1e-5,
    rms_norm=True, residual_in_fp32=True, fused_add_norm=True, layer_idx=None, device=None, dtype=None,
):
    # ... (This function remains unchanged)
    if ssm_cfg is None:
        ssm_cfg = {'spatial': {}, 'temporal': {}}
    if attn_layer_idx is None:
        attn_layer_idx = {'spatial': [], 'temporal': []}
    if attn_cfg is None:
        attn_cfg = {'spatial': {}, 'temporal': {}}
    factory_kwargs = {"device": device, "dtype": dtype}
    
    if layer_idx not in attn_layer_idx['spatial']:
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        spatial_partial_layer = SpatialLayer
        spatial_mixer_cls = partial(
            spatial_partial_layer, layer_idx=layer_idx, **ssm_cfg['spatial'], **factory_kwargs
        )
    else:
        spatial_mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg['spatial'], **factory_kwargs)
    if layer_idx not in attn_layer_idx['temporal']:
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        temporal_partial_layer = TemporalLayer
        temporal_mixer_cls = partial(
            temporal_partial_layer, layer_idx=layer_idx, **ssm_cfg['temporal'], **factory_kwargs
        )
    else:
        temporal_mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg['temporal'], **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            FeedForwardNN, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model, spatial_mixer_cls, temporal_mixer_cls, mlp_cls, drop_path=drop_path, norm_cls=norm_cls,
        fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

def _init_weights(
    module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=2,
):
    # ... (This function remains unchanged)
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def segm_init_weights(m):
    # ... (This function remains unchanged)
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class Model(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 512,
        n_layer: int = 12,
        d_intermediate: int = 512 * 4,
        channels: int = 3,
        gloss_dict: dict = None, # Made optional
        drop_path_rate: float = .1,
        head_drop_rate: float = .1,
        ssm_cfg: dict[str, dict] | None = None,
        attn_layer_idx: dict[str, list] | None = None,
        attn_cfg: dict[str, dict] | None = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.patch_embed = Stem(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=channels,
            embed_dim=d_model,
            norm_layer=partial(nn.LayerNorm, eps=norm_epsilon),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
            
        self.layers = nn.ModuleList([
            create_block(
                d_model, d_intermediate=d_intermediate, ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx,
                attn_cfg=attn_cfg, drop_path=dpr[i], norm_epsilon=norm_epsilon, rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm, layer_idx=i, **factory_kwargs
            )
            for i in range(n_layer)
        ])

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        
        # We remove the head and decoder logic as this will now be a feature encoder
        
        self.patch_embed.apply(segm_init_weights)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(
            partial(
                _init_weights, n_layer=n_layer, **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=2 if d_intermediate == 0 else 3,
            )
        )

    def forward_features(self, x: Tensor, seq_idx: Tensor | None = None):
        B_T, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B_T, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, seq_idx, residual)
        if not self.fused_add_norm:
            residual = hidden_states + residual if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states, self.norm_f.weight, self.norm_f.bias, eps=self.norm_f.eps, residual=residual,
                prenorm=False, residual_in_fp32=self.residual_in_fp32, is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        # Return features for all patches, excluding the CLS token
        return hidden_states[:, 1:, :]

    def forward(self, x: Tensor):
        """
        This is the new forward pass for the feature encoder.
        Input x: A (B*T, C, H, W) tensor of frames.
        Returns: A (B*T, D) tensor of features.
        """
        features = self.forward_features(x)
        # Average the patch features to get a single feature vector per frame
        return features.mean(dim=1)
















# import copy
# import math
# from functools import partial
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_, lecun_normal_

# import torch
# import torch.nn as nn
# from torch import Tensor
# from torch.nn.utils.rnn import pad_sequence
# from torchaudio.models.decoder import CUCTCDecoder

# from mamba_ssm.modules.mha import MHA
# from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

# try:
#     from .spatial_layer import SpatialLayer
#     from .temporal_layer import TemporalLayer
#     from .ffn import FeedForwardNN
#     from .stem import Stem
# except:
#     from spatial_layer import SpatialLayer
#     from temporal_layer import TemporalLayer
#     from ffn import FeedForwardNN
#     from stem import Stem


# class Block(nn.Module):
#     def __init__(
#         self, dim, spatial_mixer_cls, temporal_mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
#     ):
#         """
#         Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

#         This Block has a slightly different structure compared to a regular
#         prenorm Transformer block.
#         The standard block is: LN -> MHA/MLP -> Add.
#         [Ref: https://arxiv.org/abs/2002.04745]
#         Here we have: Add -> LN -> Mixer, returning both
#         the hidden_states (output of the mixer) and the residual.
#         This is purely for performance reasons, as we can fuse add and LayerNorm.
#         The residual needs to be provided (except for the very first block).
#         """
#         super().__init__()
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         if spatial_mixer_cls is not nn.Identity:
#             self.spatial_mixer = spatial_mixer_cls(dim)
#             self.spatial_norm = norm_cls(dim)
#         else:
#             self.spatial_mixer = None
#         if temporal_mixer_cls is not nn.Identity:
#             self.temporal_mixer = temporal_mixer_cls(dim)
#             self.temporal_norm = norm_cls(dim)
#         else:
#             self.temporal_mixer = None
#         if mlp_cls is not nn.Identity:
#             self.mlp = mlp_cls(dim)
#             self.mlp_norm = norm_cls(dim)
#         else:
#             self.mlp = None
#         if self.fused_add_norm:
#             assert RMSNorm is not None, "RMSNorm import fails"
#             assert isinstance(
#                 self.spatial_norm, (nn.LayerNorm, RMSNorm)
#             ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

#     def forward(
#             self, hidden_states: Tensor, seq_idx: Tensor, residual: Tensor | None = None
#     ):
#         r"""Pass the input through the encoder layer.

#         Args:
#             hidden_states: the sequence to the encoder layer (required).
#             seq_idx: the index of sequence to the encoder layer (required).
#             residual: hidden_states = Mixer(LN(residual))
#         """
#         BT, L, D = hidden_states.shape

#         if self.spatial_mixer is not None:
#             if not self.fused_add_norm:
#                 residual = hidden_states + residual if residual is not None else hidden_states
#                 hidden_states = self.spatial_norm(residual.to(dtype=self.spatial_norm.weight.dtype))
#                 if self.residual_in_fp32:
#                     residual = residual.to(torch.float32)
#             else:
#                 hidden_states, residual = layer_norm_fn(
#                     hidden_states,
#                     self.spatial_norm.weight,
#                     self.spatial_norm.bias,
#                     residual=residual,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     eps=self.spatial_norm.eps,
#                     is_rms_norm=isinstance(self.spatial_norm, RMSNorm)
#                 )
#             hidden_states = self.spatial_mixer(hidden_states)
#             hidden_states = self.drop_path(hidden_states)
            
#             # (BT, HW + 1, D) -> (HW + 1, BT, D)
#             hidden_states = hidden_states.transpose(0, 1).contiguous()
#             residual = residual.transpose(0, 1).contiguous() if residual is not None else None
#             seq_idx = seq_idx.unsqueeze(0).expand(L, -1)

#         if self.temporal_mixer is not None:
#             if not self.fused_add_norm:
#                 residual = hidden_states + residual if residual is not None else hidden_states
#                 hidden_states = self.temporal_norm(residual.to(dtype=self.temporal_norm.weight.dtype))
#                 if self.residual_in_fp32:
#                     residual = residual.to(torch.float32)
#             else:
#                 hidden_states, residual = layer_norm_fn(
#                     hidden_states,
#                     self.temporal_norm.weight,
#                     self.temporal_norm.bias,
#                     residual=residual,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     eps=self.temporal_norm.eps,
#                     is_rms_norm=isinstance(self.temporal_norm, RMSNorm)
#                 )
#             hidden_states = self.temporal_mixer(hidden_states, seq_idx)
#             hidden_states = self.drop_path(hidden_states)

#             # (HW + 1, BT, D) -> (BT, HW + 1, D)
#             hidden_states = hidden_states.transpose(0, 1).contiguous() # # (BT, HW + 1, D)
#             residual = residual.transpose(0, 1).contiguous() if residual is not None else None

#         if self.mlp is not None:
#             if not self.fused_add_norm:
#                 residual = hidden_states + residual
#                 hidden_states = self.mlp_norm(residual.to(dtype=self.mlp_norm.weight.dtype))
#                 if self.residual_in_fp32:
#                     residual = residual.to(torch.float32)
#             else:
#                 hidden_states, residual = layer_norm_fn(
#                     hidden_states,
#                     self.mlp_norm.weight,
#                     self.mlp_norm.bias,
#                     residual=residual,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     eps=self.mlp_norm.eps,
#                     is_rms_norm=isinstance(self.mlp_norm, RMSNorm)
#                 )
#             hidden_states = self.mlp(hidden_states)
#             hidden_states = self.drop_path(hidden_states)

#         return hidden_states, residual

# def create_block(
#     d_model,
#     d_intermediate,
#     ssm_cfg=None,
#     attn_layer_idx=None,
#     attn_cfg=None,
#     drop_path=0.,
#     norm_epsilon=1e-5,
#     rms_norm=True,
#     residual_in_fp32=True,
#     fused_add_norm=True,
#     layer_idx=None,
#     device=None,
#     dtype=None,
# ):
#     if ssm_cfg is None:
#         ssm_cfg = {'spatial': {}, 'temporal': {}}
#     if attn_layer_idx is None:
#         attn_layer_idx = {'spatial': [], 'temporal': []}
#     if attn_cfg is None:
#         attn_cfg = {'spatial': {}, 'temporal': {}}
#     factory_kwargs = {"device": device, "dtype": dtype}
    
#     if layer_idx not in attn_layer_idx['spatial']:
#         ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
#         spatial_partial_layer = SpatialLayer
#         spatial_mixer_cls = partial(
#             spatial_partial_layer,
#             layer_idx=layer_idx,
#             **ssm_cfg['spatial'],
#             **factory_kwargs
#         )
#     else:
#         spatial_mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg['spatial'], **factory_kwargs)
#     if layer_idx not in attn_layer_idx['temporal']:
#         ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
#         temporal_partial_layer = TemporalLayer
#         temporal_mixer_cls = partial(
#             temporal_partial_layer,
#             layer_idx=layer_idx,
#             **ssm_cfg['temporal'],
#             **factory_kwargs
#         )
#     else:
#         temporal_mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg['temporal'], **factory_kwargs)
#     norm_cls = partial(
#         nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
#     )
#     if d_intermediate == 0:
#         mlp_cls = nn.Identity
#     else:
#         mlp_cls = partial(
#             FeedForwardNN, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
#         )
#     block = Block(
#         d_model,
#         spatial_mixer_cls,
#         temporal_mixer_cls,
#         mlp_cls,
#         drop_path=drop_path,
#         norm_cls=norm_cls,
#         fused_add_norm=fused_add_norm,
#         residual_in_fp32=residual_in_fp32,
#     )
#     block.layer_idx = layer_idx
#     return block

# # https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
# def _init_weights(
#     module,
#     n_layer,
#     initializer_range=0.02,  # Now only used for embedding layer.
#     rescale_prenorm_residual=True,
#     n_residuals_per_layer=2,  # Change to 3 if we have MLP
# ):
#     if isinstance(module, nn.Linear):
#         if module.bias is not None:
#             if not getattr(module.bias, "_no_reinit", False):
#                 nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Embedding):
#         nn.init.normal_(module.weight, std=initializer_range)

#     if rescale_prenorm_residual:
#         # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
#         #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
#         #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
#         #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
#         #
#         # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
#         for name, p in module.named_parameters():
#             if name in ["out_proj.weight", "fc2.weight"]:
#                 # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
#                 # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
#                 # We need to reinit p since this code could be called multiple times
#                 # Having just p *= scale would repeatedly scale it down
#                 nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#                 with torch.no_grad():
#                     p /= math.sqrt(n_residuals_per_layer * n_layer)

# # https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py
# def segm_init_weights(m):
#     if isinstance(m, nn.Linear):
#         trunc_normal_(m.weight, std=0.02)
#         if isinstance(m, nn.Linear) and m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Conv2d):
#         # NOTE conv was left to pytorch default in my original init
#         lecun_normal_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
#         nn.init.zeros_(m.bias)
#         nn.init.ones_(m.weight)

# # Patch Embedding -> Blocks(spatial -> temporal -> mlp) -> Head
# class Model(nn.Module):
#     def __init__(
#         self,
#         img_size: int = 224,
#         patch_size: int = 16,
#         d_model: int = 512,
#         n_layer: int = 12,
#         d_intermediate: int = 512 * 4,
#         channels: int = 3,
#         gloss_dict: dict = {},
#         drop_path_rate: float = .1,
#         head_drop_rate: float = .1,
#         ssm_cfg: dict[str, dict] | None = None,
#         attn_layer_idx: dict[str, list] | None = None,
#         attn_cfg: dict[str, dict] | None = None,
#         norm_epsilon: float = 1e-5,
#         rms_norm: bool = False,
#         initializer_cfg=None,
#         fused_add_norm=True,
#         residual_in_fp32=True,
#         device=None,
#         dtype=None
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.num_classes = len(gloss_dict) + 1 # +1 for <blank> token
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm

#         ssm_cfg['spatial'].update({'img_size': img_size, 'patch_size': patch_size})

#         self.patch_embed = Stem(
#             img_size=img_size, patch_size=patch_size, in_chans=channels, out_chans=d_model,
#             norm_cls=nn.LayerNorm, norm_epsilon=norm_epsilon, **factory_kwargs
#         )
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]

#         # We change the order of residual and layer norm:
#         # Instead of LN -> Attn / MLP -> Add, we do:
#         # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
#         # the main branch (output of MLP / Mixer). The model definition is unchanged.
#         # This is for performance reason: we can fuse add + layer_norm.
#         if self.fused_add_norm:
#             if layer_norm_fn is None or rms_norm_fn is None:
#                 raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
            
#         self.layers = nn.ModuleList([
#             create_block(
#                 d_model,
#                 d_intermediate=d_intermediate,
#                 ssm_cfg=ssm_cfg,
#                 attn_layer_idx=attn_layer_idx,
#                 attn_cfg=attn_cfg,
#                 drop_path=dpr[i],
#                 norm_epsilon=norm_epsilon,
#                 rms_norm=rms_norm,
#                 residual_in_fp32=residual_in_fp32,
#                 fused_add_norm=fused_add_norm,
#                 layer_idx=i,
#                 **factory_kwargs
#             )
#             for i in range(n_layer)
#         ])

#         self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
#             d_model, eps=norm_epsilon, **factory_kwargs
#         )
        
#         self.head_drop = nn.Dropout(head_drop_rate) if head_drop_rate > 0 else nn.Identity()
#         self.head = nn.Linear(2 * self.d_model, self.num_classes) if self.num_classes > 0 else nn.Identity()

#         glosses = self.build_tokens(gloss_dict)
#         assert glosses[0] == '<blank>'
#         assert len(glosses) == self.num_classes
#         self.decoder = CUCTCDecoder(
#             vocab_list=glosses,
#             blank_id=0,
#             beam_size=5
#         )

#         # init
#         self.patch_embed.apply(segm_init_weights)
#         self.head.apply(segm_init_weights)
#         trunc_normal_(self.cls_token, std=.02)
#         self.apply(
#             partial(
#                 _init_weights,
#                 n_layer=n_layer,
#                 **(initializer_cfg if initializer_cfg is not None else {}),
#                 n_residuals_per_layer=2 if d_intermediate == 0 else 3,  # 3 if we have MLP
#             )
#         )

#     def forward_features(self, x: Tensor, seq_idx: Tensor):
#         """
#         x: (BT, C, H, W)
#         """
#         x = self.patch_embed(x) # (BT, HW, D)
#         BT, _, _ = x.shape

#         cls_token = self.cls_token.expand(BT, -1, -1) # (BT, 1, D)
#         x = torch.cat([x, cls_token], dim=1) # (BT, HW + 1, D)
        
#         residual = None
#         hidden_states = x
#         for layer in self.layers:
#             hidden_states, residual = layer(hidden_states, seq_idx, residual)
#         if not self.fused_add_norm:
#             residual = hidden_states + residual if residual is not None else hidden_states
#             hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
#         else:
#             # Set prenorm=False here since we don't need the residual
#             hidden_states = layer_norm_fn(
#                 hidden_states,
#                 self.norm_f.weight,
#                 self.norm_f.bias,
#                 eps=self.norm_f.eps,
#                 residual=residual,
#                 prenorm=False,
#                 residual_in_fp32=self.residual_in_fp32,
#                 is_rms_norm=isinstance(self.norm_f, RMSNorm)
#             )
        
#         return torch.cat([hidden_states[:, :-1, :].mean(dim=1), hidden_states[:, -1, :]], dim=-1)
    
#     def forward(self, x: Tensor, lengths: list[int], seq_idx: Tensor):
#         """
#         x(frames): (BT, C, H, W)
#         lengths: (B,)
#         seq_idx: (BT,)
#         """
#         x = self.forward_features(x, seq_idx) # (BT, D)
#         x = self.head(self.head_drop(x))

#         log_probs = x.log_softmax(dim=-1) # (BT, D)
#         log_probs = log_probs.split(lengths, dim=0) # ((T_i, D), ...)
#         log_probs = pad_sequence(log_probs, batch_first=True, padding_value=-1e9) # (B, T, D)

#         return {
#             # 'logits': x,
#             'log_probs': log_probs,
#             'lengths': torch.tensor(lengths, device=log_probs.device, dtype=torch.int32)
#         }


#     def build_tokens(self, gloss_dict: dict):
#         # gloss_dict: {gloss: [idx, ...]}, idx in 1..N
#         get_idx = lambda v: int(v[0]) if isinstance(v, (list, tuple)) else int(v)
#         N = len(gloss_dict)
#         tokens = [None] * (N + 1)
#         tokens[0] = '<blank>'
#         for g, v in gloss_dict.items():
#             idx = get_idx(v)  # 1..N
#             tokens[idx] = g
#         assert all(t is not None for t in tokens)
#         return tokens  # len = N+1, tokens[0] = '<blank>'
    
#     def get_num_layers(self):
#         return len(self.layers)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'cls_token'}
        
#     def decode(self, log_probs: Tensor, lengths: Tensor):
#         """
#         log_probs: (BT, D)
#         lengths: (B,)
#         """
#         decoded = self.decoder(log_probs, lengths)
#         return decoded
    

# if __name__ == '__main__':
#     import numpy as np
#     gloss_dict = np.load("/home/kks/workspace/slr/data/phoenix2014/gloss_dict.npy", allow_pickle=True).item()  # {str: [idx, ...]}
#     model = Model(gloss_dict=gloss_dict, ssm_cfg={'spatial': {'expand': 2, 'headdim': 64}, 'temporal': {'expand': 2, 'headdim': 64}},)
    