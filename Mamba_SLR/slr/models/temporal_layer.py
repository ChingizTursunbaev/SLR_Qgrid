# slr/models/temporal_layer.py
# Copyright (c) 2024, Tri Dao, Albert Gu.

# slr/models/temporal_layer.py
# Copyright (c) 2024

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# removed direct import for portability
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

# --- optional causal_conv1d wrapper (native if available; PyTorch fallback otherwise) ---
try:
    from causal_conv1d import causal_conv1d_fn as _native_causal_conv1d_fn
except Exception:
    _native_causal_conv1d_fn = None

def causal_conv1d_fn(x_B_D_L, weight_D_W, bias=None, activation="none", seq_idx=None):
    """
    x_B_D_L: (B, D, L), weight_D_W: (D, W), bias: (D,)
    Returns (B, D, L). Fallback keeps strict causality (no future leakage).
    """
    if _native_causal_conv1d_fn is not None:
        return _native_causal_conv1d_fn(
            x_B_D_L, weight_D_W, bias=bias, activation=activation, seq_idx=seq_idx
        )
    import torch.nn.functional as F
    D, W = weight_D_W.shape
    w = weight_D_W.unsqueeze(1)  # (D,1,W)
    y = F.conv1d(x_B_D_L, w, bias=bias, stride=1, padding=W-1, groups=D)  # (B,D,L+W-1)
    y = y[..., : x_B_D_L.shape[-1]]  # trim to L (causal)
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y
# --- end optional wrapper ---



class TemporalLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=32,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=2,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # fused/sharding options
        chunk_size=128,
        use_mem_eff_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [x, B, C, dt, x_b, B_b, C_b, dt_bwd]
        d_in_proj = self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        self.conv_dim = (self.d_inner + 2 * self.ngroups * self.d_state) // 2
        self.conv1d_fwd = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_bwd = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d_fwd.weight, -self.conv_init, self.conv_init)
            nn.init.uniform_(self.conv1d_bwd.weight, -self.conv_init, self.conv_init)

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^-1
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u: torch.Tensor, seq_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        u: (B, L, D)
        Returns: (B, L, D)
        """
        batch, seqlen, _ = u.shape

        # ensure we have a valid (B, L) seq_idx on correct device & dtype=int32 (required by causal_conv1d)
        if seq_idx is None:
            seq_idx = torch.arange(seqlen, device=u.device, dtype=torch.int32).unsqueeze(0).expand(batch, seqlen)
        else:
            seq_idx = seq_idx.to(device=u.device, dtype=torch.int32)
            if seq_idx.dim() == 1:  # (L,)
                assert seq_idx.numel() == seqlen, f"seq_idx length {seq_idx.numel()} != seqlen {seqlen}"
                seq_idx = seq_idx.unsqueeze(0).expand(batch, seqlen)
            elif seq_idx.dim() == 2:  # (B?, L)
                if seq_idx.shape[1] != seqlen:
                    raise ValueError(f"seq_idx.shape[1] ({seq_idx.shape[1]}) != seqlen ({seqlen})")
                if seq_idx.shape[0] == 1 and batch > 1:
                    seq_idx = seq_idx.expand(batch, seqlen)
                elif seq_idx.shape[0] != batch:
                    raise ValueError(f"seq_idx.shape[0] ({seq_idx.shape[0]}) != batch ({batch})")
            else:
                raise ValueError(f"seq_idx must be 1D or 2D, got shape {tuple(seq_idx.shape)}")
        seq_idx = seq_idx.contiguous()

        A = -torch.exp(self.A_log)  # (nheads)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        xbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        xBC, dt = xbcdt.split([self.conv_dim * 2, self.nheads], dim=-1)
        xBC_fwd, xBC_bwd = xBC.chunk(2, dim=-1)
        dt = F.softplus(dt + self.dt_bias)
        assert self.activation in ["silu", "swish"]

        xBC_fwd = causal_conv1d_fn(
            xBC_fwd.transpose(1, 2),
            rearrange(self.conv1d_fwd.weight, "d 1 w -> d w"),
            bias=self.conv1d_fwd.bias,
            activation=self.activation,
            seq_idx=seq_idx,
        ).transpose(1, 2)

        xBC_bwd = causal_conv1d_fn(
            xBC_bwd.flip(1).transpose(1, 2),
            rearrange(self.conv1d_bwd.weight, "d 1 w -> d w"),
            bias=self.conv1d_bwd.bias,
            activation=self.activation,
            seq_idx=seq_idx.flip(1).contiguous(),
        ).transpose(1, 2)

        # (B, L, d_inner//2), (B, L, ngroups*d_state//2), (B, L, ngroups*d_state//2)
        x_fwd, B_fwd, C_fwd = xBC_fwd.split(
            [self.d_inner // 2, self.ngroups * self.d_state // 2, self.ngroups * self.d_state // 2], dim=-1
        )
        x_bwd, B_bwd, C_bwd = xBC_bwd.split(
            [self.d_inner // 2, self.ngroups * self.d_state // 2, self.ngroups * self.d_state // 2], dim=-1
        )

        A_fwd, A_bwd = A.chunk(2, dim=-1)  # (nheads // 2)
        D_fwd, D_bwd = self.D.chunk(2, dim=-1)
        dt_fwd, dt_bwd = dt.chunk(2, dim=-1)  # (B, L, nheads // 2)
        dt_bwd = dt_bwd.flip(1)

        y_fwd = mamba_chunk_scan_combined(
            x_fwd.reshape(batch, seqlen, self.nheads // 2, self.headdim),
            dt_fwd,
            A_fwd,
            B_fwd.reshape(batch, seqlen, self.ngroups // 2, self.d_state),
            C_fwd.reshape(batch, seqlen, self.ngroups // 2, self.d_state),
            chunk_size=self.chunk_size,
            D=D_fwd,
            z=None,
            seq_idx=seq_idx,
            initial_states=None,
            **dt_limit_kwargs,
        ).reshape(batch, seqlen, -1)

        y_bwd = mamba_chunk_scan_combined(
            x_bwd.reshape(batch, seqlen, self.nheads // 2, self.headdim),
            dt_bwd,
            A_bwd,
            B_bwd.reshape(batch, seqlen, self.ngroups // 2, self.d_state),
            C_bwd.reshape(batch, seqlen, self.ngroups // 2, self.d_state),
            chunk_size=self.chunk_size,
            D=D_bwd,
            z=None,
            seq_idx=seq_idx.flip(1).contiguous(),
            initial_states=None,
            **dt_limit_kwargs,
        ).reshape(batch, seqlen, -1)

        y = torch.cat([y_fwd, y_bwd.flip(1)], dim=-1)
        y = self.norm(y)
        out = self.out_proj(y)
        return out


if __name__ == "__main__":
    layer = TemporalLayer(384, headdim=32).cuda()
    x = torch.rand(2, 8, 384, device="cuda")
    print(layer(x).shape)
    si = torch.arange(8, device="cuda", dtype=torch.int32).unsqueeze(0).expand(2, 8)
    print(layer(x, si).shape)




# # slr/models/temporal_layer.py
# # Copyright (c) 2024, Tri Dao, Albert Gu.

# # slr/models/temporal_layer.py
# # Copyright (c) 2024

# import math
# from typing import Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# # from causal_conv1d import causal_conv1d_fn
# from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

# try:
#     from causal_conv1d import causal_conv1d_fn as _native_causal_conv1d_fn
# except Exception:
#     _native_causal_conv1d_fn = None

# def causal_conv1d_fn(x_B_D_L, weight_D_W, bias=None, activation="none", seq_idx=None):
#     """
#     x_B_D_L: (B, D, L)
#     weight_D_W: (D, W)
#     bias: (D,)
#     """
#     if _native_causal_conv1d_fn is not None:
#         return _native_causal_conv1d_fn(
#             x_B_D_L, weight_D_W, bias=bias, activation=activation, seq_idx=seq_idx
#         )

#     # Fallback: depthwise 1D conv with causal padding (no future leakage)
#     import torch.nn.functional as F
#     D, W = weight_D_W.shape
#     w = weight_D_W.unsqueeze(1)  # (D,1,W)
#     y = F.conv1d(x_B_D_L, w, bias=bias, stride=1, padding=W-1, groups=D)  # (B,D,L+W-1)
#     y = y[..., :x_B_D_L.shape[-1]]  # trim back to L (causal)
#     if activation in ("silu", "swish"):
#         y = F.silu(y)
#     return y

# class TemporalLayer(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         d_state=32,
#         d_conv=4,
#         conv_init=None,
#         expand=2,
#         headdim=64,
#         ngroups=2,
#         A_init_range=(1, 16),
#         dt_min=0.001,
#         dt_max=0.1,
#         dt_init_floor=1e-4,
#         dt_limit=(0.0, float("inf")),
#         learnable_init_states=False,
#         activation="swish",
#         bias=False,
#         conv_bias=True,
#         # fused/sharding options
#         chunk_size=128,
#         use_mem_eff_path=False,
#         layer_idx=None,
#         device=None,
#         dtype=None,
#     ):
#         super().__init__()
#         factory_kwargs = {"device": device, "dtype": dtype}

#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.conv_init = conv_init
#         self.expand = expand
#         self.d_inner = self.expand * self.d_model
#         self.headdim = headdim
#         self.ngroups = ngroups
#         assert self.d_inner % self.headdim == 0
#         self.nheads = self.d_inner // self.headdim
#         self.dt_limit = dt_limit
#         self.learnable_init_states = learnable_init_states
#         self.activation = activation
#         self.chunk_size = chunk_size
#         self.use_mem_eff_path = use_mem_eff_path
#         self.layer_idx = layer_idx

#         # Order: [x, B, C, dt, x_b, B_b, C_b, dt_bwd]
#         d_in_proj = self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
#         self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

#         self.conv_dim = (self.d_inner + 2 * self.ngroups * self.d_state) // 2
#         self.conv1d_fwd = nn.Conv1d(
#             in_channels=self.conv_dim,
#             out_channels=self.conv_dim,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             groups=self.conv_dim,
#             padding=d_conv - 1,
#             **factory_kwargs,
#         )
#         self.conv1d_bwd = nn.Conv1d(
#             in_channels=self.conv_dim,
#             out_channels=self.conv_dim,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             groups=self.conv_dim,
#             padding=d_conv - 1,
#             **factory_kwargs,
#         )
#         if self.conv_init is not None:
#             nn.init.uniform_(self.conv1d_fwd.weight, -self.conv_init, self.conv_init)
#             nn.init.uniform_(self.conv1d_bwd.weight, -self.conv_init, self.conv_init)

#         if self.learnable_init_states:
#             self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
#             self.init_states._no_weight_decay = True

#         # Initialize log dt bias
#         dt = torch.exp(
#             torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
#         )
#         dt = torch.clamp(dt, min=dt_init_floor)
#         inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^-1
#         self.dt_bias = nn.Parameter(inv_dt)
#         self.dt_bias._no_weight_decay = True

#         # A parameter
#         assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
#         A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
#         A_log = torch.log(A).to(dtype=dtype)
#         self.A_log = nn.Parameter(A_log)
#         self.A_log._no_weight_decay = True

#         # D "skip" parameter
#         self.D = nn.Parameter(torch.ones(self.nheads, device=device))
#         self.D._no_weight_decay = True

#         self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **factory_kwargs)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

#     def forward(self, u: torch.Tensor, seq_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         u: (B, L, D)
#         Returns: (B, L, D)
#         """
#         batch, seqlen, _ = u.shape

#         # ensure we have a valid (B, L) seq_idx on correct device & dtype=int32 (required by causal_conv1d)
#         if seq_idx is None:
#             seq_idx = torch.arange(seqlen, device=u.device, dtype=torch.int32).unsqueeze(0).expand(batch, seqlen)
#         else:
#             seq_idx = seq_idx.to(device=u.device, dtype=torch.int32)
#             if seq_idx.dim() == 1:  # (L,)
#                 assert seq_idx.numel() == seqlen, f"seq_idx length {seq_idx.numel()} != seqlen {seqlen}"
#                 seq_idx = seq_idx.unsqueeze(0).expand(batch, seqlen)
#             elif seq_idx.dim() == 2:  # (B?, L)
#                 if seq_idx.shape[1] != seqlen:
#                     raise ValueError(f"seq_idx.shape[1] ({seq_idx.shape[1]}) != seqlen ({seqlen})")
#                 if seq_idx.shape[0] == 1 and batch > 1:
#                     seq_idx = seq_idx.expand(batch, seqlen)
#                 elif seq_idx.shape[0] != batch:
#                     raise ValueError(f"seq_idx.shape[0] ({seq_idx.shape[0]}) != batch ({batch})")
#             else:
#                 raise ValueError(f"seq_idx must be 1D or 2D, got shape {tuple(seq_idx.shape)}")
#         seq_idx = seq_idx.contiguous()

#         A = -torch.exp(self.A_log)  # (nheads)
#         dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

#         xbcdt = self.in_proj(u)  # (B, L, d_in_proj)
#         xBC, dt = xbcdt.split([self.conv_dim * 2, self.nheads], dim=-1)
#         xBC_fwd, xBC_bwd = xBC.chunk(2, dim=-1)
#         dt = F.softplus(dt + self.dt_bias)
#         assert self.activation in ["silu", "swish"]

#         xBC_fwd = causal_conv1d_fn(
#             xBC_fwd.transpose(1, 2),
#             rearrange(self.conv1d_fwd.weight, "d 1 w -> d w"),
#             bias=self.conv1d_fwd.bias,
#             activation=self.activation,
#             seq_idx=seq_idx,
#         ).transpose(1, 2)

#         xBC_bwd = causal_conv1d_fn(
#             xBC_bwd.flip(1).transpose(1, 2),
#             rearrange(self.conv1d_bwd.weight, "d 1 w -> d w"),
#             bias=self.conv1d_bwd.bias,
#             activation=self.activation,
#             seq_idx=seq_idx.flip(1).contiguous(),
#         ).transpose(1, 2)

#         # (B, L, d_inner//2), (B, L, ngroups*d_state//2), (B, L, ngroups*d_state//2)
#         x_fwd, B_fwd, C_fwd = xBC_fwd.split(
#             [self.d_inner // 2, self.ngroups * self.d_state // 2, self.ngroups * self.d_state // 2], dim=-1
#         )
#         x_bwd, B_bwd, C_bwd = xBC_bwd.split(
#             [self.d_inner // 2, self.ngroups * self.d_state // 2, self.ngroups * self.d_state // 2], dim=-1
#         )

#         A_fwd, A_bwd = A.chunk(2, dim=-1)  # (nheads // 2)
#         D_fwd, D_bwd = self.D.chunk(2, dim=-1)
#         dt_fwd, dt_bwd = dt.chunk(2, dim=-1)  # (B, L, nheads // 2)
#         dt_bwd = dt_bwd.flip(1)

#         y_fwd = mamba_chunk_scan_combined(
#             x_fwd.reshape(batch, seqlen, self.nheads // 2, self.headdim),
#             dt_fwd,
#             A_fwd,
#             B_fwd.reshape(batch, seqlen, self.ngroups // 2, self.d_state),
#             C_fwd.reshape(batch, seqlen, self.ngroups // 2, self.d_state),
#             chunk_size=self.chunk_size,
#             D=D_fwd,
#             z=None,
#             seq_idx=seq_idx,
#             initial_states=None,
#             **dt_limit_kwargs,
#         ).reshape(batch, seqlen, -1)

#         y_bwd = mamba_chunk_scan_combined(
#             x_bwd.reshape(batch, seqlen, self.nheads // 2, self.headdim),
#             dt_bwd,
#             A_bwd,
#             B_bwd.reshape(batch, seqlen, self.ngroups // 2, self.d_state),
#             C_bwd.reshape(batch, seqlen, self.ngroups // 2, self.d_state),
#             chunk_size=self.chunk_size,
#             D=D_bwd,
#             z=None,
#             seq_idx=seq_idx.flip(1).contiguous(),
#             initial_states=None,
#             **dt_limit_kwargs,
#         ).reshape(batch, seqlen, -1)

#         y = torch.cat([y_fwd, y_bwd.flip(1)], dim=-1)
#         y = self.norm(y)
#         out = self.out_proj(y)
#         return out


# if __name__ == "__main__":
#     layer = TemporalLayer(384, headdim=32).cuda()
#     x = torch.rand(2, 8, 384, device="cuda")
#     print(layer(x).shape)
#     si = torch.arange(8, device="cuda", dtype=torch.int32).unsqueeze(0).expand(2, 8)
#     print(layer(x, si).shape)

