# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


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
        # Fused kernel and sharding options
        chunk_size=128,
        use_mem_eff_path=False,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
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
        # self.conv2d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u: torch.Tensor, seq_idx: torch.Tensor):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, _ = u.shape

        # initial_states=repeat(self.init_states, "... -> b ...", b=B) if self.learnable_init_states else None
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        xbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        
        xBC, dt = xbcdt.split([self.conv_dim * 2, self.nheads], dim=-1)
        xBC_fwd, xBC_bwd = xBC.chunk(2, dim=-1)
        dt = F.softplus(dt + self.dt_bias)
        assert self.activation in ['silu', 'swish']

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
            seq_idx=seq_idx.flip(1),
        ).transpose(1, 2)
        
        
        # (batch, seqlen, d_inner // 2), (batch, seqlen, ngroups * d_state // 2), (batch, seqlen, ngroups * d_state // 2)
        x_fwd, B_fwd, C_fwd = xBC_fwd.split([self.d_inner // 2, self.ngroups * self.d_state // 2, self.ngroups * self.d_state // 2], dim=-1)
        x_bwd, B_bwd, C_bwd = xBC_bwd.split([self.d_inner // 2, self.ngroups * self.d_state // 2, self.ngroups * self.d_state // 2], dim=-1)

        A_fwd, A_bwd = A.chunk(2, dim=-1) # (nheads // 2)
        D_fwd, D_bwd = self.D.chunk(2, dim=-1) # (nheads // 2)
        dt_fwd, dt_bwd = dt.chunk(2, dim=-1) # (batch, seqlen, nheads // 2)
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
            seq_idx=seq_idx.flip(1),
            initial_states=None,
            **dt_limit_kwargs,
        ).reshape(batch, seqlen, -1)

        y = torch.cat([y_fwd, y_bwd.flip(1)], dim=-1)
        y = self.norm(y)

        out = self.out_proj(y)
        return out

    
if __name__ == '__main__':
    layer = TemporalLayer(384, headdim=32).cuda()
    print(layer(torch.rand(2, 8, 384, device='cuda'), torch.arange(8, device='cuda', dtype=torch.int32).expand(2, -1)).shape)