# slr/models/multi_modal_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model  # frame encoder (Mamba-based)

def _pool_variable_1d(x: torch.Tensor, lengths: torch.Tensor, max_tokens: int, mode: str = "mean"):
    """
    x: (B, T, D) on device
    lengths: (B,) actual lengths for each row
    Returns:
      x_pooled: (B, T', D) where T' <= max_tokens per-row
      lens_pooled: (B,)
      mask: (B, T') bool, True = PAD to ignore in attention
    """
    B, T, D = x.shape
    device = x.device
    lens = lengths.to(device)

    # compute stride per sample so that ceil(L/stride) <= max_tokens
    stride = torch.clamp(((lens + max_tokens - 1) // max_tokens), min=1)  # (B,)
    # allocate lists then pad to max pooled length across batch
    pooled_list = []
    new_lens = []
    for b in range(B):
        L = int(lens[b].item())
        s = int(stride[b].item())
        if s == 1:
            xb = x[b, :L]  # (L,D)
        else:
            # pad to multiple of s
            pad_len = (s - (L % s)) % s
            if pad_len > 0:
                pad = torch.zeros(pad_len, D, device=device, dtype=x.dtype)
                xb = torch.cat([x[b, :L], pad], dim=0)  # (L+pad, D)
            else:
                xb = x[b, :L]
            # pool by windows of size s
            xb = xb.view(-1, s, D)  # (chunks, s, D)
            if mode == "mean":
                xb = xb.mean(dim=1)
            elif mode == "max":
                xb = xb.max(dim=1).values
            elif mode == "vote":
                # ternary majority vote over each window (assumes roughly centered {-1,0,1} inputs)
                s = xb.sum(dim=1)
                xb = torch.sign(s)  # -1, 0, or 1 per feature dim
            else:
                xb = xb.mean(dim=1)
        pooled_list.append(xb)
        new_lens.append(xb.shape[0])

    T_max = max(new_lens) if new_lens else 1
    out = x.new_zeros((B, T_max, D))
    mask = torch.ones((B, T_max), dtype=torch.bool, device=device)  # True => pad
    for b, xb in enumerate(pooled_list):
        l = xb.shape[0]
        out[b, :l] = xb
        mask[b, :l] = False
    return out, torch.tensor(new_lens, device=device, dtype=torch.long), mask  # (B,T',D), (B,), (B,T')

class MultiModalMamba(nn.Module):
    """
    Image frames (B, T_img, 3, H, W)
    Qgrid       (B, T_q, 242) with values in {-1,0,1}
    Keypoints   (B, T_img, 242)

    Returns logits (B, T_img, V)
    """
    def __init__(self, img_cfg, qgrid_cfg, kp_cfg, num_classes, fusion_cfg):
        super().__init__()

        # Per-frame encoders
        # Disable fused_add_norm & rms_norm via img_cfg provided from ddp_train_multimodal.py for stability on A100
        self.image_encoder = Model(**img_cfg)

        # For qgrid we’ll project 242 -> E; we don't run the heavy image backbone on it
        embed_dim = fusion_cfg.get("embed_dim", getattr(self.image_encoder, "d_model", 512))
        self.qgrid_proj = nn.Linear(242, embed_dim)
        self.qgrid_ln   = nn.LayerNorm(embed_dim)

        # Keypoints encoder to same dim as image features (so we can add)
        kp_model_dim = kp_cfg.get("model_dim", getattr(self.image_encoder, "d_model", embed_dim))
        self.keypoint_encoder = nn.Sequential(
            nn.Linear(kp_cfg.get("input_dim", 242), kp_model_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(kp_model_dim),
        )

        # If image d_model != embed_dim, map image+kp to embed_dim for attention
        self.img_dim = getattr(self.image_encoder, "d_model", kp_model_dim)
        self.to_query = (nn.Identity() if self.img_dim == embed_dim
                         else nn.Sequential(nn.Linear(self.img_dim, embed_dim), nn.LayerNorm(embed_dim)))

        # Cross-attention: query = per-frame (image+kp), key/value = pooled Qgrid
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=fusion_cfg.get("num_heads", 8),
            dropout=fusion_cfg.get("dropout", 0.1),
            batch_first=True
        )
        self.attn_ln = nn.LayerNorm(embed_dim)

        # Final head: concat [query (img+kp), context (from qgrid)]
        fused_dim = self.img_dim + embed_dim
        self.ctc_head = nn.Linear(fused_dim, num_classes)

        # Qgrid pooling limit + mode
        self.max_kv = fusion_cfg.get("max_kv", 512)
        self.pool_mode = fusion_cfg.get("pool_mode", "mean")

    def _encode_frames(self, encoder: nn.Module, x_5d: torch.Tensor) -> torch.Tensor:
        """
        x_5d: (B, T, C, H, W) -> encoder -> (B, T, D)
        """
        B, T, C, H, W = x_5d.shape
        x = x_5d.reshape(B * T, C, H, W).contiguous()
        feats = encoder(x)  # expect (B*T, D)
        if feats.dim() == 3:
            # If encoder returns (B*T, S, D), pool S
            feats = feats.mean(dim=1)
        feats = feats.reshape(B, T, -1).contiguous()
        return feats

    def forward(self, images: torch.Tensor, qgrids: torch.Tensor, keypoints: torch.Tensor,
                qgrid_lengths: torch.Tensor = None):
        """
        images:        (B, T_img, 3, H, W)
        qgrids:        (B, T_q, 242)
        keypoints:     (B, T_img, 242)
        qgrid_lengths: (B,) optional true lengths (before padding)
        """
        B, T_img = images.shape[:2]
        device = images.device

        # 1) per-frame image features
        img_feat = self._encode_frames(self.image_encoder, images)        # (B, T_img, D_img)

        # 2) keypoints -> same dim as image, add for low-frequency enhancement
        kp_feat  = self.keypoint_encoder(keypoints)                        # (B, T_img, D_img)
        low_freq = img_feat + kp_feat                                      # (B, T_img, D_img)

        # 3) Qgrid -> pool to ≤ max_kv, project, build mask
        if self.pool_mode == "vote":
            # pool raw ternary qgrid with majority vote, then project
            if qgrid_lengths is None:
                qgrid_lengths = torch.full((B,), qgrids.size(1), dtype=torch.long, device=device)
            q_emb, q_lens, q_pad_mask = _pool_variable_1d(qgrids, qgrid_lengths, max_tokens=self.max_kv, mode="vote")
            q_emb = self.qgrid_ln(self.qgrid_proj(q_emb.float()))       # (B, T_k, E)
        else:
            q_emb = self.qgrid_ln(self.qgrid_proj(qgrids.float()))      # (B, T_q, E)
            if qgrid_lengths is None:
                qgrid_lengths = torch.full((B,), q_emb.size(1), dtype=torch.long, device=device)
            q_emb, q_lens, q_pad_mask = _pool_variable_1d(q_emb, qgrid_lengths, max_tokens=self.max_kv, mode=self.pool_mode)
        # q_pad_mask: True=pad → pass directly to MultiheadAttention as key_padding_mask

        # 4) cross-attention: queries = low_freq mapped to E
        q_queries = self.to_query(low_freq)                                # (B, T_img, E)
        q_ctx, _ = self.fusion_attn(
            query=q_queries,
            key=q_emb,
            value=q_emb,
            key_padding_mask=q_pad_mask,   # shape (B, T_k), True = IGNORE
            need_weights=False
        )
        q_ctx = self.attn_ln(q_ctx)                                        # (B, T_img, E)

        # 5) fuse + classify (CTC over T_img)
        fused = torch.cat([low_freq, q_ctx], dim=-1)                       # (B, T_img, D_img + E)
        logits = self.ctc_head(fused)                                      # (B, T_img, V)
        return logits