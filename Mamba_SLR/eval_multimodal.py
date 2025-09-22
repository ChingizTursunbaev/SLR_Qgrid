# eval_multimodal.py
# Uses your exact path defaults + robust id2gloss loading from gloss_dict
# Adds a small adapter so frame_encoder can take (B,T,C,H,W) -> (B,T,D)

import argparse
import warnings
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset
from slr.models.multi_modal_model import MultiModalMamba


# -------------------------------
# Frame-encoder adapter
# -------------------------------
class _FrameEncoderAdapter(nn.Module):
    """
    Makes an encoder that expects (B*T, C, H, W) work with (B, T, C, H, W).
    Returns (B, T, D). If input is already (B*T, C, H, W), passes through.
    """
    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # already (B*T, C, H, W)
            return self._normalize(self.enc(x), B=None, T=None)
        if x.dim() != 5:
            raise RuntimeError(f"[adapter] expected 5D (B,T,C,H,W) or 4D (B*T,C,H,W), got {tuple(x.shape)}")
        B, T, C, H, W = x.shape
        x_bt = x.reshape(B * T, C, H, W)
        y = self.enc(x_bt)  # (B*T, D) or (B*T, P, D)
        return self._normalize(y, B, T)

    @staticmethod
    def _normalize(y: torch.Tensor, B: Optional[int], T: Optional[int]) -> torch.Tensor:
        if B is None or T is None:
            return y
        if y.dim() == 2:       # (B*T, D)
            return y.view(B, T, -1)
        if y.dim() == 3:       # (B*T, P, D) -> mean pool patches
            y = y.mean(dim=1)
            return y.view(B, T, -1)
        raise RuntimeError(f"[adapter] unexpected frame_encoder output {tuple(y.shape)}")


# -------------------------------
# Args (your exact defaults)
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pt")
    p.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--force_num_classes", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# -------------------------------
# id2gloss loader (robust)
# -------------------------------
def load_id2gloss(gloss_dict_path: str) -> Dict[int, str]:
    """
    Supports both:
      - dict: {gloss(str) -> id(int)}  or  {id(int) -> gloss(str)}
      - list/array: index -> gloss
    Always returns a mapping {id(int) -> gloss(str)} and inserts id=0 as '<blank>'
    if not present.
    """
    obj = np.load(gloss_dict_path, allow_pickle=True)
    try:
        obj = obj.item()  # if it's a dict saved via np.save
    except Exception:
        pass

    id2gloss: Dict[int, str] = {}

    if isinstance(obj, dict):
        # Detect orientation
        # Case A: keys are strings (gloss), values ints (id)
        if obj and isinstance(next(iter(obj.keys())), str) and isinstance(next(iter(obj.values())), (int, np.integer)):
            for g, i in obj.items():
                id2gloss[int(i)] = str(g)
        # Case B: keys are ints (id), values strings (gloss)
        elif obj and isinstance(next(iter(obj.keys())), (int, np.integer)) and isinstance(next(iter(obj.values())), str):
            for i, g in obj.items():
                id2gloss[int(i)] = str(g)
        else:
            # Fallback: try to coerce
            for k, v in obj.items():
                try:
                    i = int(k)
                    g = str(v)
                except Exception:
                    i = int(v)
                    g = str(k)
                id2gloss[i] = g
    else:
        # Assume sequence-like container of gloss strings
        arr = list(obj)
        for i, g in enumerate(arr):
            id2gloss[int(i)] = str(g)

    # Ensure blank at index 0
    if 0 not in id2gloss:
        id2gloss[0] = "<blank>"

    return id2gloss


# -------------------------------
# Evaluation
# -------------------------------
@torch.no_grad()
def evaluate(model: nn.Module,
             dl: DataLoader,
             id2gloss: Dict[int, str],
             device: str,
             use_bf16: bool = False):
    """
    Greedy CTC-like decode (blank=0, collapse repeats) -> WER/CER.
    Assumes model(images, qgrids, keypoints, qgrid_lengths) -> (B, T', C).
    """
    model.eval()
    blank_idx = 0

    total_wer_edits = 0
    total_wer_tokens = 0
    total_cer_edits = 0
    total_cer_chars = 0
    total_samples = 0

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if (use_bf16 and device.startswith("cuda"))
                    else torch.cuda.amp.autocast(enabled=False))

    def collapse_and_map(ids):
        seq = []
        prev = None
        for t in ids:
            if t == blank_idx:
                prev = t
                continue
            if prev is not None and t == prev:
                continue
            seq.append(t)
            prev = t
        return [id2gloss.get(int(x), "<UNK>") for x in seq]

    def edit_distance(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )
        return dp[m][n]

    def cer_edits(hyp_str, ref_str):
        a = list(hyp_str)
        b = list(ref_str)
        return edit_distance(a, b), len(b)

    for batch in dl:
        if not isinstance(batch, dict):
            raise RuntimeError("Expected dict batches from MultiModalPhoenixDataset.")

        images = batch.get("images") or batch.get("frames")
        qgrids = batch.get("qgrids") or batch.get("qgrid")
        keypoints = batch.get("keypoints") or batch.get("kps") or batch.get("pose")
        qgrid_lengths = batch.get("qgrid_lengths") or batch.get("qgrid_len") or batch.get("lengths")

        y_ids = batch.get("gloss_ids") or batch.get("targets") or batch.get("labels")
        y_text = batch.get("gloss_str") or batch.get("gloss") or batch.get("text")

        images = images.to(device, non_blocking=True)
        qgrids = qgrids.to(device, non_blocking=True) if qgrids is not None else None
        keypoints = keypoints.to(device, non_blocking=True) if keypoints is not None else None
        qgrid_lengths = qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None

        with autocast_ctx:
            logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B, T', C)

        pred_ids = logits.argmax(dim=-1).cpu()  # (B, T')

        B = pred_ids.size(0)
        for b in range(B):
            hyp_toks = collapse_and_map(pred_ids[b].tolist())
            hyp_str = " ".join(hyp_toks)

            if y_text is not None:
                ref_str = y_text[b] if isinstance(y_text, (list, tuple)) else y_text
                if isinstance(ref_str, list):
                    ref_str = " ".join(ref_str)
            elif y_ids is not None:
                ids = y_ids[b].tolist()
                ref_toks = [id2gloss.get(int(t), "<UNK>") for t in ids if int(t) != blank_idx]
                ref_str = " ".join(ref_toks)
            else:
                ref_str = ""

            hyp_words = hyp_str.split()
            ref_words = ref_str.split()
            total_wer_edits += edit_distance(hyp_words, ref_words)
            total_wer_tokens += max(1, len(ref_words))

            ed, L = cer_edits(hyp_str, ref_str)
            total_cer_edits += ed
            total_cer_chars += max(1, L)

            total_samples += 1

    WER = 100.0 * total_wer_edits / max(1, total_wer_tokens)
    CER = 100.0 * total_cer_edits / max(1, total_cer_chars)
    return WER, CER, total_samples


# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    warnings.filterwarnings("once", category=FutureWarning)

    device = args.device
    map_location = torch.device(device)

    # Dataset — pass your args directly; note param names on ctor
    # IMPORTANT: kp_path must point to the exact filename present on disk.
    # If your file is the INTERPOLATED one, override via --kp_path.
    ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,         # ctor expects meta_dir_path
        gloss_dict_path=args.gloss_dict,     # ctor expects gloss_dict_path
        split=args.split,
    )

    # Build id2gloss:
    id2gloss = getattr(ds, "id2gloss", None)
    if not isinstance(id2gloss, dict) or len(id2gloss) == 0:
        # Load from gloss_dict file directly
        id2gloss = load_id2gloss(args.gloss_dict)

    # DataLoader (use dataset's collate if available)
    collate_fn = getattr(ds, "collate_fn", None)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn if callable(collate_fn) else None,
    )

    # Model
    n_classes = args.force_num_classes or len(id2gloss)
    if args.force_num_classes:
        print(f"[head] num_classes={n_classes} (forced)")
    model = MultiModalMamba(num_classes=n_classes).to(device)

    # Load checkpoint (try weights_only when available)
    print(f"[ckpt] loading: {args.ckpt}")
    try:
        ckpt = torch.load(args.ckpt, map_location=map_location, weights_only=True)  # PyTorch 2.5+
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=map_location)

    # Accept common formats
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt
        else:
            state_dict = ckpt.get("model", ckpt.get("model_state", {})) if isinstance(ckpt, dict) else {}

    # Strip common prefixes
    def _strip(sd: Dict[str, Any], prefixes=("module.", "model.")):
        out = {}
        for k, v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            out[nk] = v
        return out

    state_dict = _strip(state_dict)
    ld = model.load_state_dict(state_dict, strict=False)

    # Report unexpected/missing, but keep going
    missing = getattr(ld, "missing_keys", None) or (ld.get("missing_keys", []) if isinstance(ld, dict) else [])
    unexpected = getattr(ld, "unexpected_keys", None) or (ld.get("unexpected_keys", []) if isinstance(ld, dict) else [])
    if unexpected:
        print(f"[ckpt] strict load failed with unexpected keys ({len(unexpected)}). "
              f"Loading non-head layers and reinitializing classifier.")
    if missing:
        print(f"[ckpt] missing keys: {len(missing)}")

    # Patch frame encoder to accept (B, T, C, H, W)
    if not isinstance(model.frame_encoder, _FrameEncoderAdapter):
        model.frame_encoder = _FrameEncoderAdapter(model.frame_encoder)
    print("[patch] FrameEncoderAdapter attached.")

    # Eval
    WER, CER, N = evaluate(model, dl, id2gloss, device, use_bf16=bool(args.bf16))
    print(f"[RESULT] N={N}  WER={WER:.2f}%  CER={CER:.2f}%")


if __name__ == "__main__":
    main()
