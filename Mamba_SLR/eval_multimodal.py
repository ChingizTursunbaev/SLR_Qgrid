# eval_multimodal.py
# Evaluation script aligned with your dataset/model code.
# - Unpacks 7-tuple batches from multi_modal_collate_fn
# - Adapts frame encoder to (B,T,C,H,W) -> (B,T,D)
# - Greedy CTC decode (blank=0) for WER/CER

import argparse
import warnings
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.models.multi_modal_model import MultiModalMamba


# -------------------------------
# Frame-encoder adapter (5D -> sequence)
# -------------------------------
class FrameEncoder5D(nn.Module):
    """
    Wraps a per-frame encoder E that expects (B*T, C, H, W) and returns (B*T, D),
    to make it accept (B, T, C, H, W) and return (B, T, D).
    """
    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise RuntimeError(f"FrameEncoder5D expects 5D (B,T,C,H,W), got {tuple(x.shape)}")
        B, T, C, H, W = x.shape
        x_bt = x.reshape(B * T, C, H, W).contiguous()
        y_bt = self.enc(x_bt)  # (B*T, D) or (B*T, P, D)->avg pooled inside enc
        if y_bt.dim() == 2:
            y = y_bt.view(B, T, -1)
        elif y_bt.dim() == 3:
            y = y_bt.mean(dim=1).view(B, T, -1)
        else:
            raise RuntimeError(f"Unexpected frame encoder output {tuple(y_bt.shape)}")
        return y


# -------------------------------
# Robust gloss map loader
# -------------------------------
def load_id2gloss(gloss_dict_path: str) -> Dict[int, str]:
    raw = np.load(gloss_dict_path, allow_pickle=True)
    try:
        obj = raw.item()
    except Exception:
        obj = raw

    id2gloss: Dict[int, str] = {}
    if isinstance(obj, dict):
        # Could be {gloss:str -> id:int} OR {id:int -> gloss:str}
        k0 = next(iter(obj.keys())) if obj else None
        v0 = obj[k0] if obj else None
        if isinstance(k0, str) and isinstance(v0, (int, np.integer)):
            for g, i in obj.items():
                id2gloss[int(i)] = str(g)
        elif isinstance(k0, (int, np.integer)) and isinstance(v0, str):
            for i, g in obj.items():
                id2gloss[int(i)] = str(g)
        else:
            # fallback: coerce
            for k, v in obj.items():
                try:
                    i = int(k); g = str(v)
                except Exception:
                    i = int(v); g = str(k)
                id2gloss[i] = g
    else:
        # treat as list/array: index -> gloss
        seq = list(obj)
        for i, g in enumerate(seq):
            id2gloss[int(i)] = str(g)

    # ensure blank at 0
    if 0 not in id2gloss:
        id2gloss[0] = "<blank>"
    return id2gloss


# -------------------------------
# Decoding + metrics
# -------------------------------
def collapse_ctc(ids, blank=0):
    out = []
    prev = None
    for t in ids:
        t = int(t)
        if t == blank:
            prev = t
            continue
        if prev is not None and t == prev:
            continue
        out.append(t)
        prev = t
    return out

def edit_distance(a, b):
    # a, b: lists of tokens (words or chars)
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        ai = a[i-1]
        for j in range(1, n+1):
            cost = 0 if ai == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n]


@torch.no_grad()
def evaluate(model: nn.Module,
             dl: DataLoader,
             id2gloss: Dict[int, str],
             device: str,
             use_bf16: bool = False) -> Tuple[float, float, int]:

    model.eval()
    blank = 0

    total_wer_edits = 0
    total_wer_tokens = 0
    total_cer_edits = 0
    total_cer_chars = 0
    total_samples = 0

    use_cuda_bf16 = (use_bf16 and device.startswith("cuda"))
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_cuda_bf16 else torch.cuda.amp.autocast(enabled=False))

    for batch in dl:
        # Dataset collate returns a 7-tuple:
        # (images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths)
        images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths = batch

        images = images.to(device, non_blocking=True)          # (B,T,C,H,W)
        qgrids = qgrids.to(device, non_blocking=True)          # (B,Tq,242)
        keypoints = keypoints.to(device, non_blocking=True)    # (B,T,242)
        qgrid_lengths = qgrid_lengths.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B,T',V)

        pred_ids = logits.argmax(dim=-1).cpu()  # (B, T')

        # Build reference strings from label ids: ignore 0 (blank)
        labels_cpu = labels.cpu()
        B = pred_ids.size(0)
        for b in range(B):
            hyp_seq = collapse_ctc(pred_ids[b].tolist(), blank=blank)
            hyp_tokens = [id2gloss.get(t, "<UNK>") for t in hyp_seq]
            hyp_str = " ".join(hyp_tokens)

            ref_ids = [int(t) for t in labels_cpu[b].tolist() if int(t) != blank]
            ref_tokens = [id2gloss.get(t, "<UNK>") for t in ref_ids]
            ref_str = " ".join(ref_tokens)

            # WER
            wer_ed = edit_distance(hyp_str.split(), ref_str.split())
            total_wer_edits += wer_ed
            total_wer_tokens += max(1, len(ref_str.split()))

            # CER
            cer_ed = edit_distance(list(hyp_str), list(ref_str))
            total_cer_edits += cer_ed
            total_cer_chars += max(1, len(ref_str))

            total_samples += 1

    WER = 100.0 * total_wer_edits / max(1, total_wer_tokens)
    CER = 100.0 * total_cer_edits / max(1, total_cer_chars)
    return WER, CER, total_samples


# -------------------------------
# Args
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pt/.pth")

    # keep your defaults here only if needed at runtime; dataset itself discovers from meta
    p.add_argument("--image_prefix", default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    p.add_argument("--split", choices=["train", "val", "dev", "test"], default="test")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_kv", type=int, default=512, help="Max pooled KV length for qgrid/keypoints")
    return p.parse_args()


# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    warnings.filterwarnings("once", category=FutureWarning)

    device = args.device
    map_location = torch.device(device)

    # Dataset (uses robust pre-resolution & 7-tuple collate)
    ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,
        gloss_dict_path=args.gloss_dict,
        split=args.split if args.split != "val" else "dev",
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=multi_modal_collate_fn,
    )

    # id2gloss (+ ensure blank=0)
    id2gloss = getattr(ds, "id_to_gloss", None)
    if not isinstance(id2gloss, dict) or len(id2gloss) == 0:
        id2gloss = load_id2gloss(args.gloss_dict)
    if 0 not in id2gloss:
        id2gloss[0] = "<blank>"
    num_classes = max(id2gloss.keys()) + 1
    print(f"[head] num_classes={num_classes} (from gloss_dict + blank)")

    # Model
    model = MultiModalMamba(num_classes=num_classes, max_kv=args.max_kv)
    # Adapt frame encoder for 5D input -> (B,T,D)
    model.frame_encoder = FrameEncoder5D(model.frame_encoder)
    model = model.to(device)

    # Load checkpoint (lenient)
    print(f"[ckpt] loading: {args.ckpt}")
    try:
        ckpt = torch.load(args.ckpt, map_location=map_location, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=map_location)

    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt
        else:
            state_dict = ckpt.get("model", ckpt.get("model_state", {})) if isinstance(ckpt, dict) else {}

    # strip DDP prefixes
    def _strip(sd: Dict[str, Any], prefixes=("module.", "model.")):
        out = {}
        for k, v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            out[nk] = v
        return out

    ld = model.load_state_dict(_strip(state_dict), strict=False)
    missing = list(getattr(ld, "missing_keys", []))
    unexpected = list(getattr(ld, "unexpected_keys", []))
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[ckpt] missing keys: {len(missing)}")

    # Optional: sanity check whether classifier loaded (don’t crash; just warn)
    head_keys = {"classifier.weight", "classifier.bias"}
    if any(k in missing for k in head_keys):
        print("[warn] Classifier head did not fully load from checkpoint. "
              "If vocab differs from training, WER/CER will be high.")

    # Evaluate
    WER, CER, N = evaluate(model, dl, id2gloss, device, use_bf16=bool(args.bf16))
    print(f"[RESULT] N={N}  WER={WER:.2f}%  CER={CER:.2f}%")


if __name__ == "__main__":
    main()
