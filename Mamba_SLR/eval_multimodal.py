# eval_multimodal.py — final: head-name remap, lazy-mat before load, tuple-collate match

import argparse, warnings, re
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.models.multi_modal_model import MultiModalMamba


# ---------- Frame encoder adapter ----------
class FrameEncoder5D(nn.Module):
    """Make per-frame encoder E((B*T,C,H,W)->(B*T,D)) accept (B,T,C,H,W)->(B,T,D)."""
    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise RuntimeError(f"FrameEncoder5D needs (B,T,C,H,W), got {tuple(x.shape)}")
        B,T,C,H,W = x.shape
        y = self.enc(x.reshape(B*T, C, H, W))
        if y.dim() == 2:
            return y.view(B, T, -1)
        if y.dim() == 3:
            return y.mean(dim=1).view(B, T, -1)
        raise RuntimeError(f"Unexpected frame encoder output {tuple(y.shape)}")


# ---------- Gloss map ----------
def load_id2gloss(gloss_dict_path: str) -> Dict[int, str]:
    raw = np.load(gloss_dict_path, allow_pickle=True)
    try:
        obj = raw.item()
    except Exception:
        obj = raw
    id2 = {}
    if isinstance(obj, dict):
        k0 = next(iter(obj.keys())) if obj else None
        v0 = obj[k0] if obj else None
        if isinstance(k0, str) and isinstance(v0, (int, np.integer)):
            for g, i in obj.items(): id2[int(i)] = str(g)
        elif isinstance(k0, (int, np.integer)) and isinstance(v0, str):
            for i, g in obj.items(): id2[int(i)] = str(g)
        else:
            for k, v in obj.items():
                try: i, g = int(k), str(v)
                except: i, g = int(v), str(k)
                id2[i] = g
    else:
        for i, g in enumerate(list(obj)): id2[int(i)] = str(g)
    id2.setdefault(0, "<blank>")
    return id2


# ---------- Decode + metrics ----------
def collapse_ctc(ids, blank=0):
    out, prev = [], None
    for t in ids:
        t = int(t)
        if t == blank: prev = t; continue
        if prev is not None and t == prev: continue
        out.append(t); prev = t
    return out

def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1,m+1):
        ai = a[i-1]
        for j in range(1,n+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (ai!=b[j-1]))
    return dp[m][n]

@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, id2gloss: Dict[int,str], device: str, use_bf16: bool) -> Tuple[float,float,int]:
    model.eval()
    total_wer_e = total_wer_t = total_cer_e = total_cer_c = N = 0
    autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if (use_bf16 and device.startswith("cuda")) else torch.cuda.amp.autocast(enabled=False))
    for batch in dl:
        # Your collate returns a 7-tuple:
        # (images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths)
        images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths = batch  # :contentReference[oaicite:3]{index=3}
        images = images.to(device, non_blocking=True)
        qgrids = qgrids.to(device, non_blocking=True) if qgrids is not None else None
        keypoints = keypoints.to(device, non_blocking=True) if keypoints is not None else None
        qgrid_lengths = qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None

        with autocast_ctx:
            logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B,T,V) :contentReference[oaicite:4]{index=4}
        pred = logits.argmax(-1).cpu()
        labels = labels.cpu()

        for b in range(pred.size(0)):
            hyp_ids = collapse_ctc(pred[b].tolist(), blank=0)
            hyp = " ".join(id2gloss.get(t, "<UNK>") for t in hyp_ids)
            ref_ids = [int(t) for t in labels[b].tolist() if int(t) != 0]
            ref = " ".join(id2gloss.get(t, "<UNK>") for t in ref_ids)

            total_wer_e += edit_distance(hyp.split(), ref.split())
            total_wer_t += max(1, len(ref.split()))
            total_cer_e += edit_distance(list(hyp), list(ref))
            total_cer_c += max(1, len(ref))
            N += 1

    WER = 100.0 * total_wer_e / max(1, total_wer_t)
    CER = 100.0 * total_cer_e / max(1, total_cer_c)
    return WER, CER, N


# ---------- ckpt utils ----------
def strip_prefixes(sd: Dict[str, torch.Tensor], prefixes=("module.", "model.")) -> Dict[str, torch.Tensor]:
    out = {}
    for k,v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p): nk = nk[len(p):]
        out[nk] = v
    return out

def find_head_prefix(sd: Dict[str, torch.Tensor]) -> Optional[str]:
    for p in ("classifier","ctc_head","head","fc","final_proj"):
        if f"{p}.weight" in sd and sd[f"{p}.weight"].dim()==2:
            return p
    return None

def remap_head(sd: Dict[str, torch.Tensor], src: str, dst: str) -> Dict[str, torch.Tensor]:
    if src == dst: return sd
    rem = {}
    for k,v in sd.items():
        if k.startswith(src + "."):
            rem[f"{dst}.{k[len(src)+1:]}"] = v
        else:
            rem[k] = v
    return rem


# ---------- args ----------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    # paths (keep your defaults)
    ap.add_argument("--image_prefix", default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    ap.add_argument("--qgrid_prefix", default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    ap.add_argument("--kp_path",      default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    ap.add_argument("--meta_dir",     default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    ap.add_argument("--gloss_dict",   default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    ap.add_argument("--split", choices=["train","val","dev","test"], default="test")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_kv", type=int, default=1024)  # match training default :contentReference[oaicite:5]{index=5}
    return ap.parse_args()


# ---------- main ----------
def main():
    args = get_args()
    warnings.filterwarnings("once", category=FutureWarning)
    device = args.device
    map_location = torch.device(device)

    # Dataset + loader (same collate as training)
    ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix, qgrid_prefix=args.qgrid_prefix, kp_path=args.kp_path,
        meta_dir_path=args.meta_dir, gloss_dict_path=args.gloss_dict,
        split=("dev" if args.split=="val" else args.split),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True,
                    collate_fn=multi_modal_collate_fn, drop_last=False)  # :contentReference[oaicite:6]{index=6}

    # id2gloss (+blank)
    id2gloss = load_id2gloss(args.gloss_dict)
    num_classes = max(id2gloss.keys()) + 1
    print(f"[head] num_classes={num_classes} (from gloss_dict + blank)")

    # Build model w/ same defaults as training script
    model = MultiModalMamba(num_classes=num_classes, max_kv=args.max_kv).to(device)

    # ✅ in-place forward patch (keeps module name 'frame_encoder' so ckpt keys match)
    import types
    _fe_orig_forward = model.frame_encoder.forward

    def _fe_forward_5d(self, x, *a, **kw):   # ← add self here
        if x.dim() == 5:                     # (B,T,C,H,W)
            B, T, C, H, W = x.shape
            y = _fe_orig_forward(x.reshape(B*T, C, H, W), *a, **kw)
            if y.dim() == 2:  return y.view(B, T, -1)
            if y.dim() == 3:  return y.mean(dim=1).view(B, T, -1)
            raise RuntimeError(f"Unexpected frame_encoder output {tuple(y.shape)}")
        elif x.dim() == 4:                   # (B*T,C,H,W)
            return _fe_orig_forward(x, *a, **kw)
        else:
            raise RuntimeError(f"frame_encoder expected 4D/5D, got {tuple(x.shape)}")

    model.frame_encoder.forward = types.MethodType(_fe_forward_5d, model.frame_encoder)



    # -------- 1) Materialize LazyLinear with a dummy forward BEFORE loading --------
    try:
        first = next(iter(dl))
        images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths = first
        with torch.no_grad():
            _ = model(
                images.to(device, non_blocking=True),
                qgrids.to(device, non_blocking=True) if qgrids is not None else None,
                keypoints.to(device, non_blocking=True) if keypoints is not None else None,
                qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None,
            )
    except StopIteration:
        pass

    # -------- 2) Load checkpoint (strip DDP prefixes, remap head name) --------
    print(f"[ckpt] loading: {args.ckpt}")
    try:
        ckpt = torch.load(args.ckpt, map_location=map_location, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=map_location)

    sd = None
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state")
        if sd is None and all(isinstance(k, str) for k in ckpt.keys()):
            sd = ckpt
    if sd is None:
        sd = {}
    sd = strip_prefixes(sd)

    head_src = find_head_prefix(sd)
    if head_src and head_src != "classifier":
        sd = remap_head(sd, head_src, "classifier")

    ld = model.load_state_dict(sd, strict=False)
    missing = list(getattr(ld, "missing_keys", []))
    unexpected = list(getattr(ld, "unexpected_keys", []))
    print(f"[ckpt] unexpected keys: {len(unexpected)}")
    print(f"[ckpt] missing keys: {len(missing)}")
    if unexpected[:5]: print("[ckpt] unexpected sample:", unexpected[:5])
    if missing[:5]:    print("[ckpt] missing sample:", missing[:5])

    # missing = list(getattr(ld, "missing_keys", []))
    # unexpected = list(getattr(ld, "unexpected_keys", []))
    # if unexpected: print(f"[ckpt] unexpected keys: {len(unexpected)}")
    # if missing:    print(f"[ckpt] missing keys: {len(missing)}")

    # Warn if head still didn’t load
    if ("classifier.weight" in missing) or ("classifier.bias" in missing):
        print("[warn] Classifier head did not fully load from checkpoint. Vocab/arch mismatch will tank WER.")

    # -------- 3) Eval --------
    use_bf16 = bool(args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    WER, CER, N = evaluate(model, dl, id2gloss, device, use_bf16)
    print(f"[RESULT] N={N}  WER={WER:.2f}%  CER={CER:.2f}%")


if __name__ == "__main__":
    main()
