# eval_multimodal.py — evaluation that mirrors ddp_train_multimodal.py

import argparse, warnings, re
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.models.multi_modal_model import MultiModalMamba


# -------------------------------
# CLI
# -------------------------------
def get_args():
    p = argparse.ArgumentParser()

    # data (your defaults from training)
    p.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    p.add_argument("--split", choices=["train","dev","val","test"], default="test")

    # model (match training defaults)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=12)     # will be auto-inferred from ckpt if possible
    p.add_argument("--fusion_embed", type=int, default=512)
    p.add_argument("--fusion_heads", type=int, default=8)
    p.add_argument("--max_kv", type=int, default=1024)
    p.add_argument("--pool_mode", type=str, default="mean", choices=["mean","max","vote"])

    # eval
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--blank_idx", type=int, default=0)
    return p.parse_args()


# -------------------------------
# Utils
# -------------------------------
def load_id2gloss(gloss_dict_path: str) -> Dict[int,str]:
    """Invert gloss→id mapping; ensure id 0 exists as <blank>."""
    obj = np.load(gloss_dict_path, allow_pickle=True)
    try:
        d = obj.item()
        # d is gloss->id (strings to ints) in your repo
        id2 = {int(i): str(g) for g, i in d.items()}
    except Exception:
        seq = list(obj)
        id2 = {int(i): str(g) for i,g in enumerate(seq)}
    if 0 not in id2:
        id2[0] = "<blank>"
    return id2

def wrap_frame_encoder_for_5d(model: nn.Module):
    """Let the frame encoder accept (B,T,C,H,W) like in training."""
    fe = model.frame_encoder
    orig_forward = fe.forward
    def adapter(x: torch.Tensor, *args, **kw):
        if x.dim() == 4:
            return orig_forward(x, *args, **kw)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            y = orig_forward(x.reshape(B*T, C, H, W), *args, **kw)
            if y.dim() == 2:  # (B*T, D)
                return y.view(B, T, -1)
            if y.dim() == 3:  # (B*T, P, D)
                return y.mean(dim=1).view(B, T, -1)
            raise RuntimeError(f"Unexpected frame_encoder output {tuple(y.shape)}")
        raise RuntimeError(f"frame_encoder input must be 4D/5D, got {tuple(x.shape)}")
    fe.forward = adapter

def unpack_batch(batch: Any) -> Dict[str, Any]:
    """Training-style unpack (tuple from multi_modal_collate_fn -> dict)."""
    if isinstance(batch, dict):
        return batch
    (images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths) = batch
    return {
        "images": images, "qgrids": qgrids, "keypoints": keypoints,
        "labels": labels, "image_lengths": image_lengths,
        "label_lengths": label_lengths, "qgrid_lengths": qgrid_lengths,
    }

def infer_n_layer_from_ckpt(state_dict: Dict[str, torch.Tensor], fallback: int) -> int:
    ids = []
    for k in state_dict.keys():
        m = re.search(r"frame_encoder\.layers\.(\d+)\.", k)
        if m:
            ids.append(int(m.group(1)))
    return (max(ids) + 1) if ids else fallback

def remap_head_to_classifier(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map common head names to 'classifier.*' (your model uses classifier)."""
    for prefix in ("classifier", "head", "fc", "final_proj"):
        if f"{prefix}.weight" in state_dict:
            if prefix == "classifier":
                return state_dict  # matches model
            # remap to classifier.*
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith(prefix + "."):
                    remapped["classifier." + k[len(prefix) + 1:]] = v
                else:
                    remapped[k] = v
            return remapped
    return state_dict


# -------------------------------
# Decode & metrics
# -------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, id2gloss: Dict[int,str],
             device: str, use_bf16: bool, blank_idx: int):
    model.eval()
    total_wer_edits = total_wer_tokens = 0
    total_cer_edits = total_cer_chars = 0
    N = 0

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if (use_bf16 and device.startswith("cuda"))
                    else torch.cuda.amp.autocast(enabled=False))

    def collapse_and_map(ids):
        out, prev = [], None
        for t in ids:
            if t == blank_idx: prev = t; continue
            if prev is not None and t == prev: continue
            out.append(t); prev = t
        return [id2gloss.get(int(x), "<UNK>") for x in out]

    def ed(a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1,m+1):
            for j in range(1,n+1):
                c = 0 if a[i-1]==b[j-1] else 1
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+c)
        return dp[m][n]

    for batch in loader:
        b = unpack_batch(batch)
        images = b["images"].to(device, non_blocking=True)
        qgrids = b["qgrids"].to(device, non_blocking=True) if b["qgrids"] is not None else None
        keypoints = b["keypoints"].to(device, non_blocking=True) if b["keypoints"] is not None else None
        qgrid_lengths = b["qgrid_lengths"].to(device, non_blocking=True) if b.get("qgrid_lengths") is not None else None

        with autocast_ctx:
            logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B,T,V)

        pred_ids = logits.argmax(dim=-1).cpu()  # (B,T)
        labels = b.get("labels")
        if labels is not None:
            labels = labels.cpu()

        B = pred_ids.size(0)
        for i in range(B):
            hyp_toks = collapse_and_map(pred_ids[i].tolist())
            hyp = " ".join(hyp_toks)

            if labels is not None:
                tgt = [int(t) for t in labels[i].tolist() if int(t) != blank_idx]
                ref = " ".join(id2gloss.get(int(t), "<UNK>") for t in tgt)
            else:
                ref = ""

            hw, rw = hyp.split(), ref.split()
            total_wer_edits += ed(hw, rw); total_wer_tokens += max(1, len(rw))
            total_cer_edits += ed(list(hyp), list(ref)); total_cer_chars += max(1, len(ref))
            N += 1

    WER = 100.0 * total_wer_edits / max(1, total_wer_tokens)
    CER = 100.0 * total_cer_edits / max(1, total_cer_chars)
    return WER, CER, N


# -------------------------------
# Main
# -------------------------------
def main():
    args = get_args()
    warnings.filterwarnings("once", category=FutureWarning)
    device = args.device
    map_location = torch.device(device)

    # Dataset (same as training) + collate
    ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix, qgrid_prefix=args.qgrid_prefix, kp_path=args.kp_path,
        meta_dir_path=args.meta_dir, gloss_dict_path=args.gloss_dict, split="dev" if args.split=="val" else args.split,
    )
    # Build id2gloss: invert and add blank 0
    id2gloss = load_id2gloss(args.gloss_dict)
    # True class count = (blank + glosses)
    n_classes_from_gloss = len(id2gloss)  # should be 1296 with your file (1295 + blank)

    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=multi_modal_collate_fn, drop_last=False,
    )

    # Load ckpt first to sniff depth/head
    print(f"[ckpt] loading: {args.ckpt}")
    try:
        ckpt_obj = torch.load(args.ckpt, map_location=map_location, weights_only=True)
    except TypeError:
        ckpt_obj = torch.load(args.ckpt, map_location=map_location)

    state_dict = ckpt_obj.get("state_dict") if isinstance(ckpt_obj, dict) else None
    if state_dict is None and isinstance(ckpt_obj, dict) and all(isinstance(k, str) for k in ckpt_obj.keys()):
        state_dict = ckpt_obj
    if state_dict is None and isinstance(ckpt_obj, dict):
        state_dict = ckpt_obj.get("model", ckpt_obj.get("model_state", {})) or {}

    # strip prefixes
    def _strip(sd: Dict[str, Any], prefixes=("module.", "model.")):
        out = {}
        for k, v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p): nk = nk[len(p):]
            out[nk] = v
        return out
    state_dict = _strip(state_dict)

    # infer n_layer from ckpt (fallback to CLI)
    n_layer_infer = infer_n_layer_from_ckpt(state_dict, args.n_layer)
    # choose num_classes; prefer id2gloss (+blank)
    n_classes = n_classes_from_gloss

    print(f"[head] num_classes={n_classes} (from gloss_dict + blank)")
    print(f"[arch] n_layer inferred from ckpt: {n_layer_infer}")

    # Build model (match training hparams) and wrap FE for 5D
    model = MultiModalMamba(
        d_model=args.d_model, n_layer=n_layer_infer,
        fusion_embed=args.fusion_embed, fusion_heads=args.fusion_heads,
        num_classes=n_classes, max_kv=args.max_kv, pool_mode=args.pool_mode,
    ).to(device)
    wrap_frame_encoder_for_5d(model)

    # Materialize lazy layers with a dummy forward (like training)
    try:
        first = next(iter(dl))
        b = unpack_batch(first)
        with torch.no_grad():
            _ = model(
                b["images"].to(device, non_blocking=True),
                (b["qgrids"].to(device, non_blocking=True) if b["qgrids"] is not None else None),
                (b["keypoints"].to(device, non_blocking=True) if b["keypoints"] is not None else None),
                (b["qgrid_lengths"].to(device, non_blocking=True) if b.get("qgrid_lengths") is not None else None),
            )
    except StopIteration:
        pass

    # Remap any head names in ckpt to classifier.*
    state_dict = remap_head_to_classifier(state_dict)

    # Load weights (require classifier load to succeed)
    ld = model.load_state_dict(state_dict, strict=False)
    missing = getattr(ld, "missing_keys", None) or (ld.get("missing_keys", []) if isinstance(ld, dict) else [])
    unexpected = getattr(ld, "unexpected_keys", None) or (ld.get("unexpected_keys", []) if isinstance(ld, dict) else [])
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[ckpt] missing keys: {len(missing)}")

    # Ensure classifier actually loaded
    if any(k.startswith("classifier.") for k in missing):
        raise RuntimeError(
            "Classifier head from checkpoint did not load. "
            "This means num_classes or model hyperparams differ from training. "
            "Make sure gloss_dict matches the training run (blank + labels)."
        )

    # Fresh loader (we consumed 1 batch to materialize)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=multi_modal_collate_fn, drop_last=False,
    )

    use_bf16 = bool(args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    WER, CER, N = evaluate(model, dl, id2gloss, args.device, use_bf16, blank_idx=args.blank_idx)
    print(f"[RESULT] N={N}  WER={WER:.2f}%  CER={CER:.2f}%")


if __name__ == "__main__":
    main()

