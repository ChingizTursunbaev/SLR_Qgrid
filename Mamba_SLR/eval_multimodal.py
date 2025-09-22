#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slr.datasets.multi_modal_datasets import (
    MultiModalPhoenixDataset,
    multi_modal_collate_fn,
)
from slr.models.multi_modal_model import MultiModalMamba


# ---------------------------
# Gloss dictionary utilities
# ---------------------------
def load_id2gloss(gloss_path: str) -> List[str]:
    """Accepts dict-like npy (gloss->id) OR list-like id->gloss."""
    arr = np.load(gloss_path, allow_pickle=True)
    # dict-like stored in object scalar
    if getattr(arr, "shape", None) == () and getattr(arr, "dtype", None) == object:
        obj = arr.item()
        # build id->gloss; keep 0 as <blank>
        max_id = max(int(v) for v in obj.values())
        id2gloss = ["<blank>"] * (max_id + 1)
        for g, i in obj.items():
            id2gloss[int(i)] = str(g)
        if id2gloss and (not id2gloss[0] or id2gloss[0].lower() not in {"<blank>", "blank"}):
            id2gloss[0] = "<blank>"
        return id2gloss
    # list-like
    if hasattr(arr, "tolist"):
        id2gloss = arr.tolist()
    else:
        id2gloss = list(arr)
    if id2gloss and (not id2gloss[0] or id2gloss[0].lower() not in {"<blank>", "blank"}):
        id2gloss[0] = "<blank>"
    return id2gloss


# ---------------------------
# Decoding + metrics
# ---------------------------
def ctc_greedy_decode(logits: torch.Tensor, blank: int = 0) -> List[List[int]]:
    """
    logits: (B, T, C) unnormalized or log-probs are fine for argmax.
    Return: list of token id sequences (CTC-collapsed, blank removed).
    """
    with torch.no_grad():
        pred = logits.argmax(dim=-1)  # (B, T)
    B, T = pred.shape
    out: List[List[int]] = []
    for b in range(B):
        seq = []
        prev = None
        for t in range(T):
            p = int(pred[b, t])
            if p != blank and p != prev:
                seq.append(p)
            prev = p
        out.append(seq)
    return out


def _levenshtein(a: List[Any], b: List[Any]) -> int:
    # classic DP
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def wer(ref: List[str], hyp: List[str]) -> float:
    return _levenshtein(ref, hyp) / max(1, len(ref))


def cer(ref: List[str], hyp: List[str]) -> float:
    ref_chars = [c for w in ref for c in w]
    hyp_chars = [c for w in hyp for c in w]
    return _levenshtein(ref_chars, hyp_chars) / max(1, len(ref_chars))


# ---------------------------
# Batch normalization
# ---------------------------
def normalize_batch(batch: Any) -> Dict[str, Any]:
    """
    Accept both dict batches and your tuple batches.
    Required keys for model: images, qgrids, keypoints, qgrid_lengths
    Optional (if present, for metrics): labels, label_lengths
    """
    if isinstance(batch, dict):
        out = dict(batch)  # shallow copy
        return out

    if isinstance(batch, (list, tuple)):
        # From your inspection:
        # [0] images (B, T, 3, 224, 224)
        # [1] qgrids (B, Lq_max, 242)
        # [2] keypoints (B, T, 242)
        # [3] qgrid_lengths (B, 12)  # multi-lengths packed; we’ll take [:,0] if needed below
        images = batch[0]
        qgrids = batch[1]
        keypoints = batch[2]
        qgrid_lengths_raw = batch[3]

        # Try to extract a single length value per sample for qgrid (first column),
        # but keep the original if it is already 1D or clearly per-sample.
        if torch.is_tensor(qgrid_lengths_raw):
            if qgrid_lengths_raw.dim() == 2 and qgrid_lengths_raw.size(1) >= 1:
                qgrid_lengths = qgrid_lengths_raw[:, 0]
            else:
                qgrid_lengths = qgrid_lengths_raw
        else:
            qgrid_lengths = qgrid_lengths_raw

        out: Dict[str, Any] = {
            "images": images,            # (B, T, 3, H, W)
            "qgrids": qgrids,            # (B, Lq, 242)
            "keypoints": keypoints,      # (B, T, 242)
            "qgrid_lengths": qgrid_lengths,  # (B,) best effort
        }

        # Heuristics to find labels & lengths if collate included them
        # Scan remaining items for int tensors that look like (B, L_lab) and a (B,) length.
        for i in range(4, len(batch)):
            x = batch[i]
            if not torch.is_tensor(x):
                continue
            if x.dtype in (torch.int32, torch.int64, torch.long):
                # candidate label sequence (B, L)
                if x.dim() == 2 and x.size(0) == images.size(0):
                    out.setdefault("labels", x)
                # candidate lengths (B,)
                elif x.dim() == 1 and x.size(0) == images.size(0):
                    # Prefer to name it 'label_lengths' if labels exist
                    if "labels" in out and "label_lengths" not in out:
                        out["label_lengths"] = x
                    elif "label_lengths" not in out:
                        out["label_lengths"] = x
        return out

    raise TypeError(f"Unexpected batch type: {type(batch)}")


# ---------------------------
# Build dataset / loader
# ---------------------------
def make_dataset(split: str, args) -> MultiModalPhoenixDataset:
    return MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,
        gloss_dict_path=args.gloss_dict,
        split=split,
        transforms=None,
    )


# ---------------------------
# Build model from checkpoint
# ---------------------------
def infer_num_classes_from_ckpt(state: Dict[str, torch.Tensor]) -> Optional[int]:
    # Prefer classifier weight if present
    head_w = state.get("classifier.weight", None)
    if isinstance(head_w, torch.Tensor) and head_w.dim() == 2:
        return head_w.size(0)
    return None


def build_model(args, num_classes: int) -> nn.Module:
    # Keep these consistent with training (d_model, n_layer, etc.) if they were constants.
    model = MultiModalMamba(
        d_model=512,
        n_layer=8,
        fusion_embed=512,
        fusion_heads=8,
        num_classes=num_classes,
        max_kv=args.max_kv,
        pool_mode=args.pool_mode,
    )
    return model


# ---------------------------
# Evaluation loop
# ---------------------------
@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dl: DataLoader,
    id2gloss: List[str],
    device: torch.device,
    use_bf16: bool,
) -> Tuple[Optional[float], Optional[float], int]:
    model.eval()
    has_labels = False
    total_wer, total_cer, Nseq = 0.0, 0.0, 0

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.cuda.amp.autocast(enabled=False)
    )

    for raw in dl:
        batch = normalize_batch(raw)

        images = batch["images"].to(device, non_blocking=True)          # (B,T,3,H,W)
        qgrids = batch["qgrids"].to(device, non_blocking=True)          # (B,Lq,242)
        keypoints = batch["keypoints"].to(device, non_blocking=True)    # (B,T,242)
        qgrid_lengths = batch["qgrid_lengths"]
        if torch.is_tensor(qgrid_lengths):
            qgrid_lengths = qgrid_lengths.to(device, non_blocking=True)

        with amp_ctx:
            logits = model(images, qgrids, keypoints, qgrid_lengths)    # (B,T',C)

        # Try to gather targets if present
        labels = batch.get("labels", None)
        label_lengths = batch.get("label_lengths", None)
        if labels is not None:
            has_labels = True
            # decode predictions
            pred_ids = ctc_greedy_decode(logits, blank=0)
            B = len(pred_ids)

            # build reference sequences from labels
            if torch.is_tensor(labels):
                labels_np = labels.cpu().tolist()
            else:
                labels_np = labels
            if torch.is_tensor(label_lengths):
                lens_np = label_lengths.cpu().tolist()
            else:
                lens_np = label_lengths

            for b in range(B):
                hyp_ids = pred_ids[b]
                # truncate ref by length if provided
                if lens_np is not None:
                    L = int(lens_np[b])
                    ref_ids = labels_np[b][:L]
                else:
                    ref_ids = labels_np[b]
                # map to gloss tokens and ignore blanks
                hyp = [id2gloss[i] for i in hyp_ids if 0 <= i < len(id2gloss) and i != 0]
                ref = [id2gloss[i] for i in ref_ids if 0 <= i < len(id2gloss) and i != 0]

                total_wer += wer(ref, hyp)
                total_cer += cer(ref, hyp)
                Nseq += 1

    if has_labels and Nseq > 0:
        return total_wer / Nseq, total_cer / Nseq, Nseq
    else:
        return None, None, 0


# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    # data (match your training defaults)
    p.add_argument("--image_prefix", default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")

    # loader
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)

    # model
    p.add_argument("--max_kv", type=int, default=1024)
    p.add_argument("--pool_mode", choices=["mean", "max", "vote"], default="mean")
    p.add_argument("--force_num_classes", type=int, default=None)

    # ckpt + dtype
    p.add_argument("--ckpt", required=True)
    p.add_argument("--bf16", action="store_true")

    args = p.parse_args()

    # gloss dict
    id2gloss = load_id2gloss(args.gloss_dict)
    print(f"[gloss] loaded {len(id2gloss)} entries; blank idx=0; sample: {id2gloss[:5]}")

    # dataset + loader
    ds = make_dataset(args.split, args)
    print(f"[MultiModalPhoenixDataset] len={len(ds)} split={args.split}")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=multi_modal_collate_fn,
    )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load ckpt
    map_location = "cpu" if device.type == "cpu" else {"cuda:0": f"cuda:{torch.cuda.current_device()}"}
    ckpt_path = args.ckpt
    ckpt = torch.load(ckpt_path, map_location=map_location)

    state: Dict[str, torch.Tensor]
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        # raw state_dict
        state = ckpt

    # decide num_classes
    if args.force_num_classes is not None:
        num_classes = args.force_num_classes
        print(f"[head] num_classes={num_classes} (forced)")
    else:
        inferred = infer_num_classes_from_ckpt(state)
        if inferred is not None:
            num_classes = inferred
            print(f"[head] num_classes inferred from ckpt: {num_classes}")
        else:
            num_classes = len(id2gloss)
            print(f"[head] num_classes from gloss_dict: {num_classes}")

    # build & load model
    model = build_model(args, num_classes=num_classes)
    try:
        missing, unexpected = model.load_state_dict(state, strict=True)
        print(f"[ckpt] strict load ok. missing={len(missing)} unexpected={len(unexpected)}")
    except Exception as e:
        print(f"[ckpt] strict load failed ({e}) -> loading non-head layers and reinitializing classifier")
        # try to load everything except the classifier (head)
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_dict and not k.startswith("classifier.")}
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)
        # (classifier stays randomly initialized for num_classes)

    model.to(device)
    model.eval()

    # evaluate
    WER, CER, N = evaluate(model, dl, id2gloss, device, use_bf16=args.bf16)
    if WER is None:
        print("[eval] Ran forward pass successfully, but labels were not found in batch -> WER/CER skipped.")
        print("       If your collate provides labels, they may be at indices >3; we try to auto-detect.")
    else:
        print(f"[eval] WER={WER:.4f}  CER={CER:.4f}  over {N} samples.")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))  # respect your env
    main()
