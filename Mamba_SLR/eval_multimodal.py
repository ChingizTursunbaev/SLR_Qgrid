#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import inspect
import math
from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader

# local imports
from slr.datasets.multi_modal_datasets import (
    MultiModalPhoenixDataset,
    multi_modal_collate_fn,
)
from slr.models.multi_modal_model import MultiModalMamba


# ----------------------------
# Gloss utilities
# ----------------------------
def load_id2gloss(gloss_path: str) -> List[str]:
    """
    Accepts:
      - numpy scalar object containing dict(gloss->id) with ids 1..K
      - numpy array/list of gloss strings (id->gloss)
    Returns id2gloss list with id2gloss[0] = "<blank>" and len = max_id+1
    """
    import numpy as np

    arr = np.load(gloss_path, allow_pickle=True)
    # If it's a numpy scalar object holding a dict, arr.shape == ()
    if getattr(arr, "shape", None) == () and getattr(arr, "dtype", None) == object:
        obj = arr.item()
    else:
        obj = arr

    if isinstance(obj, dict):
        # gloss -> id with ids 1..K; reserve 0 for blank
        max_id = max(obj.values())
        id2gloss = ["<blank>"] * (max_id + 1)
        for g, i in obj.items():
            id2gloss[int(i)] = str(g)
        # Ensure 0 is explicitly a blank token
        id2gloss[0] = "<blank>"
        return id2gloss
    elif isinstance(obj, (list, tuple)):
        # Assume already id->gloss; ensure 0 is a blank-like symbol
        id2gloss = [str(x) for x in obj]
        if not id2gloss or id2gloss[0].lower() not in {"<blank>", "blank", "<pad>", ""}:
            # Be explicit
            id2gloss[0] = "<blank>"
        return id2gloss
    else:
        raise AssertionError("gloss_dict should be list-like or dict-like")


# ----------------------------
# Decoding + metrics
# ----------------------------
def ctc_collapse_and_strip_blanks(ids: List[int], blank: int = 0) -> List[int]:
    out = []
    prev = None
    for x in ids:
        if x == blank:
            prev = x
            continue
        if x != prev:
            out.append(x)
        prev = x
    return out

def levenshtein(a: List[str], b: List[str]) -> int:
    # simple DP
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # delete
                dp[i][j - 1] + 1,      # insert
                dp[i - 1][j - 1] + cost,  # substitute
            )
    return dp[n][m]

def wer(ref: List[str], hyp: List[str]) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return levenshtein(ref, hyp) / len(ref)

def cer(ref: str, hyp: str) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return levenshtein(list(ref), list(hyp)) / len(ref)


# ----------------------------
# Safe dataset construction
# ----------------------------
def make_dataset(split: str, args) -> MultiModalPhoenixDataset:
    """Pass only the kwargs that the current class signature supports."""
    sig = inspect.signature(MultiModalPhoenixDataset.__init__)
    allowed = set(sig.parameters.keys())
    # Build a candidate kwargs dict
    candidates = dict(
        split=split,
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir=args.meta_dir,
        gloss_dict=args.gloss_dict,  # some versions accept this too
    )
    kwargs = {k: v for k, v in candidates.items() if k in allowed and v is not None}
    return MultiModalPhoenixDataset(**kwargs)


# ----------------------------
# Checkpoint utilities
# ----------------------------
def load_checkpoint(ckpt_path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt

def infer_num_classes_from_state(state_dict: Dict[str, torch.Tensor], fallback: int = 1296) -> int:
    # Prefer common classifier head names
    head_keys = [
        "classifier.weight",
        "proj.weight",
        "head.weight",
        "final.weight",
        "lm_head.weight",
        "output.weight",
    ]
    for k in state_dict.keys():
        for hk in head_keys:
            if k.endswith(hk) and state_dict[k].dim() == 2:
                return state_dict[k].shape[0]
    # If not found, last resort: look for any 2D weight that matches [C, d_model] where C is large
    candidates = [(k, v.shape[0]) for k, v in state_dict.items() if torch.is_tensor(v) and v.dim() == 2]
    if candidates:
        # choose the one with the largest C (likely classifier)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][1]
    return fallback


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model, dataloader, id2gloss: List[str], device: torch.device, use_bf16: bool):
    model.eval()
    blank = 0
    total_wer = 0.0
    total_cer = 0.0
    n_utts = 0

    autocast = torch.cuda.amp.autocast if use_bf16 else torch.cpu.amp.autocast
    amp_dtype = torch.bfloat16 if use_bf16 else None

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device, non_blocking=True)
            qgrids = batch["qgrids"].to(device, non_blocking=True)
            keypoints = batch["keypoints"].to(device, non_blocking=True)
            qgrid_lengths = batch["qgrid_lengths"].to(device, non_blocking=True)

            # labels may be list[Tensor], variable-length per sample
            labels_list = batch.get("labels", None)

            with (autocast(dtype=amp_dtype) if use_bf16 else torch.no_grad()):
                logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B, T, C)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                # greedy decode
                pred = logits.argmax(dim=-1)  # (B, T)

            B = pred.size(0)
            for b in range(B):
                T = int(qgrid_lengths[b].item()) if qgrid_lengths is not None else pred.size(1)
                raw_ids = pred[b, :T].tolist()
                hyp_ids = ctc_collapse_and_strip_blanks(raw_ids, blank=blank)
                # map to tokens
                hyp_tokens = [id2gloss[i] if 0 <= i < len(id2gloss) else f"<oob:{i}>" for i in hyp_ids]

                # reference
                if labels_list is not None:
                    ref_ids = labels_list[b].tolist()
                    ref_tokens = [id2gloss[i] if 0 <= i < len(id2gloss) else f"<oob:{i}>" for i in ref_ids]
                else:
                    # if labels missing, skip metrics
                    continue

                total_wer += wer(ref_tokens, hyp_tokens)
                # join tokens with spaces to compute CER on characters of the string
                total_cer += cer(" ".join(ref_tokens), " ".join(hyp_tokens))
                n_utts += 1

    if n_utts == 0:
        return math.nan, math.nan, 0
    return total_wer / n_utts, total_cer / n_utts, n_utts


def main():
    p = argparse.ArgumentParser(description="Evaluate MultiModalMamba on PHOENIX14 with greedy CTC decode")
    # data
    p.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")

    # runtime
    p.add_argument("--ckpt",   required=True, help="path to checkpoint, e.g., checkpoints/multimodal_ddp/best.pt")
    p.add_argument("--split",  default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # model shape fallbacks (only used if not inferred from ckpt)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--fusion_embed", type=int, default=512)
    p.add_argument("--fusion_heads", type=int, default=8)
    p.add_argument("--max_kv", type=int, default=1024)
    p.add_argument("--pool_mode", default="mean", choices=["mean","max","vote"])

    args = p.parse_args()

    # 1) id->gloss
    id2gloss = load_id2gloss(args.gloss_dict)
    print(f"[gloss] loaded {len(id2gloss)} entries; blank idx=0; sample: {id2gloss[:5]}")

    # 2) dataset & loader (robust to ctor changes)
    ds = make_dataset(args.split, args)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=multi_modal_collate_fn,
    )

    # 3) build model with num_classes inferred from ckpt
    state = load_checkpoint(args.ckpt, map_location="cpu")
    num_classes = infer_num_classes_from_state(state, fallback=len(id2gloss))
    if num_classes != len(id2gloss):
        print(f"[warn] ckpt head implies num_classes={num_classes}, but gloss dict len={len(id2gloss)}. Proceeding with ckpt value.")
    model = MultiModalMamba(
        d_model=args.d_model,
        n_layer=args.n_layer,
        fusion_embed=args.fusion_embed,
        fusion_heads=args.fusion_heads,
        num_classes=num_classes,
        max_kv=args.max_kv,
        pool_mode=args.pool_mode,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded. missing={len(missing)} unexpected={len(unexpected)}")

    device = torch.device(args.device)
    model.to(device)

    # 4) run evaluation
    use_bf16 = args.bf16 and device.type == "cuda"
    WER, CER, N = evaluate(model, dl, id2gloss, device, use_bf16=use_bf16)
    print(f"[eval:{args.split}] N={N}  WER={WER:.4f}  CER={CER:.4f}")


if __name__ == "__main__":
    main()
