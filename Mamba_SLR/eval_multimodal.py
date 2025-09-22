#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import inspect
import math
from typing import List, Dict, Tuple, Union

import torch
from torch.utils.data import DataLoader

from slr.datasets.multi_modal_datasets import (
    MultiModalPhoenixDataset,
    multi_modal_collate_fn,
)
from slr.models.multi_modal_model import MultiModalMamba


# ----------------------------
# Gloss utilities
# ----------------------------
def load_id2gloss(gloss_path: str) -> List[str]:
    import numpy as np
    arr = np.load(gloss_path, allow_pickle=True)
    if getattr(arr, "shape", None) == () and getattr(arr, "dtype", None) == object:
        obj = arr.item()
    else:
        obj = arr

    if isinstance(obj, dict):
        max_id = max(obj.values())
        id2gloss = ["<blank>"] * (max_id + 1)
        for g, i in obj.items():
            id2gloss[int(i)] = str(g)
        id2gloss[0] = "<blank>"
        return id2gloss
    elif isinstance(obj, (list, tuple)):
        id2gloss = [str(x) for x in obj]
        if not id2gloss:
            raise AssertionError("empty gloss list")
        if id2gloss[0].lower() not in {"<blank>", "blank", "<pad>", ""}:
            id2gloss[0] = "<blank>"
        return id2gloss
    else:
        raise AssertionError("gloss_dict should be list-like or dict-like")


# ----------------------------
# Edit distance + metrics
# ----------------------------
def _lev(a: List[str], b: List[str]) -> int:
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
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def wer(ref: List[str], hyp: List[str]) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _lev(ref, hyp) / len(ref)


def cer(ref: str, hyp: str) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _lev(list(ref), list(hyp)) / len(ref)


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


# ----------------------------
# Dataset construction (signature-robust)
# ----------------------------
def make_dataset(split: str, args) -> MultiModalPhoenixDataset:
    sig = inspect.signature(MultiModalPhoenixDataset.__init__)
    allowed = set(sig.parameters.keys())

    kwargs = {}
    if "split" in allowed:
        kwargs["split"] = split
    if "image_prefix" in allowed:
        kwargs["image_prefix"] = args.image_prefix
    if "qgrid_prefix" in allowed:
        kwargs["qgrid_prefix"] = args.qgrid_prefix
    if "kp_path" in allowed and args.kp_path is not None:
        kwargs["kp_path"] = args.kp_path

    # meta path
    if "meta_dir_path" in allowed:
        kwargs["meta_dir_path"] = args.meta_dir
    elif "meta_dir" in allowed:
        kwargs["meta_dir"] = args.meta_dir

    # gloss dict path
    if "gloss_dict_path" in allowed:
        kwargs["gloss_dict_path"] = args.gloss_dict
    elif "gloss_dict" in allowed:
        kwargs["gloss_dict"] = args.gloss_dict

    return MultiModalPhoenixDataset(**kwargs)


# ----------------------------
# Checkpoint helpers
# ----------------------------
def load_checkpoint(ckpt_path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def find_head_class_sizes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    head_names = [
        "classifier.weight",
        "head.weight",
        "lm_head.weight",
        "output.weight",
        "proj.weight",
        "final.weight",
    ]
    sizes = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v) and v.dim() == 2:
            for hn in head_names:
                if k.endswith(hn):
                    sizes[hn] = v.shape[0]
    return sizes


def infer_num_classes(state_dict: Dict[str, torch.Tensor], gloss_len: int) -> Tuple[int, str]:
    head_sizes = find_head_class_sizes(state_dict)
    if "classifier.weight" in head_sizes:
        return head_sizes["classifier.weight"], "ckpt(classifier.weight)"
    if head_sizes:
        hn, sz = max(head_sizes.items(), key=lambda kv: kv[1])
        return sz, f"ckpt({hn})"
    return gloss_len, "gloss_dict_len"


def load_weights_lenient(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    own = model.state_dict()
    compatible = {}
    incompatible = []
    for k, v in state_dict.items():
        if k in own and torch.is_tensor(v) and own[k].shape == v.shape:
            compatible[k] = v
        else:
            incompatible.append(k)
    model.load_state_dict({**own, **compatible})
    if incompatible:
        print(f"[ckpt] skipped {len(incompatible)} incompatible keys (likely classifier or arch diff):")
        for k in incompatible[:8]:
            shp = tuple(state_dict[k].shape) if torch.is_tensor(state_dict[k]) else "obj"
            print(f"        - {k}  {shp}")
        if len(incompatible) > 8:
            print(f"        ... (+{len(incompatible)-8} more)")


# ----------------------------
# Batch normalization (robust to list/tuple)
# ----------------------------
def normalize_batch(batch: Union[dict, list, tuple]) -> dict:
    """
    Ensure we always end up with a dict like training:
      keys: images, qgrids, keypoints, qgrid_lengths, labels (optional)
    """
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (list, tuple)):
        # common cases:
        # 1) [ {sample1}, {sample2}, ... ]  -> fold with our collate
        if len(batch) > 0 and all(isinstance(b, dict) for b in batch):
            return multi_modal_collate_fn(batch)
        # 2) [ {batched_dict} ] -> unwrap
        if len(batch) == 1 and isinstance(batch[0], dict):
            return batch[0]
    raise TypeError(f"Unexpected batch type: {type(batch)}")


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model, dataloader, id2gloss: List[str], device: torch.device, use_bf16: bool):
    model.eval()
    blank = 0
    total_wer = 0.0
    total_cer = 0.0
    n_utts = 0

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else torch.amp.autocast("cuda", enabled=False)
    )

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = normalize_batch(raw_batch)

            images = batch["images"].to(device, non_blocking=True)
            qgrids = batch["qgrids"].to(device, non_blocking=True)
            keypoints = batch["keypoints"].to(device, non_blocking=True)
            qgrid_lengths = batch["qgrid_lengths"].to(device, non_blocking=True)
            labels_list = batch.get("labels", None)

            with autocast_ctx:
                logits = model(images, qgrids, keypoints, qgrid_lengths)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                pred = logits.argmax(dim=-1)

            B = pred.size(0)
            for b in range(B):
                T = int(qgrid_lengths[b].item()) if qgrid_lengths is not None else pred.size(1)
                raw_ids = pred[b, :T].tolist()
                hyp_ids = ctc_collapse_and_strip_blanks(raw_ids, blank=blank)
                hyp_tokens = [id2gloss[i] if 0 <= i < len(id2gloss) else f"<oob:{i}>" for i in hyp_ids]

                if labels_list is not None:
                    ref_ids = labels_list[b].tolist()
                    ref_tokens = [id2gloss[i] if 0 <= i < len(id2gloss) else f"<oob:{i}>" for i in ref_ids]
                else:
                    # if eval split has no labels, skip metrics accumulation
                    continue

                total_wer += wer(ref_tokens, hyp_tokens)
                total_cer += cer(" ".join(ref_tokens), " ".join(hyp_tokens))
                n_utts += 1

    if n_utts == 0:
        return math.nan, math.nan, 0
    return total_wer / n_utts, total_cer / n_utts, n_utts


def main():
    p = argparse.ArgumentParser(description="Evaluate MultiModalMamba on PHOENIX14 with greedy CTC decode")

    # data (defaults set to your box)
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

    # model shape fallbacks
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--fusion_embed", type=int, default=512)
    p.add_argument("--fusion_heads", type=int, default=8)
    p.add_argument("--max_kv", type=int, default=1024)
    p.add_argument("--pool_mode", default="mean", choices=["mean", "max", "vote"])

    # override hook
    p.add_argument("--force_num_classes", type=int, default=None, help="override class count (else inferred from ckpt head, else gloss dict length)")

    args = p.parse_args()

    # 1) id->gloss
    id2gloss = load_id2gloss(args.gloss_dict)
    print(f"[gloss] loaded {len(id2gloss)} entries; blank idx=0; sample: {id2gloss[:5]}")

    # 2) dataset / loader
    ds = make_dataset(args.split, args)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=multi_modal_collate_fn,  # even if this is bypassed, normalize_batch() will recover
    )

    # 3) model from ckpt
    state = load_checkpoint(args.ckpt, map_location="cpu")

    if args.force_num_classes is not None:
        num_classes = args.force_num_classes
        reason = "forced"
    else:
        num_classes, reason = infer_num_classes(state, gloss_len=len(id2gloss))

    print(f"[head] num_classes={num_classes} ({reason})")

    model = MultiModalMamba(
        d_model=args.d_model,
        n_layer=args.n_layer,
        fusion_embed=args.fusion_embed,
        fusion_heads=args.fusion_heads,
        num_classes=num_classes,
        max_kv=args.max_kv,
        pool_mode=args.pool_mode,
    )

    # Try strict load; if mismatch, fall back to lenient load
    try:
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError("strict load produced missing/unexpected keys")
        print("[ckpt] loaded strict.")
    except Exception:
        print("[ckpt] strict load failed -> loading non-head layers and reinitializing classifier")
        load_weights_lenient(model, state)

    device = torch.device(args.device)
    model.to(device)

    # 4) eval
    use_bf16 = args.bf16 and device.type == "cuda"
    WER, CER, N = evaluate(model, dl, id2gloss, device, use_bf16=use_bf16)
    print(f"[eval:{args.split}] N={N}  WER={WER:.4f}  CER={CER:.4f}")


if __name__ == "__main__":
    main()
