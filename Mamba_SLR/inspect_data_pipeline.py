#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deterministically inspect the PHOENIX14 multi-modal data pipeline:
- Dataset __init__ signature + file path
- __getitem__ output (type/keys/shapes)
- Collated batch output from DataLoader (type/keys/shapes)
- Gloss dict structure

Run examples:
  python inspect_data_pipeline.py --split train
  python inspect_data_pipeline.py --split val
  python inspect_data_pipeline.py --split test
"""

import argparse
import inspect
import os
from pprint import pprint
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from slr.datasets.multi_modal_datasets import (
    MultiModalPhoenixDataset,
    multi_modal_collate_fn,
)


def _shape(x: Any):
    if torch.is_tensor(x):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)) and len(x) and torch.is_tensor(x[0]):
        # list/tuple of tensors
        return [tuple(t.shape) for t in x]
    return None


def _summarize_dict(d: dict, name: str):
    print(f"\n[{name}] dict keys: {list(d.keys())}")
    for k, v in d.items():
        tname = type(v).__name__
        shp = _shape(v)
        if shp is None:
            try:
                l = len(v)  # might be a list
                print(f"  - {k}: type={tname} len={l}")
            except Exception:
                print(f"  - {k}: type={tname}")
        else:
            print(f"  - {k}: type={tname} shape={shp}")


def _summarize_sequence(seq: Iterable, name: str, max_items: int = 2):
    print(f"\n[{name}] {type(seq).__name__} length={len(seq)}")
    for i, it in enumerate(seq):
        if i >= max_items:
            print(f"  ... (showing only first {max_items})")
            break
        tname = type(it).__name__
        shp = _shape(it)
        if isinstance(it, dict):
            print(f"  [{i}] dict -> keys={list(it.keys())}")
            for k, v in it.items():
                print(f"     .{k}: type={type(v).__name__}, shape={_shape(v)}")
        elif shp is not None:
            print(f"  [{i}] type={tname}, shape={shp}")
        else:
            try:
                l = len(it)
                print(f"  [{i}] type={tname}, len={l}")
            except Exception:
                print(f"  [{i}] type={tname}")


def load_id2gloss(gloss_path: str):
    arr = np.load(gloss_path, allow_pickle=True)
    print("\n[gloss_dict numpy repr]")
    print(type(arr), getattr(arr, "shape", None), getattr(arr, "dtype", None))

    if getattr(arr, "shape", None) == () and getattr(arr, "dtype", None) == object:
        obj = arr.item()
        print(f"-> detected dict with {len(obj)} items (min_id={min(obj.values())}, max_id={max(obj.values())})")
        # Rebuild id->gloss array
        max_id = max(obj.values())
        id2gloss = ["<blank>"] * (max_id + 1)
        for g, i in obj.items():
            id2gloss[int(i)] = str(g)
        id2gloss[0] = "<blank>"
        return id2gloss
    elif isinstance(arr, (list, tuple)) or (hasattr(arr, "dtype") and arr.dtype.kind in ("U", "S", "O")):
        # assume list-like
        id2gloss = arr.tolist() if hasattr(arr, "tolist") else list(arr)
        print(f"-> detected list-like with length={len(id2gloss)}; sample={id2gloss[:5]}")
        if id2gloss and id2gloss[0].lower() not in {"<blank>", "blank", "<pad>", ""}:
            id2gloss[0] = "<blank>"
        return id2gloss
    else:
        raise AssertionError("gloss_dict should be list-like or dict-like")


def main():
    p = argparse.ArgumentParser()
    # Use your known-good defaults from training
    p.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    # 0) Dataset signature + file path
    print("=== MultiModalPhoenixDataset.__init__ signature ===")
    sig = inspect.signature(MultiModalPhoenixDataset.__init__)
    print(sig)
    try:
        src = inspect.getsourcefile(MultiModalPhoenixDataset) or "<unknown file>"
    except Exception:
        src = "<unknown file>"
    print(f"defined in: {src}")

    # 1) Load gloss dict
    id2gloss = load_id2gloss(args.gloss_dict)
    print(f"\n[gloss] length={len(id2gloss)} blank_idx=0 sample={id2gloss[:5]}")

    # 2) Build dataset with only the params it supports
    allowed = set(sig.parameters.keys())
    kwargs = {}
    if "split" in allowed:
        kwargs["split"] = args.split
    if "image_prefix" in allowed:
        kwargs["image_prefix"] = args.image_prefix
    if "qgrid_prefix" in allowed:
        kwargs["qgrid_prefix"] = args.qgrid_prefix
    if "kp_path" in allowed:
        kwargs["kp_path"] = args.kp_path
    if "meta_dir_path" in allowed:
        kwargs["meta_dir_path"] = args.meta_dir
    elif "meta_dir" in allowed:
        kwargs["meta_dir"] = args.meta_dir
    if "gloss_dict_path" in allowed:
        kwargs["gloss_dict_path"] = args.gloss_dict
    elif "gloss_dict" in allowed:
        kwargs["gloss_dict"] = args.gloss_dict

    print("\n[dataset kwargs actually used]")
    pprint(kwargs)

    ds = MultiModalPhoenixDataset(**kwargs)
    print(f"\n[dataset] len={len(ds)}  split={args.split}")

    # 3) Inspect a single sample (what __getitem__ returns)
    sample = ds[0]
    print(f"\n[__getitem__ sample type] {type(sample).__name__}")
    if isinstance(sample, dict):
        _summarize_dict(sample, "__getitem__")
    elif isinstance(sample, (list, tuple)):
        _summarize_sequence(sample, "__getitem__", max_items=8)
    else:
        print(sample)

    # 4) Build DataLoader w/ your collate_fn and inspect one batch
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=multi_modal_collate_fn,
    )
    batch = next(iter(dl))
    print(f"\n[dataloader batch type] {type(batch).__name__}")

    if isinstance(batch, dict):
        _summarize_dict(batch, "batch")
    elif isinstance(batch, (list, tuple)):
        _summarize_sequence(batch, "batch", max_items=4)
    else:
        print(batch)

    print("\n[done] If batch is not a dict with keys "
          "['images','qgrids','keypoints','qgrid_lengths',(labels)], "
          "we now know exactly what it IS, and can adapt eval/train accordingly.")


if __name__ == "__main__":
    # make sure we don't accidentally use CUDA; this is pure inspection
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
