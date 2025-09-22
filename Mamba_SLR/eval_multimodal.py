#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# external metrics
try:
    from jiwer import wer, cer
except Exception:
    raise SystemExit("Please install jiwer: pip install jiwer")

# repo imports (match training)
from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.models.multi_modal_model import MultiModalMamba


def _load_gloss_dict(path, assume_blank_idx=0):
    """Load gloss dictionary from .npy and return (id2gloss:list[str], blank_idx:int)."""
    arr = np.load(path, allow_pickle=True)

    # Case A: 0-d object ndarray holding a dict
    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
        obj = arr.item()
    else:
        obj = arr

    # If it's already list/array of strings: index == id
    if isinstance(obj, (list, tuple, np.ndarray)):
        # ensure python list of str
        id2gloss = [str(x) for x in list(obj)]
        # heuristic: if first entry not empty, still treat index 0 as blank
        return id2gloss, assume_blank_idx

    # If it's a dict, decide its direction
    if isinstance(obj, dict):
        # Detect key/value types
        sample_k = next(iter(obj.keys()))
        sample_v = obj[sample_k]

        # id -> gloss dict
        if isinstance(sample_k, (int, np.integer)) and isinstance(sample_v, str):
            max_id = max(int(k) for k in obj.keys())
            id2gloss = ["<BLANK>"] * (max_id + 1)
            for k, v in obj.items():
                kid = int(k)
                id2gloss[kid] = str(v)
            # try to locate blank if named
            blank_idx = assume_blank_idx
            for i, g in enumerate(id2gloss):
                if g.upper() in ("<BLANK>", "BLANK", "<PAD>", "PAD"):
                    blank_idx = i
                    break
            return id2gloss, blank_idx

        # gloss -> id dict
        if isinstance(sample_k, str) and isinstance(sample_v, (int, np.integer)):
            max_id = max(int(v) for v in obj.values())
            id2gloss = ["<BLANK>"] * (max_id + 1)
            for g, idx in obj.items():
                id2gloss[int(idx)] = str(g)
            # detect blank
            blank_idx = assume_blank_idx
            for g, idx in obj.items():
                if str(g).upper() in ("<BLANK>", "BLANK", "<PAD>", "PAD"):
                    blank_idx = int(idx)
                    break
            return id2gloss, blank_idx

    raise ValueError(f"Unrecognized gloss_dict format in {path}")


@torch.no_grad()
def greedy_ctc_decode(logits, blank=0):
    # logits: (B, T, C)
    pred = logits.argmax(dim=-1)  # (B, T)
    decoded = []
    for seq in pred.tolist():
        out, prev = [], blank
        for tok in seq:
            if tok != blank and tok != prev:
                out.append(tok)
            prev = tok
        decoded.append(out)
    return decoded


def ids_to_glosses(ids, id2gloss):
    return [id2gloss[i] if 0 <= i < len(id2gloss) else "<UNK>" for i in ids]


def main():
    p = argparse.ArgumentParser()

    # data (defaults match your training)
    p.add_argument("--image_prefix", default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")

    # model & eval
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt (best.pt or last.pt)")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # model shape (set to what you trained with)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--fusion_embed", type=int, default=512)
    p.add_argument("--fusion_heads", type=int, default=8)
    p.add_argument("--max_kv", type=int, default=1024)
    p.add_argument("--pool_mode", default="mean", choices=["mean", "max", "vote"])

    args = p.parse_args()

    # ---- load gloss dict
    id2gloss, blank_idx = _load_gloss_dict(args.gloss_dict, assume_blank_idx=0)
    num_classes = len(id2gloss)

    # ---- dataset
    ds = MultiModalPhoenixDataset(
        split=args.split,
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir=args.meta_dir,
        gloss_dict=args.gloss_dict,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=multi_modal_collate_fn,
    )

    # ---- model & ckpt
    model = MultiModalMamba(
        d_model=args.d_model,
        n_layer=args.n_layer,
        fusion_embed=args.fusion_embed,
        fusion_heads=args.fusion_heads,
        num_classes=num_classes,
        max_kv=args.max_kv,
        pool_mode=args.pool_mode,
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(args.device).eval()

    # ---- eval
    hyp_strs, ref_strs = [], []

    use_bf16 = args.bf16 and torch.cuda.is_available()
    autocast_ctx = torch.cuda.amp.autocast if use_bf16 else torch.cpu.amp.autocast
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    with autocast_ctx(dtype=amp_dtype):
        for batch in dl:
            images        = batch["images"].to(args.device, non_blocking=True)        # (B, T_img, C, H, W)
            qgrids        = batch["qgrids"].to(args.device, non_blocking=True)        # (B, T_q, 2)
            keypoints     = batch["keypoints"].to(args.device, non_blocking=True)     # (B, T_kp, J, 2) or similar
            qgrid_lengths = batch["qgrid_lengths"].to(args.device, non_blocking=True) # (B,)
            labels        = batch["labels"]  # list[tensor] of gold ids

            out = model(images, qgrids, keypoints, qgrid_lengths)
            logits = out[0] if isinstance(out, (tuple, list)) else out  # (B, T, C)

            pred_ids_batch = greedy_ctc_decode(logits, blank=blank_idx)
            for i, hyp_ids in enumerate(pred_ids_batch):
                hyp_strs.append(" ".join(ids_to_glosses(hyp_ids, id2gloss)))
                gold_ids = labels[i].tolist() if torch.is_tensor(labels[i]) else list(labels[i])
                ref_strs.append(" ".join(ids_to_glosses(gold_ids, id2gloss)))

    print(f"[eval] split={args.split}  ckpt={os.path.basename(args.ckpt)}  classes={num_classes}  blank_idx={blank_idx}")
    print(f"[eval] samples: {len(ref_strs)}")
    print(f"[eval] WER: {wer(ref_strs, hyp_strs)*100:.2f}%   CER: {cer(ref_strs, hyp_strs)*100:.2f}%")
