#!/usr/bin/env python
# eval_multimodal.py — evaluation that matches ddp_train_multimodal packing/wrapping
import argparse, warnings, math
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# dataset / model
from slr.datasets.multi_modal_datasets import (
    MultiModalPhoenixDataset,
    multi_modal_collate_fn,
)
from slr.models.multi_modal_model import MultiModalMamba

# -------------------------------
# args – keep your same defaults
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pt")
    p.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    p.add_argument("--split", choices=["train", "val", "test", "dev"], default="test")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--bf16", action="store_true", help="Evaluate under bfloat16 autocast when supported")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pool_mode", type=str, default="mean")  # just in case checkpoint stored this
    p.add_argument("--max_kv", type=int, default=2048)
    # optional override (rarely needed now)
    p.add_argument("--force_num_classes", type=int, default=None)
    return p.parse_args()

# -------------------------------
# frame-encoder 5D wrapper (as in training)
# -------------------------------
def wrap_frame_encoder_for_5d(model: nn.Module, verbose: bool=True):
    if not hasattr(model, "frame_encoder"):
        return
    fe = model.frame_encoder
    if getattr(fe, "_wrapped_for_5d", False):
        return
    orig_forward = fe.forward

    def adapter(x: torch.Tensor, *args, **kwargs):
        # accept (B*T,C,H,W) or (B,T,C,H,W)
        if x.dim() == 4:
            return orig_forward(x, *args, **kwargs)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            y = orig_forward(x.reshape(B * T, C, H, W), *args, **kwargs)
            # (B*T,D) OR (B*T,P,D)
            if y.dim() == 2:
                return y.view(B, T, -1)
            if y.dim() == 3:
                return y.mean(dim=1).view(B, T, -1)
            raise RuntimeError(f"[adapter] unsupported frame_encoder output {tuple(y.shape)}")
        raise RuntimeError(f"[adapter] frame_encoder input must be 4D/5D, got {tuple(x.shape)}")

    fe.forward = adapter
    fe._wrapped_for_5d = True
    if verbose:
        print("[patch] FrameEncoder adapted for 5D (B,T,C,H,W).", flush=True)

# -------------------------------
# id2gloss utilities
# -------------------------------
def coerce_id_maps(gloss_dict_path: str) -> Tuple[Dict[str,int], Dict[int,str]]:
    obj = np.load(gloss_dict_path, allow_pickle=True)
    try:
        obj = obj.item()
    except Exception:
        pass

    gloss2id: Dict[str,int] = {}
    id2gloss: Dict[int,str] = {}

    if isinstance(obj, dict):
        # detect direction
        k0 = next(iter(obj.keys()))
        v0 = obj[k0]
        if isinstance(k0, str) and isinstance(v0, (int, np.integer)):
            gloss2id = {str(k): int(v) for k, v in obj.items()}
            id2gloss = {int(v): str(k) for k, v in obj.items()}
        elif isinstance(k0, (int, np.integer)) and isinstance(v0, str):
            id2gloss = {int(k): str(v) for k, v in obj.items()}
            gloss2id = {v: k for k, v in id2gloss.items()}
        else:
            # last resort: try to coerce both ways
            for k, v in obj.items():
                try:
                    ik = int(k); sv = str(v)
                    id2gloss[ik] = sv
                except Exception:
                    gloss2id[str(k)] = int(v)
            if not id2gloss and gloss2id:
                id2gloss = {i: g for g, i in gloss2id.items()}
            if not gloss2id and id2gloss:
                gloss2id = {g: i for i, g in id2gloss.items()}
    else:
        # assume array/list of gloss strings indexed by id
        arr = list(obj)
        id2gloss = {i: str(g) for i, g in enumerate(arr)}
        gloss2id = {g: i for i, g in id2gloss.items()}

    # ensure blank at 0; real labels should be >=1
    if 0 not in id2gloss:
        id2gloss[0] = "<blank>"
    if "<blank>" not in gloss2id:
        gloss2id["<blank>"] = 0
    return gloss2id, id2gloss

def ensure_blank_zero_and_shift_if_needed(labels_sample: torch.Tensor,
                                          id2gloss: Dict[int,str]) -> Dict[int,str]:
    """If labels appear to include 0 as a *real* label (not blank),
       shift mapping so: 0=blank, real labels start at 1."""
    if labels_sample.numel() == 0:
        # nothing to check
        return id2gloss
    # Heuristic: if many labels are 0 but dataset does not intend blank inside targets,
    # we consider 0 likely a real label and need a +1 shift.
    zero_frac = float((labels_sample == 0).float().mean().item())
    if zero_frac > 0.05 and id2gloss.get(0, "<blank>") != "<blank>":
        # build shift
        shifted = {0: "<blank>"}
        for k, v in id2gloss.items():
            if k == 0: continue
            shifted[k + 1] = v
        return shifted
    return id2gloss

# -------------------------------
# quick greedy collapse
# -------------------------------
def greedy_collapse_blank0(ids: List[int]) -> List[int]:
    out = []
    prev = None
    for t in ids:
        if t == 0:  # blank
            prev = t
            continue
        if prev is not None and t == prev:
            continue
        out.append(t)
        prev = t
    return out

def edit_distance(a: List[str], b: List[str]) -> int:
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

# -------------------------------
# evaluate loop (matches training packing)
# -------------------------------
@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             id2gloss: Dict[int,str],
             device: torch.device,
             use_bf16: bool=False) -> Tuple[float,float,int]:
    model.eval()
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if (use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
           else torch.cuda.amp.autocast(enabled=False))

    total_w_edits = 0
    total_w_tok   = 0
    total_c_edits = 0
    total_c_tok   = 0
    N = 0

    blank_count = 0
    total_logits = 0

    for batch in loader:
        # training used a tuple return; mirror that via the collate_fn
        # multi_modal_collate_fn returns:
        # (images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths)
        if not isinstance(batch, (tuple, list)) or len(batch) < 7:
            raise RuntimeError("Unexpected batch format from loader; expected 7-tuple.")

        images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths = batch
        images = images.to(device, non_blocking=True) if images is not None else None
        qgrids = qgrids.to(device, non_blocking=True) if qgrids is not None else None
        keypoints = keypoints.to(device, non_blocking=True) if keypoints is not None else None
        qgrid_lengths = qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None

        with amp:
            logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B,T,V)
        if logits.dim() != 3:
            raise RuntimeError(f"Model returned logits with shape {tuple(logits.shape)}, expected (B,T,V).")

        # stats on blanks (after argmax, before collapse)
        preds = logits.argmax(dim=-1)  # (B,T)
        blank_count += int((preds == 0).sum().item())
        total_logits += int(preds.numel())

        # Greedy collapse and score
        preds = preds.cpu()
        labels = labels.cpu()
        label_lengths = label_lengths.cpu()

        B, T = preds.shape
        for b in range(B):
            hyp_ids = greedy_collapse_blank0(preds[b].tolist())
            hyp_toks = [id2gloss.get(i, "<UNK>") for i in hyp_ids]
            hyp_str = " ".join(hyp_toks)

            L = int(label_lengths[b].item())
            ref_ids = labels[b, :L].tolist() if L > 0 else []
            # strip any 0s from references (padding or accidental blanks)
            ref_ids = [i for i in ref_ids if i != 0]
            ref_toks = [id2gloss.get(i, "<UNK>") for i in ref_ids]
            ref_str = " ".join(ref_toks)

            hyp_words = hyp_str.split() if hyp_str else []
            ref_words = ref_str.split() if ref_str else []
            total_w_edits += edit_distance(hyp_words, ref_words)
            total_w_tok   += max(1, len(ref_words))

            hyp_chars = list(hyp_str)
            ref_chars = list(ref_str)
            total_c_edits += edit_distance(hyp_chars, ref_chars)
            total_c_tok   += max(1, len(ref_chars))

            N += 1

    if total_logits > 0:
        br = 100.0 * blank_count / total_logits
        print(f"[debug] blank@argmax ratio: {br:.2f}% ({blank_count}/{total_logits})", flush=True)

    WER = 100.0 * total_w_edits / max(1, total_w_tok)
    CER = 100.0 * total_c_edits / max(1, total_c_tok)
    return WER, CER, N

# -------------------------------
# main
# -------------------------------
def main():
    args = parse_args()
    warnings.filterwarnings("once", category=FutureWarning)

    device = torch.device(args.device)
    print(f"[env] device={device}  bf16={args.bf16}", flush=True)

    # ---- dataset (use same collate as training) ----
    ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,
        gloss_dict_path=args.gloss_dict,
        split={"val":"dev","dev":"dev"}.get(args.split, args.split),
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=multi_modal_collate_fn,
    )

    # ---- vocab / classes ----
    # prefer dataset's parsed maps, but coerce from file if missing
    try:
        gloss2id = getattr(ds, "gloss_dict")
        id2gloss = getattr(ds, "id_to_gloss")
        if not isinstance(gloss2id, dict) or not isinstance(id2gloss, dict):
            raise RuntimeError("ds maps missing")
    except Exception:
        gloss2id, id2gloss = coerce_id_maps(args.gloss_dict)

    # sample some labels to verify id range and shift if needed
    try:
        tmp_batch = next(iter(dl))
        lbls = tmp_batch[3]  # labels (B,L)
        lbl_lens = tmp_batch[5]  # label_lengths
        sample = []
        for i in range(min(8, lbls.size(0))):
            L = int(lbl_lens[i].item())
            if L > 0:
                sample.append(lbls[i, :L])
        sample = torch.cat(sample, dim=0) if sample else torch.zeros(0, dtype=torch.long)
    except Exception:
        sample = torch.zeros(0, dtype=torch.long)

    id2gloss = ensure_blank_zero_and_shift_if_needed(sample, id2gloss)

    # ensure blank exists & compute num_classes
    if 0 not in id2gloss:
        id2gloss[0] = "<blank>"
    vocab_max = max(id2gloss.keys()) if id2gloss else 0
    num_classes = (args.force_num_classes or (vocab_max + 1))
    if args.force_num_classes:
        print(f"[head] num_classes={num_classes} (forced)")
    else:
        print(f"[head] num_classes={num_classes} (from gloss_dict + blank)")

    # ---- model ----
    # Try to infer architecture knobs from checkpoint when present
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # after: ckpt = torch.load(args.ckpt, map_location="cpu")
    # pick the right dict
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state") or ckpt
    else:
        sd = ckpt

    # --- infer dims from checkpoint ---
    # img_proj.weight is [fusion_embed, d_model]
    fe = sd["img_proj.weight"].shape[0]      # fusion_embed (should be 512)
    dm = sd["img_proj.weight"].shape[1]      # d_model (should be 512)
    # attn.in_proj_weight is [3*fe, fe] -> sanity
    assert sd["attn.in_proj_weight"].shape[0] == 3 * fe
    # classifier.weight is [num_classes, fe]
    num_classes = sd["classifier.weight"].shape[0]

    # heads don't affect parameter shapes; choose a clean divisor of fe
    fusion_heads = 8 if fe % 8 == 0 else 4

    print(f"[arch] inferred from ckpt -> d_model={dm}, fusion_embed={fe}, fusion_heads={fusion_heads}, num_classes={num_classes}")

    # --- now build the model with inferred sizes ---
    model = MultiModalMamba(
        d_model=dm,
        n_layer=(n_layer or 12),        # keep your previously inferred n_layer if you have it
        fusion_embed=fe,
        fusion_heads=fusion_heads,
        num_classes=num_classes,        # or use your id2gloss-derived count; both are 1296 here
        max_kv=args.max_kv,
        pool_mode=args.pool_mode,
    ).to(device)

    # keep your existing frame-encoder 5D wrapper here
    wrap_frame_encoder_for_5d(model, verbose=True)

    # (optional) tiny dummy forward to materialize lazies, then load:
    # ...


    # materialize with a tiny forward to allocate shapes before load_state_dict
    try:
        b = next(iter(dl))
        images, qgrids, keypoints = b[0], b[1], b[2]
        qgrid_lengths = b[6]
        with torch.no_grad():
            _ = model(
                images.to(device, non_blocking=True),
                qgrids.to(device, non_blocking=True) if qgrids is not None else None,
                keypoints.to(device, non_blocking=True) if keypoints is not None else None,
                qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None,
            )
    except Exception:
        # if it fails, we'll still proceed to load weights
        pass

    # load checkpoint (strict, but report)
    print(f"[ckpt] loading: {args.ckpt}")
    load = model.load_state_dict(sd, strict=True)  # strict can be True now; it should match
    unexpected = list(getattr(load, "unexpected_keys", []))
    missing    = list(getattr(load, "missing_keys", []))
    print(f"[ckpt] unexpected keys: {len(unexpected)}")
    print(f"[ckpt] missing keys: {len(missing)}")

    # if classifier weights didn’t load, warn (WER likely bad)
    cls_ok = not any(k.startswith("classifier") for k in missing)
    if not cls_ok:
        print("[warn] Classifier head did not fully load from checkpoint. "
              "If vocab differs from training, WER/CER will be high.", flush=True)

    # ---- evaluate ----
    use_bf16 = args.bf16 and (device.type == "cuda") and torch.cuda.is_bf16_supported()
    WER, CER, N = evaluate(model, dl, id2gloss, device, use_bf16=use_bf16)
    print(f"[RESULT] N={N}  WER={WER:.2f}%  CER={CER:.2f}%")

if __name__ == "__main__":
    main()
