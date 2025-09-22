# eval_multimodal.py — eval that mirrors ddp_train_multimodal.py hyperparams + collate
import argparse, warnings
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.models.multi_modal_model import MultiModalMamba


# ---- frame encoder adapter (like in training) ----
def wrap_frame_encoder_for_5d(model: nn.Module):
    fe = model.frame_encoder
    orig_forward = fe.forward
    def adapter(x: torch.Tensor, *args, **kwargs):
        if x.dim() == 4:
            return orig_forward(x, *args, **kwargs)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            y = orig_forward(x.reshape(B*T, C, H, W), *args, **kwargs)
            if y.dim() == 2: return y.view(B, T, -1)
            if y.dim() == 3: return y.mean(dim=1).view(B, T, -1)
            raise RuntimeError(f"Unsupported frame_encoder output {tuple(y.shape)}")
        raise RuntimeError(f"frame_encoder input must be 4D/5D, got {tuple(x.shape)}")
    fe.forward = adapter


def get_args():
    p = argparse.ArgumentParser()
    # data — keep your defaults
    p.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    p.add_argument("--split", choices=["train","dev","test"], default="test")

    # model hyperparams (match training defaults)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--fusion_embed", type=int, default=512)
    p.add_argument("--fusion_heads", type=int, default=8)
    p.add_argument("--max_kv", type=int, default=1024)  # training default
    p.add_argument("--pool_mode", type=str, default="mean", choices=["mean","max","vote"])

    # eval
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--blank_idx", type=int, default=0)
    return p.parse_args()


def load_id2gloss(gloss_dict_path: str) -> Dict[int,str]:
    obj = np.load(gloss_dict_path, allow_pickle=True)
    try: obj = obj.item()
    except Exception: pass
    id2gloss: Dict[int,str] = {}
    if isinstance(obj, dict):
        try:
            k0,v0 = next(iter(obj.items()))
        except StopIteration:
            raise ValueError("empty gloss_dict")
        if isinstance(k0, (str, bytes)):
            for g,i in obj.items():
                id2gloss[int(i)] = (g.decode("utf-8") if isinstance(g, bytes) else str(g))
        elif isinstance(k0, (int, np.integer)):
            for i,g in obj.items():
                id2gloss[int(i)] = (g.decode("utf-8") if isinstance(g, bytes) else str(g))
        else:
            for k,v in obj.items():
                try: i = int(k); g = str(v)
                except Exception: i = int(v); g = str(k)
                id2gloss[i] = g
    else:
        arr = list(obj)
        for i,g in enumerate(arr):
            id2gloss[int(i)] = str(g)
    if 0 not in id2gloss: id2gloss[0] = "<blank>"
    return id2gloss


def unpack_batch(batch: Any) -> Dict[str, Any]:
    """Mirror training's unpack: accept tuple from multi_modal_collate_fn or dict."""
    if isinstance(batch, dict):
        return batch
    # tuple layout from multi_modal_collate_fn:
    # (images(B,T,C,H,W), qgrids(B,T,242), keypoints(B,T,242), labels(B,L),
    #  image_lengths(B,), label_lengths(B,), qgrid_lengths(B,))
    (images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths) = batch
    return {
        "images": images,
        "qgrids": qgrids,
        "keypoints": keypoints,
        "labels": labels,
        "image_lengths": image_lengths,
        "label_lengths": label_lengths,
        "qgrid_lengths": qgrid_lengths,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, id2gloss: Dict[int,str], device: str, use_bf16: bool, blank_idx: int):
    model.eval()
    total_wer_edits = total_wer_tokens = 0
    total_cer_edits = total_cer_chars = 0
    N = 0
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if (use_bf16 and device.startswith("cuda")) else torch.cuda.amp.autocast(enabled=False))

    def collapse_and_map(ids):
        seq, prev = [], None
        for t in ids:
            if t == blank_idx: prev = t; continue
            if prev is not None and t == prev: continue
            seq.append(t); prev = t
        return [id2gloss.get(int(x), "<UNK>") for x in seq]

    def ed(a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1,m+1):
            for j in range(1,n+1):
                cost = 0 if a[i-1]==b[j-1] else 1
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
        return dp[m][n]

    for batch in loader:
        b = unpack_batch(batch)

        images = b["images"].to(device, non_blocking=True)
        qgrids = b["qgrids"].to(device, non_blocking=True) if b["qgrids"] is not None else None
        keypoints = b["keypoints"].to(device, non_blocking=True) if b["keypoints"] is not None else None
        qgrid_lengths = b.get("qgrid_lengths")
        if qgrid_lengths is not None:
            qgrid_lengths = qgrid_lengths.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(images, qgrids, keypoints, qgrid_lengths)   # (B,T,V)

        preds = logits.argmax(dim=-1).cpu()
        labels = b.get("labels")
        if labels is not None:
            labels = labels.cpu()

        B = preds.size(0)
        for i in range(B):
            hyp_toks = collapse_and_map(preds[i].tolist())
            hyp = " ".join(hyp_toks)

            if labels is not None:
                ref_ids = [int(t) for t in labels[i].tolist() if int(t) != blank_idx]
                ref = " ".join(id2gloss.get(int(t), "<UNK>") for t in ref_ids)
            else:
                ref = ""

            hw, rw = hyp.split(), ref.split()
            total_wer_edits += ed(hw, rw); total_wer_tokens += max(1, len(rw))
            total_cer_edits += ed(list(hyp), list(ref)); total_cer_chars += max(1, len(ref))
            N += 1

    WER = 100.0 * total_wer_edits / max(1, total_wer_tokens)
    CER = 100.0 * total_cer_edits / max(1, total_cer_chars)
    return WER, CER, N


def main():
    args = get_args()
    warnings.filterwarnings("once", category=FutureWarning)
    device = args.device
    map_location = torch.device(device)

    # Dataset & loader (use the SAME collate as training)
    ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix, qgrid_prefix=args.qgrid_prefix, kp_path=args.kp_path,
        meta_dir_path=args.meta_dir, gloss_dict_path=args.gloss_dict, split=args.split,
    )
    id2gloss = getattr(ds, "id_to_gloss", None) or getattr(ds, "id2gloss", None)
    if not isinstance(id2gloss, dict) or len(id2gloss)==0:
        id2gloss = load_id2gloss(args.gloss_dict)
    n_classes = len(id2gloss)

    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=multi_modal_collate_fn, drop_last=False,
    )

    # Model — mirror training hparams
    model = MultiModalMamba(
        d_model=args.d_model, n_layer=args.n_layer,
        fusion_embed=args.fusion_embed, fusion_heads=args.fusion_heads,
        num_classes=n_classes, max_kv=args.max_kv, pool_mode=args.pool_mode,
    ).to(device)
    wrap_frame_encoder_for_5d(model)

    # ---- MATERIALIZE lazy layers BEFORE loading weights (like training) ----
    try:
        first_batch = next(iter(dl))
        b0 = unpack_batch(first_batch)
        with torch.no_grad():
            _ = model(
                b0["images"].to(device, non_blocking=True),
                (b0["qgrids"].to(device, non_blocking=True) if b0["qgrids"] is not None else None),
                (b0["keypoints"].to(device, non_blocking=True) if b0["keypoints"] is not None else None),
                (b0["qgrid_lengths"].to(device, non_blocking=True) if b0.get("qgrid_lengths") is not None else None),
            )
    except StopIteration:
        pass

    # Load checkpoint
    print(f"[ckpt] loading: {args.ckpt}")
    try:
        ckpt_obj = torch.load(args.ckpt, map_location=map_location, weights_only=True)
    except TypeError:
        ckpt_obj = torch.load(args.ckpt, map_location=map_location)

    # state_dict extract + strip prefixes
    state_dict = ckpt_obj.get("state_dict") if isinstance(ckpt_obj, dict) else None
    if state_dict is None and isinstance(ckpt_obj, dict) and all(isinstance(k, str) for k in ckpt_obj.keys()):
        state_dict = ckpt_obj
    if state_dict is None and isinstance(ckpt_obj, dict):
        state_dict = ckpt_obj.get("model", ckpt_obj.get("model_state", {})) or {}
    def _strip(sd: Dict[str, Any], prefixes=("module.", "model.")):
        out = {}
        for k,v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p): nk = nk[len(p):]
            out[nk]=v
        return out
    state_dict = _strip(state_dict)

    ld = model.load_state_dict(state_dict, strict=False)
    missing = getattr(ld, "missing_keys", None) or (ld.get("missing_keys", []) if isinstance(ld, dict) else [])
    unexpected = getattr(ld, "unexpected_keys", None) or (ld.get("unexpected_keys", []) if isinstance(ld, dict) else [])
    if unexpected: print(f"[ckpt] unexpected keys: {len(unexpected)}")
    if missing:    print(f"[ckpt] missing keys: {len(missing)}")
    # Make classifier head loading mandatory:
    if any((k.startswith("classifier.") or k.startswith("head.") or k.startswith("fc.")) for k in missing):
        raise RuntimeError("Classifier head from checkpoint did not load. Ensure model hparams & vocab match training.")

    # Recreate loader iterator fresh (since we consumed one batch for materialization)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=multi_modal_collate_fn, drop_last=False,
    )

    use_bf16 = bool(args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    WER, CER, N = evaluate(model, dl, id2gloss, device, use_bf16=use_bf16, blank_idx=args.blank_idx)
    print(f"[RESULT] N={N}  WER={WER:.2f}%  CER={CER:.2f}%")


if __name__ == "__main__":
    main()
