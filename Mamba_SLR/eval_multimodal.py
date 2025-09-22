# eval_multimodal.py
# Uses your exact path defaults + robust id2gloss loading from gloss_dict
# Adds an adapter so frame_encoder can take (B,T,C,H,W) -> (B,T,D)
# Pads variable-length sequences via a custom collate_fn that supports tuple/dict samples.

import argparse
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset
from slr.models.multi_modal_model import MultiModalMamba


# -------------------------------
# Collate helpers (pad variable T)
# -------------------------------
def _pad_time(x: torch.Tensor, T: int, time_dim: int) -> torch.Tensor:
    if x.shape[time_dim] == T:
        return x
    new_shape = list(x.shape)
    new_shape[time_dim] = T
    out = torch.zeros(new_shape, dtype=x.dtype)
    slc = [slice(None)] * x.ndim
    slc[time_dim] = slice(0, x.shape[time_dim])
    out[tuple(slc)] = x
    return out

def _to_TCHW(img: torch.Tensor) -> torch.Tensor:
    # Accept (C,T,H,W) or (T,C,H,W) → return (T,C,H,W)
    if img.ndim != 4:
        raise RuntimeError(f"Expected image tensor 4D, got {tuple(img.shape)}")
    if img.shape[0] in (1, 3):  # (C,T,H,W)
        return img.permute(1, 0, 2, 3).contiguous()
    return img  # already (T,C,H,W)

def _infer_T_from_images(img: torch.Tensor) -> int:
    if img.ndim != 4:
        raise RuntimeError(f"Expected image tensor 4D, got {tuple(img.shape)}")
    if img.shape[0] in (1, 3):  # (C,T,H,W)
        return int(img.shape[1])
    return int(img.shape[0])     # (T,C,H,W)

def _sample_to_dict(sample: Any) -> Dict[str, Any]:
    """
    Normalize a dataset sample (dict OR tuple/list) into a dict with:
      images (4D), qgrids (3D optional), keypoints (3D optional),
      qgrid_lengths (int/1D optional), gloss_ids/gloss_str/labels (optional)
    """
    if isinstance(sample, dict):
        return sample

    if not isinstance(sample, (tuple, list)):
        raise RuntimeError(f"Unexpected sample type: {type(sample)}")

    out: Dict[str, Any] = {}
    others: List[Any] = []

    for x in sample:
        if isinstance(x, torch.Tensor):
            if x.ndim == 4 and "images" not in out:
                out["images"] = x  # (C,T,H,W) or (T,C,H,W)
                continue
            if x.ndim == 3:
                # Heuristic: keypoints typically (T,J,D) with small D (2/3/4/6)
                if x.shape[-1] in (2, 3, 4, 6) and x.shape[1] >= 8 and "keypoints" not in out:
                    out["keypoints"] = x
                elif "qgrids" not in out:
                    out["qgrids"] = x  # assume (T,Hq,Wq) or (Hq,Wq,T)
                else:
                    others.append(x)
                continue
            if x.ndim == 1 and "qgrid_lengths" not in out and x.numel() == 1:
                out["qgrid_lengths"] = x  # scalar len as 1D
                continue
        others.append(x)

    # Try to attach common label-ish fields from leftovers
    for x in others:
        if isinstance(x, torch.Tensor) and x.ndim == 1 and "gloss_ids" not in out:
            out["gloss_ids"] = x
        elif isinstance(x, (list, tuple)) and "gloss_str" not in out:
            out["gloss_str"] = list(x)
        elif isinstance(x, str) and "gloss_str" not in out:
            out["gloss_str"] = x

    if "images" not in out:
        raise RuntimeError("Sample has no images tensor (4D)")

    return out

def collate_mm(batch: List[Any]) -> Dict[str, Any]:
    """
    Pads variable-length time across a batch.
    Returns:
      images: (B,T,C,H,W)
      qgrids: (B,T,...) or None
      keypoints: (B,T,J,D) or None
      qgrid_lengths: (B,)
      + labels (gloss_ids/gloss_str/labels) if present
    """
    batch_norm = [_sample_to_dict(s) for s in batch]

    # Compute each T_i from images
    Ts = []
    for s in batch_norm:
        img = s.get("images")
        if img is None:
            img = s.get("frames")
        if not isinstance(img, torch.Tensor):
            raise RuntimeError("Sample missing 'images' tensor")
        Ts.append(_infer_T_from_images(img))
    T_max = max(Ts)

    images_list, qgrids_list, keypts_list, lengths_list = [], [], [], []

    for i, s in enumerate(batch_norm):
        # Images → (T,C,H,W) then pad
        img = s.get("images")
        if img is None:
            img = s.get("frames")
        img_TCHW = _to_TCHW(img)
        img_TCHW = _pad_time(img_TCHW, T_max, time_dim=0)
        images_list.append(img_TCHW)
        lengths_list.append(Ts[i])

        # Qgrids: try to make time be dim 0 if 3D
        qg = s.get("qgrids")
        if qg is None:
            qg = s.get("qgrid")
        if isinstance(qg, torch.Tensor) and qg.ndim == 3:
            if qg.shape[0] == Ts[i]:
                qgT = qg
            elif qg.shape[-1] == Ts[i]:
                qgT = qg.permute(2, 0, 1).contiguous()
            else:
                qgT = qg
            qgT = _pad_time(qgT, T_max, time_dim=0)
            qgrids_list.append(qgT)
        else:
            qgrids_list.append(None)

        # Keypoints: usually (T,J,D)
        kp = s.get("keypoints")
        if kp is None:
            kp = s.get("kps")
        if kp is None:
            kp = s.get("pose")
        if isinstance(kp, torch.Tensor) and kp.ndim == 3 and kp.shape[0] == Ts[i]:
            kpT = _pad_time(kp, T_max, time_dim=0)
            keypts_list.append(kpT)
        else:
            keypts_list.append(None)

    images = torch.stack(images_list, dim=0)       # (B,T,C,H,W)

    qgrids = None
    if any(q is not None for q in qgrids_list):
        first = next(q for q in qgrids_list if q is not None)
        filled = [(q if q is not None else torch.zeros_like(first)) for q in qgrids_list]
        qgrids = torch.stack(filled, dim=0)        # (B,T,Hq,Wq)

    keypoints = None
    if any(k is not None for k in keypts_list):
        first = next(k for k in keypts_list if k is not None)
        filled = [(k if k is not None else torch.zeros_like(first)) for k in keypts_list]
        keypoints = torch.stack(filled, dim=0)     # (B,T,J,D)

    lengths = torch.tensor(lengths_list, dtype=torch.int32)  # (B,)

    # Labels: try to collate if present & stackable
    out: Dict[str, Any] = {
        "images": images,
        "qgrids": qgrids,
        "keypoints": keypoints,
        "qgrid_lengths": lengths,
    }
    for k in ("gloss_ids", "targets", "labels", "gloss_str", "gloss", "text"):
        vals = [s.get(k, None) for s in batch_norm]
        if all(v is None for v in vals):
            continue
        if all(isinstance(v, torch.Tensor) for v in vals):
            try:
                out[k] = torch.stack(vals, dim=0)
            except Exception:
                out[k] = vals
        else:
            out[k] = vals
    return out


# -------------------------------
# Frame-encoder adapter
# -------------------------------
class _FrameEncoderAdapter(nn.Module):
    """
    Makes an encoder that expects (B*T, C, H, W) work with (B, T, C, H, W).
    Returns (B, T, D). If input is already (B*T, C, H, W), passes through.
    """
    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # already (B*T, C, H, W)
            return self._normalize(self.enc(x), B=None, T=None)
        if x.dim() != 5:
            raise RuntimeError(f"[adapter] expected 5D (B,T,C,H,W) or 4D (B*T,C,H,W), got {tuple(x.shape)}")
        B, T, C, H, W = x.shape
        x_bt = x.reshape(B * T, C, H, W)
        y = self.enc(x_bt)  # (B*T, D) or (B*T, P, D)
        return self._normalize(y, B, T)

    @staticmethod
    def _normalize(y: torch.Tensor, B: Optional[int], T: Optional[int]) -> torch.Tensor:
        if B is None or T is None:
            return y
        if y.dim() == 2:       # (B*T, D)
            return y.view(B, T, -1)
        if y.dim() == 3:       # (B*T, P, D) -> mean pool patches
            y = y.mean(dim=1)
            return y.view(B, T, -1)
        raise RuntimeError(f"[adapter] unexpected frame_encoder output {tuple(y.shape)}")


# -------------------------------
# Args (your exact defaults)
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pt")
    p.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--force_num_classes", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# -------------------------------
# id2gloss loader (robust)
# -------------------------------
def load_id2gloss(gloss_dict_path: str) -> Dict[int, str]:
    """
    Supports both:
      - dict: {gloss(str) -> id(int)}  or  {id(int) -> gloss(str)}
      - list/array: index -> gloss
    Always returns a mapping {id(int) -> gloss(str)} and ensures id 0 exists.
    """
    obj = np.load(gloss_dict_path, allow_pickle=True)
    try:
        obj = obj.item()
    except Exception:
        pass

    id2gloss: Dict[int, str] = {}

    if isinstance(obj, dict):
        # Case A: keys are strings (gloss), values ints (id)
        if obj and isinstance(next(iter(obj.keys())), str) and isinstance(next(iter(obj.values())), (int, np.integer)):
            for g, i in obj.items():
                id2gloss[int(i)] = str(g)
        # Case B: keys are ints (id), values strings (gloss)
        elif obj and isinstance(next(iter(obj.keys())), (int, np.integer)) and isinstance(next(iter(obj.values())), str):
            for i, g in obj.items():
                id2gloss[int(i)] = str(g)
        else:
            # Fallback: attempt coercion
            for k, v in obj.items():
                try:
                    i = int(k); g = str(v)
                except Exception:
                    i = int(v); g = str(k)
                id2gloss[i] = g
    else:
        arr = list(obj)
        for i, g in enumerate(arr):
            id2gloss[int(i)] = str(g)

    if 0 not in id2gloss:
        id2gloss[0] = "<blank>"
    return id2gloss


# -------------------------------
# Evaluation
# -------------------------------
@torch.no_grad()
def evaluate(model: nn.Module,
             dl: DataLoader,
             id2gloss: Dict[int, str],
             device: str,
             use_bf16: bool = False):
    """
    Greedy CTC-like decode (blank=0, collapse repeats) -> WER/CER.
    Assumes model(images, qgrids, keypoints, qgrid_lengths) -> (B, T', C).
    """
    model.eval()
    blank_idx = 0

    total_wer_edits = 0
    total_wer_tokens = 0
    total_cer_edits = 0
    total_cer_chars = 0
    total_samples = 0

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if (use_bf16 and device.startswith("cuda"))
                    else torch.cuda.amp.autocast(enabled=False))

    def collapse_and_map(ids):
        seq = []
        prev = None
        for t in ids:
            if t == blank_idx:
                prev = t
                continue
            if prev is not None and t == prev:
                continue
            seq.append(t)
            prev = t
        return [id2gloss.get(int(x), "<UNK>") for x in seq]

    def edit_distance(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n]

    def cer_edits(hyp_str, ref_str):
        a = list(hyp_str); b = list(ref_str)
        return edit_distance(a, b), len(b)

    for batch in dl:
        if not isinstance(batch, dict):
            raise RuntimeError("Expected dict batches from MultiModalPhoenixDataset.")

        images = batch["images"]
        qgrids = batch.get("qgrids")
        keypoints = batch.get("keypoints")
        qgrid_lengths = batch["qgrid_lengths"]

        # Safe picks that don't coerce tensors to bool
        y_ids = batch.get("gloss_ids")
        if y_ids is None:
            y_ids = batch.get("targets")
        if y_ids is None:
            y_ids = batch.get("labels")

        y_text = batch.get("gloss_str")
        if y_text is None:
            y_text = batch.get("gloss")
        if y_text is None:
            y_text = batch.get("text")


        images = images.to(device, non_blocking=True)
        qgrids = qgrids.to(device, non_blocking=True) if qgrids is not None else None
        keypoints = keypoints.to(device, non_blocking=True) if keypoints is not None else None
        qgrid_lengths = qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None

        with autocast_ctx:
            logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B, T', C)

        pred_ids = logits.argmax(dim=-1).cpu()  # (B, T')

        B = pred_ids.size(0)
        for b in range(B):
            hyp_toks = collapse_and_map(pred_ids[b].tolist())
            hyp_str = " ".join(hyp_toks)

            if y_text is not None:
                ref_str = y_text[b] if isinstance(y_text, (list, tuple)) else y_text
                if isinstance(ref_str, list):
                    ref_str = " ".join(ref_str)
            elif y_ids is not None:
                ids = y_ids[b].tolist()
                ref_toks = [id2gloss.get(int(t), "<UNK>") for t in ids if int(t) != blank_idx]
                ref_str = " ".join(ref_toks)
            else:
                ref_str = ""

            hyp_words = hyp_str.split()
            ref_words = ref_str.split()
            total_wer_edits += edit_distance(hyp_words, ref_words)
            total_wer_tokens += max(1, len(ref_words))

            ed, L = cer_edits(hyp_str, ref_str)
            total_cer_edits += ed
            total_cer_chars += max(1, L)

            total_samples += 1

    WER = 100.0 * total_wer_edits / max(1, total_wer_tokens)
    CER = 100.0 * total_cer_edits / max(1, total_cer_chars)
    return WER, CER, total_samples


# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    warnings.filterwarnings("once", category=FutureWarning)

    device = args.device
    map_location = torch.device(device)

    # Dataset — pass your args directly; note param names on ctor
    ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,     # ctor expects meta_dir_path
        gloss_dict_path=args.gloss_dict, # ctor expects gloss_dict_path
        split=args.split,
    )

    # id2gloss
    id2gloss = getattr(ds, "id2gloss", None)
    if not isinstance(id2gloss, dict) or len(id2gloss) == 0:
        id2gloss = load_id2gloss(args.gloss_dict)

    # DataLoader with our robust collate
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_mm,
    )

    # Model
    n_classes = args.force_num_classes or len(id2gloss)
    if args.force_num_classes:
        print(f"[head] num_classes={n_classes} (forced)")
    model = MultiModalMamba(num_classes=n_classes).to(device)

        # ---------- load checkpoint first (to sniff metadata) ----------
    print(f"[ckpt] loading: {args.ckpt}")
    try:
        ckpt_obj = torch.load(args.ckpt, map_location=map_location, weights_only=True)
    except TypeError:
        ckpt_obj = torch.load(args.ckpt, map_location=map_location)

    # Pull a state_dict-like mapping
    state_dict = ckpt_obj.get("state_dict") if isinstance(ckpt_obj, dict) else None
    if state_dict is None and isinstance(ckpt_obj, dict) and all(isinstance(k, str) for k in ckpt_obj.keys()):
        state_dict = ckpt_obj
    if state_dict is None and isinstance(ckpt_obj, dict):
        state_dict = ckpt_obj.get("model", ckpt_obj.get("model_state", {})) or {}

    # Strip common prefixes
    def _strip(sd: Dict[str, Any], prefixes=("module.", "model.")):
        out = {}
        for k, v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            out[nk] = v
        return out

    state_dict = _strip(state_dict)

    # Try to infer number of classes from the checkpoint head
    num_classes_ckpt = None
    for key in ("head.weight", "classifier.weight", "fc.weight", "final_proj.weight"):
        if key in state_dict and state_dict[key].dim() == 2:
            num_classes_ckpt = int(state_dict[key].shape[0])
            break

    # ---------- dataset ----------
    ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,
        gloss_dict_path=args.gloss_dict,
        split=args.split,
    )

    # Build id2gloss (prefer from checkpoint if included)
    id2gloss = None
    # common fields people save
    for vocab_key in ("id2gloss", "vocab", "labels", "classes", "gloss_list"):
        if isinstance(ckpt_obj, dict) and vocab_key in ckpt_obj:
            raw = ckpt_obj[vocab_key]
            try:
                if isinstance(raw, dict):
                    # normalize to {int_id: str_gloss}
                    if raw and isinstance(next(iter(raw.keys())), str):
                        id2gloss = {int(v): str(k) for k, v in raw.items()}
                    else:
                        id2gloss = {int(k): str(v) for k, v in raw.items()}
                else:
                    seq = list(raw)
                    id2gloss = {int(i): str(g) for i, g in enumerate(seq)}
            except Exception:
                id2gloss = None
            if id2gloss:
                break

    if id2gloss is None:
        # fall back to dataset file
        id2gloss = load_id2gloss(args.gloss_dict)

    # Decide final num_classes
    if args.force_num_classes:
        n_classes = int(args.force_num_classes)
        print(f"[head] num_classes={n_classes} (forced)")
    elif num_classes_ckpt is not None:
        n_classes = num_classes_ckpt
        print(f"[head] num_classes={n_classes} (from checkpoint)")
    else:
        n_classes = len(id2gloss)
        print(f"[head] num_classes={n_classes} (from id2gloss)")

    # Sanity check: id2gloss length vs head
    if len(id2gloss) != n_classes:
        raise RuntimeError(
            f"Vocab size mismatch: id2gloss={len(id2gloss)} vs head={n_classes}. "
            f"Use the same gloss_dict the checkpoint was trained with, or remove --force_num_classes."
        )

    # ---------- dataloader ----------
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_mm,
    )

    # ---------- model ----------
    model = MultiModalMamba(num_classes=n_classes).to(device)

    # Load weights (now shapes should match)
    ld = model.load_state_dict(state_dict, strict=False)
    missing = getattr(ld, "missing_keys", None) or (ld.get("missing_keys", []) if isinstance(ld, dict) else [])
    unexpected = getattr(ld, "unexpected_keys", None) or (ld.get("unexpected_keys", []) if isinstance(ld, dict) else [])
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[ckpt] missing keys: {len(missing)}")

    # Frame encoder adapter
    if not isinstance(model.frame_encoder, _FrameEncoderAdapter):
        model.frame_encoder = _FrameEncoderAdapter(model.frame_encoder)
    print("[patch] FrameEncoderAdapter attached.")


    # Eval
    WER, CER, N = evaluate(model, dl, id2gloss, device, use_bf16=bool(args.bf16))
    print(f"[RESULT] N={N}  WER={WER:.2f}%  CER={CER:.2f}%")


if __name__ == "__main__":
    main()
