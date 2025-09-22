# eval_multimodal.py
# Final: robust dataset args + frame-encoder adapter for (B,T,C,H,W)

import os
import argparse
import warnings
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- repo imports (keep as-is for your tree) ---
from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset
from slr.models.multi_modal_model import MultiModalMamba


# ==========================
# Frame-Encoder Adapter
# ==========================
class _FrameEncoderAdapter(nn.Module):
    """
    Wraps the existing frame_encoder so it can take (B, T, C, H, W)
    and returns (B, T, D). If given (B*T, C, H, W), passes through.
    """
    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # Already (B*T, C, H, W)
            return self._normalize_output(self.enc(x), B=None, T=None)

        if x.dim() != 5:
            raise RuntimeError(f"[adapter] Expected (B,T,C,H,W) or (B*T,C,H,W); got {tuple(x.shape)}")

        B, T, C, H, W = x.shape
        x_bt = x.reshape(B * T, C, H, W)   # (B*T, C, H, W)
        y = self.enc(x_bt)                 # typically (B*T, D) or (B*T, P, D)
        return self._normalize_output(y, B=B, T=T)

    @staticmethod
    def _normalize_output(y: torch.Tensor, B: Optional[int], T: Optional[int]) -> torch.Tensor:
        # Normalize to (B, T, D) when B/T provided, else return as-is.
        if B is None or T is None:
            return y
        if y.dim() == 2:
            return y.view(B, T, -1)               # (B*T, D) -> (B, T, D)
        if y.dim() == 3:
            # (B*T, P, D) -> mean pool patches -> (B, T, D)
            y = y.mean(dim=1)
            return y.view(B, T, -1)
        raise RuntimeError(f"[adapter] Unexpected frame_encoder output shape {tuple(y.shape)}")


# ==========================
# Utils: path guessing
# ==========================
def _first_existing(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def _repo_root() -> str:
    return os.path.abspath(os.path.dirname(__file__))

def guess_phoenix_paths() -> Tuple[str, str, str, str, str]:
    """
    Try to guess dataset paths from common layouts and your earlier logs.
    You can still override via CLI flags or env vars.
    """
    root = _repo_root()
    base_candidates = [
        os.path.join(root, "data", "phoenix2014"),
        "/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014",  # your machine's path seen in logs
        os.path.join(root, "phoenix2014"),
    ]
    base = _first_existing(base_candidates) or base_candidates[0]

    # frames
    image_prefix = _first_existing([
        os.path.join(base, "phoenix2014-frames"),
        os.path.join(base, "frames"),
        os.path.join(base, "images"),
    ]) or os.path.join(base, "frames")

    # qgrid
    qgrid_prefix = _first_existing([
        os.path.join(base, "qgrid"),
        os.path.join(base, "qgrids"),
        os.path.join(base, "qgrid_npz"),
    ]) or os.path.join(base, "qgrid")

    # keypoints .pkl
    kp_path = _first_existing([
        os.path.join(base, "phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256_INTERPOLATED.pkl"),
        os.path.join(base, "phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl"),
        os.path.join(base, "interpolated", "phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256_INTERPOLATED.pkl"),
    ]) or os.path.join(base, "phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256_INTERPOLATED.pkl")

    # meta dir
    meta_dir_path = _first_existing([
        os.path.join(base, "_meta"),
        os.path.join(base, "meta"),
    ]) or os.path.join(base, "_meta")

    # gloss dict
    gloss_dict_path = _first_existing([
        os.path.join(base, "gloss_dict_normalized.npy"),
        os.path.join(base, "gloss_dict.npy"),
    ]) or os.path.join(base, "gloss_dict_normalized.npy")

    return image_prefix, qgrid_prefix, kp_path, meta_dir_path, gloss_dict_path


def resolve_dataset_paths(args) -> Tuple[str, str, str, str, str]:
    """
    Priority: CLI flag -> ENV -> guess
    ENV names:
      PHX_IMAGE_PREFIX, PHX_QGRID_PREFIX, PHX_KP_PATH, PHX_META_DIR, PHX_GLOSS_DICT
    """
    env = os.environ
    image_prefix = args.image_prefix or env.get("PHX_IMAGE_PREFIX")
    qgrid_prefix = args.qgrid_prefix or env.get("PHX_QGRID_PREFIX")
    kp_path      = args.kp_path      or env.get("PHX_KP_PATH")
    meta_dir     = args.meta_dir_path or env.get("PHX_META_DIR")
    gloss_dict   = args.gloss_dict_path or env.get("PHX_GLOSS_DICT")

    if all([image_prefix, qgrid_prefix, kp_path, meta_dir, gloss_dict]):
        return image_prefix, qgrid_prefix, kp_path, meta_dir, gloss_dict

    # Fall back to guessing
    gi, gq, gk, gm, gg = guess_phoenix_paths()
    return (
        image_prefix or gi,
        qgrid_prefix or gq,
        kp_path or gk,
        meta_dir or gm,
        gloss_dict or gg,
    )


# ==========================
# Arg parsing
# ==========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    ap.add_argument("--split", type=str, default="test", choices=["train", "dev", "valid", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--force_num_classes", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Optional dataset paths (if omitted, we'll auto-resolve)
    ap.add_argument("--image_prefix", type=str, default=None)
    ap.add_argument("--qgrid_prefix", type=str, default=None)
    ap.add_argument("--kp_path", type=str, default=None)
    ap.add_argument("--meta_dir_path", type=str, default=None)
    ap.add_argument("--gloss_dict_path", type=str, default=None)
    return ap.parse_args()


# ==========================
# Evaluation
# ==========================
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

    total_wer_edits, total_wer_tokens = 0, 0
    total_cer_edits, total_cer_chars = 0, 0
    total_samples = 0

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if (use_bf16 and device.startswith("cuda")) else torch.cuda.amp.autocast(enabled=False)

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
        toks = [id2gloss.get(int(x), "<UNK>") for x in seq]
        return toks

    def edit_distance(a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
        return dp[m][n]

    def cer_edits(hyp_str, ref_str):
        a = list(hyp_str)
        b = list(ref_str)
        return edit_distance(a, b), len(b)

    for batch in dl:
        # Dict-style batch from your dataset
        if isinstance(batch, dict):
            images = batch.get("images") or batch.get("frames")
            qgrids = batch.get("qgrids") or batch.get("qgrid")
            keypoints = batch.get("keypoints") or batch.get("kps") or batch.get("pose")
            qgrid_lengths = batch.get("qgrid_lengths") or batch.get("qgrid_len") or batch.get("lengths")
            # references
            y_ids = batch.get("gloss_ids") or batch.get("targets") or batch.get("labels")
            y_text = batch.get("gloss_str") or batch.get("gloss") or batch.get("text")
        else:
            raise RuntimeError("Batch format unexpected; expected dict-like from MultiModalPhoenixDataset.")

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


# ==========================
# Main
# ==========================
def main():
    args = parse_args()
    warnings.filterwarnings("once", category=FutureWarning)

    device = args.device
    map_location = torch.device(device)

    # ----- dataset -----
    # Resolve dataset paths if constructor requires them
    ds = None
    try:
        # Try the minimal constructor first (some versions support this)
        ds = MultiModalPhoenixDataset(split=args.split)
    except TypeError:
        # Fall back to full-arg constructor
        image_prefix, qgrid_prefix, kp_path, meta_dir_path, gloss_dict_path = resolve_dataset_paths(args)
        print(f"[dataset paths]\n  image_prefix = {image_prefix}\n  qgrid_prefix = {qgrid_prefix}\n  kp_path = {kp_path}\n  meta_dir = {meta_dir_path}\n  gloss_dict = {gloss_dict_path}")
        ds = MultiModalPhoenixDataset(
            image_prefix=image_prefix,
            qgrid_prefix=qgrid_prefix,
            kp_path=kp_path,
            meta_dir_path=meta_dir_path,
            gloss_dict_path=gloss_dict_path,
            split=args.split,
        )

    # id2gloss / blank idx
    id2gloss = getattr(ds, "id2gloss", None)
    if id2gloss is None:
        gloss_list = getattr(ds, "gloss_list", None)
        if gloss_list is not None and isinstance(gloss_list, (list, tuple)):
            id2gloss = {i: g for i, g in enumerate(gloss_list)}
        else:
            raise RuntimeError("Could not locate id2gloss mapping on dataset.")
    blank_idx = getattr(ds, "blank_idx", 0)
    print(f"[gloss] loaded {len(id2gloss)} entries; blank idx={blank_idx}; sample: {list(id2gloss.values())[:5]}")

    # Let the dataset report what it kept/dropped (dataset likely prints this internally)
    print(f"[MultiModalPhoenixDataset] len={len(ds)} split={args.split}")

    collate_fn = getattr(ds, "collate_fn", None)
    if callable(collate_fn):
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=collate_fn)
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ----- model -----
    n_classes = (args.force_num_classes or len(id2gloss))
    if args.force_num_classes:
        print(f"[head] num_classes={n_classes} (forced)")
    else:
        print(f"[head] num_classes={n_classes}")

    model = MultiModalMamba(num_classes=n_classes)
    model.to(device)

    # load checkpoint (try new safe API first)
    ckpt_path = args.ckpt
    print(f"[ckpt] loading: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)  # PyTorch 2.5+
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=map_location)

    # Accept common formats
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt
        else:
            state_dict = ckpt.get("model", ckpt.get("model_state", {})) if isinstance(ckpt, dict) else {}

    # strip common prefixes
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

    # Non-strict to allow head or depth mismatches
    ld = model.load_state_dict(state_dict, strict=False)
    # PyTorch <=2.4 returns NamedTuple; in 2.5 it's dict-like. Handle both.
    missing = getattr(ld, "missing_keys", None) or ld.get("missing_keys", [])
    unexpected = getattr(ld, "unexpected_keys", None) or ld.get("unexpected_keys", [])
    if unexpected:
        print(f"[ckpt] strict load failed with unexpected keys ({len(unexpected)}). "
              f"Loading non-head layers and reinitializing classifier.")
    if missing:
        print(f"[ckpt] missing keys: {len(missing)}")

    # Patch frame encoder to accept (B, T, C, H, W)
    if not isinstance(model.frame_encoder, _FrameEncoderAdapter):
        model.frame_encoder = _FrameEncoderAdapter(model.frame_encoder)
    print("[patch] FrameEncoderAdapter attached.")

    # ----- eval -----
    use_bf16 = bool(args.bf16)
    WER, CER, N = evaluate(model, dl, id2gloss, device, use_bf16=use_bf16)
    print(f"[RESULT] N={N}  WER={WER:.2f}%  CER={CER:.2f}%")


if __name__ == "__main__":
    main()
