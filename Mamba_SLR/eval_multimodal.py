# eval_multimodal.py
# Final version with a frame-encoder adapter to handle (B, T, C, H, W) inputs.
# Your original evaluation logic stays the same; only marked sections are new.

import os
import argparse
import warnings
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- your repo imports (unchanged) ----
# If your import paths differ, keep yours.
from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset
from slr.models.multi_modal_model import MultiModalMamba

# ==========================
# BEGIN: NEW ADAPTER SECTION
# ==========================
class _FrameEncoderAdapter(nn.Module):
    """
    Wraps an existing frame_encoder so it can take (B, T, C, H, W)
    and returns (B, T, D). If it already receives (B*T, C, H, W),
    it passes through unchanged.
    """
    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # Already (B*T, C, H, W)
            return self.enc(x)

        if x.dim() != 5:
            raise RuntimeError(f"[adapter] Expected (B,T,C,H,W) or (B*T,C,H,W), got {tuple(x.shape)}")

        B, T, C, H, W = x.shape
        x_bt = x.reshape(B * T, C, H, W)       # (B*T, C, H, W)
        y = self.enc(x_bt)                     # typically (B*T, D) or (B*T, P, D)

        # Normalize to (B, T, D)
        if y.dim() == 2:
            return y.view(B, T, -1)
        if y.dim() == 3:
            # If backbone emits patch tokens (B*T, P, D), mean-pool patches
            y = y.mean(dim=1)
            return y.view(B, T, -1)

        raise RuntimeError(f"[adapter] Unexpected frame_encoder output shape {tuple(y.shape)}")
# ========================
# END: NEW ADAPTER SECTION
# ========================


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    ap.add_argument("--split", type=str, default="test", choices=["train", "dev", "valid", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--force_num_classes", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


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
        # remove blanks & repeated tokens, then map to strings (space-joined)
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
        # Levenshtein on token lists
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,         # del
                    dp[i][j-1] + 1,         # ins
                    dp[i-1][j-1] + cost     # sub
                )
        return dp[m][n]

    def cer_edits(hyp_str, ref_str):
        # CER on raw strings (no spaces removed)
        a = list(hyp_str)
        b = list(ref_str)
        return edit_distance(a, b), len(b)

    for batch in dl:
        # Support both dict-style or tuple-style batches
        if isinstance(batch, dict):
            images = batch.get("images") or batch.get("frames")
            qgrids = batch.get("qgrids") or batch.get("qgrid")
            keypoints = batch.get("keypoints") or batch.get("kps") or batch.get("pose")
            qgrid_lengths = batch.get("qgrid_lengths") or batch.get("qgrid_len") or batch.get("lengths")
            # reference targets
            y_ids = batch.get("gloss_ids") or batch.get("targets") or batch.get("labels")
            y_text = batch.get("gloss_str") or batch.get("gloss") or batch.get("text")
        else:
            # If your dataset returns a tuple, keep your original unpacking here.
            raise RuntimeError("Batch format unexpected; please keep your original collate/unpack logic.")

        images = images.to(device, non_blocking=True)
        qgrids = qgrids.to(device, non_blocking=True) if qgrids is not None else None
        keypoints = keypoints.to(device, non_blocking=True) if keypoints is not None else None
        qgrid_lengths = qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None

        with autocast_ctx:
            logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B, T', C)

        # Greedy
        pred_ids = logits.argmax(dim=-1).cpu()  # (B, T')

        B = pred_ids.size(0)
        for b in range(B):
            hyp_toks = collapse_and_map(pred_ids[b].tolist())
            hyp_str = " ".join(hyp_toks)

            # build reference string
            if y_text is not None:
                # already tokens or space-joined
                ref_str = y_text[b] if isinstance(y_text, (list, tuple)) else y_text
                if isinstance(ref_str, list):
                    ref_str = " ".join(ref_str)
            elif y_ids is not None:
                ids = y_ids[b].tolist()
                ref_toks = [id2gloss.get(int(t), "<UNK>") for t in ids if int(t) != blank_idx]
                ref_str = " ".join(ref_toks)
            else:
                ref_str = ""

            # WER on whitespace tokens
            hyp_words = hyp_str.split()
            ref_words = ref_str.split()
            total_wer_edits += edit_distance(hyp_words, ref_words)
            total_wer_tokens += max(1, len(ref_words))

            # CER
            ed, L = cer_edits(hyp_str, ref_str)
            total_cer_edits += ed
            total_cer_chars += max(1, L)

            total_samples += 1

    WER = 100.0 * total_wer_edits / max(1, total_wer_tokens)
    CER = 100.0 * total_cer_edits / max(1, total_cer_chars)
    return WER, CER, total_samples


def main():
    args = parse_args()

    warnings.filterwarnings("once", category=FutureWarning)

    device = args.device
    map_location = torch.device(device)

    # ----- dataset -----
    ds = MultiModalPhoenixDataset(split=args.split)
    print(f"[MultiModalPhoenixDataset] len={len(ds)} split={args.split}")

    # Some repos attach id2gloss/blank_idx on the dataset
    id2gloss = getattr(ds, "id2gloss", None)
    if id2gloss is None:
        # Fallback: try to read from numpy list on dataset if present
        gloss_list = getattr(ds, "gloss_list", None)
        if gloss_list is not None and isinstance(gloss_list, (list, tuple)):
            id2gloss = {i: g for i, g in enumerate(gloss_list)}
        else:
            raise RuntimeError("Could not locate id2gloss mapping on dataset.")

    blank_idx = getattr(ds, "blank_idx", 0)
    print(f"[gloss] loaded {len(id2gloss)} entries; blank idx={blank_idx}; sample: {list(id2gloss.values())[:5]}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
        # If you have a custom collate in your repo, add it here:
        # collate_fn=your_collate_fn
    )

    # ----- model -----
    model = MultiModalMamba(num_classes=(args.force_num_classes or len(id2gloss)))
    model.to(device)

    # load checkpoint (safe)
    ckpt_path = args.ckpt
    print(f"[ckpt] loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # Accept common formats
    state_dict = ckpt.get("state_dict", None)
    if state_dict is None:
        # sometimes checkpoints store straight state dict
        if all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt
        else:
            # last resort
            state_dict = ckpt.get("model", ckpt.get("model_state", {}))

    # strip potential prefixes ('module.', 'model.')
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

    # Try non-strict first; classifier may be reinit if num_classes forced
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[ckpt] strict load failed with unexpected keys ({len(unexpected)}). "
              f"Loading non-head layers and reinitializing classifier.")
    if missing:
        print(f"[ckpt] missing keys: {len(missing)}")

    # ================================
    # BEGIN: NEW WRAPPING LINE (KEY!)
    # ================================
    # Make the frame encoder accept (B, T, C, H, W) and return (B, T, D)
    model.frame_encoder = _FrameEncoderAdapter(model.frame_encoder)
    print("[patch] FrameEncoderAdapter attached.")
    # ==============================
    # END: NEW WRAPPING LINE (KEY!)
    # ==============================

    # ----- eval -----
    use_bf16 = bool(args.bf16)
    WER, CER, N = evaluate(model, dl, id2gloss, device, use_bf16=use_bf16)
    print(f"[RESULT] N={N}  WER={WER:.2f}%  CER={CER:.2f}%")


if __name__ == "__main__":
    main()
