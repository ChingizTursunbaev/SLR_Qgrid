# slr/engine.py
import math, time
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
import torch.distributed as dist

try:
    from torchmetrics.functional.text import word_error_rate as tm_wer
    _HAS_TM = True
except Exception:
    _HAS_TM = False


def _is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _reduce_mean(val: torch.Tensor) -> float:
    if dist.is_available() and dist.is_initialized():
        t = val.detach().clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
        return t.item()
    return val.item()


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _ctc_input_lengths(log_probs: torch.Tensor) -> torch.Tensor:
    # log_probs: (B, T, V)
    B, T, _ = log_probs.shape
    return torch.full((B,), T, dtype=torch.long, device=log_probs.device)


def _ctc_greedy_decode(log_probs: torch.Tensor) -> List[List[int]]:
    # log_probs: (B, T, V)
    with torch.no_grad():
        pred = log_probs.argmax(dim=-1)  # (B,T)
        B, T = pred.shape
        hyps: List[List[int]] = []
        for b in range(B):
            prev = -1
            seq: List[int] = []
            for t in range(T):
                p = int(pred[b, t].item())
                if p != 0 and p != prev:  # remove blanks(0) and repeats
                    seq.append(p)
                prev = p
            hyps.append(seq)
        return hyps


def _ids_to_gloss(ids: List[int], idx2gloss: Dict[int, str]) -> str:
    toks = []
    for i in ids:
        if i == 0:
            continue
        toks.append(idx2gloss.get(i, str(i)))
    return " ".join(toks)


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch: int,
    scaler,
    amp_dtype: torch.dtype,
    max_norm: float = 1.0,
    accum: int = 1,
) -> Dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    num_samples = 0
    skipped = 0

    start = time.time()
    last = start

    for it, batch in enumerate(data_loader):
        batch = _move_batch(batch, device)

        images         = batch["images"]         # (B, T_img, 3, H, W)
        qgrids         = batch["qgrids"]         # (B, T_q, 242)  {-1,0,1}
        keypoints      = batch["keypoints"]      # (B, T_img, 242)
        targets        = batch["targets"]        # (sumL,)
        target_lengths = batch["target_lengths"] # (B,)
        qgrid_lengths  = batch.get("qgrid_lengths", None)

        B = images.size(0)
        num_samples += B

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(images, qgrids, keypoints, qgrid_lengths=qgrid_lengths)  # (B, T_img, V)
            log_probs = F.log_softmax(logits, dim=-1)                               # (B, T_img, V)
            in_lens = _ctc_input_lengths(log_probs)                                  # (B,)

            # CTC expects (T, N, C)
            loss = criterion(log_probs.transpose(0, 1), targets, in_lens, target_lengths)

        # Guard against NaNs/Infs (skip this batch)
        if (not torch.isfinite(loss)) or torch.isnan(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if _is_main():
                print(f"[warn] non-finite loss at step {it}, skipping batch")
            continue

        loss_div = loss / max(1, accum)
        scaler.scale(loss_div).backward()

        # optimizer step
        if (it + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach()
        # lightweight progress
        if _is_main() and (it % 50 == 0):
            now = time.time()
            speed = (it + 1) / max(1e-6, (now - start))
            print(f"Epoch: [{epoch}]  [{it}/{len(data_loader)}]  lr: {optimizer.param_groups[0]['lr']:.6f}  "
                  f"loss: {loss.item():.4f}  accum:{accum}  time/50b: {now-last:.1f}s")
            last = now

    # average loss across workers
    avg_loss = running_loss / max(1, len(data_loader))
    avg_loss_val = _reduce_mean(avg_loss) if torch.is_tensor(avg_loss) else float(avg_loss)
    return {"loss": float(avg_loss_val), "skipped": int(skipped)}


def evaluate(model, data_loader, device) -> float:
    model.eval()
    total_wer = 0.0
    n_batches = 0

    # build idx2gloss mapping
    ds = data_loader.dataset
    idx2gloss = None
    gd = getattr(ds, "gloss_dict", None)
    if isinstance(gd, dict):
        # assumes labels start at 1 (blank=0)
        idx2gloss = {v: k for k, v in gd.items()}
    elif isinstance(gd, (list, tuple)):
        idx2gloss = {i + 1: g for i, g in enumerate(gd)}
    else:
        idx2gloss = {}

    with torch.no_grad():
        for batch in data_loader:
            batch = _move_batch(batch, device)
            images         = batch["images"]
            qgrids         = batch["qgrids"]
            keypoints      = batch["keypoints"]
            target_flat    = batch["targets"].cpu().tolist()
            target_lengths = batch["target_lengths"].cpu().tolist()
            qgrid_lengths  = batch.get("qgrid_lengths", None)

            # split flat targets to per-sample lists
            targets = []
            offset = 0
            for L in target_lengths:
                seq = target_flat[offset:offset + L]
                targets.append(seq)
                offset += L

            logits = model(images, qgrids, keypoints, qgrid_lengths=qgrid_lengths)
            log_probs = F.log_softmax(logits, dim=-1)

            # greedy CTC decode
            pred_ids = _ctc_greedy_decode(log_probs)

            # map to strings
            refs = [_ids_to_gloss(t, idx2gloss) for t in targets]
            hyps = [_ids_to_gloss(h, idx2gloss) for h in pred_ids]

            if _HAS_TM:
                wer_val = float(tm_wer(hyps, refs).item()) * 100.0
            else:
                # fallback: average per-sample WER via simple ratio
                # NOTE: not exact but fine as a fallback
                import re
                def _wer_simple(h, r):
                    hw = re.findall(r"\S+", h)
                    rw = re.findall(r"\S+", r)
                    # simple Levenshtein for words
                    dp = [[0]*(len(hw)+1) for _ in range(len(rw)+1)]
                    for i in range(len(rw)+1): dp[i][0]=i
                    for j in range(len(hw)+1): dp[0][j]=j
                    for i in range(1, len(rw)+1):
                        for j in range(1, len(hw)+1):
                            cost = 0 if rw[i-1]==hw[j-1] else 1
                            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
                    denom = max(1, len(rw))
                    return 100.0 * dp[len(rw)][len(hw)] / denom
                wer_val = sum(_wer_simple(h, r) for h, r in zip(hyps, refs)) / max(1, len(hyps))

            total_wer += wer_val
            n_batches += 1

    # average across workers
    w = torch.tensor([total_wer, n_batches], device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(w, op=dist.ReduceOp.SUM)
    total_wer, n_batches = float(w[0].item()), int(w[1].item())
    return total_wer / max(1, n_batches)