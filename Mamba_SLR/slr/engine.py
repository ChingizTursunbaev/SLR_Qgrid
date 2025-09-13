import math
import sys
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from timm.utils import ModelEma
from . import utils
from torchmetrics.text import WordErrorRate


def _flatten_targets(labels: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:
    """Flatten padded (B, L) labels into 1D targets for CTCLoss."""
    parts = []
    for b in range(labels.size(0)):
        Lb = int(label_lengths[b].item())
        if Lb > 0:
            parts.append(labels[b, :Lb])
    if not parts:
        # avoid empty vector (CTC would error)
        return torch.zeros(1, dtype=torch.long, device=labels.device)
    return torch.cat(parts, dim=0).to(dtype=torch.long)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast,
                    max_norm: float = 0, model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=1, no_amp=False, bf16=False):

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    optimizer.zero_grad(set_to_none=True)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # lr schedule per micro-step
        micro_it = (start_steps or 0) + data_iter_step
        if lr_schedule_values is not None:
            lr_val = lr_schedule_values[micro_it]
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = lr_val * pg.get("lr_scale", 1.0)
                if wd_schedule_values is not None and pg.get("weight_decay", 0) > 0:
                    pg["weight_decay"] = wd_schedule_values[micro_it]

        images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths = batch
        images        = images.to(device, non_blocking=True)
        qgrids        = qgrids.to(device, non_blocking=True)
        keypoints     = keypoints.to(device, non_blocking=True)
        labels        = labels.to(device, non_blocking=True)
        image_lengths = image_lengths.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        qgrid_lengths = qgrid_lengths.to(device, non_blocking=True)

        with amp_autocast:
            logits    = model(images, qgrids, keypoints, qgrid_lengths=qgrid_lengths)  # (B, T, V)
            log_probs = logits.permute(1, 0, 2).log_softmax(dim=2)                      # (T, B, V)
            targets   = _flatten_targets(labels, label_lengths)                          # (sum L,)
            loss      = criterion(log_probs, targets, image_lengths, label_lengths)

        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training", flush=True)
            sys.exit(1)

        # gradient accumulation
        loss = loss / update_freq
        if loss_scaler is not None:
            # torch.amp.GradScaler path
            loss_scaler.scale(loss).backward()
            if (data_iter_step + 1) % update_freq == 0:
                if max_norm and max_norm > 0:
                    loss_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                loss_scaler.step(optimizer)
                loss_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    # model may be wrapped in DDP; ModelEma handles .module
                    model_ema.update(model)
            loss_scale_value = float(getattr(loss_scaler, "get_scale", lambda: 0.0)())
        else:
            # FP32 path
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                if max_norm and max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = 0.0

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value, loss_scale=loss_scale_value, lr=optimizer.param_groups[0]["lr"])

    # gather stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, gloss_dict, ds=True, no_amp=False, bf16=False):
    """
    Batch: images (B,T,C,H,W), qgrids (B,Tq,242), keypoints (B,T,242),
           labels (B,L), image_lengths (B), label_lengths (B), qgrid_lengths (B)
    Model returns logits (B, T_img, V).
    """
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    wer_metric = WordErrorRate().to(device)

    id_to_gloss = {v: k for k, v in gloss_dict.items()}
    blank_id = 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths = batch

        images        = images.to(device, non_blocking=True)
        qgrids        = qgrids.to(device, non_blocking=True)
        keypoints     = keypoints.to(device, non_blocking=True)
        labels        = labels.to(device, non_blocking=True)
        image_lengths = image_lengths.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        qgrid_lengths = qgrid_lengths.to(device, non_blocking=True)

        with amp_autocast:
            logits        = model(images, qgrids, keypoints, qgrid_lengths=qgrid_lengths)  # (B, T, V)
            log_probs_BTV = logits.log_softmax(dim=-1)
            log_probs_TBV = log_probs_BTV.permute(1, 0, 2)
            targets       = _flatten_targets(labels, label_lengths)
            loss = criterion(log_probs_TBV, targets, image_lengths, label_lengths)

        metric_logger.update(loss=loss.item())

        # Greedy decode for WER
        preds_B_T = log_probs_BTV.argmax(dim=-1)  # (B,T)
        decoded_preds = []
        for b in range(preds_B_T.size(0)):
            T_valid = int(image_lengths[b].item())
            seq = preds_B_T[b, :T_valid]
            seq = seq[seq != blank_id]
            if seq.numel() > 0:
                seq = torch.unique_consecutive(seq)
            words = [id_to_gloss.get(int(tok), "?") for tok in seq.tolist()]
            decoded_preds.append(" ".join(words))

        true_glosses = []
        for lbl_row, L in zip(labels, label_lengths):
            valid = lbl_row[:int(L.item())].tolist()
            words = [id_to_gloss.get(int(tok), "?") for tok in valid]
            true_glosses.append(" ".join(words))

        wer_metric.update(decoded_preds, true_glosses)

    wer_score = wer_metric.compute().item() * 100.0
    metric_logger.update(wer=wer_score)
    metric_logger.synchronize_between_processes()
    print(f"* WER {metric_logger.meters['wer'].global_avg:.2f}% loss {metric_logger.meters['loss'].global_avg:.3f}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
