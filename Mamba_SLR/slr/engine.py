import math
import sys
from typing import Iterable, Optional
import pickle

import torch
import torch.distributed as dist
from timm.utils import ModelEma
import utils
from torchmetrics.text import WordErrorRate

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, no_amp=False, bf16=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None: # For DeepSpeed
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step

        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, qgrids, keypoints, labels, image_lengths, label_lengths = batch
        images = images.to(device, non_blocking=True)
        qgrids = qgrids.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        image_lengths = image_lengths.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)

        # Autocasting for mixed precision
        with amp_autocast:
            print(f"[DEBUG] Rank {torch.distributed.get_rank()} Images shape: {images.shape}")
            outputs = model(images, qgrids, keypoints)
            loss = criterion(outputs, labels, image_lengths, label_lengths)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), flush=True)
            sys.exit(1)
        
        if loss_scaler is None: # Deepspeed backward
            loss /= update_freq
            model.backward(loss)
            model.step()
            if (data_iter_step + 1) % update_freq == 0 and model_ema is not None:
                model_ema.update(model)
            loss_scale_value = model.optimizer.loss_scale if hasattr(model.optimizer, "loss_scale") else 0
        else: # Standard PyTorch backward
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, gloss_dict, ds=True, no_amp=False, bf16=False):
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    wer_metric = WordErrorRate().to(device)
    
    id_to_gloss = {v: k for k, v in gloss_dict.items()}

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, qgrids, keypoints, labels, image_lengths, label_lengths = batch
        images = images.to(device, non_blocking=True)
        qgrids = qgrids.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        image_lengths = image_lengths.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)

        with amp_autocast:
            output = model(images, qgrids, keypoints)
            loss = criterion(output, labels, image_lengths, label_lengths)

        metric_logger.update(loss=loss.item())

        log_probs = output.permute(1, 0, 2).log_softmax(dim=2)
        predictions = torch.argmax(log_probs, dim=2)

        decoded_preds = []
        for pred_tensor in predictions:
            pred_tensor_no_blanks = pred_tensor[pred_tensor != 0]
            unique_tokens = torch.unique_consecutive(pred_tensor_no_blanks)
            decoded_words = [id_to_gloss.get(token.item(), "?") for token in unique_tokens]
            decoded_preds.append(" ".join(decoded_words))

        true_glosses = []
        for label_tensor, length in zip(labels, label_lengths):
            valid_labels = label_tensor[:length]
            true_words = [id_to_gloss.get(lbl.item(), "?") for lbl in valid_labels]
            true_glosses.append(" ".join(true_words))
        
        wer_metric.update(decoded_preds, true_glosses)

    wer_score = wer_metric.compute().item() * 100
    metric_logger.update(wer=wer_score)
    
    metric_logger.synchronize_between_processes()
    print('* WER {wer.global_avg:.2f}% loss {losses.global_avg:.3f}'
          .format(wer=metric_logger.wer, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

