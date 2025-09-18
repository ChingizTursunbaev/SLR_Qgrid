#!/usr/bin/env python
# ddp_train_multimodal.py
# Distributed training entrypoint for MultiModal Mamba SLR with length-safe model + optional BF16.
import os
import argparse
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ---- Project imports ----
from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.models.multi_modal_model import MultiModalMamba


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return device, local_rank
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0


def cleanup_ddp():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def preflight_print(gloss_path: str, shift_labels_by: int, num_classes: int, blank_idx: int):
    if is_main_process():
        try:
            import numpy as np
            arr = np.load(gloss_path, allow_pickle=True)
            if isinstance(arr.item(), dict):
                ids = list(arr.item().values())
                gloss_min, gloss_max = min(ids), max(ids)
            else:
                ids = [int(v) for v in arr.ravel() if isinstance(v, (int, np.integer))]
                gloss_min, gloss_max = (min(ids), max(ids)) if len(ids) else (1, num_classes - 1)
        except Exception:
            gloss_min, gloss_max = (1, num_classes - 1)

        print(f"[preflight] gloss_id_range=[{gloss_min}, {gloss_max}]  "
              f"shift_labels_by={shift_labels_by}  num_classes={num_classes}  blank_idx={blank_idx}", flush=True)


def find_1d_lengths(cands: List[torch.Tensor], target_dim: int) -> Optional[torch.Tensor]:
    for t in cands:
        if t is None or t.dim() != 1:
            continue
        if int(t.max().item()) <= target_dim:
            return t
    return None


def unpack_batch(batch: Any) -> Dict[str, Any]:
    """
    The dataset's collate returns a tuple. We infer components by shape/dtype:
      - images: (B,T,C,H,W)
      - qgrids/keypoints: (B,T,D)
      - labels: (B,Lmax) long
      - lengths: 1D longs that should be <= their respective padded dims
    Returns a dict with keys: images, qgrids, keypoints, qgrid_lengths, labels, label_lengths, image_lengths
    Missing pieces may be None (handled later).
    """
    if isinstance(batch, dict):
        return batch

    if not isinstance(batch, (list, tuple)):
        raise RuntimeError(f"Unexpected batch type {type(batch)}")

    imgs = None; qg = None; kp = None; labels = None
    one_d_lens: List[torch.Tensor] = []

    for item in batch:
        if torch.is_tensor(item):
            if item.dim() == 5 and item.dtype in (torch.float16, torch.bfloat16, torch.float32):
                imgs = item
            elif item.dim() == 3 and item.dtype in (torch.float16, torch.bfloat16, torch.float32):
                if qg is None:
                    qg = item
                else:
                    kp = item
            elif item.dim() == 2 and item.dtype in (torch.int32, torch.int64, torch.long):
                labels = item
            elif item.dim() == 1 and item.dtype in (torch.int32, torch.int64, torch.long):
                one_d_lens.append(item)

    B = imgs.size(0) if imgs is not None else (labels.size(0) if labels is not None else batch[0].size(0))

    T_img = imgs.size(1) if imgs is not None else None
    T_q = qg.size(1) if qg is not None else None
    L_lab = labels.size(1) if labels is not None else None

    image_lengths = find_1d_lengths(one_d_lens, T_img or 10**9)
    qgrid_lengths = find_1d_lengths(one_d_lens, T_q or 10**9)
    label_lengths = find_1d_lengths(one_d_lens, L_lab or 10**9)

    return {
        "images": imgs,
        "qgrids": qg,
        "keypoints": kp,
        "image_lengths": image_lengths if image_lengths is not None else (torch.full((B,), T_img or 1, dtype=torch.long, device=imgs.device if imgs is not None else labels.device)),
        "qgrid_lengths": qgrid_lengths,
        "labels": labels,
        "label_lengths": label_lengths,
    }


def flatten_targets(labels: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    for i in range(labels.size(0)):
        L = int(label_lengths[i].item())
        if L > 0:
            outs.append(labels[i, :L])
    if len(outs) == 0:
        return torch.zeros((0,), dtype=torch.long, device=labels.device)
    return torch.cat(outs, dim=0)


def valid_ctc_targets(targets: torch.Tensor, num_classes: int, blank_idx: int) -> bool:
    if targets.numel() == 0:
        return True
    tmin = int(targets.min().item())
    tmax = int(targets.max().item())
    if tmin < 0 or tmax >= num_classes:
        return False
    if (targets == blank_idx).any():
        return False
    return True


def train(args):
    device, local_rank = setup_ddp()
    torch.backends.cudnn.benchmark = True

    train_set = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,
        gloss_dict_path=args.gloss_dict,
        split="train",
    )
    val_set = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,
        gloss_dict_path=args.gloss_dict,
        split="dev",
    )

    train_sampler = DistributedSampler(train_set, shuffle=True) if torch.distributed.is_initialized() else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if torch.distributed.is_initialized() else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=multi_modal_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=multi_modal_collate_fn,
        drop_last=False,
    )

    model = MultiModalMamba(
        d_model=args.d_model,
        n_layer=args.n_layer,
        fusion_embed=args.fusion_embed,
        fusion_heads=args.fusion_heads,
        num_classes=args.num_classes,
        max_kv=args.max_kv,
        pool_mode=args.pool_mode,
    ).to(device)

    if torch.distributed.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    blank_idx = 0
    ctc_loss = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else None
    if is_main_process():
        preflight_print(args.gloss_dict, shift_labels_by=0, num_classes=args.num_classes, blank_idx=blank_idx)
        if args.bf16 and not use_bf16:
            print("[warn] --bf16 requested but not supported on this GPU/driver; training in fp32.", flush=True)

    step = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        running = 0.0
        seen = 0
        for it, batch in enumerate(train_loader):
            batch = unpack_batch(batch)
            images = batch["images"].to(device, non_blocking=True) if batch["images"] is not None else None
            qgrids = batch["qgrids"].to(device, non_blocking=True) if batch["qgrids"] is not None else None
            keypoints = batch["keypoints"].to(device, non_blocking=True) if batch["keypoints"] is not None else None
            qgrid_lengths = batch["qgrid_lengths"].to(device, non_blocking=True) if batch["qgrid_lengths"] is not None else None
            label_lengths = batch["label_lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B,T,V)
                    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T,B,V)
            else:
                logits = model(images, qgrids, keypoints, qgrid_lengths)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            B, T, V = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            targets = flatten_targets(labels, label_lengths)
            if not valid_ctc_targets(targets, args.num_classes, blank_idx):
                if is_main_process():
                    tmin = int(targets.min().item()) if targets.numel() else -1
                    tmax = int(targets.max().item()) if targets.numel() else -1
                    print(f"[skip-batch] invalid targets (min={tmin}, max={tmax}, V={args.num_classes}); step={step} epoch={epoch} it={it}", flush=True)
                continue

            loss = ctc_loss(log_probs, targets, input_lengths, label_lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * B
            seen += B
            step += 1
            if is_main_process() and (it % args.log_interval == 0):
                print(f"Epoch: [{epoch}]  [{it}/{len(train_loader)}]  lr: {args.lr:.6f}  loss: {running/ max(1,seen):.4f}", flush=True)

        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = unpack_batch(batch)
                images = batch["images"].to(device, non_blocking=True) if batch["images"] is not None else None
                qgrids = batch["qgrids"].to(device, non_blocking=True) if batch["qgrids"] is not None else None
                keypoints = batch["keypoints"].to(device, non_blocking=True) if batch["keypoints"] is not None else None
                qgrid_lengths = batch["qgrid_lengths"].to(device, non_blocking=True) if batch["qgrid_lengths"] is not None else None
                label_lengths = batch["label_lengths"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(images, qgrids, keypoints, qgrid_lengths)
                        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                else:
                    logits = model(images, qgrids, keypoints, qgrid_lengths)
                    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

                B, T, V = logits.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                targets = flatten_targets(labels, label_lengths)
                if not valid_ctc_targets(targets, args.num_classes, 0):
                    continue
                loss = ctc_loss(log_probs, targets, input_lengths, label_lengths)
                val_loss += float(loss.item()) * B
                val_seen += B

        if is_main_process():
            avg_train = running / max(1, seen)
            avg_val = val_loss / max(1, val_seen)
            print(f"[epoch {epoch}] train_loss={avg_train:.4f}  val_loss={avg_val:.4f}", flush=True)

    cleanup_ddp()


def get_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    p.add_argument("--out_dir",      default="checkpoints/multimodal_ddp")
    # Model
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--fusion_embed", type=int, default=512)
    p.add_argument("--fusion_heads", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=1296)
    p.add_argument("--max_kv", type=int, default=512)
    p.add_argument("--pool_mode", type=str, default="mean", choices=["mean", "max", "vote"])
    # Train
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--log_interval", type=int, default=50)
    # Precision
    p.add_argument("--bf16", action="store_true", help="enable autocast bfloat16 if supported")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)























# # ddp_train_multimodal.py
# import os, time
# import argparse
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, DistributedSampler
# from torch.amp import GradScaler, autocast
# from torch import optim

# from slr.datasets.multi_modal_datasets import (
#     MultiModalPhoenixDataset,
#     multi_modal_collate_fn,
# )
# from slr.models.multi_modal_model import MultiModalMamba
# from slr.engine import train_one_epoch, evaluate


# def parse_args():
#     ap = argparse.ArgumentParser()
#     # paths (set your own defaults if needed)
#     ap.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
#     ap.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
#     ap.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
#     ap.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
#     ap.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
#     ap.add_argument("--out_dir",      default="checkpoints/multimodal_ddp")

#     # training
#     ap.add_argument("--epochs", type=int, default=30)
#     ap.add_argument("--lr", type=float, default=1e-4)
#     ap.add_argument("--weight_decay", type=float, default=0.05)
#     ap.add_argument("--batch_size", type=int, default=1, help="per-GPU batch size")
#     ap.add_argument("--num_workers", type=int, default=4)
#     ap.add_argument("--max_norm", type=float, default=1.0)
#     ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
#     ap.add_argument("--bf16", action="store_true", help="use bfloat16 autocast on A100+ (recommended)")

#     # model sizing
#     ap.add_argument("--d_model", type=int, default=512)
#     ap.add_argument("--n_layer", type=int, default=12)
#     ap.add_argument("--fusion_embed", type=int, default=512)
#     ap.add_argument("--fusion_heads", type=int, default=8)

#     # fusion / pooling
#     ap.add_argument("--max_kv", type=int, default=512, help="pooled qgrid length")
#     ap.add_argument("--pool_mode", default="mean", choices=["mean", "max", "vote"])
#     return ap.parse_args()


# def is_main():
#     return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


# def setup_dist():
#     if dist.is_available() and not dist.is_initialized():
#         dist.init_process_group(backend="nccl", init_method="env://")


# def cleanup_dist():
#     if dist.is_available() and dist.is_initialized():
#         dist.destroy_process_group()


# def main():
#     # Safer NCCL defaults
#     os.environ.setdefault("NCCL_IB_DISABLE", "1")
#     os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
#     os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29531"))

#     # Enable TF32 for extra stability/speed on A100
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

#     args = parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)

#     setup_dist()
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     device = torch.device(f"cuda:{local_rank}")
#     torch.cuda.set_device(local_rank)
#     torch.backends.cudnn.benchmark = True

#     # Different ranks get different seeds
#     base_seed = 1337
#     if dist.is_available() and dist.is_initialized():
#         torch.manual_seed(base_seed + dist.get_rank())
#     else:
#         torch.manual_seed(base_seed)

#     # ----- data -----
#     train_ds = MultiModalPhoenixDataset(
#     image_prefix=args.image_prefix,
#     qgrid_prefix=args.qgrid_prefix,
#     kp_path=args.kp_path,
#     meta_dir_path=args.meta_dir,         # <-- renamed kwarg
#     gloss_dict_path=args.gloss_dict,     # <-- renamed kwarg
#     split="train",
#     transforms=None,
#     )
    
#     val_ds = MultiModalPhoenixDataset(
#         image_prefix=args.image_prefix,
#         qgrid_prefix=args.qgrid_prefix,
#         kp_path=args.kp_path,
#         meta_dir_path=args.meta_dir,
#         gloss_dict_path=args.gloss_dict,
#         split="dev",
#         transforms=None,
#     )

#     if dist.is_available() and dist.is_initialized():
#         train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
#         val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
#     else:
#         train_sampler = None
#         val_sampler = None

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         sampler=train_sampler,
#         shuffle=(train_sampler is None),
#         pin_memory=True,
#         drop_last=False,
#         collate_fn=multi_modal_collate_fn,
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         sampler=val_sampler,
#         shuffle=False,
#         pin_memory=True,
#         drop_last=False,
#         collate_fn=multi_modal_collate_fn,
#     )

#     # ----- model -----
#     img_cfg = {
#         "d_model": args.d_model,
#         "n_layer": args.n_layer,
#         "fused_add_norm": False,  # Triton fused LN off (stability)
#         "rms_norm": False,
#     }
#     qgrid_cfg = {"out_dim": args.fusion_embed}
#     kp_cfg = {"input_dim": 242, "model_dim": args.d_model}
#     fusion = {
#         "embed_dim": args.fusion_embed,
#         "num_heads": args.fusion_heads,
#         "dropout": 0.1,
#         "max_kv": args.max_kv,
#         "pool_mode": args.pool_mode,
#     }
#     num_classes = len(train_ds.gloss_dict) + 1  # CTC blank=0

#     model = MultiModalMamba(img_cfg, qgrid_cfg, kp_cfg, num_classes=num_classes, fusion_cfg=fusion).to(device)

#     if dist.is_initialized():
#         model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

#     # ----- loss/opt/amp -----
#     criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)  # zero_infinity helps avoid NaNs from length mismatches
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     scaler = GradScaler(device="cuda", enabled=not args.bf16)
#     amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

#     best_wer = 1e9
#     start = time.time()

#     for epoch in range(args.epochs):
#         if isinstance(train_sampler, DistributedSampler):
#             train_sampler.set_epoch(epoch)

#         stats = train_one_epoch(
#             model=model,
#             criterion=criterion,
#             data_loader=train_loader,
#             optimizer=optimizer,
#             device=device,
#             epoch=epoch,
#             scaler=scaler,
#             amp_dtype=amp_dtype,
#             max_norm=args.max_norm,
#             accum=args.accum,
#         )

#         if is_main():
#             ckpt = {
#                 "epoch": epoch,
#                 "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
#                 "optim": optimizer.state_dict(),
#                 "scaler": scaler.state_dict(),
#                 "args": vars(args),
#             }
#             os.makedirs(args.out_dir, exist_ok=True)
#             torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))

#         wer = evaluate(model, val_loader, device)

#         if is_main():
#             if wer < best_wer:
#                 best_wer = wer
#                 torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
#             print(f"[epoch {epoch}] train_loss={stats['loss']:.4f}  skipped={stats['skipped']}  val_WER={wer:.2f}  best_WER={best_wer:.2f}")

#     if is_main():
#         print(f"Done in {(time.time()-start)/3600:.2f} h. Best WER: {best_wer:.2f}%  → {args.out_dir}")

#     cleanup_dist()


# if __name__ == "__main__":
#     main()