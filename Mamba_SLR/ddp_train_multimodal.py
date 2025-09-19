#!/usr/bin/env python
# ddp_train_multimodal.py — stable bf16+CTC, NaN-guards, LR warmup+cosine

import os, math
import argparse
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# datasets / collate
from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
# model
from slr.models.multi_modal_model import MultiModalMamba


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_ddp():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), local_rank
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0


def cleanup_ddp():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def preflight_print(gloss_path: str, shift_labels_by: int, num_classes: int, blank_idx: int):
    if not is_main():
        return
    try:
        import numpy as np
        arr = np.load(gloss_path, allow_pickle=True)
        vals = (arr.item().values() if hasattr(arr, "item") else arr.ravel().tolist())
        ids = [int(v) for v in vals]
        gmin, gmax = (min(ids), max(ids)) if ids else (1, num_classes - 1)
    except Exception:
        gmin, gmax = (1, num_classes - 1)
    print(f"[preflight] gloss_id_range=[{gmin}, {gmax}]  shift_labels_by={shift_labels_by}  num_classes={num_classes}  blank_idx={blank_idx}", flush=True)


def find_1d_lengths(cands: List[torch.Tensor], target_dim: int) -> Optional[torch.Tensor]:
    for t in cands:
        if t is not None and t.dim() == 1 and int(t.max().item()) <= target_dim:
            return t
    return None


def unpack_batch(batch: Any) -> Dict[str, Any]:
    if isinstance(batch, dict):
        return batch

    imgs = qg = kp = labels = None
    one_d = []
    for item in batch:
        if torch.is_tensor(item):
            if item.dim() == 5 and item.is_floating_point():
                imgs = item
            elif item.dim() == 3 and item.is_floating_point():
                if qg is None: qg = item
                else:          kp = item
            elif item.dim() == 2 and item.dtype in (torch.int32, torch.int64, torch.long):
                labels = item
            elif item.dim() == 1 and item.dtype in (torch.int32, torch.int64, torch.long):
                one_d.append(item)

    B = (imgs.size(0) if imgs is not None else labels.size(0))
    dev = (imgs.device if imgs is not None else labels.device)
    T_img = imgs.size(1) if imgs is not None else 0
    T_q   = qg.size(1)   if qg   is not None else 0
    L_lab = labels.size(1) if labels is not None else 0

    image_lengths = find_1d_lengths(one_d, T_img) or torch.full((B,), T_img, dtype=torch.long, device=dev)
    qgrid_lengths = find_1d_lengths(one_d, T_q)   or torch.full((B,), T_q,   dtype=torch.long, device=dev)
    label_lengths = find_1d_lengths(one_d, L_lab) or torch.full((B,), L_lab, dtype=torch.long, device=dev)

    return {
        "images": imgs, "qgrids": qg, "keypoints": kp,
        "image_lengths": image_lengths, "qgrid_lengths": qgrid_lengths,
        "labels": labels, "label_lengths": label_lengths,
    }


def flatten_targets(labels: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:
    outs = []
    for i in range(labels.size(0)):
        L = int(label_lengths[i].item())
        if L > 0: outs.append(labels[i, :L])
    return torch.cat(outs, dim=0) if outs else torch.zeros((0,), device=labels.device, dtype=torch.long)


def valid_ctc_targets(targets: torch.Tensor, num_classes: int, blank_idx: int) -> bool:
    if targets.numel() == 0: return True
    tmin = int(targets.min().item()); tmax = int(targets.max().item())
    return (0 <= tmin) and (tmax < num_classes) and not (targets == blank_idx).any()


def save_ckpt(state: Dict, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    if is_main(): print(f"[ckpt] saved -> {path}", flush=True)


def wrap_frame_encoder_for_5d(model: nn.Module):
    if not hasattr(model, "frame_encoder"): return
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
    if is_main(): print("[adapter] Wrapped frame_encoder to accept 5D (B,T,C,H,W) -> (B,T,D).", flush=True)


def sanitize_(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None: return None
    return torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)


@torch.no_grad()
def materialize_params_with_dummy_forward(model: nn.Module, loader: DataLoader, device: torch.device, use_bf16: bool):
    model.eval()
    try: batch = next(iter(loader))
    except StopIteration: return
    b = unpack_batch(batch)
    images = sanitize_(b["images"]).to(device, non_blocking=True) if b["images"] is not None else None
    qgrids = sanitize_(b["qgrids"]).to(device, non_blocking=True) if b["qgrids"] is not None else None
    keypoints = sanitize_(b["keypoints"]).to(device, non_blocking=True) if b["keypoints"] is not None else None
    qgrid_lengths = b["qgrid_lengths"].to(device, non_blocking=True) if b["qgrid_lengths"] is not None else None
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.cuda.amp.autocast(enabled=False)
    with ctx: _ = model(images, qgrids, keypoints, qgrid_lengths)
    torch.cuda.synchronize(device); model.train()
    if is_main(): print("[ddp] dummy forward ran; parameters materialized.", flush=True)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float):
    # returns LambdaLR that does linear warmup to base_lr then cosine down to min_lr
    min_ratio = max(1e-8, min_lr / max(1e-12, base_lr))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def maybe_freeze_frame_encoder(model: nn.Module, freeze: bool):
    if not hasattr(model, "frame_encoder"): return
    for p in model.frame_encoder.parameters(): p.requires_grad = not freeze


def train(args):
    device, local_rank = setup_ddp()

    # datasets/loaders
    train_set = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix, qgrid_prefix=args.qgrid_prefix, kp_path=args.kp_path,
        meta_dir_path=args.meta_dir, gloss_dict_path=args.gloss_dict, split="train"
    )
    val_set = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix, qgrid_prefix=args.qgrid_prefix, kp_path=args.kp_path,
        meta_dir_path=args.meta_dir, gloss_dict_path=args.gloss_dict, split="dev"
    )
    train_sampler = DistributedSampler(train_set, shuffle=True) if torch.distributed.is_initialized() else None
    val_sampler   = DistributedSampler(val_set,   shuffle=False) if torch.distributed.is_initialized() else None

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler,
        collate_fn=multi_modal_collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=val_sampler,
        collate_fn=multi_modal_collate_fn, drop_last=False,
    )

    # model
    base_model = MultiModalMamba(
        d_model=args.d_model, n_layer=args.n_layer, fusion_embed=args.fusion_embed,
        fusion_heads=args.fusion_heads, num_classes=args.num_classes,
        max_kv=args.max_kv, pool_mode=args.pool_mode,
    ).to(device)

    wrap_frame_encoder_for_5d(base_model)

    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if is_main():
        preflight_print(args.gloss_dict, 0, args.num_classes, 0)
        if args.bf16 and not use_bf16:
            print("[warn] --bf16 requested but not supported; training in fp32.", flush=True)

    # materialize before DDP
    materialize_params_with_dummy_forward(base_model, train_loader, device, use_bf16)

    model = base_model
    if torch.distributed.is_initialized():
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    blank_idx = 0
    ctc_loss = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler setup
    steps_per_epoch = (len(train_loader) + args.accum - 1) // args.accum
    total_updates = steps_per_epoch * args.epochs
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_updates, base_lr=args.lr, min_lr=args.min_lr)

    accum = max(1, int(args.accum))
    global_update = 0
    best_val = float("inf")

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # optional early freeze to stabilize head
        freeze_now = (epoch < args.freeze_frames_epochs)
        if isinstance(model, DDP):
            maybe_freeze_frame_encoder(model.module, freeze_now)
        else:
            maybe_freeze_frame_encoder(model, freeze_now)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        seen = 0

        for it, batch in enumerate(train_loader):
            b = unpack_batch(batch)
            images = sanitize_(b["images"])
            qgrids = sanitize_(b["qgrids"])
            keypoints = sanitize_(b["keypoints"])
            labels = b["labels"]
            label_lengths = b["label_lengths"]
            qgrid_lengths = b["qgrid_lengths"]

            images = images.to(device, non_blocking=True) if images is not None else None
            qgrids = qgrids.to(device, non_blocking=True) if qgrids is not None else None
            keypoints = keypoints.to(device, non_blocking=True) if keypoints is not None else None
            labels = labels.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)
            qgrid_lengths = qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None

            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.cuda.amp.autocast(enabled=False)
            with ctx:
                logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B,T,V)

            if not torch.isfinite(logits).all():
                # keep DDP sync
                (logits.sum() * 0 / accum).backward()
            else:
                B, T, V = logits.shape
                log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # (T,B,V) fp32
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                targets = flatten_targets(labels, label_lengths)

                if not valid_ctc_targets(targets, args.num_classes, blank_idx):
                    (logits.sum() * 0 / accum).backward()
                else:
                    loss = ctc_loss(log_probs, targets, input_lengths, label_lengths)
                    if torch.isfinite(loss):
                        (loss / accum).backward()
                        running += float(loss.item()) * B
                        seen += B
                    else:
                        (logits.sum() * 0 / accum).backward()

            # optimizer step & scheduler step every 'accum' mini-batches
            do_step = (((it + 1) % accum) == 0) or (it + 1 == len(train_loader))
            if do_step:
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update += 1

            if is_main() and (it % args.log_interval == 0):
                cur_lr = optimizer.param_groups[0]["lr"]
                avg = running / max(1, seen)
                print(f"Epoch: [{epoch}]  [{it}/{len(train_loader)}]  lr: {cur_lr:.6f}  loss: {avg:.4f}  accum:{accum}", flush=True)

        # ----- validation -----
        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                b = unpack_batch(batch)
                images = sanitize_(b["images"])
                qgrids = sanitize_(b["qgrids"])
                keypoints = sanitize_(b["keypoints"])
                labels = b["labels"]
                label_lengths = b["label_lengths"]
                qgrid_lengths = b["qgrid_lengths"]

                images = images.to(device, non_blocking=True) if images is not None else None
                qgrids = qgrids.to(device, non_blocking=True) if qgrids is not None else None
                keypoints = keypoints.to(device, non_blocking=True) if keypoints is not None else None
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)
                qgrid_lengths = qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None

                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.cuda.amp.autocast(enabled=False)
                with ctx:
                    logits = model(images, qgrids, keypoints, qgrid_lengths)

                if not torch.isfinite(logits).all(): continue

                log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
                B, T, V = logits.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                targets = flatten_targets(labels, label_lengths)
                if not valid_ctc_targets(targets, args.num_classes, blank_idx): continue
                loss = ctc_loss(log_probs, targets, input_lengths, label_lengths)
                if torch.isfinite(loss):
                    val_loss += float(loss.item()) * B
                    val_seen += B

        if is_main():
            avg_train = running / max(1, seen)
            avg_val   = val_loss / max(1, val_seen)
            print(f"[epoch {epoch}] train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  (accum={accum})", flush=True)

            save_ckpt(
                {"epoch": epoch, "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                 "optimizer": optimizer.state_dict(), "avg_train_loss": avg_train, "avg_val_loss": avg_val,
                 "accum": accum, "lr": optimizer.param_groups[0]["lr"]},
                args.out_dir, "last.pt",
            )
            if avg_val < best_val:
                best_val = avg_val
                save_ckpt(
                    {"epoch": epoch, "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                     "optimizer": optimizer.state_dict(), "avg_train_loss": avg_train, "avg_val_loss": avg_val,
                     "accum": accum, "lr": optimizer.param_groups[0]["lr"]},
                    args.out_dir, "best.pt",
                )

    cleanup_ddp()


def get_args():
    p = argparse.ArgumentParser()

    # data (your defaults)
    p.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner_original/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    p.add_argument("--out_dir",      default="checkpoints/multimodal_ddp")

    # model
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--fusion_embed", type=int, default=512)
    p.add_argument("--fusion_heads", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=1296)
    p.add_argument("--max_kv", type=int, default=1024)
    p.add_argument("--pool_mode", type=str, default="mean", choices=["mean", "max", "vote"])

    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-5, help="final LR for cosine schedule")
    p.add_argument("--warmup_steps", type=int, default=1500, help="linear warmup steps")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    p.add_argument("--clip_grad", type=float, default=1.0, help="max grad-norm; 0 to disable")
    p.add_argument("--freeze_frames_epochs", type=int, default=0, help="freeze frame encoder for N starting epochs")

    # precision
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