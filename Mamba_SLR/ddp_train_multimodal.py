# Mamba_SLR/ddp_train_multimodal.py
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # precise error sites

import time
import argparse
from typing import Dict, Any, Tuple

import torch
import torch.distributed as dist
from torch import optim
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from slr.datasets.multi_modal_datasets import (
    MultiModalPhoenixDataset,
    multi_modal_collate_fn,
    load_gloss_maps,
)
from slr.models.multi_modal_model import MultiModalMamba
from slr.engine import train_one_epoch, evaluate


# ----------------------------- ID SPACE HELPERS ------------------------------ #
def _compute_id_bounds_from_map(gloss_dict_path: str) -> Tuple[int, int]:
    """Return (min_id, max_id) from the gloss dict file—no assumptions."""
    gloss_to_id, id_to_gloss = load_gloss_maps(gloss_dict_path)
    ids = []
    if isinstance(gloss_to_id, dict) and len(gloss_to_id):
        try:
            ids.extend(int(v) for v in gloss_to_id.values())
        except Exception:
            pass
    if isinstance(id_to_gloss, dict) and len(id_to_gloss):
        try:
            ids.extend(int(k) for k in id_to_gloss.keys())
        except Exception:
            pass
    if not ids:
        return 1, 1
    return min(ids), max(ids)


def make_collate_to_dict(shift_labels_by: int, blank_idx: int, num_classes: int):
    """
    Wrap repo collate to:
      - optionally shift labels to avoid blank collision,
      - build flat CTC targets,
      - VALIDATE every batch: targets in [0, num_classes-1] and targets != blank_idx.
    """
    def _collate_to_dict(samples):
        imgs, qgrids, kps, labels, image_lengths, label_lengths, qgrid_lengths = multi_modal_collate_fn(samples)

        # Ensure tensor types
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, dtype=torch.long)
        if not torch.is_tensor(label_lengths):
            label_lengths = torch.as_tensor(label_lengths, dtype=torch.long)

        if shift_labels_by:
            labels = labels + shift_labels_by

        # Build flat CTC targets vector
        parts = []
        B = labels.size(0) if labels.ndim == 2 else 0
        for i in range(B):
            L = int(label_lengths[i])
            if L > 0:
                parts.append(labels[i, :L])
        targets_flat = torch.cat(parts, dim=0) if parts else torch.zeros(0, dtype=torch.long)

        # --------- HARD VALIDATION ---------
        if targets_flat.numel():
            tmin = int(targets_flat.min().item())
            tmax = int(targets_flat.max().item())
            if tmin < 0 or tmax >= num_classes:
                raise RuntimeError(
                    f"[LabelRangeError] targets min={tmin}, max={tmax}, num_classes={num_classes}. "
                    f"shift_labels_by={shift_labels_by}, blank_idx={blank_idx}."
                )
            if (targets_flat == blank_idx).any().item():
                raise RuntimeError(
                    f"[BlankCollision] targets contain the blank index ({blank_idx}). "
                    f"shift_labels_by={shift_labels_by}."
                )
        # -----------------------------------

        batch: Dict[str, Any] = {
            "images": imgs,                             # (B, T_img, 3, H, W)
            "qgrids": qgrids,                           # (B, T_q, 242) or None
            "keypoints": kps,                           # (B, T_img, 242) or None
            "targets": targets_flat,                    # (sum(L_i),)
            "target_lengths": label_lengths.long(),     # (B,)
            "qgrid_lengths": (torch.as_tensor(qgrid_lengths, dtype=torch.long)
                              if qgrid_lengths is not None else None),  # (B,) or None
        }
        return batch
    return _collate_to_dict
# ---------------------------------------------------------------------------- #


class SafeCTC(torch.nn.Module):
    """
    Wrapper around nn.CTCLoss that asserts target ranges BEFORE CUDA kernels.
    Works whether engine passes logits or log-probs.
    """
    def __init__(self, blank: int, zero_infinity: bool, num_classes: int, expect_log_probs: bool = True):
        super().__init__()
        self.base = torch.nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)
        self.blank = int(blank)
        self.num_classes = int(num_classes)
        self.expect_log_probs = expect_log_probs

    def forward(self, x, targets, input_lengths, target_lengths):
        # Support (T,B,C) or (B,T,C)
        dims = x.dim()
        if dims == 3 and targets.dim() == 1 and x.size(0) != input_lengths.size(0):
            # assume (T, B, C) → transpose for checks only
            x_bt = x.transpose(0, 1)
        else:
            x_bt = x

        C = x_bt.size(-1)
        if C != self.num_classes:
            raise RuntimeError(f"[LogitsDimError] logits classes={C} but configured num_classes={self.num_classes}")

        if targets.numel():
            tmin = int(targets.min().item())
            tmax = int(targets.max().item())
            if tmin < 0 or tmax >= C:
                raise RuntimeError(
                    f"[CTC/TargetRangeError] targets min={tmin}, max={tmax}, classes(C)={C}. blank={self.blank}"
                )
            if (targets == self.blank).any().item():
                raise RuntimeError(f"[CTC/BlankCollision] targets contain blank={self.blank}")

        x_lp = x if self.expect_log_probs else torch.log_softmax(x, dim=-1)
        return self.base(x_lp, targets, input_lengths, target_lengths)


def parse_args():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--image_prefix", required=True)
    ap.add_argument("--qgrid_prefix", required=True)
    ap.add_argument("--kp_path", required=True)
    ap.add_argument("--meta_dir", required=True)       # passthrough → dataset meta_dir_path
    ap.add_argument("--gloss_dict", required=True)     # gloss map file path
    ap.add_argument("--out_dir", default="checkpoints/multimodal_ddp")

    # training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=2, help="per-GPU batch size")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_norm", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast")

    # model sizing
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--fusion_embed", type=int, default=512)
    ap.add_argument("--fusion_heads", type=int, default=8)

    # fusion / pooling
    ap.add_argument("--max_kv", type=int, default=512, help="pooled qgrid length")
    ap.add_argument("--pool_mode", default="mean", choices=["mean", "max", "vote"])

    return ap.parse_args()


def _is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_dist():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    setup_dist()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # seeding
    base_seed = 1337
    torch.manual_seed(base_seed + (dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0))

    # ----------------------- PREFLIGHT: LABEL ID SPACE ----------------------- #
    min_id, max_id = _compute_id_bounds_from_map(args.gloss_dict)
    # If 0 is a real label id, shift +1 so 0 is reserved for blank.
    shift_labels_by = 1 if min_id == 0 else 0
    num_classes = int(max_id + 1 + shift_labels_by)
    blank_idx = 0
    if _is_main():
        print(f"[preflight] gloss_id_range=[{min_id}, {max_id}]  "
              f"shift_labels_by={shift_labels_by}  num_classes={num_classes}  blank_idx={blank_idx}")
    # ------------------------------------------------------------------------ #

    # ---- datasets ----
    train_ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,           # correct kwarg
        gloss_dict_path=args.gloss_dict,       # correct kwarg
        split="train",
    )
    val_ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,           # correct kwarg
        gloss_dict_path=args.gloss_dict,       # correct kwarg
        split="dev",
    )

    # samplers
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    # dataloaders with deterministic collate + validation
    collate_fn = make_collate_to_dict(shift_labels_by=shift_labels_by, blank_idx=blank_idx, num_classes=num_classes)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=val_sampler,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # ---- model ----
    img_cfg = {"in_chans": 3, "embed_dim": args.d_model, "n_layer": args.n_layer}
    qgrid_cfg = {"in_dim": 242, "embed_dim": args.d_model, "n_layer": args.n_layer}
    kp_cfg = {"in_dim": 242, "embed_dim": args.d_model, "n_layer": args.n_layer}
    fusion = {
        "embed_dim": args.fusion_embed,
        "num_heads": args.fusion_heads,
        "dropout": 0.1,
        "max_kv": args.max_kv,
        "pool_mode": args.pool_mode,
    }

    model = MultiModalMamba(img_cfg, qgrid_cfg, kp_cfg, num_classes=num_classes, fusion_cfg=fusion).to(device)

    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ---- loss/opt/amp ----
    criterion = SafeCTC(blank=blank_idx, zero_infinity=True, num_classes=num_classes, expect_log_probs=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device="cuda", enabled=not args.bf16)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    best_wer = 1e9
    start = time.time()

    for epoch in range(args.epochs):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            amp_dtype=amp_dtype,
            max_norm=args.max_norm,
            accum=args.accum,
        )

        # save checkpoint
        if _is_main():
            ckpt = {
                "epoch": epoch,
                "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                "optim": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "args": {
                    **vars(args),
                    "num_classes": num_classes,
                    "shift_labels_by": shift_labels_by,
                    "blank_idx": blank_idx,
                    "gloss_id_range": (int(min_id), int(max_id)),
                },
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"epoch_{epoch:03d}.pth"))

        # validation
        wer = evaluate(model, val_loader, device)

        if _is_main():
            if wer < best_wer:
                best_wer = wer
                ckpt_best = {
                    "epoch": epoch,
                    "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                    "optim": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": {
                        **vars(args),
                        "num_classes": num_classes,
                        "shift_labels_by": shift_labels_by,
                        "blank_idx": blank_idx,
                        "gloss_id_range": (int(min_id), int(max_id)),
                    },
                }
                torch.save(ckpt_best, os.path.join(args.out_dir, "best.pth"))
            print(f"[epoch {epoch}] train_loss={stats['loss']:.4f}  skipped={stats['skipped']}  "
                  f"val_WER={wer:.2f}  best_WER={best_wer:.2f}")

    if _is_main():
        print(f"Done in {(time.time()-start)/3600:.2f} h. Best WER: {best_wer:.2f}%  → {args.out_dir}")

    cleanup_dist()


if __name__ == "__main__":
    main()





















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