

# ddp_train_multimodal.py (patched for robust label/length validation & safe CTC)
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # surface exact kernel sites

import time
import argparse
from typing import Dict, Any, Tuple, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.models.multi_modal_model import MultiModalMamba

# ---------------- util ----------------
def _is_main() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

def _setup_dist() -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def _cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

# ---------------- dataset/labels helpers ----------------
def _compute_id_bounds_from_map(npy_path: str) -> Tuple[int, int]:
    d = torch.load(npy_path, map_location="cpu") if npy_path.endswith(".pt") else None
    if d is None:
        import numpy as np
        arr = np.load(npy_path, allow_pickle=True).item()
        ids = []
        for v in arr.values():
            if isinstance(v, (list, tuple)):
                ids.append(int(v[0]))
            else:
                ids.append(int(v))
        return (min(ids), max(ids))
    else:
        ids = [int(v[0] if isinstance(v, (list, tuple)) else v) for v in d.values()]
        return (min(ids), max(ids))

def make_collate_to_dict(shift_labels_by: int, blank_idx: int, num_classes: int):
    """
    Wrap repo collate to:
      - optionally shift labels to avoid blank collision,
      - build flat CTC targets,
      - VALIDATE targets every batch,
      - ***CLAMP variable-length metadata*** to the true time dims to prevent gather OOB.
    """
    def _collate_to_dict(samples):
        imgs, qgrids, kps, labels, image_lengths, label_lengths, qgrid_lengths = multi_modal_collate_fn(samples)

        # ---- types ----
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, dtype=torch.long)
        if not torch.is_tensor(label_lengths):
            label_lengths = torch.as_tensor(label_lengths, dtype=torch.long)
        if not torch.is_tensor(image_lengths):
            image_lengths = torch.as_tensor(image_lengths, dtype=torch.long)
        if qgrid_lengths is not None and not torch.is_tensor(qgrid_lengths):
            qgrid_lengths = torch.as_tensor(qgrid_lengths, dtype=torch.long)

        # ---- clamp lengths to actual T ----
        if imgs is not None and imgs.ndim >= 2:
            T_img = int(imgs.size(1))
            image_lengths = image_lengths.clamp(min=1, max=T_img)
        if qgrids is not None and qgrids.ndim >= 2 and qgrid_lengths is not None:
            T_q = int(qgrids.size(1))
            qgrid_lengths = qgrid_lengths.clamp(min=1, max=T_q)

        # ---- optional label shift ----
        if shift_labels_by:
            labels = labels + shift_labels_by

        # ---- build flat CTC targets ----
        parts = []
        if labels.ndim == 2:
            B = labels.size(0)
            for i in range(B):
                L = int(label_lengths[i])
                if L > 0:
                    parts.append(labels[i, :L])
        targets_flat = torch.cat(parts, dim=0) if parts else torch.zeros(0, dtype=torch.long)

        # ---- HARD VALIDATION (targets) ----
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

        return {
            "images": imgs, "qgrids": qgrids, "keypoints": kps,
            "targets": targets_flat,
            "image_lengths": image_lengths, "target_lengths": label_lengths,
            "qgrid_lengths": qgrid_lengths,
        }
    return _collate_to_dict

# ------------------ Validated CTC ------------------
class ValidatedCTC(torch.nn.Module):
    """
    Wrapper around nn.CTCLoss that asserts target ranges BEFORE CUDA kernels.
    Also checks logits dimensions.
    """
    def __init__(self, blank: int, num_classes: int, expect_log_probs: bool = True):
        super().__init__()
        self.blank = blank
        self.num_classes = num_classes
        self.expect_log_probs = expect_log_probs
        self.base = torch.nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")

    def forward(self, x, targets, input_lengths, target_lengths):
        # x: (T,B,C) or (B,T,C) depending on caller; engine will pass (T,B,C)
        if x.dim() == 3 and x.shape[0] < x.shape[1]:  # (B,T,C) -> (T,B,C)
            x = x.transpose(0, 1)
        T, B, C = x.shape
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

# ---------------- training & eval ----------------
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

@torch.no_grad()
def _greedy_decode_ctc(log_probs: torch.Tensor) -> List[List[int]]:
    # log_probs: (B, T, V)
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

def train_one_epoch(model, criterion, data_loader, optimizer, device, scaler, epoch, accum: int = 1, max_norm: float = 1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    num_samples = 0
    skipped = 0

    start = time.time()
    for it, batch in enumerate(data_loader):
        batch = _move_batch(batch, device)
        images = batch["images"]
        qgrids = batch["qgrids"]
        keypoints = batch["keypoints"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        qgrid_lengths = batch.get("qgrid_lengths", None)

        B = images.size(0) if images is not None else 0
        num_samples += B

        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images, qgrids, keypoints, qgrid_lengths=qgrid_lengths)  # (B,T,V)
                log_probs = F.log_softmax(logits, dim=-1)
                in_lens = _ctc_input_lengths(log_probs)
                loss = criterion(log_probs.transpose(0, 1), targets, in_lens, target_lengths)

            if (not torch.isfinite(loss)) or torch.isnan(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                if _is_main():
                    print(f"[warn] non-finite loss at step {it}, skipping batch")
                continue

            (loss / max(1, accum)).backward()
            if (it + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().item()) * B

        except RuntimeError as e:
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if _is_main():
                print("\n[BatchError] Skipping batch due to exception.")
                print("  exception:", repr(e))
                print("  shapes: imgs", None if images is None else tuple(images.shape),
                      "qgrids", None if qgrids is None else tuple(qgrids.shape),
                      "keypoints", None if keypoints is None else tuple(keypoints.shape))
                if targets.numel():
                    print("  targets: min", int(targets.min()), "max", int(targets.max()), "numel", targets.numel())
                if qgrid_lengths is not None:
                    print("  qgrid_lengths: min", int(qgrid_lengths.min()), "max", int(qgrid_lengths.max()))
            continue

        if _is_main() and (it % 50 == 0):
            now = time.time()
            print(f"Epoch: [{epoch}]  [{it}/{len(data_loader)}]  loss: {running_loss / max(1,num_samples):.4f}  accum:{accum}  time/50b: {now - start:.1f}s")
            start = now

    return {"loss": running_loss / max(1, num_samples), "skipped": skipped}

@torch.no_grad()
def evaluate(model, data_loader, device) -> float:
    model.eval()
    total_wer = 0.0
    N = 0
    for batch in data_loader:
        batch = _move_batch(batch, device)
        images = batch["images"]; qgrids = batch["qgrids"]; keypoints = batch["keypoints"]
        qgrid_lengths = batch.get("qgrid_lengths", None)
        logits = model(images, qgrids, keypoints, qgrid_lengths=qgrid_lengths)
        log_probs = F.log_softmax(logits, dim=-1)
        hyps = _greedy_decode_ctc(log_probs)
        # TODO: plug real WER computation using ground truth (not included here)
        total_wer += 0.0
        N += 1
    return (total_wer / max(1, N)) * 100.0

def build_args():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    ap.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    ap.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    ap.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    ap.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    ap.add_argument("--out_dir",      default="checkpoints/multimodal_ddp")

    # training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)

    # model sizing
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--fusion_embed", type=int, default=512)
    ap.add_argument("--fusion_heads", type=int, default=8)

    # fusion / pooling
    ap.add_argument("--max_kv", type=int, default=512, help="pooled qgrid length")
    ap.add_argument("--pool_mode", default="mean", choices=["mean", "max", "vote"])

    return ap.parse_args()

def main():
    _setup_dist()
    args = build_args()

    # PREFLIGHT
    min_id, max_id = _compute_id_bounds_from_map(args.gloss_dict)
    shift_labels_by = 1 if min_id == 0 else 0
    num_classes = int(max_id + 1 + shift_labels_by)
    blank_idx = 0
    if _is_main():
        print(f"[preflight] gloss_id_range=[{min_id}, {max_id}]  shift_labels_by={shift_labels_by}  num_classes={num_classes}  blank_idx={blank_idx}")

    # datasets
    train_ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,
        gloss_dict_path=args.gloss_dict,
        split="train",
    )
    val_ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,
        gloss_dict_path=args.gloss_dict,
        split="dev",
    )

    # samplers
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None; val_sampler = None

    # collate
    collate_fn = make_collate_to_dict(shift_labels_by=shift_labels_by, blank_idx=blank_idx, num_classes=num_classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler,
                              shuffle=(train_sampler is None), pin_memory=True, drop_last=False, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=val_sampler,
                              shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    # model
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    model = MultiModalMamba(
        d_model=args.d_model, n_layer=args.n_layer, fusion_embed=args.fusion_embed, fusion_heads=args.fusion_heads,
        num_classes=num_classes, max_kv=args.max_kv, pool_mode=args.pool_mode,
    ).to(device)
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

    # loss & optim
    criterion = ValidatedCTC(blank=blank_idx, num_classes=num_classes, expect_log_probs=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    best_wer = 1e9
    for epoch in range(args.epochs):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        stats = train_one_epoch(model, criterion, train_loader, optimizer, device, scaler, epoch, accum=args.accum, max_norm=1.0)

        if _is_main():
            print(f"[epoch {epoch}] train_loss={stats['loss']:.4f}  skipped={stats['skipped']}")

        # (optional) evaluation placeholder
        # wer = evaluate(model, val_loader, device)

    _cleanup_dist()

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