# ddp_train_multimodal.py
import os, time
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast
from torch import optim

from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.models.multi_modal_model import MultiModalMamba
from slr.engine import train_one_epoch, evaluate

def parse_args():
    ap = argparse.ArgumentParser()
    # paths
    ap.add_argument("--image_prefix", default="/nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px")
    ap.add_argument("--qgrid_prefix", default="/nas/Dataset/Phoenix/Phoenix-2014_cleaned/interpolated_original/Qgrid_npy")
    ap.add_argument("--kp_path",      default="/home/chingiz/SLR/Mamba_SLR/data/phoenix2014/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    ap.add_argument("--meta_dir",     default="data/phoenix2014")
    ap.add_argument("--gloss_dict",   default="data/phoenix2014/gloss_dict_normalized.npy")
    ap.add_argument("--out_dir",      default="checkpoints/multimodal_ddp")
    # training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=1, help="per-GPU batch size")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_norm", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (update_freq)")
    ap.add_argument("--bf16", action="store_true", help="use bf16 autocast")
    # fusion / pooling
    ap.add_argument("--max_kv", type=int, default=512)
    ap.add_argument("--pool_mode", default="mean", choices=["mean","max"])
    # mamba cfgs could be expanded as args if needed
    return ap.parse_args()

def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def setup_dist():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def cosine(step, total, base_lr):
    import math
    return base_lr * 0.5 * (1 + math.cos(math.pi * step / max(1, total)))

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    setup_dist()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1337 + (dist.get_rank() if dist.is_initialized() else 0))

    # ----- datasets -----
    train_ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix, qgrid_prefix=args.qgrid_prefix, kp_path=args.kp_path,
        meta_dir_path=args.meta_dir, gloss_dict_path=args.gloss_dict, split="train", transforms=None
    )
    dev_ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix, qgrid_prefix=args.qgrid_prefix, kp_path=args.kp_path,
        meta_dir_path=args.meta_dir, gloss_dict_path=args.gloss_dict, split="dev", transforms=None
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False) if dist.is_initialized() else None
    dev_sampler   = DistributedSampler(dev_ds, shuffle=False, drop_last=False) if dist.is_initialized() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=multi_modal_collate_fn
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=dev_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=multi_modal_collate_fn
    )

    # ----- model -----
    img_cfg   = {"out_dim": 512}
    qgrid_cfg = {"out_dim": 512}
    kp_cfg    = {"input_dim": 242, "model_dim": 512}
    fusion    = {"embed_dim": 512, "num_heads": 8, "dropout": 0.1, "max_kv": args.max_kv, "pool_mode": args.pool_mode}

    num_classes = len(train_ds.gloss_dict) + 1  # blank=0
    model = MultiModalMamba(img_cfg, qgrid_cfg, kp_cfg, num_classes=num_classes, fusion_cfg=fusion).to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ----- loss/opt/amp -----
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device="cuda", enabled=not args.bf16)
    amp_ctx = autocast(device_type="cuda", dtype=(torch.bfloat16 if args.bf16 else torch.float16))

    # ----- schedule -----
    total_micro_steps = len(train_loader) * args.epochs
    lr_sched = [cosine(s, total_micro_steps - 1, args.lr) for s in range(total_micro_steps)]

    best_wer = float("inf")
    start = time.time()

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        stats = train_one_epoch(
            model=model, criterion=criterion, data_loader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, loss_scaler=scaler, amp_autocast=amp_ctx,
            max_norm=args.max_norm, model_ema=None, log_writer=None,
            start_steps=epoch * len(train_loader), lr_schedule_values=lr_sched, wd_schedule_values=None,
            num_training_steps_per_epoch=len(train_loader), update_freq=args.accum, no_amp=False, bf16=args.bf16
        )

        val_stats = evaluate(
            data_loader=dev_loader, model=model, device=device, amp_autocast=amp_ctx,
            gloss_dict=train_ds.gloss_dict, ds=True, no_amp=False, bf16=args.bf16
        )

        if is_main():
            ckpt = {
                "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "train_stats": stats,
                "val_stats": val_stats,
                "config": {
                    "batch_size_per_gpu": args.batch_size,
                    "world_size": (dist.get_world_size() if dist.is_initialized() else 1),
                    "lr": args.lr, "weight_decay": args.weight_decay,
                    "fusion": fusion, "img_cfg": img_cfg, "kp_cfg": kp_cfg
                }
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"epoch{epoch:03d}.pth"))

            wer = float(val_stats.get("wer", 1e9))
            if wer < best_wer:
                best_wer = wer
                torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))

            print(f"[epoch {epoch}] train_loss={stats['loss']:.4f}  val_loss={val_stats['loss']:.4f}  val_WER={wer:.2f}  best_WER={best_wer:.2f}")

    dur = (time.time() - start) / 3600
    if is_main():
        print(f"Done. ~{dur:.2f} h. Best WER: {best_wer:.2f}%  → {args.out_dir}")

    cleanup_dist()

if __name__ == "__main__":
    main()
