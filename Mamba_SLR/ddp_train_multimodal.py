# ddp_train_multimodal.py
import os, time
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast
from torch import optim

from slr.datasets.multi_modal_datasets import (
    MultiModalPhoenixDataset,
    multi_modal_collate_fn,
)
from slr.models.multi_modal_model import MultiModalMamba
from slr.engine import train_one_epoch, evaluate


def parse_args():
    ap = argparse.ArgumentParser()
    # paths (set your own defaults if needed)
    ap.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    ap.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    ap.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    ap.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    ap.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    ap.add_argument("--out_dir",      default="checkpoints/multimodal_ddp")

    # training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=1, help="per-GPU batch size")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_norm", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--bf16", action="store_true", help="use bfloat16 autocast on A100+ (recommended)")

    # model sizing
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--fusion_embed", type=int, default=512)
    ap.add_argument("--fusion_heads", type=int, default=8)

    # fusion / pooling
    ap.add_argument("--max_kv", type=int, default=512, help="pooled qgrid length")
    ap.add_argument("--pool_mode", default="mean", choices=["mean", "max", "vote"])
    return ap.parse_args()


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_dist():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Safer NCCL defaults
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29531"))

    # Enable TF32 for extra stability/speed on A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    setup_dist()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # Different ranks get different seeds
    base_seed = 1337
    if dist.is_available() and dist.is_initialized():
        torch.manual_seed(base_seed + dist.get_rank())
    else:
        torch.manual_seed(base_seed)

    # ----- data -----
    train_ds = MultiModalPhoenixDataset(
    image_prefix=args.image_prefix,
    qgrid_prefix=args.qgrid_prefix,
    kp_path=args.kp_path,
    meta_dir_path=args.meta_dir,         # <-- renamed kwarg
    gloss_dict_path=args.gloss_dict,     # <-- renamed kwarg
    split="train",
    transforms=None,
    )
    
    val_ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,
        gloss_dict_path=args.gloss_dict,
        split="dev",
        transforms=None,
    )

    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=True,
        drop_last=False,
        collate_fn=multi_modal_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=val_sampler,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=multi_modal_collate_fn,
    )

    # ----- model -----
    img_cfg = {
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "fused_add_norm": False,  # Triton fused LN off (stability)
        "rms_norm": False,
    }
    qgrid_cfg = {"out_dim": args.fusion_embed}
    kp_cfg = {"input_dim": 242, "model_dim": args.d_model}
    fusion = {
        "embed_dim": args.fusion_embed,
        "num_heads": args.fusion_heads,
        "dropout": 0.1,
        "max_kv": args.max_kv,
        "pool_mode": args.pool_mode,
    }
    num_classes = len(train_ds.gloss_dict) + 1  # CTC blank=0

    model = MultiModalMamba(img_cfg, qgrid_cfg, kp_cfg, num_classes=num_classes, fusion_cfg=fusion).to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ----- loss/opt/amp -----
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)  # zero_infinity helps avoid NaNs from length mismatches
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

        if is_main():
            ckpt = {
                "epoch": epoch,
                "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                "optim": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
            }
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))

        wer = evaluate(model, val_loader, device)

        if is_main():
            if wer < best_wer:
                best_wer = wer
                torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"[epoch {epoch}] train_loss={stats['loss']:.4f}  skipped={stats['skipped']}  val_WER={wer:.2f}  best_WER={best_wer:.2f}")

    if is_main():
        print(f"Done in {(time.time()-start)/3600:.2f} h. Best WER: {best_wer:.2f}%  → {args.out_dir}")

    cleanup_dist()


if __name__ == "__main__":
    main()