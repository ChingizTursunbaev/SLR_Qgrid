
# train_multimodal.py
import os, time, json
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.models.multi_modal_model import MultiModalMamba
from slr.engine import train_one_epoch, evaluate

def main():
    # --------- paths (EDIT if needed) ----------
    IMAGE_PREFIX = "/nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px"
    QGRID_PREFIX = "/nas/Dataset/Phoenix/Phoenix-2014_cleaned/interpolated_original/Qgrid_npy"
    KP_PATH      = "/home/chingiz/SLR/Mamba_SLR/data/phoenix2014/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl"
    META_DIR     = "data/phoenix2014"
    GLOSS_DICT   = "data/phoenix2014/gloss_dict_normalized.npy"

    OUT_DIR      = "checkpoints/multimodal_run"
    os.makedirs(OUT_DIR, exist_ok=True)

    # --------- hyperparams ----------
    seed       = 1337
    batch_size = 2           # raise if GPU allows
    epochs     = 30
    lr         = 1e-4
    weight_decay = 0.05
    num_workers  = 4
    max_norm     = 1.0       # grad clip
    use_bf16     = False     # set True if your GPU supports bf16 well (A100/H100)

    torch.manual_seed(seed)
    assert torch.cuda.is_available(), "Need a CUDA GPU"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # --------- datasets ----------
    train_ds = MultiModalPhoenixDataset(
        image_prefix=IMAGE_PREFIX, qgrid_prefix=QGRID_PREFIX, kp_path=KP_PATH,
        meta_dir_path=META_DIR, gloss_dict_path=GLOSS_DICT, split="train", transforms=None
    )
    dev_ds = MultiModalPhoenixDataset(
        image_prefix=IMAGE_PREFIX, qgrid_prefix=QGRID_PREFIX, kp_path=KP_PATH,
        meta_dir_path=META_DIR, gloss_dict_path=GLOSS_DICT, split="dev", transforms=None
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=multi_modal_collate_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=multi_modal_collate_fn
    )

    # --------- model ----------
    img_cfg   = {"out_dim": 512}  # matches what worked in your tiny test
    qgrid_cfg = {"out_dim": 512}
    kp_cfg    = {"input_dim": 242, "model_dim": 512}
    fusion    = {"embed_dim": 512, "num_heads": 8, "dropout": 0.1, "max_kv": 512, "pool_mode": "mean"}

    num_classes = len(train_ds.gloss_dict) + 1  # blank=0, your gloss ids start at 1
    model = MultiModalMamba(img_cfg, qgrid_cfg, kp_cfg, num_classes=num_classes, fusion_cfg=fusion).to(device)

    # --------- loss/opt/amp ----------
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=not use_bf16)
    amp_ctx = (autocast(dtype=torch.bfloat16) if use_bf16 else autocast())

    # --------- schedules (optional simple cosine) ----------
    total_steps = len(train_loader) * epochs
    def cosine(step, total, base_lr):  # quick inline LR schedule
        import math
        return base_lr * 0.5 * (1 + math.cos(math.pi * step / total))
    lr_sched = [cosine(s, total_steps-1, lr) for s in range(total_steps)]

    best_wer = 1e9
    start = time.time()

    for epoch in range(epochs):
        stats = train_one_epoch(
            model=model, criterion=criterion, data_loader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, loss_scaler=scaler, amp_autocast=amp_ctx,
            max_norm=max_norm, model_ema=None, log_writer=None,
            start_steps=epoch * len(train_loader), lr_schedule_values=lr_sched, wd_schedule_values=None,
            num_training_steps_per_epoch=len(train_loader), update_freq=1, no_amp=False, bf16=use_bf16
        )
        val_stats = evaluate(
            data_loader=dev_loader, model=model, device=device, amp_autocast=amp_ctx,
            gloss_dict=train_ds.gloss_dict, ds=True, no_amp=False, bf16=use_bf16
        )

        # Save checkpoints
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "train_stats": stats,
            "val_stats": val_stats,
            "config": {
                "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                "fusion": fusion, "img_cfg": img_cfg, "kp_cfg": kp_cfg
            }
        }
        torch.save(ckpt, os.path.join(OUT_DIR, f"epoch{epoch:03d}.pth"))

        wer = float(val_stats.get("wer", 1e9))
        if wer < best_wer:
            best_wer = wer
            torch.save(ckpt, os.path.join(OUT_DIR, "best.pth"))

        # quick log line
        print(f"[epoch {epoch}] train_loss={stats['loss']:.4f}  val_loss={val_stats['loss']:.4f}  val_WER={wer:.2f}  best_WER={best_wer:.2f}")

    dur = (time.time() - start) / 3600
    print(f"Done. Total ~{dur:.2f} h. Best WER: {best_wer:.2f}%  Checkpoints → {OUT_DIR}")

if __name__ == "__main__":
    main()
