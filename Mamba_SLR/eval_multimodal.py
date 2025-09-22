#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from jiwer import wer, cer

# repo imports (match your training)
from slr.datasets.multi_modal_datasets import MultiModalPhoenixDataset, multi_modal_collate_fn
from slr.models.multi_modal_model import MultiModalMamba

@torch.no_grad()
def greedy_ctc_decode(logits, blank=0):
    # logits: (B, T, C)
    pred = logits.argmax(dim=-1)            # (B, T)
    decoded = []
    for seq in pred:
        prev = blank
        out = []
        for token in seq.tolist():
            if token != blank and token != prev:
                out.append(token)
            prev = token
        decoded.append(out)
    return decoded

def ids_to_glosses(ids, id2gloss):
    # ids are integers with blank excluded already
    # id2gloss is list/array so id maps directly
    return [id2gloss[i] if 0 <= i < len(id2gloss) else "<UNK>" for i in ids]

def join_glosses(gloss_list):
    # space-separated gloss transcription
    return " ".join(gloss_list)

def main():
    p = argparse.ArgumentParser()
    # data (defaults match your training msg)
    p.add_argument("--image_prefix", default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")

    # model & eval
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt (best.pt or last.pt)")
    p.add_argument("--split", default="val", choices=["val","test"], help="Which split to evaluate")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--blank_idx", type=int, default=0)
    args = p.parse_args()

    # ---- load gloss dict (indexing must match training) ----
    # Expect a numpy array/list where index == label id (0 is blank; 1..N are gloss IDs)
    id2gloss = np.load(args.gloss_dict, allow_pickle=True).tolist()
    assert isinstance(id2gloss, (list, tuple)), "gloss_dict should be a list-like"
    num_classes = len(id2gloss)

    # ---- dataset & loader ----
    ds = MultiModalPhoenixDataset(
        split=args.split,
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir=args.meta_dir,
        gloss_dict=args.gloss_dict,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=multi_modal_collate_fn,
    )

    # ---- build model skeleton & load checkpoint ----
    # Use same hyperparams as your training defaults (edit if you trained with different sizes)
    model = MultiModalMamba(
        d_model=512,
        n_layer=8,
        fusion_embed=512,
        fusion_heads=8,
        num_classes=num_classes,
        max_kv=1024,
        pool_mode="mean",
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)  # support plain state_dict or wrapped
    model.load_state_dict(state, strict=True)

    model = model.to(args.device)
    model.eval()

    # ---- evaluation loop ----
    hyp_strs, ref_strs = [], []

    amp_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    autocast = torch.cuda.amp.autocast if amp_dtype is torch.bfloat16 else torch.cpu.amp.autocast

    for batch in dl:
        images         = batch["images"].to(args.device, non_blocking=True)         # (B, T_img, C, H, W)
        qgrids         = batch["qgrids"].to(args.device, non_blocking=True)         # (B, T_q, 2)
        keypoints      = batch["keypoints"].to(args.device, non_blocking=True)      # (B, T_kp, J, 2) or similar
        qgrid_lengths  = batch["qgrid_lengths"].to(args.device, non_blocking=True)  # (B,)
        labels         = batch["labels"]                                           # list of 1D tensors (target ids)
        # some collate_fn variants return padded tensors and label_lengths
        label_lengths  = batch.get("label_lengths", None)

        with autocast(dtype=amp_dtype):
            logits = model(images, qgrids, keypoints, qgrid_lengths)  # expect (B, T_out, C)

        # Greedy decode
        pred_ids = greedy_ctc_decode(logits, blank=args.blank_idx)  # list[List[int]]

        # Convert both preds and refs to gloss strings
        for i, hyp_ids in enumerate(pred_ids):
            hyp_glosses = ids_to_glosses(hyp_ids, id2gloss)
            hyp_str     = join_glosses(hyp_glosses)
            hyp_strs.append(hyp_str)

            # References: collapse blanks/repeats are not present in labels; labels is gold gloss id sequence
            ref_ids = labels[i].tolist() if torch.is_tensor(labels[i]) else list(labels[i])
            ref_glosses = ids_to_glosses(ref_ids, id2gloss)
            ref_str     = join_glosses(ref_glosses)
            ref_strs.append(ref_str)

    # ---- metrics ----
    # jiwer expects raw strings; we compute WER & CER over full corpus
    corpus_wer = wer(ref_strs, hyp_strs)
    corpus_cer = cer(ref_strs, hyp_strs)

    print(f"[eval] split={args.split}  ckpt={os.path.basename(args.ckpt)}")
    print(f"[eval] WER: {corpus_wer*100:.2f}%   CER: {corpus_cer*100:.2f}%")
    print(f"[eval] samples: {len(ref_strs)}")

if __name__ == "__main__":
    main()
