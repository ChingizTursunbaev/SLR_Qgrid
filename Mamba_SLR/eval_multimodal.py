#!/usr/bin/env python
# eval_multimodal.py — evaluation that matches ddp_train_multimodal packing/wrapping
import argparse, warnings
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# dataset / model
from slr.datasets.multi_modal_datasets import (
    MultiModalPhoenixDataset,
    multi_modal_collate_fn,
)
from slr.models.multi_modal_model import MultiModalMamba


# -------------------------------
# args – keep your same defaults
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pt")
    p.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
    p.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
    p.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
    p.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
    p.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
    p.add_argument("--split", choices=["train", "val", "test", "dev"], default="test")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--bf16", action="store_true", help="Evaluate under bfloat16 autocast when supported")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pool_mode", type=str, default="mean")
    p.add_argument("--max_kv", type=int, default=2048)
    p.add_argument("--force_num_classes", type=int, default=None)
    return p.parse_args()


# -------------------------------
# frame-encoder 5D wrapper (as in training)
# -------------------------------
def wrap_frame_encoder_for_5d(model: nn.Module, verbose: bool=True):
    if not hasattr(model, "frame_encoder"):
        return
    fe = model.frame_encoder
    if getattr(fe, "_wrapped_for_5d", False):
        return
    orig_forward = fe.forward

    def adapter(x: torch.Tensor, *args, **kwargs):
        # accept (B*T,C,H,W) or (B,T,C,H,W)
        if x.dim() == 4:
            return orig_forward(x, *args, **kwargs)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            y = orig_forward(x.reshape(B * T, C, H, W), *args, **kwargs)
            # (B*T,D) OR (B*T,P,D)
            if y.dim() == 2:
                return y.view(B, T, -1)
            if y.dim() == 3:
                return y.mean(dim=1).view(B, T, -1)
            raise RuntimeError(f"[adapter] unsupported frame_encoder output {tuple(y.shape)}")
        raise RuntimeError(f"[adapter] frame_encoder input must be 4D/5D, got {tuple(x.shape)}")

    fe.forward = adapter
    fe._wrapped_for_5d = True
    if verbose:
        print("[patch] FrameEncoder adapted for 5D (B,T,C,H,W).", flush=True)


# -------------------------------
# id2gloss utilities
# -------------------------------
def coerce_id_maps(gloss_dict_path: str):
    obj = np.load(gloss_dict_path, allow_pickle=True)
    try:
        obj = obj.item()
    except Exception:
        pass

    gloss2id = {}
    id2gloss = {}

    if isinstance(obj, dict):
        k0 = next(iter(obj.keys()))
        v0 = obj[k0]
        if isinstance(k0, str) and isinstance(v0, (int, np.integer)):
            gloss2id = {str(k): int(v) for k, v in obj.items()}
            id2gloss = {int(v): str(k) for k, v in obj.items()}
        elif isinstance(k0, (int, np.integer)) and isinstance(v0, str):
            id2gloss = {int(k): str(v) for k, v in obj.items()}
            gloss2id = {v: k for k, v in id2gloss.items()}
        else:
            for k, v in obj.items():
                try:
                    ik = int(k); sv = str(v)
                    id2gloss[ik] = sv
                except Exception:
                    gloss2id[str(k)] = int(v)
            if not id2gloss and gloss2id:
                id2gloss = {i: g for g, i in gloss2id.items()}
            if not gloss2id and id2gloss:
                gloss2id = {g: i for i, g in id2gloss.items()}
    else:
        arr = list(obj)
        id2gloss = {i: str(g) for i, g in enumerate(arr)}
        gloss2id = {g: i for i, g in id2gloss.items()}

    if 0 not in id2gloss:
        id2gloss[0] = "<blank>"
    if "<blank>" not in gloss2id:
        gloss2id["<blank>"] = 0
    return gloss2id, id2gloss


def ensure_blank_zero_and_shift_if_needed(labels_sample: torch.Tensor,
                                          id2gloss: Dict[int,str]) -> Dict[int,str]:
    if labels_sample.numel() == 0:
        return id2gloss
    zero_frac = float((labels_sample == 0).float().mean().item())
    if zero_frac > 0.05 and id2gloss.get(0, "<blank>") != "<blank>":
        shifted = {0: "<blank>"}
        for k, v in id2gloss.items():
            if k == 0: continue
            shifted[k + 1] = v
        return shifted
    return id2gloss


def edit_distance(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        ai = a[i-1]
        for j in range(1, n+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (ai != b[j-1]))
    return dp[m][n]


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             id2gloss: Dict[int,str],
             device: torch.device,
             use_bf16: bool=False,
             force_blank_id: Optional[int]=None):

    model.eval()

    # ----- warm-up to detect blank id -----
    blank_id = 0 if force_blank_id is None else int(force_blank_id)
    needs_auto_blank = force_blank_id is None

    warm_batches = 8
    checked = 0
    if needs_auto_blank:
        vocab_size = max(id2gloss.keys()) + 1 if id2gloss else 1
        freq = torch.zeros(vocab_size, dtype=torch.long)

        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if (use_bf16 and device.type=="cuda" and torch.cuda.is_bf16_supported())
                        else torch.amp.autocast("cuda", enabled=False))
        for b_idx, batch in enumerate(loader):
            if b_idx >= warm_batches: break
            images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths = batch
            images = images.to(device, non_blocking=True)
            qgrids = qgrids.to(device, non_blocking=True) if qgrids is not None else None
            keypoints = keypoints.to(device, non_blocking=True) if keypoints is not None else None
            qgrid_lengths = qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None
            with autocast_ctx:
                logits = model(images, qgrids, keypoints, qgrid_lengths)
            pred = logits.argmax(-1)  # (B,T)
            vals, counts = pred.unique(return_counts=True)
            freq[vals.cpu()] += counts.cpu()
            checked += pred.numel()

        if checked > 0:
            blank_id = int(freq.argmax().item())
            total = int(freq.sum().item())
            top = int(freq.max().item())
            top_share = 100.0 * top / max(1, total)
            if top_share < 50.0:
                blank_id = 0
            print(f"[auto] detected blank_id={blank_id} (top-share ~{top_share:.2f}%)", flush=True)
        else:
            print("[auto] could not sample warm-up batches; defaulting blank_id=0", flush=True)

    # ----- full pass -----
    def do_pass(dtype_bf16: bool):
        total_w_edits = total_w_tok = 0
        total_c_edits = total_c_tok = 0
        N = 0
        total_logits = 0
        blank_hits = 0

        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if (dtype_bf16 and device.type=="cuda" and torch.cuda.is_bf16_supported())
                        else torch.amp.autocast("cuda", enabled=False))
        for batch in loader:
            images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths = batch
            images = images.to(device, non_blocking=True)
            qgrids = qgrids.to(device, non_blocking=True) if qgrids is not None else None
            keypoints = keypoints.to(device, non_blocking=True) if keypoints is not None else None
            qgrid_lengths = qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None

            with autocast_ctx:
                logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B,T,V)

            pred = logits.argmax(-1).cpu()  # (B,T)
            blank_hits += int((pred == blank_id).sum().item())
            total_logits += int(pred.numel())

            labels = labels.cpu()
            label_lengths = label_lengths.cpu()

            B, _T = pred.shape
            for b in range(B):
                # greedy collapse using detected blank_id
                seq = []
                prev = None
                for t in pred[b].tolist():
                    if t == blank_id:
                        prev = t; continue
                    if prev is not None and t == prev:
                        continue
                    seq.append(t); prev = t
                hyp = " ".join(id2gloss.get(int(t), "<UNK>") for t in seq)

                L = int(label_lengths[b].item())
                ref_ids = labels[b, :L].tolist() if L>0 else []
                # strip only the detected blank index
                ref_ids = [int(t) for t in ref_ids if int(t) != blank_id]
                ref = " ".join(id2gloss.get(int(t), "<UNK>") for t in ref_ids)

                # WER
                hw, rw = hyp.split(), ref.split()
                m, n = len(hw), len(rw)
                dp = [[0]*(n+1) for _ in range(m+1)]
                for i in range(m+1): dp[i][0] = i
                for j in range(n+1): dp[0][j] = j
                for i in range(1, m+1):
                    for j in range(1, n+1):
                        dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (hw[i-1] != rw[j-1]))
                total_w_edits += dp[m][n]
                total_w_tok   += max(1, n)

                # CER
                hc, rc = list(hyp), list(ref)
                mc, nc = len(hc), len(rc)
                dp2 = [[0]*(nc+1) for _ in range(mc+1)]
                for i in range(mc+1): dp2[i][0] = i
                for j in range(n+1): dp2[0][j] = j
                for i in range(1, mc+1):
                    for j in range(1, nc+1):
                        dp2[i][j] = min(dp2[i-1][j]+1, dp2[i][j-1]+1, dp2[i-1][j-1] + (hc[i-1] != rc[j-1]))
                total_c_edits += dp2[mc][nc]
                total_c_tok   += max(1, nc)

                N += 1

        blank_ratio = 100.0 * blank_hits / max(1, total_logits)
        WER = 100.0 * total_w_edits / max(1, total_w_tok)
        CER = 100.0 * total_c_edits / max(1, total_c_tok)
        return WER, CER, N, blank_ratio

    # first pass (requested precision)
    W1, C1, N1, BR1 = do_pass(use_bf16)
    print(f"[debug] pass-1 blank@argmax (id={blank_id}): {BR1:.2f}%")

    # auto-retry in fp32 if bf16 collapsed
    if use_bf16 and BR1 > 95.0:
        print("[debug] high blank ratio under bf16 — retrying in fp32 once.")
        W2, C2, N2, BR2 = do_pass(False)
        print(f"[debug] pass-2 (fp32) blank@argmax (id={blank_id}): {BR2:.2f}%")
        return W2, C2, N2, blank_id, BR2

    return W1, C1, N1, blank_id, BR1


def main():
    args = parse_args()
    warnings.filterwarnings("once", category=FutureWarning)

    device = torch.device(args.device)
    print(f"[env] device={device}  bf16={args.bf16}", flush=True)

    # ---- dataset (use same collate as training) ----
    ds = MultiModalPhoenixDataset(
        image_prefix=args.image_prefix,
        qgrid_prefix=args.qgrid_prefix,
        kp_path=args.kp_path,
        meta_dir_path=args.meta_dir,
        gloss_dict_path=args.gloss_dict,
        split={"val":"dev","dev":"dev"}.get(args.split, args.split),
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=multi_modal_collate_fn,
    )

    # ---- vocab / classes ----
    try:
        gloss2id = getattr(ds, "gloss_dict")
        id2gloss = getattr(ds, "id_to_gloss")
        if not isinstance(gloss2id, dict) or not isinstance(id2gloss, dict):
            raise RuntimeError("ds maps missing")
    except Exception:
        gloss2id, id2gloss = coerce_id_maps(args.gloss_dict)

    # quick sample to check ids; keep mapping as-is
    try:
        tmp_batch = next(iter(dl))
        lbls = tmp_batch[3]          # labels (B,L)
        lbl_lens = tmp_batch[5]      # label_lengths
        sample = []
        for i in range(min(8, lbls.size(0))):
            L = int(lbl_lens[i].item())
            if L > 0:
                sample.append(lbls[i, :L])
        sample = torch.cat(sample, dim=0) if sample else torch.zeros(0, dtype=torch.long)
    except Exception:
        sample = torch.zeros(0, dtype=torch.long)

    id2gloss = ensure_blank_zero_and_shift_if_needed(sample, id2gloss)

    if 0 not in id2gloss:
        id2gloss[0] = "<blank>"
    vocab_max = max(id2gloss.keys()) if id2gloss else 0
    num_classes = (args.force_num_classes or (vocab_max + 1))
    if args.force_num_classes:
        print(f"[head] num_classes={num_classes} (forced)")
    else:
        print(f"[head] num_classes={num_classes} (from gloss_dict + blank)")

    # ---- model ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state") if isinstance(ckpt, dict) else ckpt
    if sd is None:
        sd = ckpt

    # infer dims from checkpoint
    fe = sd["img_proj.weight"].shape[0]      # fusion_embed
    dm = sd["img_proj.weight"].shape[1]      # d_model
    fusion_heads = 8 if fe % 8 == 0 else 4

    # infer n_layer from keys if present
    n_layer = None
    try:
        layer_ids = []
        for k in sd.keys():
            if "layers." in k:
                try:
                    layer_ids.append(int(k.split("layers.")[1].split(".")[0]))
                except Exception:
                    pass
        if layer_ids:
            n_layer = max(layer_ids) + 1
            print(f"[arch] n_layer inferred from ckpt: {n_layer}")
    except Exception:
        pass

    print(f"[arch] inferred from ckpt -> d_model={dm}, fusion_embed={fe}, fusion_heads={fusion_heads}, num_classes={sd['classifier.weight'].shape[0]}")

    model = MultiModalMamba(
        d_model=dm,
        n_layer=(n_layer or 12),
        fusion_embed=fe,
        fusion_heads=fusion_heads,
        num_classes=num_classes,   # keep dataset-derived classes (should match ckpt)
        max_kv=args.max_kv,
        pool_mode=args.pool_mode,
    ).to(device)

    wrap_frame_encoder_for_5d(model, verbose=True)

    # materialize with a tiny forward before load
    try:
        b = next(iter(dl))
        images, qgrids, keypoints = b[0], b[1], b[2]
        qgrid_lengths = b[6]
        with torch.no_grad():
            _ = model(
                images.to(device, non_blocking=True),
                qgrids.to(device, non_blocking=True) if qgrids is not None else None,
                keypoints.to(device, non_blocking=True) if keypoints is not None else None,
                qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None,
            )
    except Exception:
        pass

    print(f"[ckpt] loading: {args.ckpt}")
    load = model.load_state_dict(sd, strict=True)
    unexpected = list(getattr(load, "unexpected_keys", []))
    missing    = list(getattr(load, "missing_keys", []))
    print(f"[ckpt] unexpected keys: {len(unexpected)}")
    print(f"[ckpt] missing keys: {len(missing)}")

    use_bf16 = args.bf16 and (device.type == "cuda") and torch.cuda.is_bf16_supported()
    WER, CER, N, blank_id, blank_ratio = evaluate(model, dl, id2gloss, device, use_bf16=use_bf16)
    print(f"[RESULT] N={N}  WER={WER:.2f}%  CER={CER:.2f}%  (blank_id={blank_id}, blank@argmax={blank_ratio:.2f}%)")


if __name__ == "__main__":
    main()



































# #!/usr/bin/env python
# # eval_multimodal.py — evaluation that matches ddp_train_multimodal packing/wrapping
# import argparse, warnings
# from typing import Dict, Optional, List, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

# # dataset / model
# from slr.datasets.multi_modal_datasets import (
#     MultiModalPhoenixDataset,
#     multi_modal_collate_fn,
# )
# from slr.models.multi_modal_model import MultiModalMamba


# # -------------------------------
# # args – keep your same defaults
# # -------------------------------
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pt")
#     p.add_argument("--image_prefix", required=False, default="/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px")
#     p.add_argument("--qgrid_prefix", required=False, default="/shared/home/xvoice/Chingiz/datasets/Qgrid_npy")
#     p.add_argument("--kp_path",      required=False, default="/shared/home/xvoice/Chingiz/datasets/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl")
#     p.add_argument("--meta_dir",     required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014")
#     p.add_argument("--gloss_dict",   required=False, default="/shared/home/xvoice/Chingiz/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy")
#     p.add_argument("--split", choices=["train", "val", "test", "dev"], default="test")

#     p.add_argument("--batch_size", type=int, default=4)
#     p.add_argument("--num_workers", type=int, default=2)
#     p.add_argument("--bf16", action="store_true", help="Evaluate under bfloat16 autocast when supported")
#     p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     p.add_argument("--pool_mode", type=str, default="mean")
#     p.add_argument("--max_kv", type=int, default=2048)
#     p.add_argument("--force_num_classes", type=int, default=None)
#     return p.parse_args()


# # -------------------------------
# # frame-encoder 5D wrapper (as in training)
# # -------------------------------
# def wrap_frame_encoder_for_5d(model: nn.Module, verbose: bool=True):
#     if not hasattr(model, "frame_encoder"):
#         return
#     fe = model.frame_encoder
#     if getattr(fe, "_wrapped_for_5d", False):
#         return
#     orig_forward = fe.forward

#     def adapter(x: torch.Tensor, *args, **kwargs):
#         # accept (B*T,C,H,W) or (B,T,C,H,W)
#         if x.dim() == 4:
#             return orig_forward(x, *args, **kwargs)
#         if x.dim() == 5:
#             B, T, C, H, W = x.shape
#             y = orig_forward(x.reshape(B * T, C, H, W), *args, **kwargs)
#             # (B*T,D) OR (B*T,P,D)
#             if y.dim() == 2:
#                 return y.view(B, T, -1)
#             if y.dim() == 3:
#                 return y.mean(dim=1).view(B, T, -1)
#             raise RuntimeError(f"[adapter] unsupported frame_encoder output {tuple(y.shape)}")
#         raise RuntimeError(f"[adapter] frame_encoder input must be 4D/5D, got {tuple(x.shape)}")

#     fe.forward = adapter
#     fe._wrapped_for_5d = True
#     if verbose:
#         print("[patch] FrameEncoder adapted for 5D (B,T,C,H,W).", flush=True)


# # -------------------------------
# # id2gloss utilities
# # -------------------------------
# def coerce_id_maps(gloss_dict_path: str) -> Tuple[Dict[str,int], Dict[int,str]]:
#     obj = np.load(gloss_dict_path, allow_pickle=True)
#     try:
#         obj = obj.item()
#     except Exception:
#         pass

#     gloss2id: Dict[str,int] = {}
#     id2gloss: Dict[int,str] = {}

#     if isinstance(obj, dict):
#         k0 = next(iter(obj.keys()))
#         v0 = obj[k0]
#         if isinstance(k0, str) and isinstance(v0, (int, np.integer)):
#             gloss2id = {str(k): int(v) for k, v in obj.items()}
#             id2gloss = {int(v): str(k) for k, v in obj.items()}
#         elif isinstance(k0, (int, np.integer)) and isinstance(v0, str):
#             id2gloss = {int(k): str(v) for k, v in obj.items()}
#             gloss2id = {v: k for k, v in id2gloss.items()}
#         else:
#             for k, v in obj.items():
#                 try:
#                     ik = int(k); sv = str(v)
#                     id2gloss[ik] = sv
#                 except Exception:
#                     gloss2id[str(k)] = int(v)
#             if not id2gloss and gloss2id:
#                 id2gloss = {i: g for g, i in gloss2id.items()}
#             if not gloss2id and id2gloss:
#                 gloss2id = {g: i for i, g in id2gloss.items()}
#     else:
#         arr = list(obj)
#         id2gloss = {i: str(g) for i, g in enumerate(arr)}
#         gloss2id = {g: i for i, g in id2gloss.items()}

#     if 0 not in id2gloss:
#         id2gloss[0] = "<blank>"
#     if "<blank>" not in gloss2id:
#         gloss2id["<blank>"] = 0
#     return gloss2id, id2gloss


# def ensure_blank_zero_and_shift_if_needed(labels_sample: torch.Tensor,
#                                           id2gloss: Dict[int,str]) -> Dict[int,str]:
#     if labels_sample.numel() == 0:
#         return id2gloss
#     zero_frac = float((labels_sample == 0).float().mean().item())
#     if zero_frac > 0.05 and id2gloss.get(0, "<blank>") != "<blank>":
#         shifted = {0: "<blank>"}
#         for k, v in id2gloss.items():
#             if k == 0: continue
#             shifted[k + 1] = v
#         return shifted
#     return id2gloss


# # -------------------------------
# # quick greedy collapse
# # -------------------------------
# def edit_distance(a: List[str], b: List[str]) -> int:
#     m, n = len(a), len(b)
#     dp = [[0]*(n+1) for _ in range(m+1)]
#     for i in range(m+1): dp[i][0] = i
#     for j in range(n+1): dp[0][j] = j
#     for i in range(1, m+1):
#         ai = a[i-1]
#         for j in range(1, n+1):
#             dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (ai != b[j-1]))
#     return dp[m][n]


# # -------------------------------
# # evaluate loop (matches training packing)
# # -------------------------------
# @torch.no_grad()
# def evaluate(model: nn.Module,
#              loader: DataLoader,
#              id2gloss: Dict[int,str],
#              device: torch.device,
#              use_bf16: bool=False,
#              force_blank_id: Optional[int]=None) -> Tuple[float,float,int,int,float]:

#     model.eval()

#     # ----- warm-up to detect blank id -----
#     blank_id = 0 if force_blank_id is None else int(force_blank_id)
#     needs_auto_blank = force_blank_id is None

#     warm_batches = 8
#     checked = 0
#     if needs_auto_blank:
#         # frequency over vocab size inferred from mapping
#         vocab_size = max(id2gloss.keys()) + 1 if id2gloss else 1
#         freq = torch.zeros(vocab_size, dtype=torch.long)

#         autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
#                         if (use_bf16 and device.type=="cuda" and torch.cuda.is_bf16_supported())
#                         else torch.cuda.amp.autocast(enabled=False))
#         for b_idx, batch in enumerate(loader):
#             if b_idx >= warm_batches: break
#             images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths = batch
#             images = images.to(device, non_blocking=True)
#             qgrids = qgrids.to(device, non_blocking=True) if qgrids is not None else None
#             keypoints = keypoints.to(device, non_blocking=True) if keypoints is not None else None
#             qgrid_lengths = qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None
#             with autocast_ctx:
#                 logits = model(images, qgrids, keypoints, qgrid_lengths)
#             pred = logits.argmax(-1)  # (B,T)
#             vals, counts = pred.unique(return_counts=True)
#             freq[vals.cpu()] += counts.cpu()
#             checked += pred.numel()

#         if checked > 0:
#             blank_id = int(freq.argmax().item())
#             total = int(freq.sum().item())
#             top = int(freq.max().item())
#             top_share = 100.0 * top / max(1, total)
#             if top_share < 50.0:
#                 blank_id = 0
#             print(f"[auto] detected blank_id={blank_id} (top-share ~{top_share:.2f}%)", flush=True)
#         else:
#             print("[auto] could not sample warm-up batches; defaulting blank_id=0", flush=True)

#     # ----- full pass -----
#     def do_pass(dtype_bf16: bool) -> Tuple[float,float,int,float]:
#         total_w_edits = total_w_tok = 0
#         total_c_edits = total_c_tok = 0
#         N = 0
#         total_logits = 0
#         blank_hits = 0

#         autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
#                         if (dtype_bf16 and device.type=="cuda" and torch.cuda.is_bf16_supported())
#                         else torch.cuda.amp.autocast(enabled=False))
#         for batch in loader:
#             images, qgrids, keypoints, labels, image_lengths, label_lengths, qgrid_lengths = batch
#             images = images.to(device, non_blocking=True)
#             qgrids = qgrids.to(device, non_blocking=True) if qgrids is not None else None
#             keypoints = keypoints.to(device, non_blocking=True) if keypoints is not None else None
#             qgrid_lengths = qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None

#             with autocast_ctx:
#                 logits = model(images, qgrids, keypoints, qgrid_lengths)  # (B,T,V)

#             pred = logits.argmax(-1).cpu()  # (B,T)
#             blank_hits += int((pred == blank_id).sum().item())
#             total_logits += int(pred.numel())

#             labels = labels.cpu()
#             label_lengths = label_lengths.cpu()

#             B, _T = pred.shape
#             for b in range(B):
#                 # greedy collapse using detected blank_id
#                 seq = []
#                 prev = None
#                 for t in pred[b].tolist():
#                     if t == blank_id:
#                         prev = t; continue
#                     if prev is not None and t == prev:
#                         continue
#                     seq.append(t); prev = t
#                 hyp = " ".join(id2gloss.get(int(t), "<UNK>") for t in seq)

#                 L = int(label_lengths[b].item())
#                 ref_ids = labels[b, :L].tolist() if L>0 else []
#                 # strip only the detected blank index
#                 ref_ids = [int(t) for t in ref_ids if int(t) != blank_id]
#                 ref = " ".join(id2gloss.get(int(t), "<UNK>") for t in ref_ids)

#                 # WER
#                 hw, rw = hyp.split(), ref.split()
#                 m, n = len(hw), len(rw)
#                 dp = [[0]*(n+1) for _ in range(m+1)]
#                 for i in range(m+1): dp[i][0] = i
#                 for j in range(n+1): dp[0][j] = j
#                 for i in range(1, m+1):
#                     for j in range(1, n+1):
#                         dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (hw[i-1] != rw[j-1]))
#                 total_w_edits += dp[m][n]
#                 total_w_tok   += max(1, n)

#                 # CER
#                 hc, rc = list(hyp), list(ref)
#                 mc, nc = len(hc), len(rc)
#                 dp2 = [[0]*(nc+1) for _ in range(mc+1)]
#                 for i in range(mc+1): dp2[i][0] = i
#                 for j in range(nc+1): dp2[0][j] = j
#                 for i in range(1, mc+1):
#                     for j in range(1, nc+1):
#                         dp2[i][j] = min(dp2[i-1][j]+1, dp2[i][j-1]+1, dp2[i-1][j-1] + (hc[i-1] != rc[j-1]))
#                 total_c_edits += dp2[mc][nc]
#                 total_c_tok   += max(1, nc)

#                 N += 1

#         blank_ratio = 100.0 * blank_hits / max(1, total_logits)
#         WER = 100.0 * total_w_edits / max(1, total_w_tok)
#         CER = 100.0 * total_c_edits / max(1, total_c_tok)
#         return WER, CER, N, blank_ratio

#     # first pass (requested precision)
#     W1, C1, N1, BR1 = do_pass(use_bf16)
#     print(f"[debug] pass-1 blank@argmax (id={blank_id}): {BR1:.2f}%")

#     # auto-retry in fp32 if bf16 collapsed
#     if use_bf16 and BR1 > 95.0:
#         print("[debug] high blank ratio under bf16 — retrying in fp32 once.")
#         W2, C2, N2, BR2 = do_pass(False)
#         print(f"[debug] pass-2 (fp32) blank@argmax (id={blank_id}): {BR2:.2f}%")
#         return W2, C2, N2, blank_id, BR2

#     return W1, C1, N1, blank_id, BR1


# # -------------------------------
# # main
# # -------------------------------
# def main():
#     args = parse_args()
#     warnings.filterwarnings("once", category=FutureWarning)

#     device = torch.device(args.device)
#     print(f"[env] device={device}  bf16={args.bf16}", flush=True)

#     # ---- dataset (use same collate as training) ----
#     ds = MultiModalPhoenixDataset(
#         image_prefix=args.image_prefix,
#         qgrid_prefix=args.qgrid_prefix,
#         kp_path=args.kp_path,
#         meta_dir_path=args.meta_dir,
#         gloss_dict_path=args.gloss_dict,
#         split={"val":"dev","dev":"dev"}.get(args.split, args.split),
#     )
#     dl = DataLoader(
#         ds,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=False,
#         collate_fn=multi_modal_collate_fn,
#     )

#     # ---- vocab / classes ----
#     try:
#         gloss2id = getattr(ds, "gloss_dict")
#         id2gloss = getattr(ds, "id_to_gloss")
#         if not isinstance(gloss2id, dict) or not isinstance(id2gloss, dict):
#             raise RuntimeError("ds maps missing")
#     except Exception:
#         gloss2id, id2gloss = coerce_id_maps(args.gloss_dict)

#     # quick sample to check ids; keep mapping as-is
#     try:
#         tmp_batch = next(iter(dl))
#         lbls = tmp_batch[3]          # labels (B,L)
#         lbl_lens = tmp_batch[5]      # label_lengths
#         sample = []
#         for i in range(min(8, lbls.size(0))):
#             L = int(lbl_lens[i].item())
#             if L > 0:
#                 sample.append(lbls[i, :L])
#         sample = torch.cat(sample, dim=0) if sample else torch.zeros(0, dtype=torch.long)
#     except Exception:
#         sample = torch.zeros(0, dtype=torch.long)

#     id2gloss = ensure_blank_zero_and_shift_if_needed(sample, id2gloss)

#     if 0 not in id2gloss:
#         id2gloss[0] = "<blank>"
#     vocab_max = max(id2gloss.keys()) if id2gloss else 0
#     num_classes = (args.force_num_classes or (vocab_max + 1))
#     if args.force_num_classes:
#         print(f"[head] num_classes={num_classes} (forced)")
#     else:
#         print(f"[head] num_classes={num_classes} (from gloss_dict + blank)")

#     # ---- model ----
#     ckpt = torch.load(args.ckpt, map_location="cpu")
#     sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state") if isinstance(ckpt, dict) else ckpt
#     if sd is None:
#         sd = ckpt

#     # infer dims from checkpoint
#     fe = sd["img_proj.weight"].shape[0]      # fusion_embed
#     dm = sd["img_proj.weight"].shape[1]      # d_model
#     fusion_heads = 8 if fe % 8 == 0 else 4

#     # infer n_layer from keys if present
#     n_layer = None
#     try:
#         layer_ids = []
#         for k in sd.keys():
#             if "layers." in k:
#                 try:
#                     layer_ids.append(int(k.split("layers.")[1].split(".")[0]))
#                 except Exception:
#                     pass
#         if layer_ids:
#             n_layer = max(layer_ids) + 1
#             print(f"[arch] n_layer inferred from ckpt: {n_layer}")
#     except Exception:
#         pass

#     print(f"[arch] inferred from ckpt -> d_model={dm}, fusion_embed={fe}, fusion_heads={fusion_heads}, num_classes={sd['classifier.weight'].shape[0]}")

#     model = MultiModalMamba(
#         d_model=dm,
#         n_layer=(n_layer or 12),
#         fusion_embed=fe,
#         fusion_heads=fusion_heads,
#         num_classes=num_classes,   # keep dataset-derived classes (should match ckpt)
#         max_kv=args.max_kv,
#         pool_mode=args.pool_mode,
#     ).to(device)

#     wrap_frame_encoder_for_5d(model, verbose=True)

#     # materialize with a tiny forward before load
#     try:
#         b = next(iter(dl))
#         images, qgrids, keypoints = b[0], b[1], b[2]
#         qgrid_lengths = b[6]
#         with torch.no_grad():
#             _ = model(
#                 images.to(device, non_blocking=True),
#                 qgrids.to(device, non_blocking=True) if qgrids is not None else None,
#                 keypoints.to(device, non_blocking=True) if keypoints is not None else None,
#                 qgrid_lengths.to(device, non_blocking=True) if qgrid_lengths is not None else None,
#             )
#     except Exception:
#         pass

#     print(f"[ckpt] loading: {args.ckpt}")
#     load = model.load_state_dict(sd, strict=True)
#     unexpected = list(getattr(load, "unexpected_keys", []))
#     missing    = list(getattr(load, "missing_keys", []))
#     print(f"[ckpt] unexpected keys: {len(unexpected)}")
#     print(f"[ckpt] missing keys: {len(missing)}")

#     use_bf16 = args.bf16 and (device.type == "cuda") and torch.cuda.is_bf16_supported()
#     WER, CER, N, blank_id, blank_ratio = evaluate(model, dl, id2gloss, device, use_bf16=use_bf16)
#     print(f"[RESULT] N={N}  WER={WER:.2f}%  CER={CER:.2f}%  (blank_id={blank_id}, blank@argmax={blank_ratio:.2f}%)")


# if __name__ == "__main__":
#     main()
