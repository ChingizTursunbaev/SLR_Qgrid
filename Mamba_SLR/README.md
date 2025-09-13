# Sign Language Recognition (SLR) Repo – Beginner’s Guide

Welcome to the SLR codebase! This document walks you through everything you need to know to understand, run, and extend this project from day one.

---

## Table of Contents

- [Sign Language Recognition (SLR) Repo – Beginner’s Guide](#sign-language-recognition-slr-repo--beginners-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Prerequisites \& Installation](#2-prerequisites--installation)
  - [3. Directory Structure](#3-directory-structure)
  - [4. Data \& Data Loading](#4-data--data-loading)
    - [4.1 Data Directories \& Formats](#41-data-directories--formats)
    - [4.2 Dataset Class](#42-dataset-class)
    - [4.3 Collate Functions](#43-collate-functions)
  - [5. Model Architecture](#5-model-architecture)
  - [6. Training Engine](#6-training-engine)
    - [6.1 `train_one_epoch`](#61-train_one_epoch)
    - [6.2 `evaluate_one_epoch`](#62-evaluate_one_epoch)
  - [7. Utilities \& Helpers](#7-utilities--helpers)
    - [7.1 Logging](#71-logging)
    - [7.2 Scheduling](#72-scheduling)
    - [7.3 Distributed Training](#73-distributed-training)
    - [7.4 Checkpointing](#74-checkpointing)
  - [8. Experiment Scripts](#8-experiment-scripts)
  - [9. How to Start Training](#9-how-to-start-training)
  - [10. Tips for Beginners](#10-tips-for-beginners)

---

## 1. Project Overview

This repository implements an end-to-end pipeline for **Sign Language Recognition** on the Phoenix2014 dataset (German sign language).  
The key components are:

- **Data loading** (`datasets/`)  
- **Model** (`models/model.py`): a spatio-temporal Vision-Transformer–based network  
- **Training & evaluation** (`engine.py`, `main.py`)  
- **Utilities** (`utils.py`, `optim_factory.py`)  
- **Example scripts** (`exp/pheonix-2014/`)  

---

## 2. Prerequisites & Installation

- Python 3.8+  
- PyTorch (1.9+) with CUDA  
- `timm`  
- `deepspeed` (if `--enable_deepspeed` is used)  
- `tensorboardX`  
- `torchmetrics`  

Install core dependencies:

```bash
pip install torch torchvision timm deepspeed tensorboardX torchmetrics
```

---

## 3. Directory Structure

```
slr/
└─ slr/                           # <-- main package
   ├─ main.py                     # Entry point (arg parsing, pipeline orchestration)
   ├─ engine.py                   # train_one_epoch & evaluate_one_epoch
   ├─ utils.py                    # logging, schedulers, distributed helpers, collate
   ├─ optim_factory.py            # optimizer creation & layer-wise decay
   ├─ datasets/
   │   ├─ datasets.py             # Phoenix2014 Dataset & collate_fn
   │   └─ rand_augment.py         # RandAugment, AutoAugment implementations
   ├─ models/
   │   └─ model.py                # Vision-Transformer–based Model class
   └─ exp/
       └─ pheonix-2014/
           └─ run_phoenix.sh      # example shell script with flags
```

---

## 4. Data & Data Loading

### 4.1 Data Directories & Formats

1. **`--prefix`**  
   The root of your video data.  
   Expected structure:
   ```
   PREFIX/
   ├─ train/
   │   ├─ video_0001/ frame0001.jpg, frame0002.jpg, …
   │   └─ …
   ├─ dev/    (same layout)
   └─ test/   (same layout)
   ```
2. **`--meta_dir_path`**  
   Path to JSON metadata for Phoenix2014:
   ```
   META_DIR_PATH/
   ├─ train.corpus.json
   ├─ dev.corpus.json
   └─ test.corpus.json
   ```
   Each file maps video IDs → gloss sequences.
3. **`--gloss_dict_path`**  
   A NumPy `.npy` file containing a Python dict `{ gloss_str: int_id }`.
4. **`--kp_path`** (optional)  
   A `.pkl` with precomputed pose keypoints per frame.

### 4.2 Dataset Class

File: `datasets/datasets.py`  
Class: `Phoenix2014`  
- Reads frames, optional heatmaps, labels.  
- Returns a tuple:
  ```python
  (videos, heatmaps, video_lengths, labels, glosses, label_lengths, frame_ids)
  ```

### 4.3 Collate Functions

Defined in `utils.py`:

- `multiple_samples_collate`: for repeated augmentations  
- `multiple_pretrain_samples_collate`: similar but supports `fold=True`  

In `main.py`, the default is:
```python
from datasets.datasets import collate_fn
# …
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, sampler=sampler_train,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=True,
    collate_fn=collate_fn,
    persistent_workers=True
)
```

---

## 5. Model Architecture

File: `models/model.py`  
Class: `Model`  
- A patch-embed Vision Transformer with spatial & temporal mixing layers  
- Configurable via:
  - `img_size`, `patch_size`, `d_model`, `n_layer`, `drop_path_rate`, etc.  
  - `ssm_cfg` for structured state-space mixers  
  - `attn_cfg` for attention heads  

The model outputs frame-level features, then applies a linear head and CTC for sequence prediction.

---

## 6. Training Engine

File: `engine.py`

### 6.1 `train_one_epoch`

- Puts model in `train()`  
- Iterates over data via `MetricLogger.log_every`  
- Handles gradient accumulation (`update_freq`) and mixed-precision / DeepSpeed  
- Updates learning rate & weight decay schedules per step  
- Computes CTC loss and backpropagates  

### 6.2 `evaluate_one_epoch`

- Runs validation or test in `torch.no_grad()`  
- Decodes logits via CTC beam or greedy  
- Computes **Word Error Rate** (`torchmetrics.text.WordErrorRate`)  

---

## 7. Utilities & Helpers

File: `utils.py`

### 7.1 Logging

- `SmoothedValue`: rolling statistics (avg, max)  
- `MetricLogger`: prints metrics every _n_ iterations  
- `TensorboardLogger`: wraps `tensorboardX.SummaryWriter`  

### 7.2 Scheduling

- `cosine_scheduler`: cosine learning‐rate & weight‐decay schedules

### 7.3 Distributed Training

- `init_distributed_mode`: sets up `torch.distributed`  
- `get_world_size` / `get_rank`  
- `setup_for_distributed`: mutes print on non-master processes  

### 7.4 Checkpointing

- `auto_load_model`: resumes weights, optimizer, scaler, EMA if present  
- `save_model`: saves model, optimizer, scaler, EMA state  

---

## 8. Experiment Scripts

Example: `exp/pheonix-2014/run_phoenix.sh`  
Shows how to launch with Deepspeed, bf16, finetuning, EMA, distributed eval, etc.

---

## 9. How to Start Training

1. **Prepare your data**  
   - Download Phoenix2014 frames under `PREFIX/{train,dev,test}`  
   - Place `*.corpus.json` in `META_DIR_PATH/`  
   - Generate or download `gloss_dict.npy` and optional `kp_path.pkl`

2. **Configure & launch**  
   ```bash
   python main.py \
     --batch_size 4 \
     --epochs 50 \
     --lr 1e-4 \
     --prefix /path/to/phoenix-2014-multisigner \
     --meta_dir_path /path/to/phoenix2014/meta \
     --gloss_dict_path /path/to/gloss_dict.npy \
     --kp_path /path/to/keypoints.pkl \
     --output_dir ./outputs \
     --log_dir ./logs \
     --enable_deepspeed \
     --bf16 \
     --model_ema \
     --dist_eval
   ```

3. **Monitor**  
   - Check console for real-time logging (loss, lr, ETA)  
   - Start TensorBoard:
     ```bash
     tensorboard --logdir ./logs
     ```

4. **Checkpoints**  
   - **latest** saved every epoch  
   - **best** saved based on validation WER  

---

## 10. Tips for Beginners

- Start with `--disable_eval_during_finetuning` to speed up training.  
- Try small `--batch_size` and fewer `--epochs` to verify setup.  
- Use `--eval` (no training) to validate data loading & model forward pass.  
- Read through `MetricLogger.log_every` in `utils.py` to understand logging.  
- Explore `LayerDecayValueAssigner` in `optim_factory.py` for fine-tuning.