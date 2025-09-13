+148
-0

# Mamba SLR Repository Overview

This repository implements a sign language recognition (SLR) pipeline that
leverages Mamba-based neural network layers.  The code is organised as a
self-contained project under `slr/` and includes a sample dataset under
`data/`.

The sections below describe the repository layout, data preparation, and how to
start training.

## Repository Structure

```
Mamba_SLR/
├── data/                   # example metadata for the PHOENIX-2014 dataset
│   └── phoenix2014/
│       ├── dev_info.npy
│       ├── test_info.npy
│       ├── train_info.npy
│       ├── gloss_dict.npy
│       └── phoenix2014-groundtruth-*.stm
├── slr/                    # source code for model, dataset and training
│   └── slr/
│       ├── datasets/       # dataloading utilities and augmentations
│       ├── models/         # Mamba based model definition
│       ├── exp/            # example training script & deepspeed config
│       ├── main.py         # entry point for training
│       ├── engine.py       # train / eval loops
│       └── utils.py        # helper utilities
└── pip_list.txt            # example list of python packages
```

`pip_list.txt` shows the Python packages originally used to run the code.  It
is provided for reference when setting up your environment.

## Dataset Format

The code expects the [RWTH-PHOENIX-Weather 2014 T](https://www.phoenixrwth.de/phoenix-2014/) dataset.
A minimal subset of metadata is included in `data/phoenix2014` to illustrate the
required files:

- `*_info.npy` – numpy dictionaries describing each video sample.  Each entry is
  a dict with keys like `fileid`, `folder`, `signer`, `label`, and
  `num_frames`.
- `gloss_dict.npy` – mapping from gloss strings to integer indices.  The model
  uses this vocabulary for CTC decoding.
- `phoenix2014-groundtruth-*.stm` – original ground‑truth transcripts.

`prefix` paths (specified when running the training script) must point to the
video frames extracted from the dataset.  The frame files are expected under:

```
<dataset_root>/features/fullFrame-256x256px/<split>/<video_id>/<frame>.png
```

where `<split>` is `train`, `dev` or `test`.  Keypoint heatmaps can also be
provided via `--kp_path` if you want to use pose information.

The included `.npy` files are small demo versions and do not contain the actual
videos.

## Data Loading

The `Phoenix2014` class in `slr/slr/datasets/datasets.py` handles loading the
video frames and labels.  Important arguments include:

- `prefix` – path to the dataset root containing the extracted frames.
- `gloss_dict` – dictionary mapping gloss string to index (loaded from
  `gloss_dict.npy`).
- `meta_dir_path` – directory containing the `*_info.npy` metadata files.
- `frame_interval` – temporal sampling stride (e.g. 1 loads every frame).
- `use_heatmap` and `kp_path` – enable loading keypoint data and converting it
  to heatmap images used as an additional input channel.

For training the dataset returns `(frames, heatmaps, label_tensor, gloss_text,
label_length)` where:

- `frames` is a tensor of shape `(T, 3, H, W)`.
- `heatmaps` is `None` or `(T, H, W)` if keypoint heatmaps are enabled.
- `label_tensor` contains the integer encoded gloss sequence.
- `gloss_text` is the original string transcript.
- `label_length` stores the number of gloss tokens.

`datasets.collate_fn` merges variable length sequences inside a batch and also
computes a `frame_ids` tensor used by the model for positional indexing.

Augmentations such as random cropping, horizontal flip and RandAugment are
implemented in `datasets.pair_rand_augment` and `video_transforms`.

## Model Overview

The model architecture is defined in `models/model.py` and is composed of:

1. **Stem** (`models/stem.py`) – converts input frames into patches.
2. **Blocks** – a sequence of spatial and temporal layers, implemented with
   Mamba‑style operations (`spatial_layer.py` and `temporal_layer.py`).
3. **CTC Head** – after feature extraction the network predicts gloss
   probabilities which are decoded with a CTC decoder from `torchaudio`.

The model expects sequences of patch embeddings arranged as `(batch*time,
height*width+1, dim)`.  The `decode` method uses the `CUCTCDecoder` for WER
computation.

## Training and Evaluation

The main training entry point is `slr/slr/main.py`.  The repository includes an
example script `slr/slr/exp/pheonix-2014/run_phoenix.sh` which launches training
via `torchrun`.  Key options in this script are:

- model size and depth (`--d_model`, `--depth`, etc.)
- optimisation hyper‑parameters (`--lr`, `--weight_decay`, etc.)
- dataset paths (`--prefix`, `--gloss_dict_path`, `--meta_dir_path`)
- optional keypoint heatmap path (`--kp_path`)
- enabling DeepSpeed with BF16 for distributed training

To start training from scratch or fine‑tune a checkpoint you would typically:

1. Prepare the dataset directory with extracted frame images following the
   expected layout.
2. Adjust paths in `run_phoenix.sh` to match your environment.
3. Run the script:

   ```bash
   cd slr/slr/exp/pheonix-2014
   bash run_phoenix.sh
   ```

During training checkpoints and logs are saved into the directory specified by
`--output_dir` and `--log_dir`.  After each epoch the validation WER is printed
and the best model can be automatically evaluated on the test set with
`--test_best`.

## Notes for Beginners

- **Environment** – create a Python environment (Python 3.12 was used in the
  example `pip_list.txt`) and install PyTorch with GPU support.  Additional
  libraries such as `mamba-ssm`, `deepspeed` and `torchvision` are required.
- **Data** – the repository only provides metadata, not the actual PHOENIX-2014
  videos.  You must obtain the dataset separately and set `--prefix` to the
  directory containing the frame images.
- **Configuration** – almost all training hyper‑parameters have command line
  flags in `main.py`.  The provided shell script is a good starting point.

With the dataset placed correctly and dependencies installed, running the shell
script will launch distributed training across the specified number of GPUs.
Model checkpoints, logs and results can then be found in the chosen output
folder.