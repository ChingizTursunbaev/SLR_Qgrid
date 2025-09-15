#!/usr/bin/env bash
set -euo pipefail

cd ~/SLR/SLR_Qgrid/Mamba_SLR

# Safer single-node NCCL defaults
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_PORT=${MASTER_PORT:-29531}

# Train on 2x A100 80GB; BF16 to avoid FP16 overflows
torchrun --standalone --nproc_per_node=2 ddp_train_multimodal.py \
  --batch_size 2 \
  --accum 2 \
  --num_workers 4 \
  --bf16 \
  --max_kv 1024 \
  --pool_mode mean









# #!/bin/bash
# set -e

# # --- GPUs ---
# N_GPU=${N_GPU:-2}
# MASTER_PORT=${MASTER_PORT:-29501}

# # --- Paths (EDIT THESE to your machine) ---
# IMAGE_PREFIX="/nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px"
# QGRID_PREFIX="/nas/Dataset/Phoenix/Phoenix-2014_cleaned/interpolated_original/Qgrid_npy"
# KEYPOINTS_PATH="/home/chingiz/SLR/Mamba_SLR/data/phoenix2014/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl"
# METADATA_DIR="/home/chingiz/SLR/SLR_Qgrid/Mamba_SLR/data/phoenix2014"
# GLOSS_DICT_PATH=/home/chingiz/SLR/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict_normalized.npy       #"/home/chingiz/SLR/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict.npy"

# OUTPUT_DIR="./output/multimodal_mamba_experiment_1"

# # --- Model cfg (make sure Model returns features with these out_dims) ---
# IMG_OUT=512
# QGD_OUT=512
# EMBED=512
# HEADS=8
# DROP=0.1

# torchrun --nproc_per_node=${N_GPU} --master_port=${MASTER_PORT} slr/main.py \
#   --image_prefix "${IMAGE_PREFIX}" \
#   --qgrid_prefix "${QGRID_PREFIX}" \
#   --kp_path "${KEYPOINTS_PATH}" \
#   --meta_dir_path "${METADATA_DIR}" \
#   --gloss_dict_path "${GLOSS_DICT_PATH}" \
#   --output_dir "${OUTPUT_DIR}" \
#   --epochs 40 --batch_size 8 --update_freq 1 \
#   --num_workers 6 \
#   --input_size 224 --patch_size 16 \
#   --d_model 512 --headdim 64 --depth 12 --qgrid_depth 18 --expand 2 \
#   --drop_path 0.2 --head_drop_rate 0.1 \
#   --opt adamw --lr 1e-4 --weight_decay 0.05 \
#   --warmup_epochs 5 --min_lr 1e-6 \
#   --model_ema --model_ema_decay 0.999 --model_ema_eval \
#   --enable_deepspeed \
#   --bf16 \
#   --fusion_embed_dim ${EMBED} --fusion_num_heads ${HEADS} --fusion_dropout ${DROP} \
#   --img_out_dim ${IMG_OUT} --qgrid_out_dim ${QGD_OUT}















# #!/bin/bash

# # --- Configuration ---
# # Set this to the number of GPUs you have
# N_GPU=2

# # Set the master port for distributed training
# MASTER_PORT=$((12000 + RANDOM % 20000))

# # --- Paths ---
# # [CORRECTED] - Removed the space after the '=' for all variables.
# IMAGE_PREFIX="/nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px"
# QGRID_PREFIX="/nas/Dataset/Phoenix/Phoenix-2014_cleaned/interpolated_original/Qgrid_npy"
# KEYPOINTS_PATH="/home/chingiz/SLR/Mamba_SLR/data/phoenix2014/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl"
# METADATA_DIR="/home/chingiz/SLR/SLR_Qgrid/Mamba_SLR/data/phoenix2014"                    #"/home/chingiz/SLR/SLR_Qgrid/Mamba_SLR/data/phoenix2014"
# GLOSS_DICT_PATH="/home/chingiz/SLR/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict.npy"

# # --- Output Directories ---
# OUTPUT_DIR="./output/multimodal_mamba_experiment_1"
# LOG_DIR="./logs/multimodal_mamba_experiment_1"

# # --- Model Hyperparameters ---
# D_MODEL=512
# IMG_DEPTH=12
# QGRID_DEPTH=18
# BATCH_SIZE=2

# # --- Deepspeed Execution ---
# # [CORRECTED] - The path to main.py is correct, but we must run this script from the project root.
# deepspeed --num_gpus=${N_GPU} --master_port=${MASTER_PORT} slr/main.py \
#     --prefix "${IMAGE_PREFIX}" \
#     --meta_dir_path "${METADATA_DIR}" \
#     --gloss_dict_path "${GLOSS_DICT_PATH}" \
#     --qgrid_prefix "${QGRID_PREFIX}" \
#     --kp_path "${KEYPOINTS_PATH}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --log_dir "${LOG_DIR}" \
#     \
#     --batch_size ${BATCH_SIZE} \
#     --epochs 100 \
#     --d_model ${D_MODEL} \
#     --depth ${IMG_DEPTH} \
#     --qgrid_depth ${QGRID_DEPTH} \
#     \
#     --lr 1e-4 \
#     --warmup_epochs 5 \
#     --min_lr 1e-6 \
#     --weight_decay 0.05 \
#     --layer_decay 0.75 \
#     \
#     --enable_deepspeed \
#     --bf16 \
#     --model_ema \
#     --dist_eval

# echo "--- Training script finished ---"

















# #!/bin/bash

# # --- Configuration ---
# # Set this to the number of GPUs you have
# N_GPU=2

# # Set the master port for distributed training
# MASTER_PORT=$((12000 + RANDOM % 20000))

# # --- Paths ---
# # IMPORTANT: Replace these placeholder paths with the actual paths to your data
# IMAGE_PREFIX="/nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px"
# #IMAGE_PREFIX="/nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner/"
# QGRID_PREFIX="/nas/Dataset/Phoenix/Phoenix-2014_cleaned/interpolated_original/Qgrid_npy"
# KEYPOINTS_PATH="/home/chingiz/SLR/Mamba_SLR/data/phoenix2014/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl"
# METADATA_DIR="/home/chingiz/SLR/SLR_Qgrid/Mamba_SLR/data/phoenix2014"
# GLOSS_DICT_PATH="/home/chingiz/SLR/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict.npy"

# # --- Output Directories ---
# OUTPUT_DIR="./output/multimodal_mamba_experiment_1"
# LOG_DIR="./logs/multimodal_mamba_experiment_1"

# # --- Model Hyperparameters ---
# # These are starting points. You will likely need to tune them.
# D_MODEL=512
# IMG_DEPTH=12
# QGRID_DEPTH=18 # A deeper encoder for the more detailed Qgrid data
# BATCH_SIZE=2   # Start with a small batch size due to the large model

# # --- Deepspeed Execution ---
# deepspeed --num_gpus=${N_GPU} --master_port=${MASTER_PORT} slr/main.py \
#     --prefix "${IMAGE_PREFIX}" \
#     --meta_dir_path "${METADATA_DIR}" \
#     --gloss_dict_path "${GLOSS_DICT_PATH}" \
#     --qgrid_prefix "${QGRID_PREFIX}" \
#     --kp_path "${KEYPOINTS_PATH}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --log_dir "${LOG_DIR}" \
#     \
#     --batch_size ${BATCH_SIZE} \
#     --epochs 100 \
#     --d_model ${D_MODEL} \
#     --depth ${IMG_DEPTH} \
#     --qgrid_depth ${QGRID_DEPTH} \
#     \
#     --lr 1e-4 \
#     --warmup_epochs 5 \
#     --min_lr 1e-6 \
#     --weight_decay 0.05 \
#     --layer_decay 0.75 \
#     \
#     --enable_deepspeed \
#     --bf16 \
#     --model_ema \
#     --dist_eval

# echo "--- Training script finished ---"