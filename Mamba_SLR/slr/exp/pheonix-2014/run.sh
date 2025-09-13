#!/bin/bash

# --- Configuration ---
N_GPU=2
MASTER_PORT=$((12000 + RANDOM % 20000))

# --- Paths ---
IMAGE_PREFIX="/nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px"
QGRID_PREFIX="/nas/Dataset/Phoenix/Phoenix-2014_cleaned/interpolated_original/Qgrid_npy"
KEYPOINTS_PATH="/home/chingiz/SLR/Mamba_SLR/data/phoenix2014/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256.pkl"
METADATA_DIR="/home/chingiz/SLR/SLR_Qgrid/Mamba_SLR/data/phoenix2014"
GLOSS_DICT_PATH="/home/chingiz/SLR/SLR_Qgrid/Mamba_SLR/data/phoenix2014/gloss_dict.npy"

# --- Output Directories ---
OUTPUT_DIR="./output/multimodal_mamba_experiment_1"
LOG_DIR="./logs/multimodal_mamba_experiment_1"

# --- Model Hyperparameters ---
D_MODEL=512
IMG_DEPTH=12
QGRID_DEPTH=18
BATCH_SIZE=2

# --- Deepspeed Execution ---
deepspeed --num_gpus=${N_GPU} --master_port=${MASTER_PORT} slr/main.py \
    --prefix "${IMAGE_PREFIX}" \
    --meta_dir_path "${METADATA_DIR}" \
    --gloss_dict_path "${GLOSS_DICT_PATH}" \
    --qgrid_prefix "${QGRID_PREFIX}" \
    --kp_path "${KEYPOINTS_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --epochs 100 \
    --d_model ${D_MODEL} \
    --depth ${IMG_DEPTH} \
    --qgrid_depth ${QGRID_DEPTH} \
    --lr 1e-4 \
    --warmup_epochs 5 \
    --min_lr 1e-6 \
    --weight_decay 0.05 \
    --layer_decay 0.75 \
    --enable_deepspeed \
    --bf16 \
    --model_ema \
    --dist_eval

echo "--- Training script finished ---"














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