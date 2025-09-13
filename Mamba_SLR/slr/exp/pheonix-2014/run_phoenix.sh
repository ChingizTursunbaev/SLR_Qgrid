# prepend to run_phoenix.sh (or your shell init)
# (no filepath—just add these lines before torchrun)
# export CUDA_HOME=/usr/local/cuda-11.8
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# export PYTHONPATH=/shared/home/xvoice/nirmal/mambaslr/Mamba_SLR/slr:$PYTHONPATH

# Dynamically determine number of GPUs
if command -v nvidia-smi &> /dev/null; then
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    N_GPUS=$(python - <<EOF
import torch
print(torch.cuda.device_count())
EOF
)
fi
MASTER_PORT=$((12000 + $RANDOM % 20000))


torchrun --nproc_per_node="$N_GPUS" --master_port="$MASTER_PORT" ../../main.py \
    --batch_size 2 \
    --epochs 100 \
    --update_freq 1 \
    --input_size 224 \
    --patch_size 16 \
    --d_model 512 \
    --headdim 64 \
    --depth 12 \
    --expand 2 \
    --head_drop_rate 0.1 \
    --drop_path 0.4 \
    --model_ema \
    --model_ema_decay 0.9999 \
    --model_ema_eval \
    --opt 'adamw' \
    --opt_betas 0.9 0.999 \
    --clip_grad 5.0 \
    --weight_decay 0.05 \
    --lr 1e-3 \
    --layer_decay 0.75 \
    --warmup_lr 1e-4 \
    --min_lr 1e-5 \
    --warmup_epochs 5 \
    --aa 'rand-m5-mstd0.25' \
    --temp_scale 0.0 \
    --model_key 'model_ema' \
    --prefix '/shared/home/xvoice/nirmal/SlowFastSign/dataset/phoenix2014' \
    --gloss_dict_path '/shared/home/xvoice/nirmal/mambaslr/Mamba_SLR/data/phoenix2014/gloss_dict.npy' \
    --meta_dir_path '/shared/home/xvoice/nirmal/mambaslr/Mamba_SLR/data/phoenix2014' \
    --kp_path '/shared/home/xvoice/nirmal/mambaslr/Mamba_SLR/data/phoenix2014/phoenix-2014-keypoints_hrnet-filtered_SMOOTH_v2-256x256_INTERPOLATED.pkl' \
    --output_dir '/shared/home/xvoice/nirmal/mambaslr/Mamba_SLR/slr/exp/pheonix-2014' \
    --log_dir '/shared/home/xvoice/nirmal/mambaslr/Mamba_SLR/slr/exp/pheonix-2014' \
    --resume '/shared/home/xvoice/nirmal/mambaslr/Mamba_SLR/slr/exp/pheonix-2014' \
    --auto_resume \
    --save_ckpt \
    --test_best \
    --num_workers 12 \
    --pin_mem \
    --enable_deepspeed \
    --bf16 \
    # --finetune '/home/kks/workspace/slr/image_backbone/exp/convnextStem-dim_512-depths_12-noz-cls/checkpoint-best-ema.pth' \