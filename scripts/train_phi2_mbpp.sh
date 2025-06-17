#!/bin/bash

# Training script for Phi-2 on MBPP dataset
# Default setting: dynamic
# Other settings (uncomment to use):
# --original
# --original_paper
# --random
# --all

# For other models, change model_name to:
# - "meta-llama/Llama-2-7b-hf" for Llama-2
# - "microsoft/Phi-3-mini-4k-instruct" for Phi-3

python cli.py \
    --model_name "microsoft/phi-2" \
    --dataset "mbpp" \
    --output_dir "checkpoints/phi2_mbpp_dynamic" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "accuracy" \
    --greater_is_better True \
    --seed 42 \
    --fp16 True 