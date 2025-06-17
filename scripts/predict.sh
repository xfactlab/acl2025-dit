#!/bin/bash

# Prediction script for all datasets and models
# Usage: ./predict.sh <dataset> <checkpoint_dir> [--with_pause] [--original_paper]
# Example: ./predict.sh gsm8k checkpoints/phi2_gsm8k_dynamic --with_pause

# Default settings:
# - Model: Phi-2
# - Dataset: specified in argument
# - Pause: disabled by default
# - Original paper: disabled by default

# For other models, change model_name to:
# - "meta-llama/Llama-2-7b-hf" for Llama-2
# - "microsoft/Phi-3-mini-4k-instruct" for Phi-3

if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> <checkpoint_dir> [--with_pause] [--original_paper]"
    exit 1
fi

DATASET=$1
CHECKPOINT_DIR=$2
MODEL_NAME="microsoft/phi-2"
PAUSE_ARGS=""
ORIGINAL_PAPER_ARGS=""

# Process optional arguments
for arg in "${@:3}"; do
    if [ "$arg" == "--with_pause" ]; then
        PAUSE_ARGS="--with_pause --pause_threshold 0.5"
    elif [ "$arg" == "--original_paper" ]; then
        ORIGINAL_PAPER_ARGS="--original_paper"
    fi
done

python generate_cli.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --save_dir "predictions" \
    $PAUSE_ARGS \
    $ORIGINAL_PAPER_ARGS

#sbatch —-gpus=1 —-time=23:59:59 -—cpus-per-gpu=8 -—partition=rtx6000 predict.sh