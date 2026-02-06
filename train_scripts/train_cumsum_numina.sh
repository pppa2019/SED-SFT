#!/bin/bash
# ============================================================================
# SFT Training Script with SED Loss
# ============================================================================
# This script trains a language model using Selective Entropy-based Distillation
# (SED) loss with top-k cumulative probability ratio masking.
#
# Before running:
# 1. Set YOUR_WORK_DIR to your project directory
# 2. Set YOUR_PRETRAINED_MODEL_PATH to where your backbone models are stored
# 3. Ensure the tokenized data files exist (run tokenize_data.sh first)
# ============================================================================

set -e  # Exit on error
set -x  # Print commands

# ============================================================================
# Configuration - MODIFY THESE PATHS
# ============================================================================
YOUR_WORK_DIR="/path/to/your/workdir"         # TODO: Set your work directory
YOUR_PRETRAINED_MODEL_PATH="/path/to/models"  # TODO: Set your models directory

# ============================================================================
# Environment Setup
# ============================================================================
cd "$YOUR_WORK_DIR"

# Cache directories
CACHE_DIR="$YOUR_WORK_DIR/cache"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCH_HOME="$CACHE_DIR/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions"
export HF_HOME="$CACHE_DIR/huggingface"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface"

# Offline mode (set to 0 if you need to download models)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Other settings
export CXX=g++
export WANDB_DISABLED=true
export OMP_NUM_THREADS=20
export FLASH_ATTENTION_DETERMINISTIC="1"

# ============================================================================
# GPU Configuration
# ============================================================================
NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR="${CHIEF_IP:=localhost}"

# ============================================================================
# Training Configuration
# ============================================================================
# Loss function
LOSS_FUNC="sed_with_topk_cumsum_ratio"

# Model
MODEL_NAME="Llama-3.2-3B-Instruct"
MODEL_NAME_OR_PATH="$YOUR_PRETRAINED_MODEL_PATH/$MODEL_NAME"

# Data
TRAIN_TOKENIZED_FILE="$YOUR_WORK_DIR/data/miromind_sampled_formated_sft_train_llama3o2_tokenized-full.jsonl"
TEST_TOKENIZED_FILE="$YOUR_WORK_DIR/data/miromind_sampled_formated_sft_test_llama3o2_tokenized-full.jsonl"

# Training hyperparameters
SEED=42
LEARNING_RATE=2e-5
NUM_EPOCHS=2
BATCH_SIZE=1
GRAD_ACCUM_STEPS=16
BLOCK_SIZE=1024
WARMUP_RATIO=0.03
CUMSUM_RATIO=0.7

# Output
TIME_STEP=$(date "+%Y-%m-%d")
OUTPUT_DIR="./logs/${MODEL_NAME}-${LOSS_FUNC}-${SEED}-${TIME_STEP}"

# ============================================================================
# Create output directory
# ============================================================================
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Run Training
# ============================================================================
deepspeed train.py \
    --seed "$SEED" \
    --deepspeed "$YOUR_WORK_DIR/configs/zero2.json" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --train_tokenized_file "$TRAIN_TOKENIZED_FILE" \
    --test_tokenized_file "$TEST_TOKENIZED_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --save_strategy "steps" \
    --save_steps 50 \
    --loss "$LOSS_FUNC" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --warmup_ratio "$WARMUP_RATIO" \
    --block_size "$BLOCK_SIZE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --overwrite_output_dir \
    --bf16 True \
    --cumsum_ratio "$CUMSUM_RATIO" \
    --use_flash_attn False \
    2>&1 | tee "$OUTPUT_DIR/training.log"
