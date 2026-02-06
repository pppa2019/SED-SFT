#!/bin/bash
# ============================================================================
# Data Tokenization Script
# ============================================================================
# This script tokenizes raw conversation data into a format suitable for
# supervised fine-tuning training.
#
# Before running:
# 1. Set YOUR_WORK_DIR to your project directory
# 2. Set YOUR_PRETRAINED_MODEL_PATH to where your backbone models are stored
# 3. Ensure the raw dataset exists at the specified path
# ============================================================================

set -e  # Exit on error
set -x  # Print commands

# ============================================================================
# Configuration - MODIFY THESE PATHS
# ============================================================================
YOUR_WORK_DIR="/path/to/your/workdir"         # TODO: Set your work directory
YOUR_PRETRAINED_MODEL_PATH="/path/to/models"  # TODO: Set your models directory
DATASET_PATH="$YOUR_WORK_DIR/train_data/MiroMind-M1-SFT-719K"

# ============================================================================
# Environment Setup
# ============================================================================
cd "$YOUR_WORK_DIR"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC="1"
export CUDA_VISIBLE_DEVICES="0"

# ============================================================================
# Tokenization Configuration
# ============================================================================
MODEL_NAME="Llama-3.2-3B-Instruct"
TOKENIZER_PATH="$YOUR_PRETRAINED_MODEL_PATH/$MODEL_NAME"
MAX_SEQ_LENGTH=32000
OUTPUT_DIR="./data"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Tokenize Training Data
# ============================================================================
echo "=== Tokenizing training data ==="
python3 preprocess_data.py \
    --dataset_name_or_path "$DATASET_PATH" \
    --split "train" \
    --tokenizer_name_or_path "$TOKENIZER_PATH" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --output_file "$OUTPUT_DIR/miromind_sampled_formated_sft_train_llama3o2_tokenized-full.jsonl" \
    --start 0 \
    --end 20000

# ============================================================================
# Tokenize Test Data
# ============================================================================
echo "=== Tokenizing test data ==="
python3 preprocess_data.py \
    --dataset_name_or_path "$DATASET_PATH" \
    --split "test" \
    --tokenizer_name_or_path "$TOKENIZER_PATH" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --output_file "$OUTPUT_DIR/miromind_sampled_formated_sft_test_llama3o2_tokenized-full.jsonl" \
    --start 20000 \
    --end 21000

echo "=== Tokenization complete ==="
echo "Training data: $OUTPUT_DIR/miromind_sampled_formated_sft_train_llama3o2_tokenized-full.jsonl"
echo "Test data: $OUTPUT_DIR/miromind_sampled_formated_sft_test_llama3o2_tokenized-full.jsonl"