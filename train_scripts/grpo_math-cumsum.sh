#!/bin/bash
# ============================================================================
# GRPO (Group Relative Policy Optimization) Training Script
# ============================================================================
# This script runs reinforcement learning training using GRPO algorithm
# on mathematical reasoning tasks.
#
# Prerequisites:
# - Ray cluster must be running
# - verl package must be installed
# - SFT model checkpoint from previous training
#
# Before running:
# 1. Set YOUR_WORK_DIR to your project directory
# 2. Set the correct MODEL_PATH to your SFT checkpoint
# 3. Ensure Ray cluster is properly configured
# ============================================================================

# ============================================================================
# Configuration - MODIFY THESE PATHS
# ============================================================================
YOUR_WORK_DIR="/path/to/your/workdir"  # TODO: Set your work directory
VERL_DIR="$YOUR_WORK_DIR/verl"

# Model configuration
SFT_MODEL_NAME="your_sft_model_name"   # TODO: Set your SFT model name
LOSS_FUNC="ce"
MODEL_TAG="Qwen2.5-7B-Math-Instruction-${LOSS_FUNC}-42-math-full-1024"
MODEL_PATH="$YOUR_WORK_DIR/logs/$MODEL_TAG"
MODEL_NAME="Math_filtered-${MODEL_TAG}"
SAVE_DIR="$YOUR_WORK_DIR/rl_ckpt/${MODEL_NAME}"

# WandB configuration
PROJECT_NAME="GRPO-math-revised-filtered_Qwen2.5-7B-Math-Instruction"
EXP_NAME="EXP_${LOSS_FUNC}"

# Data paths
TRAIN_FILE="$YOUR_WORK_DIR/data/math_train_data_filtered_qwen25_revised.parquet"
TEST_FILE="$YOUR_WORK_DIR/data/aime-2024.parquet"

# ============================================================================
# Environment Setup
# ============================================================================
export HYDRA_FULL_ERROR=1
export NUMEXPR_MAX_THREADS=32

# Ray environment variables
declare -A ray_vars
ray_vars["VLLM_ATTENTION_BACKEND"]="XFORMERS"
ray_vars["GLOO_SOCKET_IFNAME"]="bond1"
ray_vars["NCCL_SOCKET_IFNAME"]="bond1"
ray_vars["NCCL_IB_GID_INDEX"]="3"
ray_vars["NCCL_IB_SL"]="3"
ray_vars["NCCL_CHECK_DISABLE"]="1"
ray_vars["NCCL_P2P_DISABLE"]="0"
ray_vars["NCCL_IB_DISABLE"]="0"
ray_vars["NCCL_LL_THRESHOLD"]="16384"
ray_vars["NCCL_IB_CUDA_SUPPORT"]="1"
ray_vars["UCX_NET_DEVICES"]="bond1"
ray_vars["NCCL_IB_HCA"]="mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"
ray_vars["NCCL_COLLNET_ENABLE"]="0"
ray_vars["SHARP_COLL_ENABLE_SAT"]="0"
ray_vars["NCCL_NET_GDR_LEVEL"]="2"
ray_vars["NCCL_IB_QPS_PER_CONNECTION"]="4"
ray_vars["NCCL_IB_TC"]="160"
ray_vars["NCCL_PXN_DISABLE"]="0"
ray_vars["NCCL_IB_TIMEOUT"]="22"

# ============================================================================
# Algorithm Hyperparameters
# ============================================================================
ADV_ESTIMATOR="grpo"
KL_COEF=0.0
KL_LOSS_COEF=0.00
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
USE_TOKEN_LEVEL_LOSS=True
ENABLE_FILTER_GROUPS=False
ENABLE_OVERLONG_BUFFER=False
OVERLONG_PENALTY_FACTOR=1.0
FILTER_GROUPS_METRIC="acc"
MAX_NUM_GEN_BATCHES=10

# Sequence length configuration
MAX_PROMPT_LENGTH=$((1024 * 1))
MAX_RESPONSE_LENGTH=$((1024 * 3))
TRAIN_PROMPT_BSZ=256
GEN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ}
TRAIN_PROMPT_MINI_BSZ=32
N_RESP_PER_PROMPT=8
VAL_TOP_K=-1  # 0 for HF rollout, -1 for vLLM rollout

# Performance configuration
SP_SIZE=1
USE_DYNAMIC_BSZ=True
ACTOR_PPO_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
INFER_PPO_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
OFFLOAD=True
GEN_TP=2

# Ray configuration
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
RUNTIME_ENV=${RUNTIME_ENV:-"./verl/trainer/runtime_env.yaml"}
NNODES=${WORLD_SIZE}
PORT=6379

# ============================================================================
# Setup Directories
# ============================================================================
rm -rf "${SAVE_DIR}/ray_ready"
if [ ! -d "${SAVE_DIR}" ]; then
    mkdir -p "${SAVE_DIR}/ray_ready"
    chmod -R 777 "${SAVE_DIR}"
fi

cd "${SAVE_DIR}"
rm -f "${SAVE_DIR}/ray_ready/"*

# Setup WandB
wandb offline
ray_vars["WANDB_MODE"]="offline"
ray_vars["WANDB_DIR"]="${SAVE_DIR}"

# Save this script for reference
cp "$0" "$SAVE_DIR/"

# Build JSON environment variables for Ray
env_vars_json=()
keys=("${!ray_vars[@]}")
last_index=$(( ${#keys[@]} - 1 ))
for index in "${!keys[@]}"; do
    key="${keys[$index]}"
    value="${ray_vars[$key]}"
    if [[ $index -eq $last_index ]]; then
        env_vars_json+=("\"$key\": \"$value\"")
    else
        env_vars_json+=("\"$key\": \"$value\",")
    fi
done

json_env_vars="{\"env_vars\": {$(IFS=; echo "${env_vars_json[*]}")}}"
echo "Environment variables: ${json_env_vars}"

# ============================================================================
# Run Training
# ============================================================================
set -euxo pipefail

if [ "$RANK" -eq 0 ]; then
    # Master node: submit training job
    ray job submit --address="http://127.0.0.1:8265" \
        --working-dir="${VERL_DIR}" \
        --runtime-env-json="${json_env_vars}" \
        -- python3 -um verl.trainer.main_ppo \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.train_batch_size=${TRAIN_PROMPT_BSZ} \
        actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
        algorithm.adv_estimator=${ADV_ESTIMATOR} \
        algorithm.kl_ctrl.kl_coef=${KL_COEF} \
        actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
        actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
        actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN} \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN} \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        +actor_rollout_ref.model.override_config.attention_dropout=0. \
        +actor_rollout_ref.model.override_config.embd_pdrop=0. \
        +actor_rollout_ref.model.override_config.resid_pdrop=0. \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_MINI_BSZ} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
        actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K}" \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE} \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        actor_rollout_ref.rollout.name="vllm" \
        trainer.logger=['console','wandb'] \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXP_NAME}" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes="${NNODES}" \
        trainer.test_freq=5000 \
        trainer.save_freq=20 \
        trainer.total_epochs=10 \
        trainer.default_local_dir="${SAVE_DIR}" \
        trainer.resume_mode=auto \
        trainer.val_before_train=False \
        2>&1 | tee "${SAVE_DIR}/log.$(date +%Y-%m-%d-%H)"

    echo "Training completed on rank 0."

else
    # Worker nodes: wait for Ray cluster to stop
    echo "Worker rank $RANK waiting for Ray cluster..."
    
    while true; do
        ray status 1>/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Ray cluster stopped. Exiting worker..."
            break
        fi
        sleep 5m
    done
fi

echo "Rank $RANK script completed."
