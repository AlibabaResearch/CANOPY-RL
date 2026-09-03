#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CANOPY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
VERL_ROOT="${CANOPY_ROOT}"

# Edit these local paths before launching. Data, checkpoints, and outputs are
# intentionally not distributed with the source release.
MODEL_ROOT="/path/to/models"
DATA_ROOT="/path/to/verl_data"
OUTPUT_ROOT="/path/to/outputs"
RUNTIME_ROOT="/path/to/shared/canopy-runtime"
RAY_ADDRESS="http://127.0.0.1:8265"

export CANOPY_ROOT VERL_ROOT
cd "${CANOPY_ROOT}"

export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

ulimit -n 65535

project_name="appworld_agentrl"
model_base="${MODEL_ROOT}"


model_name="Qwen3-14B"
model_path=${model_base}/${model_name}



entropy_coeff=0
kl_loss_coef=0.0001

# exp_name="0116_qwen3_14b_32k_50turns_microbs4"
exp_name="0118_qwen3_14b_32k_50turns_microbs4_kl0001"

output_base="${OUTPUT_ROOT}"
output_dir="${output_base}/${project_name}/${exp_name}"

data_base="${DATA_ROOT}"
data_name="appworld"
train_files="${data_base}/${data_name}/train.parquet"
val_files="[${data_base}/${data_name}/dev.parquet]"


CONFIG_PATH="../../recipe/appworld/config"

server_url_config_folder="${RUNTIME_ROOT}/appworld_urls"



actor_pp=1
actor_tp=8
actor_cp=2

ref_pp=${actor_pp}
ref_tp=${actor_tp}
ref_cp=${actor_cp}

rollout_tp=8

ray job submit --address="${RAY_ADDRESS}" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='appworld_agent_megatron.yaml' \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.train_batch_size=90 \
    data.val_batch_size=90 \
    data.max_prompt_length=4096 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=90 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp} \
    actor_rollout_ref.actor.megatron.seed=42 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ref_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ref_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${ref_cp} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.response_length=32768 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=50 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.appworld_custom_config.server_url_config_folder=${server_url_config_folder} \
    actor_rollout_ref.appworld_custom_config.sparse_reward=True \
    actor_rollout_ref.appworld_custom_config.max_completion_tokens=2048 \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.megatron.use_remove_padding=True \
    actor_rollout_ref.ref.megatron.use_remove_padding=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.model.use_remove_padding=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.default_local_dir="${output_dir}" \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.rollout_data_dir="${output_dir}/rollout_exps/rollout_log" \
    trainer.validation_data_dir="${output_dir}/val_exps/validation_log" \
    trainer.total_epochs=90
