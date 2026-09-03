#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CANOPY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
VERL_ROOT="${CANOPY_ROOT}"

MODEL_PATH="/path/to/checkpoints/appworld-step-90/actor/huggingface"
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

project_name="appworld_eval"
model_path="${MODEL_PATH}"


entropy_coeff=0
kl_loss_coef=0

# exp_name="0712_qwen3_14b_mean4_dev_normal_challenge_temp06_resp61k_100turn"
# exp_name="0712_0118_step90_mean4_dev_normal_challenge_temp06_resp61k_100turn"
# exp_name="0722_0118_step90_mean4_dev_normal_challenge_temp06_resp32k_50turn"
# exp_name="0724_0118_step90_mean4_dev_normal_challenge_temp06_resp32k_50turn"
# exp_name="07242_0118_step90_mean4_dev_normal_challenge_temp06_resp32k_50turn"
exp_name="07243_0118_step90_mean4_dev_normal_challenge_temp06_resp32k_50turn"

output_base="${OUTPUT_ROOT}"
output_dir="${output_base}/${project_name}/${exp_name}"

data_base="${DATA_ROOT}"
data_name="appworld"
dprefix="${data_base}/${data_name}"
train_files="${data_base}/${data_name}/train.parquet"
val_files="[${data_base}/${data_name}/dev.parquet,${data_base}/${data_name}/test_normal.parquet,${data_base}/${data_name}/test_challenge.parquet]"
# val_files="[${dprefix}/test_normal.parquet,${dprefix}/test_challenge.parquet]"
# val_files="[${dprefix}/test_normal.parquet]"
# val_files="[${dprefix}/dev.parquet,${dprefix}/test_normal.parquet,${dprefix}/test_challenge.parquet,${dprefix}/dev_level1.parquet,${dprefix}/test_normal_level1.parquet,${dprefix}/test_challenge_level1.parquet,${dprefix}/dev_level2.parquet,${dprefix}/test_normal_level2.parquet,${dprefix}/test_challenge_level2.parquet,${dprefix}/dev_level3.parquet,${dprefix}/test_normal_level3.parquet,${dprefix}/test_challenge_level3.parquet]"
# val_files="[${data_base}/${data_name}/dev.parquet,${data_base}/${data_name}/test_normal.parquet]"
# val_files="[${data_base}/${data_name}/test_challenge.parquet]"

experiments_outputs_directory="${output_dir}/appworld_env_outputs"

CONFIG_PATH="../../recipe/appworld/config"

server_url_config_folder="${RUNTIME_ROOT}/appworld_urls"


actor_pp=1
actor_tp=8
actor_cp=2

ref_pp=${actor_pp}
ref_tp=${actor_tp}
ref_cp=${actor_cp}

rollout_tp=8

# export SGLANG_OVERRIDE_ARGS='{"rope_scaling": {"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768}}'
# export SGLANG_OVERRIDE_ARGS='{"rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768, "rope_theta": 1000000}}'
prompt_length=4096
response_length=32768
# context_length=65536

# 100轮
# max_assistant_turns=100
# 0表示不限制轮数，只限制长度，看看是否会有提升
max_assistant_turns=50

# prompt_length=4092
# response_length=61440
# context_length=65536

# prompt_length=4092
# response_length=32768
# context_length=40960

# temperature=0.6
# top_p=0.95
# top_k=20

# actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
# actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
# actor_rollout_ref.rollout.val_kwargs.top_k=20 \



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
    data.val_batch_size=150 \
    data.max_prompt_length=${prompt_length} \
    data.max_response_length=${response_length} \
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
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ref_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ref_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${ref_cp} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.response_length=${response_length} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.appworld_custom_config.server_url_config_folder=${server_url_config_folder} \
    actor_rollout_ref.appworld_custom_config.sparse_reward=True \
    actor_rollout_ref.appworld_custom_config.max_completion_tokens=2048 \
    actor_rollout_ref.appworld_custom_config.experiments_outputs_directory=${experiments_outputs_directory} \
    actor_rollout_ref.appworld_custom_config.rm_outdir_after_finished=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.save_freq=3000 \
    trainer.test_freq=10 \
    trainer.default_local_dir="${output_dir}" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.rollout_data_dir="${output_dir}/rollout_exps/rollout_log" \
    trainer.validation_data_dir="${output_dir}/val_exps/validation_log" \
    trainer.total_epochs=90
