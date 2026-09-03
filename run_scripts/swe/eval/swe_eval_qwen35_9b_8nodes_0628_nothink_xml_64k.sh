#!/usr/bin/env bash
set -euo pipefail

# The archived command named recipe.swe.main_ppo; this retained launcher uses
# the current synchronous entry point, recipe.swe.main_ppo_sync.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CANOPY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
VERL_ROOT="${CANOPY_ROOT}"

MODEL_ROOT="/path/to/models"
DATA_ROOT="/path/to/verl_data"
OUTPUT_ROOT="/path/to/outputs"
RAY_ADDRESS="http://127.0.0.1:8265"

export CANOPY_ROOT VERL_ROOT
cd "${CANOPY_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export RAY_DEDUP_LOGS=0
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export RAY_ENABLE_OPEN_TELEMETRY=0


ulimit -n 65535

project_name="swe_eval"
model_base="${MODEL_ROOT}"


model_name="Qwen3.5-9B"
model_path=${model_base}/${model_name}

entropy_coeff=0
kl_loss_coef=0.01
nnodes=8

# 在原始xml基础上，做了些prompt优化。可以使用思考，检查提交，禁止解释markdown等内容，严格输出action，同时也调整了参数。64k
agent_config_path="recipe/swe/config/swe_agent_xml_config_nothink.yaml"
CONFIG_PATH="config"

output_base="${OUTPUT_ROOT}"
data_base="${DATA_ROOT}"
vbench_base="${data_base}"

val_fname="swe_bench_verified_maxlen4k_group8"
val_files="[${vbench_base}/${val_fname}.parquet]"
# val_only still initializes a train dataloader. Reuse the Verified metadata
# for that unused loader instead of requiring unrelated ReBench training data.
train_files="${val_files}"


actor_pp=2
actor_tp=4
actor_cp=1

ref_pp=${actor_pp}
ref_tp=${actor_tp}
ref_cp=${actor_cp}

# rollout_tp=${actor_tp}
rollout_tp=2

bs=64
ppo_mini_bs=${bs}
ppo_micro_batch_size_per_gpu=1
val_bs=168
# val_bs=200
rolloutn=16
valn=4

use_sparse_reward=True

respk=64
response_length=$((1024 * ${respk}))
prompt_length=$((1024 * 4))
max_assistant_turns=0

exp_name="0628_qwen35_9b_eval_swev_nothink_resp64k"
# rollout env 启动时间
env_start_timeout=360
# 单步action执行时间
action_execute_timeout=120
# 评估时env启动时间
eval_env_start_timeout=360
# 执行评估的时间
# eval_timeout=480
eval_timeout=500
# 每个容器启动前sleep的最大时间
max_env_start_sleep_seconds=60
# 一次rollout过程中，step的最多超时时间，超过则丢弃该数据
max_env_timeout_cnt=3
# 整个轨迹交互最多时间，超时则丢弃
max_rollout_trajectory_timeout=5400


env_resource_tokens=3.5
env_cpu_limit="2"
env_mem_limit="6g"
ray_env_actor_num_cpus=0.01


# 其他默认都先试试-0.2
default_terminated_reward_score=-0.2
default_timeout_reward_score=-0.2
default_no_patch_reward_score=-0.2
default_apply_patch_failed_reward_score=-0.2
default_no_eval_reward_score=0

output_dir="${output_base}/${project_name}/${exp_name}"
use_remove_padding=False

ray job submit --address="${RAY_ADDRESS}" \
    --working-dir="${CANOPY_ROOT}" \
    --runtime-env-json '{
        "env_vars": {
            "RAY_SCHEDULING_STRATEGY": "STRICT_SPREAD",
            "RAY_OP_RESOURCES_STRICT_SPREAD": "1",
            "PYTHONHASHSEED": "0",
            "HYDRA_FULL_ERROR": "1",
            "CANOPY_ROOT": ".",
            "VERL_ROOT": "."
        }
    }' \
    --no-wait \
    -- \
    python3 -m recipe.swe.main_ppo_sync \
    --config-path="$CONFIG_PATH" \
    --config-name='swe_agent_megatron.yaml' \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.train_batch_size=${bs} \
    data.val_batch_size=${val_bs} \
    data.validation_shuffle=True \
    data.max_prompt_length=${prompt_length} \
    data.max_response_length=${response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.optim.lr=4e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_bs} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
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
    actor_rollout_ref.rollout.response_length=${response_length} \
    actor_rollout_ref.rollout.prompt_length=${prompt_length} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${rolloutn} \
    actor_rollout_ref.rollout.val_kwargs.n=${valn} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.swe_custom_config.agent_config_path=${agent_config_path} \
    actor_rollout_ref.swe_custom_config.enable_thinking=False \
    actor_rollout_ref.swe_custom_config.env_start_timeout=${env_start_timeout} \
    actor_rollout_ref.swe_custom_config.action_execute_timeout=${action_execute_timeout} \
    actor_rollout_ref.swe_custom_config.eval_env_start_timeout=${eval_env_start_timeout} \
    actor_rollout_ref.swe_custom_config.eval_timeout=${eval_timeout} \
    actor_rollout_ref.swe_custom_config.max_env_start_sleep_seconds=${max_env_start_sleep_seconds} \
    actor_rollout_ref.swe_custom_config.max_env_timeout_cnt=${max_env_timeout_cnt} \
    actor_rollout_ref.swe_custom_config.env_resource_tokens=${env_resource_tokens} \
    actor_rollout_ref.swe_custom_config.env_cpu_limit=${env_cpu_limit} \
    actor_rollout_ref.swe_custom_config.env_mem_limit=${env_mem_limit} \
    actor_rollout_ref.swe_custom_config.ray_env_actor_num_cpus=${ray_env_actor_num_cpus} \
    actor_rollout_ref.swe_custom_config.use_sparse_reward=${use_sparse_reward} \
    actor_rollout_ref.swe_custom_config.default_timeout_reward_score=${default_timeout_reward_score} \
    actor_rollout_ref.swe_custom_config.default_no_patch_reward_score=${default_no_patch_reward_score} \
    actor_rollout_ref.swe_custom_config.default_apply_patch_failed_reward_score=${default_apply_patch_failed_reward_score} \
    actor_rollout_ref.swe_custom_config.default_terminated_reward_score=${default_terminated_reward_score} \
    actor_rollout_ref.swe_custom_config.default_no_eval_reward_score=${default_no_eval_reward_score} \
    actor_rollout_ref.swe_custom_config.max_rollout_trajectory_timeout=${max_rollout_trajectory_timeout} \
    actor_rollout_ref.swe_custom_config.map_testbed_to_tmpfs=True \
    actor_rollout_ref.swe_custom_config.drop_action_parse_failed_assistant_message=False \
    actor_rollout_ref.actor.megatron.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.ref.megatron.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnodes} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_local_dir="${output_dir}" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.rollout_data_dir="${output_dir}/rollout_exps/rollout_log" \
    trainer.validation_data_dir="${output_dir}/val_exps/validation_log" \
    trainer.total_epochs=3
