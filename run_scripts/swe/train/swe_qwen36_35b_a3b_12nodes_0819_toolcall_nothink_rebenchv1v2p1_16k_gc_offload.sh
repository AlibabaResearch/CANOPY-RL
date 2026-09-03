#!/usr/bin/env bash
set -euo pipefail

# This later configuration adds bounded per-node container GC, 16 GiB task
# containers, a 300-second action timeout, and Megatron optimizer offload. It
# remains a guarded porting reference until every capability below is verified.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CANOPY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
VERL_ROOT="${CANOPY_ROOT}"

# This latest Qwen3.6 configuration requires remaining correctness capabilities
# that are not present in the bundled verl/ snapshot.
# Fail before Ray submission instead of running numerically incorrect CP=4
# per-token-loss training. A port with equivalent code may adapt these checks.
require_source_marker() {
    local file="$1"
    local marker="$2"
    local label="$3"
    if ! grep -Fq -- "${marker}" "${CANOPY_ROOT}/${file}"; then
        printf 'Missing required Qwen3.6 training capability (%s): %s\n' "${label}" "${file}" >&2
        return 1
    fi
}

verify_qwen36_training_stack() {
    local failed=0
    require_source_marker "verl/workers/engine/megatron/transformer_impl.py" \
        "if not forward_only and self.tf_config.calculate_per_token_loss:" \
        "CP-aware per-token loss guard" || failed=1
    require_source_marker "verl/workers/engine/megatron/transformer_impl.py" \
        "mpu.get_data_parallel_group(with_context_parallel=True).size()" \
        "DP x CP loss normalization" || failed=1
    require_source_marker "verl/workers/engine/megatron/transformer_impl.py" \
        "scaled_loss = scaled_loss / dp_cp_size" \
        "DP x CP scaled-loss compensation" || failed=1
    require_source_marker "verl/utils/megatron/router_replay_patch.py" \
        "def rebuild_router_replay_registry(" \
        "live R2 router registry" || failed=1
    require_source_marker "verl/workers/engine/megatron/transformer_impl.py" \
        "rebuild_router_replay_registry(" \
        "live R2 router registry integration" || failed=1
    require_source_marker "verl/utils/net_utils.py" \
        "def reserve_port(" \
        "SGLang NCCL port reservation" || failed=1
    require_source_marker "verl/workers/rollout/sglang_rollout/async_sglang_server.py" \
        "VERL_SGLANG_RESERVE_NCCL_PORT" \
        "SGLang NCCL reservation integration" || failed=1
    require_source_marker "verl/workers/rollout/sglang_rollout/async_sglang_server.py" \
        "self._nccl_port, self._nccl_sock = reserve_port(" \
        "SGLang NCCL reserved-socket lifetime" || failed=1
    require_source_marker "recipe/swe/config/swe_agent_megatron.yaml" \
        "mask_action_parse_failed_assistant_message:" \
        "native-action credit mask config" || failed=1
    require_source_marker "recipe/swe/swe_agent_loop.py" \
        "self.mask_action_parse_failed_assistant_message" \
        "native-action credit mask runtime" || failed=1
    require_source_marker "recipe/swe/swe_agent_loop.py" \
        "agent_data.response_mask[prev_response_mask_len:] = [0] * (" \
        "native-action generated-token masking" || failed=1
    require_source_marker "recipe/swe/env_server/container_gc.py" \
        "class ContainerGCManager:" \
        "bounded per-node container GC" || failed=1
    require_source_marker "recipe/swe/main_ppo_sync.py" \
        "ContainerGCManager.start(" \
        "container GC lifecycle integration" || failed=1
    require_source_marker "recipe/swe/env_server/swe_env_client.py" \
        "def _configure_container_gc_identity(" \
        "container ownership labels" || failed=1
    if ((failed)); then
        printf '%s\n' \
            "This launcher is a guarded porting reference and is not runnable with bundled verl/." \
            "Port and verify the capabilities in docs/VERL_COMPATIBILITY.md, then run a one-batch grad-norm smoke test." >&2
        exit 2
    fi
}

verify_qwen36_training_stack

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
export RAY_enable_open_telemetry=0

ulimit -n 65535

project_name="swe_agent"

model_path="${MODEL_ROOT}/Qwen3.6-35B-A3B"
agent_config_path="recipe/swe/config/swe_agent_qwen3_native_toolcall_nothink.yaml"
env_config_path="recipe/swe/config/swe_env_config_2h.yaml"
config_path="config"

output_base="${OUTPUT_ROOT}"

swe_rebench_v1_train_file="${DATA_ROOT}/rebench_v1_maxlen6k_group12.parquet"
swe_rebench_v2_part1_train_file="${DATA_ROOT}/rebench_v2_part1_maxlen6k_group12.parquet"
train_files="[${swe_rebench_v1_train_file},${swe_rebench_v2_part1_train_file}]"

swe_bench_verified_val_file="${DATA_ROOT}/swe_bench_verified_maxlen6k_group12.parquet"
swe_bench_multilingual_val_file="${DATA_ROOT}/swe_bench_multilingual_maxlen6k_group12.parquet"
swe_rebench_leaderboard_val_file="${DATA_ROOT}/rebench_leaderboard_maxlen6k_group12.parquet"
swe_bench_pro_val_file="${DATA_ROOT}/swe_bench_pro_maxlen6k_group12.parquet"
val_files="[${swe_bench_verified_val_file},${swe_bench_multilingual_val_file},${swe_rebench_leaderboard_val_file},${swe_bench_pro_val_file}]"

nnodes=12
actor_tp=2
actor_pp=2
actor_ep=4
actor_etp=1
actor_cp=4
rollout_tp=2

train_batch_size=120
# Keep the PPO mini-batch equal to the prompt train batch. With rollout_n=16,
# the actor sees the full 120 x 16 = 1920 on-policy trajectories together.
ppo_mini_batch_size=120

rollout_n=16
val_n=4
val_batch_size=360


respk=64
promptk=6
response_length=$((1024 * ${respk}))
prompt_length=$((1024 * ${promptk}))
max_assistant_turns=200

learning_rate=1e-6
lr_warmup_steps=2
kl_loss_coef=0.01
total_epochs=3
save_freq=20
test_freq=50
eval_timeout=500
# Online monitoring uses a bounded Pro timeout. Set this to 3600 for the
# final, official-style checkpoint evaluation.
swe_bench_pro_eval_timeout=1000
enable_lxcfs_cpu_view=true
# Each Ray group has capacity 1000. 10 caps a node at 100 live ENV actors.
env_resource_tokens=10
val_before_train=True
action_execute_timeout=300
max_env_timeout_cnt=3
max_rollout_trajectory_timeout=3600
repeated_action_warning_threshold=2
repeated_action_termination_threshold=3



exp_name="0819_swe_qwen36_35b_a3b_rebenchv1v2p1_16k_b120_n16_cpu2_mem16g_envtok10_act300_ep3_eval50_gc1_optimizer_offload_proeval${swe_bench_pro_eval_timeout}_val4bench_prompt${promptk}k_resp${respk}k_turns${max_assistant_turns}_interaction3600_repeat3"
output_dir="${output_base}/${project_name}/${exp_name}"

ray job submit --address="${RAY_ADDRESS}" \
    --working-dir="${CANOPY_ROOT}" \
    --runtime-env-json '{
        "env_vars": {
            "RAY_SCHEDULING_STRATEGY": "STRICT_SPREAD",
            "RAY_OP_RESOURCES_STRICT_SPREAD": "1",
            "PYTHONHASHSEED": "0",
            "HYDRA_FULL_ERROR": "1",
            "RAY_enable_open_telemetry": "0",
            "SGLANG_NUMA_BIND_V2": "0",
            "VERL_SGLANG_RESERVE_NCCL_PORT": "1",
            "VERL_SGLANG_NCCL_PORT_BASE": "20000",
            "VERL_SGLANG_NCCL_PORT_STRIDE": "512",
            "VERL_SGLANG_NCCL_PORT_ATTEMPTS": "16",
            "CANOPY_ROOT": ".",
            "VERL_ROOT": "."
        }
    }' \
    --no-wait \
    -- \
    python3 -m recipe.swe.main_ppo_sync \
    --config-path="${config_path}" \
    --config-name=swe_agent_megatron.yaml \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.val_max_samples=-1 \
    data.seed=42 \
    data.validation_shuffle=True \
    data.max_prompt_length=${prompt_length} \
    data.max_response_length=${response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp} \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.router_replay.mode=R2 \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.cp_comm_type=null \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=true \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=0 \
    actor_rollout_ref.actor.megatron.seed=42 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${actor_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${actor_etp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${actor_cp} \
    actor_rollout_ref.ref.megatron.use_remove_padding=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.response_length=${response_length} \
    actor_rollout_ref.rollout.prompt_length=${prompt_length} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.num_workers=24 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.swe_custom_config.agent_config_path="${agent_config_path}" \
    actor_rollout_ref.swe_custom_config.env_config_path="${env_config_path}" \
    actor_rollout_ref.swe_custom_config.enable_thinking=False \
    actor_rollout_ref.swe_custom_config.env_start_timeout=360 \
    actor_rollout_ref.swe_custom_config.action_execute_timeout="${action_execute_timeout}" \
    actor_rollout_ref.swe_custom_config.eval_env_start_timeout=360 \
    actor_rollout_ref.swe_custom_config.eval_timeout="${eval_timeout}" \
    actor_rollout_ref.swe_custom_config.swe_bench_pro_eval_timeout="${swe_bench_pro_eval_timeout}" \
    actor_rollout_ref.swe_custom_config.max_env_start_sleep_seconds=40 \
    actor_rollout_ref.swe_custom_config.max_env_timeout_cnt="${max_env_timeout_cnt}" \
    actor_rollout_ref.swe_custom_config.repeated_action_warning_threshold="${repeated_action_warning_threshold}" \
    actor_rollout_ref.swe_custom_config.repeated_action_termination_threshold="${repeated_action_termination_threshold}" \
    actor_rollout_ref.swe_custom_config.env_resource_tokens="${env_resource_tokens}" \
    actor_rollout_ref.swe_custom_config.container_gc_enabled=True \
    actor_rollout_ref.swe_custom_config.container_gc_workers_per_node=1 \
    actor_rollout_ref.swe_custom_config.enable_lxcfs_cpu_view=${enable_lxcfs_cpu_view} \
    actor_rollout_ref.swe_custom_config.env_cpu_limit=2 \
    actor_rollout_ref.swe_custom_config.env_mem_limit=16g \
    actor_rollout_ref.swe_custom_config.ray_env_actor_num_cpus=0 \
    actor_rollout_ref.swe_custom_config.use_sparse_reward=True \
    actor_rollout_ref.swe_custom_config.default_timeout_reward_score=-0.2 \
    actor_rollout_ref.swe_custom_config.default_no_patch_reward_score=-0.2 \
    actor_rollout_ref.swe_custom_config.default_apply_patch_failed_reward_score=-0.2 \
    actor_rollout_ref.swe_custom_config.default_terminated_reward_score=-0.2 \
    actor_rollout_ref.swe_custom_config.default_no_eval_reward_score=0 \
    actor_rollout_ref.swe_custom_config.max_rollout_trajectory_timeout="${max_rollout_trajectory_timeout}" \
    actor_rollout_ref.swe_custom_config.reset_git_log=True \
    actor_rollout_ref.swe_custom_config.disable_manual_write_patch_cmd=True \
    actor_rollout_ref.swe_custom_config.map_testbed_to_tmpfs=False \
    actor_rollout_ref.swe_custom_config.drop_action_parse_failed_assistant_message=False \
    ++actor_rollout_ref.swe_custom_config.mask_action_parse_failed_assistant_message=True \
    actor_rollout_ref.swe_custom_config.enable_message_aware_prompt_truncation=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnodes} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=null \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.default_local_dir="${output_dir}" \
    trainer.resume_mode=disable \
    trainer.val_before_train=${val_before_train} \
    trainer.val_only=False \
    trainer.swe_step_diagnostics.enable=True \
    trainer.rollout_data_dir="${output_dir}/rollout_exps/rollout_log" \
    trainer.validation_data_dir="${output_dir}/val_exps/validation_log"
