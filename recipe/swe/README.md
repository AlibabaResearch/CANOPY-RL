# CANOPY SWE recipe

This directory contains CANOPY's SWE agent loop, synchronous verl entry point,
node-aware image routing and preloading tools, Podman environment client,
SWE-bench/Pro grading integration, and data builder. It is designed for an
operator-managed verl/GPU environment; it is not a standalone Python package.

For the end-to-end data, image, cluster, training, and evaluation workflow, see
the [SWE run guide](../../run_scripts/swe_readme.md) or its
[Chinese version](../../run_scripts/swe_readme.zh-CN.md).

## Verl reference and portability

- The bundled `verl/` snapshot is the audited source-layout reference for the
  retained commands; it is not an end-to-end result-reproduction claim.
- The recipe can be ported to another verl checkout, but
  `recipe.swe.main_ppo_sync` depends on verl's internal APIs. Adapt and test it
  against the exact target commit rather than treating `VERL_ROOT` as a
  universal compatibility switch.

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) and
[`docs/VERL_COMPATIBILITY.md`](../../docs/VERL_COMPATIBILITY.md). The recorded
patch applies only to its stated upstream commit.

## External prerequisites

The operator supplies a compatible verl environment, Ray, Docker/Podman,
legally obtained SWE datasets and images, model/checkpoint files, and any
official SWE-bench Pro per-instance evaluator scripts. Those evaluator files
must be available at the same absolute path on every eligible Ray worker.
CANOPY neither installs
their full dependency graph nor redistributes task data, images, models,
checkpoints, generated Parquet, or predictions. Reference top-level versions
are disclosed in
[`docs/TESTED_ENVIRONMENT.md`](../../docs/TESTED_ENVIRONMENT.md).

## Main entry points

From the CANOPY root:

```bash
# Generate a task manifest, then pull and archive its images.
python3 recipe/swe/image_download.py generate --help
python3 recipe/swe/image_download.py run --help

# Build a verl-compatible, group-routed Parquet.
python3 recipe/swe/swe_data_process.py build --help

# Preview and then preload only the images referenced by selected Parquet files.
python3 recipe/swe/preload_images.py /path/to/data.parquet --dry-run

# Inspect the guarded four-benchmark configuration; it exits until the
# required Qwen3.6 runtime capabilities have been ported.
bash run_scripts/swe/train/swe_qwen36_35b_a3b_12nodes_0819_toolcall_nothink_rebenchv1v2p1_16k_gc_offload.sh
```

The Qwen3.6 0811/0817/0819 launchers are parameter records and fail-closed porting
references, not runnable claims for the bundled Verl snapshot. They check the
required loss scaling, R2 registry, SGLang port-reservation, and rejected-action
masking integrations before Ray submission. The older 0528 launcher and the
evaluation-only launchers do not perform the affected CP=4 per-token-loss
backward pass. See `docs/VERL_COMPATIBILITY.md` before adapting any Qwen3.6
training command.

`swe_data_process.py` filters missing image archives by default and assigns
each retained instance to a deterministic Ray `group_id`. Generated Parquet
records operator-local archive paths for preloading; keep it outside the source
repository and do not redistribute it without permission. For public reruns,
repeat `--exclude-instance-ids` with every evaluation Parquet when building
training data, then independently confirm the train/eval ID intersection is
empty.

The step diagnostics exporter currently uses tracker layout version 6. Its
phase-specific SwanLab keys are intentionally not backward-compatible with the
earlier layout, and its displayed `fake_rate` is a tracker classification rather
than a direct alias for raw `AgentData.is_fake`. The raw classification remains
available in the persisted diagnostics JSON.

## Optional dependency mirrors

`env_server/dependency_mirrors.py` can configure public HTTPS mirrors for
package managers inside disposable SWE containers. It covers APT/Alpine plus
language-scoped Python, Go, Node.js, PHP, R, Ruby, JVM, Cargo, and Rustup
settings. The global `dependency_mirror_enabled` switch defaults to `False` and
is a true no-op; each ecosystem also has its own gate. Existing image or
operator configuration is preserved, and the setup never modifies files in the
checked-out task repository. The bundled endpoints come from the
[Alibaba Cloud Open Source Mirror Site](https://developer.aliyun.com/mirror/)
and use its public, not ECS/VPC-only, addresses.

This option does not enable network access. The public container default remains
`network_mode: none`. If a task must install missing dependencies, opt in to a
restricted bridge or custom network separately on dedicated, disposable workers
with no credentials, then enable only the required ecosystem gates. Public
mirror availability and package integrity remain the operator's responsibility.

## Optional online container cleanup

`env_server/container_gc.py` provides a bounded per-node Podman collector for
high-concurrency SWE jobs. It is disabled by default. When enabled, the client
assigns each managed container a generated name plus job, node, group, role,
nonce, and request labels. The collector resolves one exact container, verifies
all labels and the full 64-character ID, and only then removes that ID. It does
not enumerate containers, prune storage, remove images, or manipulate mounts.

Enable it only on a dedicated cluster after a small Ray/Podman smoke test. The
queue, deletion timeout, retry policy, drain timeout, and one-or-two workers per
node are controlled by the `container_gc_*` fields in
`config/swe_agent_megatron.yaml`. The retained 0819 launcher shows the enabled
configuration. Unit tests cover the fail-closed identity and queue behavior;
the public release does not claim a real multi-node Podman end-to-end test.

## Security boundary

SWE evaluation executes untrusted repository code. Use dedicated workers with
least privilege, no secrets, restricted egress, and disposable storage. The
public environment defaults container networking to `none` and rollout image
pulling to `never`; review dataset-supplied images and Pro evaluator scripts
before use.
