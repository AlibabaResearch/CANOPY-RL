# SWE training and evaluation

[中文说明](swe_readme.zh-CN.md)

This guide describes the complete public workflow for the CANOPY SWE recipe:
obtain task data and images, save images as local archives, build node-routed
Parquet files, start a Ray/Podman cluster, preload only the required images,
then train or evaluate. Task data, container images, model weights,
checkpoints, and predictions are **not** distributed with this repository.

## Supported data

| Role | Dataset | `--dataset-name` | Example source | Split | `--data-docker-source` |
| --- | --- | --- | --- | --- | --- |
| Train | ReBench V1 | `rebench_v1` | `nebius/SWE-rebench` | `filtered` | `SWE-rebench` |
| Train | ReBench V2 | `rebench_v2` | `nebius/SWE-rebench-v2` | `train` | `SWE-rebench-v2` |
| Evaluate | SWE-bench Verified | `swe_bench_verified` | `princeton-nlp/SWE-bench_Verified` | `test` | `SWE-bench` |
| Evaluate | SWE-bench Multilingual | `swe_bench_multilingual` | `SWE-bench/SWE-bench_Multilingual` | `test` | `SWE-bench-multilingual` |
| Evaluate | ReBench leaderboard | `rebench_leaderboard` | `nebius/SWE-rebench-leaderboard` | `2026_03` | `SWE-rebench-leaderboard` |
| Evaluate | SWE-bench Pro | `swe_bench_pro` | `ScaleAI/SWE-bench_Pro` | `test` | `SWE-bench-pro` |

Dataset names, splits, image registries, and access conditions can change.
Pin and record a dataset revision, verify the selected schema, and follow the
dataset, repository, and container-image licenses. A local dataset or Parquet
path can be passed to the same tools instead of a hosted dataset ID.

## Prerequisites and local layout

You need a compatible verl/GPU environment, Ray on all nodes, Docker on the
image-download host, Podman on the rollout nodes, authorized dataset/image
access, and a local model or checkpoint. Define operator-owned locations; do
not place large or restricted artifacts in the Git repository:

```bash
export CANOPY_ROOT=/path/to/CANOPY
export IMAGE_ROOT=/data/canopy-swe-images
export DATA_ROOT=/data/canopy-swe-parquet
export VERIFIED_REVISION=replace-with-a-pinned-dataset-revision
mkdir -p "$IMAGE_ROOT" "$DATA_ROOT"
cd "$CANOPY_ROOT"
```

Each retained train/eval launcher is self-contained and has no `common.sh`.
Edit its local model, data, output, Ray-address, and topology values directly
before use.

## 1. Generate and execute image-download tasks

First generate a JSON manifest. This example uses SWE-bench Verified:

```bash
python3 recipe/swe/image_download.py generate \
  --source princeton-nlp/SWE-bench_Verified \
  --revision "$VERIFIED_REVISION" \
  --split test \
  --dataset-name swe_bench_verified \
  --data-docker-source SWE-bench \
  --image-root "$IMAGE_ROOT" \
  --task-file "$IMAGE_ROOT/tasks-swe_bench_verified.json"
```

Repeat `generate` for each required row in the table, keeping
`--dataset-name` identical in the download and data-build commands. For
ReBench V2 and SWE-bench Pro, image names are read from their dataset fields.
Then execute each manifest on a host with Docker access:

```bash
python3 recipe/swe/image_download.py run \
  --task-file "$IMAGE_ROOT/tasks-swe_bench_verified.json" \
  --image-root "$IMAGE_ROOT" \
  --engine docker \
  --max-workers 8
```

`run` accepts archives only under the same explicit `--image-root` used to
generate the manifest. It pulls each image and writes a gzip-compressed Docker
archive whose filename ends in `.tar`, requires a successful `docker save`,
and checks the minimum size and gzip signature before marking the task
successful. Resumed runs skip archives that pass the same lightweight checks;
record registry digests separately when immutable image identity is required.
Image removal is opt-in
through `--remove-image-after-save`; the tool never performs a global image
prune.

## 2. Build training and evaluation Parquet files

Build the Parquet only after the corresponding archives exist. For example:

```bash
python3 recipe/swe/swe_data_process.py build \
  --source princeton-nlp/SWE-bench_Verified \
  --revision "$VERIFIED_REVISION" \
  --split test \
  --dataset-name swe_bench_verified \
  --data-docker-source SWE-bench \
  --image-root "$IMAGE_ROOT" \
  --output "$DATA_ROOT/swe_bench_verified_maxlen6k_group12.parquet" \
  --num-groups 12 \
  --tokenizer /path/to/Qwen3.6-35B-A3B \
  --agent-config recipe/swe/config/swe_agent_qwen3_native_toolcall_nothink.yaml \
  --max-prompt-length 6144
```

Retained filenames encode both prompt filtering and Ray topology, for example
`_maxlen4k_group8` or `_maxlen6k_group12`. Build every file with the
`--max-prompt-length` and `--num-groups` required by its launcher; do not reuse
a routed Parquet across different topologies.

Use the source, split, dataset key, and Docker source from the table for the
other datasets. Additional retained-workflow options are:

- ReBench V2 part 1: `--part-size 10000 --part-index 1 --seed 42`. Parts are
  one-indexed, contain at most `part-size` rows, and the final part may be
  shorter.
- When constructing a training set, `--exclude-instance-ids` can exclude the
  `instance_id` values from pinned evaluation splits. It accepts
  a generated evaluation Parquet directly (reading `extra_info.instance_id`),
  or a JSON string list / one-ID-per-line file, and may be repeated. The
  exclusion happens before deterministic part selection. Verify independently
  that the final train/eval ID intersection is empty.
- ReBench V1 can additionally use `--exclude-repos-from` for stricter
  repository-level separation and `--instance-ids` for an explicit allowlist.
- SWE-bench Pro requires
  `--swe-bench-pro-run-scripts-dir /authorized/path/to/official/per-instance-scripts`;
  the directory must contain `<instance_id>/run_script.sh` and `parser.py`.
  The same absolute path and files must be visible on every Ray worker that can
  receive a Pro group; validation reads them again on the worker, not only on
  the machine that builds the Parquet.
- `--max-count`, `--part-size`, and prompt-length filtering deliberately alter
  the selected population; record them with the experiment.

By default, `swe_data_process.py` omits rows whose expected `.tar` archive is
missing. `--keep-missing-images` disables that filter. For official evaluation,
use a complete authorized image set and verify the final row count; never
report a score on a missing-image-filtered subset as the full benchmark.

The builder first balances repository workloads across `--num-groups`, then
deterministically hashes each instance to one of the assigned groups. It stores
`group_id`, the image tag, and the operator's archive path in `extra_info`.
Generated Parquet therefore contains machine-local paths and must not be added
to the public source package. The `clean` subcommand removes the known image-
archive and Pro-evaluator host-path keys from an existing Parquet; review all
remaining fields separately. Cleaning does not grant redistribution permission.

## 3. Start the Ray and Podman cluster

The retained 12-node layout uses `group_0` on the head and one distinct group
on each worker. First review `run_scripts/swe/cluster/storage.conf` on every
node. Its `graphroot` must point to a large node-local disk, not shared NAS.
Initialize rootful Podman with the standalone script:

```bash
cd "$CANOPY_ROOT"
sudo bash run_scripts/swe/cluster/podman.sh
```

The script installs the adjacent configuration at
`/run/canopy-podman/storage.conf` and validates it with `podman info`; it does
not overwrite `/etc/containers/storage.conf` or start a Podman API service.
The head and worker launchers call it again idempotently and export the same
configuration to Ray.

Start the head as `group_0`. By default its address file is under
`runtime/ray/`; set `HEAD_IP_FILE` to the same shared path on every node when
the repository itself is not shared:

```bash
cd "$CANOPY_ROOT"
sudo bash run_scripts/swe/cluster/start_head.sh

# If the repository is not shared across nodes:
sudo env HEAD_IP_FILE=/shared/canopy/head_ip.txt \
  bash run_scripts/swe/cluster/start_head.sh
```

On a worker named like `mf_dsw_<batch>_<index>`, `start_worker.sh` derives
`GROUP_NUM` from the instance name or DSW agent log. If that name is not
available, set the group and head address manually:

```bash
sudo env HEAD_IP_FILE=/shared/canopy/head_ip.txt \
  bash run_scripts/swe/cluster/start_worker.sh

# Or set both values explicitly:
sudo env GROUP_NUM=3 HEAD_IP=head-node.example \
  bash run_scripts/swe/cluster/start_worker.sh
```

The core worker command is:

```bash
ray start --address="${HEAD_IP}:6379" \
  --resources="{\"group_${GROUP_NUM}\": 1000}"
```

Worker groups must be unique integers from `1` through `11`; `group_0` belongs
to the head. Group count in the Parquet, advertised Ray groups, and
`trainer.nnodes` must agree. The retained 0811/0817/0819 launchers enable the LXCFS
CPU view, so provide `/var/lib/lxcfs` on every node or disable that option.
These scripts require rootful Podman and run `ray stop --force`, which stops
existing Ray work on the node. Podman setup may create `/dev/fuse` and sets its
mode to `0666`, so use dedicated nodes. The dashboard defaults to `127.0.0.1`;
expose it only with an explicit `DASHBOARD_HOST` and appropriate firewalling.

## 4. Preload only the routed images

Pass every Parquet used by the run so the preloader can deduplicate image
archives and send each image only to its assigned Ray group:

```bash
python3 recipe/swe/preload_images.py \
  "$DATA_ROOT/rebench_v1_maxlen6k_group12.parquet" \
  "$DATA_ROOT/rebench_v2_part1_maxlen6k_group12.parquet" \
  "$DATA_ROOT/swe_bench_verified_maxlen6k_group12.parquet" \
  "$DATA_ROOT/swe_bench_multilingual_maxlen6k_group12.parquet" \
  "$DATA_ROOT/rebench_leaderboard_maxlen6k_group12.parquet" \
  "$DATA_ROOT/swe_bench_pro_maxlen6k_group12.parquet" \
  --ray-address auto \
  --engine podman \
  --resource-units 100 \
  --inspect-timeout 60 \
  --load-timeout 3600 \
  --dry-run
```

Every routed worker must be able to read each archive at the same absolute
`--image-root` path recorded in the Parquet. Use shared storage or replicate
the archives to that identical path on the corresponding workers.

Inspect the group summary, then repeat without `--dry-run`. The loader validates
that every required Ray group exists with enough capacity before scheduling.
Per-command timeouts prevent a stuck Podman process from waiting forever. It
then checks whether the tag already exists and loads the local archive when
needed; a missing archive or failed load fails the command. With a group
capacity of 1,000, the default `--resource-units 100` limits each node to ten concurrent
loads. Preloading makes rollout startup a container-create operation instead
of a registry pull.

Capacity is workload-specific. As operational reference—not a hardware
requirement—an earlier 9B run preloaded a little over 6,000 images. The later
Qwen3.6-35B-A3B experiment used more than 16,000 images; its 12 nodes consumed
about 3.1 TB of local storage per node. Recalculate from the selected Parquet,
archive sizes, node count, filesystem overhead, and free local disk.

### Optional online container cleanup

The 0819 launcher enables the optional per-node collector in
`recipe/swe/env_server/container_gc.py`. It uses a bounded queue and one worker
per node. Every deletion is fail-closed: the collector resolves one generated
MiniSWE container name, verifies its full 64-character ID and all job/node/
group/role/request ownership labels, and removes only that ID. It never prunes
storage, deletes images, or enumerates unrelated containers. When disabled or
unavailable, the existing container timeout plus `--rm` lifecycle remains the
fallback.

Keep `container_gc_enabled=False` until a small job has passed on the target
Ray/Podman cluster. The unit tests cover queueing, retries, cancellation, and
identity checks, but this source release does not claim a real multi-node
Podman end-to-end validation. Adjust only the documented `container_gc_*`
settings, and keep deletion concurrency at one or two workers per node.

### Optional dependency mirrors

The environment client includes a default-off public HTTPS mirror setup for
package installation inside disposable task containers. Set
`dependency_mirror_enabled=True` in `swe_custom_config`, then keep only the
required `dependency_mirror_*_enabled` ecosystem gates enabled. Existing
package-manager environment variables and user configuration are retained;
the setup does not rewrite files in the checked-out repository. Endpoints are
the public HTTPS addresses published by the
[Alibaba Cloud Open Source Mirror Site](https://developer.aliyun.com/mirror/),
not ECS/VPC-only endpoints.

This switch does not grant egress. The public default remains
`network_mode: none`, so preloaded images continue to run offline. If a selected
task cannot run without downloading a dependency, separately choose a
restricted bridge or custom Podman network on dedicated, disposable workers
with no secrets. Do not enable host networking or disable seccomp merely to use
a mirror. Test the exact image and dataset subset before a distributed run.

## 5. Train or evaluate

### Training launchers

| Launcher | Model/topology | Data represented by the retained command |
| --- | --- | --- |
| `swe/train/swe_qwen35_9b_8nodes_0528_origin_xml_36k.sh` | Qwen3.5-9B, 8 nodes, XML, 36K | ReBench V1 training; Verified validation |
| `swe/train/swe_qwen36_35b_a3b_12nodes_0811_toolcall_nothink_rebenchv1_val4bench.sh` | Qwen3.6-35B-A3B, 12 nodes, native tool call, no thinking; **guarded porting reference** | ReBench V1 training; the retained command lists ReBench leaderboard, Verified, and Multilingual validation files |
| `swe/train/swe_qwen36_35b_a3b_12nodes_0817_toolcall_nothink_rebenchv1v2p1_16k_b120_val4bench.sh` | Qwen3.6-35B-A3B, 12 nodes, native tool call, no thinking; **guarded porting reference** | ReBench V1 + deterministic ReBench V2 part 1 training; joint validation on Verified, Multilingual, ReBench leaderboard, and Pro |
| `swe/train/swe_qwen36_35b_a3b_12nodes_0819_toolcall_nothink_rebenchv1v2p1_16k_gc_offload.sh` | Qwen3.6-35B-A3B, 12 nodes, native tool call, no thinking, optimizer offload and optional online container GC; **guarded porting reference** | Same V1 + V2 part 1 and four-benchmark layout, with 16 GiB task containers and a 300-second action timeout |

The 0817 and 0819 launchers record the four-benchmark joint-validation layout.
They use all four validation Parquet files during training; any final score
still needs its dataset's official protocol and evaluator settings. The 0819
variant additionally enables the bounded per-node collector and Megatron
optimizer offload.

The bundled patch includes live R2 registry rebinding and deterministic SGLang
NCCL-port reservation. It still lacks CP-aware per-token-loss compensation and
rejected-native-action credit masking. The 0811/0817/0819 scripts check the
definition and integration of all required capabilities and exit before Ray
submission when any are absent. Do not bypass this guard; port equivalent
behavior and complete the one-batch gradient-norm smoke test described in
`docs/VERL_COMPATIBILITY.md`.

### Evaluation launchers

The six independent evaluation launchers currently target **SWE-bench
Verified only**. They are retained model/prompt/context variants, not separate
launchers for Multilingual, ReBench leaderboard, or Pro:

| Launcher | Variant |
| --- | --- |
| `swe/eval/swe_eval_qwen35_9b_8nodes_0621_origin_xml_36k.sh` | Qwen3.5-9B, original XML with thinking, 36K, two greedy validation samples |
| `swe/eval/swe_eval_qwen35_9b_8nodes_0628_nothink_xml_64k.sh` | Qwen3.5-9B, no-thinking XML, 64K |
| `swe/eval/swe_eval_qwen35_9b_8nodes_0806_toolcall_nothink_64k.sh` | Qwen3.5-9B, native tool call, no thinking, 64K |
| `swe/eval/swe_eval_qwen35_9b_8nodes_0806_toolcall_think_64k.sh` | Qwen3.5-9B, native tool call, thinking, 64K |
| `swe/eval/swe_eval_qwen36_35b_a3b_8nodes_0806_toolcall_nothink.sh` | Qwen3.6-35B-A3B, native tool call, no thinking |
| `swe/eval/swe_eval_qwen36_35b_a3b_8nodes_0806_toolcall_think.sh` | Qwen3.6-35B-A3B, native tool call, thinking |

These are retained runnable commands, not controlled one-variable ablations.
In particular, some thinking/no-thinking pairs also use different validation
temperature or top-p settings. Preserve and disclose each launcher's actual
sampling overrides instead of attributing their difference only to thinking.

The synchronous trainer initializes a train dataloader before honoring
`trainer.val_only=True`. To avoid an unrelated ReBench-file prerequisite, each
public eval launcher points that unused loader at the same Verified Parquet.
No optimizer step is run; the file is evaluated only through `data.val_files`.

The bundled rollout server includes an opt-in deterministic SGLang NCCL-port
reservation. The retained standalone evaluation scripts do not enable it. For
many colocated rollout replicas, validate the candidate port range and enable
the reviewed reservation path or provide another operator-controlled startup
mitigation.

After editing the paths in a launcher, run it from the repository root:

```bash
bash run_scripts/swe/train/swe_qwen35_9b_8nodes_0528_origin_xml_36k.sh

# Or a Verified-only prediction/evaluation command:
bash run_scripts/swe/eval/swe_eval_qwen35_9b_8nodes_0806_toolcall_nothink_64k.sh

# Inspect the guarded current Qwen3.6 configuration (it exits until ported):
bash run_scripts/swe/train/swe_qwen36_35b_a3b_12nodes_0819_toolcall_nothink_rebenchv1v2p1_16k_gc_offload.sh
```

Training uses `recipe.swe.main_ppo_sync`; evaluation launchers set
`trainer.val_only=True`. Verl writes rollout and validation artifacts under the
configured output root. Preserve the exact dataset revision, selected row
count, image set, checkpoint, launcher, sampling parameters, timeouts, and
official evaluator version with every reported result.

## Step diagnostics

`recipe/swe/step_diagnostics.py` exports tracker layout version 6 with separate
training and validation metric prefixes. This layout renames the earlier
SwanLab keys, so do not splice old and new curves by name without an explicit
mapping. Its displayed `fake_rate` is a tracker classification that includes
evaluation failures and missing diagnostics; the raw runtime `AgentData.is_fake`
classification remains separately available in the diagnostics JSON.

## Isolation and security

SWE tasks execute untrusted repository code inside benchmark containers. Use
dedicated, non-production workers with least privilege, no secrets, restricted
egress, firewalled Ray services, and disposable storage. The public environment
configuration defaults container networking to `none` and uses preloaded images
(`--pull never`) during rollout. Review every image and any dataset-supplied Pro
script/parser before execution. Do not mount credentials, source workspaces,
or sensitive host paths into task containers.
