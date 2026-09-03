# AppWorld training and evaluation

[中文说明](appworld_readme.zh-CN.md)

This guide covers the retained AppWorld data, environment, training, and
prediction commands. CANOPY provides split-metadata generation and agent
integration code. Obtain AppWorld materials through their authorized upstream
channel and observe the applicable terms.

## 1. Prepare the four splits

Use an authorized AppWorld installation to generate the four local Parquet
files used by the launchers:

```bash
export CANOPY_ROOT=/path/to/CANOPY
export APPWORLD_ROOT=/authorized/path/to/appworld
export DATA_ROOT=/path/to/canopy-data
cd "$CANOPY_ROOT"

python3 recipe/appworld/appworld_data_process.py \
  --appworld-root "$APPWORLD_ROOT" \
  --output-dir "$DATA_ROOT/appworld" \
  --splits train dev test_normal test_challenge
```

The output directory contains `train.parquet`, `dev.parquet`,
`test_normal.parquet`, and `test_challenge.parquet`. These files contain task
identifiers and split metadata derived from the operator's AppWorld install.
They are not included in this repository. Do not redistribute them unless the
AppWorld terms expressly permit the intended form of redistribution.

## 2. Start the AppWorld services

The rollout workers read endpoint files produced by the local service
launcher. The paper experiment prompt is bundled in
`recipe/appworld/env_server/prompts.py`, based on AppWorld's public legacy
ReAct instructions at revision `ba33afb...` under Apache-2.0. CANOPY removes
the final task placeholder because the service sends the actual task as a
separate user message. `CANOPY_APPWORLD_PROMPT_FILE` is an optional local
override; leave it unset to use the paper prompt. The prompt alone does not
make the overall run an exact replay; see `docs/REPRODUCIBILITY.md`. For
example:

```bash
cd "$CANOPY_ROOT"
export APPWORLD_ROOT=/authorized/path/to/appworld
export APPWORLD_ALLOWED_OUTPUT_ROOT=/path/to/appworld-run-outputs
export APPWORLD_SERVER_URL_DIR=/path/to/shared/canopy-runtime/appworld_urls
export APPWORLD_NUM_SERVERS=8
bash recipe/appworld/env_server/start_server.sh
```

Start enough services for the requested rollout concurrency and make the
generated URL directory visible to the training process. AppWorld executes
model-generated Python, so bind it only to trusted local interfaces and run it
on isolated compute. The public launcher defaults to loopback.

## 3. Select a launcher

All paths and experiment parameters are inside the corresponding file. Update
the model/checkpoint path, data root, output root, `RUNTIME_ROOT`, Ray address,
and topology before running it.

| Purpose | Launcher | Retained behavior |
| --- | --- | --- |
| Train | `appworld/train/appworld_grpo_qwen3_14b_8nodes_0118.sh` | Qwen3-14B GRPO training; 8 nodes with 8 NVIDIA H20 GPUs per node (64 GPUs total) |
| Evaluate | `appworld/eval/appworld_eval_qwen3_14b_10nodes_0129.sh` | Step-90 checkpoint; Test-Normal and Test-Challenge; long-context YaRN command |
| Evaluate | `appworld/eval/appworld_eval_qwen3_14b_8nodes_0712.sh` | Dev, Test-Normal, and Test-Challenge; four validation samples; YaRN command |
| Evaluate | `appworld/eval/appworld_eval_qwen3_14b_8nodes_0722_noyarn.sh` | Dev, Test-Normal, and Test-Challenge; four validation samples; 32K no-YaRN command |

The retained 0129 and 0712 commands are not a single resolved rope-scaling
configuration: their SGLang rollout override uses YaRN factor `4.0`, while the
actor/model override uses factor `2.0` (0712 also places `rope_theta` in the
retained nested override). These historical values are disclosed rather than
silently normalized. Verify both backends when adapting either command and do
not cite their difference as a controlled YaRN ablation.

For example:

```bash
cd "$CANOPY_ROOT"
bash run_scripts/appworld/train/appworld_grpo_qwen3_14b_8nodes_0118.sh

# Or, after setting the checkpoint and evaluation paths in the file:
bash run_scripts/appworld/eval/appworld_eval_qwen3_14b_8nodes_0722_noyarn.sh
```

The launchers submit jobs to Ray and write verl rollout/validation artifacts
under the configured output directory. AppWorld environment outputs are kept
under the configured authorized output root. No run output is part of this
source release.

## Result reporting

- Keep model licenses, AppWorld permissions, and any organization-specific
  disclosure rules with the corresponding artifacts.
- Report the exact launcher, checkpoint step, split, sampling count, and
  context configuration with every score.
