# CANOPY AppWorld recipe

This directory contains the AppWorld agent loop, local environment service,
configuration, and split-metadata builder used by the retained CANOPY commands.
It is designed to run inside a compatible verl environment; it is not a
standalone Python distribution.

## Verl reference and portability

For the recorded paper-command setup, use the bundled `verl/` snapshot; the retained
launchers default to that layout. The recipe source can be ported to another
verl checkout when it provides equivalent interfaces, but compatibility is
best-effort and must be verified for that exact commit. Setting `VERL_ROOT` to
a second checkout by itself is not a supported runtime switch.

The exact tested commit and patch boundary are recorded in
[`REQUIRED_VERL.txt`](REQUIRED_VERL.txt). The patch is not a rolling patch for
arbitrary verl revisions.

## External prerequisites

Obtain AppWorld and its data through the authorized upstream distribution.
CANOPY does not redistribute protected tasks, API data, databases, evaluators,
or generated benchmark derivatives. The environment service uses packages
supplied by the operator's verl/AppWorld environment; reference versions are
disclosed in
[`docs/TESTED_ENVIRONMENT.md`](../../docs/TESTED_ENVIRONMENT.md), not installed
by this repository.

The paper experiment prompt is bundled in `env_server/prompts.py`. It is based
on AppWorld's public legacy ReAct instructions at revision `ba33afb...` under
Apache-2.0. CANOPY removes the upstream template's final
`Task: {{ input_str }}` line because `server.py` sends the actual task as a
separate user message. `CANOPY_APPWORLD_PROMPT_FILE` is an optional local
override; leave it unset to use the paper prompt. The prompt alone does not
make the overall run an exact replay; see `docs/REPRODUCIBILITY.md`.

## Entry points

From the CANOPY root:

```bash
export APPWORLD_ROOT=/authorized/path/to/appworld
export DATA_ROOT=/authorized/path/to/canopy-data
python3 recipe/appworld/appworld_data_process.py \
  --appworld-root "$APPWORLD_ROOT" \
  --output-dir "$DATA_ROOT/appworld"

bash recipe/appworld/env_server/start_server.sh
bash run_scripts/appworld/train/appworld_grpo_qwen3_14b_8nodes_0118.sh
```

The data builder emits task-index metadata locally. Do not publish its generated
Parquet files unless the AppWorld terms expressly permit that redistribution.
See the [AppWorld launcher guide](../../run_scripts/appworld_readme.md) for the
training and evaluation commands.
