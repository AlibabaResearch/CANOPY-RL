# CANOPY

[中文说明](README.zh-CN.md)

CANOPY (Coverage-ANchored On-PolicY RL) contains reinforcement-learning
recipes for long-horizon interactive LLM agents. It accompanies
[“Explore More, Drift Less: Outcome-Only Reinforcement Learning Can Suffice
for Long-Horizon Interactive Agents”](https://arxiv.org/abs/2609.01245)
(arXiv:2609.01245) and currently covers AppWorld and SWE-style software
engineering environments.

This is a source release for an existing multi-node Verl/Ray/GPU stack. It is
not a standalone Python package or a single-machine quickstart.

## What is included

| Path | Contents |
| --- | --- |
| `recipe/appworld/` | AppWorld agent loop, data-index builder, and local service |
| `recipe/swe/` | SWE agent loop, synchronous trainer, grading, data/image tools, and tests |
| `run_scripts/appworld/` | One retained training command and three evaluation commands |
| `run_scripts/swe/` | Curated cluster, training, and evaluation commands |
| `verl/` | Paper-command reference snapshot of the upstream Verl Python source |
| `patches/` | Auditable delta for the bundled Verl snapshot |
| `THIRD_PARTY_*` | Source provenance, modifications, and license records |

The repository and public ZIP contain no model weights, checkpoints, Parquet
data, AppWorld protected task/API data, databases, evaluators, generated
benchmark derivatives, SWE container images, trajectories, raw logs,
predictions, credentials, personal paths, or internal cluster identifiers.

## Verl reference and portability

The bundled source is based on:

```text
upstream commit: 19c6af5de10de2b5272c83c0e82aa715c8c621f3
describe:         v0.8.0-11-g19c6af5d
```

This is a Verl 0.8.0 development snapshot, not Verl 0.7.0. The exact delta is
documented in [`patches/README.md`](patches/README.md). It is the reference
layout for the retained commands, but source compatibility alone is not proof
of an end-to-end result replay.

The August Qwen3.6 SWE commands were captured while the SWE runtime was still
being developed. The bundled patch now includes live R2 registry rebinding and
deterministic SGLang NCCL-port reservation, but it does not include the
CP-aware per-token-loss compensation or rejected-native-action credit masking
required by the retained 0811/0817/0819 configurations. Those launchers
therefore fail closed and remain guarded porting references, not claims that
the bundled tree can run the jobs. Port equivalent behavior to the selected
Verl revision and pass a one-batch gradient-norm smoke test before a costly
run. Do not apply the recorded patch blindly to another Verl revision.

Recipes can be ported to another exact Verl checkout that provides equivalent
interfaces. This is best-effort portability, not a claim that arbitrary old,
new, or stock versions work unchanged. See
[`docs/VERL_COMPATIBILITY.md`](docs/VERL_COMPATIBILITY.md) and each recipe's
`REQUIRED_VERL.txt`.

## Environment and launchers

CANOPY intentionally has no root dependency installer or lockfile. The
operator supplies and separately scans the selected Verl, benchmark, CUDA,
model, and container environment. Historical top-level versions are recorded
in [`docs/TESTED_ENVIRONMENT.md`](docs/TESTED_ENVIRONMENT.md); they are not a
current security baseline or a compatibility range.

Each launcher contains its own paths and experiment parameters; there is no
shared `common.sh`. Before running a file, edit its `MODEL_ROOT` or
`MODEL_PATH`, `DATA_ROOT`, `OUTPUT_ROOT`, `RAY_ADDRESS`, and topology values.
All paths consumed by remote Ray workers must be accessible on those workers.

- [AppWorld training/evaluation guide](run_scripts/appworld_readme.md) ·
  [中文](run_scripts/appworld_readme.zh-CN.md)
- [SWE data/image/cluster/training guide](run_scripts/swe_readme.md) ·
  [中文](run_scripts/swe_readme.zh-CN.md)

## AppWorld data

Generate `train`, `dev`, `test_normal`, and `test_challenge` metadata locally
from an authorized AppWorld installation. Training and evaluation launchers
reference these split names directly.

The exact system prompt used in the paper experiment is bundled in
[`recipe/appworld/env_server/prompts.py`](recipe/appworld/env_server/prompts.py).
It is based on AppWorld's public legacy ReAct prompt at revision `ba33afb...`
under Apache-2.0; the paper-experiment form omits the final task placeholder
because the service sends each task as a separate user message. Set
`CANOPY_APPWORLD_PROMPT_FILE` only to override this default with another local
template. The prompt alone does not make the overall run an exact replay; see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## SWE workflow and data boundary

The SWE guide covers the complete public workflow:

1. generate image-download manifests and save Docker images as local archives;
2. build node-routed Parquet for ReBench V1/V2 and the supported evaluations;
3. start Ray groups with node-local Podman storage;
4. preload only images referenced by the selected Parquet files;
5. submit a self-contained training or evaluation launcher.

Supported evaluation integrations are SWE-bench Verified, SWE-bench
Multilingual, ReBench leaderboard, and SWE-bench Pro. The six independent
evaluation launchers currently target Verified; the guarded 0817 and 0819
training configurations record four-benchmark joint validation.

## Reproducibility and security

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for external artifacts
and known boundaries. AppWorld and SWE execute model-generated or benchmark
code; use isolated non-production workers, no secrets, restricted egress,
firewalled Ray services, and disposable storage.

Public archives are built from a clean Git commit by
`tools/build_public_release.sh`, allowing the scanned commit and ZIP SHA-256
to be matched.

## Citation

If you use CANOPY, please cite the associated paper:

```bibtex
@misc{pu2026explore,
  title         = {Explore More, Drift Less: Outcome-Only Reinforcement Learning Can Suffice for Long-Horizon Interactive Agents},
  author        = {Pu, Liming and Li, Xiaoxia and Liu, Yifu and Cao, Teng and Yang, Bin},
  year          = {2026},
  eprint        = {2609.01245},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2609.01245},
  url           = {https://arxiv.org/abs/2609.01245}
}
```

Machine-readable citation metadata is available in
[`CITATION.cff`](CITATION.cff).

## License

CANOPY-original code is Apache-2.0. Bundled and modified third-party source
retains its upstream license. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE),
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and
[`THIRD_PARTY_COMPONENTS.yml`](THIRD_PARTY_COMPONENTS.yml).

## Acknowledgements

CANOPY incorporates and modifies open-source software and interfaces with
public research benchmarks. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`THIRD_PARTY_COMPONENTS.yml`](THIRD_PARTY_COMPONENTS.yml) for upstream
provenance, modifications, and license details.
