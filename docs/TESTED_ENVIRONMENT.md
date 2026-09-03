# Tested environment record

This document preserves the top-level values that were previously recorded
during CANOPY paper execution and release preparation. It is a top-level
environment record, **not** an SBOM, a standalone CANOPY installation manifest,
a minimum-version claim, or a promise that these versions work on every CUDA
stack. The scope column distinguishes paper-environment observations from later
release-review records.

CANOPY runs inside an operator-provided verl environment. The public source
archive does not install or redistribute Python wheels, CUDA libraries, GPU
kernels, models, datasets, or containers. Operators using another verl checkout
must resolve, scan, and maintain that environment separately.

## Tested verl source

| Item | Recorded value |
| --- | --- |
| Upstream | `https://github.com/verl-project/verl.git` |
| Commit | `19c6af5de10de2b5272c83c0e82aa715c8c621f3` |
| Describe | `v0.8.0-11-g19c6af5d` |
| CANOPY delta | `patches/verl-canopy.patch` |
| Bundled path | `verl/` |

The bundled source snapshot plus the patch are the recorded paper-command
source reference. This records provenance and interface layout; it does not by
itself prove an end-to-end replay of every result. The recipe source may be
ported to another compatible verl checkout, but that is a new, best-effort
compatibility target. See [`VERL_COMPATIBILITY.md`](VERL_COMPATIBILITY.md).

## Recorded top-level versions

| Component | Version | Scope |
| --- | --- | --- |
| Python | 3.12 | Paper/runtime environment |
| accelerate | 1.11.0 | Recorded operator environment |
| codetiming | 1.4.0 | Recorded operator environment |
| datasets | 4.3.0 | Recorded operator environment |
| dill | 0.4.0 | Recorded operator environment |
| FastAPI | 0.135.1 | Recorded AppWorld service environment |
| Hydra Core | 1.3.2 | Recorded operator environment |
| Jinja2 | 3.1.6 | Recorded operator environment |
| loguru | 0.7.3 | Recorded operator environment |
| NumPy | 2.2.6 | Recorded operator environment |
| OmegaConf | 2.3.0 | Recorded operator environment |
| packaging | 25.0 | Recorded operator environment |
| PEFT | 0.15.0 | Recorded operator environment |
| psutil | 7.1.1 | Recorded operator environment |
| torch | 2.9.1 | Operator-provided verl/GPU environment |
| torchdata | 0.11.0 | Operator-provided verl environment |
| Ray | 2.54.0 | Existing multi-node cluster |
| SGLang | 0.5.10.post1 | Tested rollout backend |
| Transformers | 5.3.0 | Tested model stack |
| TensorDict | 0.10.0 | Tested training stack |
| TransferQueue | 0.1.6 | Tested synchronous SWE path |
| MBridge | 0.15.1 | Tested Megatron path |
| Megatron Bridge | 0.4.2 | Tested Megatron path |
| flash-attn | 2.8.3 | CUDA-sensitive tested environment |
| aiohttp | 3.14.3 | Post-run source-review compatibility update; not rerun evidence |
| pandas | 2.3.3 | Local data preparation |
| PyArrow | 22.0.0 | Historical data-preparation environment only |
| pybind11 | 3.0.1 | Recorded operator environment/build tooling |
| Pydantic | 2.12.3 | Recorded operator environment |
| pylatexenc | 2.10 | Recorded bundled-Verl environment |
| PyYAML | 6.0.3 | Recorded operator environment |
| regex | 2025.10.23 | Recorded operator environment |
| tensorboard | 2.16.2 | Recorded operator environment |
| tqdm | 4.67.1 | Recorded operator environment |
| Uvicorn | 0.38.0 | Recorded AppWorld service environment |
| Weights & Biases | 0.22.2 | Recorded operator environment; retained commands use console logging |
| Ruff | 0.14.5 | Release-development tooling only |

CUDA, NCCL, compiled kernels, and transitive packages must match the selected
host driver, image, and verl checkout. Versions in this table are disclosed for
traceability; CANOPY does not automatically install them. A deployment or new
experiment should scan its resolved environment rather than treating this
record as a current security lockfile.

## External benchmark interfaces

The retained recipes were exercised with:

- AppWorld source revision `ba33afb327152803956fdc16f2c3b94a88377453`
  (observed version `0.2.0.dev0`); protected benchmark content is not public.
- SWE-rebench's SWE-bench fork revision
  `71aaff544c63b57943056b05a43271afc475e7b7` (observed version `4.0.3`).

These are compatibility records, not redistributed packages. Obtain them from
their authorized upstream sources and review their terms independently.
