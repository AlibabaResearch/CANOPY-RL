# Third-party notices

CANOPY's original code is licensed under Apache-2.0. The following third-party
components are redistributed as source, represented by a patch, or documented
as external benchmark interfaces. This inventory makes the source-release SCA
findings auditable; it does not replace upstream licenses or an independent scan
of the operator's runtime environment.

| Component | Version / revision | Distribution in this repository | License | Modified |
| --- | --- | --- | --- | --- |
| Verl | `19c6af5de10de2b5272c83c0e82aa715c8c621f3` | Tested modified package-source snapshot, audit patch, and derived SWE trainer entry point | Apache-2.0 | Yes |
| PyTorch copied portions | v2.7.0 | `verl/third_party/torch/` through upstream verl | BSD-3-Clause | No CANOPY changes |
| AppWorld | `ba33afb327152803956fdc16f2c3b94a88377453` | Runtime API integration and modified public legacy ReAct prompt | Apache-2.0 plus protected-content rules | Yes |
| mini-swe-agent | v2.2.6, `e68906fa7cd9f859b1cf483b685284285a379865` | Modified action, environment, configuration, and prompt files | MIT | Yes |
| SWE-bench | v4.1.0, `726c5461e2ef52d83cf1ea2107870a8bb3328d57` | Two modified grading/test-spec files | MIT | Yes |
| SWE-rebench SWE-bench fork | `71aaff544c63b57943056b05a43271afc475e7b7` | External prerequisite only | MIT | Not in this repository |

The machine-readable record, including every distributed path and modification
summary, is [`THIRD_PARTY_COMPONENTS.yml`](THIRD_PARTY_COMPONENTS.yml).

## Verl-derived source

CANOPY bundles the tested upstream `verl/` Python package source so the retained
launchers keep their recorded import/layout assumptions. A unified audit patch
records nine CANOPY content modifications plus one filename normalization
(removal of a trailing U+200E format character) and is documented in
[`patches/README.md`](patches/README.md). In addition,
`recipe/swe/main_ppo_sync.py` is a modified copy of the same-revision Verl
synchronous trainer entry point with SWE diagnostics, dense/nested dump
handling, and optional container-GC lifecycle management;
`recipe/swe/hermes_tool_parser.py` adapts the same snapshot's Hermes parser for
strict SWE diagnostics. Both retain the ByteDance Apache-2.0 header. Two
focused tests cover the added R2 registry and port-reservation behavior.
The Apache-2.0 license and upstream notice are retained under
`THIRD_PARTY_LICENSES/verl/`.

This is a tested package-source snapshot, not a complete upstream repository
checkout or a claim that every Verl revision is compatible. The recipe source
may be ported to another exact commit after separate interface and runtime
validation; the recorded patch must not be applied blindly to that revision.

The bundled upstream snapshot itself contains code adapted from other projects.
In particular, `verl/third_party/torch/` copies PyTorch 2.7.0 distributed
checkpoint code under BSD-3-Clause; its copyright headers are preserved and
the license is at `THIRD_PARTY_LICENSES/PyTorch/LICENSE`. Other dependencies and
source-similarity findings must be resolved in the SCA report for the exact
release commit rather than inferred from package names alone.

## AppWorld-derived prompt

`recipe/appworld/env_server/prompts.py` bundles the public AppWorld legacy
ReAct code-agent instructions from revision `ba33afb...` under Apache-2.0.
CANOPY removes the final `Task: {{ input_str }}` line because `server.py`
sends each actual task as a separate user message, and retains an optional
local template override. The upstream source and full attribution are recorded
in `THIRD_PARTY_LICENSES/AppWorld/NOTICE.md`.

## mini-swe-agent-derived files

The following files are based on mini-swe-agent v2.2.6 and remain under MIT:

- `recipe/swe/env_server/actions_text.py`
- `recipe/swe/env_server/actions_toolcall.py`
- `recipe/swe/env_server/config.py`
- `recipe/swe/env_server/environments.py`
- `recipe/swe/env_server/exceptions.py`
- `recipe/swe/config/swe_agent_qwen35_9b_native_toolcall_nothink.yaml`
- `recipe/swe/config/swe_agent_qwen3_native_toolcall_nothink.yaml`
- `recipe/swe/config/swe_agent_qwen3_native_toolcall_think.yaml`
- `recipe/swe/config/swe_agent_xml_config.yaml`
- `recipe/swe/config/swe_agent_xml_config_nothink.yaml`

Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez. CANOPY changes the
imports, container execution, configuration, prompt formats, and agent/tool-call
representations; adds resource and parsing safeguards, diagnostics, and
fallback identifiers plus an optional public dependency-mirror integration;
and omits the upstream multimodal expansion path. The complete license is at
`THIRD_PARTY_LICENSES/mini-swe-agent/LICENSE.md`.

## SWE-bench-derived files

`recipe/swe/env_server/evaluate.py` is based on SWE-bench's
`swebench/harness/grading.py`; `recipe/swe/env_server/test_spec.py` is based on
`swebench/harness/test_spec/test_spec.py`. The source-code baseline is
SWE-bench v4.1.0 (`726c5461...`). Both remain under MIT.

Copyright (c) 2023 Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao,
Kexin Pei, Ofir Press, Karthik R Narasimhan. CANOPY adds the runtime fork's
install configuration and parser interfaces, richer failure diagnostics,
dense-reward calculation, report fields, and TestSpec construction. The exact
runtime interface is pinned to the SWE-rebench fork commit `71aaff544...`.
The complete license is at `THIRD_PARTY_LICENSES/SWE-bench/LICENSE`.

## Data, models, and containers

No benchmark data, model weights, checkpoints, or container images are
distributed in the public source archive. AppWorld's protected content and
derivatives are subject to additional redistribution rules; this repository
contains none of them. SWE task instances originate in third-party source
repositories and must be obtained and reviewed under their applicable terms.
Model users must review the relevant model card and license separately.

## External execution environment

CANOPY is a source-only recipe release and intentionally distributes no root
Python installation manifest. Runtime packages, GPU libraries, benchmark
packages, models, datasets, and containers are provisioned by the operator and
are not bundled. Versions observed in the retained environment are disclosed
for transparency in [`docs/TESTED_ENVIRONMENT.md`](docs/TESTED_ENVIRONMENT.md);
that record is not an install constraint or security baseline.

The release SCA covers the exact distributed source commit, including the
bundled Verl package source and the third-party-derived files listed above. It
does not resolve the operator-selected Python, accelerator, benchmark, or
container environment. Each deployment must scan and review that resolved
environment independently.
