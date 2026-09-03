# CANOPY

[English](README.md)

CANOPY（Coverage-ANchored On-PolicY RL）提供面向长时程交互式 LLM Agent
的强化学习 Recipe，配套论文
[“Explore More, Drift Less: Outcome-Only Reinforcement Learning Can Suffice
for Long-Horizon Interactive Agents”](https://arxiv.org/abs/2609.01245)
（arXiv:2609.01245）。目前涵盖 AppWorld 和 SWE 类软件工程环境。

这是面向现有多节点 Verl/Ray/GPU 栈的源码发布，不是独立 Python 包或单机
Quickstart。

## 开源内容

| 路径 | 内容 |
| --- | --- |
| `recipe/appworld/` | AppWorld Agent Loop、数据索引生成与本地服务 |
| `recipe/swe/` | SWE Agent Loop、同步 Trainer、评测、数据/镜像工具与测试 |
| `run_scripts/appworld/` | 1 个保留训练命令和 3 个评测命令 |
| `run_scripts/swe/` | 精选的集群、训练与评测命令 |
| `verl/` | 论文命令参考用的上游 Verl Python 源码快照 |
| `patches/` | 内置 Verl 快照的可审计差异 |
| `THIRD_PARTY_*` | 第三方来源、修改与许可证记录 |

仓库和公开 ZIP 不包含模型权重、Checkpoint、Parquet 数据、AppWorld 受保护的
任务/API 数据、数据库、Evaluator、Benchmark 衍生物、SWE 容器镜像、轨迹、
原始日志、预测结果、凭据、个人路径或内部集群标识。

## Verl 参考版本与可移植性

内置源码基于：

```text
upstream commit: 19c6af5de10de2b5272c83c0e82aa715c8c621f3
describe:         v0.8.0-11-g19c6af5d
```

这是 Verl 0.8.0 的开发快照，不是 0.7.0。准确差异见
[`patches/README.md`](patches/README.md)。它是保留命令的参考源码布局，但仅有
源码兼容性并不能证明端到端结果复现。

8 月的 Qwen3.6 SWE 命令形成时，SWE Runtime 仍在开发。当前内置 Patch 已包含
运行中 R2 Registry 重绑定和确定性的 SGLang NCCL 端口预留，但仍不包含
0811/0817/0819 配置所需的支持 CP 的逐 Token Loss 补偿和 Native Action
解析失败 Credit Mask。因此这些脚本会 Fail Closed，仍只是带保护的移植参考，
并不声称内置源码可以直接运行它们。必须先把等价能力移植到所选 Verl Revision，
并完成单 Batch Gradient-norm Smoke Test 后再启动大任务。也不能把记录的 Patch
盲目套到其他 Verl Revision。

Recipe 可以移植到提供等价接口的其他准确 Verl Commit，但这是 Best-effort
适配，并不表示任意旧版、新版或 Stock Verl 能不经修改运行。边界见
[`docs/VERL_COMPATIBILITY.md`](docs/VERL_COMPATIBILITY.md) 及两个 Recipe 的
`REQUIRED_VERL.txt`。

## 环境与运行脚本

CANOPY 有意不提供根依赖安装器或 Lockfile。操作者需要提供并单独扫描所选择的
Verl、Benchmark、CUDA、模型与容器环境。历史顶层版本记录在
[`docs/TESTED_ENVIRONMENT.md`](docs/TESTED_ENVIRONMENT.md)，它不是当前安全
基线或兼容版本范围。

每个 Launcher 都在文件内包含自己的路径和实验参数，不再依赖 `common.sh`。
运行前直接修改其中的 `MODEL_ROOT` 或 `MODEL_PATH`、`DATA_ROOT`、
`OUTPUT_ROOT`、`RAY_ADDRESS` 及拓扑参数。远端 Ray Worker 使用的路径必须能在
对应 Worker 上访问。

- [AppWorld 训练/评测说明](run_scripts/appworld_readme.zh-CN.md) ·
  [English](run_scripts/appworld_readme.md)
- [SWE 数据/镜像/集群/训练说明](run_scripts/swe_readme.zh-CN.md) ·
  [English](run_scripts/swe_readme.md)

## AppWorld 数据

请从已获授权的 AppWorld 安装在本地生成 `train`、`dev`、`test_normal` 和
`test_challenge` 元数据。训练与评测脚本直接引用这些切分名称。

论文实验使用的完整系统 Prompt 已内置于
[`recipe/appworld/env_server/prompts.py`](recipe/appworld/env_server/prompts.py)。
它基于 AppWorld 在 `ba33afb...` 版本公开的 legacy ReAct Prompt，并遵循
Apache-2.0；论文实验版本移除了末尾的任务占位符，因为服务会把具体任务另作一条
User Message 发送。`CANOPY_APPWORLD_PROMPT_FILE` 只用于可选地覆盖该默认模板。
仅有该 Prompt 并不代表整个运行能够精确重放，详见
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。

## SWE 流程与数据边界

SWE 指南覆盖完整公开流程：

1. 生成镜像下载任务，将 Docker 镜像保存成本地归档；
2. 为 ReBench V1/V2 及支持的评测集生成按节点路由的 Parquet；
3. 使用节点本地 Podman 存储启动 Ray Group；
4. 只预加载所选 Parquet 引用的镜像；
5. 提交文件内自包含的训练或评测 Launcher。

支持的评测集成包括 SWE-bench Verified、SWE-bench Multilingual、ReBench
Leaderboard 和 SWE-bench Pro。当前 6 个独立评测脚本都针对 Verified；带保护的
0817 和 0819 训练配置记录了四个 Benchmark 的联合验证。

## 复现与安全

外部材料和已知边界见
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。AppWorld 与 SWE 会执行
模型生成代码或 Benchmark 代码；请使用隔离的非生产 Worker、无密钥环境、
受限出网、受防火墙保护的 Ray 服务和可丢弃存储。

公开压缩包只由 `tools/build_public_release.sh` 从干净 Git Commit 构建，便于将
扫描 Commit 与 ZIP SHA-256 对应。

## 引用

如果使用 CANOPY，请引用配套论文：

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

机器可读的引用信息见 [`CITATION.cff`](CITATION.cff)。

## 许可证

CANOPY 原创代码采用 Apache-2.0；内置或修改的第三方源码继续遵循上游许可证。
详见 [`LICENSE`](LICENSE)、[`NOTICE`](NOTICE)、
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与
[`THIRD_PARTY_COMPONENTS.yml`](THIRD_PARTY_COMPONENTS.yml)。

## 致谢

CANOPY 使用并修改了开源软件，同时对接公开研究 Benchmark。上游来源、修改和
许可证详情见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与
[`THIRD_PARTY_COMPONENTS.yml`](THIRD_PARTY_COMPONENTS.yml)。
