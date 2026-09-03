# AppWorld 训练与评测

[English](appworld_readme.md)

本文说明保留的 AppWorld 数据、环境、训练与预测命令。CANOPY 提供切分元数据
生成和 Agent 集成代码；AppWorld 材料请通过已授权的上游渠道获取，并遵守相应
条款。

## 1. 准备四个切分

使用已获授权的 AppWorld 安装生成运行脚本所需的四个本地 Parquet：

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

输出目录包含 `train.parquet`、`dev.parquet`、`test_normal.parquet` 和
`test_challenge.parquet`。这些文件含有从操作者 AppWorld 安装派生出的任务标识
与切分元数据，不随本仓库分发。除非 AppWorld 条款明确允许预期的分发形式，
请勿再分发这些文件。

## 2. 启动 AppWorld 服务

Rollout Worker 会读取本地服务启动器生成的端点文件。论文实验 Prompt 已内置在
`recipe/appworld/env_server/prompts.py`；它基于 AppWorld 在 `ba33afb...` 版本
公开的 legacy ReAct Prompt，并遵循 Apache-2.0。论文实验版本移除末尾任务占位符，
因为服务会把具体任务另作一条 User Message 发送。
`CANOPY_APPWORLD_PROMPT_FILE` 只用于可选的本地覆盖；不设置时即使用论文 Prompt。
仅有该 Prompt 也不代表整个运行能够精确重放，详见
`docs/REPRODUCIBILITY.md`。例如：

```bash
cd "$CANOPY_ROOT"
export APPWORLD_ROOT=/authorized/path/to/appworld
export APPWORLD_ALLOWED_OUTPUT_ROOT=/path/to/appworld-run-outputs
export APPWORLD_SERVER_URL_DIR=/path/to/shared/canopy-runtime/appworld_urls
export APPWORLD_NUM_SERVERS=8
bash recipe/appworld/env_server/start_server.sh
```

请按 Rollout 并发量启动足够的服务，并让训练进程能够访问所生成的 URL 目录。
AppWorld 会执行模型生成的 Python，因此只能绑定到可信的本地接口，并应在隔离
计算环境中运行。公开启动器默认绑定回环地址。

## 3. 选择运行脚本

每个文件都包含完整的路径和实验参数。运行前请修改模型或 Checkpoint 路径、
数据根目录、输出根目录、`RUNTIME_ROOT`、Ray 地址和集群拓扑。

| 用途 | 脚本 | 保留的行为 |
| --- | --- | --- |
| 训练 | `appworld/train/appworld_grpo_qwen3_14b_8nodes_0118.sh` | Qwen3-14B GRPO 训练；8 个节点，每节点 8 张 NVIDIA H20 GPU，共 64 张 GPU |
| 评测 | `appworld/eval/appworld_eval_qwen3_14b_10nodes_0129.sh` | Step 90 Checkpoint；Test-Normal 与 Test-Challenge；长上下文 YaRN 命令 |
| 评测 | `appworld/eval/appworld_eval_qwen3_14b_8nodes_0712.sh` | Dev、Test-Normal 与 Test-Challenge；4 个验证采样；YaRN 命令 |
| 评测 | `appworld/eval/appworld_eval_qwen3_14b_8nodes_0722_noyarn.sh` | Dev、Test-Normal 与 Test-Challenge；4 个验证采样；32K、无 YaRN 命令 |

保留的 0129 与 0712 命令并不是单一的已解析 Rope Scaling 配置：SGLang Rollout
Override 使用 YaRN factor `4.0`，而 Actor/Model Override 使用 factor `2.0`
（0712 还保留了嵌套 Override 中的 `rope_theta`）。这里选择如实披露历史值，而
不是静默统一。移植任一命令时都要分别验证两个 Backend，也不能把这些命令之间
的差异引用为严格控制变量的 YaRN 消融。

例如：

```bash
cd "$CANOPY_ROOT"
bash run_scripts/appworld/train/appworld_grpo_qwen3_14b_8nodes_0118.sh

# 或先在文件中设置 Checkpoint 和评测路径，再执行：
bash run_scripts/appworld/eval/appworld_eval_qwen3_14b_8nodes_0722_noyarn.sh
```

运行脚本向 Ray 提交任务，并把 verl 的 Rollout/验证材料写入所配置的输出目录。
AppWorld 环境输出写入已授权的配置目录。本次源码发布不包含任何运行结果。

## 结果记录

- 模型许可证、AppWorld 权限以及组织内部披露规则应与相应材料一同管理。
- 每个分数都应同时报告准确的运行脚本、Checkpoint Step、切分、采样次数与
  上下文配置。
