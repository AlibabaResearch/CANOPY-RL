# SWE 训练与评测

[English](swe_readme.md)

本文说明 CANOPY SWE Recipe 的完整公开流程：获取任务数据和镜像、将镜像保存为
本地归档、生成按节点路由的 Parquet、启动 Ray/Podman 集群、只预加载本次任务
所需镜像，最后训练或评测。本仓库**不**分发任务数据、容器镜像、模型权重、
Checkpoint 或预测结果。

## 支持的数据

| 用途 | 数据集 | `--dataset-name` | 示例来源 | 切分 | `--data-docker-source` |
| --- | --- | --- | --- | --- | --- |
| 训练 | ReBench V1 | `rebench_v1` | `nebius/SWE-rebench` | `filtered` | `SWE-rebench` |
| 训练 | ReBench V2 | `rebench_v2` | `nebius/SWE-rebench-v2` | `train` | `SWE-rebench-v2` |
| 评测 | SWE-bench Verified | `swe_bench_verified` | `princeton-nlp/SWE-bench_Verified` | `test` | `SWE-bench` |
| 评测 | SWE-bench Multilingual | `swe_bench_multilingual` | `SWE-bench/SWE-bench_Multilingual` | `test` | `SWE-bench-multilingual` |
| 评测 | ReBench leaderboard | `rebench_leaderboard` | `nebius/SWE-rebench-leaderboard` | `2026_03` | `SWE-rebench-leaderboard` |
| 评测 | SWE-bench Pro | `swe_bench_pro` | `ScaleAI/SWE-bench_Pro` | `test` | `SWE-bench-pro` |

数据集名称、切分、镜像 Registry 与访问条件可能变化。请固定并记录数据集
Revision，确认所选 Schema，并遵守数据集、代码仓库及容器镜像的许可证。相同
工具也支持传入本地数据集或 Parquet 路径，而不是托管数据集 ID。

## 前置条件与本地目录

需要兼容的 verl/GPU 环境、所有节点上的 Ray、下载节点上的 Docker、Rollout
节点上的 Podman、已授权的数据集/镜像访问权限，以及本地模型或 Checkpoint。
请定义由操作者管理的目录，不要把大文件或受限材料放进 Git 仓库：

```bash
export CANOPY_ROOT=/path/to/CANOPY
export IMAGE_ROOT=/data/canopy-swe-images
export DATA_ROOT=/data/canopy-swe-parquet
export VERIFIED_REVISION=replace-with-a-pinned-dataset-revision
mkdir -p "$IMAGE_ROOT" "$DATA_ROOT"
cd "$CANOPY_ROOT"
```

每个保留的训练/评测脚本都是自包含的，不依赖 `common.sh`。使用前请直接在脚本
中修改本地模型、数据、输出、Ray 地址和拓扑参数。

## 1. 生成并执行镜像下载任务

首先生成 JSON 任务清单。下面以 SWE-bench Verified 为例：

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

对表中本次需要的每一行重复执行 `generate`，并确保下载和数据构建命令使用
完全一致的 `--dataset-name`。ReBench V2 与 SWE-bench Pro 的镜像名取自其数据
字段。然后在能够访问 Docker Registry 的节点执行每份清单：

```bash
python3 recipe/swe/image_download.py run \
  --task-file "$IMAGE_ROOT/tasks-swe_bench_verified.json" \
  --image-root "$IMAGE_ROOT" \
  --engine docker \
  --max-workers 8
```

`run` 只接受位于生成清单时同一个显式 `--image-root` 下的归档路径。命令会
拉取每个镜像，并写成文件名以 `.tar` 结尾的 gzip 压缩 Docker 归档；任务成功
前会要求 `docker save` 成功，并检查最小文件大小与 gzip 标识；续跑时会按同样
的轻量检查跳过已有归档。如需不可变镜像身份，还应另行记录 registry digest。只有显式传入
`--remove-image-after-save` 才会删除已保存的镜像；工具不会执行全局镜像清理。

## 2. 生成训练与评测 Parquet

对应的镜像归档准备好后，再生成 Parquet。例如：

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

保留文件名同时编码 Prompt 过滤长度与 Ray 拓扑，例如 `_maxlen4k_group8` 或
`_maxlen6k_group12`。每个文件都必须使用相应 Launcher 所需的
`--max-prompt-length` 与 `--num-groups` 构建；不能跨不同拓扑复用已路由的
Parquet。

其他数据集请采用表中的来源、切分、数据集名称和 Docker Source。保留流程还会
使用以下选项：

- ReBench V2 第 1 部分：`--part-size 10000 --part-index 1 --seed 42`。Part 从
  1 开始编号，每个最多包含 `part-size` 行，最后一个 Part 可以更短。
- 构建训练集时，可使用 `--exclude-instance-ids` 排除固定 Revision 评测切分中的
  `instance_id`。它可以直接接收已生成的评测 Parquet
  （读取 `extra_info.instance_id`），也可以接收 JSON 字符串列表或每行一个 ID
  的文件，并可重复传入。该排除发生在确定性 Part 选择之前；还应独立确认最终
  训练/评测 ID 的交集为空。
- ReBench V1 还可以用 `--exclude-repos-from` 做更严格的仓库级隔离，并用
  `--instance-ids` 指定显式白名单。
- SWE-bench Pro 必须传入
  `--swe-bench-pro-run-scripts-dir /authorized/path/to/official/per-instance-scripts`；
  该目录中应存在 `<instance_id>/run_script.sh` 和 `parser.py`。所有可能承载 Pro
  分组的 Ray Worker 都必须能通过同一绝对路径读取这些文件；运行时会在 Worker
  上再次读取，而不只是构建 Parquet 的机器上检查一次。
- `--max-count`、`--part-size` 与 Prompt 长度过滤都会有意改变样本集合；应随
  实验一同记录。

`swe_data_process.py` 默认过滤掉预期 `.tar` 归档不存在的样本；
`--keep-missing-images` 可以关闭此过滤。正式评测必须使用完整、已授权的镜像
集合并核对最终行数；不得把因缺失镜像过滤后的子集分数当作完整 Benchmark
结果报告。

数据构建器先按 `--num-groups` 在不同 Group 间平衡各代码仓库的任务量，然后
把每个实例确定性地哈希到该仓库对应的某个 Group。它会把 `group_id`、镜像
Tag 和操作者的归档路径写进 `extra_info`。因此，生成的 Parquet 含机器本地
路径，不能加入公开源码包。`clean` 子命令可以从现有 Parquet 中删除已知的
镜像归档与 Pro 评测器宿主机路径键；其余字段仍需单独检查。清理操作不会因此
产生再分发授权。

## 3. 启动 Ray 与 Podman 集群

保留的 12 节点拓扑在 Head 上使用 `group_0`，每个 Worker 使用一个互不重复的
Group。先在每个节点检查 `run_scripts/swe/cluster/storage.conf`，其中
`graphroot` 必须指向大容量节点本地磁盘，不能指向共享 NAS。然后用独立脚本
初始化 Rootful Podman：

```bash
cd "$CANOPY_ROOT"
sudo bash run_scripts/swe/cluster/podman.sh
```

该脚本把同目录配置安装到 `/run/canopy-podman/storage.conf`，并通过
`podman info` 验证；它不会覆盖 `/etc/containers/storage.conf`，也不会启动
Podman API Service。Head/Worker 启动脚本会幂等地再次调用它，并把相同配置
传给 Ray。

在 Head 上启动 `group_0`。默认地址文件写到 `runtime/ray/`；如果代码仓库不是
共享目录，请通过 `HEAD_IP_FILE` 为所有节点指定同一个共享路径：

```bash
cd "$CANOPY_ROOT"
sudo bash run_scripts/swe/cluster/start_head.sh

# 如果代码仓库不是多节点共享目录：
sudo env HEAD_IP_FILE=/shared/canopy/head_ip.txt \
  bash run_scripts/swe/cluster/start_head.sh
```

当 Worker 名称符合 `mf_dsw_<batch>_<index>` 时，`start_worker.sh` 会从实例名或
DSW Agent 日志自动得到 `GROUP_NUM`。如果没有可识别名称，则手动指定 Group 和
Head 地址：

```bash
sudo env HEAD_IP_FILE=/shared/canopy/head_ip.txt \
  bash run_scripts/swe/cluster/start_worker.sh

# 或者同时显式指定两个值：
sudo env GROUP_NUM=3 HEAD_IP=head-node.example \
  bash run_scripts/swe/cluster/start_worker.sh
```

核心 Worker 启动命令为：

```bash
ray start --address="${HEAD_IP}:6379" \
  --resources="{\"group_${GROUP_NUM}\": 1000}"
```

Worker Group 必须是 `1` 到 `11` 的唯一整数，`group_0` 留给 Head。Parquet 中的
Group 数量、Ray 注册的 Group 和 `trainer.nnodes` 必须一致。0811/0817/0819 脚本启用
LXCFS CPU 视图，因此每个节点都要提供 `/var/lib/lxcfs`，否则应关闭该选项。
这些脚本需要 Rootful Podman，并会执行 `ray stop --force`，从而停止节点上已有的
Ray 任务。Podman 初始化可能创建 `/dev/fuse` 并把权限设为 `0666`，因此应使用
专用节点。Dashboard 默认只绑定 `127.0.0.1`；只有在配好防火墙时才显式设置
`DASHBOARD_HOST` 对外暴露。

## 4. 只预加载本次路由需要的镜像

传入本次运行涉及的全部 Parquet；预加载工具会对归档去重，并只把每个镜像
发送到其指定 Ray Group：

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

每个被路由的 Worker 都必须能在 Parquet 记录的同一绝对 `--image-root` 路径
读取镜像归档。请使用共享存储，或把归档同步到对应 Worker 的相同路径。

检查 Group 汇总后，去掉 `--dry-run` 再执行一次。加载器会先验证所需 Ray
Group 是否存在且容量足够，并为 Podman 检查和加载设置超时；随后检查 Tag
是否已存在，需要时加载本地归档。归档缺失或加载失败都会让命令失败。当 Group 容量为
1,000 时，默认的 `--resource-units 100` 会把每个节点的并发加载数限制为 10。
预加载后，Rollout 启动只需要创建容器，而不必现场从 Registry 拉取镜像。

容量取决于具体任务。以下只是运维经验，**不是硬件要求**：早期 9B 实验预
加载了 6,000 多个镜像；后续 Qwen3.6-35B-A3B 实验使用了 16,000 多个镜像，12 个
节点每个节点约占用 3.1 TB 本地空间。应根据所选 Parquet、归档大小、节点数、
文件系统开销与本地可用空间重新计算。

### 可选的在线容器回收

0819 脚本启用了 `recipe/swe/env_server/container_gc.py` 中的可选逐节点回收器，
每个节点使用有界队列和一个 Worker。每次删除都会 Fail Closed：回收器只解析一个
自动生成的 MiniSWE 容器名，核对完整 64 位 ID 以及 Job、节点、Group、Role、
Request 等 Ownership Label，随后只删除这个 ID。它不会 Prune 存储、删除镜像或
枚举无关容器。功能禁用或不可用时，原有 Container Timeout 加 `--rm` 仍是兜底。

在目标 Ray/Podman 集群完成小任务验证前，应保持 `container_gc_enabled=False`。
单元测试覆盖队列、重试、取消和身份校验，但本源码发布不声称已完成真实多节点
Podman 端到端验证。只调整文档列出的 `container_gc_*` 参数，并把每节点删除并发
保持在一或两个 Worker。

### 可选的依赖镜像

环境客户端提供默认关闭的公网 HTTPS 依赖镜像配置，用于可丢弃的任务容器内安装
依赖。在 `swe_custom_config` 中设置 `dependency_mirror_enabled=True`，并只保留本次
任务需要的 `dependency_mirror_*_enabled` 语言生态开关。已有的包管理器环境变量和
用户配置优先，初始化过程不会改写被评测代码仓库中的文件。内置地址来自
[阿里云开源镜像站](https://developer.aliyun.com/mirror/)，仅使用公网 HTTPS 地址，
不包含 ECS/VPC 专用地址。

该开关不会自动授予容器出网能力。公开默认值仍是 `network_mode: none`，预加载镜像
仍可离线运行。如果某些任务确实必须下载缺失依赖，应在无密钥、可丢弃的专用 Worker
上单独选择受限的 Bridge 或自定义 Podman 网络；不要仅为使用镜像而启用 Host 网络
或关闭 seccomp。分布式运行前，应先用完全相同的镜像和数据子集做小规模验证。

## 5. 训练或评测

### 训练脚本

| 脚本 | 模型/拓扑 | 保留命令所代表的数据 |
| --- | --- | --- |
| `swe/train/swe_qwen35_9b_8nodes_0528_origin_xml_36k.sh` | Qwen3.5-9B，8 节点，XML，36K | ReBench V1 训练；Verified 验证 |
| `swe/train/swe_qwen36_35b_a3b_12nodes_0811_toolcall_nothink_rebenchv1_val4bench.sh` | Qwen3.6-35B-A3B，12 节点，原生工具调用，不思考；**带保护的移植参考** | ReBench V1 训练；保留命令列出 ReBench leaderboard、Verified 与 Multilingual 三个验证文件 |
| `swe/train/swe_qwen36_35b_a3b_12nodes_0817_toolcall_nothink_rebenchv1v2p1_16k_b120_val4bench.sh` | Qwen3.6-35B-A3B，12 节点，原生工具调用，不思考；**带保护的移植参考** | ReBench V1 + 确定性选择的 ReBench V2 第 1 部分训练；在 Verified、Multilingual、ReBench leaderboard 与 Pro 上联合验证 |
| `swe/train/swe_qwen36_35b_a3b_12nodes_0819_toolcall_nothink_rebenchv1v2p1_16k_gc_offload.sh` | Qwen3.6-35B-A3B，12 节点，原生工具调用，不思考，Optimizer Offload 与可选在线容器回收；**带保护的移植参考** | 同一 V1 + V2 第 1 部分及四 Benchmark 布局，任务容器内存为 16 GiB，Action Timeout 为 300 秒 |

0817 与 0819 脚本记录了四 Benchmark 联合验证布局。训练过程中会使用四份验证
Parquet；但最终分数仍必须遵循各数据集的官方协议和评测器设置。0819 还启用了
有界的逐节点容器回收和 Megatron Optimizer Offload。

内置 Patch 已包含运行中 R2 Registry 重绑定与确定性的 SGLang NCCL 端口预留，
但仍缺少支持 CP 的逐 Token Loss 补偿和 Native Action 解析失败时的 Credit Mask。
0811/0817/0819 会检查所有所需能力的定义与接入，缺少任一项都会在 Ray 提交前
退出。不要绕过该保护；必须先移植等价行为，并完成
`docs/VERL_COMPATIBILITY.md` 所述的单 Batch Gradient-norm Smoke Test。

### 评测脚本

当前六个独立评测脚本都**只针对 SWE-bench Verified**。它们是不同模型、
Prompt 和上下文设置的保留变体，并不是 Multilingual、ReBench leaderboard
或 Pro 的独立启动器：

| 脚本 | 变体 |
| --- | --- |
| `swe/eval/swe_eval_qwen35_9b_8nodes_0621_origin_xml_36k.sh` | Qwen3.5-9B，原始 XML、开启思考、36K、2 次贪心验证采样 |
| `swe/eval/swe_eval_qwen35_9b_8nodes_0628_nothink_xml_64k.sh` | Qwen3.5-9B，不思考 XML，64K |
| `swe/eval/swe_eval_qwen35_9b_8nodes_0806_toolcall_nothink_64k.sh` | Qwen3.5-9B，原生工具调用，不思考，64K |
| `swe/eval/swe_eval_qwen35_9b_8nodes_0806_toolcall_think_64k.sh` | Qwen3.5-9B，原生工具调用，思考，64K |
| `swe/eval/swe_eval_qwen36_35b_a3b_8nodes_0806_toolcall_nothink.sh` | Qwen3.6-35B-A3B，原生工具调用，不思考 |
| `swe/eval/swe_eval_qwen36_35b_a3b_8nodes_0806_toolcall_think.sh` | Qwen3.6-35B-A3B，原生工具调用，思考 |

这些是保留的可运行命令，不是严格控制单一变量的消融实验。部分思考/不思考
脚本还使用了不同的验证 temperature 或 top-p；应按每个脚本的实际采样参数
完整披露，不能把结果差异只归因于是否思考。

同步 Trainer 会在处理 `trainer.val_only=True` 前初始化训练 DataLoader。为避免
额外依赖无关的 ReBench 训练文件，公开评测脚本把这个不会被优化步骤使用的
Loader 指向同一份 Verified Parquet；实际评测仍只由 `data.val_files` 驱动，且
不会执行参数更新。

内置 Rollout Server 已包含可选的确定性 SGLang NCCL 端口预留，但保留的独立
评测脚本没有启用它。大量 Rollout Replica 同机启动时，应先核对候选端口范围和
集群策略，再启用该已审查路径，或由操作者提供其他端口竞争缓解方案。

在脚本中修改路径后，从仓库根目录运行：

```bash
bash run_scripts/swe/train/swe_qwen35_9b_8nodes_0528_origin_xml_36k.sh

# 或执行只针对 Verified 的预测/评测命令：
bash run_scripts/swe/eval/swe_eval_qwen35_9b_8nodes_0806_toolcall_nothink_64k.sh

# 查看当前带保护的 Qwen3.6 配置（未完成移植时会退出）：
bash run_scripts/swe/train/swe_qwen36_35b_a3b_12nodes_0819_toolcall_nothink_rebenchv1v2p1_16k_gc_offload.sh
```

训练使用 `recipe.swe.main_ppo_sync`；评测脚本设置 `trainer.val_only=True`。
Verl 会把 Rollout 和验证材料写进所配置的输出根目录。每个报告结果都应保留
准确的数据集 Revision、样本行数、镜像集合、Checkpoint、运行脚本、采样
参数、超时设置与官方评测器版本。

## Step 统计

`recipe/swe/step_diagnostics.py` 当前输出 Tracker Layout Version 6，并为训练和
验证使用独立的指标前缀。该版本重命名了此前的 SwanLab Key，因此不能在没有
显式映射的情况下把新旧曲线按名称直接拼接。界面中的 `fake_rate` 是 Tracker
展示分类，包含评测失败和诊断缺失；运行时原始的 `AgentData.is_fake` 分类仍单独
保存在诊断 JSON 中。

## 隔离与安全

SWE 任务会在 Benchmark 容器中执行不可信的代码仓库内容。请使用专用的非生产
Worker、最小权限、无密钥环境、受限出网、受防火墙保护的 Ray 服务和可丢弃
存储。公开环境配置默认把容器网络设为 `none`，Rollout 期间只使用预加载镜像
（`--pull never`）。执行前应审查每个镜像以及数据集提供的 Pro 脚本/Parser。
不要把凭证、源码工作区或敏感宿主机路径挂载到任务容器。
