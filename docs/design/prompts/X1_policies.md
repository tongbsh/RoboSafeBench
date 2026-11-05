# X1 — 策略矩阵与服务

起点分支：`hazard/integration`　依赖：对应基准的 E00　需要 GPU：是　先读：T00，`02_benchmark_profiles.md` §8。
**参数 `BENCH`**（可一次处理多个，逐个出汇报）。任务分支：`hazard/X1-<BENCH>`。

## 目标
弄清每个基准上**可以跑起来**的学习策略，下载并服务至多两个（一个小而快、一个强），为后续实验备料。**这不是评测任务，不跑大规模评测。**

## 范围与产出

1. **矩阵** `docs/design/policy_matrix.md`（多个任务会话合并时按基准分节追加，各自只写自己的节）：
   每个基准一张表：策略 | 是否有该基准的公开检查点 | 来源链接 | 大小 | 依赖 / 环境冲突 | 接入方式（本仓库 `policies/adapters`、`transports`；RoboDojo 的 XPolicyLab；CALVIN 的 `CalvinBaseModel.step(obs, goal)`）| 在本机能否跑通 | 备注。
2. **服务**：为选定的策略写启动脚本（放 `scripts/hazard/serve_<BENCH>_<policy>.sh`）并验证：
   - LIBERO / SimplerEnv / RoboTwin：优先复用本仓库既有适配与下载脚本（`scripts/setup/download_assets.py --model <stem>`，见 README 与 `configs/models/*.yaml`），既有模型清单见 `02_benchmark_profiles.md` §8。
   - RoboDojo：通过 XPolicyLab（子模块）；检查点必须是**在 RoboDojo 数据上训练**的，先确认有无公开检查点，没有就如实写“不可得”，**不要**自行大规模训练。
   - RoboCasa：先调研有无公开检查点（LeRobot 集成等），没有就写“不可得”。
   - CALVIN：调研公开检查点（至少看 README 列出的方法），选一个能在本机跑通的。
3. **冒烟验证**：每个选定策略在对应基准的 `L0`（无危险）上跑 **1 个任务 × 3 个回合**，贴出成功率与耗时；只证明“能跑”，不做统计结论。

## 验收标准（贴输出）

1. 矩阵每格有来源或明确写“不可得 / 未核实”。
2. 选定策略的服务脚本可重复启动，冒烟输出完整。
3. 下载量与磁盘占用记录；**单次下载 > 5 GB 前已征得用户同意**。
4. 与环境互斥原则一致：策略依赖与基准环境冲突时，用**独立环境 + 服务（HTTP / WebSocket / ZMQ，本仓库 `policies/transports`）**隔离，不要硬装进基准环境。

## 不要做
不要训练或微调；不要把检查点放进 git；不要在没有公开检查点时伪造“等效”策略。

## 汇报
`docs/design/reports/X1-<BENCH>.md`。
