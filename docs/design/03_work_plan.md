# 任务切分与执行计划

读者：协调者（用户）与在服务器上执行任务的 Claude Code 会话。前置：读 `00_paradigm.md`、`01_integration_rules.md`、`02_benchmark_profiles.md`。

---

## 1. 任务总表

| ID | 任务 | 依赖 | 是否需要仿真器 / GPU | 产出（核心） | 预计规模 |
|---|---|---|---|---|---|
| **T01** | 核心接口与规格对象（`hazardlib` 骨架） | 无 | 否 | 数据类、`HazardHost` 协议、BD 加载 / 校验、哈希、`configs/hazard/` 骨架、双版本 Python 测试 | 小 |
| **T02** | 运动学与校准数学 | T01 | 否 | 秒为单位的运动原语、τ 反解、子步细分、T_stop / Σ 计算 | 小 |
| **T02-F** | T02 收尾：参数重命名、文档对齐决定 D1–D6、配置骨架（见 `prompts/T02_followup.md`） | T02 | 否 | 同一分支上的小提交 | 小 |
| **T03** | 安全语义 | T01 | 否 | 事件与接触图、严重度、倾倒、无效样本、`HazardEventLog` 读写 | 小到中 |
| **T03-F** | T03 收尾：落实 D10（归因窗口、BD 匹配前缀、`first_scene_contact_t_s`、YAML 键、卫生测试白名单，见 `prompts/T03_followup.md`） | T03 | 否 | 同一分支上的小提交 | 小 |
| **T04** | `HazardRuntime` + `FakeHost` | T01–T03 | 否 | 控制反转运行时、回滚释放、事件记录、纯 Python 假世界 | 中 |
| **T04-F** | T04 收尾：落实 D12（`EventKind` 改名 `dynamic`、`conformance.yaml` 补 C13 块；宿主钩子语义写进 01 §3.3） | T04 | 否 | 同一分支上的小提交 | 小 |
| **T05** | 一致性测试套件 C1–C13 | T04 | 否 | pytest 套件 + 变异体验证（每个测试都能抓到对应缺陷） | 中 |
| **T06** | 流水线：探针 / 实例化 / 门控 / 校准 / 冻结 | T04 | 否 | `hazardctl` 子命令，FakeHost 上端到端 | 中到大 |
| **T05-F** | T05 收尾：落实 D13 h（C5 落地时间容差按物理步、statistical 宿主的 C2 / C7 / C12 判据与按观测键的 C8 上限、报告路径不含本机绝对路径，见 `prompts/T05_followup.md`） | T05、T06 | 否 | 小提交 | 小 |
| **T06-F** | T06 收尾：落实 D13 a–d（冻结绑定一致性报告、Manifest 增 `conformance_report_hash` / `calibration_hash`、运行时特权访问器 `hazard_state`，见 `prompts/T06_followup.md`） | T05、T06 | 否 | 小提交 | 小到中 |
| **T07** | 核心收尾：落实 D15 a、b、e（机器人几何求 standoff 与 G0 早接触、碰撞体路径不含抓持物、宿主注册回退，见 `prompts/T07_followup.md`） | T05-F、T06-F、`fix-hook-times` | 否 | 小提交 | 中 |
| **E00** | 基准环境搭建（按基准参数化） | 无 | 是 | `docs/design/env_notes/<bench>.md`，基准自带最小评测跑通 | 视基准 |
| **H1** | CALVIN 适配器 | T01–T06（含 T05-F、T06-F）、E00(calvin) | 是 | 适配器、BD、一致性测试通过、特权运行、成本账本 | 中 |
| **H2** | RoboCasa365 适配器 | T01–T06（含 T05-F、T06-F）、T07、E00(robocasa) | 是 | 同上 | 中 |
| **H3** | LIBERO（SafeLIBERO）适配器（D16 a：布局 = 演示初始状态） | T01–T06（含 T05-F、T06-F）、**T07**、E00(libero) | 是 | 同上 | 中 |
| **H4** | SimplerEnv 适配器 | T01–T06（含 T05-F、T06-F）、E00(simplerenv) | 是 | 同上 | 中 |
| **H5** | RoboTwin 适配器 | T01–T06（含 T05-F、T06-F）、E00(robotwin) | 是 | 同上 | 中到大 |
| **H6** | RoboDojo 适配器（含 E1–E8 引擎探针） | T01–T06（含 T05-F、T06-F）、E00(robodojo) | 是 | 探针报告 + 适配器 | 大 |
| **X1** | 策略矩阵与服务 | E00 | 是 | `policy_matrix.md`、每基准至多两个策略的服务脚本 | 中 |
| **R1** | Run 阶段核心（D16 c）：`hazardctl run` / `report`、对照回合、汇总表、视频帧记录接口，见 `prompts/R1_run.md` | T06（可与 T07 并行，不碰 T07 改的文件） | 否 | FakeHost 上端到端 | 中 |
| **R1-F** | R1 收尾（D17 a、b）：`run.yaml` / `report.yaml` 拆分、`run.yaml` 进 `config_hashes`、冻结目录带 TaskContext 与 BD | R1 | 否 | 小提交 | 小 |
| **R1-L** | LIBERO 端到端：策略桥接（复用 `src/dynisafety/policies`）、视频编码、spatial 4 任务 × 1–2 个策略的试点运行（prompt 在 H3 后写） | R1、H3、X1(libero) 或既有服务脚本 | 是 | 视频、成功率与安全指标表（试点） | 中 |

**阶段与并行。**
- 阶段 A（无需仿真器，可立刻开始）：T01 先做（约一两个小时的量）；T01 合并后 T02、T03 并行（T02 后跟一个小收尾 T02-F；**T04 在 T02-F 与 T03（含 T03-F）都合并后才放行**）；T04 之后 T05、T06 并行；两者合并后 T05-F、T06-F 并行（D13）。
- 阶段 B（需要环境）：E00 对各基准可与阶段 A **同时**进行；H1–H6 在 T01–T06（含 T05-F、T06-F）合并并且对应 E00 完成后开始。
- 阶段 C（实验 H0–H4）：在 H1–H3 至少完成、P0 修订落地后再写 prompt；本批**不提供**，因为它们取决于阶段 B 的结果（特别是 T_stop 与跨引擎探针是否可行）。

**D16 调整（2026-11-09）：** 先打通 LIBERO：E00(libero) ∥ T07 ∥ R1 → H3 → R1-L；H2 在 T07 后照常进行。LIBERO 结果在 §5 门槛达到前一律标“试点”。

**建议的启动顺序（一台服务器、串行时）：** T01 → T02 ∥ T03 → T04 → T05 ∥ T06；E00(calvin) → H1；然后 E00(robocasa) → H2；E00(robodojo) 由用户安装环境，`doctor` 通过后再启动 H6。

---

## 2. 分支与合并

- 集成分支：`hazard/integration`（由协调者从 `claude/sweet-newton-2clmpw` 创建）。
- 每个任务会话建分支 `hazard/<ID>`：T01 与 E00 从 `claude/sweet-newton-2clmpw` 起；依赖 T01 及之后的任务从 `hazard/integration` 起（协调者在 T01 通过后合并进去）。
- 任务会话**只提交并推送自己的分支**，**不合并、不开 PR、不改 `docs/design/00–03`**；对设计文档的修订意见写进自己的汇报。
- 协调者合并顺序：T01 → T02、T03 → T04 → T05、T06 → T05-F、T06-F → H*。D15 之后 T07 先于 H2 合并；H1 接入 D15 a 的机器人几何另做收尾（H1-F），在 T07 合并之后。

---

## 3. 汇报协议

每个任务在 `docs/design/reports/<ID>.md` 写汇报，模板见 `prompts/T00_shared_context.md` §汇报模板。要求：
1. 验收命令与**真实输出**（粘贴，不得改写）；
2. 未通过的测试逐项列出；
3. 偏离设计文档之处与理由；
4. H 类任务填写适配成本账本；
5. 任何“已验证”都要附命令或行号。

---

## 4. 服务器环境矩阵（每基准一个环境，互斥）

| 基准 | 环境名（建议） | Python | 关键约束 |
|---|---|---|---|
| LIBERO | `openvla-oft`（README 约定）| 3.10 **[推断]** | `numpy<2.0` |
| SimplerEnv | 既有 SimplerEnv 环境 | 以 `docs/user/install.md` 为准 | `sapien==2.2.2` |
| RoboTwin | 既有 RoboTwin 环境 | 同上 | `sapien==3.0.0b1` |
| RoboCasa | `dynisafety-robocasa` | 3.11 | `numpy==2.2.5`、`mujoco==3.3.1`；资产约 10 GB |
| RoboDojo | `rsb_robodojo61`（自建，见 `env_notes/robodojo.md`） | 3.12 | Isaac Sim 6.1（E00 决定的移植）、`numpy==2.3.1`、`torch==2.11.0`；自建克隆，`Assets` 只读共享 |
| CALVIN | `calvin_venv` | **3.8** | 官方 `install.sh`；`hazardlib` 必须兼容 |

纯 Python 开发（T01–T06）另建 `hz38` 与 `hz311` 两个轻量环境，用于双版本测试。

---

## 5. 总验收（进入阶段 C 的门槛）

- T01–T06 合并后，`pytest hazardlib` 在 Python 3.8 与 3.11 全绿，`FakeHost` 通过 C1–C13，变异体全部被抓到。
- 至少 H1（CALVIN）与 H2（RoboCasa）通过各自的 C1–C13，并产出 T_stop 与特权运行结果。
- H6 的 E1–E8 探针报告完成，并给出 RoboDojo 的 Tier 结论（A / B / 不可行）。
- 适配成本账本齐全。

未达门槛不得进入阶段 C。
