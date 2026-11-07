# docs/design — 动态安全扰动范式 v3：设计、接入规则与任务

| 文件 | 内容 |
|---|---|
| `00_paradigm.md` | 范式：主张、对象、流水线、危险族、实例化规则、统一安全语义、校准、证书、方法学纪律 |
| `01_integration_rules.md` | 通用接入规则：目录、BD 字段、`HazardHost` 契约、规则 R0–R16、一致性测试 C1–C13、适配器验收流程 |
| `02_benchmark_profiles.md` | 六个基准（LIBERO、SimplerEnv、RoboTwin、RoboCasa365、RoboDojo、CALVIN）各自怎么接 |
| `03_work_plan.md` | 任务切分、依赖、分支与合并、汇报协议、服务器环境矩阵、进入实验阶段的门槛 |
| `prompts/` | 喂给服务器上 Claude Code 新会话的任务文件 |
| `reports/` | 各任务汇报（由任务会话写入） |
| `env_notes/` | 各基准环境笔记（由 E00 写入） |

前置审计：`docs/audit/p0_semantics_audit.md`、`docs/audit/r1_benchmark_capability.md`。

## 任务与文件

| 任务 | 文件 | 起点分支 | 参数 |
|---|---|---|---|
| T01 核心接口 | `prompts/T01_core_spec.md` | `claude/sweet-newton-2clmpw` | — |
| T02 运动学 | `prompts/T02_kinematics.md` | `hazard/integration` | — |
| T02-F T02 收尾（合并前） | `prompts/T02_followup.md` | `hazard/T02`（同一分支继续） | — |
| T03 安全语义 | `prompts/T03_semantics.md` | `hazard/integration` | — |
| T03-F T03 收尾（合并前） | `prompts/T03_followup.md` | `hazard/T03`（同一分支继续） | — |
| T04 运行时与 FakeHost | `prompts/T04_runtime_fakehost.md` | `hazard/integration` | — |
| T05 一致性套件 | `prompts/T05_conformance.md` | `hazard/integration` | — |
| T06 流水线 | `prompts/T06_pipeline.md` | `hazard/integration` | — |
| E00 环境搭建 | `prompts/E00_env_bringup.md` | `claude/sweet-newton-2clmpw` | `BENCH` |
| H1 CALVIN | `prompts/H00_host_template.md` + `prompts/H1_calvin.md` | `hazard/integration` | — |
| H2 RoboCasa | `prompts/H00_host_template.md` + `prompts/H2_robocasa.md` | `hazard/integration` | — |
| H3 LIBERO | `prompts/H00_host_template.md` + `prompts/H3_libero.md` | `hazard/integration` | — |
| H4 SimplerEnv | `prompts/H00_host_template.md` + `prompts/H4_simplerenv.md` | `hazard/integration` | — |
| H5 RoboTwin | `prompts/H00_host_template.md` + `prompts/H5_robotwin.md` | `hazard/integration` | — |
| H6 RoboDojo | `prompts/H00_host_template.md` + `prompts/H6_robodojo.md` | `hazard/integration` | — |
| X1 策略矩阵 | `prompts/X1_policies.md` | `hazard/integration` | `BENCH` |

所有任务都先读 `prompts/T00_shared_context.md`。

## 启动一个任务会话（复制后替换尖括号）

在服务器上，**RoboSafeBench 仓库根目录**启动 Claude Code，粘贴：

```text
你在 RoboSafeBench 仓库根目录运行。任务编号：<ID>。基准参数（如适用）：<BENCH>。
1. 先完整阅读 docs/design/prompts/T00_shared_context.md，并按其中第 0 节切到任务分支（起点分支见 docs/design/README.md 的任务表）。
2. 再阅读任务文件：docs/design/prompts/<任务文件>（H 类任务先读 H00_host_template.md 再读对应的 Hn 文件）。
3. 先用不超过 10 行复述你理解的范围、产出和验收标准，等我确认后再动手。
4. 严格按任务文件的范围、验收标准和汇报格式执行；遇到阻塞按 T00 第 3 节处理，不要扩大范围。
```

## 协调者备忘

1. 创建集成分支：`git fetch origin && git checkout -B hazard/integration origin/claude/sweet-newton-2clmpw && git push -u origin hazard/integration`。
2. T01 完成后审阅并合并到 `hazard/integration`，再放行 T02、T03；之后依次 T04 → T05 ∥ T06 → H*。
3. 任务会话不合并、不开 PR；合并由协调者执行。
4. 进入实验阶段（H0–H4）前，需满足 `03_work_plan.md` §5 的门槛。
