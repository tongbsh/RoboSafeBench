# `scripts/run/` — 一键执行脚本

每个脚本都可直接 `bash scripts/run/xx_*.sh` 执行，会**自动切换到该模型/backend 需要的 conda 环境**，无需手动 `conda activate`。

## 为什么需要自动切换环境

不同 VLA 依赖互斥（NumPy/TF/torch 版本冲突），本机有 18 个 conda 环境。一次评测需要的环境必须同时满足两件事：

- **仿真器**（由 backend 决定）：LIBERO 需要 `libero`，SimplerEnv 需要 ManiSkill2，RoboTwin 需要 sapien + **curobo**
- **策略**（由模型决定）：如 StarVLA 服务端需要 `starVLA` 环境

映射关系的唯一来源是 `configs/models/*.yaml`：

| 字段 | 含义 |
|------|------|
| `extra.eval_env` | 跑评测客户端的环境 |
| `server.env` / `extra.serve_env` | 跑推理服务的环境 |

脚本通过 `tools/model_env.py` 读取该 YAML，不在 shell 里硬编码环境名。对 backend 无关的模型（如 `random`，`eval_env: base`），回落到 backend 的仿真器环境。

## 执行顺序

```bash
bash scripts/run/01_setup.sh          # 一次性：submodule + patch + pip install -e + 审计 + doctor
bash scripts/run/02_doctor.sh         # 环境自检（每次换机器/装完依赖后跑）
bash scripts/run/03_smoke.sh          # 5 分钟冒烟：单测 + 配置解析 + backend dry-run + 一次真实 rollout
```

冒烟过了再往下：

```bash
# 单次评测：<backend> <model> <suite> <level> [额外 flag]
bash scripts/run/10_eval.sh simplerenv random       bridge             2 --task 0 --episodes 1
bash scripts/run/10_eval.sh robotwin   internvla_a1 main20             0 --task 0 --episodes 2
bash scripts/run/10_eval.sh libero     openvla_oft  safelibero_spatial 3 --episodes 5
# RoboCasa kitchen-nav（numpy 2 env；不在 paper_main）
bash scripts/run/10_eval.sh robocasa   random_mobile robocasa_nav      0 --task 0 --episodes 1

# 服务型模型：先在一个终端起服务，再在另一个终端评测
bash scripts/run/20_serve.sh starvla --gpu 0
bash scripts/run/10_eval.sh robotwin starvla main20 3 --task 1

# 权重按 suite 区分的服务型模型（如 univla）必须传 --suite：
# 权重是服务端启动时加载的，一个服务只能对应一个 suite
bash scripts/run/20_serve.sh univla --suite safelibero_object --gpu 0
bash scripts/run/10_eval.sh libero univla safelibero_object 3 --task 0

# 大规模 sweep（取代原来 ~276 个 shell 脚本）
bash scripts/run/30_sweep.sh --dry-run            # 先看计划，务必
DYNI_GPUS=0 bash scripts/run/30_sweep.sh          # 真跑
bash scripts/run/30_sweep.sh --resume             # 断点续跑
bash scripts/run/30_sweep.sh --retry-failed       # 只重跑失败的

# 出报表
bash scripts/run/40_report.sh --validate          # 只看覆盖度
bash scripts/run/40_report.sh                     # 出论文表

# 重构正确性验证（对比重构前的归档结果）
bash scripts/run/50_verify_golden.sh --extract    # 解开归档日志看一眼
bash scripts/run/50_verify_golden.sh              # 跑一致性测试

# 环境审计：哪个模型缺哪个 conda 环境
bash scripts/run/99_audit_envs.sh
```

## 脚本清单

| 脚本 | 作用 |
|------|------|
| `00_env.sh` | 被其他脚本 source，不直接执行。定义路径、conda 激活、环境解析、`dyni_cli` |
| `01_setup.sh` | 一次性安装 |
| `02_doctor.sh` | 环境自检，透传 `--backend/--strict/--online` |
| `03_smoke.sh` | 冒烟，`--offline` 可跳过真实 rollout |
| `10_eval.sh` | 单次评测 |
| `20_serve.sh` | 起模型服务 |
| `30_sweep.sh` | 大规模 sweep |
| `40_report.sh` | 聚合出表 |
| `50_verify_golden.sh` | golden 一致性验证 |
| `99_audit_envs.sh` | 环境审计 |

## 可覆盖的环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `DYNI_ROOT` | 仓库根 | 仓库位置 |
| `DYNI_CONDA_SH` | `$HOME/miniconda3/etc/profile.d/conda.sh` | conda hook（注意是 conda **安装**位置，不是 env 目录） |
| `DYNI_TOOL_ENV` | `base` | 跑 doctor/report/config 等无需仿真器的工具用的环境 |
| `DYNI_ENV_LIBERO` | `openvla-oft` | LIBERO 仿真器环境 |
| `DYNI_ENV_SIMPLERENV` | `simpler_env` | SimplerEnv 仿真器环境 |
| `DYNI_ENV_ROBOTWIN` | `internvla_a1` | RoboTwin 仿真器环境（**本机只有这个 env 装了 curobo**） |
| `DYNI_ENV_ROBOCASA` | `dynisafety-robocasa` | RoboCasa 厨房轮式导航（NumPy 2；与其它 backend 互斥） |
| `DYNI_SUITE` | `paper_main` | sweep 用的 suite |
| `DYNI_GPUS` | 自动发现 | 逗号分隔 GPU id |
| `DYNI_JOBS_PER_GPU` | 由 model YAML 决定 | 全局并发上限 |
| `DYNI_RESULTS` / `DYNI_REPORTS` | `results/` / `reports/` | 输入输出目录 |
| `DYNI_GRID` | `configs/report/grids/paper_main.yaml` | 报表 grid 规格 |
| `DYNI_GOLDEN_TAR` | `$HOME/autodl-tmp/dynisafety_archive/golden_logs_20260805.tar.gz` | 归档 golden |

## 当前已知状态（2026-08-06 实测）

| backend | live 跑通 | 实测命令 |
|---------|----------|---------|
| SimplerEnv | ✅ | `bash scripts/run/10_eval.sh simplerenv random bridge 2 --task 0 --episodes 1` |
| RoboTwin | ✅ | `bash scripts/run/10_eval.sh robotwin internvla_a1 main20 0 --task 21 --episodes 1 --seed 0`（真实 400 步 episode） |
| LIBERO | ✅ | `bash scripts/run/10_eval.sh libero random safelibero_spatial 3 --task 0 --episodes 1`（真实 220 步，L3 下测到 peak_force 3.74 N）|
| RoboCasa | ✅ L0–L3 已接线（2026-08-13/14；**不在** `paper_main`） | `bash scripts/run/10_eval.sh robocasa random_mobile robocasa_nav 0 --task 0 --episodes 1`（厨房轮式地板障碍；步数/力不可与上表绝对值比） |

### 判断"是否真的跑通"的方法

`eval done` 只说明流程走完了，**不代表 episode 有效**。必须看 `episodes.jsonl`：

```bash
D=$(ls -dt results/l*/* | head -1)
# legacy flat runs (pre level-bucket): ls -dt results/* | head -1
python -c "
import json; r=json.loads(open('$D/episodes.jsonl').readline())
print(r['task_name'], 'steps=%s'%r['steps'], 'term=%s'%r['termination'])
"
```

`steps=0` + `termination=error` 表示 rollout 立刻失败了（外层仍会打印 `eval done`）。

### 已知的模型/backend 组合限制

- **`random` 跑不了 RoboTwin**：`configs/models/random.yaml` 是 `action_space: delta_ee_7d`，RoboTwin 需要 `qpos_14d`。会得到 `steps=0 / termination=error`。RoboTwin 请用 `internvla_a1`。
- **RoboTwin 的 expert-check 与任务/seed 有关**：`adjust_bottle`（task 0）在 seed 2000/16000 附近 200 次尝试内都找不到可行 seed；`place_a2b_right`（task 21）可以。
- **`dreamvla` / `octo-server`** 引用了本机不存在的 conda 环境，两者均未用于论文。
- **`openvla_oft` 跑不了 LIBERO**：`openvla-oft` 环境内 NumPy 2 与 TensorFlow 二进制不兼容（`_ARRAY_API not found`）。LIBERO backend 本身没问题。
- **LIBERO 依赖 `~/.dynisafety_libero/config.yaml`**（家目录，不在仓库内）。该文件缺失时 LIBERO 会调 `input()` 交互提问导致挂死；移动仓库后其中的绝对路径也要同步改。

本地维护者的重构交接与剩余清单：`old_cache/docs_internal/HANDOFF.md`（gitignored，不随开源发布）。
