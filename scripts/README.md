# `scripts/` 执行指南

按功能分模块。权威评测入口始终是 CLI：`dynisafety eval` / `dynisafety sweep`（见 [`docs/migration.md`](../docs/migration.md)）。

| 模块 | 用途 | 常用入口 |
|------|------|----------|
| [`run/`](run/README.md) | 日常评测 / serve / sweep / report（自动切 conda） | `bash scripts/run/10_eval.sh …` |
| [`install/`](install/) | 后端安装、patch、权重下载、editable 修复 | `bash scripts/install/install_backend.sh libero` |
| [`debug/`](debug/) | 调参小跑 / 长矩阵 serve+eval | `bash scripts/debug/eval_debug.sh --help` |
| [`feasibility/`](feasibility/) | 授权后的 Oracle / AEGIS 可行性矩阵（默认拒 GPU） | `bash scripts/feasibility/run_authorized_aegis_feasibility.sh` |
| [`models/`](models/) | 模型专用包装（如 LingBot-VA serve） | `bash scripts/models/lingbot_va/serve.sh` |

```text
scripts/
├── README.md
├── run/                 # 正式一键评测
├── install/             # 安装 / patch / 资产
├── debug/               # 调参小跑（跨 backend 入口 + 按 backend 分类）
│   ├── eval_debug.sh / launch_vla_{serve,eval}.sh
│   ├── libero/
│   ├── robotwin/
│   └── simplerenv/
├── feasibility/         # 授权可行性
└── models/              # 模型专用
    └── lingbot_va/
```

---

## 0. 安装（一次性）

```bash
# 四选一（extras / numpy 互斥，不要装进同一 conda env）
make install-libero          # = bash scripts/install/install_backend.sh libero
make install-simplerenv
make install-robotwin
make install-robocasa        # numpy 2 env dynisafety-robocasa

dynisafety doctor --backend libero
python scripts/install/download_assets.py --help
```

更完整：[`docs/installation.md`](../docs/installation.md)。也可：`bash scripts/run/01_setup.sh`。

---

## 1. Debug 评测

单格调参：`scripts/debug/eval_debug.sh`（走 `scripts/run/10_eval.sh`）。

长矩阵（screen + 日志）：`launch_vla_serve.sh` + `launch_vla_eval.sh` 已接通 **libero / robotwin / simplerenv**。
`xvla_official`（RoboTwin / SimplerEnv）另有薄包装；LIBERO 上的 X-VLA 用 stem `xvla`，不要用 `xvla_official`。

```bash
# --- Dynamic LIBERO（例：π₀.₅，端口 8011）---
screen -S vla_serve
bash scripts/debug/launch_vla_serve.sh --model pi05 --port 8011
screen -S vla_eval
bash scripts/debug/launch_vla_eval.sh --model pi05 --backend libero \
  --level 1 --suite spatial --task 0 --episodes 1

# --- xvla_official × RoboTwin（端口 8010，权重 X-VLA-RoboTwin2）---
screen -S xvla_rt_serve
bash scripts/debug/robotwin/launch_xvla_official_robotwin_serve.sh
screen -S xvla_rt_eval
bash scripts/debug/robotwin/launch_xvla_official_robotwin_eval.sh --level 0 --task 21 --episodes 1

# --- xvla_official × SimplerEnv（端口 8010，权重 xvla-simpler-widowx）---
screen -S xvla_se_serve
bash scripts/debug/simplerenv/launch_xvla_official_simplerenv_serve.sh
screen -S xvla_se_eval
bash scripts/debug/simplerenv/launch_xvla_official_simplerenv_eval.sh --level 0 --task 0 --episodes 1
```

一个 serve 只对应一套权重：换 RoboTwin ↔ SimplerEnv 必须重启 serve（`--suite main20` vs `bridge`）。
`--task all` 在 RoboTwin 上是 catalog id（`0,1,2,5,…,48`），不是 `0..19`。SimplerEnv `all` 是论文三题 `0,1,2`（eggplant 显式传 `--task 3`）。

内部按 `configs/models/<model>.yaml` **自动激活 conda**。serve 在 **tool env** 里跑 `python -m dynisafety.cli serve`（不要用 `scripts/run/20_serve.sh`）。

### 默认值（仅 debug 脚本提供，正式 CLI 无业务默认）

| 项 | `eval_debug.sh` | `launch_vla_eval.sh` |
|----|-----------------|----------------------|
| `--backend` | `libero` | `libero` |
| `--model` | `pi05` | 随 backend：libero→`pi05`，robotwin/simplerenv→`xvla_official` |
| `--suite` | 单 suite 默认 | `all`（libero 四套 / robotwin `main20` / simplerenv `bridge`） |
| `--level` | `0` | `1,2,3` |
| `--task` | `0` | `all` |
| `--episodes` / `--seed` | `1` / `7` | `1` / `7` |
| `--port` | （走 10_eval） | `xvla*` → `8010`，其它 → `8011` |
| `--out` | `results/l{level}/…` | `artifacts/debug/<model>_<backend>/` |

### 常用命令

```bash
bash scripts/debug/libero/run_pi0_libero_spatial.sh
bash scripts/debug/eval_debug.sh --model pi05 --level 1 --episodes 1 --dry-run

bash scripts/debug/eval_debug.sh \
  --backend libero --model pi05 --suite safelibero_spatial \
  --level all --task 0 --episodes 3

bash scripts/debug/eval_debug.sh \
  --backend robocasa --model random_mobile --suite robocasa_nav \
  --level 0 --task 0 --episodes 1 --dry-run

bash scripts/debug/eval_debug.sh \
  --model pi05 --level 1 --episodes 1 --oracle --aegis
```

服务型模型另开终端：`bash scripts/run/20_serve.sh pi05 --gpu 0`。

结果在 `results/l{level}/<run_id>/`。是否真 rollout：看 `episodes.jsonl` 的 `steps` / `termination`（[`run/README.md`](run/README.md)）。

---

## 2. 正式 / 大规模（不要用 debug 脚本）

| 目标 | 命令 |
|------|------|
| 冒烟 | `bash scripts/run/03_smoke.sh` 或 `dynisafety sweep --suite smoke` |
| 论文主表 | `bash scripts/run/30_sweep.sh --dry-run` → `DYNI_GPUS=0 bash scripts/run/30_sweep.sh` |
| 单格正式 | `bash scripts/run/10_eval.sh <backend> <model> <suite> <level> --task N --episodes K` |
| 出表 | `bash scripts/run/40_report.sh` |

矩阵权威：`configs/suites/paper_main.yaml`。细节：[`run/README.md`](run/README.md)。

---

## 3. 可行性（Oracle / AEGIS）

默认 **拒 GPU**（`BLOCKED_UNTIL_AUTHORIZED`）：

```bash
bash scripts/feasibility/run_authorized_aegis_feasibility.sh
AUTHORIZE_GPU=1 bash scripts/feasibility/run_authorized_aegis_feasibility.sh
```

说明：[`docs/experiment_upgrade/pi05_libero_aegis_mount.md`](../docs/experiment_upgrade/pi05_libero_aegis_mount.md)。

---

## 4. Backend × Suite × 常见模型速查

| `--backend` | 常用 `--suite` | 调试常用 `--model` |
|-------------|----------------|-------------------|
| `libero` | `safelibero_{spatial,object,goal,long}` | `pi05`, `pi0`, `xvla`, `openvla_oft`, `random` |
| `simplerenv` | `bridge` | `xvla_official`, `starvla`, `internvla_m1` |
| `robotwin` | `main20` | `xvla_official`, `internvla_a1`, `starvla` |
| `robocasa` | `robocasa_nav` | `random_mobile`（**不在** `paper_main`） |

模型 stem：`ls configs/models/*.yaml`。  
台面三 backend 的 level：`0` 无扰动，`1` 突降，`2` 滑动障碍，`3` 人手伸入。  
RoboCasa 是厨房轮式地板障碍，不要和台面 drop / arc / shared reaching 混比。
