# E00-robodojo 汇报
状态：部分完成（doctor 核对完成；烟雾测试 BLOCKED，等用户决定仿真器版本）
分支 / 提交：hazard/E00-robodojo（起点 `claude/sweet-newton-2clmpw` @ 147c17a）
环境：现成的 `RoboDojo`（Python 3.11.16，isaacsim 5.1.0.0，torch 2.7.0+cu128）与 `robodojo61`（Python 3.12.14，isaacsim 6.1.0.0，torch 2.11.0），均来自 embodySafeAgent 项目的 `robodojo-ws`；3 × RTX PRO 6000 Blackwell，驱动 595.99.02。`pip freeze` 见下文。

## BLOCKED：烟雾测试需要用户决定用哪个 Isaac Sim

- **现象**：本机驱动 595.99.02，Isaac Sim 5.1 官方要求 580.65.06。`robodojo-ws/LOG.md` 第 197 行记录：5.1 在 R590 / R595 驱动上一渲染就在 `librtx.scenedb.plugin.so` 段错误，headless 也一样，官方答复是降驱动。E00 的烟雾测试（`create_eval_env` → `reset()` → 保持动作）会真正启动渲染。
- **已经做的**：在两套现成安装上跑了只读的 `doctor`，都是 0 fail（全文见下）。没有启动 Isaac Sim，也没有复现段错误。
- **需要用户决定**：
  1. 用 6.1 非官方移植（`robodojo61`）做烟雾测试和后续 H6。代价：仿真器版本成了新增的实验变量；流体任务（如 `pour_liquid_into_cup`）在 6.1 上不可用；结果不是官方榜单口径。
  2. 由管理员把驱动降到 580.65.06 后用官方 5.1。这需要 sudo，影响整台机器和其他用户，我不能做。
  3. 先跑一次 5.1 烟雾测试，确认段错误在本机确实复现（只为留下证据，预计失败）。
  4. RoboDojo 暂缓，H6 往后排。

## 做了什么

1. 记录机器信息：`nvidia-smi`、`df -h`、`lsb_release`、`conda env list`。
2. 定位现成安装：`robodojo-ws/RoboDojo-eval`（5.1）与 `robodojo-ws/isaac61/RoboDojo`（6.1 移植），两者都在 RoboDojo 提交 `726e9aa`，共用 39 GB 资源缓存。
3. 在两个环境里各跑一次 `bash scripts/robodojo.sh doctor`（先读过 `scripts/internal/verify_install.sh`，确认它只读）。
4. 把两个环境的 `pip freeze` 存到仓库外。
5. 读了那边项目的 `LOG.md`，提取与本机驱动、渲染、6.1 移植有关的事实，写进 env_notes 并标 [未核实]。

仓库里的 `CLAUDE.md` 等文件只当文档读，没有照做。

## 产出文件

- `docs/design/env_notes/robodojo.md`
- `docs/design/reports/E00-robodojo.md`（本文件）
- 仓库外：`/data0/tongbaoshun/benchmarks/_logs/robodojo_doctor_{51,61}.{log,json}`、`robodojo_{RoboDojo,robodojo61}_pip_freeze.txt`

## 验收：命令与真实输出

**1. env_notes**：见 `docs/design/env_notes/robodojo.md`。其中烟雾测试一节标为“尚未做”。

**2. doctor 全文**

5.1（`conda activate RoboDojo`，在 `RoboDojo-eval` 下）：

```
python3=/data0/tongbaoshun/workspace/embodySafeAgent/robodojo-ws/conda/envs/RoboDojo/bin/python3 Python 3.11.16
[verify_install] root=/data0/tongbaoshun/workspace/embodySafeAgent/robodojo-ws/RoboDojo-eval
[verify_install] policy checks disabled
[PASS] env package - .../RoboDojo-eval/env
[PASS] robodojo launcher - .../RoboDojo-eval/scripts/robodojo.sh
[PASS] task inventory helper - .../RoboDojo-eval/scripts/internal/task_inventory.py
[PASS] env_cfg - .../RoboDojo-eval/env_cfg/arx_x5.yml
[PASS] task config - .../RoboDojo-eval/task/RoboDojo/config/stack_bowls.yml
[PASS] robot assets - .../RoboDojo-eval/Assets/Robots
[PASS] object assets - .../RoboDojo-eval/Assets/Object/RoboDojo
[PASS] eval layouts - .../RoboDojo-eval/Assets/Eval_Layout/RoboDojo
[PASS] materials - .../RoboDojo-eval/Assets/Material
[WARN] policy checks - skipped; pass --policy-dir to validate a policy
[PASS] task inventory - {'config_only': 0, 'missing_class': 0, 'missing_config': 0, 'runnable': 54, 'tasks': 54}
[PASS] env_cfg references - all referenced sim/scene/robot/camera files exist
[WARN] policy processor - skipped; pass --policy-dir to validate
[PASS] python import - env.global_configs imports
[PASS] sim conda env - RoboDojo
[WARN] policy conda env - skipped; pass --policy-env to validate
[PASS] Isaac imports - isaacsim and isaaclab import in RoboDojo
[verify_install] summary=/data0/tongbaoshun/benchmarks/_logs/robodojo_doctor_51.json
[verify_install] pass=14 warn=3 fail=0
real	0m1.900s
```

6.1（`conda activate robodojo61`，在 `isaac61/RoboDojo` 下，加 `--sim-env robodojo61`）：

```
python3=/data0/tongbaoshun/workspace/embodySafeAgent/robodojo-ws/conda/envs/robodojo61/bin/python3 Python 3.12.14
[verify_install] root=/data0/tongbaoshun/workspace/embodySafeAgent/robodojo-ws/isaac61/RoboDojo
[verify_install] policy checks disabled
[PASS] env package - .../isaac61/RoboDojo/env
[PASS] robodojo launcher - .../isaac61/RoboDojo/scripts/robodojo.sh
[PASS] task inventory helper - .../isaac61/RoboDojo/scripts/internal/task_inventory.py
[PASS] env_cfg - .../isaac61/RoboDojo/env_cfg/arx_x5.yml
[PASS] task config - .../isaac61/RoboDojo/task/RoboDojo/config/stack_bowls.yml
[PASS] robot assets - .../isaac61/RoboDojo/Assets/Robots
[PASS] object assets - .../isaac61/RoboDojo/Assets/Object/RoboDojo
[PASS] eval layouts - .../isaac61/RoboDojo/Assets/Eval_Layout/RoboDojo
[PASS] materials - .../isaac61/RoboDojo/Assets/Material
[WARN] policy checks - skipped; pass --policy-dir to validate a policy
[PASS] task inventory - {'config_only': 0, 'missing_class': 0, 'missing_config': 0, 'runnable': 54, 'tasks': 54}
[PASS] env_cfg references - all referenced sim/scene/robot/camera files exist
[WARN] policy processor - skipped; pass --policy-dir to validate
[PASS] python import - env.global_configs imports
[PASS] sim conda env - robodojo61
[WARN] policy conda env - skipped; pass --policy-env to validate
[PASS] Isaac imports - isaacsim and isaaclab import in robodojo61
[verify_install] summary=/data0/tongbaoshun/benchmarks/_logs/robodojo_doctor_61.json
[verify_install] pass=14 warn=3 fail=0
real	0m1.913s
```

（为了可读性，把重复的 `robodojo-ws` 前缀缩写成 `...`；原始日志未改动。）

驱动对照：

```
index, name, driver_version, memory.used [MiB], memory.total [MiB]
0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 595.99.02, 82460 MiB, 97887 MiB
1, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 595.99.02, 13305 MiB, 97887 MiB
2, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 595.99.02, 92388 MiB, 97887 MiB
```

官方要求 580.65.06，本机是 595.99.02，**不一致**。

**无策略烟雾测试**：未做，见 BLOCKED。

**3. pip freeze**：`/data0/tongbaoshun/benchmarks/_logs/robodojo_RoboDojo_pip_freeze.txt`（306 行）、`robodojo_robodojo61_pip_freeze.txt`（279 行）。

**4. 磁盘与耗时**：资源缓存 39 GB，`RoboDojo` 环境 21 GB，`robodojo61` 环境 31 GB；都不是本任务安装的，没有安装耗时可记；doctor 每次约 1.9 s。

## 偏离设计文档之处与理由

1. **没有运行 `git submodule update --init --recursive` 与 `init_assets.sh`**：两个克隆都有那边项目的本地修改，运行它们会改动别人的工作区；资源已经齐全，doctor 的资源检查全部通过。
2. **没有自己克隆、自己建环境**：用户指明“conda 列表里已有 RoboDojo 环境，先跑 doctor 核对”；Isaac Sim 本来也约定由用户安装。

## 未解决与阻塞

- 烟雾测试等待用户决定仿真器版本（见顶部）。
- 两个现成环境属于另一个项目。H6 要往环境里 `pip install -e hazardlib --no-deps`，直接装进去会改动别人的环境。

## 对设计文档的修订建议（只提议，不改文件）

1. `E00_env_bringup.md` 与 `H6_robodojo.md` 默认的是官方 Isaac Sim 5.1。本机驱动下 5.1 无法渲染 [未核实，依据为那边 LOG 和公开报告]，所以 RoboDojo 这一列的“引擎”实际可能是 Isaac Sim 6.1 移植。若采用，`02_benchmark_profiles.md` 的 RoboDojo 档案、BD 的 `engine.version`、论文里的引擎列表都要写明“6.1 非官方移植”，并把流体任务排除在外。
2. H6 开工前建议：在 `/data0/tongbaoshun/benchmarks/robodojo` 下另建自己的克隆（同一提交 `726e9aa`），资源目录以只读软链复用那份 39 GB 缓存（不重新下载）；环境用 `conda create --clone` 复制一份（本地复制，不下载，约 21–31 GB 磁盘），再装 `hazardlib`。这样不碰那边项目的克隆和环境。
3. 核对时 GPU 0 与 GPU 2 各被占用 80–90 GB，GPU 1 较空。H6 与 X1 排期时要考虑与其他项目共用 GPU。

## 下一步建议

1. 用户在 BLOCKED 一节的四个选项里选一个。
2. 选定后补跑烟雾测试（脚本放 `scratch/`，不进 git），并按上面第 2 条另建克隆与环境。
