# E00-robodojo 汇报
状态：完成（doctor 核对完成；Isaac Sim 6.1 移植上的无策略烟雾测试通过）
分支 / 提交：hazard/E00-robodojo（起点 `claude/sweet-newton-2clmpw` @ 147c17a；doctor 部分在 f4df44b）
环境：现成的 `RoboDojo`（Python 3.11.16，isaacsim 5.1.0.0，torch 2.7.0+cu128）与 `robodojo61`（Python 3.12.14，isaacsim 6.1.0.0，torch 2.11.0），均来自 embodySafeAgent 项目的 `robodojo-ws`；3 × RTX PRO 6000 Blackwell，驱动 595.99.02。`pip freeze` 见下文。烟雾测试只用 `robodojo61`。

## 用户决定（2026-11-07）

1. **仿真器**：用 Isaac Sim 6.1 非官方移植（`isaac61/RoboDojo` + `robodojo61`）。代价照旧：仿真器版本成了新增的实验变量，流体任务在 6.1 上不可用，结果不是官方榜单口径。
2. **对借用环境的写入**：只允许 Kit 写 cache / logs / data（复用已有缓存，不改代码与包），跑完记录新增文件清单。清单见验收第 5 条。

## 做了什么

1. 记录机器信息：`nvidia-smi`、`df -h`、`lsb_release`、`conda env list`。
2. 定位现成安装：`robodojo-ws/RoboDojo-eval`（5.1）与 `robodojo-ws/isaac61/RoboDojo`（6.1 移植），两者都在 RoboDojo 提交 `726e9aa`，共用 39 GB 资源缓存。
3. 在两个环境里各跑一次 `bash scripts/robodojo.sh doctor`（先读过 `scripts/internal/verify_install.sh`，确认它只读）。
4. 把两个环境的 `pip freeze` 存到仓库外。
5. 读了那边项目的 `LOG.md`，提取与本机驱动、渲染、6.1 移植有关的事实，写进 env_notes 并标 [未核实]。
6. 读 6.1 克隆的 `src/eval_client/main.py`、`eval_env.py`、`scripts/eval_policy.sh` 和那边的 6.1 启动脚本，照着在 `scratch/robodojo_smoke.py` 里拼配置和启动参数（不进 git，已加进本仓库的 `.git/info/exclude`）。
7. 跑前在 `_logs/` 下建时间戳标记文件，在 GPU 1（当时唯一空闲的卡）上跑烟雾测试两次。第一次暴露了脚本漏调 `run_reward()` 的问题（见验收第 4 条），修正后第二次通过。
8. 跑后用 `find -newer <标记>` 列出借用的环境、克隆、资源缓存、家目录和 `/tmp`、`/dev/shm` 里所有新增或修改的文件。

仓库里的 `CLAUDE.md` 等文件只当文档读，没有照做。

## 产出文件

- `docs/design/env_notes/robodojo.md`
- `docs/design/reports/E00-robodojo.md`（本文件）
- 不进 git：`scratch/robodojo_smoke.py`
- 仓库外：`/data0/tongbaoshun/benchmarks/_logs/robodojo_doctor_{51,61}.{log,json}`、`robodojo_{RoboDojo,robodojo61}_pip_freeze.txt`、`robodojo_smoke_newer_files.txt`（写入清单）
- 仓库外 `/data0/tongbaoshun/benchmarks/robodojo_smoke/`：`smoke.log`、`smoke_report.json`、`gpu1.csv`（第二次），`*.attempt1.*`（第一次），`tmp/`（Isaac Lab 日志与 warp 缓存，209 MB），`eval_result/`（第一次运行留下的空 `_stream` 目录）

## 验收：命令与真实输出

**1. env_notes**：见 `docs/design/env_notes/robodojo.md`，含烟雾测试的启动方式、接口事实和注意事项。

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

**3. pip freeze**：`/data0/tongbaoshun/benchmarks/_logs/robodojo_RoboDojo_pip_freeze.txt`（306 行）、`robodojo_robodojo61_pip_freeze.txt`（279 行）。

**4. 无策略烟雾测试（6.1，`stack_bowls`）**

启动环境和命令见 env_notes“烟雾测试”一节，外面套 `timeout 1800`，后台每 5 s 采一次 GPU 1 显存。脚本的步骤：

1. 解析参数后强制 `headless`、`enable_cameras`，`kit_args` 加上 `eval_policy.sh` 启用的两个扩展，再用 RoboDojo 自己的 `add_zero_delay_kit_args` 补零延迟渲染设置，然后 `AppLauncher(args)`。
2. 在 `sys.modules` 里放一个空的 `client_server.ws.model_client`（`WsModelClient` 是只记录调用的桩），再导入 `create_eval_env`。原因见“未解决与阻塞”第 2 条。
3. 按 `main.py` 拼 `env_cfg`：`arx_x5.yml` 读出 `eval_cfg`，`num_envs=1`、`eval_batch=False`、`seed=0`、`physx_monitor_enabled=False`（`stack_bowls` 配置里没有 `Articulation`，`main.py` 本来也会关掉监控）；`deploy_cfg` 只填占位值；再过 `process_randomization` 与 `process_config`。
4. `create_eval_env` → `seed_manager.get_seeds(max_count=1)` → `reset(seed=[0])` → `run_reward()`、`get_score()`（照 `run_eval()` 开头的做法）。
5. 保持动作：每条目标臂的当前关节角 + 当前夹爪开度换算回 [0, 1]，先过 `validate_action_dict`，再对同一个动作调 20 次 `take_action`。
6. 每次之后检查：两条臂的 `joint_pos` / `joint_vel` / 末端位姿、三个碗的位置和姿态、`obs_manager.get_obs()` 的全部数值字段；同时记录物理步计数、`end_flag`、关节偏移、碗位移。
7. `env.close()`、`app.close()`，最后 `os._exit`（同 `main.py`）。

第二次运行的日志关键行（全文 416 行在 `robodojo_smoke/smoke.log`）：

```
start 2026-11-07T11:14:18+08:00 python=/data0/tongbaoshun/workspace/embodySafeAgent/robodojo-ws/conda/envs/robodojo61/bin/python VK=/usr/share/vulkan/icd.d/nvidia_icd.json
[INFO][AppLauncher]: Using device: cuda:0
[10.564s] app ready
[smoke] ok=True rc=0 report=/data0/tongbaoshun/benchmarks/robodojo_smoke/smoke_report.json
[41.023s] Simulation App Shutting Down
RUN_DONE rc=0 elapsed_s=42 end 2026-11-07T11:15:00+08:00
```

`smoke_report.json` 摘录：

```
"ok": true,
"timing_s": {"app_launch": 11.23, "create_eval_env": 14.76, "reset": 11.36, "hold_steps": 2.29, "env_close": 1.08, "total": 40.85},
"model_client": "NullModelClient", "model_client_calls": ["reset"], "registered_checks": 1,
"robot_action_dim_info": {"arm_dim": [6, 6], "ee_dim": [1, 1]}, "step_lim": 800, "collect_interval": 10, "sim_dt_s": 0.004,
"reset_tries": [{"seeds": [0], "result": "ok"}], "object_labels": ["bowl0", "bowl1", "bowl2"], "initial_nonfinite": [],
"hold_action": {"left_arm_joint_state": [5.1e-13, -1.2e-15, 1.8e-17, 7.9e-16, -1.0e-12, 2.5e-12], "left_ee_joint_state": [0.9999944076493935],
                "right_arm_joint_state": [5.1e-13, -1.2e-15, 1.8e-17, 7.9e-16, -1.0e-12, 2.5e-12], "right_ee_joint_state": [0.9999944076493935]}
{"step": 1,  "physics_steps": 10, "sim_time_s": 0.04, "take_action_cnt": 1,  "end_flag": false, "nonfinite": [], "max_joint_dev_rad": 0.0, "max_joint_vel": 7.929571665954427e-08, "max_object_disp_m": 0.0}
{"step": 2,  "physics_steps": 10, "sim_time_s": 0.08, "take_action_cnt": 2,  "end_flag": false, "nonfinite": [], "max_joint_dev_rad": 0.0, "max_joint_vel": 7.929545375873204e-08, "max_object_disp_m": 0.0}
...（第 3–19 步同样 physics_steps=10、nonfinite=[]、end_flag=false）
{"step": 20, "physics_steps": 10, "sim_time_s": 0.8,  "take_action_cnt": 20, "end_flag": false, "nonfinite": [], "max_joint_dev_rad": 0.0, "max_joint_vel": 7.929541112616789e-08, "max_object_disp_m": 0.0}
```

（`hold_action` 里的关节角在这里保留两位有效数字，原文是完整的 float。）

判定：20 次 `take_action` 全部执行（`take_action_cnt` 1→20），每次 10 个物理步，共 0.8 s 仿真时间；初始状态和每一步之后，所有浮点字段都有限；`end_flag` 始终为 false。观测里三路相机各出一张 480×640×3 的图，说明 6.1 在 595 驱动上能渲染。Kit 日志 0 条 `[Error]`、37 条 `[Warning]`（`omni.rtx` 12、`rtx.neuraylib.plugin` 4、`carb.windowing-glfw.plugin` 4、`carb` 4、`isaacsim.sensors.camera.camera` 3 等），日志里没有 NaN / Inf 字样。

**第一次运行（`*.attempt1.*`）为什么不算**：它同样 `ok=True`、退出码 0、69 s（冷缓存），但脚本当时没调 `run_reward()`。`get_reward()` 在检查列表为空时返回 1.0，第一次 `take_action` 后 `is_episode_end()` 就判成功，后面 19 次被静默跳过：

```
{"step": 1, "take_action_cnt": 1, "end_flag": true, ...}
{"step": 2, "take_action_cnt": 1, "end_flag": true, ...}
```

修正方法是照 `run_eval()` 补调 `run_reward()` / `get_score()`，同时把判定改严：要求 `take_action_cnt` 等于请求步数、每步物理步数大于 0、最后 `end_flag` 为 false。

GPU 1 显存峰值（5 s 采样，可能漏掉瞬时峰值）：第一次 18.9 GB（结束时触发了视频流写入），第二次 8.5 GB。跑前 GPU 1 占用 18 MiB。

**5. 写入清单**（`find -newer _logs/robodojo_smoke.marker`，标记建于 11:12:31，两次运行都在其后；全文 `_logs/robodojo_smoke_newer_files.txt`）

| 位置 | 新增 / 修改 | 归属 |
|---|---|---|
| `robodojo61/.../site-packages/isaacsim/kit/logs/` | `Kit/Isaac-Sim/5.1/kit_20261107_111235.log`、`kit_20261107_111418.log`、`omni.telemetry.transmitter.log` | Kit 日志（批准范围内） |
| `robodojo61/.../isaacsim/kit/cache/` | `Kit/110.3/00c488ae/material_cache.json`、`DerivedDataCache/`（`cachelru.dat` 与 3 个实例锁）、`nv_shadercache/GLCache/.../9cd601fd3f14d49d.{bin,toc}` | Kit 缓存（批准范围内） |
| `robodojo61/.../isaacsim/kit/data/` | `Kit/Isaac-Sim/5.1/user.config.json` | Kit 用户设置文件，Kit 退出时写回（批准范围内；目录名里的 5.1 是移植沿用的应用名） |
| `isaac61/RoboDojo`（克隆） | 只有 `.git/modules/third_party/curobo` 这个目录本身的 mtime 变了（11:14:29，第二次运行的 `create_eval_env` 阶段），目录下所有文件的 mtime 都没变 | 推测是导入链里某处调了 git，短暂建了又删了锁文件；具体调用方没找到 [未核实]。没有文件内容改变 |
| `XPolicyLab`（软链目标）、资源缓存 `robodojo_assets_hf` | 无 | — |
| `~/.nvidia-omniverse/` | `logs/omni.{kit.app,kit,kit.sysinfo,processlifetime,replicator.extinfo}.log`、`pycache/` | Kit 日志与缓存（在家目录，不在借用环境里） |
| `~/.nv/ComputeCache/` | 261 个条目，时间落在两次运行期间（11:12:48–11:14:36） | CUDA 驱动的 JIT 缓存 |
| `/dev/shm/sem.carbonite-sharedmemory` | 新增 | Kit（Carbonite）信号量 |
| `~/.npm`、`~/.nvm`（11:14:00–11:14:04）、`~/.local/state/labmon`（11:15:06）、`/tmp/cc-daemon-1010/` | 有变动 | 时间落在两次运行之间或之后，或属于别的工具，与本次运行无关 |

另外：6.1 克隆的 `.git` 目录本身 mtime 是今天 10:24，早于标记，也早于烟雾测试；目录下没有文件被改。推测来自之前 doctor 或只读 `git status` 建删 `index.lock`。

**6. 磁盘与耗时**：资源缓存 39 GB，`RoboDojo` 环境 21 GB，`robodojo61` 环境 31 GB；都不是本任务安装的，没有安装耗时可记；doctor 每次约 1.9 s；烟雾测试 42 s（热缓存）/ 69 s（冷缓存），运行目录 `robodojo_smoke/` 约 210 MB（主要是 `tmp/`）。

## 偏离设计文档之处与理由

1. **没有运行 `git submodule update --init --recursive` 与 `init_assets.sh`**：两个克隆都有那边项目的本地修改，运行它们会改动别人的工作区；资源已经齐全，doctor 的资源检查全部通过。
2. **没有自己克隆、自己建环境**：用户指明“conda 列表里已有 RoboDojo 环境，先跑 doctor 核对”；Isaac Sim 本来也约定由用户安装。
3. **烟雾测试用的是 6.1 非官方移植，不是 E00 默认的 5.1**：用户决定，理由见 doctor 部分的驱动对照。
4. **策略客户端在导入层面换成了桩**：E00 只要求“不带策略”，但原样导入 `eval_env.py` 在这套借用环境里会失败（见下）。桩只替换 `client_server.ws.model_client`，RoboDojo 自己的代码一行没动。
5. **除 `reset()` 外还调了 `run_reward()` / `get_score()`**：不调的话 `stack_bowls` 第一步就被判成功，保持动作根本不执行。这是照 `run_eval()` 的做法，不是额外的改动。

## 未解决与阻塞

1. 两个现成环境属于另一个项目。H6 要往环境里 `pip install -e hazardlib --no-deps`，直接装进去会改动别人的环境。
2. **范围外问题（只记录，未修）**：`robodojo61` 里 `client_server` 导入不了。环境中 XPolicyLab 的可编辑安装（`__editable___xpolicylab_0_0_1_finder.py`）把 `client_server` 映射到 `isaac61/RoboDojo/XPolicyLab/client_server`；那边项目后来把 `isaac61/RoboDojo/XPolicyLab` 换成了指向 RoboProbe 版 XPolicyLab 的软链（原目录移到 `.XPolicyLab.upstream-d6332bf`），而 RoboProbe 版的 `client_server` 在 `utils/` 下。按 `eval_policy.sh` 的 `PYTHONPATH` 实测 `import client_server` 报 `ModuleNotFoundError`。推论（未实跑）：这套借用环境里，RoboDojo 原生的 `eval_policy.sh` → `main.py` 评测路径会在导入阶段失败。这是那边项目的布局，不影响本任务的结论，但支持下面第 2 条建议（H6 用自己的克隆，子模块用上游 XPolicyLab）。
3. 烟雾测试只覆盖了“静止保持”：200 个物理步里机器人关节和三个碗的位置都逐位不变（推测是 PhysX 休眠加夹爪在关节限位 [未核实]），没有覆盖接触和运动下的数值稳定性。

## 对设计文档的修订建议（只提议，不改文件）

1. `E00_env_bringup.md` 与 `H6_robodojo.md` 默认的是官方 Isaac Sim 5.1。本机驱动下 5.1 无法渲染 [未核实，依据为那边 LOG 和公开报告]，所以 RoboDojo 这一列的“引擎”实际可能是 Isaac Sim 6.1 移植。若采用，`02_benchmark_profiles.md` 的 RoboDojo 档案、BD 的 `engine.version`、论文里的引擎列表都要写明“6.1 非官方移植”，并把流体任务排除在外。
2. H6 开工前建议：在 `/data0/tongbaoshun/benchmarks/robodojo` 下另建自己的克隆（同一提交 `726e9aa`），资源目录以只读软链复用那份 39 GB 缓存（不重新下载）；环境用 `conda create --clone` 复制一份（本地复制，不下载，约 21–31 GB 磁盘），再装 `hazardlib`。这样不碰那边项目的克隆和环境。
3. 核对时 GPU 0 与 GPU 2 各被占用 80–97 GB，GPU 1 较空。另外，`robodojo-ws/env.sh` 写明 GPU 1 是那个项目的卡（Isaac Sim 与 vLLM 共用）；本次烟雾测试借用 GPU 1 约 2 分钟。H6 与 X1 排期时要和那边约定 GPU。
4. H6 的 RoboDojo 适配器要把这次发现的隐式宿主要求写进 01 §3.3 的清单：
   - 不走 `run_eval()` 时，`reset()` 之后必须调 `run_reward()`（以及 `get_score()`），否则 `get_reward()` 对空检查列表返回 1.0，回合立刻判成功。
   - 一次 `take_action` 是 `collect_interval` 个物理步（`stack_bowls` 上是 10 步 × 0.004 s）。HazardRuntime 的 `pre_step` / `post_step` 要挂在物理步上，不能挂在 `take_action` 上；`max_step_ratio` 的检查也要按 0.004 s 算。
   - 放置或释放危险物体时要唤醒它和可能接触的刚体（本次观察到静止场景完全不动，推测有休眠 [未核实]）。
   - 物理默认跑在 CPU（`BaseEnv` 的 `DEFAULT_SIM_DEVICE = "cpu"`），CCD 等 PhysX 设置要在 `sim.physx` 里显式给出。`DEFAULT_PHYSX_CONFIG` 里 `enable_ccd` 默认是 False，和 R1 的 `hazard_ccd` 有关。
5. `E00_env_bringup.md` 的 RoboDojo 烟雾测试一节建议补一句：“若不经 `run_eval()`，需在 `reset()` 后调用 `run_reward()`，并检查 `take_action_cnt` 确实递增”，否则烟雾测试会假通过（本次第一次运行就是这样）。

## 下一步建议

1. H6 开工前按第 2 条另建自己的克隆与环境（`conda create --clone` 约 31 GB 本地复制，不下载；子模块用上游 XPolicyLab），在自己的环境里重跑这个烟雾测试，作为 H6 的基线。
2. 和 embodySafeAgent 那边约定 GPU 1 的使用时段，或等 GPU 0 / GPU 2 有空。
3. 视需要在 H6 里补一个“非静止”烟雾测试：让机械臂走一小段关节轨迹并碰到一只碗，检查接触下无 NaN。
