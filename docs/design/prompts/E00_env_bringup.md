# E00 — 基准环境搭建（按基准参数化）

起点分支：`claude/sweet-newton-2clmpw`　依赖：无　需要仿真器 / GPU：是　先读：T00。
**参数 `BENCH`**：`calvin | robocasa | libero | simplerenv | robotwin | robodojo`（协调者会告诉你）。任务分支：`hazard/E00-<BENCH>`。

## 目标
在服务器上为 `BENCH` 建好**独立**的环境，并让基准**自带的最小评测或最小 reset + step 流程**跑通。**不写任何危险体相关代码。**

## 通用要求

1. 各基准环境**互斥**（见 `docs/design/03_work_plan.md` §4），**不要**把一个基准装进另一个基准的环境，也不要动其他基准的环境。
2. 先 `nvidia-smi`、`df -h`、`python --version`、`conda env list`，把 GPU 型号 / 数量 / 驱动 / 磁盘余量写进 env_notes。单次下载 > 5 GB 或需要 `sudo` 前先问用户。**不要**改动驱动或系统 CUDA。
3. 全程记录：精确命令、版本（`pip freeze` 存到仓库外并在汇报里给路径）、安装耗时、遇到的问题与解法。
4. 第三方仓库克隆到仓库外（如 `/data/benchmarks/<name>`），记录提交号；不要把它们放进 RoboSafeBench 的 git 里。
5. 产出 `docs/design/env_notes/<BENCH>.md`：激活方式、版本表、提交号、烟雾测试命令与输出、已知坑、磁盘占用。

## 按基准的步骤（来源：各仓库 README 与本仓库文档，细节以实际仓库为准）

**calvin**（Python 3.8）
- `git clone --recurse-submodules https://github.com/mees/calvin.git`；`conda create -n calvin_venv python=3.8`；`sh install.sh`（README 提示 pyhash 安装失败时降级 setuptools 到 58 以下）。
- 数据：`cd dataset && sh download_data.sh debug`（约 1.3 GB）。
- 烟雾测试：用 `calvin_env.envs.play_table_env.get_env(<dataset>/validation, show_gui=False)` 构建环境，`reset` 到验证集的一个初始状态，执行 30 步零相对动作（7 维，夹爪保持），确认返回的 `rgb_obs`（静态相机与夹爪相机）、`robot_obs`、`scene_obs` 形状，并记录是否需要 EGL。

**robocasa**
- 使用本仓库既有的 `dynisafety-robocasa`（`make install-robocasa` 或 `scripts/setup/install_robocasa.sh`，见 `docs/user/install.md`）。需要 `MUJOCO_GL=egl`。资产约 10 GB，**下载前先问用户**。
- 烟雾测试：`dynisafety doctor --backend robocasa`；`dynisafety eval --backend robocasa --model random_mobile --suite robocasa_nav --level 0` 跑 1 个 episode；另外直接用 robocasa 的 API 构建一个原子操作任务（例如 `PickPlaceCounterToCabinet`，类名以源码为准），`reset` 后执行 20 步零动作，打印 `env.fixtures` 与 `env.objects` 的键。

**libero**（2026-11-09 改写，D16；起点分支 `hazard/integration`）
- **现状（协调者核对）**：本机**没有** README 说的 `openvla-oft` / `dynisafety-libero` conda 环境；`third_party/dyniSafety_libero` 不存在，`external/simulators/dyniSafety_libero` 也不存在。已有、**只读不改**的参考：
  - SafeLIBERO 原版树 `/data0/tongbaoshun/workspace/embodySafeAgent/vlsa-aegis/safelibero`（vlsa-aegis 仓库 `57b1aef`；有 `safelibero_{spatial,goal,object,long}` 的 BDDL、`init_files`、`assets/obstacle_objects`）；
  - RLinf 的 venv（`/data0/tongbaoshun/workspace/RLinf/.venv`、`/home/tongbaoshun/venvs/rlinf-openpi-libero`：libero、robosuite 1.4.1、mujoco 3.8.1、torch 2.11.0+cu130），`~/.libero/config.yaml` 指向 RLinf venv 里的 libero；
  - 官方演示 `/data0/dataset/LIBERO/{libero_spatial,libero_object,libero_goal,libero_10}`（hdf5，共 94 GB）。
- **SafeLIBERO 树**：`scripts/setup/install_libero.sh` 优先用 DyniSafety 的 fork（带 `safelibero_*` BDDL、人手资产、障碍物资产），否则克隆上游 LIBERO（缺这些）。先在本机找 fork（`DYNISAFETY_LIBERO_SRC`、其他 workspace）；找不到时比较 vlsa-aegis 的 SafeLIBERO 树与仓库 `src/dynisafety/backends/libero` 所需的文件（BDDL、`hand_objects`、`obstacle_objects`、`backends/libero/assets/obstacle_pack_gso`），列出差异，**先问用户**再决定用哪棵树；不要从网上找来路不明的 fork。树放仓库外（如 `/data0/tongbaoshun/benchmarks/libero/`），`third_party/dyniSafety_libero` 用软链并加进 `.git/info/exclude`（与 robocasa 的做法相同）。
- **环境**：按 `environment/dynisafety-libero.yml` 新建 `dynisafety-libero`（Python 3.10，`numpy<2`）。**注意 Blackwell sm_120**：配方里的 torch 2.4.1 cu124 不支持 sm_120（CALVIN、RoboCasa 都遇到过）。仿真与渲染不需要 GPU torch；若评测进程要在本进程里跑策略，需要支持 sm_120 的 torch（cu128 及以上），先核实 `requirements/libero.txt` 其余依赖能否共存，不能就让策略走独立环境 + 服务（`policies/transports`），写清楚选了哪条路。不要改 RLinf 的 venv。
- **渲染**：`MUJOCO_GL=egl`，只用 GPU 2（`CUDA_VISIBLE_DEVICES=2`、`MUJOCO_EGL_DEVICE_ID=2`），`nvidia-smi` 核实。
- **烟雾测试**：
  1. `dynisafety doctor --backend libero`；`dynisafety eval --backend libero --model random --suite safelibero_spatial --level 0` 跑 1 个回合，确认 SafeLIBERO 场景可加载、L0 能跑完（产出与耗时贴进汇报）；
  2. **演示回放**（D16 a 的前提）：对 `libero_spatial` 的 2 个任务各取 3 集演示，在**普通 LIBERO** 场景里按演示初始状态复位、开环执行演示动作，记录成功率与逐步状态误差（演示里存的 `states` 与回放所得之差，取末步与最大值）；
  3. **同一初始状态放进 SafeLIBERO 场景**：同一任务的 SafeLIBERO 场景多了障碍物（可能还有人手）的自由度，演示的 `states` 不能直接设。查明两者状态向量的差别，按关节 / 刚体名映射把演示初始状态设进 SafeLIBERO 场景（障碍物停放在远处），开环回放同样 3 集，记录成功率与误差。做不到就如实写原因，这是 H3 的阻塞信息；
  4. 记录：`control_freq`、物理步长、每个动作的子步数（从 robosuite 读出并实测），演示的集数与每集步数分布，演示里有没有 `model_file` / `init_state` / `ep_meta` 之类的字段。
- 回放脚本放 `scratch/`，不进 git；数值与输出写进 env_notes 与汇报。

**simplerenv**
- 使用既有环境（`make install-simplerenv`）；烟雾测试：`dynisafety doctor --backend simplerenv`；`dynisafety eval --backend simplerenv --model random --suite bridge --level 0` 1 个 episode。

**robotwin**
- 使用既有环境（`make install-robotwin`）；烟雾测试：`dynisafety doctor --backend robotwin`；`dynisafety eval --backend robotwin --model random --suite main20 --level 0` 1 个 episode。

**robodojo**（**由用户安装 Isaac Sim 环境；你不要自行安装 Isaac Sim、驱动或 CUDA**）
- 你只做核对与烟雾测试：用户装好后，在 RoboDojo 克隆里 `git submodule update --init --recursive`，`bash scripts/init_assets.sh`，`bash scripts/robodojo.sh doctor`；记录 `doctor` 全文输出。
- 机器：RTX PRO 6000；Isaac Sim 5.1 官方要求页给出的 Linux 驱动为 580.65.06、Ubuntu 22.04 / 24.04（以官方页为准）。将 `nvidia-smi` 的驱动版本与之对照并写进笔记。
- 烟雾测试：`doctor` 通过后，写一个**不带策略**的小脚本：在内存里构造配置、调用 `create_eval_env`，对任务 `stack_bowls` 执行 `reset()` 后若干步“保持”动作（动作字典格式以 `validate_action_dict` 为准），确认能无 NaN 地运行；脚本放 `scratch/`，不进 git。

## 验收标准

1. `docs/design/env_notes/<BENCH>.md` 完整。
2. 基准自带最小流程跑通，输出贴在汇报里（RoboDojo 为 `doctor` 全文 + 无策略 reset/step 烟雾测试输出）。
3. `pip freeze`（或 `conda list`）保存到仓库外，路径写入汇报。
4. 磁盘占用、耗时记录在案。

## 不要做
不要写危险体代码；不要安装 `hazardlib`（T01 之后才有）；不要升级系统组件；不要动其他基准的环境。

## 汇报
`docs/design/reports/E00-<BENCH>.md`。
