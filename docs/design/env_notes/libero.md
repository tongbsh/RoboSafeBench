# 当前入口更正（2026-11-10，D18）

新 H3 / R1-L 使用**原始 LIBERO**，四个官方套件各 10 任务。`LIBERO_CONFIG_PATH` 使用本文件中 **upstream** 配置目录，不使用下方历史 SafeLIBERO 配置。原始 checkout 为 `third_party/libero_upstream`；没有改动上游。环境仍为 `dynisafety-libero`，其他数值以 H3 汇报为准。以下 E00 记录保留作历史。

# LIBERO / SafeLIBERO 环境笔记（E00-libero，2026-11-09）

环境 `dynisafety-libero` 是本任务按仓库配方新建的（本机之前没有，也没有 `openvla-oft`）。克隆、配置、日志都在仓库外；没有动 RLinf 的 venv、`~/.libero/config.yaml`、`embodySafeAgent/` 下的任何文件，也没有动其他基准的环境。

**SafeLIBERO 树：用户 2026-11-09 选 vlsa-aegis `57b1aef`**（本机找不到 DyniSafety 的 fork）。旧 L3 的人手（`hand_pusher_1`）在这棵树里没有，H3 第 7 条对照表的 L3 一列空缺；找到 DyniSafety fork 后可以替换（`DYNISAFETY_LIBERO_SRC=<fork> bash scripts/setup/install_libero.sh`，或改软链）。树已原样复制到 `/data0/tongbaoshun/benchmarks/libero/safelibero_aegis57b1aef/`，主 worktree 的 `third_party/dyniSafety_libero` 软链过去。定树之前的烟雾测试是在只读源树上临时跑的（`DYNISAFETY_LIBERO_ROOT` 覆盖，源树 `git status` 始终为空）；定树之后用正式路径（不设覆盖变量）重跑了 `doctor`、L0 评测与 3 集映射回放，结果见“烟雾测试”。

## 机器

| 项 | 值 |
|---|---|
| 系统 | Ubuntu 22.04.5 LTS，内核 6.8.0-138-generic |
| GPU | 3 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition（sm_120），每张 97,887 MiB |
| 驱动 | 595.99.02（`nvidia-smi` 报 CUDA 13.2） |
| 显存（开始时） | GPU 0、GPU 2 各约 92.4 GB（用户的占位进程，可让出）；GPU 1 是别人的进程，不用 |
| 磁盘 | `/data0` 剩 4.1 TB；`/` 剩 417 GB |
| 内存 | 251 GB |

## 用了哪张卡、显存实际多少

全部只用 GPU 2（`CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2`），没有用 GPU 0 / 1。

- **EGL 渲染进程（回放、评测）**：占位进程**不让出**，两者共存。`nvidia-smi` 里我们的进程是 `G` 类型、558 MiB（`gpu_during_replay_full.log`）；10 集回放期间 GPU 2 总占用 92,955–93,121 MiB（`replay/gpu_sampling_10ep.log`）。剩下约 4.5 GB 够渲染用。
- **CUDA（torch）进程**：占位进程**让出**。torch 自检时 `mem_get_info` 报可用 51,326 / 97,249 MiB；之后 `nvidia-smi` 显示 GPU 2 只剩 18 MiB，占位进程（PID 2632503）退出；22:31:16 以新 PID 3537719 重新占回 92,362 MiB（`torch_gpu_check.log`）。在本进程里跑策略能拿到多少显存要看让出时机，没测更大的分配。

## 位置与激活

| 项 | 路径 |
|---|---|
| 上游 LIBERO 克隆 | `/data0/tongbaoshun/benchmarks/libero/LIBERO`，`8f1084e`（= `install_libero.sh` 的 `LIBERO_SHA`），743 MB |
| SafeLIBERO 树（选定） | `/data0/tongbaoshun/benchmarks/libero/safelibero_aegis57b1aef/`：vlsa-aegis（`https://github.com/THU-RCSCT/vlsa-aegis`）`57b1aef306f2` 的 `safelibero/` 原样复制，982 个文件，705 MB（736,594,107 B）。来源记录 `copy/SOURCE.txt`，sha256 清单 `copy/sha256_manifest.txt`（与源树逐文件一致）。源树 `/data0/tongbaoshun/workspace/embodySafeAgent/vlsa-aegis/safelibero` 未改动 |
| 仓库内入口 | 主 worktree `third_party/dyniSafety_libero` → 上面的副本；`third_party/libero_upstream` → 上游克隆；两个软链都在 `.git/info/exclude` 里 |
| LIBERO 配置（自建，不碰 `~/.libero`） | `/data0/tongbaoshun/benchmarks/libero/config/safelibero/config.yaml`（指副本）、`.../config/upstream/config.yaml`（指上游树）；`datasets` 都指 `/data0/dataset/LIBERO`。`.../config/_provisional_safelibero_aegis_readonly_ref/` 是定树前指源树的旧配置，只为复现旧日志保留 |
| conda 环境 | `/data0/tongbaoshun/conda/envs/dynisafety-libero`（8.3 GB） |
| `dynisafety` | editable，指向主 worktree `/data0/tongbaoshun/workspace/RoboSafeBench/src`（`hazard/integration`，与 robocasa 环境相同做法） |
| 官方演示 | `/data0/dataset/LIBERO/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}`（只读） |
| 日志、脚本、`pip freeze` | `/data0/tongbaoshun/benchmarks/_logs/e00_libero/`（脚本副本在 `scripts/`） |
| 对照用临时 venv | `/data0/tongbaoshun/benchmarks/libero/_venv_mj237`（37 MB，`--system-site-packages` + `mujoco==2.3.7`，只为下面的版本对照，可删） |

```bash
source /home/tongbaoshun/miniconda3/etc/profile.d/conda.sh
conda activate dynisafety-libero
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2
# 普通 LIBERO（上游树）：
export LIBERO_CONFIG_PATH=/data0/tongbaoshun/benchmarks/libero/config/upstream
#   并把 /data0/tongbaoshun/benchmarks/libero/LIBERO 放到 sys.path（libero 没有 pip 安装，见“已知坑”4）
# SafeLIBERO（dynisafety 后端；树由 third_party/dyniSafety_libero 解析，不设 DYNISAFETY_LIBERO_ROOT）：
export LIBERO_CONFIG_PATH=/data0/tongbaoshun/benchmarks/libero/config/safelibero
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # 见“已知坑”1
cd /data0/tongbaoshun/workspace/RoboSafeBench   # dynisafety 按当前目录找 configs/models，见“已知坑”3
```

`LIBERO_CONFIG_PATH` 是 LIBERO 自己读的变量（上游 `libero/libero/__init__.py:5-7`，默认 `~/.libero`；SafeLIBERO 树默认改成了 `~/.safelibero`）。配置文件不存在时 LIBERO 会在 import 时 `input()` 交互提问，所以必须先建好。

## 版本

| 项 | 值 |
|---|---|
| Python | 3.10.22 |
| numpy | 1.26.4（配方 `requirements/libero.txt` 没限 numpy，conda 建环境时先装上了 2.2.6，随后按 `pyproject` 的 `<2.0` 换成 1.26.4；之后所有 pip 都带 `-c constraints.txt` 钉住） |
| mujoco | 3.6.0（`requirements/libero.txt` 的 `>=3.1,<3.7` 解析结果） |
| robosuite | 1.4.0（上游 LIBERO `requirements.txt` 的固定值；SafeLIBERO 树不固定版本） |
| bddl / gym / easydict | 1.0.1 / 0.25.2 / 1.9（上游 LIBERO 固定值） |
| torch | **2.8.0+cu128**（PyPI 默认 cu128 轮子，支持 sm_120，见下） |
| numba / scipy / h5py / opencv-python | 0.68.1 / 1.15.3 / 3.16.0 / 4.11.0.86 |
| requests / msgpack / websockets / imageio-ffmpeg | 2.34.2 / 1.2.4 / 16.1.1 / 0.8.1（`dynisafety` 的传输层要，`pyproject` 没声明，见“已知坑”3） |
| dynisafety | 0.1.0，`pip install -e ".[libero]"` |

完整列表：`pip_freeze.txt`（92 行）、`conda_list.txt`；`pip check`：No broken requirements found。

**torch 的选择**：配方写 torch 2.4.1 cu124，不支持 sm_120。改装 PyPI 的 torch 2.8.0（自带 CUDA 12.8）。自检（GPU 2）：`torch.cuda.get_arch_list()` 含 `sm_120`，`torch.ones(4)*2` 求和 8.0，1024² 矩阵乘与 CPU 差 7e-5（`torch_gpu_check.log`）。`requirements/libero.txt` 其余依赖（mujoco、einops、opencv、cloudpickle）与它共存，`pip check` 无冲突。仿真与渲染不用 torch，只有 LIBERO 读 `init_files` 时用 `torch.load`。**策略走哪条路**：本环境可以在进程内跑支持 sm_120 的 torch 策略，但各 VLA 的依赖（transformers 等版本）没有核实，与 LIBERO 的 numpy 1.x / robosuite 1.4 是否兼容要按模型逐个看；默认建议策略走独立环境 + `policies/transports` 服务，这个决定留给 R1-L。
先试过从 PyTorch 官方索引装 2.7.1+cu128，经代理只有约 1.5 MB/s，13 分钟后中止（`pip_torch271_cu128_aborted.log`）；2.8.0 走阿里云镜像，全部命中本机 pip 缓存，49 s 装完。

## 安装命令（实际执行的）

```bash
# 上游 LIBERO（上一会话已克隆，HEAD = 8f1084e）
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /data0/tongbaoshun/benchmarks/libero/LIBERO   # 1 m 29 s

cd /data0/tongbaoshun/workspace/RoboSafeBench-E00-libero
conda env create -n dynisafety-libero -f environment/dynisafety-libero.yml       # 24 s，exit 0
conda activate dynisafety-libero
echo "numpy==1.26.4" > <logs>/constraints.txt
pip install "numpy==1.26.4" "torch==2.8.0"                                       # 49 s（缓存）
pip install -c <logs>/constraints.txt robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 easydict==1.9 \
    termcolor tqdm imageio h5py matplotlib pyyaml future                         # 36 s
cd /data0/tongbaoshun/workspace/RoboSafeBench && pip install -c <logs>/constraints.txt -e ".[libero]"   # 5 s
pip install -c <logs>/constraints.txt requests msgpack websockets imageio-ffmpeg # 评测时发现缺 requests
ln -s /data0/tongbaoshun/benchmarks/libero/LIBERO third_party/libero_upstream    # 主 worktree，已加 exclude
```

LIBERO 本身**没有** `pip install -e`：两个树的包名都是 `libero`，装哪一个都会和另一个打架；`dynisafety` 后端本来就靠 `ensure_libero_sys_path()` 把树插到 `sys.path[0]`（`src/dynisafety/backends/libero/env.py:32-85`），回放脚本同样用 `sys.path`。LIBERO 的训练依赖（transformers 4.21、robomimic、wandb、hydra、thop）没装，仿真用不到。

## 场景与接口事实（给 H3）

**控制与物理**（robosuite 读出并实测，两棵树相同）：`control_freq` 20 Hz，`control_timestep` 0.05 s，`model_timestep` = `opt.timestep` 0.002 s（`robosuite/macros.py:11` `SIMULATION_TIMESTEP`），每个动作 25 个子步（每步前后 `sim.data.time` 差 / `opt.timestep` 实测 25.000，所有回放步都一样）。子步循环在 `robosuite/environments/base.py:388-393`（`MujocoEnv.step`：`for i in range(int(control_timestep / model_timestep))` → `sim.forward()` → `_pre_action(action, policy_step)` → `sim.step()`），子步钩子可挂在这里或子类的 `_pre_action`（`base.py:406`）。`horizon` 1000（LIBERO `ControlEnv` 默认），控制器 OSC_POSE（`OperationalSpaceController`），动作 7 维（Δpos 3 + Δaxis-angle 3 + 夹爪，夹爪 −1 张开 / +1 闭合）。LIBERO 的 `step` 返回的 `done` 就是 `_check_success()`（`bddl_base_domain.py:800-809`）。

**状态向量**：`sim.get_state().flatten()` = `[time, qpos(nq), qvel(nv)]`。

| 场景 | nq / nv | 状态长度 | 关节数 | 刚体数 |
|---|---|---|---|---|
| 普通 LIBERO `libero_spatial`（两个任务相同） | 48 / 43 | 92 | 18 | 38 |
| SafeLIBERO `safelibero_spatial`（同两个任务） | 97 / 85 | 183 | 25 | 46 |

SafeLIBERO 多出的 7 个关节全是 free joint：`moka_pot_obstacle_1_joint0`、`white_storage_box_obstacle_1_joint0`、`milk_obstacle_1_joint0`、`wine_bottle_obstacle_1_joint0`、`red_coffee_mug_obstacle_1_joint0`、`yellow_book_obstacle_1_joint0`、`box_base_1_joint0`；多出的刚体是这 7 个物体的 `*_main` 加场景 XML 里的 `eef_marker`（无关节、`contype=conaffinity=0`、透明，SafeLIBERO 树改了 `libero_*_base_style.xml`）。普通场景的 18 个关节在 SafeLIBERO 里**全部同名存在**，所以按关节名映射是完整的。注意：物体与关节的地址**不是**简单的前缀关系，障碍物插在中间（BDDL 的对象顺序），必须按名字查 `jnt_qposadr` / `jnt_dofadr`，不能截断拼接。仓库里的 `pad_init_state_for_hand`（`env.py:290-318`）只处理“手”这一个 7 维块，不能拿来做障碍物映射。

**演示文件**（`libero_spatial`，`inspect_demo_*.log`）：
- 每个 hdf5 50 集；`data` 属性有 `bddl_file_name`、`env_args`（JSON：`env_name` Libero_Tabletop_Manipulation，OSC_POSE 参数，`control_freq` 20，相机 `robot0_eye_in_hand`、`agentview` 128×128）、`env_name`、`macros_image_convention`（opengl）、`num_demos`、`problem_info`（含语言指令）、`tag`（libero-v1）、`total`。
- 每集属性：`init_state`（92 维，与 `states[0]` 逐位相同）、`model_file`（采集时的 MJCF 字符串）、`num_samples`。**没有** `ep_meta`。
- 每集数据：`actions (T,7)`、`states (T,92)`、`robot_states (T,9)`、`rewards`、`dones`（只有末步为 1）、`obs/{agentview_rgb, eye_in_hand_rgb (T,128,128,3), ee_pos, ee_ori, ee_states, gripper_states, joint_states}`。
- 步数：`on_the_stove` 120–197（中位 137，均值 142.2）；`between_the_plate_and_the_ramekin` 75–168（中位 95，均值 101.4）。
- **对齐关系**（上游 `scripts/create_dataset.py:125-230`）：数据集是把原始人类演示重放一遍生成的，丢掉前 5 步；`states[k]` 是动作 `k` 执行**之前**的原始状态，`obs/*[k]` 是重放时动作 `k` 执行**之后**的观测。回放误差因此拿第 `k` 步之后的状态对 `states[k+1]`、拿 EEF 对 `obs/ee_pos[k]`。
- `model_file` 里物体网格的路径是采集者机器上的绝对路径（`/Users/yifengz/workspace/libero-dev/chiliocosm/assets/...`），上游的 `postprocess_model_xml` 只修 robosuite 的路径，直接 `reset_from_xml_string` 会报文件找不到；把 `.../chiliocosm/assets/` 替换成上游 `assets` 目录后可以加载。

## 烟雾测试

### 1. `doctor` 与 L0 评测（SafeLIBERO）

**正式路径（定树后，`third_party/dyniSafety_libero` → 副本，不设 `DYNISAFETY_LIBERO_ROOT`，`LIBERO_CONFIG_PATH=.../config/safelibero`，主 worktree 下执行）：**

`dynisafety doctor --backend libero`（`doctor_official.log`，exit 1）：`libero_tree`、`asset_libero_package` 变成 OK；剩下的 11 个 FAIL 是 `patches`（`doctor.py:372` 的路径错误，robocasa 也有）和 10 个本机不存在的模型服务环境（`conda_yaml:*`），都与 LIBERO 无关。

```text
OK      FAIL  python_torch_cuda         python 3.10.22; torch 2.8.0+cu128; cuda 12.8; device0=NVIDIA RTX PRO 6000 Blackwell ...
OK      FAIL  libero_tree               tree=/data0/tongbaoshun/workspace/RoboSafeBench/third_party/dyniSafety_libero; upstream_ref=8f1084e…
FAIL    FAIL  patches                   missing /data0/tongbaoshun/workspace/RoboSafeBench/scripts/apply_patches.sh
OK      FAIL  sim_mujoco                mujoco 3.6.0
OK      FAIL  numpy_mutex               numpy 1.26.4; ...
OK      FAIL  asset_libero_package      libero package under /data0/tongbaoshun/workspace/RoboSafeBench/third_party/dyniSafety_libero
OK      FAIL  conda_backend:dynisafet…  conda env dynisafety-libero (found)
OK      FAIL  gpu                       n=1; 0:NVIDIA RTX PRO 6000 Blackwell Workstation Edition 95.0GiB
summary: 35 checks, 11 fail, 14 warn
```

`dynisafety eval --backend libero --model random --suite safelibero_spatial --level 0 --task 0 --episodes 1 --output-root <logs>/eval_out_official`（`eval_l0_official.log`，exit 0，总 21.4 s）：

```text
[rescale] milk_obstacle_1 x0.85 geoms=5 meshes=1
.../third_party/dyniSafety_libero/libero/libero/benchmark/__init__.py:146: UserWarning: Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, ... forcing weights_only=False.
eval done task_id=0 episodes=1 successes=0 dir=.../eval_out_official/eval/libero/safelibero_spatial/l0/libero-random-safelibero_spatial-t0-l0-s7-20261109T145117
```

220 步，`termination: max_steps`，`success: false`，0 个接触事件，回合墙钟 16.3 s。与定树前那次比：`episodes.jsonl` 只有 `run_id`、`config_hash`、`wall_time_s` 不同；逐步 trace（221 行）只有 `run_id` 不同。

**定树前（临时，只读源树，`DYNISAFETY_LIBERO_ROOT` 覆盖）：**

`dynisafety doctor --backend libero`（`doctor.log`，exit 1）：torch/CUDA、mujoco、numpy 互斥、conda 环境、GPU 都 OK。与 LIBERO 有关的 FAIL：`libero_tree` 与 `asset_libero_package`（`doctor` 只认 `<workspace>/third_party/dyniSafety_libero` 或 `external/simulators/dyniSafety_libero`，不看 `DYNISAFETY_LIBERO_ROOT`；树定了建软链就会过）；`patches` 是 `doctor.py:372` 自己的路径错误（与 robocasa 笔记“已知坑”6 相同）。其余 FAIL 是 10 个模型服务环境（`openvla-oft`、`openpi` 等）本机不存在，与本任务无关。

```text
OK      FAIL  python_torch_cuda         python 3.10.22; torch 2.8.0+cu128; cuda 12.8; device0=NVIDIA RTX PRO 6000 Blackwell ...
FAIL    FAIL  libero_tree               tree=/data0/tongbaoshun/workspace/RoboSafeBench/third_party/dyniSafety_libero; upstream_ref=8f1084e…
FAIL    FAIL  patches                   missing /data0/tongbaoshun/workspace/RoboSafeBench/scripts/apply_patches.sh
OK      FAIL  sim_mujoco                mujoco 3.6.0
OK      FAIL  numpy_mutex               numpy 1.26.4; ...
FAIL    FAIL  asset_libero_package      libero package under /data0/tongbaoshun/workspace/RoboSafeBench/external/simulators/dyniSafety_libe…
OK      FAIL  conda_backend:dynisafet…  conda env dynisafety-libero (found)
OK      FAIL  gpu                       n=1; 0:NVIDIA RTX PRO 6000 Blackwell Workstation Edition 95.0GiB
summary: 35 checks, 13 fail, 14 warn
```

`dynisafety eval --backend libero --model random --suite safelibero_spatial --level 0 --task 0 --episodes 1 --output-root <logs>/eval_out`（主 worktree 下执行，`eval_l0.log`，exit 0，总 21.3 s）：

```text
[rescale] milk_obstacle_1 x0.85 geoms=5 meshes=1
eval done task_id=0 episodes=1 successes=0 dir=.../eval_out/eval/libero/safelibero_spatial/l0/libero-random-safelibero_spatial-t0-l0-s7-20261109T143853
```

`episodes.jsonl`：220 步（`max_steps` 220，另有 `num_steps_wait` 10），`termination: max_steps`，`success: false`，0 个接触事件，回合墙钟 16.3 s；`save_video: false`，没有视频。前两次失败与解法见“已知坑”1、3（`eval_l0_try1_no_requests.log`、`eval_l0_try2_weights_only.log`）。

附加探针（定树前，同一树内容、同一命令改 `--level`，`--no-video`）：L1 exit 0（15.7 s，`active_obstacle: moka_pot_obstacle_1`）；L2 exit 0（12.0 s，同一障碍物）；**L3 失败**：`eval failed for task_id=0 (...): 'hand_pusher_1'`（`eval_l3_probe.log`）——这棵树的 BDDL 里没有人手物体，旧 L3 跑不了。

### 2. 演示回放（普通 LIBERO，上游树）

脚本 `scripts/replay_demo.py --mode plain`：按 BDDL 建 `OffScreenRenderEnv`（128×128），`reset()` 后 `set_init_state(states[0])`，开环执行演示动作，不加等待步。任务取 `libero_spatial` 里同时属于 `safelibero_spatial` 的两个：`pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`（简称 stove）与 `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`（简称 between）。误差列：`q@1` / `q@10` 是第 1 / 10 步后全部 qpos 的最大绝对差；`robot` 是机器人关节（含两个手指）qpos 最大差（rad / m）；`obj` 是物体 free joint 位置分量的最大差（m）；`ee` 是 EEF 位置与 `obs/ee_pos` 的欧氏距离（m）。

规定的前 3 集（`summary_plain_xml.txt`）：

| 任务 | 集 | T | 成功（步） | q@1 | 末步 robot / obj | 全程最大 robot / obj | 末步 / 最大 ee |
|---|---|---|---|---|---|---|---|
| stove | 0 | 155 | **失败** | 5.4e-3 | 0.176 / 0.039 | 0.176 / 0.042 | 0.039 / 0.041 |
| stove | 1 | 125 | 111 | 5.4e-3 | 0.114 / 0.017 | 0.120 / 0.031 | 0.017 / 0.021 |
| stove | 2 | 124 | 115 | 5.4e-3 | 0.010 / 0.008 | 0.021 / 0.019 | 0.010 / 0.014 |
| between | 0 | 98 | 75 | 5.4e-3 | 0.025 / 0.010 | 0.025 / 0.015 | 0.006 / 0.007 |
| between | 1 | 84 | 71 | 5.5e-3 | 0.027 / 0.005 | 0.032 / 0.009 | 0.009 / 0.009 |
| between | 2 | 90 | 78 | 5.4e-3 | 0.020 / 0.006 | 0.020 / 0.010 | 0.003 / 0.005 |

成功 5/6。扩到每任务前 10 集（`summary_10ep.txt`）：stove 8/10（第 0、4 集失败），between 10/10；物体位置最大差 0.4–8.9 cm，EEF 最大差 0.4–4.1 cm。每步墙钟 15–18 ms（含 2 路 128×128 渲染）。

**第 1 步就有约 5.4e-3 的偏差**，几乎所有集相同：主要是两个手指关节（5.4e-3 m），其次是臂关节 1–2e-3 rad（`replay/diag_step1.log`）。两个对照排除了两种解释：
- **不是 MuJoCo 版本**：在 `mujoco==2.3.7` 的临时 venv 里重放，第 1 步误差逐位相同（`diag_step1_mj237.log`）；成功率 stove 3/3、between 3/3（3.6.0 下 stove 是 2/3），说明后面的分叉对版本敏感，第 1 步的偏差不是版本造成的（`summary_mj237.txt`）；
- **不是模型 XML 差别**：用演示自带的 `model_file`（修好路径）`reset_from_xml_string` 后重放，第 1 步误差相同，成功数相同（stove 2/3、between 3/3，`xml_*.json`）。
- 试过把夹爪控制器的内部状态预设成 −1，误差变大（`diag_step1_preset_gripper.log`），这个假设不成立。真正原因**[未核实]**；可能与求解器热启动等不在 `get_state()` 里的状态有关。

### 3. 同一初始状态放进 SafeLIBERO 场景

脚本 `--mode mapped`：先在普通场景导出关节布局（`replay/layout_plain_*.json`），在 SafeLIBERO 场景里按关节名把演示 `states[0]` 的 qpos / qvel 逐块拷过去；7 个障碍物（含 `box_base_1`）放到 (10, 10+i, 0.2)、单位四元数、速度 0（|x|、|y| 都 > 5，旧代码 `detect_visible_obstacles` 认为不可见）；时间分量照拷。误差按同样的名字映射投影回普通布局后与演示比。

- 映射后第 0 步的投影误差 0.0（初始状态完全设进去了）；障碍物落到地面后停在 z = −0.002…0.106，整集不动（`obstacle_final_pos`）。
- 前 3 集：stove 2/3（第 0 集失败）、between 3/3，与普通场景相同。between 的所有误差数字与普通场景**逐位相同**；stove 第 1 集末步 robot 误差 0.054（普通 0.114），成功步 115（普通 111），其余集相同——多出的物体改变了接触求解的顺序，个别集有微小分叉。
- 每任务 10 集：stove 8/10、between 10/10，失败的集与普通场景是同两集（`summary_10ep.txt`）。
- 每步墙钟 21–25 ms（SafeLIBERO 树的 `OffScreenRenderEnv` 多渲染一个 `backview` 相机，物体也多）。
- 以上在定树前用只读源树跑。**定树后用副本（`--tree <主 worktree>/third_party/dyniSafety_libero`，`LIBERO_CONFIG_PATH=.../config/safelibero`）重跑 3 集**（`replay/mapped_official_*.json`、`summary_mapped_official.txt`）：stove 2/3、between 3/3；6 集的所有结果字段（成功步、各步误差、障碍物终点）与定树前**完全相同**，只有墙钟不同。

结论：演示初始状态可以按关节名映射进 SafeLIBERO 场景，障碍物停远处不影响回放；回放失败来自演示本身的开环分叉，与场景替换无关。

### 4. 记录项

`control_freq` / 物理步长 / 子步数、演示集数与步数分布、演示字段见上面“场景与接口事实”。

## 磁盘占用

| 项 | 大小 |
|---|---|
| conda 环境 `dynisafety-libero` | 8.3 GB |
| 上游 LIBERO 克隆 | 743 MB |
| `_venv_mj237`（可删） | 37 MB |
| 日志目录 | < 1 MB |
| SafeLIBERO 副本 `safelibero_aegis57b1aef/` | 705 MB（含 97 MB 的 `obstacle_objects.zip`，见“已知坑”11） |

## 已知坑

1. **torch ≥ 2.6 读不了 LIBERO 的 `init_files`**：`benchmark/__init__.py` 里的 `torch.load(init_states_path)` 没传 `weights_only`，2.6 起默认 `True`，报 `Unsupported global: numpy.core.multiarray._reconstruct`。而 sm_120 需要 torch ≥ 2.7，两者没法用旧 torch 绕开。解法：启动时设 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`（torch 会打一条 UserWarning），不改上游；只对本机可信的 `init_files` 用。H3 / R1-L 的启动脚本要带上它。
2. **LIBERO 首次 import 会 `input()`**：配置文件不存在时 `libero/libero/__init__.py` 在 import 阶段交互提问，无人值守会卡住或 EOF。用 `LIBERO_CONFIG_PATH` 指向事先写好的配置（本机两份，见上）。SafeLIBERO 树的默认目录是 `~/.safelibero`，不是 `~/.libero`。
3. **`dynisafety` 的随机策略要 `requests`**：`policies/transports/http.py` 顶层 `import requests`，`pyproject` 没声明；缺了它所有适配器都注册失败，`eval` 只报 `policy/model 'random' is not registered. Available: (none)`，不说缺什么。另外策略 / 模型目录是按**当前目录**往上找 `configs/models` 的（`policies/registry.py:48-56`），要在仓库根目录下运行。
4. **两个 `libero` 包不能同时装**：上游树和 SafeLIBERO 树的包名都是 `libero`，同一进程只能有一个；普通场景与 SafeLIBERO 场景要分进程跑，配置也要分开（本机两份 `LIBERO_CONFIG_PATH`）。
5. **conda 配方装上了 numpy 2**：`environment/dynisafety-libero.yml` 只装 `requirements/libero.txt`，里面没限 numpy，解析出 2.2.6；后面装 robosuite / numba 也会再拉 numpy 2。所有 pip 都要带 `-c constraints.txt`（`numpy==1.26.4`）。
6. **演示的 `model_file` 不能直接加载**（路径是采集者机器上的），见上。按 BDDL 重建场景再设状态与用 `model_file` 结果一样，H3 用前者即可。
7. **开环回放不是逐位复现**：第 1 步就有毫米级偏差，10–15 步后增长到厘米级；本机 2 个任务 20 集里 2 集失败。H3 的探针要像 prompt 写的那样“只保留回放成功的集”，并按集记录失败原因。
8. **EGL 设备**：与 robocasa 笔记“已知坑”5 相同，`MUJOCO_EGL_DEVICE_ID` 必须出现在 `CUDA_VISIBLE_DEVICES` 里，两个都设 2。进程退出时 robosuite 的 `MjRenderContext.__del__` 会打印 `EGLError` 异常栈（`Exception ignored in ...`），不影响结果。
9. `gym` 0.25.2 每次 import 打印“Gym has been unmaintained...”，robosuite 打印“没有私有宏文件”的警告，都无影响。
10. 评测 Manifest 的 `git_dirty: true` 来自主 worktree 里未跟踪的 `docs/researchProgress.md`，与本环境无关。
11. **`assets/obstacle_objects.zip` 不是目录的重复**：没有任何文件引用它（运行时读的是 `assets/obstacle_objects/`），但逐文件对比，161 个文件里 152 个相同，3 个不同（`moka_pot_obstacle/moka_pot_handle.xml`、`moka_pot_obstacle.xml`、`moka_pot_small_obstacle.xml`），6 个只在 zip 里（`black_book_obstacle/`）。所以复制时保留了它（`copy/zip_vs_dir.txt`）。
