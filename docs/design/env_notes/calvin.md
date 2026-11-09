# CALVIN 环境笔记（E00-calvin，2026-11-07）

本任务新装的独立环境：克隆在仓库外，conda 环境 `calvin_venv` 只给 CALVIN 用，没有动其他基准的环境。

## 机器

| 项 | 值 |
|---|---|
| 系统 | Ubuntu 22.04.5 LTS，内核 6.8.0-138-generic |
| GPU | 3 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition（sm_120），每张 97,887 MiB |
| 驱动 | 595.99.02 |
| 核对时的显存占用 | GPU 0 约 82.8 GB、GPU 1 约 13.5 GB、GPU 2 约 92.4 GB（其他项目的进程） |
| 磁盘 | `/data0` 剩 4.1 TB；`/` 剩 102 GB（已用 88%） |

## 位置与激活

| 项 | 路径 |
|---|---|
| 克隆 | `/data0/tongbaoshun/benchmarks/calvin` |
| 数据 | `/data0/tongbaoshun/benchmarks/calvin/dataset/calvin_debug_dataset/{training,validation}` |
| conda 环境 | `/data0/tongbaoshun/conda/envs/calvin_venv` |
| 日志、脚本、`pip freeze` | `/data0/tongbaoshun/benchmarks/_logs/`（`*calvin*`） |

```bash
source /home/tongbaoshun/miniconda3/etc/profile.d/conda.sh
conda activate calvin_venv
```

无头渲染不需要额外环境变量：数据集配置里 `env.use_egl: true`，`PlayTableSimEnv` 会自己加载 PyBullet 的 `eglRenderer` 插件并设置 `PYOPENGL_PLATFORM=egl`。

## 版本

| 项 | 值 |
|---|---|
| Python | 3.8.20 |
| calvin（超级仓库） | `fa03f01`（2025-09-08） |
| calvin_env 子模块 | **`797142c`**（上游 main，2022-12-23）。超级仓库锁定的是 `1431a46`（2022-04-29），见“已知坑”第 6 条 |
| tacto 子模块 | `dd53360` |
| pybullet | 3.2.7 |
| torch / torchvision | 1.13.1+cu117 / 0.14.1（**GPU 不可用**，见“已知坑”第 3 条） |
| numpy | 1.23.5（手动从 1.24.4 降下来，见“已知坑”第 1 条） |
| hydra-core / omegaconf | 1.1.1 / 2.1.2 |
| gym | 0.26.2 |
| pyrender / urdfpy / networkx | 0.1.45 / 0.0.22 / 2.2 |
| transformers / tokenizers | 4.46.3 / 0.20.3 |
| pytorch-lightning | 1.8.6 |
| setuptools / cmake | 57.5.0 / 3.18.4.post1 |
| rich | 14.3.4（只有锁定版本 `1431a46` 需要，为对照测试装的） |

完整列表：`/data0/tongbaoshun/benchmarks/_logs/calvin_venv_pip_freeze.txt`（112 行）。`pip check` 无冲突。

## 安装命令（实际执行的）

```bash
cd /data0/tongbaoshun/benchmarks
git clone --recurse-submodules https://github.com/mees/calvin.git
conda create -n calvin_venv python=3.8 -y && conda activate calvin_venv
cd calvin && sh install.sh                       # 失败：见下面第 2 条，tacto 与 calvin_env 已装上
pip install "setuptools==57.5.0" wheel "cmake==3.18.4.post1"
cd calvin_models && pip install -e .             # 失败：tokenizers 0.20.4 没有 cp38 轮子
pip install -e . --only-binary=tokenizers        # 成功，解析到 tokenizers 0.20.3
pip install "numpy==1.23.5"                      # 否则 import 失败，见第 1 条
cd ../dataset && sh download_data.sh debug       # 1.3 GB zip，解压后保留 zip
```

脚本与完整日志：`_logs/install_calvin{,_fix,_fix2}.{sh,log}`、`install_calvin_numpy_pin.log`、`install_calvin_rich.log`、`download_calvin_debug.{sh,log}`。

| 步骤 | 耗时 |
|---|---|
| `install.sh`（失败在 calvin_models） | 56 s |
| 第一次重试 | 2 m 10 s（失败） |
| 第二次重试 | 1 m 02 s |
| 下载 debug 数据 | 3 m 44 s |

## 磁盘占用

| 项 | 大小 |
|---|---|
| 克隆（含数据） | 2.6 GB |
| 其中 `dataset/` | 2.5 GB（zip 1.3 GB + 解压后约 1.2 GB；zip 可删） |
| `calvin_venv` | 4.9 GB |

## 场景与接口事实（给 H1 用）

- debug 数据：training 1 段（帧 358482–361252），validation 1 段（帧 553567–555241），场景都是 `calvin_scene_D`，`global_scaling: 0.8`。
- 控制频率 30 Hz（`control_freq: 30`），`action_repeat=8`，物理 240 Hz（`bullet_time_step: 240.0`）。
- 动作：7 维 numpy 数组按相对笛卡尔位移处理（位置 3、欧拉角 3、夹爪 ±1），由 `Robot.relative_to_absolute` 乘 `max_rel_pos` / `max_rel_orn` 缩放；`use_target_pose: true`，相对量累加在目标位姿上而不是当前位姿上。`797142c` 另外支持字典动作 `{"action": ..., "type": "cartesian_rel|cartesian_abs|joint_rel|joint_abs"}`。
- 观测（烟雾测试实测）：

| 键 | 形状 | dtype |
|---|---|---|
| `rgb_obs.rgb_static` | (200, 200, 3) | uint8 |
| `rgb_obs.rgb_gripper` | (84, 84, 3) | uint8 |
| `rgb_obs.rgb_tactile` | (160, 120, 6) | uint8 |
| `depth_obs.depth_static` | (200, 200) | float32 |
| `depth_obs.depth_gripper` | (84, 84) | float32 |
| `depth_obs.depth_tactile` | (160, 120, 2) | float32 |
| `robot_obs` | (15,) | float64 |
| `scene_obs` | (24,) | float64 |

- `robot_obs`：末端位置 3、末端欧拉角 3、夹爪开口宽度 1、关节角 7、夹爪动作 1（README）。
- `scene_obs`（`PlayTableScene.get_obs` 的拼接顺序）：滑门、抽屉、按钮、开关、灯泡、绿灯，然后 `block_red`、`block_blue`、`block_pink` 各 6 维（位置 3 + 欧拉角 3）。
- 支撑面（配置 `scene.surfaces`）：`table`、`slider_left`、`slider_right`。

## 烟雾测试

脚本 `/data0/tongbaoshun/benchmarks/_logs/calvin_smoke.py`（仓库外）：`get_env(<validation>, show_gui=False)`，`reset` 到 validation 第一段起始帧存的 `robot_obs` / `scene_obs`，执行 30 步 `[0, 0, 0, 0, 0, 0, g]`（`g` 取当前夹爪状态的符号），打印形状、检查有限性、测 TCP 漂移。

```bash
cd /data0/tongbaoshun/benchmarks/calvin && conda activate calvin_venv
python /data0/tongbaoshun/benchmarks/_logs/calvin_smoke.py $PWD/dataset/calvin_debug_dataset/validation            # EGL
PYOPENGL_PLATFORM=egl python .../calvin_smoke.py $PWD/dataset/calvin_debug_dataset/validation --no-egl             # PyBullet 用 CPU TinyRenderer
```

| 配置 | 结果 | 每步 | 复位误差（TCP / scene_obs 最大） | 30 步后 TCP 漂移 |
|---|---|---|---|---|
| EGL，calvin_env `797142c` | 通过，全部有限，图像非空 | 45.5 ms | 1.05e-3 m / 4.12e-3 | 8.36e-4 m |
| EGL，calvin_env `1431a46` | 通过，数值与上一行逐位相同 | 53.6 ms | 同上 | 同上 |
| 不加载 EGL 插件 + `PYOPENGL_PLATFORM=egl` | 通过，物理数值相同，**图像不同** | 67.9 ms | 同上 | 同上 |
| 不加载 EGL 插件，不设 `PYOPENGL_PLATFORM` | **失败**：tacto 的 pyrender 去连 X display，`NoSuchDisplayException`，随后进程挂住 | — | — | — |

结论：无头运行需要 EGL；PyBullet 相机本身可以退回 TinyRenderer，但 tacto 触觉传感器的 pyrender 仍要 EGL（或 OSMesa，未试）。复位差异逐项定位见 `_logs/calvin_reset_diff.log`：全部在三个方块的 rx / ry 欧拉角上（最大 4.1e-3 rad，约 0.24°），滑门、抽屉、按钮、开关、灯的状态完全一致；`robot_obs` 最大差 2.2e-3（第 3 维，末端 roll）。

## 已知坑

1. **numpy ≥ 1.24 时 `import` 就失败**：`urdfpy` 依赖的 `networkx 2.2` 在 `readwrite/graphml.py:346` 用了 `np.int`，报错链经 `calvin_env.camera.tactile_sensor.TactileSensor` 抛出。`tacto/renderer.py:469` 也有 `np.float`。降到 `numpy==1.23.5` 解决，与 numba（≥1.22）、scipy、pandas、matplotlib 的约束都兼容。
2. **`install.sh` 在 2026 年跑不通**：`cmake==3.18.4` 已从索引下架（用 `3.18.4.post1`）；pyhash 依赖 `use_2to3`，要 `setuptools<58`（calvin_models 的 requirements 自己就锁了 57.5.0）；tokenizers 0.20.4 没有 cp38 轮子、源码包要 Rust，加 `--only-binary=tokenizers`。
3. **torch 1.13.1 在 Blackwell（sm_120）上不能用，而且会悄悄算错**：编译架构只到 sm_86，只给一条 UserWarning，不报错。实测 `torch.ones(4, device="cuda")*2` 求和得 0.0（应为 8）；第二次试 `torch.arange` 直接挂住 120 s 被超时杀掉。所以 `calvin_venv` 里**不能在 GPU 上跑策略**。PyTorch 对 sm_120 的支持从 2.7（cu128）开始，而 torch 从 2.5 起不再支持 Python 3.8 [未在本机核实]，Python 3.8 环境里大概率没有能用的 GPU torch。E00 的烟雾测试不需要 torch GPU；H1 若要在 GPU 上跑 CALVIN 策略，需要另开一个 Python ≥ 3.9 的策略环境，与仿真环境分开。
4. **env 不显式 `close()` 时进程可能在退出时挂住**：`env` 若是模块级全局变量，`__del__ → close() → p.disconnect` 会拖到解释器收尾时执行，实测挂住直到超时；`env` 是 `main()` 的局部变量时能正常退出；显式 `env.close()` 后 2.4 s 正常退出。宿主实现里要显式 `close()`。构造到一半抛异常时也会挂住（见上表最后一行）。
5. **EGL 与 TinyRenderer 渲染出的图像不同**（同一状态下 `rgb_static` 均值 144.9 vs 133.0）。数据集是按 `use_egl: true` 录的，评测与数据生成要用 EGL，不要在两种渲染器之间混用。
6. **calvin_env 版本**：子模块检出的是上游 main `797142c`，比超级仓库锁定的 `1431a46` 新 7 个提交（`d517115` 去掉 rich、`6e7ceaa` 换场景后物体尺寸、`27a27a2` 动作在 step 后被改写、`5a0eb8a` 关节空间控制、`cc6c8c9` / `77f853c` 补 `__init__.py`、`797142c` 复位时按钮状态）。与我们相关的差异：
   - `1431a46` 的 `relative_to_absolute` 会**原地缩放调用方传入的动作数组**，调用方之后再读这个数组（比如写进日志）拿到的是缩放后的值；`797142c` 先 `np.copy`。
   - `1431a46` 复位时先复位按钮再复位灯，且按钮状态一律设为 OFF；`797142c` 先复位灯，按钮状态跟随灯。复位到灯亮的 `scene_obs` 时两者行为不同。
   - `1431a46` 在 import 时 `rich.traceback.install(show_locals=True)`，需要 `rich`。
   本次烟雾测试（灯是灭的、零动作）两个版本结果逐位相同，所以上面两处差异都没有被这个测试覆盖。用哪个版本需要协调者定，定下后写进 Manifest 的 `benchmark_commits`。
7. **EGL 设备选择**：日志里是 `EGL device choice: -1 of 7`，`get_env` 不设 `EGL_VISIBLE_DEVICES`，`CUDA_VISIBLE_DEVICES` 也不影响它。CALVIN 自己的 `play_lmp_wrapper.set_egl_device` 会根据 CUDA 设备号找对应的 EGL 设备号。本次实际渲染在哪张卡上没有核实 [未核实]。
8. 无害的警告：gym 0.26.2 的停更提示；`urdfpy/urdf.py:2169` 的 `invalid value encountered in divide`。

## H1 补充（H-calvin，2026-11-07）

**测试依赖**（H1 第 12 条）：`calvin_venv` 里用 `pip install pytest` 新装了 `pytest 8.3.5`（Python 3.8 下解析到的版本），连带 `exceptiongroup 1.3.1`、`iniconfig 2.1.0`、`pluggy 1.5.0`、`tomli 2.5.0`；没有装或升级别的包。`PYTHONPATH=<worktree>/hazardlib/src python -m pytest hazardlib/tests -q -p no:cacheprovider`：1357 passed, 1 skipped（跳过的是 `test_source_hygiene.py:103`，需要 3.10+ 的 `sys.stdlib_module_names`），132 s。

**EGL 设备（解决第 7 条的 [未核实]）**：`hazardhosts.calvin.probes.egl_devices` 列出 7 个 EGL 设备，EGL 序号 0/1/2 对应 CUDA 0/1/2，序号 3–5 是 Mesa 的 DRM 设备。`nvidia-smi` 的 GPU 2 是 PCI `00000000:41:00.0`。`CalvinHost` 在建 env 前设置 `EGL_VISIBLE_DEVICES=2`（PyBullet `eglRendererPlugin`）和 `EGL_DEVICE_ID=2`（pyrender，即 tacto），日志变成 `EGL device choice: 2 of 7 (from EGL_VISIBLE_DEVICES)`。运行期间 `nvidia-smi` 的进程表里，该进程只出现在 GPU 2 上，类型 G，约 256–264 MiB，无论带不带触觉相机都一样；它不在计算进程列表里（`--query-compute-apps`）。

**导入 `calvin_agent.evaluation.*` 失败**：环境里的 `fsspec 2025.3.1` 在 `fsspec/caching.py:40` 用了 `list[[int, int]]`，Python 3.8 下报 `TypeError: 'type' object is not subscriptable`。适配器的原生评测循环（`hazardhosts/calvin/native.py`）因此不 import 这些模块，而是从固定提交 `fa03f01` 的 `evaluation/utils.py`、`evaluate_policy.py` 里按名字取出顶层定义执行，并在执行前检查这两个文件相对固定提交没有改动。

**同种子复现**：上游 `env.reset(robot_obs, scene_obs)` 本身不能保证跨回合复现。同一布局、同一动作序列，中间插入别的回合之后，从第 8 个控制步起开始不同，`rgb_static` 最多差 187 个灰度级，`scene_obs` 最多差 2.7e-5。PyBullet 的 `saveState` / `restoreState` 也不行：从第 0 个控制步起就不同，`scene_obs` 最多差 0.073。CALVIN 自带的 `serialize` / `reset_from_storage` 差得更多：`robot_obs` 3.5e-4，`scene_obs` 0.42。另外 `reset_from_storage` 要求 pickle 里有录制文件才有的 `state_obs` / `done` / `info` 三个键，直接喂 `serialize()` 的结果会报 `KeyError`。只有每回合重建整个 PyBullet 世界（`CalvinHost.reset_episode`，每次约 0.08 s）才能让 4 次重复的所有观测逐位相同。复现命令：`python -m hazardhosts.calvin.probes.determinism ... [--no-rebuild [--restore bullet|storage]]`。

**滚动摩擦**：PyBullet 动态体的滚动摩擦默认是 0。释放到桌面上的球体会以释放速度的约 5/7 一直滚下去，5 s 内不减速，直到滚出桌面落到地面。

**退出时挂住（第 4 条的又一种情况）**：建 env 时如果带触觉相机（tacto / pyrender），`env.close()` 之后 `PlayTableSimEnv.__del__` 会再调一次 `close()`，抛 `pybullet.error: Not connected to physics server.`，进程随后挂住，只能 kill。`CalvinHost` 只开静态相机和夹爪相机（`configs/hazard/calvin.yaml` 的 `cameras`）。

**判定器与接触**：危险体只是停放、或运动学放置但不接触任何东西时，`robot_obs`、`scene_obs`、整个 `get_info()` 以及所有任务的判定结果都与无危险运行逐位一致。危险体只落在桌面上时，只有 `get_info()['scene_info']['fixed_objects']['table']` 里桌子的接触信息不同，其余仍逐位一致。危险体碰到机械臂时，它会出现在 `robot_info['contacts']` 里，但判定结果不变。危险体碰到被举起的方块时，`lift_*` 判定为假，因为判定器读取方块的接触列表。危险体停放后立即读 `get_info()`，不会残留旧的接触（`probes/oracle_isolation.py`）。
