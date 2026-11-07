# E00-calvin 汇报
状态：完成（有两个问题要协调者定，见下）
分支 / 提交：hazard/E00-calvin（起点 `claude/sweet-newton-2clmpw` @ 147c17a）
环境：新建的 `calvin_venv`（Python 3.8.20，pybullet 3.2.7，torch 1.13.1+cu117，numpy 1.23.5）；calvin `fa03f01`，calvin_env `797142c`，tacto `dd53360`；3 × RTX PRO 6000 Blackwell，驱动 595.99.02。

## 要协调者定的事

1. **calvin_env 用哪个提交。** 子模块检出的是上游 main `797142c`，超级仓库锁定的是 `1431a46`，相差 7 个提交。两处差异与我们直接相关：`1431a46` 会原地缩放调用方传入的动作数组；复位到灯亮的状态时，`1431a46` 会把按钮状态错设成 OFF。烟雾测试两个版本结果逐位相同，但这个测试没有覆盖这两处差异（灯是灭的，零动作）。建议用 `797142c`，并写进 Manifest 的 `benchmark_commits`。细节见 env_notes“已知坑”第 6 条。
2. **H1 的策略环境。** torch 1.13.1 在 sm_120 上不报错但会算错，第二次测试直接挂住（env_notes“已知坑”第 3 条），所以 `calvin_venv` 里只能在 CPU 上跑 torch。H1 如果要在 GPU 上跑 CALVIN 策略，需要另开一个 Python ≥ 3.9 的策略环境，与仿真环境分开。这会影响 H1 的工作量估计。

## 做了什么

1. 记录机器信息（`nvidia-smi`、`df -h`、`conda env list`）。
2. 按 README 克隆（`--recurse-submodules`），建 `calvin_venv`，跑 `install.sh`。`install.sh` 跑不通，前后补了三处：cmake 版本、setuptools 降级、tokenizers 只用二进制轮子。
3. 下载 debug 数据（1.3 GB）。
4. 烟雾测试第一次在 import 阶段就失败（numpy 1.24 删掉了 `np.int`），把 numpy 降到 1.23.5 后通过。
5. 对照测试：不加载 EGL 插件跑一次；临时切到锁定版本 `1431a46` 跑一次（为此装了 `rich`），跑完切回 `797142c`。
6. 定位复位误差落在哪几维；核实 torch 在 Blackwell 上的行为；查明进程退出时挂住的原因。

没有写任何危险体代码，没有装 `hazardlib`，没有动其他基准的环境，也没有用 sudo。期间杀掉的几个挂住的进程都是本任务自己起的。

## 产出文件

- `docs/design/env_notes/calvin.md`
- `docs/design/reports/E00-calvin.md`（本文件）
- 仓库外 `/data0/tongbaoshun/benchmarks/_logs/`：
  - 安装：`install_calvin{,_fix,_fix2}.{sh,log}`、`install_calvin_numpy_pin.log`、`install_calvin_rich.log`、`download_calvin_debug.{sh,log}`
  - 版本：`calvin_venv_pip_freeze.txt`
  - 烟雾测试：`calvin_smoke.py`、`calvin_smoke_egl.log`、`calvin_smoke_egl_pin1431a46.log`、`calvin_smoke_noegl.log`、`calvin_smoke_noegl_pyrender_egl.log`
  - 复位误差：`calvin_reset_diff.py`、`calvin_reset_diff{,_close}.log`

## 验收：命令与真实输出

**1. env_notes**：见 `docs/design/env_notes/calvin.md`，包括激活方式、版本表、提交号、烟雾测试命令与输出、已知坑、磁盘占用。

**2. 最小 reset + step 流程**

```bash
cd /data0/tongbaoshun/benchmarks/calvin && conda activate calvin_venv
python /data0/tongbaoshun/benchmarks/_logs/calvin_smoke.py $PWD/dataset/calvin_debug_dataset/validation
```

输出（去掉 pybullet 启动行与 gym 停更提示）：

```
EGL device choice: -1 of 7.
Loaded EGL 1.5 after reload.
GL_VENDOR=NVIDIA Corporation
GL_RENDERER=NVIDIA RTX PRO 6000 Blackwell Workstation Edition/PCIe/SSE2
GL_VERSION=3.3.0 NVIDIA 595.99.02
use_egl=True first validation episode start=553567 (of 1 episodes)
stored robot_obs (15,) scene_obs (24,)
env built in 1.9 s: PlayTableSimEnv
bullet_time_step=None action_repeat=8 control_freq=30
reset tcp error vs stored: 1.05e-03 m
scene_obs error vs stored (max abs): 4.12e-03
  depth_obs.depth_gripper  (84, 84)         float32
  depth_obs.depth_static   (200, 200)       float32
  depth_obs.depth_tactile  (160, 120, 2)    float32
  rgb_obs.rgb_gripper      (84, 84, 3)      uint8
  rgb_obs.rgb_static       (200, 200, 3)    uint8
  rgb_obs.rgb_tactile      (160, 120, 6)    uint8
  robot_obs                (15,)            float64
  scene_obs                (24,)            float64
30 steps of [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] in 1.37 s (45.5 ms/step); all finite=True
tcp drift after 30 zero steps: 8.36e-04 m
  (30 步后各键形状与上面相同)
  rgb_gripper mean=137.7 std=64.3 (non-blank image check)
  rgb_static mean=144.9 std=66.4 (non-blank image check)
  rgb_tactile mean=149.8 std=25.5 (non-blank image check)
disconnecting id 0 from server
real	0m4.129s
```

`bullet_time_step=None` 只是脚本取错了属性名，配置里是 `bullet_time_step: 240.0`，与 `action_repeat=8 × control_freq=30` 一致。

**是否需要 EGL**：需要。不加载 EGL 插件、也不设 `PYOPENGL_PLATFORM` 时，tacto 的 pyrender 去连 X display，抛 `NoSuchDisplayException` 后进程挂住。设 `PYOPENGL_PLATFORM=egl` 后，PyBullet 相机可以退回 CPU TinyRenderer（67.9 ms/步），物理数值与 EGL 逐位相同，但图像不同（`rgb_static` 均值 133.0 vs 144.9）。数据集是 EGL 录的，所以评测也用 EGL。

**对照：锁定版本 `1431a46`**（`calvin_smoke_egl_pin1431a46.log`）：通过，53.6 ms/步，复位误差、漂移、图像均值与 `797142c` 逐位相同。

**复位误差定位**（`calvin_reset_diff.log`）：

```
pink_rx        stored= 0.0000 reset= 0.0041 diff= 4.12e-03
blue_ry        stored=-0.0000 reset=-0.0039 diff=-3.85e-03
red_rx         stored=-0.0015 reset= 0.0001 diff= 1.59e-03
blue_rx        stored=-0.0000 reset= 0.0015 diff= 1.51e-03
red_ry         stored=-0.0007 reset=-0.0012 diff=-5.59e-04
pink_ry        stored=-0.0000 reset=-0.0002 diff=-1.76e-04
robot_obs max abs diff idx 3 2.16e-03; gripper stored/reset 1.0 1.0
lights stored [0. 0.]
```

**torch 在 sm_120 上**（`CUDA_VISIBLE_DEVICES=1`）：

```
UserWarning: NVIDIA RTX PRO 6000 Blackwell Workstation Edition with CUDA capability sm_120 is not compatible with the current PyTorch installation.
arch_list ['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86'] capability (12, 0)
cuda op ok 0.0          # torch.ones(4).cuda()*2 求和，应为 8
```

第二次试 `torch.arange(4., device="cuda")`：120 s 无输出，被 `timeout` 杀掉。

**3. `pip freeze`**：`/data0/tongbaoshun/benchmarks/_logs/calvin_venv_pip_freeze.txt`（112 行，`pip check` 无冲突）。

**4. 磁盘与耗时**：克隆含数据 2.6 GB（数据 2.5 GB，其中 zip 1.3 GB 可删），`calvin_venv` 4.9 GB。`install.sh` 56 s（失败），两次重试分别 2 m 10 s（失败）与 1 m 02 s，下载 3 m 44 s，烟雾测试单次约 4 s。

## 偏离 prompt 的地方

- prompt 说 pyhash 失败时“降级 setuptools 到 58 以下”，实际还要补 cmake 和 tokenizers 两处，另外要降 numpy。这些都只改了 `calvin_venv`，没有改上游代码。
- 为了做对照测试，在 `calvin_venv` 里多装了 `rich`。它对 `797142c` 无影响，留着是为了随时能切回 `1431a46`。

## 给 H1 的提示

- 保持动作就是零相对动作：`use_target_pose: true` 时相对量累加在目标位姿上，零动作让目标不动，实测 30 步漂移 0.84 mm。
- 复位到存档状态时，方块姿态会有约 0.24° 的偏差，铰接物体与灯的状态是精确的。G1 / G2 的可重复性检查要把这个量级考虑进去，但阈值不由本任务定。
- 宿主要显式 `env.close()`，否则进程可能在退出时挂住。
- 支撑面在 `scene.surfaces` 里有 `table`、`slider_left`、`slider_right` 三个，slide 族的支撑面约束（T06 发现 5）可以从这里取。
- EGL 设备号要像 CALVIN 的 `play_lmp_wrapper.set_egl_device` 那样显式设，否则用的是默认设备。
