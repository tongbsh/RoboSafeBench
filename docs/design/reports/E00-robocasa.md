# E00-robocasa 汇报
状态：完成。四件事里前三件用户已定（固定 `921c9a5`、纹理与生成纹理换官方版、不下 `aigen_objs`），第 4 件留给 H2，见末尾“后续”
分支 / 提交：hazard/E00-robocasa（起点 `claude/sweet-newton-2clmpw` @ 147c17a）
环境：新建的 `dynisafety-robocasa`（Python 3.11.18，mujoco 3.3.1，numpy 2.2.5，robosuite 1.5.2 `5ce6643`，robocasa 1.0.1 `921c9a5`，torch 2.7.1+cu126）；3 × RTX PRO 6000 Blackwell，驱动 595.99.02；渲染固定 GPU 2。

## 要协调者定的事

1. **robocasa 用哪个提交。** 仓库安装脚本与 `docs/user/install.md` 固定 `921c9a5`，H2 prompt 写 `3443cdf`（上游当前 main）。`921c9a5` 是 `3443cdf` 的祖先，差 14 个提交，版本号都是 1.0.1；运行时代码只有 `sink.py` 删掉一个重复定义的函数，其余是子任务标注流水线、脚本、测试和 README；H2 引用的 `kitchen.py` 行号两边都对得上。本环境是 `921c9a5`（安装脚本强制切回）。建议沿用 `921c9a5`，把 H2 prompt 的固定值改过来，不改安装脚本。
2. **资产来源。** 官方 Box 链接已失效，安装脚本回退到第三方 HF 镜像下载了纹理、生成纹理、lightwheel 固定装置与物体；这些文件与官方 `robocasa/robocasa-assets` 的哈希都不同，对应哪个上游版本未核实。objaverse 是我从官方仓库下的（sha256 已核对）。两个选择：保持现状（导航后端已有结果用的就是镜像资产），或把纹理与生成纹理换成官方版本（约 1.7 GB；lightwheel 两类官方 HF 仓库里没有，只能用镜像）。我倾向保持现状，并在 Manifest 里记录资产来源。
3. **`aigen_objs` 要不要下。** 官方仓库里是单个 5.77 GB 的 zip，超过“单次 > 5 GB 先问”，我没下。`Kitchen` 默认物体库是 `objaverse` + `lightwheel`，烟雾测试不需要它。
4. **H2 的探针素材。** 官方演示数据的 Box 链接同样 404；HF 上有第三方转存，大小与一致性未核实。H2 第 9 条要求下载前先问用户，env_notes 里列了候选。

## 做了什么

1. 记录机器信息（`nvidia-smi`、`df -h`、`python --version`、`conda env list`）。本机原先没有 `dynisafety-robocasa`。
2. robosuite、robocasa 克隆到仓库外，在主 worktree 的 `third_party/` 下建软链（安装脚本只认 `${ROOT}/third_party`；软链加进 `.git/info/exclude`）。
3. 按仓库配方建环境（`environment/dynisafety-robocasa.yml`），跑 `scripts/setup/install_backend.sh robocasa`：一次通过，7 m 35 s；轻量资产全部来自 HF 镜像（Box 报 SSL EOF）。
4. 烟雾测试：`doctor`；`robocasa_nav` 评测 1 回合；原子任务 `PickPlaceCounterToCabinet` 第一次在 `reset` 失败（轻量资产里没有 objaverse 物体）。
5. 查明失败原因后，先用 `obj_registries=("lightwheel",)` 验证原子任务本身能跑；再从官方 HF 仓库下载 objaverse（2.16 GB，按 LFS sha256 校验），默认配置通过。中途发现最初用的是第三方镜像的 objaverse，就停下改用官方版本，未完成的下载已删除。
6. 装 objaverse 之后重跑导航评测，结果逐位相同；记录 `tests/test_backends_robocasa.py` 的基线（给 H2 的“前后一致”用）。
7. 核对两个 robocasa 提交的差异、EGL 设备选择逻辑、`doctor` 两项失败的原因。

没有写危险体代码，没有装 `hazardlib`，没有动其他基准的环境，没有用 sudo，没有改驱动、CUDA 或系统包，没有改上游代码（安装脚本跑的 `setup_macros` 在克隆里生成了上游 gitignore 的 `macros_private.py`）。

## 产出文件

- `docs/design/env_notes/robocasa.md`
- `docs/design/reports/E00-robocasa.md`（本文件）
- 仓库外 `/data0/tongbaoshun/benchmarks/_logs/e00_robocasa/`：
  - 安装：`clone_{robosuite,robocasa}.log`、`conda_create.log`、`install_backend.log`、`download_objaverse.log`、`download_objaverse_livfour_aborted.log`
  - 版本：`pip_freeze.txt`（142 行）、`conda_list.txt`、`pip_check.log`、`system_info.log`
  - 烟雾测试：`doctor.log`、`eval_nav.log`、`eval_nav_after_objaverse.log`、`eval_out/`、`eval_out_after_objaverse/`、`scripts/smoke_atomic.py`、`smoke_atomic.log`、`smoke_atomic_lw.log`、`smoke_atomic_before_objaverse.log`、`pytest_backends_robocasa.log`
  - GPU：`gpu_uuids.log`、`eval_nvidia_smi_apps.log`

## 验收：命令与真实输出

**1. env_notes**：见 `docs/design/env_notes/robocasa.md`，包括激活方式、版本表、提交号、资产来源、烟雾测试命令与输出、给 H2 的场景事实、已知坑、磁盘占用。

**2. 基准自带的最小流程**

`dynisafety doctor --backend robocasa`（exit 1，RoboCasa 相关项全部 OK）：

```text
OK      FAIL  robocasa_tree             path=.../third_party/robocasa head=921c9a5736a8d0ea558965789…
OK      FAIL  robosuite_tree            path=.../third_party/robosuite head=5ce6643f3092639d08f7b0f9…
FAIL    FAIL  patches                   missing .../scripts/apply_patches.sh         git checkout HEAD -- scripts/setup/apply_patches.sh
OK      FAIL  sim_robocasa              robocasa 1.0.1
OK      FAIL  sim_robosuite             robosuite 1.5.2 (git master, not PyPI 1.4)
OK      FAIL  sim_mujoco_robocasa       mujoco 3.3.1 (need 3.3.x; robocasa pin 3.3.1)
OK      FAIL  numpy_mutex               numpy 2.2.5 (robocasa pin 2.2.5; mutually exclusive with libero/simplerenv/robotwin)
OK      FAIL  asset_robocasa_tex        tex .../robocasa/models/assets/textures…
OK      FAIL  asset_robocasa_objs_lw    objs_lw .../robocasa/models/assets/obje…
FAIL    FAIL  conda_yaml:mibot          configs/models references env 'mibot' (missing); ...
summary: 19 checks, 2 fail, 3 warn
```

（节选；第二列是严重度。）两个 FAIL：`patches` 是 `doctor.py:372` 的路径写错（找 `scripts/apply_patches.sh`，脚本在 `scripts/setup/`），`mibot` 是 xiaomi 模型的服务环境，都与 RoboCasa 无关。

`dynisafety eval --backend robocasa --model random_mobile --suite robocasa_nav --level 0 --task 0 --episodes 1 --output-root <logs>/eval_out`（exit 0，54.6 s）：

```text
eval done task_id=0 episodes=1 successes=0 dir=.../robocasa-random_mobile-robocasa_nav-t0-l0-s7-20261107T083727
```

回合记录：`nav_sink`，450 步跑满（`max_steps`），未成功（随机策略），0 个接触事件，`wall_time_s` 28.3。

原子任务（`python smoke_atomic.py`，`create_env("PickPlaceCounterToCabinet", seed=0)`，`reset` 后 20 步零动作）：

```text
pid 218474
versions: robocasa 1.0.1 robosuite 1.5.2 mujoco 3.3.1 numpy 2.2.5
env: MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2
create_env('PickPlaceCounterToCabinet', seed=0, {})
create 0.0s reset 10.8s
class PickPlaceCounterToCabinet robots ['WheeledRobot:PandaOmron']
control_freq 20 model_timestep 0.002 substeps/control 25
layout_id 12 style_id 32
action_dim 12
lang Pick the hot dog from the counter and place it in the cabinet.
fixtures (114): ['bottom_right_group_1_1', 'bottom_right_group_1_base', 'box_1_island_group', 'box_1_left_group', ...]
objects (3): ['distr_cab', 'distr_counter', 'obj']
nbody 384 ngeom 2027; ...
robot-side bodies: {'robot0_': 10, 'gripper0_': 6, 'mobilebase0_': 4, 'panda0_': 0}
20 zero-action steps in 0.9s; obs finite: True; reward 0.0; success False
eef drift over 20 steps: 0.000004 m
base body mobilebase0_base drift: 0.000000 m
agentview_left image mean 89.65 std 65.55
open /dev/nvidia* fds: ['/dev/nvidia-modeset', '/dev/nvidia0', '/dev/nvidia1', '/dev/nvidia2', '/dev/nvidiactl']
|    2   N/A  N/A          218474      G   python                                  865MiB |
```

最后一行是环境存活时的 `nvidia-smi`：本进程只在 GPU 2 上（G 类型，EGL）。完整的 114 个 fixture 键与 32 个观测键见 `smoke_atomic.log`。

装 objaverse 之前同一命令的失败（`smoke_atomic_before_objaverse.log`）：

```text
  File ".../robocasa/models/objects/kitchen_object_utils.py", line 463, in sample_kitchen_object_helper
    chosen_reg = rng.choice(
ValueError: Probabilities contain NaN
```

**3. `pip freeze`**：`/data0/tongbaoshun/benchmarks/_logs/e00_robocasa/pip_freeze.txt`（142 行）。`pip check` 只报 `dynisafety` 的 `numpy<2.0` 声明，配方有意用 `--no-deps` 绕开。

**4. 磁盘与耗时**：环境 7.2 GB；robosuite 克隆 1.3 GB；robocasa 克隆 11 GB（其中资产 10 GB）；objaverse zip 2.1 GB（已解压，可删）。克隆 2 m 35 s（两个并行），建环境约 1 分钟（pip 轮子多数在缓存里），安装脚本 7 m 35 s，objaverse 下载加校验加解压 2 m 25 s，原子任务烟雾测试单次 13 s（`reset` 约 11 s），导航评测 1 回合 44–55 s。会话总长约 30 分钟（16:18 克隆开始，16:46 写文档）。

## 偏离 prompt 的地方

- prompt 说克隆放仓库外；安装脚本只认 `${ROOT}/third_party/`。我把克隆放在 `/data0/tongbaoshun/benchmarks/robocasa/`，在主 worktree 的 `third_party/` 下建软链，并加进 `.git/info/exclude`。
- 安装脚本从主 worktree（`hazard/integration` @ `84961b4`）跑，`dynisafety` 的 editable 安装因此指向主 worktree，而不是本任务分支。两者的 `src/dynisafety`、`scripts/setup`、`requirements`、`environment`、`docs/user` 完全相同（`git diff` 为空）；主 worktree 不会被删，环境不会因为本分支的 worktree 被清理而失效。
- 下了 objaverse（2.16 GB）：仓库脚本默认只下轻量资产，但原子任务离不开它；用户已同意约 10 GB 资产。`aigen_objs` 没下（见“要协调者定的事”第 3 条）。

## 给 H2 的提示

- 激活后一定要 `export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2`。`MUJOCO_EGL_DEVICE_ID` 会覆盖 robosuite 的 `render_gpu_device_id` 参数。
- 物理步 0.002 s，控制 20 Hz，每个控制步 25 个子步。子步钩子的位置（`_pre_action` 是否每个子步调用）本任务没有查，留给 H2 第 3 条。
- 布局随种子程序化生成，`reset` 一次约 11 s（建场景加编译模型），探针预算要按这个算。
- 机器人侧刚体前缀实测是 `robot0_`（10 个）、`gripper0_`（6 个）、`mobilebase0_`（4 个）；H2 prompt 列的 `panda0_` 在本场景里没有。
- 零动作 1 s 内 EEF 漂移 4–5 µm，底盘不动；H2 第 7 条要的是“保持”动作（夹爪保持），零向量对夹爪的含义要再核对。
- 原子任务默认从 objaverse + lightwheel 采样物体；如果以后要只用 lightwheel，建环境时传 `obj_registries=("lightwheel",)`，但物体类别会少到 58 个。
- `import dynisafety` 在本环境里解析到主 worktree 的 `src/dynisafety`（H2 第 6 条要复用 `mujoco_contact`）。H2 自己的 worktree 里改 `hazardlib` 不受影响（`PYTHONPATH`）。
- `tests/test_backends_robocasa.py` 的基线：`40 passed, 1 skipped`（缺 SMPL 模型）。
- 纹理已换成官方版。上面原子任务输出里的图像均值 89.65 是换之前的数，换之后是 97.43（见“后续”）。

## 后续（2026-11-07，用户定了前三件事之后）

用户的决定：robocasa 沿用 `921c9a5`，H2 prompt 由协调者改；纹理与生成纹理换成官方版；`aigen_objs` 不下。

替换纹理的过程：
1. 从官方 `robocasa/robocasa-assets@1b92c3d` 下载 `textures.zip`（543 MB）与 `generative_textures.zip`（1185 MB），用时 1 m 38 s。sha256 与 LFS oid 一致（`download_official_textures.log`：两行 `OK`）。
2. 解压前逐文件比对镜像与官方：
   - 镜像是官方的子集，同名文件逐字节相同；
   - 镜像缺 `textures/window_bkgs/` 的 5 张图（代码未引用），以及生成纹理里全部 450 个 `tex*.png`。官方的 `tex*.png` 都是同目录 UUID 文件的副本。
3. 旧目录移到 `_downloads/mirror_backup/`，解压官方版本：`textures` 225 个文件，`generative_textures` 900 个文件。

发现一个问题：替换前，生成纹理目录里有 400 个 `tex*.png` 软链，其中 399 个指错了文件。
- 软链是仓库导航后端的 `ensure_generative_texture_tex_aliases` 建的（`src/dynisafety/backends/robocasa/env.py:1187-1209`）。第一次导航评测时它把排序后的 UUID 文件依次链成 `tex001.png`…，每类 100 个（`eval_out` 记 `gentex_tex_aliases` 每类 100）。
- fixture 注册表在默认模式下也引用这些名字，所以替换前地板、墙、柜体、台面的纹理都不对，原子任务的图像也因此不同。
- 物理不受影响：导航评测的终点逐位相同。
- 用镜像资产的机器都会这样。代码在 `src/dynisafety`，按约束没改，记给仓库维护。

替换后重跑（GPU 2）：

| 测试 | 结果 |
|---|---|
| `doctor --backend robocasa` | 与之前相同：14 OK / 3 WARN / 2 FAIL（`patches`、`mibot`，与 RoboCasa 无关） |
| 导航评测 1 回合 | exit 0，45 s；`base_xy_end`、底盘关节终值与之前逐位相同；新建软链 0 个 |
| 原子任务（默认） | exit 0；layout 12 / style 32、物体、漂移与之前相同；图像均值 / 标准差 89.65 / 65.55 → 97.43 / 70.05 |
| 原子任务 `--generative-textures 100p` | exit 0；116.80 / 49.44 |

对照实验确认图像的变化来自生成纹理：
- 两个纹理目录都换回镜像，复现出 89.65 / 65.55；
- 只把 `textures` 换回镜像，仍是 97.43 / 70.05；
- 官方版连跑两次都是 97.43 / 70.05。

对照实验后已恢复为官方版（`textures` 225 个文件，`generative_textures` 900 个，软链 0 个），镜像备份已删除。新增日志：`download_official_textures.log`、`doctor_official_tex.log`、`eval_nav_official_tex.log`、`eval_out_official_tex/`、`smoke_atomic_official_tex.log`、`smoke_atomic_gen100p.log`、`ab_official_{1,2}.log`、`ab_mirror.log`、`ab_mirror_both.log`。

磁盘：资产约 11 GB；下载缓存 3.8 GB（3 个 zip，可删）。
