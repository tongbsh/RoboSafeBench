# RoboCasa 环境笔记（E00-robocasa，2026-11-07）

环境 `dynisafety-robocasa` 是本任务按仓库配方新建的（本机之前没有）。克隆和资产都在仓库外，没有动其他基准的环境。

## 机器

| 项 | 值 |
|---|---|
| 系统 | Ubuntu 22.04.5 LTS，内核 6.8.0-138-generic |
| GPU | 3 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition（sm_120），每张 97,887 MiB |
| 驱动 | 595.99.02（`nvidia-smi` 报 CUDA 13.2） |
| 核对时的显存占用 | GPU 0 88,812 MiB（其他用户）；GPU 1 19 MiB；GPU 2 1,456 MiB（用户的 `gfnvla` 进程 2 × 712 MiB），我们的渲染进程上卡后再加约 0.87 GB |
| 磁盘 | `/data0` 剩 4.1 TB；`/` 剩 102 GB（已用 88%） |

## 位置与激活

| 项 | 路径 |
|---|---|
| robosuite 克隆 | `/data0/tongbaoshun/benchmarks/robocasa/robosuite` |
| robocasa 克隆（含资产） | `/data0/tongbaoshun/benchmarks/robocasa/robocasa`，资产在 `robocasa/models/assets/` |
| 仓库内的入口 | 主 worktree 的 `third_party/robosuite`、`third_party/robocasa` 是指向上面两个克隆的软链（`.git/info/exclude` 里排除，不进 git） |
| conda 环境 | `/data0/tongbaoshun/conda/envs/dynisafety-robocasa` |
| `dynisafety` | editable 安装，指向主 worktree `/data0/tongbaoshun/workspace/RoboSafeBench/src`（`hazard/integration`） |
| 日志、脚本、`pip freeze` | `/data0/tongbaoshun/benchmarks/_logs/e00_robocasa/` |
| 下载缓存 | `/data0/tongbaoshun/benchmarks/robocasa/_downloads/`：`objaverse.zip` 2.1 GB、`textures.zip` 0.5 GB、`generative_textures.zip` 1.2 GB（都已解压，可删） |

```bash
source /home/tongbaoshun/miniconda3/etc/profile.d/conda.sh
conda activate dynisafety-robocasa
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2
```

`MUJOCO_EGL_DEVICE_ID` 是 EGL 设备序号，不是 CUDA 序号；本机 EGL 0–2 对应 CUDA 0–2（H1 的 `egl_devices` 探针）。见“已知坑”第 5 条。

## 版本

| 项 | 值 |
|---|---|
| Python | 3.11.18 |
| robosuite | 1.5.2，git `5ce6643`（2026-07-11，上游 master，安装脚本的固定值） |
| robocasa | 1.0.1，git **`921c9a5`**（2026-08-07，安装脚本的固定值；用户定沿用它，H2 prompt 已改，见“已知坑”第 1 条） |
| mujoco | 3.3.1 |
| numpy | 2.2.5 |
| numba / scipy | 0.61.2 / 1.15.3 |
| torch | 2.7.1+cu126（**不支持 sm_120**，见“已知坑”第 3 条） |
| lerobot / tianshou / gymnasium | 0.3.3 / 0.4.10 / 0.29.1（robocasa 的 `install_requires` 带进来） |
| opencv-python / h5py | 5.0.0.93 / 3.16.0 |
| dynisafety | 0.1.0（`--no-deps`） |

完整列表：`/data0/tongbaoshun/benchmarks/_logs/e00_robocasa/pip_freeze.txt`（142 行），`conda_list.txt` 同目录。`pip check` 只报一条：`dynisafety` 声明 `numpy<2.0`，这是配方有意绕开的（`--no-deps`）。

## 安装命令（实际执行的）

```bash
# 克隆到仓库外，再软链进主 worktree 的 third_party/（安装脚本只认 ${ROOT}/third_party）
cd /data0/tongbaoshun/benchmarks/robocasa
git clone https://github.com/ARISE-Initiative/robosuite.git robosuite
git clone https://github.com/robocasa/robocasa.git robocasa
cd /data0/tongbaoshun/workspace/RoboSafeBench
ln -s /data0/tongbaoshun/benchmarks/robocasa/robosuite third_party/robosuite
ln -s /data0/tongbaoshun/benchmarks/robocasa/robocasa  third_party/robocasa

# 仓库配方（docs/user/install.md）
conda env create -n dynisafety-robocasa -f environment/dynisafety-robocasa.yml
conda activate dynisafety-robocasa
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2
bash scripts/setup/install_backend.sh robocasa       # 7 m 35 s，exit 0；轻量资产

# 原子任务需要 objaverse 物体（见“资产”）：从官方 HF 仓库取，按 LFS sha256 校验
cd /data0/tongbaoshun/benchmarks/robocasa/_downloads
wget -c https://hf-mirror.com/datasets/robocasa/robocasa-assets/resolve/1b92c3d02ca4354984fec961357db0bff7b32166/objaverse.zip
echo "66a7eebef3bac855301964d35d905fc2cb44d691f007f97af08d96e83db5cc08  objaverse.zip" | sha256sum -c -
python -c "from zipfile import ZipFile; ZipFile('objaverse.zip').extractall('<robocasa>/robocasa/models/assets/objects')"

# 纹理与生成纹理换成官方版（用户定，替换安装脚本从第三方镜像取的那份；旧目录先移出再解压）
for f in textures generative_textures; do
  wget -c https://hf-mirror.com/datasets/robocasa/robocasa-assets/resolve/1b92c3d02ca4354984fec961357db0bff7b32166/$f.zip
done
printf '%s  %s\n' ba3a4dcef96e199cbd601c7cdb7eb46077fb6672611b7ac74c25a191bf0be392 textures.zip \
  bc05095af306222d9a799a9b0b5b16662aa546998622ee8fd3072566ea4c31ec generative_textures.zip | sha256sum -c -
cd <robocasa>/robocasa/models/assets && unzip -q <downloads>/textures.zip && unzip -q <downloads>/generative_textures.zip
```

安装脚本做的事：把 robosuite 检出到固定值（已是）、robocasa 从 `3443cdf` 切到 `921c9a5`，`pip install -e` 两个克隆，跑 `robocasa.scripts.setup_macros`（在克隆里写了被上游 gitignore 的 `macros_private.py`），下载轻量资产，`pip install -e . --no-deps` 与 `requirements/robocasa.txt`。

## 资产

| 包 | 用途 | 来源 | 大小（解压后） |
|---|---|---|---|
| `tex` | 环境纹理（225 个文件） | **官方** `robocasa/robocasa-assets@1b92c3d`，sha256 `ba3a4dce…`（与 LFS oid 一致）。安装脚本原先装的是第三方镜像 `Livfour/robocasa-assets` 的版本，已替换 | 521 MB |
| `tex_generative` | 生成纹理（900 个文件，含 450 个 `tex*.png`） | **官方**，同上，sha256 `bc05095a…`。替换前同样来自第三方镜像，而且被导航后端建了 400 个错误的软链，见“已知坑”第 10 条 | 1.2 GB |
| `fixtures_lw` | lightwheel 固定装置 | 安装脚本：第三方镜像 `IIFAN/robocasa365-lightwheel-fixtures` | 1.4 GB |
| `objs_lw` | lightwheel 物体（658 个实例，58 个类别） | 安装脚本：第三方镜像 `IIFAN/robocasa365-lightwheel-objects` | 1.5 GB |
| `objs_objaverse` | objaverse 物体（81 个类别目录） | **官方** `robocasa/robocasa-assets@1b92c3d`，sha256 `66a7eebe…`（与 LFS oid 一致） | 6.2 GB |
| `objs_aigen` | AI 生成物体 | **未下载**：官方仓库里单个 zip 5.77 GB，超过“单次 > 5 GB 先问”；`Kitchen` 默认物体库不含它 | — |

- `Kitchen` 的默认 `obj_registries = ("objaverse", "lightwheel")`（`kitchen.py:397`）。只有轻量资产时 objaverse 库为空，凡是抽到 lightwheel 里没有的类别就报 `ValueError: Probabilities contain NaN`（见“已知坑”第 4 条）。
- 第三方镜像的 zip 与官方的哈希不同。逐文件比对后：镜像纹理是官方的子集，同名文件逐字节相同；镜像缺 `textures/window_bkgs/`（5 张，代码未引用）和生成纹理里全部 450 个 `tex*.png`（官方的 `tex*.png` 是同目录 UUID 文件的副本）。lightwheel 固定装置与物体官方仓库没有，仍用镜像。见“已知坑”第 2 条。
- 资产合计约 11 GB（`robocasa/models/assets/`，`du -sh`）；下载缓存另占 3.8 GB。

## 磁盘占用

| 项 | 大小 |
|---|---|
| conda 环境 | 7.2 GB |
| robosuite 克隆 | 1.3 GB |
| robocasa 克隆（含资产约 11 GB） | 11 GB |
| 下载缓存（3 个 zip，可删） | 3.8 GB |
| 日志与评测输出 | < 10 MB |

## 场景与接口事实（给 H2 用）

以下来自 `PickPlaceCounterToCabinet`、种子 0 的烟雾测试（`smoke_atomic.log`）与导航评测输出：

| 项 | 值 |
|---|---|
| 机器人 | `PandaOmron`（`WheeledRobot`） |
| `control_freq` / `model.opt.timestep` | 20 Hz / 0.002 s，即每个控制步 25 个物理子步 |
| 动作 | 12 维；切分（导航后端记录）：臂 `[0,6)`、夹爪 `[6,7)`、底盘 `[7,10)`、躯干 `[10,11)`、底盘模式 `[11,12)` |
| 布局 | 种子决定布局：默认物体库下 layout 12 / style 32，只用 lightwheel 时 layout 28 / style 18 |
| `env.fixtures` | 114 个键（只用 lightwheel 时 93 个），名字形如 `counter_1_main_group`、`cab_1_main_group`、`dishwasher_1_main_group` |
| `env.objects` | `obj`、`distr_counter`、`distr_cab` |
| 观测 | 3 个相机 128×128×3 uint8（`robot0_agentview_left/right_image`、`robot0_eye_in_hand_image`），`object-state` 42 维，`robot0_proprio-state` 68 维，另有逐物体的位置、四元数与相对 EEF 量 |
| 模型规模 | 384 个刚体、2027 个几何体 |
| 机器人侧刚体名前缀 | `robot0_` 10、`gripper0_` 6、`mobilebase0_` 4；**`panda0_` 为 0**（H2 prompt 第 5 条列了它） |
| 零动作 20 步（1 s） | EEF 漂移 4–5 µm，底盘 `mobilebase0_base` 漂移 0 |
| 耗时 | `reset` 10.8–12.3 s（含程序化建场景与编译模型）；20 步 0.9 s；导航评测 450 步 28.3 s |
| H2 引用的 `kitchen.py` 行号 | `edit_model_xml` 1216、`_post_action` 1476、`_check_success` 1542、`register_fixture_ref` 1747、`self.fixtures` 632、`self.objects` 853：在 `921c9a5` 与 `3443cdf` 上都对得上 |

演示数据（H2 第 9 条，只调研、未下载）：

- 官方下载脚本 `robocasa/scripts/download_datasets.py` 按 `box_links_ds.json`（410 个 tar）取数据，`PickPlaceCounterToCabinet`、`OpenDrawer`、`OpenCabinet` 各有 pretrain 人类、target 人类、pretrain MimicGen 三份，`CloseDrawer` 没有 target。抽查一条 Box 链接返回 404。
- HF 上有第三方转存（如 `ember-lab-berkeley/robocasa365-pretrain-atomic`、`…-pretrain-mg`、按任务拆分的 `jellyho/robocasa365-<Task>`），大小与是否与官方一致 **[未核实]**。下载前要先问用户。

## 烟雾测试

1. `dynisafety doctor --backend robocasa`（`doctor.log`，exit 1）：14 项 OK（torch/CUDA、两个源码树及其固定值、robocasa / robosuite / mujoco 版本、numpy 互斥、robocasa 包与 4 类轻量资产、conda 环境、GPU）；3 项 WARN 是 MANO 资产与两个模型权重，与本任务无关。2 个 FAIL 也与 RoboCasa 无关（“已知坑”第 6、7 条）：

   ```text
   FAIL    FAIL  patches                   missing .../scripts/apply_patches.sh        git checkout HEAD -- scripts/setup/apply_patches.sh
   FAIL    FAIL  conda_yaml:mibot          configs/models references env 'mibot' (missing); ...
   summary: 19 checks, 2 fail, 3 warn
   ```

2. `dynisafety eval --backend robocasa --model random_mobile --suite robocasa_nav --level 0 --task 0 --episodes 1 --output-root <logs>/eval_out`（`eval_nav.log`，exit 0，54.6 s）：

   ```text
   eval done task_id=0 episodes=1 successes=0 dir=.../robocasa-random_mobile-robocasa_nav-t0-l0-s7-20261107T083727
   ```

   `episodes.jsonl`：`nav_sink`，450 步，`termination: max_steps`，`success: false`，0 个接触事件。装 objaverse 之后重跑一次（`eval_nav_after_objaverse.log`），终点 `base_xy_end` 逐位相同。

3. 原子任务（`scripts/smoke_atomic.py`，用 `robocasa.utils.env_utils.create_env`，`smoke_atomic.log`，exit 0）：

   ```text
   create_env('PickPlaceCounterToCabinet', seed=0, {})
   create 0.0s reset 10.8s
   class PickPlaceCounterToCabinet robots ['WheeledRobot:PandaOmron']
   control_freq 20 model_timestep 0.002 substeps/control 25
   layout_id 12 style_id 32
   action_dim 12
   lang Pick the hot dog from the counter and place it in the cabinet.
   fixtures (114): ['bottom_right_group_1_1', 'bottom_right_group_1_base', 'box_1_island_group', ...]
   objects (3): ['distr_cab', 'distr_counter', 'obj']
   20 zero-action steps in 0.9s; obs finite: True; reward 0.0; success False
   eef drift over 20 steps: 0.000004 m
   base body mobilebase0_base drift: 0.000000 m
   |    2   N/A  N/A          218474      G   python                                  865MiB |
   ```

   最后一行是脚本在环境存活时抓的 `nvidia-smi`：本进程只出现在 GPU 2，类型 G（EGL 图形上下文）。完整的键表与观测形状见日志。

4. `pytest tests/test_backends_robocasa.py`（主 worktree，本环境，`pytest_backends_robocasa.log`）：`40 passed, 1 skipped`（跳过原因：`SMPL-family model.npz is not installed under external/assets`）。该文件没有 `sim` 标记的测试。H2 的“前后一致”以此为基线。
5. **换成官方纹理后重跑**（`*_official_tex.log`、`smoke_atomic_gen100p.log`，GPU 2）：
   - `doctor`：与上面相同，14 OK / 3 WARN / 2 FAIL，两个 FAIL 仍是“已知坑”第 6、7 条；
   - 导航评测：exit 0，`base_xy_end`、底盘关节终值与换之前逐位相同；
   - 原子任务：layout 12 / style 32、物体、漂移都与之前相同。`robot0_agentview_left_image` 的均值 / 标准差从 89.65 / 65.55 变成 97.43 / 70.05，原因是之前的生成纹理软链指错了文件（“已知坑”第 10 条）。对照实验：两个纹理目录都换回镜像，复现出 89.65 / 65.55；只把 `textures` 换回镜像，仍是 97.43 / 70.05；官方版连跑两次都是 97.43 / 70.05；
   - `--generative-textures 100p`（脚本新加的参数）：exit 0，`reset` 正常，均值 / 标准差 116.80 / 49.44。

## 已知坑

1. **robocasa 固定值两处不一致**：安装脚本与 `docs/user/install.md` 固定 `921c9a5`，H2 prompt 原来写 `3443cdf`（上游当前 main）。`921c9a5` 是 `3443cdf` 的祖先，差 14 个提交，`setup.py` 都是 1.0.1；运行时代码只有 `models/fixtures/sink.py` 删掉一个重复定义的函数，其余是子任务标注流水线、脚本、测试与 README。安装脚本会强制切回 `921c9a5`，本环境就是 `921c9a5`。用户定沿用 `921c9a5`，H2 prompt 已改。
2. **资产来源**：官方 Box 链接已失效（`shared/static/*.zip` 404，安装时 4 个包都报 SSL EOF），安装脚本回退到第三方 HF 镜像，镜像文件与官方 `robocasa/robocasa-assets` 的哈希都不同。官方 HF 仓库（CC BY 4.0，2026-02-13 更新）有 `textures`、`generative_textures`、`fixtures`、`objaverse`、`aigen_objs` 五个包，合计 9.89 GB，但**没有** lightwheel 物体与 lightwheel 固定装置，这两类目前只能从第三方镜像取。本环境的纹理、生成纹理、objaverse 已换成官方版。
3. **torch 2.7.1+cu126 不支持 sm_120**：import 时警告 “sm_120 is not compatible with the current PyTorch installation”。仿真与 MuJoCo 渲染不用 torch，不受影响；但本环境里的 lerobot / 策略不能用 GPU。要用得换 cu128 的 torch 轮子，本任务没有动。
4. **只有轻量资产时原子任务 `reset` 失败**：`kitchen_object_utils.py:463` 的 `rng.choice(..., p=...)` 报 `Probabilities contain NaN`（之前有 `RuntimeWarning: invalid value encountered in divide`）。要么装 objaverse（本环境已装），要么建环境时传 `obj_registries=("lightwheel",)`（`smoke_atomic_lw.log` 验证可用，但只覆盖 58 个物体类别）。导航任务不受影响。
5. **EGL 设备**：robosuite 的 `egl_context.py:36-65` 在 `MUJOCO_EGL_DEVICE_ID` 是纯数字时直接用它作 EGL 设备序号，忽略构造参数 `render_gpu_device_id`；`binding_utils.py:29-35` 还断言它出现在 `CUDA_VISIBLE_DEVICES` 字符串里。所以两个变量都设成 2。进程会打开 `/dev/nvidia0`–`2`（EGL 枚举设备），但 `nvidia-smi` 里图形上下文只在 GPU 2。
6. **`doctor` 的 `patches` 项是 `doctor` 自己的路径错误**：`src/dynisafety/cli/doctor.py:372` 找 `scripts/apply_patches.sh`，脚本实际在 `scripts/setup/apply_patches.sh`，按 FIX 列执行也修不好。RoboCasa 本来就没有补丁。按约束没改 `src/dynisafety/`。
7. **`doctor` 的 `conda_yaml:mibot`**：`configs/models/xiaomi.yaml` 的服务环境 `mibot` 本机没有，与 RoboCasa 无关。
8. robosuite 启动时的警告（没有私有宏文件、`robosuite_models` 与 `mink` 没装）对 `PandaOmron` 无影响；`mimicgen` 没装的警告同样无影响。
9. 评测 Manifest 的 `git_dirty: true` 来自主 worktree 里未跟踪的 `docs/researchProgress.md`，与本环境无关。
10. **导航后端会给生成纹理建错误的软链**：`src/dynisafety/backends/robocasa/env.py:1187-1209` 的 `ensure_generative_texture_tex_aliases` 在每次 `make_robocasa_env` 时运行。如果 `generative_textures/<类别>/` 下没有 `tex*.png`（第三方镜像就没有），它就把排序后的 UUID 文件依次软链成 `tex001.png`、`tex002.png`……。fixture 注册表（`fixtures/fixture_registry/{floor,wall,cabinet,counter}.yaml`）在默认模式下也引用这些名字。和官方文件对比，400 个软链里有 399 个指错，所以替换前，默认风格下的地板、墙、柜体、台面纹理都不对，原子任务的图像也因此不同。本环境第一次导航评测时建了这 400 个软链（`eval_out` 的 `gentex_tex_aliases` 每类 100），现在随旧目录一起移走了；有了官方的 `tex*.png`，这个函数遇到已存在的文件会跳过，记 0。用镜像资产的其他机器有同样的问题。这段代码属于 `src/dynisafety`，按约束没有改。
11. robocasa 克隆里唯一的未跟踪文件是 `robocasa/models/assets/README.md`，是安装脚本下载的 lightwheel 物体包带来的（指向 nvidia 的 HF 数据集）。
