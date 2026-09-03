# 安装与 conda

仿真器安装、conda 名字、短配方、可选一键建 env。权重下载见 [assets.md](assets.md)；跑评测见 [eval.md](eval.md)。Pin / 许可证细节：[third_party.md](install.md)、[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。

**硬规则：** 一个 Python/conda 只装一个 backend。SimplerEnv = SAPIEN **2.2.2**；RoboTwin = SAPIEN **3.0.0b1**；LIBERO = MuJoCo + NumPy 1.x；RoboCasa = MuJoCo **3.3.1** + NumPy **2.2.5**。不要去掉 `pyproject.toml` 的 `numpy>=1.26,<2.0`。

调度认的 conda **名字**来自 `configs/models/*.yaml`（`extra.eval_env` / `server.env`）。**不要重命名**已经能跑的 env。`environment/dynisafety-*.yml` 只是新机器配方，`name:` 不是 `paper_main` 查找名。

---

## 1. 克隆

```bash
git clone <THIS_REPO_URL> dynisafety
cd dynisafety
# 子模块也可由 install 脚本再拉
git submodule update --init --recursive
```

Linux x86_64 + Conda/Mamba；yml 目标 Python 3.10，RoboCasa 是 **3.11**。权重、视频、MANO、API key 不要进 git。

---

## 2. 三种 conda 角色

| 角色 | 谁在里面跑 | 必须有 |
|------|------------|--------|
| **Eval / 仿真** | `dynisafety eval` | 该 backend 的仿真器 |
| **Serve** | `dynisafety serve` | 上游 VLA 栈，通常不要仿真器 |
| **合体** | 进程内加载权重 | 仿真器 **和** 模型（无 `serve_env`） |

合体（无 server）：`openvla_oft`、`openvla_oft_rlinf`、`internvla_a1`、`fastwam`。

---

## 3. `paper_main` 名字（调度真相）

权威矩阵：`configs/suites/paper_main.yaml`。同一模型换 backend 可能换 `eval_env`（`per_backend`）。

### LIBERO — eval `openvla-oft`

| 模型 | Eval | Serve |
|------|------|-------|
| `openvla_oft` / `openvla_oft_rlinf` | `openvla-oft` | — |
| `xvla` | `openvla-oft` | `XVLA` |
| `starvla` | `openvla-oft` | `starVLA` |
| `pi0` / `pi05` | `openvla-oft` | `openpi` |
| `univla` | `openvla-oft` | `univla` |

### RoboTwin — eval `internvla_a1`

不是遗留的 `RoboTwin`，也不是 `dynisafety-robotwin`。

| 模型 | Eval | Serve |
|------|------|-------|
| `internvla_a1` / `fastwam` | `internvla_a1` | — |
| `lingbot_va` | `internvla_a1` | `lingbot_va` |
| `starvla` | `internvla_a1` | `starVLA` |
| `xvla_official` | `internvla_a1` | `XVLA` |

### SimplerEnv — eval `simpler_env`

| 模型 | Eval | Serve |
|------|------|-------|
| `starvla` | `simpler_env` | `starVLA` |
| `internvla_m1` | `simpler_env` | `internvla-m1` |
| `xvla_official` | `simpler_env` | `XVLA` |

最少九个：`openvla-oft` `simpler_env` `internvla_a1` `XVLA` `starVLA` `openpi` `univla` `internvla-m1` `lingbot_va`。

RoboCasa **不在** `paper_main`，eval 名 `dynisafety-robocasa`。

Extra（不在 `paper_main`）：`turbovla` eval=`openvla-oft`，serve=`turbovla`。

磁盘上可能还有 `RoboTwin`、`fastwam`、`openvla` 等，不是主表入口，不要按 `conda env list` 全装。

---

## 4. 可选一键（只建缺失的 eval env）

**不会**改已经存在的 conda。Serve 型 VLA 不装（跟上游）。

```bash
bash scripts/setup/install_conda_env.sh --list
bash scripts/setup/install_conda_env.sh                          # 交互选 backend + 模型
bash scripts/setup/install_conda_env.sh --backend libero --model openvla_oft --yes
bash scripts/setup/install_conda_env.sh --backend robotwin --model starvla --dry-run
```

| Flag | 作用 |
|------|------|
| `--yes` | 跳过确认 |
| `--dry-run` | 只打印计划 |
| `--skip-torch` | 不装参考 torch |
| `--install-backend` | 即使 eval env 已存在也跑 `install_backend.sh`（默认不跑） |
| `--paper` | 交互列表只显示 `paper_main` 模型 |

已有 `openvla-oft` / `simpler_env` / `internvla_a1` 时会打印 `EXISTS — skip create`。`internvla_a1` / `fastwam` 新建时只有 sapien 3 仿真层，权重仍走 `extra.repo_root`。

---

## 5. 手装一个 backend

`make install-*` / `install_backend.sh` **不创建** conda。先 activate YAML eval 名：

```bash
conda activate openvla-oft && make install-libero
# conda activate simpler_env && make install-simplerenv
# conda activate internvla_a1 && make install-robotwin
# conda activate dynisafety-robocasa && export MUJOCO_GL=egl && make install-robocasa
```

等价：`scripts/setup/install_backend.sh libero|simplerenv|robotwin|robocasa`（`--skip-pip` / `--skip-patch`）。

步骤：同步树 → 打补丁 → `pip install -e ".[<backend>]"`。RoboCasa：`pip install -e . --no-deps` 再 `pip install -r requirements/robocasa.txt`（禁止普通 `pip install -e .`，会拉 `numpy<2`）。

新机器也可用配方，**`-n` 必须是 YAML 名**：

```bash
conda env create -n openvla-oft -f environment/dynisafety-libero.yml
conda env create -n simpler_env -f environment/dynisafety-simplerenv.yml
conda env create -n dynisafety-robocasa -f environment/dynisafety-robocasa.yml
# robotwin 论文 eval 是 internvla_a1（上游合体栈），不要建成 dynisafety-robotwin / RoboTwin
```

| Backend | 配方 `name:` | 建成（YAML） | pip extra |
|---------|--------------|--------------|-----------|
| LIBERO | `dynisafety-libero` | `openvla-oft` | `.[libero]` |
| SimplerEnv | `dynisafety-simplerenv` | `simpler_env` | `.[simplerenv]` |
| RoboTwin | `dynisafety-robotwin` | `internvla_a1` | `.[robotwin]` |
| RoboCasa | `dynisafety-robocasa` | `dynisafety-robocasa` | `--no-deps` + `robocasa.txt` |

禁止：`pip install -e ".[libero,simplerenv,robotwin]"`。

然后按 [../../requirements/README.md](../../requirements/README.md) **先装 torch**，再 `pip install -r requirements/<backend>.txt`。参考：libero cu124 / 2.4.x；simplerenv cu118 / 2.3.x；`internvla_a1` cu128 / 2.7.x。Serve env 各自保留上游 torch，不要统一。

---

## 6. 依赖怎么给（不要 freeze）

仓库**不提供** `conda env export` / `pip freeze`。现场 env 常有“元数据冲突但能跑”；重放 dump 会求解失败或被 pip“修好”装崩。

| 层 | 文件 | 管什么 |
|----|------|--------|
| Python | `environment/dynisafety-*.yml` | 3.10/3.11 + `-r requirements/*.txt` |
| 仿真短 pin | `requirements/*.txt` + `pyproject.toml` extras | sapien / numpy / mujoco |
| Torch | **没有文件** | 按上一节手装 |
| 仿真树 | `scripts/setup/install_*.sh` | editable `third_party/` |
| VLA / serve | 上游 `server.root` / `extra.repo_root` | 本仓库不写 |

`conda list` 只当私人日志，不当安装器。

---

## 7. 各 backend 要点

**LIBERO：** `scripts/setup/install_libero.sh` 装到 `third_party/dyniSafety_libero`（不是 submodule）。废弃路径 `simulator_env/safelibero/`。可选 Docker：根 `Dockerfile`，说明 [../../docker/README.md](../../docker/README.md)。

**SimplerEnv：** submodule `third_party/SimplerEnv-OpenVLA`，SAPIEN 2.2.2。Bridge 全模型 `runtime.max_steps: 120`，不要和文献 60 步比。

**RoboTwin：** submodule pin 见 [third_party.md](install.md)，SAPIEN 3.0.0b1。论文评测 env 是 `internvla_a1`。

**RoboCasa：** 独立 NumPy 2 env，Python 3.11，`export MUJOCO_GL=egl`。厨房地面轮式导航 `robocasa_nav`，**不在** `paper_main`。轻量资产缺了 `doctor` 会 FAIL → `install_robocasa.sh`。

```bash
conda activate dynisafety-robocasa
export MUJOCO_GL=egl
make install-robocasa
dynisafety doctor --backend robocasa
dynisafety sweep --suite smoke_robocasa --dry-run
```

---

## 8. Docker（仅 LIBERO）

```bash
docker build -t dynisafety-libero:latest -f Dockerfile .
docker run --rm --gpus all -it dynisafety-libero:latest
```

宿主机论文 eval 名是 `openvla-oft`，不是镜像里的 `dynisafety-libero`。

---

## 9. 验收

```bash
bash scripts/setup/audit_envs.sh
python tools/model_env.py --audit
python tools/model_env.py starvla eval --backend robotwin   # internvla_a1
dynisafety doctor --backend <backend>
dynisafety sweep --suite smoke --dry-run
make test    # pytest -m "not sim"
```

`doctor` 的 `conda_backend:dynisafety-*` 在论文机（只有 YAML 名）上 FAIL 是预期，以 `audit_envs.sh` 为准。

现场参考版本（不是 pin）：`openvla-oft` 2.4.1+cu124；`simpler_env` 2.3.1+cu118 sapien 2.2.2；`internvla_a1` 2.7.1+cu128 sapien 3；serve 的 torch 各不相同。

---

## 10. 排障

| 现象 | 处理 |
|------|------|
| freeze/export 重装失败 | 用短配方，不要 dump |
| extras 互殴 | 一个 env 只装一个 extra |
| RoboCasa numpy/mujoco | `--no-deps` + `numpy==2.2.5` |
| sapien 2/3 | 分 env：`simpler_env` vs `internvla_a1` |
| 补丁 drift | `scripts/setup/apply_patches.sh <backend>` |
| 子模块空 | `install_<backend>.sh` |
| serve env 缺 | 按 YAML 精确名字从上游建 |
| 旧 bash 入口 | [migration.md](repo.md) |


---

# 第三方 pin / 补丁 / 许可证

（原 third_party.md）

# Third-party components (DyniSafety)

This document lists simulators and restricted assets that DyniSafety depends on:
upstream location, pinned commit, license, our patches, install entrypoints, and
intended runtime path keys.

Path placeholders follow config resolution: `${workspace}`, `${third_party}`
(`<workspace>/third_party`). Do not hard-code machine-absolute paths in YAML.

**Install (user guide):** **[install.md](install.md)**（conda / 一键建 env / Docker）。
权重：[assets.md](assets.md)。总目录：[README.md](README.md)。

Short pin-check reminder (after submodule init):

```bash
scripts/setup/apply_patches.sh --check all   # exit 2 => MISSING or DRIFTED
scripts/setup/install_backend.sh <backend> # or: make install-libero|simplerenv|robotwin|robocasa
dynisafety doctor --backend <backend>
```

Conda recipes: `environment/dynisafety-{libero,simplerenv,robotwin,robocasa}.yml`
(override `-n` to the YAML name; [install.md](install.md)).
Docker (libero only): root `Dockerfile` + `docker/README.md`.
This document remains the **pin / license / patch** authority; it is not a
second install manual.

License texts / notices: see [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)
(coverage = **core simulators + MANO + RoboCasa/robosuite**).

---

## Summary

| Component | Layout | Pin | License | Install |
|-----------|--------|-----|---------|---------|
| SimplerEnv-OpenVLA | submodule `third_party/SimplerEnv-OpenVLA` | `ccfe3809766839a2fcfb7a3d3c9abff585189188` | MIT | `scripts/setup/install_simplerenv.sh` |
| ManiSkill2_real2sim | nested under SimplerEnv | `91d154bfd864577f8d2e80f3fc2f8b4d9df9ae5c` | Apache-2.0 | via SimplerEnv + `apply_patches.sh` |
| RoboTwin | submodule `third_party/RoboTwin` | `afd5f1bbc5759ac1512dc42b3cdaf8a218fb7c00` (`dynisafety-pin-20260806`) | MIT | `scripts/setup/install_robotwin.sh` |
| dyniSafety_libero (SafeLIBERO) | **not** a submodule; created by install | install tree; upstream ref `8f1084e3132a39270c3a13ebe37270a43ece2a01` | MIT | `scripts/setup/install_libero.sh` |
| robosuite | **not** a submodule; `third_party/robosuite` | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` (git **master**, ≥1.5; not PyPI 1.4) | MIT | `scripts/setup/install_robocasa.sh` |
| RoboCasa | **not** a submodule; `third_party/robocasa` | `921c9a5736a8d0ea5589657898aadcfa55a6a195` | MIT (code); CC BY 4.0 (assets) | `scripts/setup/install_robocasa.sh` |
| MANO / SMPL+H | **not distributed** | n/a | MPI/MPG non-commercial | `scripts/setup/download_assets.py --mano-notice` |

Deprecated: `simulator_env/safelibero/`. Use `dyniSafety_libero` only.

---

## 1. SimplerEnv-OpenVLA

| Field | Value |
|-------|-------|
| Upstream | `https://github.com/DelinQu/SimplerEnv-OpenVLA.git` (fork of simpler-env / OpenVLA SimplerEnv) |
| Path | `${third_party}/SimplerEnv-OpenVLA` |
| Pin | `ccfe3809766839a2fcfb7a3d3c9abff585189188` (`gr00t support action queue`) |
| License | MIT — `third_party/SimplerEnv-OpenVLA/LICENSE` |
| Runtime key | intended YAML: `extra.simplerenv.sim_root: ${third_party}/SimplerEnv-OpenVLA` (wiring in a later batch) |
| Install | `scripts/setup/install_simplerenv.sh` then `scripts/setup/apply_patches.sh simplerenv` |

### Our changes (`patches/simplerenv/`)

| Patch | Why |
|-------|-----|
| `requirements_json_numpy.patch` | HTTP VLA clients need `json_numpy` |
| `maniskill2_evaluator_optional_visualize.patch` | Guard missing `visualize_epoch` on some policies |
| `openpi_websocket_host_and_obs_keys.patch` | Connect to `127.0.0.1`; LiberoInputs observation key names |
| `maniskill2_submodule_sha.patch` | **Not** a unified diff — pins nested ManiSkill2 via `git checkout` |

Not patched: local `main_inference.py` / `scripts/run_*.sh` edits; untracked
`simpler_env/policies/{dreamvla,…}` (policy logic lives under
`src/dynisafety/policies/adapters/`).

---

## 2. ManiSkill2_real2sim

| Field | Value |
|-------|-------|
| Upstream | `https://github.com/allenzren/ManiSkill2_real2sim.git` (as recorded in SimplerEnv `.gitmodules`) |
| Path | `${third_party}/SimplerEnv-OpenVLA/ManiSkill2_real2sim` |
| Pin | `91d154bfd864577f8d2e80f3fc2f8b4d9df9ae5c` (upstream SimplerEnv@`ccfe380` records `cd45dd2`) |
| License | Apache-2.0 — `…/ManiSkill2_real2sim/LICENSE` |
| Apply | `scripts/setup/apply_patches.sh simplerenv` (checkout helper) |

---

## 3. RoboTwin

| Field | Value |
|-------|-------|
| Upstream | [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) |
| Path | `${third_party}/RoboTwin` |
| Pin | `afd5f1bbc5759ac1512dc42b3cdaf8a218fb7c00` / tag `dynisafety-pin-20260806` |
| License | MIT — `third_party/RoboTwin/LICENSE` |
| Runtime key | intended YAML: `extra.robotwin.sim_root: ${third_party}/RoboTwin` |
| Install | `scripts/setup/install_robotwin.sh` then `scripts/setup/apply_patches.sh robotwin` |
| Assets | Large `assets/` (~16GB) are **not** in the submodule; obtain separately (do not commit) |

### Our changes (`patches/robotwin/`)

| Patch | Why |
|-------|-----|
| `dynisafety_hooks.patch` | Minimal `Base_Task` hooks (`enable`/`disable`/`before_step`/`after_step`/`update_info`/close); business logic in `dynisafety.backends.robotwin.hooks` |

Per-task `get_dynisafety_*` helpers are applied at runtime, not in the patch.

Pin tree excludes `envs/dynisafety/` (logic lives in `src/dynisafety/backends/robotwin/`).

---

## 4. LIBERO / dyniSafety_libero (SafeLIBERO)

| Field | Value |
|-------|-------|
| Upstream | `https://github.com/Lifelong-Robot-Learning/LIBERO.git` |
| Path | `${third_party}/dyniSafety_libero` (created by install; **not** listed in `.gitmodules`) |
| Pin / source | Set `DYNISAFETY_LIBERO_SRC` to reuse a local tree; otherwise clone upstream @ `8f1084e3132a39270c3a13ebe37270a43ece2a01` (may lack DyniSafety BDDL/assets) |
| License | MIT — upstream / fork `LICENSE` |
| Runtime key | intended YAML: `extra.libero.sim_root: ${third_party}/dyniSafety_libero` |
| Install | `scripts/setup/install_libero.sh` |
| Patches | none (`patches/libero/` empty by design) |

`simulator_env/safelibero/` is **deprecated**.

---

## 5. robosuite (RoboCasa backend)

| Field | Value |
|-------|-------|
| Upstream | `https://github.com/ARISE-Initiative/robosuite.git` |
| Path | `${third_party}/robosuite` (**not** a git submodule) |
| Pin | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` (git **master**; package version 1.5.2) |
| License | MIT — `third_party/robosuite/LICENSE` |
| Install | `scripts/setup/install_robocasa.sh` then `pip install -e` this tree |
| Forbidden | PyPI `robosuite==1.4.*` — RoboCasa asserts robosuite ≥ 1.5.2 |

Clone may use `https://ghfast.top/https://github.com/ARISE-Initiative/robosuite.git` when GitHub TLS fails. Override with `DYNISAFETY_ROBOSUITE_URL`.

---

## 6. RoboCasa

| Field | Value |
|-------|-------|
| Upstream | `https://github.com/robocasa/robocasa.git` |
| Path | `${third_party}/robocasa` (**not** a git submodule) |
| Pin | `921c9a5736a8d0ea5589657898aadcfa55a6a195` (package version 1.0.1) |
| License (code) | MIT — `third_party/robocasa/LICENSE` |
| License (assets / datasets) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Runtime key | YAML: `extra.robocasa.sim_root: ${third_party}/robocasa` |
| Install | `scripts/setup/install_robocasa.sh` |
| Conda env | `dynisafety-robocasa` (Python 3.11 only) |
| Official pins | `numpy==2.2.5`, `mujoco==3.3.1` (`import robocasa` asserts both) |
| NumPy mutex | Do **not** install into the LIBERO / SimplerEnv / RoboTwin eval envs (`openvla-oft`, `simpler_env`, `internvla_a1`). Do **not** remove `numpy>=1.26,<2.0` from `pyproject.toml`. This env uses `pip install -e . --no-deps` then `requirements/robocasa.txt`. |
| Assets (default) | lightweight: `tex`, `tex_generative`, `fixtures_lw`, `objs_lw`. `dynisafety doctor --backend robocasa` **FAIL**s if they are missing (FIX: `install_robocasa.sh`). |
| Assets (full ~10GB) | only if `DYNISAFETY_ROBOCASA_FULL_ASSETS=1` |
| Asset host | Official UTexas Box `shared/static/*.zip` links in `box_links_assets.json` currently **404**. `install_robocasa.sh` falls back to Hugging Face mirrors (`Livfour/robocasa-assets`, `IIFAN/robocasa365-lightwheel-*`). If `huggingface.co` times out, it retries `https://hf-mirror.com`. |
| Headless | `export MUJOCO_GL=egl` |
| Patches | none (`patches/robocasa/` not used) |
| Eval | kitchen **wheeled** navigation (`NavigateKitchen`, suite `robocasa_nav`) + floor disturbances L0–L3. Not countertop PickPlace, not bipedal. **Not** in `paper_main`. Camera `robot0_agentview_center` looks at the stove — spawn is `body z = floor`. |

Clone may use `https://ghfast.top/https://github.com/robocasa/robocasa.git`. Override with `DYNISAFETY_ROBOCASA_URL`.

---

## 7. MANO / SMPL+H (not redistributed)

| Field | Value |
|-------|-------|
| Owner | Max Planck Institute for Intelligent Systems / Max-Planck-Gesellschaft |
| License | Proprietary non-commercial research terms — see `THIRD_PARTY_NOTICES.md` |
| Distribution | **Must not** ship in this repository or releases |
| Obtain | Accept official MANO terms yourself; place under `<workspace>/assets/mano_v1_2/` (gitignored) |
| Helper | `python scripts/setup/download_assets.py --mano-notice` |

Note: RoboTwin L3 default hand mesh in DyniSafety is the small `hand_pusher` asset
(see backend YAML / `hand_objects`), not MANO. RoboCasa L3 default pedestrian
visuals are original in-tree OBJ parts under
`src/dynisafety/backends/robocasa/assets/pedestrian/`, not SMPL.

---

## 8. Checkpoints & other downloads

Use `scripts/setup/download_assets.py` (canonical). Root `download_hf.py` is a
**deprecated wrapper** that forwards here (no hard-coded machine paths):

- `--list` — catalog（可加 `--paper` / `--model` 过滤）
- `--paper` / `--model STEM` / `--asset NAME` — 按 `paper_main`、模型 YAML 或目录名下载
- 落盘：`--dest` 或 `DYNISAFETY_CHECKPOINTS` 或 `<workspace>/external/checkpoints`
- `--mano-notice` — MANO instructions only (no binary fetch)

详见 [assets.md](assets.md)。

Individual HF model/dataset licenses are those of their respective repos; they are
outside the notices in `THIRD_PARTY_NOTICES.md`.

---

## 9. Verify pins & patches

```bash
git submodule status
# expect:
#   afd5f1b… third_party/RoboTwin (dynisafety-pin-20260806)
#   ccfe380… third_party/SimplerEnv-OpenVLA

# RoboCasa trees are not submodules:
git -C third_party/robosuite rev-parse HEAD
# expect 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
git -C third_party/robocasa rev-parse HEAD
# expect 921c9a5736a8d0ea5589657898aadcfa55a6a195

scripts/setup/apply_patches.sh --check all
# exit 0 = all APPLIED; exit 2 = MISSING or DRIFTED
```

`dynisafety doctor --backend <backend>` is the full environment gate:
submodule pins, `apply_patches.sh --check`, simulator imports, assets,
checkpoints, conda env names, GPU. Optional gaps are WARN unless `--strict`.
