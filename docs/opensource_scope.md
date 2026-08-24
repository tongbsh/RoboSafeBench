# DyniSafety 开源范围与 Git 清单

本文档盘点当前工作树（约 **440 GB**）里每个顶层目录的功能，并给出开源发布时应纳入 / 不应纳入 git 的结论。  
盘点日期：2026-08-24。结论按**新统一框架**（`src/dynisafety` + `configs/` + `tests/`）来写，不是按当前 `origin/main` 上那棵旧树。

> **当前 git 与磁盘严重脱节（开源前必须先处理）。**  
> `origin/main` 只跟踪约 **163** 个文件，主体仍是旧三套框架：`dynisafety_libero/`、`dynamic_SimplerEnv/`、`dynamic_RoboTwin/`。  
> 新框架几乎全部未入库：`src/dynisafety/backends/`、`configs/`、`tests/`、`docs/`、`scripts/` 跟踪数为 **0**。  
> 因此「把现在的仓库直接 push 公开」会发布一套**已经废弃的旧代码**，同时漏掉真正的 benchmark。

---

## 0. 一句话对照

| 类别 | 磁盘现状 | 开源 git 应收 |
|------|----------|----------------|
| 评测框架、扰动、指标、诊断、CLI | 在 `src/`，**尚未入库** | **必须开源** |
| 协议 YAML（L0–L3 / suite / 模型） | 在 `configs/`，**尚未入库** | **必须开源** |
| 用户文档、安装脚本、补丁、CI、许可证 | 已写好，多数未入库 | **必须开源** |
| 仿真器克隆、VLA 权重、数据集 | `external/` ~401 GB + `third_party/` ~26 GB | **禁止入库** |
| 本机评测 / 调参 / 标定产物 | `results/`、`artifacts/` | **禁止入库** |
| 重构前旧框架、内部交接、审稿 Prompt | `old_cache/`、`docs/plan/`、`docs/refine_iclr_prompt/` | **禁止开源** |
| 旁路数据采集（WRM 训练包） | `A_robosafe_capture_datasets/` | **不要放进本 benchmark 仓** |

---

## 1. 必须开源的核心

这些目录/文件构成「别人 clone 之后，按文档装环境、跑 smoke、复现论文矩阵」所需的全部源码与协议。缺任何一块，仓库都不是完整 benchmark。

### 1.1 根文件（仓库身份）

| 路径 | 功能 | 开源理由 |
|------|------|----------|
| `README.md` | 项目入口：四 backend、L0–L3、5 分钟 smoke、指标口径 | 用户第一页 |
| `LICENSE` | MIT | 法律必需 |
| `CITATION.cff` | 引用元数据（当前仍是匿名 `TODO`） | 学术引用；公开前改真实作者 |
| `THIRD_PARTY_NOTICES.md` | 仿真器 / MANO / GSO / RoboCasa 资产许可证全文或摘要 | 再分发合规 |
| `pyproject.toml` | 包名 `dynisafety`、依赖、CLI 入口、ruff/mypy/pytest | 可安装性 |
| `Makefile` | `install-*` / `lint` / `test` / `coverage` | 标准入口 |
| `Dockerfile` + `.dockerignore` | LIBERO / MuJoCo 镜像 | 可复现安装 |
| `.gitignore` | 挡住权重、结果、缓存、密钥 | 防止误提交 |
| `.gitmodules` | 声明 SimplerEnv / RoboTwin 子模块 | **必须改掉本机绝对路径后再公开**（见 §4.1） |
| `.pre-commit-config.yaml` | ruff + 大文件 >5 MiB 拦截 | 防止权重/视频进仓 |
| `.github/workflows/ci.yml` | 无 GPU 的 lint / mypy / `pytest -m "not sim"` | 贡献者质量门 |

### 1.2 `src/dynisafety/` — 评测框架本体（约 14 MB 源码 + 轻量 mesh）

包布局与职责：

| 子包 | 功能 |
|------|------|
| `cli/` | 用户入口：`eval` / `sweep` / `serve` / `doctor` / `config` / `report` |
| `config/` | YAML 分层加载、占位符 `${workspace}` / `${third_party}` / `${env:}`、`RunConfig` 契约 |
| `core/` | **结局契约**：`EpisodeRecord`（schema 2.0）、接触/力、SR/CR/SASR、联合象限 SS/US/CFF/CIF、JSONL 写出 |
| `diagnostics/` | **过程契约**：逐步 trace、四轴（反应/停止/规避/恢复）、9 类行为、离线标定 |
| `backends/` | 四仿真器适配 + **L0–L3 扰动实现**（红线目录）。只做仿真器翻译，不定义指标 |
| `backends/_engine/` | 跨 backend 的 MuJoCo 接触等公共引擎 |
| `backends/*/disturbance/` | L1 突降 / L2 滑动或地面接近 / L3 人手或行人；几何与语义以 YAML 为准 |
| `backends/*/assets/` | 本仓原创或已声明可再分发的轻量视觉：L3 `hand_pusher` STL、RoboCasa 行人/障碍 OBJ、可选 GSO overlay（~6.4 MB，CC BY 4.0） |
| `policies/` | VLA 适配器、动作空间、HTTP/WS/ZMQ transport、Oracle 暂停、AEGIS CBF-QP |
| `policies/adapters/` | `pi05` / `openvla_oft` / `xvla` / `starvla` / `xiaomi` / `random` 等模型接线 |
| `policies/_libero_impl/`、`_robotwin_impl/` | 从旧框架抽出的模型实现子集，评测运行时依赖 |
| `policies/safety/` | Oracle Stop-Wait-Resume、AEGIS 滤波器（默认关，不影响原始 VLA） |
| `runner/` | rollout 循环、sweep 调度、conda 环境与端口分配 |
| `report/` | 从 JSONL 出表图（`table_main` / `table_behavior` / 行为分解图），禁止从日志正则反解析 |
| `feasibility/` | Base vs Oracle vs AEGIS 可行性矩阵（论文「场景可安全解决」证据） |

**必须入库**：整个 `src/dynisafety/`，包括 `backends/` 与其轻量 `assets/`。  
交接文档已写明：`backends/*/disturbance/**` 目前**尚未纳入 git**，开源前这是第一优先级。

**不要入库**：`src/dynisafety.egg-info/`（本地 `pip install -e .` 产物）。

### 1.3 `configs/` — 冻结的评测协议

默认值只允许出现在 YAML，Python/CLI 不写业务默认。合并顺序：`base` → 全局 `levels/` → `backends/{b}` → `hand.yaml` → `backends/{b}/levels/` → suite → task → model → `--set`。

| 子目录 | 功能 |
|--------|------|
| `base.yaml` | 全局默认：seed、伤害力阈值、SASR φ、行为诊断阈值（P5 已冻结） |
| `levels/` | L0–L3 叙事默认（台面语义） |
| `backends/<name>.yaml` | 每仿真器接触源、时钟、`extra.*` |
| `backends/<name>/levels/` | **该仿真器的扰动几何与触发**（红线：开源后一行都不该被静默改掉） |
| `backends/<name>/suites/`、`tasks/` | 任务目录与指令 |
| `backends/<name>/hand.yaml` | L3 人手/行人几何 |
| `models/*.yaml` | 每个 VLA 的 adapter、动作空间、checkpoint **占位路径**（指向 `external/`，不包含权重） |
| `suites/paper_main.yaml` | 论文主矩阵权威（约 884 job） |
| `suites/smoke*.yaml` | 5 分钟冒烟 |
| `suites/calib_behavior_diagnostics.yaml` | 行为诊断标定矩阵 |
| `report/` | 报表 grid 与展示设置 |
| `safety/`、`backends/libero/{oracle,aegis}/` | Oracle / AEGIS 开关（默认关） |
| `feasibility/` | 可行性小矩阵 |

全部应开源。YAML 里的路径必须保持 `${workspace}` / `${env:VAR}`，禁止本机绝对路径。

### 1.4 `tests/` — 无仿真器的契约锁

| 内容 | 功能 |
|------|------|
| `test_*.py` | 契约、指标、诊断、backend 接线、doctor、卫生检查；`pytest -m "not sim"` 可在 CI 跑 |
| `golden/` | 扰动几何 / rollout 黄金快照，防止 L1–L3 被改歪 |
| `fixtures/` | 可行性等离线夹具 |

必须开源。`tests/__pycache__/` 不要入库。

### 1.5 用户文档（`docs/` 的公开子集）

`docs/README.md` 已列出用户文档。下列应随仓发布：

| 文件 | 功能 |
|------|------|
| `docs/README.md` | 文档目录 |
| `installation.md` | conda / Docker / 四 backend 安装 |
| `configuration.md` | YAML 与 CLI |
| `contracts.md` | `EpisodeRecord`、输出布局、禁止事项 |
| `metrics.md` | SR / CR / force / SASR / SSR 口径 |
| `behavior_diagnostics.md` | 四轴与 9 类（已冻结） |
| `benchmark_design.md` | L0–L3 台面 + RoboCasa 地面导航 |
| `spawn_plan.md` | L2 spawn 多样性字段 |
| `reproducing_paper.md` | 从干净 clone 到论文表图 |
| `adding_a_model.md` | 新模型适配 6 步 |
| `third_party.md` | pin、许可证、补丁、安装入口 |
| `migration.md` | 旧命令 → 新 CLI |
| `leaderboard.md` | 结果提交草案 |

**不要**把 `docs/plan/`、`docs/refine_iclr_prompt/`、`docs/experiment_upgrade/` 原样公开（见 §2.5）。

### 1.6 `scripts/` — 安装与正式评测入口

| 子目录 | 功能 | 开源 |
|--------|------|------|
| `install/` | `install_backend.sh`、四 backend 安装、`apply_patches.sh`、`download_assets.py`（只下权重，不带 MANO 二进制） | **必须** |
| `run/` | `01_setup` → `03_smoke` → `10_eval` / `20_serve` / `30_sweep` / `40_report`，自动切 conda | **必须** |
| `libero/materialize_obstacle_pack_gso.py` | 从 GSO 生成 LIBERO overlay（默认关） | **建议** |
| `models/lingbot_va/` | LingBot-VA serve 包装与小补丁 | **建议**（该模型在 `paper_main`） |
| `debug/` | 调参小跑、长矩阵 serve+eval | 灰色，见 §3.1 |
| `feasibility/` | 授权后的 Oracle/AEGIS GPU 矩阵 | 灰色，见 §3.2 |

权威用户入口是 CLI：`dynisafety eval` / `sweep` / `report`。脚本只是 conda 切换包装。

### 1.7 `patches/` — 上游最小补丁

| 路径 | 功能 |
|------|------|
| `patches/simplerenv/*.patch` | `json_numpy`、visualize 守卫、openpi websocket、ManiSkill2 SHA |
| `patches/robotwin/dynisafety_hooks.patch` | `Base_Task` 上 6 个挂钩；业务在 `backends/robotwin/hooks.py` |

必须开源。完整仿真器树由安装脚本拉取后再 `apply_patches.sh`。

### 1.8 环境、示例、模板

| 路径 | 功能 |
|------|------|
| `environment/dynisafety-{libero,simplerenv,robotwin,robocasa}.yml` | 四套互斥 conda 环境（RoboCasa 为 NumPy 2） |
| `requirements/{libero,simplerenv,robotwin,robocasa}.txt` | 短 pin；禁止同一 env 装多个 backend |
| `docker/README.md` | LIBERO 镜像用法 |
| `examples/example_agent/` | 玩具 VLA：本地 + HTTP 适配示范（数字不得当论文结果） |
| `templates/model_adapter_template.py` | 新 adapter 模板 |

均应开源。`examples/**/__pycache__/`、玩具权重若变大则只留生成脚本。

### 1.9 `tools/` 中应随仓走的入口

| 文件 | 功能 |
|------|------|
| `calibrate_diagnostics.py` | P5：只读 trace，扫阈值，出 FPR / 敏感性（不跑仿真） |
| `audit_models.py` | 检查 `paper_main` 各模型的 repo / checkpoint 是否存在 |
| `model_env.py` | `scripts/run/*` 用来解析 conda 环境名 |
| `make_report_grids.py` | 生成 `configs/report/grids/` |
| `feasibility_runner.py` | 可行性矩阵；默认拒 GPU |

一次性排障脚本（`audit_l2_spawn.py`、`diagnose_l2_spawn_*.py`、`probe_l2_spawn.py`、`recount_robotwin_contact.py`）见 §3.3。

---

## 2. 不能加入 git / 不能随仓开源

原则：别人可以用**安装脚本 + 上游仓库 + 自己接受的许可证**得到同样的东西，就不要把二进制或克隆树推进本仓。含审稿策略、本机路径、密钥、限制协议资产的，一律不公开。

### 2.1 超大体量：仿真器、权重、数据集

| 路径 | 约体积 | 功能 | 不能入库的原因 |
|------|--------|------|----------------|
| `external/` | **401 GB** | 本机安装根：仿真器源码、VLA 仓、权重、数据集、MANO、bare mirror | `.gitignore` 已忽略。由 `scripts/install/*` 与 `download_assets.py` 获得 |
| `external/simulators/` | ~35 GB | `dyniSafety_libero`、`RoboTwin`（含 ~16 GB assets）、冗余 `RoboTwin_simenv_copy`（17 GB）、旧 `safelibero` / RLBench | 上游或 fork 的完整树；RoboTwin assets 单独下载 |
| `external/model_repos/` | ~54 GB | `openpi`、`X-VLA`、`starVLA`、`InternVLA-*`、`Xiaomi-Robotics-1` 等推理仓 | 各模型自己的 git；YAML 用 `${workspace}/external/model_repos/...` |
| `external/checkpoints/` | **271 GB** | 论文/调试用权重（univla 71G、RLinf 57G、lingbot 42G…） | 权重许可证各异；体积不可接受 |
| `external/datasets/` | ~46 GB | RLBench 等 | 与本 benchmark 主路径无关的大数据 |
| `external/assets/mano_v1_2/` | 257 MB | MANO / SMPL+H 手模 | **MPI/MPG 非商业协议，禁止再分发** |
| `external/mirrors/RoboTwin.git` | 8 MB | 本机 bare git，供 `.gitmodules` 使用 | 私有 pin；公开仓不能指向 `/root/autodl-tmp/...` |
| `checkpoints/` | 12 GB | 根目录另一份权重（DiffSynth / Wan-AI） | 与评测无关的生成模型权重 |
| `third_party/robocasa` | **23 GB** | RoboCasa 安装树（含厨房资产） | **不是 submodule**；代码 MIT，厨房资产 CC BY 4.0，由 `install_robocasa.sh` 拉取 |
| `third_party/robosuite` | 1.3 GB | robosuite ≥1.5（git master） | 同上，安装脚本克隆，不要 vendor |
| `third_party/vlsa-aegis` | 1.8 GB | AEGIS / GroundingDINO / openpi 旁路树 | 可选安全层依赖，应文档化 clone，不要整树提交 |
| `third_party/dyniSafety_libero` | （安装后出现） | SafeLIBERO / LIBERO 评测树 | 设计上就不是 submodule，由 `install_libero.sh` 生成 |

`third_party/SimplerEnv-OpenVLA`（~702 MB）与 `third_party/RoboTwin`（~20 MB 代码树，不含 16 GB assets）是 **submodule 工作副本**。git 只应记录 **gitlink（一个 SHA）**，不要把工作树文件当普通文件提交。RoboTwin 的公开 URL 必须先从本机绝对路径改成可克隆的上游或官方 mirror（见 §4.1）。

### 2.2 运行产物（每次评测都会再生）

| 路径 | 约体积 | 功能 | 不能入库的原因 |
|------|--------|------|----------------|
| `results/` | 72 MB（会持续涨） | 规范评测输出：`l{0–3}/<run_id>/{episodes.jsonl,manifest.json,resolved_config.yaml,traces/,videos/}` | 契约规定的 canonical 输出；含本机路径、视频、未发表数字。`.gitignore` 已忽略 |
| `artifacts/` | **672 MB** | 调参 / 可行性 / P5 标定落盘：`debug/`、`feasibility/`、`calibration/` | **当前 `.gitignore` 未覆盖**，开源前必须补上。含 GPU 日志、失败 screen log、sweep 中间结果 |
| `**/videos/`、`**/*.mp4` 等 | — | 评测与采集视频 | 已 ignore；体积大且非源码 |
| `*.log`、`logs/`、`wandb/`、`mlruns/` | — | 实验追踪 | 本机噪声 |

论文表图应在发表后单独放 release / 网盘 / Hugging Face，不要塞进 git。若要给复现者一个小样例，最多放 `tests/fixtures/` 里几条合成 JSONL。

### 2.3 限制协议与不可再分发资产

| 路径 | 功能 | 原因 |
|------|------|------|
| `external/assets/mano_v1_2/`、`assets/mano_v1_2/` | MANO 手模 | MPI/MPG：**本仓与 release 都不得附带**。用户自行接受协议后放置。RoboTwin L3 默认用的是仓内 `hand_pusher`，不依赖 MANO |
| RoboCasa 完整厨房 zip（~10 GB） | 纹理 / fixture | 安装脚本下载，不提交 |
| 各 VLA 权重 `*.safetensors` / `*.pt` / `*.ckpt` | 推理 | 许可证与体积；用 `download_assets.py` + Hugging Face |

仓内已声明可带的轻量资产（应开源）：

- `src/dynisafety/backends/robotwin/assets/hand_objects/`、`simplerenv/assets/hand_pusher_*.stl`
- `src/dynisafety/backends/robocasa/assets/{pedestrian,obstacles}/*.obj`（本仓原创低模）
- `src/dynisafety/backends/libero/assets/obstacle_pack_gso/`（GSO，CC BY 4.0，默认 `enabled: false`）

### 2.4 重构前旧框架与本地留档

| 路径 | 功能 | 原因 |
|------|------|------|
| `old_cache/` | 旧 `dynisafety_libero` / `dynamic_SimplerEnv` / `dynamic_RoboTwin`、内部交接稿、重构 prompt | `.gitignore` 已忽略；**不属于发布版**。活代码已迁到 `src/` |
| `dynisafety_libero/`、`dynamic_SimplerEnv/`、`dynamic_RoboTwin/` | **当前 git 仍在跟踪的旧三套评测** | 已被 `src/dynisafety` 取代。公开前应从 index **删除**（工作区副本可留本机） |
| `legacy/`、`scripts/dynamicRLBench/` | 未恢复的 RLBench / BridgeVLA 旧脚本 | 不在四 backend 承诺范围内 |
| `download_hf.py` | 旧权重下载入口 | 已弃用，逻辑在 `scripts/install/download_assets.py` |
| `vlaModelHome/`、`simulator_env/`、`past_files_cache/`、`mirrors/` | 历史本机布局 | 已 ignore；功能并入 `external/` / `third_party/` |

### 2.5 内部过程文档（含审稿策略，不宜公开）

| 路径 | 功能 | 不能开源的原因 |
|------|------|----------------|
| `docs/refine_iclr_prompt/` | ICLR 改进总计划、审稿意见拆解、多 Agent Prompt、handoff | 含拒稿原因、论文叙事策略、未发表实验设计；用户不需要「怎么喂 Cursor Agent」 |
| `docs/plan/` | Agent 交接（如 P5/P6 行为诊断、本机 GPU PID、sweep 路径） | 过程备忘，含本机绝对路径与运行中作业 |
| `docs/experiment_upgrade/` | Oracle / AEGIS / RoboCasa 窗口交接 | 科学内容应吸收进 `benchmark_design.md` / `metrics.md` 后再公开；handoff 原文不要原样挂出 |
| `old_cache/docs_internal/` | `HANDOFF.md`、M0–M12 进度、过时设计稿 | 已 ignore |

对外只保留 §1.5 的用户文档。

### 2.6 旁路项目：`A_robosafe_capture_datasets/`

| 项 | 说明 |
|----|------|
| 功能 | **隔离模块**：hook 现有 `make_env` / `rollout`，采集 L0–L3 成功/失败轨迹，写成 action-conditioned world reward model 的 episode 包（视频 + npy + `annotation.json`） |
| 与 benchmark 的关系 | **刻意不改** `src/dynisafety`；产物在本目录 `outputs/` |
| 为何不放进本仓 | 1）不是评测协议，是另一条训练数据管线；2）名称带临时前缀 `A_`；3）公开后会让用户把「benchmark」和「WRM 数据集」混为一谈。应独立仓库或晚于 benchmark 再发 |

### 2.7 IDE、缓存、密钥

| 路径 | 功能 | 原因 |
|------|------|------|
| `.claude/`、`.cursor/`、`.vscode/` | 编辑器/Agent 本机设置 | 已 ignore；可能含本机路径 |
| `.mypy_cache/`、`.pytest_cache/`、`.ruff_cache/`、`__pycache__/` | 工具缓存 | 已 ignore |
| `.env`、`*.pem`、`credentials.json`、`wandb_api_key*` | 密钥 | 已 ignore。`configs/models/lingbot_va.yaml` 的 `api_key: null` 可保留（占位，不是真密钥） |

---

## 3. 灰色地带（建议默认处理）

不是「违法」，而是公开后维护成本高或容易误导。建议如下。

### 3.1 `scripts/debug/` — 建议开源，但要去本机化

功能：日常调参（`eval_debug.sh`、`launch_vla_{serve,eval}.sh`），默认值偏开发机（`pi05`、固定端口、输出到 `artifacts/debug/`）。

- **建议**：随仓发布，README 写明「非正式矩阵」。
- **发布前**：去掉机器相关假设；输出目录与正式 `results/` 对齐。

### 3.2 `scripts/feasibility/` 与 `configs/feasibility/`

功能：Oracle / AEGIS 可行性；脚本默认 `BLOCKED_UNTIL_AUTHORIZED`。

- 若论文包含「场景可安全解决」：代码 + YAML + 用户文档应开源。
- 本机 `artifacts/feasibility/**` 的真实 run **不要**开源。

### 3.3 一次性 `tools/` 排障脚本

`audit_l2_spawn.py`、`diagnose_l2_spawn_ab.py`、`diagnose_l2_spawn_rollback.py`、`probe_l2_spawn.py`、`recount_robotwin_contact.py`：某次 L2/接触修 bug 的离线审计。

- **建议**：不进公开仓，或移到内部 `old_cache/tools/`。
- 锁住语义的应是 `tests/test_*_l2_spawn.py` 与 golden，不是这些脚本。

### 3.4 `artifacts/calibration/` 的「官方数字」

P5 冻结结论应写进 `docs/behavior_diagnostics.md`。  
`artifacts/calibration/p5_libero_pi05/` 下的 csv/png 是本机标定过程，不要整目录提交。若需要可复现附件，另做小型 release，不要进 git。

### 3.5 `third_party/` 在公开仓里应是什么

公开仓里的 `third_party/` 应接近空目录 + submodule 指针：

```text
third_party/
├── SimplerEnv-OpenVLA   ← submodule gitlink（SHA ccfe380…）
├── RoboTwin             ← submodule gitlink（公开 URL + SHA afd5f1b…）
└── （robosuite / robocasa / dyniSafety_libero / vlsa-aegis 不出现，由 install_*.sh 创建）
```

不要把 26 GB 克隆树推进 GitHub。

---

## 4. 开源前必须改的仓库级问题

### 4.1 `.gitmodules` 指向本机绝对路径

当前：

```ini
[submodule "third_party/RoboTwin"]
    url = /root/autodl-tmp/code/dynamic_safty_bench/external/mirrors/RoboTwin.git
```

外人 `git submodule update` 会失败。公开前应改为可访问的 HTTPS（官方 [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 或你们托管的公开 pin），并在 `docs/third_party.md` 写清 SHA `afd5f1bbc5759ac1512dc42b3cdaf8a218fb7c00`。  
`third_party/SimplerEnv-OpenVLA` 现为 `git@github.com:DelinQu/SimplerEnv-OpenVLA.git`，公开仓更宜用 `https://`。

### 4.2 旧树仍在 git index 里

公开前应从版本库移除（本机可保留）：

- `dynisafety_libero/`
- `dynamic_SimplerEnv/`
- `dynamic_RoboTwin/`
- `legacy/`
- 根目录 `download_hf.py`

同时把 §1 全部 **add** 进新的发布分支。

### 4.3 `.gitignore` 缺口

下列磁盘目录**现在不会被忽略**，`git add .` 会误加：

```gitignore
# 本机实验落盘（调参 / 可行性 / 标定）
artifacts/

# 旁路 WRM 采集（独立项目）
A_robosafe_capture_datasets/outputs/
# 若采集代码不随本仓发布，整目录忽略：
# A_robosafe_capture_datasets/

# 内部过程文档
docs/plan/
docs/refine_iclr_prompt/
docs/experiment_upgrade/

# 安装脚本生成的 third_party 工作树（保留 submodule gitlink）
third_party/robocasa/
third_party/robosuite/
third_party/vlsa-aegis/
third_party/dyniSafety_libero/
```

`external/`、`results/`、`old_cache/`、权重后缀、`*.mp4` 已在现有 `.gitignore` 中。

### 4.4 元数据

- `CITATION.cff` / README 匿名 `TODO`：双盲期间可留，公开库必须换成真实作者与仓库 URL。
- `LICENSE` 版权年为 2025，发布时按实际更新。

---

## 5. 建议的公开仓库树

```text
dynisafety/                          # 公开 git
├── README.md, LICENSE, CITATION.cff, THIRD_PARTY_NOTICES.md
├── pyproject.toml, Makefile, Dockerfile, .dockerignore
├── .gitignore, .gitmodules, .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/dynisafety/                  # 框架 + 四 backend + 轻量 mesh
├── configs/                         # 协议 YAML
├── tests/                           # 含 golden / fixtures
├── docs/                            # 仅 §1.5 用户文档（可含本文作维护说明）
├── scripts/install/  scripts/run/
├── scripts/libero/  scripts/models/ # 可选
├── patches/
├── environment/  requirements/  docker/
├── examples/example_agent/
├── templates/
└── tools/                           # 仅 §1.9 入口
```

用户机器上**另外出现、永不提交**：

```text
external/          # 权重 + 模型仓 + MANO
third_party/*      # install 拉下来的仿真器工作树
results/           # 评测 JSONL / traces / 视频
```

---

## 6. 按功能再扫一遍顶层目录

| 顶层 | 功能（一句话） | Git / 开源 |
|------|----------------|------------|
| `src/` | 统一评测包 | **必须开源**（backends 目前未跟踪，要补） |
| `configs/` | 扰动与矩阵的唯一默认值 | **必须开源** |
| `tests/` | 契约与黄金测试 | **必须开源** |
| `docs/` | 用户手册 + 内部稿混放 | **只开源用户手册** |
| `scripts/` | 安装 / 正式跑 / 调试 / 可行性 | **install + run 必须**；debug/feasibility 见 §3 |
| `patches/` | 上游最小 diff | **必须开源** |
| `environment/`、`requirements/`、`docker/` | 可复现环境 | **必须开源** |
| `examples/`、`templates/` | 接入示范 | **必须开源** |
| `tools/` | 标定 / 审计 / 一次性探针 | **部分开源**（§1.9 / §3.3） |
| `.github/` | CI | **必须开源** |
| `third_party/` | 仿真器与 AEGIS 落盘位置 | **只开源 submodule 指针** |
| `external/` | 401 GB 本机安装件 | **禁止** |
| `checkpoints/` | 额外权重 | **禁止** |
| `results/` | 规范 run 输出 | **禁止** |
| `artifacts/` | 调试/标定/可行性落盘 | **禁止**（gitignore 需补） |
| `old_cache/` | 重构前留档 | **禁止** |
| `A_robosafe_capture_datasets/` | WRM 轨迹采集 | **不要进本仓** |
| `.claude/` `.cursor/` `.vscode/` | 本机 IDE | **禁止** |
| 旧 `dynisafety_libero/` 等 | 已废弃三框架 | **从 git 删除，勿公开** |

---

## 7. 维护者检查清单（发布前）

1. 新建发布分支，**不要**直接把当前 `main`（旧 163 文件）公开。
2. 加入 §1 全部路径；确认 `src/dynisafety/backends/**` 与 `configs/backends/**/levels/**` 在版本库里。
3. 从 index 删除旧三框架与 `download_hf.py`。
4. 按 §4.3 补 `.gitignore`，再执行 `git add .` 后检查 `git status`：不应出现 `external/`、`results/`、`artifacts/`、`*.safetensors`、`*.mp4`。
5. 改 `.gitmodules` 为公开 HTTPS；`git submodule update --init` 在干净机器上能过。
6. 扫一遍 YAML / 文档，无 `/root/autodl-tmp/`、无真实 API key。
7. `pytest -m "not sim"` 与 `ruff check src tests examples` 全绿。
8. 决定 `A_robosafe_capture_datasets/` 是独立仓还是永远本机。
9. 更新 `CITATION.cff` 与 README 的匿名占位（若已过双盲）。
10. 在干净 clone + `make install-libero` + `dynisafety sweep --suite smoke --dry-run` 上走通一遍。
