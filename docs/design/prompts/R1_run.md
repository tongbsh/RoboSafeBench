# R1 — Run 阶段核心：`hazardctl run` / `report`（D16 c）

起点分支：从 `hazard/integration` 新建 `hazard/R1`　依赖：T06（含 T06-F）已合并；与 T07 并行　需要仿真器：否　先读：T00；`00_paradigm.md` §3、§5.4、§6、§7、§8 与 §13 的 **D16（本任务的依据：c）**、D10 h、D12 a、D13 a；`01_integration_rules.md` §3.1a、§3.3；`hazardlib/src/hazardlib/driving.py`（`EpisodeSession.run_episode`、`ReferencePolicy`）、`gates/`（证书标签、`in_main_score`、`annotate_exposure`）、`semantics/invalid.py`（`metric_denominator`）、`calibrate/protocol.py`（`sigma_from_summary`）、`manifest/freeze.py`（`verify_manifest`）。

## 目标

补上流水线最后一步：给定冻结的实例（或试点实例）与一个被测策略，批量运行危险回合与配对的无危险对照回合，记录事件日志、成败、策略看到的画面，并汇总成按族、τ / Σ 分档、证书标签分桶的指标表。全部在 FakeHost 上端到端验证。接真实策略与视频编码是 R1-L 的事。

## 与 T07 的边界（并行进行，避免冲突）

**不要改**：`instantiate/`、`probe/`、`conformance/`、`gates/g0.py`、`host.py`、`spec/benchmark_descriptor.py`（T07 在改）。
**可以改**：新包 `hazardlib/src/hazardlib/run/`；`cli.py`（只加子命令）；`driving.py` 只做下面第 2 条那一处改动；新配置 `configs/hazard/run.yaml` 与其加载；测试；`docs/design/reports/R1.md`。若发现必须改上面“不要改”的文件，停下来写进汇报，不要自行改。

## 范围

1. **输入与模式（D16 c）**
   - 正式：`--manifest-dir DIR`。先调 `verify_manifest`，不通过就拒绝；实例、TaskContext、BD、校准从冻结目录读取；输出记 `manifest_hash`。
   - 试点：`--instances gated.jsonl --task-context TC.yaml ... --bd bd.yaml [--calibration calibration.yaml] --pilot`，配合 `--config-dir` 指向试点配置目录。不带 `--pilot` 时拒绝非冻结输入；输出的每条记录与汇总都带 `pilot: true`。
   - 两种模式的结果**不得混合**：`report` 遇到 `pilot` 不一致或 `manifest_hash` 不一致的输入就拒绝。
   - 证书标签过滤 `--labels`（默认全跑；`unsolvable_by_probes`、`baseline_unusable` 照常运行但只进单列的桶）。

2. **配对对照回合**
   - 每个危险实例配一个同 `task_id`、`layout_id`、`seed` 的无危险回合：`instance=None`、`control_id` 由实例 id 派生、`control_onset_s` = 实例的名义 onset（D12 a），用于 `toppled` 的净值与无危险成功率。
   - `EpisodeSession.run_episode` 现在调用 `runtime.arm(env, None, control_id=...)` 时不传 `control_onset_s`。在 `run_episode` 签名里加可选关键字 `control_onset_s`，原样透传给 `arm`，只改这一处（T07 也改 `driving.py`，但改的是 trace 部分，协调者合并时处理）。
   - 同一布局的多个实例共用一个对照回合时要去重；结果里写明对照回合与实例的对应关系。

3. **回合运行**
   - 被测策略经 `ReferencePolicy` 协议接入（`reset(task_id, layout_id, seed)`、`act(observation)`），沿用 `--policy M:F` 加载约定。仿真在策略推理期间不推进（`run_episode` 已是同步的）；推理墙钟时间按步记录，只作诊断，不进任何指标。
   - 回合长度、是否成功即停等数值放进新的 `configs/hazard/run.yaml`（全部 `PLACEHOLDER`，代码无默认；按既有配置加载方式实现 strict 加载与哈希，`freeze` 的 `config_hashes` 是否要包含它，写进汇报的建议，**不改 freeze**）。
   - **§5.4 检查**：被测策略的名字出现在 TaskContext 的探针来源里时拒绝运行（报错信息指出是哪个 TaskContext）。名字用 `--policy-name` 显式给出并写进每条记录；不给就拒绝。
   - 每个回合写一条记录：`HazardEventLog`（含 `task_success`；用 `annotate_exposure` 补 `exposure_low`）、实例 id、族、`spec_id`、τ、Σ（有校准时经 `sigma_from_summary`，删失按 D8 处理并标记）、证书标签、`in_main_score`、`success_t_s`、`official_contact_count`、`first_visible_t_s`、策略名与 `describe_policy`、推理耗时摘要、对照回合 id、帧文件路径（如有）、`pilot` / `manifest_hash`、种子。
   - **可续跑**：输出目录里已有完整记录的（实例, 策略, 种子）跳过；中断不丢已写记录（逐条追加）。
   - 只用一个 `--env`；多环境并行留给后续。

4. **画面记录（核心只定接口）**
   - `FrameRecorder`：一个包装 `ReferencePolicy` 的策略，把每次 `act` 收到的观测里指定键（`--frame-keys`）的图像交给 `FrameSink`，同时记录该帧的仿真时间（从宿主的 `time_state` 读）与危险体阶段（可取时）。记录的就是策略实际看到的画面。
   - `FrameSink` 协议：`open(episode_id, meta)`、`add(key, frame, t_s, phase)`、`close() -> paths`。核心里只提供测试用的内存实现，以及一个只用 numpy 的 `.npy` 写盘实现；mp4 编码放在核心之外（R1-L 在 `hazardhosts/` 下实现），核心不 import imageio / cv2。
   - `--frame-sink M:F` 加载外部实现；不给就不录。

5. **汇总 `hazardctl report`**
   - 输入一个或多个 run 输出目录，输出 `summary.yaml` 与一份 Markdown 表。
   - 分组：策略 × 族 × τ 分档（有校准时另出 Σ 分档；分档边界放 `run.yaml`，PLACEHOLDER）。
   - 每组给出：回合数；危险回合成功率与对照回合成功率；官方接触率；`J`、`vₙ` 的中位数与四分位距（只在有官方接触的回合上）；`min_separation_m`、`ttc_min_s` 的中位数；`clamp` 比例；`toppled` 率与相对对照的净值（D10 h）；`exposure_low` 比例；无效样本按三类计数（`metric_denominator`）。
   - 主表只含 `in_main_score` 为真、非 `exposure_low`、非 `setup_invalid` / `sim_unstable` 的回合；`hazard_induced_unstable` 留在分母里；其余全部单列，不静默丢弃。`uncertified` 进主表并加脚注。
   - 比例给区间估计（方法与置信水平写进 `run.yaml`，PLACEHOLDER；方法在汇报里说明理由）。
   - `engine_peak_force_n` 只出现在附表，不参与排序（§6）。

6. **测试**（只增加）
   - 用 FakeHost 与几个脚本策略（例如：照名义路径走、从不动、在危险体逼近时停下），覆盖：对照回合的配对与去重；`control_onset_s` 透传后 `toppled` 净值正确；§5.4 拒绝；正式模式下 `verify_manifest` 失败即拒绝；试点与正式混合时 `report` 拒绝；续跑跳过已完成回合；`FrameRecorder` 帧数等于控制步数且时间单调；汇总表的每个数与手算一致（至少一个小数据集逐项对）；主表与单列桶的划分。
   - 端到端：`record → probe → instantiate → gate → calibrate → freeze → verify → run → report` 在 FakeHost 上走通一次（正式模式），再用试点目录走一次（试点模式）。
   - 既有卫生测试照常通过（无仿真库 import、无数值默认值、Python 3.8 兼容）。

7. **汇报** `docs/design/reports/R1.md`：验收输出、CLI 用法示例、输出文件格式（字段表）、汇总表样例（FakeHost 数据）、`run.yaml` 键表与每个键的含义、偏离与理由、给 R1-L 的接入说明（策略桥接如何实现 `ReferencePolicy`、mp4 `FrameSink` 放在哪里、LIBERO 观测键）。

## 验收标准（贴输出）

1. `python -m pytest hazardlib -q` 在 `hz38`、`hz311` 全绿（`PYTHONPATH=<worktree>/hazardlib/src`；对比集成分支基线：`hz38` 1388 passed、1 skipped，`hz311` 1389 passed）。
2. FakeHost 上正式模式与试点模式的端到端命令序列与输出，含 `report` 生成的 Markdown 表。
3. `python -m hazardlib.conformance --host fake` 仍是 13 项 pass。
4. `git status` 干净；`git diff --stat origin/hazard/integration` 只含上面允许的范围。

## 不要做

- 不改 `docs/design/00–03`、`src/dynisafety/`；
- 不接真实策略、不跑仿真器、不实现 mp4 编码（R1-L）；
- 不给 PLACEHOLDER 赋值（测试用临时配置目录）；
- 不改 T07 在改的文件；不改 `freeze` 与 Manifest 格式（建议写进汇报）。
