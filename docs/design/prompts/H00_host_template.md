# H00 — 适配器任务通用模板（H1–H6 都先读本文件）

起点分支：`hazard/integration`（T01–T06 及 T05-F、T06-F 已合并）　依赖：T01–T06、对应的 E00　需要仿真器：是　先读：T00。
**参数 `BENCH`** 由具体任务文件（H1–H6）给出。任务分支：`hazard/H-<BENCH>`。

## 目标
为 `BENCH` 实现 `HazardHost` 适配器与 BD，使其通过 `hazardlib` 一致性测试 C1–C13，并完成特权运行与适配成本账本。**适配器只实现引擎原语；一切危险逻辑都在 `hazardlib` 里。**

## 步骤

1. **环境**：激活 `BENCH` 的环境（`docs/design/env_notes/<BENCH>.md`），`pip install -e hazardlib --no-deps`。确认 `python -c "import hazardlib"` 成功，Python 版本满足 BD。
2. **BD 草稿**：编辑 `configs/hazard/bd/<BENCH>.yaml`。已核实项照 `02_benchmark_profiles.md` 填；所有 `unknown` 列成“探针清单”写进汇报。`robot_contact_bodies` 的每一项必须带匹配方式前缀（`exact:` / `prefix:` / `substr:`，D10 f），先列出该基准全部碰撞体名再写模式，并检查没有场景物体被误匹配；`HazardHost` 有 14 个必需方法，含 `hazard_state`（质心位姿与线 / 角速度，D10 e），`robot_state` 要尽量给出 `collision_body_names`、`collision_body_lin_vel_mps` 与 `held_object_bodies`；另须实现 `EpisodeDriver` 的 `reset_episode` / `step` / `policy_observation`（D11 a），`policy_observation` 返回的必须是策略实际看到的观测；`partner_name` 按 body / link 粒度上报，同一连杆的多个几何体报同一个名字（否则会被当成“第二伙伴”，误判 `constrained` / `clamp`）。`hazard_ccd` 只有在确认引擎对危险体启用了连续碰撞检测时才写 `true`（D11 d）。
3. **探针清单消除**：对每个 `unknown`，写一个最小脚本（放 `hazardlib/src/hazardhosts/<BENCH>/probes/`）实测，输出贴进汇报；结论回填 BD。**不得**用背景知识代替实测。同时实测并回填：`gravity_mps2`（自由落体拟合，D2）、危险体线 / 角阻尼能否置 0（R7；不能置 0 则不满足 Tier A，D13 h）、`eef_contact_radius_m`（用 C13 检验）、工作区上界与桌面高度（01 §3.4，D13 k）。
4. **实现适配器**：`hazardlib/src/hazardhosts/<BENCH>/`。遵守 R0–R16：
   - R2：**禁止**以任务名为条件的分支；逐任务信息写进 `configs/hazard/annotations/<BENCH>.yaml` 并统计行数；
   - R5/R6：危险体不进状态向量与成功判定；停放不交互；
   - R7/R8：释放速度不加下限，且由运行时按固定窗口给出，适配器**不得**自行用钩子步长计算；运动学重叠由运行时回滚处理，适配器只提供位姿 / 释放 / 接触原语；`fall` 族无运动学阶段；
   - 钩子语义：逐条满足 `01_integration_rules.md` §3.3 的 16 条要求（D12 e、D13 c），尤其是钩子包住每个最细物理步、`begin_episode` 在重置完成之后、危险体伙伴名等于 `HazardHandle.key`、`hazard_contacts` 是“取走”语义、运动学重叠在同一步上报、回合结束处调用 `runtime.end_episode`、一个宿主只用一个运行时（驱动、门控、校准共用同一个会话）；运行时只支持 Tier A（D12 d），在汇报里逐条写明适配器如何满足；
   - 第 14 条要在**基准原生评测循环**上验收（一致性套件覆盖不到，D13 j）：用原生循环跑若干回合，断言每个回合恰好调用一次 `end_episode`、`hazard_events.jsonl` 的条数等于回合数，输出贴进汇报；
   - 探针 trace 按 01 §3.4 录制：机器人碰撞体位置与 `held_object`、夹爪信号是测量值还是命令值、低控制频率时物理步分辨率的夹爪事件；
   - R13：不改上游；需要复制上游代码时固定提交号并断言；补丁存 `patches/<BENCH>/`；
   - R15：缺能力 → `CapabilityError`，不静默降级。
5. **注册宿主**：`register_host("<BENCH>", factory)`，使 `python -m hazardlib.conformance --host <BENCH>` 可用。`ConformanceTarget` 要提供场景几何、可重复的名义回放（最好附名义 EEF 路径）、`build_hit_instance`（本宿主的实例构造器，C13 一并检查它）与 `inject_engine_fault(env, hazard)`（用本宿主的方式强制引擎故障；不提供则 C11 判 `fail`，D13 h）。
6. **一致性测试**：C1–C13 全部通过（规则允许的跳过除外；C13 若无可重复名义回放，报告为 `unverified`，**不算通过**，须在汇报中说明）。失败项先查适配器，其次才怀疑套件；若确认套件有误，在汇报里提出，**不得**自行改容差，也**不改** `conformance.yaml`（D13 i）：汇报里给出每项的正常误差 e_ok 与建议容差（表格：检查、键、现值、本宿主 e_ok、建议值、理由），由协调者统一收紧。`runtime.yaml` / `semantics.yaml` / `intensity.yaml` 冻结前得到的报告只作开发参考，冻结要用冻结版配置重跑（D13 a）。
7. **特权运行**（不依赖学习策略，除非任务文件另有规定）：
   - 零 / 保持动作运行：确认机器人不漂移，测 **T_stop**（`hazardctl calibrate`，至少 3 个名义速度、每个 N 次）；
   - 演示回放或脚本专家：在 20 个无危险回合上记录成功率与 trace（作为探针素材）；
   - Oracle 停等：在 ≥10 个“名义必碰”实例上运行并汇总证书标签分布。
8. **成本账本**：适配器代码行数（`cloc` 或 `wc -l`，不含测试与探针）、注解行数、上游补丁数、首次运行一致性测试通过数÷总数、会话时长、阻塞。

## 验收标准（贴输出）

1. `python -m hazardlib.conformance --host <BENCH> --out report.json` 的摘要：通过 / 失败 / 跳过及原因；失败为零或逐项说明。
2. BD 中无 `unknown`（`strict=True` 加载通过）；`tier()` 的结果与实测一致。
3. T_stop 表（名义速度 × 重复次数 → 中位数与四分位距）。
4. 特权运行结果：无危险回合成功率；证书标签分布；任何被排除样本的数量与类别。
5. 成本账本齐全。
6. `git status` 干净；无大文件；`src/dynisafety/` 无改动（`git diff --stat origin/hazard/integration -- src/dynisafety` 为空）。

## 不要做
不要实现探针 / 实例化 / 门控（T06 已做）；不要跑学习策略的大规模评测；不要改既有后端；不要为通过测试放宽容差。

## 汇报
`docs/design/reports/H-<BENCH>.md`，按 T00 模板，并附“对契约的反馈”：哪些方法签名在这个基准上别扭、缺什么、多了什么。
