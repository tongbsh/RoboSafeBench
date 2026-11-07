# T04 — `HazardRuntime` 与 `FakeHost`

起点分支：`hazard/integration`（T01、T02、T03 已合并）　依赖：T01–T03　需要仿真器：否　先读：T00。

## 目标
实现控制反转的 `HazardRuntime`（`01_integration_rules.md` §3.3）和一个**纯 numpy 的假世界** `FakeHost`，用于在没有任何仿真器的情况下测试运行时、一致性套件与流水线。

## 范围与产出

0. **前置契约改动（T02-F、T03 审阅与 D10、D11；每个小项单独成一次提交，先于其余工作；T01–T03 的既有测试必须保持绿，因契约改动必须同步修改的测试在汇报里逐个列出）**：
   - a. 在 T01 的 `BenchmarkDescriptor`（`hazardlib/spec/benchmark_descriptor.py`）新增三个顶层字段：`gravity_mps2`、`eef_contact_radius_m`（类型 float），以及 `hazard_ccd`（类型 bool，D11 d）。三者都与 `timebase.physics_dt_s` 一样允许取 `"unknown"`，并进入 `unknown_fields()` / `require_complete()` 的检查；`tier()` 不受影响。更新 `configs/hazard/bd/calvin.yaml` 草稿（三者写 `unknown`）。补测试：缺字段时报错、`strict=True` 下它们作为 unknown 被列出、BD 哈希随字段值变化。**原因**：现有加载器遇到未声明的键会直接报错（`_records.py` 的 `unexpected key(s)`），而 D1、D2、D11 d 要求每个宿主声明这三项；FakeHost 的 BD 也要给出具体值。
   - b. 在 `configs/hazard/intensity.yaml` 删除 `intrusion.slide.precursor_lead_s`（D5：可见提前量是诊断量 τ_vis，不是强度输入），并同步任何引用它的测试。
   - c.（D10 e，单独一次提交）`hazardlib/host.py`：新增 `HazardState`（`pos`、`quat`（wxyz）、`lin_vel_mps`、`ang_vel_radps`，构造时转 float64 并校验形状与有限性，与 `Pose` 同风格）；`HazardHost` 增加第 14 个必需方法 `hazard_state(h) -> HazardState`（质心的世界系位姿与速度）；`RobotState` 增加可选字段 `held_object_bodies: Optional[List[str]] = None`；`missing_methods / require_methods`、`__init__` 导出、T01 里断言“13 个必需方法”的测试同步改。**原因**：释放后运行时读不到危险体的位姿与速度，近失、TTC、倾倒、`vₙ`、不稳定归因的距离都无从计算（T06 的 `PrivilegedAvoider` 接口也已假定有 `hazard_state`）。
   - d.（D10 f，单独一次提交）BD 加载器校验 `robot_contact_bodies.include / exclude` 的每一项：必须是 `exact:<名>`、`prefix:<名>`、`substr:<名>`，或（仅 `include`）恰为类别记号 `held_object`；否则报错并指出该项。更新 `bd/calvin.yaml` 草稿与测试（缺前缀被拒、`strict` 行为不变、哈希随条目变化），并把 `tests/conftest.py` 里 BD 夹具的裸写条目改成带前缀的写法。
   - e.（D11 a，单独一次提交）`hazardlib/host.py` 新增协议 `EpisodeDriver`：`reset_episode(env, task_id, layout_id, seed)`、`step(env, action)`、`policy_observation(env) -> Dict[str, Any]`，以及常量 `DRIVER_METHODS`。它与 `HazardHost` 分开声明，`REQUIRED_METHODS` 保持 14 个；`require_methods(host, DRIVER_METHODS)` 可单独检查。同步 `__init__` 导出并补测试。**原因**：C2、C7–C10、C13、G1、T_stop 都要驱动回合、比较策略观测，现有契约里没有对应方法。
   - f.（D11 b，单独一次提交）`hazardlib/spec/instance.py`：`HazardInstance` 增加必需字段 `spec_id: str`、`family: str`、`agent: AgentSpec`（放在 `instance_id` 之后，非空校验），三者进入实例哈希。同步更新 T01 里构造实例、Manifest 与哈希的测试；金标哈希若变化，在汇报中列出旧值与新值。**原因**：运行时要靠 `family` 决定是否回滚，靠 `agent` 调 `alloc_hazard` 并取最小半尺寸与质量；不得再从 `release_time_s == onset_time_s` 推断族。
   - g.（D11 c，单独一次提交）`hazardlib/semantics/event_log.py` 新增可选字段：`bd_hash`、`adapter_version`、`manifest_hash`（非空字符串或 `None`），`benchmark_commits`（`Dict[str, str]` 或 `None`），`task_success`（bool 或 `None`），`max_tilt_deg`（有限且 ≥ 0，或 `None`），`r1_violation_count`（int ≥ 0 或 `None`）。`schema_version` 保持 `"1"`；往返、校验与“缺失为 `None` 不为 0”的测试照 T03 的写法补齐。`hazardlib/spec/manifest.py`：`Manifest` 增加 `config_hashes: Dict[str, str]`（配置文件名 → 哈希；非空），由 T06 的冻结步骤填写。
1. `hazardlib/runtime.py` — `HazardRuntime(host, cfg, provenance=None)`：
   - `arm(env, instance, task_entities=None, control_id=None)`、`begin_episode(env)`、`pre_step(env)`、`post_step(env)`、`end_episode(env) -> HazardEventLog`（D11 a）。`begin_episode` 由宿主的重置钩子触发，使用已登记的实例；未 `arm` 就开始回合时抛错。`instance=None` 是无危险对照回合：必须给 `control_id`（用作 `instance_id`），危险相关字段记 `None`，`task_success`、`toppled`、`max_tilt_deg` 照常记录（D10 h 的配对相减要用）。`task_entities` 是要监测倾倒的任务实体名单（来自 TaskContext，D10 h），为 `None` 时 `toppled` 与 `max_tilt_deg` 记 `None`，不猜。支持多个环境下标，彼此独立；
   - `provenance` 提供 `bd_hash`、`adapter_version`、`benchmark_commits`、`manifest_hash`，原样写进每条日志；开发期没有 Manifest 时这些字段为 `None`。`task_success` 在 `end_episode` 时读 `host.task_success(env)`（D11 c）；
   - 按**秒**推进关键帧；**R1 只复核、不插步**（D11 d）：逐钩子用 `hazard_state` 算危险体位移，超过 `max_step_ratio × agent 最小半尺寸` 就累加 `r1_violation_count`。BD `hazard_ccd: true` 时照常计数，但不视为错误；
   - 运动学阶段检测机器人侧重叠（经 `host.hazard_contacts`）：首次重叠 → 记 `kinematic_overlap`，**回滚**到最后一个无重叠位姿，以 R7 的速度（**实例路径在固定窗口 `release_velocity_window_s` 上的后向差分，结束于回滚时刻，不是引擎钩子步长**；未取下限，D4）调用 `release_to_physics`，之后的接触才是官方接触（R8）；
   - 按 `instance.family` 分派（D11 b）：`intrusion.slide` / `intrusion.reach` 走上面的回滚释放；`intrusion.fall`（释放速度 0，D2）在 onset 时摆放并释放，**之后不做运动学重叠检测与回滚**，onset 之后与机器人侧的接触都是官方接触，与 `scene` 的接触只记事件（R8、D10 a）；其他族抛 `CapabilityError`。不得用 `release_time_s == onset_time_s` 判断族；
   - 停放 / 取出（R6）：回合开始前保证危险体处于停放状态，回合结束后恢复停放；
   - 事件记录：用 T03 的语义计算 `v_rel_normal_mps`、`impulse_analytic_ns`、`contact_duration_s`、`constraint_state`、`min_separation_m`、`ttc_min_s`。要点（D10）：
     - **官方接触 = 释放后与机器人侧伙伴的接触**（a）；与 `scene` 的接触记事件，并写 `first_scene_contact_t_s`；`first_official_contact_t_s` 等量只看机器人侧；
     - **`vₙ` 取碰撞前**（c）：每个钩子步保存一份 `hazard_state` 与 `robot_state` 的快照，首次机器人侧接触用**上一步**快照里的危险体质心速度与被接触碰撞体速度（按 `collision_body_names` 匹配接触伙伴名）；取不到被接触碰撞体的速度时 `v_rel_normal_mps` 与 `impulse_analytic_ns` 记 `None`，不记 0，不用碰撞后读数；
     - **逐钩子步**把接触列表（含空列表）交给 T03 的 `summarize_contacts`（只有记录下来的空步才能证明接触中断）；
     - 抓持物分类：`classify_partner` 的 `held_object_names` 传 `RobotState.held_object_bodies`，没有时传 `[held_object]`；
     - `toppled` 与 `max_tilt_deg`：`task_entities` 中每个实体在 onset 时刻记参考姿态（对照回合用同一名义 onset 时刻），`max_tilt_deg` 是回合内所有实体上轴倾角的最大值，超过阈值时 `toppled` 为真；日志记原始值，与无危险运行相减留给分析；
     - 运行时保留事件列表，提供 `runtime.contact_events(env)`（含 `scene` 与 `kinematic_overlap`），供一致性测试（如落地时间）使用；
   - 不稳定监测接口：宿主可调用 `runtime.report_instability(env, kind)`，签名 `report_instability(env, kind, entity_is_hazard=False, entity_pos=None)`；运行时调用 T03 的 `classify_invalid`（D9 d、D10 b）得到 `sim_unstable` 或 `hazard_induced_unstable`，**不得在运行时里自己再写一套判据**；窗口取 `[onset_time_s, nominal_hit_time_s + stability.attribution_window_s]`（只由实例决定），距离由 `entity_pos` 与 `hazard_state` 的质心位置算出，`entity_pos` 为 `None` 时按 T03 规则保留并带 `attribution_ambiguous`；阈值取自 `semantics.yaml`；
   - 所有阈值与模式从 `cfg`（来自 `configs/hazard/*.yaml`）读取。
   - 新建 `configs/hazard/runtime.yaml`：`max_step_ratio: 0.5`（R1 的已定值，D3）、`release_velocity_window_s`、`keyframe_dt_s`、`keyframe_interp_tol_m`（后三项写 `PLACEHOLDER`，试点后预注册；测试里用显式传入的字面量，不读 PLACEHOLDER）。
2. `hazardlib/conformance/fakehost.py` — `FakeHost`：纯 numpy 的小世界：
   - 一个按给定轨迹运动的“机器人 EEF”（带一组碰撞球），若干静态场景体（含桌面平面），可建 / 停放 / 摆放 / 释放的危险体（重力、简单球 / 盒碰撞，足以产生相对法向速度）；
   - 实现 `HazardHost` 全部必需与可选方法（可选方法可由构造参数关闭，以便测试降级路径），并实现 `EpisodeDriver`（D11 a）：`reset_episode` 按 `(task_id, layout_id, seed)` 确定机器人轨迹与初始场景，`step` 推进一个控制步并在内部逐物理步触发钩子，`policy_observation` 返回一张小的合成图像和一个不含危险体的状态向量（R5）；
   - 时基可配置（物理步长、控制步长、每动作子步数），用于测试 R1 与时钟换算；内部碰撞默认做连续检测（BD 声明 `hazard_ccd: true`），也可以关掉；
   - **可注入缺陷的变异开关**（供 T05 使用）：`tunneling`（关闭内部连续检测，但 BD 仍声明 `hazard_ccd: true`）、`residual_velocity`（`set_hazard_pose` 后残留速度）、`state_leak`（危险体进入状态向量）、`wrong_dt`（声明的时基与实际不符）、`no_rollback`、`park_interacts`、`oracle_depends_on_hazard`、`bad_standoff`（实例构造忽略接触余量，首次接触早于名义 `t_hit`）、`hook_dependent_release`（释放速度取引擎钩子步长上的差分而非固定窗口）、`post_impact_vn`（`vₙ` 用碰撞后的速度读数，D10 c）。`FakeHost` 实现 `hazard_state`，运动学阶段读回刚设置的位姿且速度为 0。
3. 单元测试覆盖上述全部。

## 验收标准（贴输出）

1. `pytest hazardlib/tests/runtime -q` 在 `hz38` 与 `hz311` 全绿。
2. 场景测试（在 `FakeHost` 上）：
   - **名义必碰**：用 T02 构造的实例（调用方已按 `standoff = r_hazard + eef_contact_radius_m` 把接触点换算成质心命中点，D1），在“机器人不做任何反应”下，最终得到一个官方接触，且 `|first_official_contact_t − nominal_hit_time| < 容差`（这是 C13 的 FakeHost 版本，容差来自 YAML）；**忽略 standoff 时必须失败**（变异开关 `bad_standoff`）；
   - **机器人反应**：机器人在 τ 之内停下 / 让开时无官方接触；
   - **不穿透与 R1 复核（D11 d）**：高速危险体（单钩子位移超过半尺寸）在 FakeHost 内部连续检测下仍产生接触，且 `r1_violation_count > 0`；变异开关 `tunneling` 下漏检，而 `r1_violation_count > 0` 能把它暴露出来；低速时 `r1_violation_count == 0`。运行时本身不插步；
   - **回滚释放**：运动学重叠被记为 `kinematic_overlap` 而非官方接触；释放速度等于实例路径在固定窗口 `release_velocity_window_s` 上的后向差分（未取下限），并且**与 FakeHost 的钩子步长无关**：同一实例在三种钩子步长下释放速度一致（容差来自 YAML）；变异开关 `hook_dependent_release` 必须使它失败；
   - **多环境独立**：环境 0 的危险体不影响环境 1 的任何读数；
   - **停放**：回合前后危险体均处于停放状态，期间无接触、无观测变化；
   - **时钟**：控制步 0.2 s、物理步 0.002 s 的配置下，事件时间以秒记录且与解析值一致；
   - **fall 族**：onset 零速释放；无 `kinematic_overlap`；落地时间与 FakeHost 的实测重力（`gravity_mps2`）一致；让 FakeHost 取 9.8 与 9.81 两种重力，证明求解器用的是实测值而不是默认值。
   - **碰撞前速度（D10 c）**：危险体以已知速度撞静止的机器人碰撞体，`v_rel_normal_mps` 等于已知速度；变异开关 `post_impact_vn` 必须使它失败；取不到碰撞体速度时为 `None`；
   - **先撞场景再撞机器人**：`fall` 族危险体先落到桌面反弹再撞机器人，`first_scene_contact_t_s < first_official_contact_t_s`，且 `first_official_contact_t_s` 是机器人侧那次；只与桌面接触时 `first_official_contact_t_s` 为 `None`；
   - **不稳定归因（D10 b）**：窗口内且距离内 → `hazard_induced_unstable`；窗口之后很久的不稳定 → `sim_unstable`；危险体自身不稳定 → `hazard_induced_unstable`；`entity_pos=None` → 保留并带 `attribution_ambiguous`；
   - **倾倒（D10 h）**：`task_entities` 中的物体被撞倒 → `toppled` 为真；纯偏航不算；`task_entities=None` → `toppled` 为 `None`；
   - **步骤 0 的契约改动**：`hazard_state` 缺失时 `require_methods` 报错；`robot_contact_bodies` 无前缀的条目被 BD 加载器拒绝；缺 `EpisodeDriver` 方法时 `require_methods(host, DRIVER_METHODS)` 报错；`HazardInstance` 缺 `family` / `agent` / `spec_id` 时报错，且三者改变实例哈希；
   - **回合驱动（D11 a）**：经 `arm` → `reset_episode` → 若干次 `step` → `end_episode` 跑完一回合，得到的日志与直接调用钩子的结果一致；未 `arm` 就重置会抛错；对照回合（`instance=None`）返回的日志中危险相关字段全为 `None`，`task_success` 有值，`instance_id` 等于 `control_id`；
   - **族分派（D11 b）**：`family` 决定是否回滚；未知族抛 `CapabilityError`；构造一个 `release_time_s == onset_time_s` 的 `reach` 实例，仍按 `reach` 回滚处理；
   - **来源字段（D11 c）**：日志带 `provenance` 中的四个字段，以及 `task_success`、`max_tilt_deg`；不给 `provenance` 时这四个字段为 `None`。
3. 事件记录字段完整，缺失量为 `None` 而非 0。
4. 运行时不 import 任何仿真库（`grep` 检查）。

## 不要做
不要实现一致性测试套件本身（T05）、探针 / 实例化 / 门控（T06）；不要为通过测试而在运行时里写阈值默认值。

## 汇报
`docs/design/reports/T04.md`。列出运行时对宿主提出的**隐含要求**（设计文档没写明、但实现时发现需要的），这会直接影响各基准适配器。
