# 通用接入规则（BD、HazardHost 契约、规则 R1–R16、一致性测试）

适用对象：任何要接入动态安全扰动的静态操作基准。前置：读 `00_paradigm.md`。
关键词：**必须**（MUST）、**禁止**（MUST NOT）、**应当**（SHOULD）。

---

## 1. 包与目录

```
hazardlib/                         新增，独立包，numpy + pyyaml，Python >= 3.8
  pyproject.toml
  src/hazardlib/                   核心（不得 import 任何仿真库）
    spec/                          HazardSpec / BenchmarkDescriptor / TaskContext / HazardInstance / Certificate / Manifest
    host.py                        HazardHost 协议、Tier、CapabilityError
    runtime.py                     HazardRuntime（控制反转）
    kinematics/                    秒为单位的运动原语与 τ 反解
    semantics/                     事件、接触图、严重度、倾倒、无效样本分类
    probe/  instantiate/  gates/  calibrate/  manifest/
    conformance/                   一致性测试套件 + FakeHost
  src/hazardhosts/<bench>/         各基准的适配器（只在对应基准环境里 import）
  tests/
configs/hazard/
  bd/<bench>.yaml                  BD
  annotations/<bench>.yaml         逐任务注解（预算内的手工部分，见 R2）
  families.yaml  intensity.yaml  gates.yaml  semantics.yaml   所有数值默认值只放这里
docs/design/reports/<TASK_ID>.md   每个任务的汇报
```

- **R0（环境隔离）。** `hazardlib` 以 `pip install -e hazardlib --no-deps` 装进**每个基准自己的环境**；核心不得依赖 scipy / torch / 任何仿真库；不得 import `dynisafety`。适配器可以 import 基准库；仅在环境满足 Python ≥ 3.10 且已装 `dynisafety` 时，才可复用 `dynisafety.backends._engine.*` 的接触封装。
- **Python 兼容：** 核心代码使用 `typing.List` 等而非 `list[int]` 的运行时求值；禁止 `match`、`dataclass(slots=True)`、`zip(strict=)`、`X | Y` 运行时类型联合；numpy 1.x 与 2.x 都要能跑。
- **数值默认值唯一性（沿用仓库铁律）：** 阈值与默认值只出现在 `configs/hazard/*.yaml`；代码函数一律显式传参。物理常数（重力加速度等）除外。

---

## 2. BenchmarkDescriptor（BD）

YAML，一个基准一份，放 `configs/hazard/bd/<bench>.yaml`。字段（`?` 表示可为 `unknown`，但 `unknown` 必须被一次探针消除后才能进入正式实验）：

```yaml
name: calvin
bd_version: "1.0"
benchmark: {repo: mees/calvin, commits: {calvin: fa03f01, calvin_env: 797142c}, license: MIT}
python: {required: "3.8", env_name: calvin_venv}
engine: {name: pybullet, version: ?}
timebase: {physics_dt_s: 0.004167, control_dt_s: 0.0333, substeps_per_action: 8}
gravity_mps2: ?                    # 宿主自由落体实测值，不得默认 9.81（CALVIN 源码里是 -9.8）；fall 族的 τ 反解用它（D2）
hazard_ccd: ?                      # 宿主是否对危险体启用连续碰撞检测；为 true 时 G0 不按 R1 位移比例拒绝实例（D11 d）
parallel: {num_envs: 1, batched: false}
loop_owner: user_script            # user_script | policy_side | env_internal
hook_point: "wrap env.step; override step to run hazard hooks per physics substep"
spawn_mode: runtime                # runtime | pooled | preload_xml
kinematic_pose_set: true
release_with_velocity: true
contact_tier: A                    # A: 逐接触 + 可得相对法向速度 | B: 仅布尔 / 伙伴 | C: 无
distance: geometric                # geometric | surrogate | none；none 与 surrogate 一样按质心距离计算并记 separation_kind=surrogate（D12 c）
robot_contact_bodies: {include: ["prefix:robot0_", "exact:panda_hand", held_object], exclude: ["substr:camera"]}   # 每项必须带匹配方式前缀；held_object 是类别记号（D10 f）
eef_contact_radius_m: ?            # 机器人有效接触半径，用于把质心命中点换算成首次表面接触（D1）；有机器人几何时 standoff 改用几何（D15 a）；由 C13 检验
robot_point_spacing_m: ?           # robot_collision_points 采样点的最大间距；unknown 表示宿主不提供机器人几何（D15 a）
state_access: {eef: true, gripper: true, base: false, objects: true, fixtures: true, support_surfaces: true}
task_entity_source: oracle_config  # static_labels | named_refs | oracle_config | bddl_goal | probe_inferred | annotation
success_oracle: {kind: state_diff, readonly: true}
snapshot_restore: true
determinism: {same_seed_repro: exact}      # exact | statistical | unknown
invalid_sample_mapping: {setup_invalid: ..., sim_unstable: ...}
observation_isolation: {hazard_visible_in_rgb: true, hazard_in_state_vectors: false}
policy_interface: {kind: calvin_model_step, notes: "..."}
```

**Tier 规则（BD 必须自己声明，运行时由一致性测试核对）：**
- **Tier A**：`contact_tier=A` 且 `kinematic_pose_set` 且 `release_with_velocity` 且 `state_access.eef/objects`。允许完整评测（接触语义 + 严重度）。
- **Tier B**：没有可靠接触信息或不能释放到动力学。只允许运动学危险，只报近失类指标与任务结果，禁止报严重度。
- **Tier C**：无物理。超出范围，**禁止**接入。
- 降级必须在 BD 与每条回合记录里显式写出，**禁止**静默降级（R15）。

---

## 3. HazardHost 契约

适配器实现以下方法；核心的 `HazardRuntime` 通过它们操作引擎。所有量用 SI 单位，世界系，`wxyz` 四元数。`env` 参数是环境下标（单环境基准恒为 0）。

### 3.1 必需（Tier A）

| 方法 | 语义与约束 |
|---|---|
| `describe() -> BenchmarkDescriptor` | 返回 BD；运行时核对时基等声明（见 C1） |
| `alloc_hazard(env, agent) -> HazardHandle` | 创建或从预置池认领一个危险体，形状 / 质量按 `agent`。**必须**在回合开始前可调用 |
| `free_hazard(h)` | 归还 / 销毁，恢复到停放状态 |
| `set_hazard_pose(h, pos, quat)` | 运动学瞬移；调用后该体为运动学（不受力）。**禁止**留下残余速度 |
| `set_hazard_visible(h, visible)` | 停放 / 取出。停放时**必须**不与任何物体交互、不在任何相机视野内 |
| `release_to_physics(h, lin_vel, ang_vel)` | 变为动力学并具有给定速度；速度**必须**是调用者给的值，不得被下限修改 |
| `robot_state(env) -> RobotState` | EEF 位姿与线速度、夹爪开度、底盘位姿（若有）、机器人碰撞体位置列表（及可选的名称、线速度与朝向 `collision_body_quat`，D15 a）、`held_object` 名称与其碰撞体名列表 `held_object_bodies`（若有，D10 e） |
| `hazard_state(h) -> HazardState` | 危险体质心的世界系位姿与线 / 角速度（D10 e）。运行时释放后靠它读取危险体运动：近失、TTC、倾倒、`vₙ`、不稳定归因的距离都依赖它。运动学阶段读回的应是刚刚设置的位姿，速度为 0 |
| `object_poses(env) -> Dict[str, Pose]` | 命名的任务物体与固定装置位姿；支撑面信息放在 `support_surfaces` |
| `hazard_contacts(h) -> List[ContactObs]` | 自上次调用以来的接触观测：伙伴名称与类别（`robot_link` / `held_object` / `scene` / `hazard`）、世界系位置、法向（**方向：由伙伴指向危险体**，即伙伴推危险体的方向，D9）、引擎冲量（若有）、时间戳。相对法向速度由核心用运动学数据补算；引擎若直接给也可附带 |
| `time_state(env) -> TimeState` | `sim_time_s`、`physics_dt_s`、`control_dt_s` |
| `task_success(env) -> bool` | **只读**转发基准自己的成功判定，**禁止**让它受危险体影响（见 R5） |
| `hold_action(env) -> Action` | 基准原生的“保持 / 无动作”动作，用于 T_stop 与 Oracle 门 |
| `install_hooks(runtime)` | 把 `runtime.begin_episode / pre_step / post_step / end_episode` 挂到基准的重置与**最细物理步**上（与 §3.3 的 `HazardRuntime` 方法名一致；T01 汇报修订建议 1） |

### 3.1a 回合驱动（`EpisodeDriver`，D11 a）

与 `HazardHost` 分开声明；凡要跑一致性测试、门控（G1 / G2）或 T_stop 校准的宿主都**必须**实现。

| 方法 | 语义与约束 |
|---|---|
| `reset_episode(env, task_id, layout_id, seed)` | 把基准重置到指定任务、布局（初始状态）与种子；重置过程中触发已安装的重置钩子（即 `runtime.begin_episode(env)`） |
| `step(env, action)` | 用基准原生动作推进**一个控制步**；逐物理步的钩子在其内部触发 |
| `policy_observation(env) -> Dict[str, Any]` | 策略实际看到的观测：图像（名 → `H×W×3 uint8`）与非视觉状态向量（名 → 一维 float 数组）。键名在同一宿主内保持稳定；C2、C7、C9 用它与无危险运行比较 |

回合绑定：调用方先 `runtime.arm(env, instance, task_entities=None, control_id=None, control_onset_s=None)` 登记下一回合的实例（`instance=None` 表示无危险对照回合，此时必须给 `control_id`，用作日志的 `instance_id`；对照回合若给了 `task_entities`，还必须给 `control_onset_s`，即配对实例的名义 onset，作为倾倒的参考时刻，D12 a），再 `reset_episode`。基准原生评测循环（不经过 `reset_episode`）走的也是“先 `arm`、再由重置钩子触发 `begin_episode`”这条路径。

### 3.2 可选

| 方法 | 作用 | 缺失时 |
|---|---|---|
| `min_distance(h) -> float` | 危险体到机器人的**真几何**最小距离 | 记 `separation_kind=surrogate` |
| `snapshot(env)` / `restore(env, snap)` | 精确保存 / 恢复 | 配对对照退化为统计配对（BD `determinism`） |
| `support_surfaces(env) -> List[Surface]` | 台面、柜体、门等支撑面 | 由探针从物体位姿估计，标注来源 |
| `render_check(h) -> bool` | 危险体是否出现在策略相机中（C9 辅助） | 用图像差分 |
| `robot_collision_points(env) -> Dict[str, ndarray]` | 机器人自身每个碰撞体（不含抓持物）的碰撞几何表面采样点（M × 3，该刚体系，回合内不变），点间距不超过 BD 的 `robot_point_spacing_m`；名字与 `collision_body_names` 一致。提供它时 `RobotState.collision_body_quat` 必填。核心用它求 standoff 与 G0 的早接触（D15 a） | standoff 退回 `r_hazard + eef_contact_radius_m`，G0 只用刚体原点 |

### 3.3 HazardRuntime（核心，不由适配器实现）

`arm(env, instance, task_entities=None, control_id=None, control_onset_s=None)`、`begin_episode(env)`、`pre_step(env)`、`post_step(env)`、`end_episode(env) -> HazardEventLog`（D11 a、D12 a）。运行时负责：按秒推进关键帧、R1 复核（D11 d）、碰撞前回滚与释放（R8）、事件记录、不稳定监测。适配器**只**转发钩子。运行时只实现 Tier A（D12 d）。

**宿主必须满足的钩子语义（D12 e）。** 下列要求决定适配器怎么接钩子，违反时运行时多半不报错、而是记出错误的量，由一致性测试去抓：

1. **钩子粒度与时刻**：`pre_step` / `post_step` 包住**每一个最细物理步**；`time_state().sim_time_s` 在 `pre_step` 里是步首 t0，在 `post_step` 里是步末 t0 + `physics_dt_s`；`physics_dt_s` 必须是真实步长（运行时用它做半步取整与 R1 的步间位移）。
2. **重置钩子的位置**：`begin_episode` 在场景重置**完成之后**调用；运行时在其中读 `object_poses` 检查 `task_entities`，记回合零点，分配并停放危险体。
3. **回合零点**：实例里的时间都相对 `begin_episode` 时的 `sim_time_s`；宿主时钟可以跨回合连续，不必在重置时归零。
4. **接触同钟**：`ContactObs.time_s` 与 `sim_time_s` 是同一个绝对时钟，运行时自己减去回合零点。
5. **危险体伙伴名**：接触里上报的危险体伙伴名必须等于 `HazardHandle.key`，否则另一个危险体会被当成 `scene` 或机器人。运行时目前每个环境每回合只布一个实例，这条暂不会触发，一致性测试也不覆盖（D13 j）。
6. **`hazard_contacts(h)` 是“取走”语义**：返回自上次调用以来的全部接触；运行时每个 `post_step` 调一次，停放期间也调，停放期间的接触被丢弃（这类宿主缺陷由 C2 检验）。
7. **运动学重叠同步上报**：`set_hazard_pose` 之后的那个物理步内，引擎要检测与机器人侧的重叠，并在该步的 `hazard_contacts` 里用碰撞体名上报；只有机器人侧的重叠触发回滚，运动学阶段与 `scene` 的穿插不记录。
8. **释放后速度立即可读**：`release_to_physics` 之后紧接着的 `hazard_state` 必须已经反映释放速度；运行时把读回值记进回合信息，供 C5 与下发值比较。
9. **碰撞体名一致**：`RobotState.collision_body_names` 与 `ContactObs.partner_name` 用同一套名字，`collision_body_lin_vel_mps` 与之一一对应；否则 `vₙ` 记 `None`（D10 c，不回退到 EEF 速度）。
10. **多体抓持物给 `held_object_bodies`**：不给时运行时只按 `[held_object]` 匹配，各部件名匹配不上会被归为 `scene`，官方接触因此丢失。
11. **成功判定的读取时刻**：`task_success(env)` 在 `end_episode` 里、危险体停放**之前**读取，返回 bool（numpy bool 亦可）。
12. **不稳定由宿主检测**：宿主在回合进行中调用 `report_instability`（`entity_pos` 为世界系），只保留第一次报告；归因由 T03 的规则函数完成（R11）。
13. **启动条件**：BD 为 Tier A；`robot_contact_bodies`、`hazard_ccd`、`distance` 不得为 `unknown`；`distance: geometric` 必须提供 `min_distance`；运行时在构造时调用一次 `host.install_hooks(runtime)`。
14. **回合结束由调用方负责**：`EpisodeDriver` 没有“结束”方法，我们的驱动或基准原生评测循环的回合结束处必须调用 `runtime.end_episode(env)`。一致性套件自己驱动回合，覆盖不到这条；H 任务在原生评测循环上验收：每个回合恰好调用一次 `end_episode`，日志条数等于回合数（D13 j）。
15. **四元数为 wxyz**：`object_poses` 与 `hazard_state` 的姿态均为 wxyz，倾角按此计算。
16. **一个宿主一个有效运行时（D13 c）**：`HazardRuntime` 构造时调用 `install_hooks`，后构造的运行时接管钩子，旧的随即作废，继续用它只会记出错误的量。驱动、门控（G1 / G2）与 T_stop 协议因此接收会话（持有唯一的运行时），而不是宿主；Oracle / G2 经运行时的特权访问器 `hazard_state(env)` 读危险体真值，策略看不到它（R5）。

### 3.4 探针 trace 约定（D13 k）

探针 trace 是 `TaskContext` 的输入，宿主录制时应满足：
- 报告机器人碰撞体的位置（提供几何时连同朝向）与 `held_object`。核心只对机器人自身刚体求路径，抓持物单列，名单在运行之间不一致时只丢掉不稳定的刚体（D15 b）。没有碰撞体位置时，G0 的 `early_contact` 只检查 EEF，`provenance` 记 `early_contact_check: eef_only`，汇报须说明 `fall` 落向夹爪时腕部 / 前臂可能先被砸到；
- 声明夹爪信号是测量值还是命令值；
- 控制频率低的基准，在物理步分辨率上报夹爪事件（或在阈值处插值），否则探针阶段边界的分辨率只有一个控制步；
- 在 `env_notes/<bench>.md` 或 BD 中给出工作区上界与桌面高度的数值，拿不到的标 [待探针]，用于事先计算 `fall` 族的可行 τ 区间。

---

## 4. 规则 R1–R16

**R1 单位与时钟。** 规格里一律用秒与米，**禁止**出现步数常数。适配器在引擎能提供的**最细**粒度调用钩子。每次钩子调用中危险体位移**必须** ≤ 0.5 × 危险体最小半尺寸。钩子已在最细物理步时，运行时无法在两个物理步之间插步，因此这是**检查项**（D11 d）：G0 预先拒绝违反的实例（运动学阶段按路径检查，动力学阶段按预测最大速度检查），除非 BD 声明 `hazard_ccd: true`；运行时逐钩子用 `hazard_state` 复核，并记 `r1_violation_count`（C3 检验）。

**R2 无逐任务代码。** 适配器里**禁止**出现以任务名为条件的分支。逐任务信息只能写在 `configs/hazard/annotations/<bench>.yaml`（角色、主臂等声明式字段）。每一行注解计入「注解预算」，在汇报中报告；某任务无法用声明式注解接入，则标为 `unsupported` 并写明原因，**不得**为它写特例代码。

**R3 任务相关实体的发现顺序。** 静态声明（YAML 标签、命名引用、BDDL 目标谓词、任务判定器配置）→ 探针推断（被抓取 / 被移动的物体）→ 注解。每个实体**必须**记录来源。

**R4 探针不含被测策略。** 探针集在版本内冻结并写入 Manifest；至少两类来源（演示回放 / 脚本专家 / 参考策略 L0 回放）。对每个被测策略报告暴露率。

**R5 危险体隔离。** 危险体**可以**出现在渲染图像中（这是目的），但**禁止**进入策略可见的**非视觉**状态向量，**禁止**进入基准的成功判定输入。C7 用“危险体存在但不接触任何东西”的对照来验证状态向量与判定结果不变。

**R6 停放与预置。** 运行时生成不可用或不安全时，在场景构建期预置危险体池并停放：位置在工作区与所有相机视野之外，碰撞过滤保证停放体不与任何物体交互，不受重力影响。

**R7 运动学阶段与释放。** `slide` / `reach` 族的脚本运动全程为运动学。**释放速度 = 实例路径在固定窗口 `release_velocity_window_s`（`runtime.yaml`，所有宿主相同）上的后向差分，结束于回滚时刻；禁止使用引擎钩子步长**（D4：否则同一实例在不同引擎上得到不同释放速度）。**不得**加速度下限（官方严重度使用未取下限速度；运动下限只可用于“让物体继续运动”的非官方用途，且必须在记录里标注）。`fall` 族在 onset 以零速度释放，速度为 0（D2）。没有重叠时的**名义释放**发生在“步末最接近 `release_time_s`”那一物理步的 `pre_step`：危险体从 p(T−dt) 以实例的释放速度出发、步末到达释放位姿，避免把危险体运动学地摆在贴合位姿上而产生假重叠（D12 b）。危险体的线阻尼与角阻尼**必须为 0**，由 C5 检验落地时间；BD 没有阻尼字段，宿主无法置 0 时不满足 Tier A（D13 h）。

**R8 官方接触与回滚。** 对 `slide` / `reach` 族：危险体处于运动学状态时的任何与机器人侧碰撞体的重叠，只记 `kinematic_overlap`。运行时检测到首次重叠后：回滚危险体到最后一个无重叠位姿 → 以 R7 的速度释放 → 其后的接触记为 `dynamic` 事件，其中伙伴为机器人侧的才是官方接触（D10 a、D12 f）。没有无重叠位姿（第一次摆放就重叠），或回滚时刻之前容不下差分窗口时，该实例记 `setup_invalid`，危险体重新停放（D12 g）。适配器若无法回滚，则声明 Tier B（运行时目前不支持 Tier B，D12 d）。对 `fall` 族：危险体从 onset 起即为动力学体，**没有运动学重叠阶段，不回滚**，onset 起的接触都按 `dynamic` 事件记录；其中“官方接触”仍特指与机器人侧伙伴的接触，与 `scene`（如桌面）的接触按伙伴类别 `scene` 记事件、不计入官方接触量（D10 a）。

**R9 严重度只用运动学解析量。** `J = m·(1+e)·vₙ`；引擎峰值力 / 冲量只作辅助，**禁止**用于排序或进入主表。

**R10 倾倒。** 上轴倾角，忽略绕 z 的偏航，阈值来自 YAML。

**R11 无效样本三分类。** `setup_invalid` / `sim_unstable` / `hazard_induced_unstable`。前两类映射基准原生机制并从分母排除、分别计数；第三类**必须**保留在分母中并单独汇报。归因规则（窗口 `[onset, 名义命中 + 窗口长度]` 内、距离 ≤ 阈值；危险体自身不稳定一律归第三类；距离不可得时保留并计 `attribution_ambiguous`）见 `00_paradigm.md` D9 d、D10 b，由 T03 的 `classify_invalid` 实现，运行时不得另写判据。

**R12 确定性与配对。** 实例是 (规格, 任务上下文, 基准任务编号, 布局 / 种子, 实例种子) 的纯函数。BD 声明 `same_seed_repro`。`exact` 时用快照做配对对照；否则用同布局、同策略种子的无危险运行做统计配对。

**R13 不改上游。** 通过包装、子类、混入或在我们自己的启动脚本里做猴子补丁来接入。若子类覆盖需要复制上游代码，**必须**固定上游提交号并在 import 时断言；若补丁无法避免，存为 `patches/<bench>/*.patch`（仓库既有约定）并在汇报中列出。

**R14 日志。** 每回合写 `hazard_events.jsonl`，字段见 `00_paradigm.md` §6，另含来源字段 `bd_hash`、`adapter_version`、`benchmark_commits`、`manifest_hash`，以及 `task_success`（D11 c）。Manifest 含所用 `configs/hazard/*.yaml` 的逐文件哈希，以及一致性测试报告哈希 `conformance_report_hash` 与校准哈希 `calibration_hash`（D13 a、b）；冻结时配置必须能 strict 加载。逐步 trace（危险体位置、EEF 位置、接触事件、阶段）按控制频率写入，路径与格式沿用仓库诊断 trace 的约定。

**R15 大声失败。** 缺少声明的能力时，启动阶段就抛 `CapabilityError`；**禁止**静默降级、**禁止**用默认值掩盖缺失的接口。

**R16 视觉有效性。** 危险体在策略相机视野中必须真的可见（C9）。对使用真实背景叠加的基准（如 SimplerEnv 的 `rgb_overlay`），必须验证危险体被正确合成。

---

## 5. 一致性测试套件（验收标准）

放在 `hazardlib/src/hazardlib/conformance/`，用 `python -m hazardlib.conformance --host <bench> --out report.json`（或 `pytest --host=<bench>`）在对应基准环境里运行；`FakeHost`（纯 Python 的运动学小世界）用来测试套件本身与核心逻辑，**无需仿真器**。**禁止**为了通过测试而放宽阈值；阈值写在 `configs/hazard/conformance.yaml`，任何改动都要在汇报中说明理由。

- 每项结果为 `pass` / `fail` / `skip` / `unverified` / `error`：宿主缺能力（`CapabilityError`）记 `fail`；`error` 只留给套件自身崩溃。报告是冻结的前提（D13 a）：`verdict` 须为 `pass`，且须在冻结版 `runtime.yaml` / `semantics.yaml` / `intensity.yaml` 上运行，不得用 `conformance.yaml` 的 `probe_runtime` / `probe_calibration` 回退值。
- **容差收紧程序（D13 i）**：`conformance.yaml` 是所有宿主共用的一份，现值由 FakeHost 推出（多数偏松）。H 任务**不改**它，只在汇报里给出本宿主各项的正常误差与建议值；H1、H2 都汇报后由协调者统一收紧并另记决定，收紧后的值仍须小于对应变异体误差的一半。
- **statistical 宿主（D13 h）**：BD 声明 `same_seed_repro: statistical` 时，C2 / C7 / C12 不要求逐值一致，而是与 C8 实测的同种子差异上限比较；该上限按观测种类（状态向量、图像；图像对每个相机分别比较）分别给出，值保持 PLACEHOLDER，由该宿主的 H 任务实测后定；未定时 C8 记 `unverified`。

| 编号 | 检验 | 通过标准（数值见 YAML） |
|---|---|---|
| C1 | BD 与运行时一致 | 实测 `physics_dt` / `control_dt` / 子步数与声明相差不超过容差；同时检查 §3.3 第 1–3 条（钩子粒度与时刻、重置钩子位置、回合零点） |
| C2 | 停放不可见且无交互 | 停放期间状态向量、成功判定、渲染图像与无危险运行完全一致（statistical 宿主见上） |
| C3 | 位姿保真与不穿透 | 运动学路径误差在容差内（位姿经 `hazard_state` 读回）；穿过机器人碰撞体时必有接触事件，穿越速度：BD 声明 `hazard_ccd: true` 时在半个物理步内穿过重叠窗口，否则取 R1 极限速度的一半（D13 h） |
| C4 | 接触检测与伙伴分类 | 与机器人重叠 → `robot_link`；与桌面重叠 → `scene`，不计机器人接触；抓持物（若可构造）→ `held_object` |
| C5 | 释放与重力 | 释放后经 `hazard_state` 读回的速度与设定值在容差内；先实测 `gravity_mps2`（自由落体拟合），再验证落地时间与 `√(2h/g_测)` 一致；按阻尼为 0 检验（R7），落地时间容差以物理步为单位（D13 h） |
| C6 | 相对法向速度 | 以已知速度撞击静止的机器人碰撞体，`vₙ`（取首次接触**前**最后一个钩子步的质心速度，D10 c）与解析冲量在容差内 |
| C7 | 隔离 | 危险体存在但不触碰任何物体时，状态向量与成功判定与无危险运行一致（statistical 宿主见上） |
| C8 | 确定性等级 | 同种子重复复位，测出差异量级，与 BD 的 `same_seed_repro` 一致；`exact` 要求 `restore(snapshot)` 精确回放；statistical 的量级写进报告，按观测种类与上限比较 |
| C9 | 可见性 | 在视野内时策略相机图像相对无危险有像素差；停放时无差异 |
| C10 | 保持动作与 T_stop | 零动作下机器人不漂移；T_stop 在 N 次重复内中位数稳定、四分位距在容差内 |
| C11 | 无效样本映射 | 强制不可放置 → `setup_invalid` 且被排除；适配器提供 `inject_engine_fault(env, hazard)`，用本宿主的方式强制引擎故障 → `sim_unstable` / `hazard_induced_unstable` 按 R11 归类；不提供即判 `fail`。FakeHost 没有真实引擎故障，只验证映射路径（D13 h） |
| C12 | 多环境独立（`num_envs > 1` 的宿主，D13 h） | 环境 i 的危险体不影响环境 j 的任何观测（statistical 宿主见上） |
| C13 | 名义命中时机（D1） | 令机器人沿一条已知名义路径运动（演示回放 / 脚本，且可重复），危险体按 `build_hit_path` 构造，standoff 按 D15 a 换算（有机器人几何时为沿 n 的首次接触距离，否则 `r_hazard + eef_contact_radius_m`）；**首次官方接触时刻与名义 `t_hit` 之差在 `tol_hit_time_s`（`c13_nominal_hit.hit_time_abs_tol_s`）内**，并报告残差；残差超标则重新测定机器人几何或 `eef_contact_radius_m`，不得放宽容差。容差至少覆盖一个物理步：reach 的释放速度是 R7 的窗口差分而不是路径在 `t_hit` 的瞬时速度，名义释放又提前一步（D12 b），命中时刻误差为 O(dt·\|v(t_hit) − v_rel\| / \|v_rel\|) |

---

## 6. 适配器验收流程（每个基准都一样）

1. **环境就绪**：装好基准与策略依赖，跑通基准自带的最小评测（带版本记录），写 `docs/design/env_notes/<bench>.md`。
2. **BD 草稿**：填写 `configs/hazard/bd/<bench>.yaml`，`unknown` 项列成探针清单。
3. **实现适配器**：`hazardhosts/<bench>/`。
4. **一致性测试**：C1–C13 全部通过（Tier B 允许 C6 跳过并说明）；汇报给出本宿主各项的正常误差与建议容差，不改 `conformance.yaml`（D13 i）。
5. **特权运行**：零动作 / 保持动作、演示回放或脚本专家、Oracle-SWR；记录 T_stop、探针轨迹、参考策略 L0 成功率。正式配置预注册之前，用试点目录 `configs/hazard/pilot/<bench>/` 的试点值运行，结果标“试点”，不得冻结（D14）。
6. **产出**：BD（所有 `unknown` 消除）、适配成本账本、汇报。

适配成本账本（H3 证据，**必须**如实填写）：适配器代码行数、注解行数、上游补丁数、首次运行一致性测试通过数 / 总数、会话时长、遇到的阻塞。
