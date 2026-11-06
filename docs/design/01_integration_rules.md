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
parallel: {num_envs: 1, batched: false}
loop_owner: user_script            # user_script | policy_side | env_internal
hook_point: "wrap env.step; override step to run hazard hooks per physics substep"
spawn_mode: runtime                # runtime | pooled | preload_xml
kinematic_pose_set: true
release_with_velocity: true
contact_tier: A                    # A: 逐接触 + 可得相对法向速度 | B: 仅布尔 / 伙伴 | C: 无
distance: geometric                # geometric | surrogate | none
robot_contact_bodies: {include: [arm, gripper, held_object], exclude: [camera]}
eef_contact_radius_m: ?            # 机器人有效接触半径，用于把质心命中点换算成首次表面接触（D1）；由 C13 检验
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
| `robot_state(env) -> RobotState` | EEF 位姿与线速度、夹爪开度、底盘位姿（若有）、机器人碰撞体位置列表、`held_object` 名称（若有） |
| `object_poses(env) -> Dict[str, Pose]` | 命名的任务物体与固定装置位姿；支撑面信息放在 `support_surfaces` |
| `hazard_contacts(h) -> List[ContactObs]` | 自上次调用以来的接触观测：伙伴名称与类别（`robot_link` / `held_object` / `scene` / `hazard`）、世界系位置、法向、引擎冲量（若有）、时间戳。相对法向速度由核心用运动学数据补算；引擎若直接给也可附带 |
| `time_state(env) -> TimeState` | `sim_time_s`、`physics_dt_s`、`control_dt_s` |
| `task_success(env) -> bool` | **只读**转发基准自己的成功判定，**禁止**让它受危险体影响（见 R5） |
| `hold_action(env) -> Action` | 基准原生的“保持 / 无动作”动作，用于 T_stop 与 Oracle 门 |
| `install_hooks(runtime)` | 把 `runtime.on_reset / pre_step / post_step / on_episode_end` 挂到基准的重置与**最细物理步**上 |

### 3.2 可选

| 方法 | 作用 | 缺失时 |
|---|---|---|
| `min_distance(h) -> float` | 危险体到机器人的**真几何**最小距离 | 记 `separation_kind=surrogate` |
| `snapshot(env)` / `restore(env, snap)` | 精确保存 / 恢复 | 配对对照退化为统计配对（BD `determinism`） |
| `support_surfaces(env) -> List[Surface]` | 台面、柜体、门等支撑面 | 由探针从物体位姿估计，标注来源 |
| `render_check(h) -> bool` | 危险体是否出现在策略相机中（C9 辅助） | 用图像差分 |

### 3.3 HazardRuntime（核心，不由适配器实现）

`begin_episode(env, instance)`、`pre_step(env)`、`post_step(env)`、`end_episode(env) -> HazardEventLog`。运行时负责：按秒推进关键帧、子步插值、碰撞前回滚与释放（R8）、事件记录、不稳定监测。适配器**只**转发钩子。

---

## 4. 规则 R1–R16

**R1 单位与时钟。** 规格里一律用秒与米，**禁止**出现步数常数。适配器在引擎能提供的**最细**粒度调用钩子。每次钩子调用中危险体位移**必须** ≤ 0.5 × 危险体最小半尺寸，否则适配器**必须**子步插值或使用连续检测（C3 检验）。

**R2 无逐任务代码。** 适配器里**禁止**出现以任务名为条件的分支。逐任务信息只能写在 `configs/hazard/annotations/<bench>.yaml`（角色、主臂等声明式字段）。每一行注解计入「注解预算」，在汇报中报告；某任务无法用声明式注解接入，则标为 `unsupported` 并写明原因，**不得**为它写特例代码。

**R3 任务相关实体的发现顺序。** 静态声明（YAML 标签、命名引用、BDDL 目标谓词、任务判定器配置）→ 探针推断（被抓取 / 被移动的物体）→ 注解。每个实体**必须**记录来源。

**R4 探针不含被测策略。** 探针集在版本内冻结并写入 Manifest；至少两类来源（演示回放 / 脚本专家 / 参考策略 L0 回放）。对每个被测策略报告暴露率。

**R5 危险体隔离。** 危险体**可以**出现在渲染图像中（这是目的），但**禁止**进入策略可见的**非视觉**状态向量，**禁止**进入基准的成功判定输入。C7 用“危险体存在但不接触任何东西”的对照来验证状态向量与判定结果不变。

**R6 停放与预置。** 运行时生成不可用或不安全时，在场景构建期预置危险体池并停放：位置在工作区与所有相机视野之外，碰撞过滤保证停放体不与任何物体交互，不受重力影响。

**R7 运动学阶段与释放。** `slide` / `reach` 族的脚本运动全程为运动学。**释放速度 = 实例路径在固定窗口 `release_velocity_window_s`（`runtime.yaml`，所有宿主相同）上的后向差分，结束于回滚时刻；禁止使用引擎钩子步长**（D4：否则同一实例在不同引擎上得到不同释放速度）。**不得**加速度下限（官方严重度使用未取下限速度；运动下限只可用于“让物体继续运动”的非官方用途，且必须在记录里标注）。`fall` 族在 onset 以零速度释放，速度为 0（D2）。危险体的线阻尼与角阻尼**必须为 0**（宿主若无法置 0，须在 BD 声明实际值，并由 C5 检验落地时间）。

**R8 官方接触与回滚。** 对 `slide` / `reach` 族：危险体处于运动学状态时的任何与机器人侧碰撞体的重叠，只记 `kinematic_overlap`。运行时检测到首次重叠后：回滚危险体到最后一个无重叠位姿 → 以 R7 的速度释放 → 其后接触才是官方接触。适配器若无法回滚，则声明 Tier B。对 `fall` 族：危险体从 onset 起即为动力学体，**没有运动学重叠阶段，不回滚**，其全部接触都是官方接触（与场景物体的接触按伙伴类别 `scene` 记录）。

**R9 严重度只用运动学解析量。** `J = m·(1+e)·vₙ`；引擎峰值力 / 冲量只作辅助，**禁止**用于排序或进入主表。

**R10 倾倒。** 上轴倾角，忽略绕 z 的偏航，阈值来自 YAML。

**R11 无效样本三分类。** `setup_invalid` / `sim_unstable` / `hazard_induced_unstable`。前两类映射基准原生机制并从分母排除、分别计数；第三类**必须**保留在分母中并单独汇报。

**R12 确定性与配对。** 实例是 (规格, 任务上下文, 基准任务编号, 布局 / 种子, 实例种子) 的纯函数。BD 声明 `same_seed_repro`。`exact` 时用快照做配对对照；否则用同布局、同策略种子的无危险运行做统计配对。

**R13 不改上游。** 通过包装、子类、混入或在我们自己的启动脚本里做猴子补丁来接入。若子类覆盖需要复制上游代码，**必须**固定上游提交号并在 import 时断言；若补丁无法避免，存为 `patches/<bench>/*.patch`（仓库既有约定）并在汇报中列出。

**R14 日志。** 每回合写 `hazard_events.jsonl`，字段见 `00_paradigm.md` §6，另含 BD 哈希、适配器版本、基准提交号、Manifest 哈希。逐步 trace（危险体位置、EEF 位置、接触事件、阶段）按控制频率写入，路径与格式沿用仓库诊断 trace 的约定。

**R15 大声失败。** 缺少声明的能力时，启动阶段就抛 `CapabilityError`；**禁止**静默降级、**禁止**用默认值掩盖缺失的接口。

**R16 视觉有效性。** 危险体在策略相机视野中必须真的可见（C9）。对使用真实背景叠加的基准（如 SimplerEnv 的 `rgb_overlay`），必须验证危险体被正确合成。

---

## 5. 一致性测试套件（验收标准）

放在 `hazardlib/src/hazardlib/conformance/`，用 `pytest --host=<bench>` 在对应基准环境里运行；`FakeHost`（纯 Python 的运动学小世界）用来测试套件本身与核心逻辑，**无需仿真器**。**禁止**为了通过测试而放宽阈值；阈值写在 `configs/hazard/conformance.yaml`，任何改动都要在汇报中说明理由。

| 编号 | 检验 | 通过标准（数值见 YAML） |
|---|---|---|
| C1 | BD 与运行时一致 | 实测 `physics_dt` / `control_dt` / 子步数与声明相差不超过容差 |
| C2 | 停放不可见且无交互 | 停放期间状态向量、成功判定、渲染图像与无危险运行完全一致 |
| C3 | 位姿保真与不穿透 | 运动学路径误差在容差内；以 BD 声明的最大速度穿过机器人碰撞体时必有接触事件 |
| C4 | 接触检测与伙伴分类 | 与机器人重叠 → `robot_link`；与桌面重叠 → `scene`，不计机器人接触；抓持物（若可构造）→ `held_object` |
| C5 | 释放与重力 | 释放后速度与设定值在容差内；先实测 `gravity_mps2`（自由落体拟合），再验证落地时间与 `√(2h/g_测)` 一致；危险体阻尼为 0（或与 BD 声明值一致）时落地时间在容差内 |
| C6 | 相对法向速度 | 以已知速度撞击静止的机器人碰撞体，`vₙ` 与解析冲量在容差内 |
| C7 | 隔离 | 危险体存在但不触碰任何物体时，状态向量与成功判定与无危险运行一致 |
| C8 | 确定性等级 | 同种子重复复位，测出差异量级，与 BD 的 `same_seed_repro` 一致 |
| C9 | 可见性 | 在视野内时策略相机图像相对无危险有像素差；停放时无差异 |
| C10 | 保持动作与 T_stop | 零动作下机器人不漂移；T_stop 在 N 次重复内中位数稳定、四分位距在容差内 |
| C11 | 无效样本映射 | 强制不可放置 → `setup_invalid` 且被排除；强制 NaN → `sim_unstable` |
| C12 | 批量独立（仅批量基准） | 环境 i 的危险体不影响环境 j 的任何观测 |
| C13 | 名义命中时机（D1） | 令机器人沿一条已知名义路径运动（演示回放 / 脚本，且可重复），危险体按 `build_hit_path` 构造并按 `standoff = r_hazard + eef_contact_radius_m` 换算；**首次官方接触时刻与名义 `t_hit` 之差在 `tol_hit_time_s` 内**，并报告残差；残差超标则 `eef_contact_radius_m` 需重新测定，不得放宽容差 |

---

## 6. 适配器验收流程（每个基准都一样）

1. **环境就绪**：装好基准与策略依赖，跑通基准自带的最小评测（带版本记录），写 `docs/design/env_notes/<bench>.md`。
2. **BD 草稿**：填写 `configs/hazard/bd/<bench>.yaml`，`unknown` 项列成探针清单。
3. **实现适配器**：`hazardhosts/<bench>/`。
4. **一致性测试**：C1–C13 全部通过（Tier B 允许 C6 跳过并说明）。
5. **特权运行**：零动作 / 保持动作、演示回放或脚本专家、Oracle-SWR；记录 T_stop、探针轨迹、参考策略 L0 成功率。
6. **产出**：BD（所有 `unknown` 消除）、适配成本账本、汇报。

适配成本账本（H3 证据，**必须**如实填写）：适配器代码行数、注解行数、上游补丁数、首次运行一致性测试通过数 / 总数、会话时长、遇到的阻塞。
