# T04 — `HazardRuntime` 与 `FakeHost`

起点分支：`hazard/integration`（T01、T02、T03 已合并）　依赖：T01–T03　需要仿真器：否　先读：T00。

## 目标
实现控制反转的 `HazardRuntime`（`01_integration_rules.md` §3.3）和一个**纯 numpy 的假世界** `FakeHost`，用于在没有任何仿真器的情况下测试运行时、一致性套件与流水线。

## 范围与产出

1. `hazardlib/runtime.py` — `HazardRuntime(host, cfg)`：
   - `begin_episode(env, instance)`、`pre_step(env)`、`post_step(env)`、`end_episode(env) -> HazardEventLog`；支持多个环境下标并彼此独立；
   - 按**秒**推进关键帧；对每个钩子间隔调用 T02 的子步细分，保证 R1；
   - 运动学阶段检测机器人侧重叠（经 `host.hazard_contacts`）：首次重叠 → 记 `kinematic_overlap`，**回滚**到最后一个无重叠位姿，以 R7 的**未取下限**速度调用 `release_to_physics`，之后的接触才是官方接触（R8）；
   - 停放 / 取出（R6）：回合开始前保证危险体处于停放状态，回合结束后恢复停放；
   - 事件记录：用 T03 的语义计算 `v_rel_normal_mps`、`impulse_analytic_ns`、`contact_duration_s`、`constraint_state`、`min_separation_m`、`ttc_min_s`；
   - 不稳定监测接口：宿主可调用 `runtime.report_instability(env, kind)`，运行时记 `sim_unstable` 或 `hazard_induced_unstable`（后者当且仅当不稳定发生在危险体活动窗口内且危险体距机器人小于参数给定距离）；
   - 所有阈值与模式从 `cfg`（来自 `configs/hazard/*.yaml`）读取。
2. `hazardlib/conformance/fakehost.py` — `FakeHost`：纯 numpy 的小世界：
   - 一个按给定轨迹运动的“机器人 EEF”（带一组碰撞球），若干静态场景体（含桌面平面），可建 / 停放 / 摆放 / 释放的危险体（重力、简单球 / 盒碰撞，足以产生相对法向速度）；
   - 实现 `HazardHost` 全部必需与可选方法（可选方法可由构造参数关闭，以便测试降级路径）；
   - 时基可配置（物理步长、控制步长、每动作子步数），用于测试 R1 与时钟换算；
   - **可注入缺陷的变异开关**（供 T05 使用）：`tunneling`（不做子步）、`residual_velocity`（`set_hazard_pose` 后残留速度）、`state_leak`（危险体进入状态向量）、`wrong_dt`（声明的时基与实际不符）、`no_rollback`、`park_interacts`、`oracle_depends_on_hazard`。
3. 单元测试覆盖上述全部。

## 验收标准（贴输出）

1. `pytest hazardlib/tests/runtime -q` 在 `hz38` 与 `hz311` 全绿。
2. 场景测试（在 `FakeHost` 上）：
   - **名义必碰**：用 T02 构造的实例，在“机器人不做任何反应”下，最终得到一个官方接触，且 `|first_official_contact_t − nominal_hit_time| < 容差`；
   - **机器人反应**：机器人在 τ 之内停下 / 让开时无官方接触；
   - **不穿透**：高速危险体（位移超过半尺寸）在子步细分下仍产生接触；关闭子步（变异开关）则漏检——测试两种情形；
   - **回滚释放**：运动学重叠被记为 `kinematic_overlap` 而非官方接触；释放速度与重叠前的有限差分速度一致（未取下限）；
   - **多环境独立**：环境 0 的危险体不影响环境 1 的任何读数；
   - **停放**：回合前后危险体均处于停放状态，期间无接触、无观测变化；
   - **时钟**：控制步 0.2 s、物理步 0.002 s 的配置下，事件时间以秒记录且与解析值一致。
3. 事件记录字段完整，缺失量为 `None` 而非 0。
4. 运行时不 import 任何仿真库（`grep` 检查）。

## 不要做
不要实现一致性测试套件本身（T05）、探针 / 实例化 / 门控（T06）；不要为通过测试而在运行时里写阈值默认值。

## 汇报
`docs/design/reports/T04.md`。列出运行时对宿主提出的**隐含要求**（设计文档没写明、但实现时发现需要的），这会直接影响各基准适配器。
