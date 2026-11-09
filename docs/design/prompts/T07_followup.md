# T07 — 核心收尾：机器人几何 standoff、碰撞体路径、宿主注册回退（D15 a、b、e）

起点分支：从 `hazard/integration` 新建 `hazard/T07`　依赖：T05-F、T06-F、`fix-hook-times` 已合并　需要仿真器：否　先读：T00；`00_paradigm.md` §5.2 与 §13 的 **D15（本任务的依据：a、b、e）**；`01_integration_rules.md` §2（BD 的 `robot_point_spacing_m`）、§3.1、§3.2（`robot_collision_points`）、§3.4、§5 的 C13；H1 汇报的 §1 “C13 失败分析”、§3.4 与“对契约的反馈”第 3–5 条。H1 已于 2026-11-09 合并进 `hazard/integration`（`ca2c8fa`），汇报在 `docs/design/reports/H-calvin.md`。

本任务**允许**改动 T01 的 `host.py`、`spec/benchmark_descriptor.py`，T05 的 `conformance/`（含 FakeHost），T06 的 `probe/`、`instantiate/`、`gates/g0.py`、`driving.py`（D15 批准的跨任务修改），只做下面列出的改动。

## 目标

把 D15 a、b、e 落到代码里，并在 FakeHost 上复现、再修好 H1 的 C13 fall 失败（危险体从上方先碰到腕部）。每项一个或几个小提交；既有测试保持绿，不放宽任何断言。

## 范围

1. **契约（D15 a）**
   - `RobotState` 增可选 `collision_body_quat`：N × 4，wxyz，与 `collision_body_pos` 一一对应；校验形状、有限、范数非零。
   - `OptionalHazardHost` 与 `OPTIONAL_METHODS` 增 `robot_collision_points(env) -> Dict[str, np.ndarray]`：机器人自身每个碰撞体（不含抓持物）在其刚体系下的表面采样点（M × 3），回合内不变，键与 `collision_body_names` 一致。宿主提供它而 `RobotState` 没给 `collision_body_quat` 时，核心报错，不静默退回。
   - BD 增必填字段 `robot_point_spacing_m`（可为 `unknown`；数值时 > 0），进 BD 哈希，`tier()` 不变。`configs/hazard/bd/calvin.yaml`（H1 补全的版本）补 `robot_point_spacing_m: unknown`，加在文件末尾；测试夹具与 FakeHost 的 BD 同步补上。`test_benchmark_descriptor.py` 只增加用例。

2. **探针 trace 与 TaskContext（D15 a、b）**
   - 录制（`driving.py` 的 trace 步）：
     - `robot_bodies` 只放机器人自身刚体：按 BD 规则归为 `robot_link` 的名字，不含抓持物及其 `held_object_bodies`；
     - 宿主给了 `collision_body_quat` 时，新增 `robot_body_quats`（名 → wxyz）；
     - 抓持物刚体另记为 `held_object_bodies_pos`（名 → 位置，没抓东西时为 null）；
     - trace 头新增 `robot_geometry`：宿主提供 `robot_collision_points` 时记录采样点与 BD 的 `robot_point_spacing_m`，每条 trace 只录一次。
     - 格式变了，按既有约定提升 `PROBE_TRACE_SCHEMA_VERSION`。
   - `build_task_context`：
     - 刚体路径只用机器人自身刚体。运行之间名单不一致时，只保留每条运行每一步都出现的刚体，`provenance.robot_body_paths_dropped` 列出丢掉的名字，不再整组丢弃（`probe/context.py:189-199`）。
     - 有朝向时同样求 `robot_body_quat_paths`，跨运行取中位朝向，做法与 `_median_quats` 一致。
     - 各运行的几何必须一致，否则报错；`provenance.robot_geometry` 存一份。
     - 抓持物另存 `held_object_path`：时间与位置，只含被抓持的步。
     - `occupancy` 的约定不变；它是否用到抓持物，在汇报里说明。

3. **几何 standoff（D15 a）**
   - **新纯函数**：放在 `kinematics/` 或 `instantiate/`，不 import 仿真库。
     - 输入：`agent`（球，或单位姿态的盒，与 `support_radius_m` 支持的形状一致，其他形状拒绝）、`p_ref`、单位法向 `n`、世界系采样点（带刚体名）。
     - 输出：standoff `s*` 与首个接触刚体。危险体中心在 `p_ref + s·n`，`s` 从 +∞ 减小时首次碰到任一点的 `s` 就是 `s*`。
     - 球：对每个点 `q`，若到轴线的距离 `d⊥ ≤ r`，`s = (q − p_ref)·n + sqrt(r² − d⊥²)`，取最大。盒：每个点给出一个可行区间，取上端的最大值。
     - 所有点都碰不到时返回“无接触”。
   - **刚体位姿插值到 `t_hit`**：位置线性；朝向用与 `_median_quats` 一致的归一化插值（或 slerp），在文档字符串里写清楚。
   - **三处构造器**在有几何时都用它：
     - `instantiate/builder.py`，几何来自 TaskContext；
     - `conformance/harness.build_nominal_hit_instance`，C13 用第一次重放的 trace 里的刚体位姿和宿主几何；宿主不提供几何时沿用 BD 的 `eef_contact_radius_m`；
     - `FakeHost.nominal_hit_instance`。
     - `provenance` 记 `standoff_source`（`robot_geometry` / `eef_contact_radius`）、`standoff_body`、`robot_point_spacing_m`。
     - 新拒绝原因 `no_robot_contact_along_normal` 加进 `REJECT_REASONS`，流水线的拒绝统计照常计数。
   - **slide 的法向倾斜**：现在用二分解 `standoff(φ)·sin φ = dz`，默认 `standoff(φ)` 单调。几何 standoff 不保证单调：先检查括号条件；找不到解就拒绝（沿用已有原因或新增一个，选法写进汇报），不要静默取近似。
   - **G0 `early_contact`**：
     - 有几何时，在 probe 时刻插值刚体位姿，判定危险体形状对采样点的穿入深度是否超过 `g0.early_contact_tol_m`。这与现在的“距离 < standoff − tol”是同一语义。
     - `checks.early_contact_check` 记 `robot_geometry`；`held_object_path` 当作点参与判定。
     - 没有几何时，行为与现在逐位相同。

4. **FakeHost 与一致性套件**
   - FakeHost 实现 `robot_collision_points` 与 `collision_body_quat`：在 `robot_bodies` 的球面上按给定间距采样，坐标在刚体系。BD 给 `robot_point_spacing_m`。几何可经配置关闭（例如 `world_overrides`），以覆盖“无几何”分支。
   - 加一个测试用的 FakeHost 世界，在 EEF 上方放一个“腕部”球：**标量 standoff 下 C13 fall 失败，几何 standoff 下通过**，复现 H1 的失败模式。默认世界仍是 13 项 pass。
   - 新变异体 `bad_geometry`（例如采样点整体偏移一个可测的距离）必须被 C13 抓到；更新 `MUTATIONS`、文档与变异体测试。
   - C13 的说明文字与失败提示改为“重新测定机器人几何或 `eef_contact_radius_m`”。`conformance.yaml` 不加键、不改值。

5. **宿主注册回退（D15 e）**：`conformance/target.py` 的 `resolve_target` 按顺序尝试：
   1. import `modules`；
   2. 名字仍未注册，import `hazardhosts.<name>`；
   3. 仍未注册，再 import `hazardhosts.<name>.target`；
   4. 都不行时，错误信息列出两次 import 的结果。

   测试在 `tmp_path` 里建假包（临时插入 `sys.path`），覆盖注册在 `__init__`、注册在 `target`、两处都没有三种情况。

6. **测试**（只增加用例）
   - 几何函数：
     - 球 / 盒对单点、对平面点阵，与解析解比较；
     - 点从侧面错过时返回“无接触”；
     - 法向不是单位向量时报错；
     - 点云取半径 `r_e` 的球面时，结果与标量公式 `r_hazard + r_e` 之差在采样误差以内（球对球为 `r − sqrt(r² − (g/2)²)` 量级，`g` 为间距；按你的推导写清楚）。
   - trace / TaskContext：
     - 抓持物进出时刚体路径仍在（复现 H1 的 28/54 退化，用 FakeHost 或手写 trace）；
     - 丢弃名单；
     - 几何不一致时报错；
     - schema 版本。
   - 构造器：
     - 三处在同一输入上给出相同的 standoff；
     - `standoff_source` 正确；
     - 无几何时与旧行为逐位相同。
   - G0：
     - 几何下“手腕先被碰”的实例被拒；
     - 无几何时结果不变。
   - 端到端：`record → probe → instantiate → gate → calibrate → freeze → verify` 在开启几何的 FakeHost 上至少走一条。

7. **汇报**：`docs/design/reports/T07.md`，包括：
   - 每项的验证命令与真实输出；
   - 接口新旧对照（`RobotState`、可选原语、BD、trace、TaskContext 的 `provenance` 键）；
   - “腕部”世界里 C13 fall 的残差，标量与几何各一次；
   - 采样间距对 standoff 误差的实测；
   - 偏离与理由；
   - 给 H1-F 与 H2 的接入说明，写成建议、不实现：MuJoCo 可从 `model.geom_*` 与 `model.mesh_vert` 取碰撞几何；PyBullet 可用 `getCollisionShapeData` 加网格文件，或在引擎里做射线采样。

## 验收标准（贴输出）

1. `python -m pytest hazardlib -q` 在 `hz38`、`hz311` 全绿（`PYTHONPATH=<worktree>/hazardlib/src`；贴总数，对比集成分支 `ca2c8fa` 的基线：`hz38` 1388 passed、1 skipped，`hz311` 1389 passed）。
2. `python -m hazardlib.conformance --host fake` 仍是 13 项 pass；“腕部”世界里，标量 standoff 的 C13 fall 失败、几何 standoff 通过（贴两次的 C13 行）。
3. 全部变异体被抓到，含 `bad_geometry`。
4. 无数值默认值、无仿真库 import、Python 3.8 兼容（既有卫生测试通过）。
5. `git status` 干净；只改了上面列出的范围（贴 `git diff --stat origin/hazard/integration`）。

## 不要做

- 不改 `docs/design/00–03`、`src/dynisafety/`；
- 不改 `hazardlib/src/hazardhosts/calvin/`（CALVIN 宿主接入几何是 H1-F 的事）。例外：本任务的接口改动若让 CALVIN 宿主的 import 或测试失败，只做保持原行为的最小兼容修改，并在汇报里逐处列出；
- 不实现真实宿主；
- 不给 PLACEHOLDER 赋值；不改 `conformance.yaml` 的键与值；
- 不为通过测试放宽阈值。
