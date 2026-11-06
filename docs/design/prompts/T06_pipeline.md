# T06 — 流水线：探针 / 实例化 / 门控 / 校准 / 冻结

起点分支：`hazard/integration`（T04 已合并）　依赖：T04　需要仿真器：否（在 `FakeHost` 上验证）　先读：T00，`00_paradigm.md` §3、§5、§7、§8。可与 T05 并行。

## 目标
实现 `hazardctl` 流水线的**引擎无关**部分，并在 `FakeHost` 上端到端验证。真实基准的接入在 H 类任务里。

## 范围与产出

1. **探针**（`hazardlib/probe/`）：
   - 输入：一组探针运行的 trace（格式沿用仓库诊断 trace 约定：逐步 `eef_pos`、`gripper`、`object_poses`、`in_contact`、`phase`、`t_s`）与成功事件；
   - 输出 `TaskContext`：阶段时间线（由夹爪闭合 / 张开事件与成功时刻切分；阈值走 YAML）、任务相关实体（**顺序**：静态声明 → 被抓取或被移动的物体 → 注解；每个实体记录来源）、名义 EEF 路径（跨探针运行按阶段归一化时间取逐点中位）、占用分布、工作区边界；
   - 探针集描述（来源类别、运行数、参考策略名）进入 `provenance` 并参与哈希；**拒绝**任何标记为“被测策略”的来源（R4）。
2. **实例化**（`hazardlib/instantiate/`）：`instantiate(spec, task_context, seed, bd) -> HazardInstance`：按 `00_paradigm.md` §5.2 选命中点与命中时刻，调用 T02 的 `build_hit_path`；是 `(spec, TC, 任务编号, 布局, 实例种子)` 的**纯函数**（用独立的随机数流，不共享全局随机状态）。
   - **换算与记录**：调用 `build_hit_path` 之前，先按 `standoff = r_hazard(n̂) + eef_contact_radius_m` 把接触点换算成质心命中点，并把 `standoff_m` 写入 `provenance`（D1）；释放窗口参数传 `runtime.yaml` 的 `release_velocity_window_s`，**不是**引擎钩子步长（D4）；
   - **关键帧**：把 T02 的解析 `Path` 采样成 `HazardInstance.keyframes`，间隔 `keyframe_dt_s`（YAML），并**测试**线性插值回放与解析路径的位置误差 ≤ `keyframe_interp_tol_m`（YAML）；危险体姿态取 `HazardSpec.agent` 指定的固定姿态（默认单位四元数），写入 `provenance`；
   - **拒绝**：带严重度的 `reach` 实例 `overshoot_m` 必须 > 0，否则拒绝（`overshoot_m = 0` 属于 `occupancy.dwell`，D6）；
3. **门控**（`hazardlib/gates/`）：
   - G0：几何有效（工作区内、与静态场景体不穿透、停放位置合法）；拒绝 `diagnostics["onset_margin_s"] < 0` 的实例；记录 `τ_vis`（宿主提供 `render_check` 或可由相机几何估计时；否则为 `None` 并在证书备注里写明），**只记录、不过滤**（D5）；
   - G1：`run_g1(host, instance, reference_policy)` 编排“Oracle 停等恢复”运行；`ReferencePolicy` 抽象 = `{demo_replay, scripted, external_policy}`；先要求参考策略在**无危险**时成功，否则标 `baseline_unusable`；
   - G2：`PrivilegedAvoider` 抽象接口（`filter_action(action, robot_state, hazard_state)`），无实现时证书标 `uncertified`，**不得假称可解**；
   - 证书标签与 `00_paradigm.md` §8 一致。
4. **校准**（`hazardlib/calibrate/`）：`run_t_stop_protocol(host, ...)` 按 §7 流程在宿主上测 T_stop（名义速度取自 TaskContext 的 EEF 速度中位数）；输出 `calibration.yaml`；T_stop 有删失（`n_censored > 0`，机器人未停下）时标记为**下界**，Σ 的计算必须拒绝或显式带标记（D6）。
5. **冻结**（`hazardlib/manifest/`）：生成 `Manifest`，含各哈希；`verify_manifest` 校验未被篡改。
6. **CLI**：`hazardctl probe|instantiate|gate|calibrate|freeze`，输入输出为文件（YAML / JSONL），便于在不同环境的进程之间传递。

## 验收标准（贴输出）

1. `pytest hazardlib/tests/pipeline -q` 在 `hz38`、`hz311` 全绿。
2. `FakeHost` 上端到端：探针 → TaskContext → 实例化 → 门控 → 校准 → 冻结，一条命令链跑通。
3. **确定性**：相同输入两次运行得到相同的实例与相同的 Manifest 哈希；改任一输入（种子、探针集、BD）哈希变化。
4. **名义必碰**：在 `FakeHost` 上，无反应的机器人必被撞；Oracle 停等运行无接触且成功 → `wait_solvable`。
5. **不可解**：构造 τ 小于 `FakeHost` 的 T_stop 的实例，证书必须为 `unsolvable_by_probes`（或在无 G2 实现时 `uncertified`），且不进入“主分”集合（给出过滤函数及测试）。
6. **暴露率**：被测机器人轨迹远离命中点时，实例被标 `exposure_low`；贴近时不标。
7. **探针拒绝**：把“被测策略”作为探针来源会抛错。
8. 流水线代码不 import 仿真库；Python 3.8 兼容。
9. **standoff**：`FakeHost` 上按含 standoff 的实例，首次官方接触与名义 `t_hit` 之差 < 容差；忽略 standoff 时失败；`standoff_m` 出现在 `provenance`。
10. **关键帧回放**：实例关键帧经线性插值回放，与解析路径的位置误差 ≤ `keyframe_interp_tol_m`；由关键帧得到的释放速度与 T02 的固定窗口后向差分一致。
11. **拒绝**：`onset_margin_s < 0` 的实例、`overshoot_m = 0` 的带严重度 `reach` 实例，都被拒绝并有测试。
12. **T_stop 删失**：构造“未停下”的重复，`aggregate` 的 `n_censored > 0` 时校准输出带“下界”标记，且 Σ 计算拒绝使用未标记的值。

## 不要做
不要实现任何真实基准的宿主；不要写 G2 的具体避障算法；不要做实验。

## 汇报
`docs/design/reports/T06.md`。列出 TaskContext 估计中你做的假设（如名义路径的聚合方式）及其对真实基准可能的失效情形。
