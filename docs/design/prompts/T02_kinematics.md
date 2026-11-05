# T02 — 运动学与校准数学

起点分支：`hazard/integration`（T01 已合并）　依赖：T01　需要仿真器：否　先读：T00。可与 T03 并行。

## 目标
在 `hazardlib/src/hazardlib/kinematics/` 与 `hazardlib/src/hazardlib/calibrate/` 实现**纯数学、秒为单位**的运动原语与校准计算。全部函数显式传参，无阈值默认值（阈值从调用者传入，测试里用字面量）。

## 范围与产出

1. **运动原语**（`kinematics/primitives.py`），输入输出均为 numpy 数组、单位 SI：
   - `min_jerk(tau01)` 及其一阶导数（标量 / 数组）；
   - `fall_height_for_tau(tau_s, g)`：自由落体，使从静止释放到落至参考平面的时间为 τ；以及 `fall_time_for_height(h, g)`；
   - `straight_path(p_start, p_end, t_start, t_end)`、`arc_path(p_start, p_end, t_start, t_end, arc_height)`（抛物线抬升）、`reach_path(p_origin, p_end, t_start, t_end, speed_profile)`（肢体伸入，min-jerk）；返回统一的 `Path`（时间戳 + 位置 [+ 姿态]），支持 `pos(t)`、`vel(t)`；
   - `Path.finite_difference_velocity(t, dt)`：用于 R7 的“最后一步有限差分速度”，**不得加下限**。
2. **名义必碰构造**（`kinematics/hit.py`）：
   - `build_hit_path(family, nominal_ee_path, t_hit, hit_point, tau_s, params)`：给定名义 EEF 路径上的命中点与命中时刻，反解危险体路径，使“不反应的名义机器人”在 `t_hit` 与危险体相撞，且 `onset = t_hit − τ`；对 `intrusion.fall / slide / reach` 三族各一个实现；
   - 返回路径及 `onset_time_s`、`release_time_s`、`release_velocity`。
3. **子步细分**（`kinematics/subdivide.py`）：`subdivide(path, max_step_m, hook_dt_s)`：保证每个钩子间隔内位移 ≤ `max_step_m`（对应 R1：≤ 0.5 × 最小半尺寸），必要时插入子步；并给出 `needs_substeps(path, max_step_m, hook_dt_s)`。
4. **校准**（`calibrate/stop.py`）：
   - `estimate_t_stop(times_s, speeds_mps, v_threshold)`：保持动作后速度降到阈值以下的首个时刻（相对保持起点）；对缺失 / 噪声鲁棒（至少连续 k 个样本低于阈值，k 为参数）；
   - `aggregate_t_stop(samples)`：中位数与四分位距；
   - `margin_sigma(tau_s, t_stop_s)`；
   - `staircase_update(...)`：τ₅₀ 自适应阶梯法的一步更新（简单上下阶梯；仅提供函数与测试，不做实验）。

## 验收标准（贴输出）

1. `pytest hazardlib/tests/kinematics hazardlib/tests/calibrate -q` 在 `hz38` 与 `hz311` 都全绿。
2. 数值测试至少包括：
   - 自由落体：`fall_time_for_height(fall_height_for_tau(τ, g), g) == τ`（相对误差 < 1e-9）；
   - min-jerk：端点位置与速度为 0、中点位置为 0.5；
   - 名义必碰：对 `fall / slide / reach` 各取随机命中点与 τ，用数值积分验证在 `t_hit` 危险体质心与命中点距离 < 1e-6 m，`onset = t_hit − τ`；
   - 子步细分：对高速路径，细分后每个钩子间隔位移 ≤ `max_step_m`；
   - `estimate_t_stop`：构造指数衰减速度序列，估计值与解析值一致；含噪声序列在连续 k 样本规则下不被单点毛刺误判。
3. 所有函数在 numpy 1.x 与 2.x 下行为一致。
4. 没有阈值数值默认值（`grep` 检查函数签名里没有数值默认）。

## 不要做
不要实现接触语义、运行时、探针；不要依赖 scipy。

## 汇报
`docs/design/reports/T02.md`。列出各族“名义必碰”的参数化假设（例如落点相对命中点的偏移约定），便于 T06 对接。
