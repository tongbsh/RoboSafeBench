# T03 — 安全语义

起点分支：`hazard/integration`（T01 已合并）　依赖：T01　需要仿真器：否　先读：T00，以及 `docs/audit/p0_semantics_audit.md` 全文。可与 T02 并行。

## 目标
在 `hazardlib/src/hazardlib/semantics/` 实现 `00_paradigm.md` §6 的统一安全语义，全部是**引擎无关的纯函数与数据类**。这是 C1，论文的第一贡献，质量优先于速度。

## 范围与产出

1. **接触观测与事件**（`events.py`）：
   - `PartnerClass`（`robot_link / held_object / scene / hazard`）、`ContactObs` 与 `ContactEvent`（官方接触 / `kinematic_overlap` 两类）；
   - `classify_partner(...)`：按 BD 的 `robot_contact_bodies` 与当前抓持物名称分类。
2. **严重度**（`severity.py`）：
   - `analytic_impulse_ns(mass_kg, v_normal_mps, restitution)` = `m(1+e)vₙ`；
   - `closing_normal_speed(v_hazard, v_partner, normal_from_hazard)`（分离时为 0）；
   - 参考实现与仓库既有 `src/dynisafety/core/newton_force.py` 的 `l3_closing_speed_mps`、`l3_newton_normal_impulse_ns` 做**等价性测试**（在测试里按文件路径动态加载该文件，避免 import `dynisafety`；该文件仅依赖 numpy 与 `dynisafety.core.contact` 的一个常量——测试中用同值桩替代）。
3. **接触图语义**（`contact_graph.py`）：
   - 输入逐步的“同时接触的伙伴集合与法向”，输出：连续官方接触时长（秒）、约束状态 `{unconstrained, constrained, clamp}`、`t_q` 判定；`t_q` 与角度阈值由参数传入；
   - 对 `clamp`：危险体同时接触一个静态场景体或机器人另一部件，且两个法向夹角大于阈值（相对），持续 ≥ `t_q`。
4. **近失与接触前剩余时间**（`proximity.py`）：从逐步 trace（危险体位置、EEF 位置，可选真几何距离）计算 `min_separation_m`、`separation_kind`、`ttc_series`、`ttc_min_s`；有真几何距离时优先用，并标注种类。
5. **倾倒**（`topple.py`）：`tilt_deg_up_axis(ref_quat, cur_quat)`（忽略绕 z 的偏航）、`is_toppled(tilt, threshold)`（严格大于）。**回归测试**必须复现 `docs/audit/p0_semantics_audit.md` 附录表：纯偏航 40° 与 90° → 0°；绕 x 轴 40° → 40°；绕 x 轴 20° → 20°；并断言这与仓库旧的“相对四元数总角”口径在纯偏航输入上**不同**（把旧公式作为测试内的对照实现）。
6. **无效样本**（`invalid.py`）：`InvalidClass`（`setup_invalid / sim_unstable / hazard_induced_unstable / none`）、`classify_invalid(...)` 的规则函数、分母口径函数 `metric_denominator(records)`：前两类排除、第三类保留，且三类都要计数输出。
7. **事件记录**（`event_log.py`）：`HazardEventLog` 数据类（字段见 `00_paradigm.md` §6）、JSONL 写 / 读、字段校验；缺失字段显式为 `None`，不得填 0。

## 验收标准（贴输出）

1. `pytest hazardlib/tests/semantics -q` 在 `hz38` 与 `hz311` 全绿。
2. 倾倒回归测试复现 P0 附录的四行数值。
3. 等价性测试：对 200 组随机输入，`analytic_impulse_ns` 与仓库旧实现一致（相对误差 < 1e-9）。
4. 接触图：构造“危险体夹在机器人与桌面之间 0.3 s / 0.8 s”两个序列，`t_q=0.5` 时只有后者判 `clamp`；单侧接触判 `unconstrained`。
5. 近失：构造已知最小间距与 TTC 的轨迹，数值与解析值一致；无真几何距离时 `separation_kind == "surrogate"`。
6. 无效样本分母：构造含三类样本的集合，前两类被排除、第三类保留、计数正确。
7. `HazardEventLog` 往返读写无损；缺失字段读回为 `None`。
8. `grep` 确认没有任何阈值数值默认值（阈值由参数传入）。

## 不要做
不要实现运行时、探针、适配器；不要修改 `src/dynisafety/`；不要在代码里写 `t_q`、角度阈值、倾倒阈值的默认值。

## 汇报
`docs/design/reports/T03.md`。列出你认为设计文档 §6 中有歧义或可能不自洽的地方（例如 `clamp` 的法向判据、`hazard_induced_unstable` 的识别条件），作为修订建议。
