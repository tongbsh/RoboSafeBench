# T03 — 安全语义

起点分支：`hazard/integration`（T01、T02、T02-F 已合并）　依赖：T01　需要仿真器：否　先读：T00，`00_paradigm.md` §6 与 §13（**D9 是本任务的补充决定**），以及 `docs/audit/p0_semantics_audit.md` 全文。

## 目标
在 `hazardlib/src/hazardlib/semantics/` 实现 `00_paradigm.md` §6 的统一安全语义，全部是**引擎无关的纯函数与数据类**。这是 C1，论文的第一贡献，质量优先于速度。

## 与已合并代码的关系（先读）
- **`PartnerClass` 与 `ContactObs` 已在 T01 的 `hazardlib/host.py` 里定义**（`ContactObs` 含 `partner_name, partner_class, position, normal, time_s, impulse_ns, rel_normal_speed_mps`）。**直接 import，不得重新定义**，也不要改 `host.py`。
- `ContactObs.normal` 的方向是**由伙伴指向危险体**（伙伴推危险体的方向；D9 a）。本任务所有接受法向的函数都用这个约定。
- `configs/hazard/semantics.yaml` 是 T01 留的骨架（全是 `PLACEHOLDER`），**归你所有**，键名可改：按 D9 把 `stability.hazard_induced_unstable_speed_mps` 换成 `stability.hazard_induced_max_distance_m`，新增 `contact.gap_tolerance_s`；叶子一律保持 `PLACEHOLDER`，不得编造数值。同步更新 `tests/test_config.py` 等引用骨架键名的测试。
- 时间一律用秒；缺失量用 `None`，不得用 0 代替。

## 范围与产出

1. **接触事件**（`events.py`）：
   - `ContactEvent`（`kind ∈ {official, kinematic_overlap}`，含时间、伙伴、`vₙ`、位置等）；
   - `classify_partner(...)`：按 BD 的 `robot_contact_bodies` 与当前抓持物名称把一个伙伴名分到 `PartnerClass`。
2. **严重度**（`severity.py`）：
   - `analytic_impulse_ns(mass_kg, v_normal_mps, restitution)` = `m(1+e)vₙ`；`restitution` 是**必填参数**，无默认（D9 b），`e < 0` 抛错；
   - `closing_normal_speed(v_hazard, v_partner, normal_partner_to_hazard)` = `max(0, (v_partner − v_hazard)·n̂)`（D9 a）；分离时为 0；**零向量法向抛错**，不得回退到速度方向；
   - 与仓库旧实现 `src/dynisafety/core/newton_force.py` 的 `l3_closing_speed_mps`、`l3_newton_normal_impulse_ns` 做**等价性测试**（按文件路径动态加载该文件，避免 import `dynisafety`；它只依赖 numpy 与 `dynisafety.core.contact` 的一个常量，测试里用同值桩替代）。旧实现的法向是“由手指向伙伴”，新约定相反：测试里把法向取反再喂旧实现，并**单列一个测试**说明两者在零法向输入上行为不同（旧：回退速度方向；新：抛错），不要掩盖这个差异。
3. **接触图语义**（`contact_graph.py`）：
   - 输入逐步的“同时接触的伙伴集合与法向”，输出：连续官方接触时长（秒）、约束状态 `{unconstrained, constrained, clamp}`、`t_q` 判定；`t_q`、角度阈值、**间隙容忍 `gap_tolerance_s`** 都由参数传入（D9 c）：接触中断的时间不超过 `gap_tolerance_s` 时仍算连续；
   - `clamp`：危险体同时接触一个静态场景体或机器人另一部件，且两个法向的夹角与 π 的差小于 `opposing_normals_tol_rad`（法向“大致相对”），持续 ≥ `t_q`。
4. **近失与接触前剩余时间**（`proximity.py`）：从逐步 trace（危险体位置、EEF 位置，可选真几何距离）计算 `min_separation_m`、`separation_kind ∈ {geometric, surrogate}`、`ttc_series`、`ttc_min_s`；有真几何距离时优先用，并标注种类。
5. **倾倒**（`topple.py`）：`tilt_deg_up_axis(ref_quat, cur_quat)`（忽略绕 z 的偏航）、`is_toppled(tilt, threshold)`（严格大于）。**回归测试**必须复现 `docs/audit/p0_semantics_audit.md` 附录表：纯偏航 40° 与 90° → 0°；绕 x 轴 40° → 40°；绕 x 轴 20° → 20°；并断言这与仓库旧的“相对四元数总角”口径在纯偏航输入上**不同**（把旧公式作为测试内的对照实现）。
6. **无效样本**（`invalid.py`）：
   - `InvalidClass`（`setup_invalid / sim_unstable / hazard_induced_unstable / none`）；
   - `classify_invalid(...)`：**归因规则函数**（D9 d）。输入：`setup_failed`、`engine_fault`（宿主检测到不稳定 / NaN / 引擎异常）、不稳定发生时刻、危险体 onset 时刻、不稳定实体与危险体质心的距离（可为 `None`）、`max_distance_m`（来自 YAML，参数传入）。规则：`setup_failed` → `setup_invalid`；未发生不稳定 → `none`；发生不稳定且（时刻 ≥ onset 且距离 ≤ `max_distance_m`）→ `hazard_induced_unstable`；其余发生不稳定的 → `sim_unstable`；**距离为 `None` 且时刻 ≥ onset → `hazard_induced_unstable`，并在返回值里带 `attribution_ambiguous=True`**（宁可保留，不得静默剔除）；
   - `metric_denominator(records)`：前两类排除、第三类保留；输出里**三类计数 + `attribution_ambiguous` 计数**都要有。
7. **暴露**（`exposure.py`）：`is_exposure_low(eef_positions, times_s, hit_point, t_start_s, t_end_s, radius_m)`：窗口内被测机器人 EEF 到命中点的最小距离 > `radius_m` 则为低暴露；窗口与半径都是**参数**（D9 e）；窗口内无样本时抛错。T06 调用它，不得重写。
8. **事件记录**（`event_log.py`）：`HazardEventLog` 数据类（字段见 `00_paradigm.md` §6，另加 `schema_version`）、JSONL 写 / 读、字段校验；`partner_class` 用 T01 的 `PartnerClass`，`constraint_state / separation_kind / invalid_class` 用枚举；缺失字段显式为 `None`，不得填 0。

## 验收标准（贴输出）

1. `pytest hazardlib/tests/semantics -q` 在 `hz38` 与 `hz311` 全绿；整个 `pytest hazardlib` 也全绿（含 T01、T02 既有测试）。
2. 倾倒回归测试复现 P0 附录的四行数值。
3. 等价性测试：对 200 组随机输入，`analytic_impulse_ns` 与仓库旧实现一致（相对误差 < 1e-9，法向已按约定取反）；零法向差异单列测试存在。
4. **法向方向**：桌面静止（伙伴）、危险体以 v 向下落、法向 +z → `vₙ = v`；危险体远离 → 0；用 T01 的 `ContactObs` 直接构造并调用，不自造类型。
5. 接触图：构造“危险体夹在机器人与桌面之间 0.3 s / 0.8 s”两个序列，`t_q=0.5` 时只有后者判 `clamp`；单侧接触判 `unconstrained`；**间隙**：在 0.8 s 序列中间挖一个短缺口，`gap_tolerance_s` ≥ 缺口时仍判 `clamp`，< 缺口时不判。
6. 近失：构造已知最小间距与 TTC 的轨迹，数值与解析值一致；无真几何距离时 `separation_kind == "surrogate"`。
7. 无效样本：规则表逐行测试（onset 前不稳定 → `sim_unstable`；onset 后且距离 ≤ 阈值 → `hazard_induced_unstable`；onset 后距离 > 阈值 → `sim_unstable`；距离 `None` → `hazard_induced_unstable` 且 `attribution_ambiguous`）；分母测试：构造含三类样本的集合，前两类被排除、第三类保留、计数正确。
8. 暴露：EEF 轨迹贴近命中点 → 不标；远离 → 标；窗口无样本 → 抛错。
9. `HazardEventLog` 往返读写无损；缺失字段读回为 `None`。
10. **类型复用**：`grep -rn "class PartnerClass\|class ContactObs" hazardlib/src/hazardlib/semantics` 无输出（贴命令与输出）。
11. `grep` 确认没有任何阈值数值默认值（阈值、`t_q`、角度、间隙、距离、半径、`e` 都由参数传入；用 AST 检查函数签名里无数值默认，沿用 T02 的做法）。
12. `semantics.yaml` 仍全是 `PLACEHOLDER`，且 `strict` 加载失败、非 strict 加载成功（沿用 T01 的骨架测试）。

## 不要做
不要实现运行时、探针、适配器；不要修改 `src/dynisafety/` 与 `hazardlib/host.py`；不要在代码里写 `t_q`、角度阈值、倾倒阈值、`e` 的默认值；不要改 `docs/design/00–03`（修订意见写进汇报）。

## 汇报
`docs/design/reports/T03.md`。列出你认为设计文档 §6 与 D9 中有歧义或可能不自洽的地方（例如 `clamp` 的法向判据、`hazard_induced_unstable` 的归因窗口是否应有结束时间），作为修订建议。
