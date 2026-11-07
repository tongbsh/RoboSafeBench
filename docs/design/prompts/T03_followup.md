# T03-F — T03 收尾（合并前，同一分支继续）

起点分支：`hazard/T03`（**同一分支继续提交，不要新开分支**）　依赖：T03　需要仿真器：否　先读：T00，你自己的 `reports/T03.md`，`00_paradigm.md` §6 与 §13（**D10 是本任务的依据**）。协调者先把 `claude/sweet-newton-2clmpw` 的最新提交合并进 `hazard/integration`，你再把 `hazard/integration` 合并进 `hazard/T03`（或 rebase），保证能看到 D10。

## 目标
把 T03 审阅后定下的 D10 落到代码与配置里。每一项一个小提交，T01、T02、T03 的既有测试必须保持绿；**不改任何算法之外的东西**。

## 范围

1. **`classify_invalid`（D10 b）**。新签名：`classify_invalid(setup_failed, engine_fault, instability_time_s, window_start_s, window_end_s, distance_m, max_distance_m, unstable_entity_is_hazard)`。规则按顺序：
   1. `setup_failed` → `setup_invalid`；
   2. 无不稳定 → `none`；
   3. `unstable_entity_is_hazard` 为真且 `instability_time_s >= window_start_s` → `hazard_induced_unstable`（不看窗口末端与距离）；
   4. `window_start_s <= instability_time_s <= window_end_s` 且距离 ≤ `max_distance_m` → `hazard_induced_unstable`；距离为 `None` → 同类并带 `attribution_ambiguous=True`；
   5. 其余 → `sim_unstable`。
   窗口两端闭区间，`window_end_s >= window_start_s` 否则抛错。更新规则表测试：窗口之后很久的不稳定（距离再近也）→ `sim_unstable`；危险体自身不稳定在窗口之后 → `hazard_induced_unstable`。`metric_denominator` 不变。
2. **`classify_partner`（D10 f）与抓持物名单**：
   - `robot_include` / `robot_exclude` 的每一项必须带匹配方式前缀 `exact:`、`prefix:`、`substr:`（`held_object` 仍是类别记号，仍被跳过）；**没有前缀的项抛错**，信息里给出该项；
   - 抓持物参数改为名单 `held_object_names: Sequence[str]`（空名单合法），逐项精确匹配；调用方（T04）传 `RobotState.held_object_bodies`，没有时传 `[held_object]`；
   - **回归测试**：`include=["prefix:robot0_", "exact:gripper"]` 时 `armchair`、`alarm_clock`、`farm_table` 都归 `scene`；旧的裸 `arm` 写法被拒绝；相机排除规则不变。
3. **`HazardEventLog` 新增可选字段 `first_scene_contact_t_s`**（D10 a；默认 `None`，校验有限；往返读写；`schema_version` 保持 `"1"`，现在还没有任何已写出的数据）。
4. **`semantics.yaml`**：删除 `contact.impulse_gate_ns`（D10 j）；新增 `stability.attribution_window_s`（`PLACEHOLDER`，注释写明窗口定义 `[t_onset, nominal_hit_time + 该值]`）；叶子仍全是 `PLACEHOLDER`，同步更新测试。
5. **文档字符串**：`contact_graph.py` 里写明 `clamp` 的对向伙伴是“另一个非危险体伙伴”，**含可移动的任务物体**（D10 g）；`events.py` / `exposure.py` 不用改。
6. **T01 的 `tests/test_source_hygiene.py`（本任务唯一允许改动的 T01 文件）**：Python 3.8 下固定标准库白名单太窄（`dataclasses enum functools hashlib json math numbers os pathlib re typing __future__`），T04–T06 一用 `collections / itertools / copy / bisect / abc / io / contextlib / operator` 就只在 `hz38` 上失败。把 3.8 的回退集合换成**完整标准库模块名的固定副本**（取 Python 3.11 的 `sys.stdlib_module_names` 写死进测试，并注释来源）；再加一个在 ≥3.10 上运行的测试，断言 `hazardlib/src` 实际 import 的标准库模块都在这份副本里。**不放宽“禁止仿真库 / scipy / torch”的检查。**
7. **汇报**：在 `docs/design/reports/T03.md` 末尾追加 “T03-F” 一节：每项的验证命令与真实输出、变异检查（新规则 / 新字段 / 新键各至少注入一个缺陷并确认被抓到）、偏离与理由。

## 验收标准（贴输出）
1. `pytest hazardlib -q` 在 `hz38`、`hz311` 全绿（贴总数，对比 T03 的 684）。
2. `classify_invalid` 规则表逐行（含上面新增两行与窗口两端边界）；`classify_partner` 回归测试；`first_scene_contact_t_s` 往返；`semantics.yaml` 仍全是 `PLACEHOLDER` 且 strict 加载失败。
3. 无数值默认值（沿用 AST 检查，贴输出）；`grep` 确认 `impulse_gate_ns` 已无引用。
4. 卫生测试：在 `hz38` 下临时加一个 `import collections` / `import itertools` 的源文件，确认**不再**失败；再加一个 `import scipy`，确认**仍然**失败（之后删除，贴输出）。
5. 新增变异体全部被抓到；`git status` 干净。

## 不要做
不改 `host.py`、`src/dynisafety/`、`docs/design/00–03`；不实现运行时；不为通过测试放宽任何阈值。`HazardHost.hazard_state` 与 `RobotState.held_object_bodies` 由 T04 的步骤 0 加，不在本任务里。
