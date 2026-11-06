# T02-F — T02 收尾（合并前的小改动）

起点分支：`hazard/T02`（**在同一分支上继续**，提交后推送，不另开分支）　依赖：T02　需要仿真器：否　先读：T00，你自己的 `docs/design/reports/T02.md`，以及 `docs/design/00_paradigm.md` §5.1–§5.2 与 §13（决定 D1–D6，已更新）。

T02 的审阅已完成，验收通过；下面是合并前要补的几处小改动，**只改命名、文档和配置骨架，不改任何算法**。

## 范围与产出

1. **重命名** `build_hit_path` 的参数 `hook_dt_s` → `release_window_s`（`slide` 与 `reach` 两族）。语义（D4）：释放速度是实例路径在这个**固定窗口**上的后向差分；调用者传入 YAML 的 `release_velocity_window_s`，**不是**引擎钩子步长。同步修改 `hit.py` 的模块文档、`HitPlan.diagnostics`（新增键 `release_window_s`）、全部测试、`hazardlib/README.md`。旧名字**不留别名**（还没有调用者）。
2. **文档字符串对齐决定**：
   - `hit.py`：把“首次物理接触比 `t_hit` 早约 …”改成 D1 的表述——τ 以**首次表面接触**为准；`hit_point` 是**质心**目标位置；调用方负责按 `standoff = r_hazard(n̂) + r_eef_contact` 换算并记录 `standoff_m`；
   - `reach` 的 `speed_profile ∈ {min_jerk, linear}`；`slide` 匀速；
   - `overshoot_m = 0` 的 `min_jerk` 末端速度为 0，**不得用于带严重度的实例**（D6），应归入 `occupancy.dwell`；
   - `fall`：onset 零速释放，重力 `g` 必须是宿主实测值，不得默认（D2）。
3. **配置骨架（D3）**：在 `configs/hazard/intensity.yaml` 的 `calibration:` 下新增键 `t_stop_min_consecutive_samples`、`t_stop_repeats`，以及 `staircase:`（`step_s`、`tau_min_s`、`tau_max_s`）。**值一律写 `PLACEHOLDER`，不要编造数值**；`t_stop_speed_threshold_mps` 保持原位。补测试：`strict=True` 加载这些键会抛可读错误，`strict=False` 可加载。
4. **汇报追加**：在 `docs/design/reports/T02.md` 末尾**追加**“T02-F 收尾”一节（做了什么、测试输出）；**不要改写**已有内容。

## 验收标准（贴输出）

1. `pytest hazardlib/tests -q` 在 `hz38` 与 `hz311` 全绿（整套，含 T01）。
2. 重命名后你原来的 40 个变异体检查**重新跑一遍**，报告存活数（应为 0）；若变异脚本因重命名需要调整，说明调整了什么。
3. `grep -rn "hook_dt_s" hazardlib docs/design/reports/T02.md` 只应在汇报的**历史描述**里出现（并标注已改名），源码、测试、README 里为零。
4. 签名里仍然没有任何数值默认值（沿用 T02 的 AST 检查）。
5. `git diff --stat` 只涉及：`kinematics/hit.py`、对应测试、`README.md`、`configs/hazard/intensity.yaml`、一个配置测试、`reports/T02.md`；**没有**其他文件。

## 不要做
不改算法；不新增数值默认值；不做 T04–T06 的工作；不修改 `docs/design/00–03`。

## 汇报
追加在 `docs/design/reports/T02.md` 末尾（见上）。完成后告诉协调者，由协调者合并 `hazard/T02`。
