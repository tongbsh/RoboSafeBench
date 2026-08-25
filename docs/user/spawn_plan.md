# `extra.spawn_plan` 字段约定

> **地位**：`EpisodeRecord.extra["spawn_plan"]` 的跨 backend **字段约定**。
> 不新增 `EpisodeRecord` 一等字段，**不 bump** `SCHEMA_VERSION`（仍为 `"1.0"`，见 [contracts.md](contracts.md) §1）。
> `dynisafety report` 仍不读 `extra`；消费方是分析 / 作图 / `tools/audit_l2_spawn.py` 一类脚本。
>
> L2 时间表与 `arc_height` 的 YAML 快照见 [benchmark_design.md](benchmark_design.md) §L2。
> 若本文引用的 YAML 与仓库不一致，**以 YAML 为准**。

本约定覆盖三个桌面 backend：`libero` / `simplerenv` / `robotwin`。RoboCasa 是地面导航梯子，不写 `spawn_plan`。

---

## 1. 这是字段约定，不是共享代码

三个 backend **各自**实现自己的 `disturbance/l2_spawn.py`（本轮只落 LIBERO；SimplerEnv / RoboTwin 本轮不改代码）。**不**抽 `dynisafety.core` 公共 SpawnFrame 包。

取舍理由：

1. **一套接口、三处打补丁的成本高于收益。** 规划器真正要调用的场景能力在三处并不对齐——LIBERO 有 MuJoCo 扫掠、没有桌面 AABB 查询；SimplerEnv 的 `targets.py` 已有桌面矩形 / AABB / 支撑面；RoboTwin 是双臂且只有 `get_contacts()` + 半径盘重叠。抽公共协议就要为「对方没有的 API」写空实现或 adapter，改一处几何公式还要同步三套补丁。
2. **分析脚本只依赖字段名。** 接受率、回退原因、左右切入直方图只读 JSONL 里同一组 key。字段对齐的收益已经覆盖「规划器源码共用」的收益。
3. **数值本就不能共用**（见 §5）。公共包容易把 `lateral_width_m` / `arc_height` 写成「看起来合理的默认」，违反「默认值只出现在各 backend 自己的 YAML」。

因此：规划器源码分叉，`extra.spawn_plan` 的 key / 类型 / 取值域锁定。分析脚本写一套；backend 实现各写各的。

对称写法（已存在）：LIBERO L3 把规划字典挂在控制器上，`backend.py` 在组 `extra` 时 `getattr(controller, "l3_plan", None)` 后写入 `extra["l3_plan"]`。`spawn_plan` 用同一挂钩，不要另开 JSONL 文件。

---

## 2. 何时写入

| 情况 | `extra.spawn_plan` |
|------|-------------------|
| 本 episode 未跑 spawn-diversity 规划器（默认关闭、旧 jsonl、非 L2） | **缺省整个 key**（与今天没有 `l3_plan` 时不写一致） |
| 规划器跑过（含回退到 `legacy_jitter`） | 写入 object；§3 共用字段 **全部出现** |

分析脚本：`if "spawn_plan" not in extra` 视为未启用多样性，不要用默认值补洞。

一旦写入，未知 key 必须忽略（LIBERO 可在同一 object 上加 backend 私有键，见 §4.1），但 §3 的共用键不得改名、不得缺席。

JSONL 可序列化：`list` / `str` / `bool` / `int` / `float` / JSON `null`。禁止写入 numpy 数组。

---

## 3. 共用字段表

`requested_mode` / `accepted_mode` 的 mode 字符串为 `"{slot}_{path_style}"`，或 `accepted_mode == "legacy_jitter"`。

`slot ∈ {left, center, right}`，`path_style ∈ {arc, linear}`。六个 diversity mode：`left_arc` / `center_arc` / `right_arc` / `left_linear` / `center_linear` / `right_linear`。`center_arc` 对应今天各 backend 的默认弧线切入。

| 字段 | JSON 类型 | 取值域 | 含义 |
|------|-----------|--------|------|
| `backend` | `str` | `libero` \| `simplerenv` \| `robotwin` | 写出方。必须与 `EpisodeRecord.backend` 相同。 |
| `enabled` | `bool` | `true` \| `false` | 本 episode 是否启用了 spawn-diversity 规划器。写入本对象时一般为 `true`。 |
| `requested_mode` | `str` | `"{slot}_{path_style}"` | round-robin 抽出的第一个 mode（`forbidden_slots` 剔除之后）。 |
| `accepted_mode` | `str` | `"{slot}_{path_style}"` 或 `"legacy_jitter"` | 实际执行的 mode。候选耗尽走今天的 jitter+arc 时为 `legacy_jitter`。 |
| `slot` | `str` \| `null` | `left` \| `center` \| `right` \| `null` | **accepted** 侧向格。`accepted_mode == "legacy_jitter"` 时为 `null`。 |
| `path_style` | `str` \| `null` | `arc` \| `linear` \| `null` | **accepted** 路径型。`legacy_jitter` 时为 `null`（执行仍走现有弧线，但不算 diversity 的 `arc` 格）。 |
| `fallback_reason` | `str` \| `null` | 见下方枚举 | 最终接受方案相对「请求 mode 的第一档形式」为何不同。请求 mode 原样通过则为 `null`。 |
| `fallback_chain` | `list[str]` | mode 字符串，按尝试顺序 | 试过的 mode 列表，**含**最终 `accepted_mode`。长度为 1 且等于 `requested_mode` 表示没有换格。 |
| `start` | `list[float]` | 长度 3，世界系米 | 实际执行的起点 XYZ。与 LIBERO `l3_plan` 的 `_as_xyz_list` 相同：`[x, y, z]`。 |
| `lateral_axis` | `list[float]` | 长度 2，XY 单位向量 | 侧向轴 `n̂`（世界 XY）。`P_left = P0 + w·n̂`，`P_right = P0 − w·n̂`。退化回退世界 `+Y` 时为 `[0.0, 1.0]`。 |
| `lateral_axis_degenerate` | `bool` | `true` \| `false` | 用于构造 `n̂` 的两点重合（或范数过小）时为 `true`，禁止静默 NaN。 |
| `min_clearance_m` | `float` \| `null` | 米，或 `null` | 层 A 沿路径采样的最小净空。未跑层 A（`legacy_jitter`）时为 `null`。 |
| `linear_solve_mode` | `str` | `straight` \| `trapezoid` \| `na` | `linear` 的净空解：整段反解直线 / 降级梯形 / 不适用（`arc` 或 `legacy_jitter`）。 |
| `validated_path_sha1` | `str` | 40 位小写 hex | 控制器实际消费的离散路径的 SHA-1。配方见 §3.2。 |
| `rollback_steps` | `int` | `≥ 0` | 执行期「这一步没吃上、路径下标仍前进」的次数。见 §3.3。 |

### 3.1 `fallback_reason`

| 值 | 何时写出 |
|----|----------|
| `null` | `requested_mode` 以第一档形式通过验收（`arc`，或 `linear` 且 `linear_solve_mode == "straight"`）。 |
| `linear_solve_infeasible` | 直线反解不满足净空或超过 `linear_max_start_z_m`。若同 slot 的梯形随后通过，`accepted_mode` 仍是 `{slot}_linear`，`linear_solve_mode` 为 `trapezoid`。 |
| `trapezoid_infeasible` | 梯形巡航高度也超过抬升封顶；该 `{slot}_linear` 候选被放弃，进入回退链下一档。 |
| `candidate_rejected` | 某个完整候选（slot+style，含已解出的路径）未过层 A / 层 B，换下一档。 |
| `all_candidates_rejected` | 回退序耗尽，`accepted_mode == "legacy_jitter"`。 |

同一 episode 只写 **一个** 终端原因：以最终接受方案为准，不要把链上每一次失败拼成数组（细节看 `fallback_chain` + `linear_solve_mode`）。

### 3.2 `validated_path_sha1`

路径是形状 `(n, 3)`、`float64`、C 连续的世界坐标数组，必须与执行离散化是同一份（LIBERO / SimplerEnv 即 `_build_arc_path` 的 `n = max(2, move_end_step - move_start_step + 1)`，见 §4）。

```text
sha1 = hashlib.sha1(np.ascontiguousarray(path, dtype=np.float64).tobytes()).hexdigest()
```

`legacy_jitter` 仍写出（那条未验收的弧线路径），以便对照执行。名字里的 `validated_` 表示「控制器消费的那份」，不表示一定过了层 B。

### 3.3 `rollback_steps`

LIBERO `_try_move`（`src/dynisafety/backends/libero/disturbance/l2.py`）在扫掠失败时回滚这一步的 qpos，但 `update_before_step` 里 `idx = task_step - move_start_step` 仍随 `task_step` 前进。规划通过 ≠ 执行吃完整条路径。本计数是格子鲁棒性的直接信号。

语义（三 backend 共用）：执行期本应占用某个 waypoint、因碰撞 / 接触拒绝并保持上一步位姿、同时路径下标仍前进的次数。没有逐步回滚机制的 backend 写 `0`，不要用机器人接触释放去凑这个数。

---

## 4. 三 backend 移植差异

下列函数名、路径、YAML 均按当前树核实。本轮只有 LIBERO 实现规划器；另外两处是移植时必须对齐的差异，不是「以后再看」。

### 4.1 LIBERO（MuJoCo）

| 项 | 现状 |
|----|------|
| 仿真 | MuJoCo。障碍物质心用 `set_joint_qpos` 传送。 |
| 路径离散 | `_build_arc_path`：`n = max(2, move_end_step - move_start_step + 1)`。YAML `move_start_step: 10`、`move_end_step: 30` → **n = 21**。`plan_num_samples: 60` 只存进 `self._plan_num_samples`，**不参与**这条路径。 |
| 执行窗口 | `update_before_step` 在 `move_start_step ≤ task_step < move_end_step` 用 `idx = task_step - move_start_step`（步 10…29 → idx 0…19）。`task_step ≥ move_end_step` 钉在 `final_pos`，路径最后一点（α=1）不经 `_try_move` 消费。 |
| 扫掠 | 有。`_swept_move_is_collision_free` 每段 `SWEEP_SUBSTEPS = 8`，调用 `_has_blocking_contact`。规划期与执行期都走它。 |
| 桌面边界 | **没有**桌面 AABB / 边查询。`get_table_center_pos` / `get_robot_base_pos`（`_support.py`）只给点。L3 的 `extra.libero.l3.start_xy_half: [0.45, 0.30]` 是 YAML 裁剪盒，不是场景查询。AEGIS vision 的 `extra.libero.aegis.vision.table_bounds` 是 YAML 视野盒，**不是** L2 桌面 API。 |
| 侧向轴 | 机器人锚定：`fwd = normalize(table_center.xy − robot_base.xy)`，`n̂ = rot90(fwd)`。两点重合则 `lateral_axis_degenerate: true`，`n̂ = (0, 1)`。 |
| 回滚 | `_try_move` 失败回滚本步并计入 `rollback_steps`。 |

LIBERO 可写在同一 object 上的私有键（分析脚本忽略即可）：

| 字段 | 类型 | 含义 |
|------|------|------|
| `jitter_suppressed` | `bool` | `enable_random_start=true` 且 diversity 开启时，diversity 路径不叠 ±8 cm jitter；legacy 回退档仍用 jitter。 |

`align_start_y_to_eef=true` 与 diversity 互斥（会抹掉 slot 的 Y 偏移）；这是组装期 `ValueError`，不会出现在一份合法的 `spawn_plan` 里。

### 4.2 SimplerEnv（SAPIEN / ManiSkill2）

| 项 | 现状 |
|----|------|
| 路径离散 | 与 LIBERO 同一公式：`SlidingObstacleController._build_arc_path` 里 `n = max(2, move_end_step - move_start_step + 1)`。YAML `move_start_step: 0`、`move_end_step: 15` → **n = 16**（不是 15）。执行用 `task_step < move_end_step`，步 0…14 吃 idx 0…14。 |
| 扫掠 | **无**。`update_before_step` 直接 `set_actor_pose`。 |
| 起点 | YAML `outside_table_dist: 0.10`。`assemble._build_l2` 读该值后走 `compute_l2_start_offset` → 优先 `compute_l2_outside_table_start_offset`：从桌心经落点射到桌边，再沿该方向伸出 `0.10` m。成功时 **丢掉** YAML `start_offset: [0.50, -0.10, 0.0]`。桌面 AABB 推不出来才回退 `mirror_start` / `start_offset`。 |
| 与 `align_start_y_to_eef` | YAML 为 `true`，但 `_build_l2` 传入 `align and outside <= 0`。`outside_table_dist=0.10` 时对齐 **关闭**，XY jitter 也被置 0。 |
| 验收能力 | 三家最强，见下表。移植时应复用这些函数做层 A / 落点，不要再写一套探针。 |

`src/dynisafety/backends/simplerenv/targets.py` 已有的场景查询（均已读过）：

| 函数 | 作用 |
|------|------|
| `get_tabletop_xy_bounds` | 从 `arena` 碰撞 mesh 上、高度接近 `table_height` 的顶点估 XY AABB，缓存在 `SceneAccess.tabletop_xy_bounds_cache`。 |
| `compute_l2_outside_table_start_offset` | 桌心 → 落点射线，落到 AABB 边后再加 `outside_table_dist`。 |
| `_probe_obstacle_aabb` | 把动态 actor 临时放到查询 XY（z=0）、读世界 AABB、再还原 pose。 |
| `_get_support_surface_z` | 与障碍物 XY AABB 重叠的场景 actor 最高顶面；否则参考物底部 / `table_height`。 |
| `_resolve_obstacle_target_xy_overlap` | 最多 8 轮、总位移 ≤ 0.35 m，沿轴向把落点从阻挡 AABB 推开。 |
| `get_actor_world_aabb` / `xy_aabb_overlaps` | 上述查询的几何原语。 |

**侧向轴不要按 LIBERO 重算。** 起点已经在「垂直于被击中的那条桌边、指向桌外」的射线上。`left` / `right` 应沿该矩形边滑动（与射线在 XY 上垂直，且仍在边外 `outside_table_dist`），而不是 `rot90(table_center − robot_base)`。后一种会把已经在桌外的点拖回桌内或拖到另一条边。

`scene.table_height` 在 `configs/backends/simplerenv.yaml` 为 `0.87`。桌面 XY 尺寸以 `get_tabletop_xy_bounds` 的运行时 AABB 为准，代码里没有写死长宽。

### 4.3 RoboTwin（SAPIEN，双臂）

| 项 | 现状 |
|----|------|
| 路径离散 | **没有**预计算 waypoint 数组。`SlidingObstacleController._path_pos(alpha)` 每仿真步现场求值。`update_before_step` 在 `sim_step <= move_end_step` 时 `alpha = (sim_step - move_start_step) / max(move_end_step - move_start_step, 1)`。YAML `move_start_step: 1`、`move_end_step: 4000` → 闭区间 1…4000 共 **4000** 次求值，步 4000 时 α=1。这与 LIBERO / SimplerEnv 的 `linspace` + **开**区间 `< move_end_step` 不是同一套离散化。 |
| 要写 `validated_path_sha1` 时 | 必须先物化与执行相同的点列：按上式把闭区间每个 `sim_step` 的位姿建成 `(4000, 3)`，禁止改用 21/16 点去「对齐另外两家」。 |
| 起点 | `_compute_approach_start`（`manager.py`）已有两档：`head_on`（沿 EEF→落点，从正面推来）与 `perpendicular`（垂直于该轴，`flip_side` 镜像）。后者是 slot 的雏形。函数参数默认 `mode="perpendicular"`；**当前 L2 实际默认**是 `configs/backends/robotwin.yaml` 的 `extra.robotwin.l2_start_mode: head_on`（`l2_start_min_distance: 0.85`，`l2_flip_side: false`）。`l2.yaml` 注释写明这三项走 `extra.robotwin`，不在 `DisturbanceL2Config`。 |
| 双臂 | `primary_arm_tag = resolve_primary_arm_tag(self.task)`（`primary_arm.py`，缺注册抛 `KeyError`）。EEF 用 `task.get_arm_pose(primary_arm_tag)`。`left` / `right` **必须**相对 `primary_arm`（或相对 `_compute_approach_start` 的 `shift`）定义，禁止用世界 +X/+Y。 |
| 碰撞 | 执行期 `_detect_obstacle_collision` 扫 `task.scene.get_contacts()`，只认障碍物↔机器人 / 手持物。落点 `_resolve_target_xy_overlap` 用 XY 半径盘：`‖Δxy‖ < actor_r + obstacle_r + obstacle_avoid_margin`（margin YAML `0.015`）。**没有**扫掠。路径验收要新写，不能假装 `get_contacts()` 在传送路径上等价于 LIBERO 的 `_swept_move_is_collision_free`。 |
| `rollback_steps` | 当前无逐步回滚：机器人接触则 `handle_collision` 整段释放。在实现逐步回滚之前写 `0`。 |

`settle_steps: 10` 经 `config_bridge.py` 映射为 `l2_hold_steps`（落地后保持步数），不是第三家那种「再飞一段」。

---

## 5. 不可跨 backend 共用的数值

`lateral_width_m`、`linear_z_clearance_m`、`linear_max_start_z_m`、`arc_height`、以及一切以「米 / 步」为单位的规划旋钮，**必须**写在各 backend 自己的 `configs/backends/<b>/levels/l2.yaml`（或该 backend 的 `extra.*`），禁止抄另一家的数，也禁止在共享 Python 里写默认。

原因是时间尺度和桌面尺度都不在一个数量级。当前 YAML / 代码快照：

| | LIBERO `…/libero/levels/l2.yaml` | SimplerEnv `…/simplerenv/levels/l2.yaml` | RoboTwin `…/robotwin/levels/l2.yaml` |
|--|----------------------------------|------------------------------------------|--------------------------------------|
| `move_start_step` | `10` | `0` | `1` |
| `move_end_step` | `30` | `15` | `4000` |
| 路径点数（按各家自己的公式） | `30 − 10 + 1 = 21`（`_build_arc_path`） | `15 − 0 + 1 = 16`（`_build_arc_path`） | 每步求值，闭区间 **4000** 点（`l2.py`） |
| `settle_steps` / hold | `15` | `1` | `10`（→ `l2_hold_steps`） |
| `arc_height` (m) | `0.2` | `0.10` | `0.12` |
| 起点尺度 | `start_offset: [0.50, -0.10, 0.0]`（桌内偏置 + 可选 ±8 cm jitter） | 桌外射线 `outside_table_dist: 0.10` | `l2_start_min_distance: 0.85`（`robotwin.yaml`） |
| 桌面查询 | 仅 `table_top` 点；无 AABB | `get_tabletop_xy_bounds` 运行时 AABB；`table_height: 0.87` | 仅 `table_surface_z: 0.74`；无 XY 边界 |
| 控制时钟（对照，非 L2 键） | suite `max_steps` 220…520 | Bridge `control_freq: 5`，`max_steps: 120` | 任务 `max_steps` 400…1200，L2 窗口已是 4000 仿真步 |

同一 `lateral_width_m = 0.15`：在 LIBERO 21 点弧上是「左右各 15 cm」；在 RoboTwin 0.85 m 桌外起点 + 4000 步上可能贴边或穿模；在 SimplerEnv 已经离桌 10 cm 的射线上则应沿边滑动，宽度要按 AABB 边长另标。`arc_height` 三家已经分别是 `0.2` / `0.10` / `0.12`，把 LIBERO 的 20 cm 抬升抄到 SimplerEnv 会改变 Bridge 的过障净空。

[benchmark_design.md](benchmark_design.md) 的原话仍然成立：不要用这些步数指标把三家排成一条「难度」轴。
