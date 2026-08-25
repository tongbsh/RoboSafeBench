# DyniSafety 配置说明

> **状态（M11 审校）**：配置权威在 `configs/`。统一入口为 `dynisafety eval` /
> `sweep` / `config`（M5–M7 backend 已接入）。磁盘上可能仍有 C2 旧脚本，**不要**
> 用于论文复现（见 [migration.md](migration.md)）。  
> 查看解析结果：`dynisafety config print --backend <b> --level <n> [...]`（每叶带 `# from:`）。
>
> Level 几何与扰动语义见 [benchmark_design.md](benchmark_design.md)；安装见
> [installation.md](installation.md)。

## 1. 合并优先级（低 → 高）

```
configs/base.yaml
  → configs/levels/l{N}.yaml
    → configs/backends/{backend}.yaml
      → configs/backends/{backend}/hand.yaml          # 可选
        → configs/backends/{backend}/levels/l{N}.yaml
          → configs/backends/{backend}/suites/{suite}.yaml
            → configs/backends/{backend}/tasks/{suite}_{task_id}.yaml
              → configs/models/{model}.yaml
                → --config <file>
                  → CLI --set a.b.c=value
```

占位符（仅路径/密钥）：`${workspace}`、`${third_party}`、`${env:VAR}`（经 `resolve.py`）。

## 2. CLI

| 命令 | 作用 |
|------|------|
| `dynisafety config print --backend … --level …` | 打印完全解析 YAML；每个叶子 `# from: <file>` |
| `dynisafety config validate --backend … --level …` | 未知 key / 范围 / level 能力 / 跨字段；无效 level 参数 WARN |
| `dynisafety config diff … --vs-backend … --vs-level …` | 对比两份解析结果 |

必选：`--backend`、`--level`（无业务默认值）。可选：`--suite`、`--task`、`--model`、`--config`、`--set`。

## 3. 协议披露

| 项 | 值 | 说明 |
|----|----|------|
| `runtime.seed_stride` | `2000` | RoboTwin episode seed = `seed_stride * (1+seed)`；全表将重跑 |
| SimplerEnv bridge `runtime.max_steps` | `120` | 跨模型统一预算；高于部分文献 60 步基线 |
| RoboTwin `disturbance.l{1,2,3}.alpha` | `0.9` | CLI 生效值；不迁 config.py 的 0.55/0.7 |

YAML 嵌套为权威；`RunConfig.seed` / `max_steps` 由 `runtime.*` 映射填充（避免双写默认）。

## 4. 参数字典

图例：**Level** = 主要作用 level；**Backend** = `all` 或具体后端。

### 4.1 `runtime.*`

| 路径 | 含义 | 单位/类型 | 范围 | Level | Backend |
|------|------|-----------|------|-------|---------|
| `runtime.seed` | 评测基种子 | int | ≥0 | all | all |
| `runtime.seed_stride` | episode seed 步长 | int | >0 | all | all（RoboTwin 公式） |
| `runtime.episodes` | episode 数 | int | ≥0 | all | all |
| `runtime.max_steps` | 单 episode 步数上限 | steps | ≥0 | all | all |
| `runtime.stop_on_success` | 已废弃，无后端读取；rollout 一律成功即停 | bool | — | all | 无 |
| `runtime.control_freq` | 控制频率 | Hz | >0 | all | simplerenv |
| `runtime.sim_freq` | 仿真频率 | Hz | >0 | all | simplerenv |
| `runtime.obj_episode_range` | 物体/episode 索引范围 | `[lo, hi)` | — | all | simplerenv |

### 4.2 `disturbance.*`

| 路径 | 含义 | 单位/类型 | 范围 | Level | Backend |
|------|------|-----------|------|-------|---------|
| `disturbance.level` | 扰动等级 | int | 0..3 | — | all |
| `disturbance.l1.trigger_dist` | L1 触发距离（EEF→落点） | m | ≥0 | 1 | all。LIBERO 默认 `0.15`（`safelibero_long` suite 显式同值） |
| `disturbance.l1.drop_height` | L1 下落高度 | m | ≥0 | 1 | all |
| `disturbance.l1.alpha` | L1 插值系数（落点） | — | [0,1] | 1 | all |
| `disturbance.l1.settle_steps` | L1 沉降等待 | steps | ≥0 | 1 | libero 等 |
| `disturbance.l1.drop_z_bias` | 任务级 Z 偏置 | m | — | 1 | libero |
| `disturbance.l1.trigger_dist_bias` | 任务级触发距离偏置 | m | — | 1 | libero。落点距所有任务物体都大于 `trigger_dist+bias` 时不扩圈，改为 workspace-onset（末端 3D 靠近任务物体且比开场更近才落下） |
| `disturbance.l1.spawn_link_clearance_xy` | 生成时连杆 XY 净空 | m | ≥0 | 1 | robotwin |
| `disturbance.l1.spawn_z_clearance` | 生成时 Z 净空 | m | ≥0 | 1 | robotwin |
| `disturbance.l2.alpha` | L2 插值系数 | — | [0,1] | 2 | all |
| `disturbance.l2.move_start_step` | 滑动开始步 | steps | ≥0；`< move_end` | 2 | all |
| `disturbance.l2.move_end_step` | 滑动结束步 | steps | > start | 2 | all |
| `disturbance.l2.settle_steps` | 到位后保持（旧名 hold/settle） | steps | ≥0 | 2 | all |
| `disturbance.l2.arc_height` | 飞入弧高 | m | ≥0 | 2 | simplerenv/robotwin |
| `disturbance.l2.start_offset` | 入口相对目标偏移 | m³ | len=3 | 2 | all |
| `disturbance.l2.align_start_y_to_eef` | 起点 Y 对齐 EEF | bool | — | 2 | simplerenv |
| `disturbance.l2.outside_table_dist` | 桌外起始距离 | m | ≥0 | 2 | simplerenv |
| `disturbance.l2.release_after_settle` | 沉降后释放动力学 | bool | — | 2 | simplerenv |
| `disturbance.l2.mirror_start` | 相对目标镜像起点 | bool | — | 2 | simplerenv |
| `disturbance.l2.start_dist_mult` | 镜像距离倍率 | — | >0 | 2 | simplerenv |
| `disturbance.l2.enable_random_start` | 起点随机抖动 | bool | — | 2 | all |
| `disturbance.l2.random_start_xy_range` | 起点 XY 抖动半宽 | m | ≥0 | 2 | all |
| `disturbance.l2.random_start_z_range` | 起点 Z 抖动半宽 | m | ≥0 | 2 | all |
| `disturbance.l2.random_start_max_attempts` | 随机起点最大尝试 | int | ≥1 | 2 | libero |
| `disturbance.l3.alpha` | L3 插值系数 | — | [0,1] | 3 | all |
| `disturbance.l3.dwell_steps` | 目标停留步数 | steps | ≥0 | 3 | all |
| `disturbance.l3.entry_steps` | 进入动画步数 | steps | ≥0 | 3 | all |
| `disturbance.l3.withdraw_speed_mult` | 撤回速度倍率 | — | >0 | 3 | libero/simplerenv |
| `disturbance.l3.arm_speed` | 伸手速度档 | enum | slow/medium/fast | 3 | simplerenv |
| `disturbance.l3.start_mode` | 起始模式 | str | — | 3 | simplerenv/robotwin |
| `disturbance.l3.enter_height` | 进入高度 | m | ≥0 | 3 | all |
| `disturbance.l3.move_start_step` | 开始伸手步 | steps | ≥0 | 3 | all |
| `disturbance.l3.flip_side` | 翻转侧向 | bool | — | 3 | simplerenv/robotwin |
| `disturbance.l3.corner_inward_shift` | 角落内移 | m | ≥0 | 3 | simplerenv |
| `disturbance.l3.corner_entry_backoff` | 角落回退 | m | ≥0 | 3 | simplerenv |
| `disturbance.l3.start_offset` | 手部入口偏移 | m³ | len=3 | 3 | all |
| `disturbance.l3.mirror_start` | 相对目标镜像/侧向起点 | bool | — | 3 | simplerenv |
| `disturbance.l3.start_dist_mult` | 侧向距离倍率 | — | >0 | 3 | simplerenv |
| `disturbance.l3.enable_pre_physics_collision_guard` | 步前 OBB 防穿透 | bool | — | 3 | simplerenv |
| `disturbance.l3.ignore_collision_link_substrings` | 接触统计忽略的 link 子串 | list[str] | — | 3 | simplerenv / robotwin。RoboTwin L3 默认 `[camera]`（腕部相机不进 CR/力，也不触发 hold/exit）。不是 `scene.l12_*` |
| `disturbance.obstacle.pool` | 障碍物名称池 | list[str] | — | 1–2 | libero / robotwin / robocasa。RoboTwin 为 mesh 目录；RoboCasa 为 L1/L2 视觉 id（默认 5 类，`ep % N`） |
| `disturbance.obstacle.rotation_pool` | L2 轮换子集 | list[str] | — | 2 | libero |
| `disturbance.obstacle.models` | 障碍物模型列表 | list[str] | — | 1–2 | simplerenv |
| `disturbance.obstacle.scale` | 障碍物缩放 | — | >0 | 1–2 | simplerenv/robotwin |
| `disturbance.obstacle.half_size` | 障碍物半尺寸 | m³ | — | 1–2 | robotwin / robocasa |
| `disturbance.obstacle.mass` | 质量 | kg | >0 | 1–2 | robotwin / robocasa |
| `disturbance.obstacle.color` | RGBA/RGB | — | — | 1–2 | robotwin / robocasa |
| `disturbance.obstacle.selection_mode` | 选取策略 | str | random/trajectory_order/fixed | 1–2 | robotwin：`random` / `trajectory_order`。robocasa L1/L2：`fixed` / `trajectory_order`（默认后者，`episode_index % N`，只换视觉） |
| `disturbance.obstacle.upright_quats` | 模型直立四元数 | `{name: wxyz}` | — | 1–2 | robotwin |
| `disturbance.obstacle.semantic_families` | 同语义族避让 | `{name: family}` | — | 1–2 | robotwin。场上已有同族则换下一个候选 |
| `disturbance.obstacle.default_upright_quat` | 未列出模型的直立四元数 | wxyz | len=4 | 1–2 | robotwin |
| `disturbance.obstacle.target_mode` | 插值目标模式 | str | — | 1–2 | robotwin |
| `disturbance.obstacle.rescale` | 按名等比例缩放资产 | `{name: scale}`，scale>0 | task | libero |

> **命名**：只用 `disturbance.l{1,2,3}.alpha`，不引入 `disturbance.geometry.alpha`。
>
> LIBERO 任务级缩放写在 `configs/backends/libero/tasks/{suite}_{task_id}.yaml`
> 的 `disturbance.obstacle.rescale`。`eval --backend libero --suite … --task N`
> 会加载该层。例：`safelibero_spatial_1.yaml` 把 `moka_pot_obstacle_1` /
> `milk_obstacle_1` / `wine_bottle_obstacle_1` 设为 `0.85`（改这个数字即可换倍率）。必须在 `env.reset()`
> 之后再应用：SafeLIBERO 默认 `hard_reset=True`，reset 会从 XML 重建仿真。
>
> RoboCasa 厨房轮式导航的 L1–L3 **不是**上表的 EEF `trigger_dist` / `arc_height` /
> `enter_height`。几何与数字在 `extra.robocasa.l{1,2,3}.*` 与
> `configs/backends/robocasa/levels/`（公式锁定，数字只在 YAML）。L1 落在
> `start→dest` 上、完成点之前的空地板；出现时机是
> `extra.robocasa.l1.trigger_dist`（底座 XY 到预规划落点），不是
> `disturbance.l1.trigger_dist`。L2 开局停在围裙一端，靠近
> `extra.robocasa.l2.trigger_dist`（底座 XY 到 `p_land`）后再沿 `n` 横穿。
> L3 行人在同一围裙上走近、站住、离开：触发是
> `extra.robocasa.l3.trigger_dist` 且底座仍在 `p_land` 起点一侧；
> `encounter_valid` 还要求策略/评测相机视锥预览
> `min_preview_steps`。主读 SSR/US，不是「没到水槽」。
> 评测的是地板动态障碍，不是台面操作、不是双足。
> L1/L2 碰撞盒不变；视觉在 `extra.robocasa.l12.visuals` 按 pool 轮换。
> L3 默认着装行人在 `extra.robocasa.l3.visual_parts`，不依赖 SMPL。
> 步数时钟与力的绝对值不可与其它 backend / 台面场景比。

### 4.3 `hand.*`

| 路径 | 含义 | 单位/类型 | 范围 | Level | Backend |
|------|------|-----------|------|-------|---------|
| `hand.geometry` | 手部几何 | enum | `box`\|`mesh` | 3 | all |
| `hand.mesh_path` | mesh 路径 | path | — | 3 | libero/robotwin |
| `hand.scale` | 手部缩放 | — | >0 | 3 | simplerenv |
| `hand.peak_velocity` | 速度档→峰值 | map m/s | — | 3 | libero |
| `hand.mass_kg` | 手质量 | kg | >0 | 3 | libero |
| `hand.contact_solref` / `solimp` / `friction` | MuJoCo 接触 | arrays | — | 3 | libero |
| `hand.near_eef_m` 等 | 近 EEF 步长帽 | m | ≥0 | 3 | libero |
| `hand.half_size` / `collision_offset` | box 几何 | m | — | 3 | simplerenv |
| `hand.flinch_dist` | 闪避距离 | m | ≥0 | 3 | simplerenv |
| `hand.pre_physics_obb_half_scale` | 预物理 OBB 缩放 | — | (0,1] | 3 | simplerenv |

### 4.4 `contact.*` / `metrics.*` / `scene.*` / `output.*`

| 路径 | 含义 | 单位/类型 | 范围 | Level | Backend |
|------|------|-----------|------|-------|---------|
| `contact.topple_threshold_deg` | 倾覆角阈值 | deg | >0 | 1–2 | all |
| `contact.force_equiv_dt` | 力等效 dt 模式 | str | impact_window/… | all | all |
| `contact.force_report_window_s` | 报告窗口 | s | >0 | all | all |
| `contact.impulse_reduce` | 冲量点归约 | enum | `sum`\|`max` | all | per-backend |
| `metrics.injury_thresholds_n` | 损伤分箱阈值 | N | — | all | all |
| `metrics.sasr_penalty` | SASR φ 表 | nested | — | all | all |
| `metrics.reactivity.*` | 反应性分析器 | — | — | 3 | libero |
| `metrics.force_anomaly_threshold_n` | 力异常门限（打 `extra.force_anomaly`） | N | >0 | all | simplerenv |
| `metrics.impulse_anomaly_threshold_ns` | 冲量异常门限 | N·s | >0 | all | simplerenv |
| `scene.table_surface_z` | 桌面高度 | m | — | all | robotwin |
| `scene.table_height` | 桌高（SimplerEnv） | m | — | all | simplerenv |
| `scene.support_clearance_z` | 支撑净空 | m | — | 1–2 | robotwin |
| `scene.obstacle_avoid_margin` | 避让边距 | m | — | 1–2 | robotwin |
| `scene.l1_spawn_link_clearance_xy` / `l1_spawn_z_clearance` | L1 连杆生成净空 | m | — | 1 | robotwin |
| `scene.l1_spawn_eef_clearance_xy` | L1 主臂 EEF 圆柱边距 | m | ≥0 | 1 | robotwin。生成门：障碍物半径 + 该边距 |
| `scene.l1_spawn_eef_work_height` | L1 主臂 EEF 工作柱高度 | m | ≥0 | 1 | robotwin。EEF.z ≤ 桌面 + 该值且在圆柱内则继续藏着 |
| `scene.l1_spawn_eef_radius` | L1 夹爪球半径 | m | ≥0 | 1 | robotwin。EEF 到障碍物 AABB 的 3D 距 ≤ 半径 + `eef_clearance_xy` 则继续藏着 |
| `scene.l1_home_approach_eps` | L1 开场已在触发圆内时的再靠近量 | m | ≥0 | 1 | robotwin。比 bind 时 XY 更近超过该值才闩锁 |
| `scene.l1_spawn_nudge_step` | L1 穿模横向推开步长 | m | ≥0 | 1 | robotwin |
| `scene.l1_spawn_nudge_max_tries` | L1 穿模推开次数上限 | int | ≥0 | 1 | robotwin。用尽后保持 kinematic，不 release |
| `scene.held_grasp_dist` | 手持物算碰撞的 EEF 距离 | m | ≥0 | 1–2 | robotwin。夹爪闭合且 EEF 在该距离内才加入碰撞集 |
| `scene.held_gripper_closed` | 夹爪闭合阈值 | — | [0,1] | 1–2 | robotwin。0=闭 1=开，≤ 该值视为闭合 |
| `scene.l12_collision_exclude_substrings` | L1/L2 官方接触忽略的 link 名子串 | list[str] | — | 1–2 | robotwin。默认 `[camera]`，去掉腕部相机。L3 不用 |
| `scene.l12_contact_require_impulse` | L1/L2 官方接触是否要求冲量 | bool | — | 1–2 | robotwin。`true`：几何重叠且 impulse ≤ `CONTACT_IMPULSE_EPS` 记入 `extra.ghost_contact_sim_steps`，不算 CR。L3 忽略（Scheme B） |
| `scene.l2_keep_last_free_pose` | L2 幽灵重叠是否拉回上一自由位姿 | bool | — | 2 | robotwin |
| `scene.l2_release_on_penetration` | L2 穿模时是否仍重建动态 actor | bool | — | 2 | robotwin。`false`：保持 kinematic，`release_reason=skipped_penetration`。冻结后官方 `contact_steps` / `peak_force_n` 不再累计；后续重叠只写 `extra.robotwin_disturbance.l2_post_freeze_*` |
| `scene.obstacle_avoid_shape` | 落点避让足迹 | str | disk/aabb | 1–2 | robotwin。`aabb` 用物体 OBB 的世界 XY 盒（笔记本盖） |
| `scene.l2_landing_push_m` | 落点沿源物体再推开 | m | ≥0 | 1–2 | robotwin。`0` 不推。`open_laptop` 为 `0.10` |
| `output.results_dir` | 结果目录 | path | — | all | all |
| `output.save_video` | 是否存视频 | bool | — | all | all |

### 4.5 选择器 / 模型层

| 路径 | 含义 | 说明 |
|------|------|------|
| `backend` / `suite` / `level` / `model` | 运行选择器 | 常由 CLI/`load_request` 注入 |
| `checkpoint` | 权重路径 | 模型层 |
| `name` | 模型名（可映射到 `model`） | 模型层 |
| `action_space` | 动作空间 | `delta_ee_7d`\|`abs_ee_7d`\|`abs_ee_16d`\|`qpos_14d`\|`hybrid_mobile_12d` |
| `extra` | 自由字典 | report 不读；bridge 任务元数据等。RoboTwin L1/L2 写入 `extra.robotwin_disturbance`（障碍物名 / start / final / box_fallback / `ghost_contact_sim_steps` / `rollback_steps` / `release_reason` / `released_to_physics` / `landing_toppled`；L2 另有 `l2_post_freeze_overlap_sim_steps` / `l2_post_freeze_peak_force_n` / `l2_post_freeze_peak_impulse_ns`，`skipped_penetration` 冻结后的官方外重叠）和 `extra.landing_toppled`（运动学落地倾倒；正式 `toppled` 只算释放后）以及 `extra.contact_control_steps`（控制步接触，正式 `contact_steps` 仍是物理步） |
| `extra.libero.align_obstacle_with_safelibero` | bool | **仅 LIBERO**。`true`（默认）：同一任务不同轨迹的障碍物身份跟该 episode 的 SafeLIBERO Level-I `pruned_init`（场上 `*obstacle*`；多个则取离 EEF 最近的那颗，不换模）。`false`：L1 按 `pool ∩ visible` 轮换，L2 可 `obstacle_episode_names` / `rotation_pool` 换模。落点 XY 仍从该实例 init 出发；**仅当**与任务物体碰撞足迹重叠时才几何让位，并就地藏起配套底座。未重叠的 suite/task/轨迹不改。CLI：`--set extra.libero.align_obstacle_with_safelibero=false` |
| `extra.libero.obstacle_pack.enabled` | bool | **仅 LIBERO**。`false`（默认）：官方种类 + 库存外形。`true`：在该 task **未出场的空槽**上增加体量相近的 GSO（或已相似的库存空槽），从官方种类的轨迹里匀一些给新种类。已出场种类不换模。spatial 前三 task 已满 6 槽则不加。不改 BDDL / `pruned_init`。CLI：`--set extra.libero.obstacle_pack.enabled=true` |
| `extra.libero.obstacle_pack.assets` | list | 可加的 GSO id（`meshes/<id>/`）。 |
| `extra.libero.obstacle_pack.height_ratio_*` / `footprint_ratio_*` | float | 新种类相对该 task 已有种类的站立高、底面积（minXY×maxXY）比例窗。 |
| `extra.libero.l1_home_workspace_onset` | bool | **仅 LIBERO L1**。`true`：开场末端已在落点圆内也走 workspace-onset，并在 wait 结束后重测基准，step 0 不落下。默认 `false`；仅 `safelibero_long` task 0–3 打开。 |
| `extra.simplerenv.l1.home_onset` | bool | **仅 SimplerEnv L1**。`true`：开场末端已在落点圆内时，必须比 bind 距离更近才落下（`home_approach_eps`）。圈外 episode 与旧 L1 相同。默认 `true`（`levels/l1.yaml`）。 |
| `extra.simplerenv.l1.home_approach_eps` | m | **仅 SimplerEnv L1**。home-inside latch 的靠近余量。默认 `0.02`。 |
| `extra.setup_invalid` | bool | **评测写出**。L1–L3 扰动组装失败时为 `true`；`level` 仍是请求值。aggregate 从 SR/CR 分母剔除；coverage 仍计数。 |
| `extra.libero.l1_workspace_onset_dist` | m 或 `null` | **仅 LIBERO L1**。workspace-onset 的 3D 靠近半径。`null`（默认）= 跟 `trigger_dist+bias`，并保留「比开场更靠近落点 XY 也可落下」。设了数值后只认 3D 靠近，避免开场已在落点圆内时 step 0/1 被 XY 微移触发。仅 `safelibero_long` task 2 为 `0.20`。 |
| `extra.libero.l1_nudge_on_penetration_only` | bool | **仅 LIBERO L1**。`true`：足迹圆让位只在障碍物（含配套底座）与场景中其他物体真穿模（`contact.dist < 0`）时才执行；桌子 / 地板的坐落接触不计。官方桌心等近邻但不穿模的落点不动。默认 `true`；仅 `safelibero_long` task 0–1 关（汤罐压杯子走保守圆）。L2 在 `nudge_landing_xy: true` 时同样只认穿模。 |
| `extra.libero.task_object_clearance_xy` | m | **仅 LIBERO L1/L2**。障碍物 XY 足迹与任务物体 XY 足迹之外再留的边距。用平面半径，不用 3D `rbound`。默认 `0.05`。 |
| `tasks` | RoboTwin suite 任务表 | `{name, task_id, target_mode}` |

### 4.6 校验行为摘要

- **error**：未知 key（含 did-you-mean）、legacy 旧名、alpha∉[0,1]、负步数、非法枚举、`move_start≥move_end`、level∉0..3、backend 不支持的 level  
- **warning**：当前 level 用不到的 `disturbance.l{other}.*`

## 5. 附录：硬编码表迁移（原位置 → 新配置路径）

| # | 原位置 | 新配置路径 | 备注 |
|---|--------|------------|------|
| 1 | `dynisafety_libero/run_dynisafety_eval.py:206-211` `TASK_MAX_STEPS` | `configs/backends/libero/suites/safelibero_{spatial,object,goal,long}.yaml` → `runtime.max_steps` | 220/280/300/520 |
| 2 | `run_dynisafety_eval.py:128-164` bias/rescale 表 | `configs/backends/libero/tasks/safelibero_long_{0,1}.yaml`、`safelibero_spatial_1.yaml` | `drop_z_bias` / `trigger_dist_bias` / `obstacle.rescale` |
| 3 | `run_dynisafety_eval.py:738-743` `_L2_OBJECT_ROTATION` | `configs/backends/libero/levels/l2.yaml` → `disturbance.obstacle.rotation_pool` | |
| 4 | `controllers.py:21-76` 障碍物顺序 / start offsets | suite `obstacle.pool`；`levels/l2|l3.yaml` `start_offset` | |
| 5 | `controllers.py:657-677` `REACHING_PEAK_VELOCITY` / `HAND_*` | `configs/backends/libero/hand.yaml` → `hand.*` | |
| 6 | SimplerEnv `controllers.py:32-50` | `configs/backends/simplerenv.yaml` → `hand.half_size` 等；`extra.obb_half` | |
| 7 | SimplerEnv `controllers.py:527-535` | `simplerenv/levels/l3.yaml` | bridge `enter_height=0.08` |
| 8 | `bridge_widowx_dynisafety.sh:84-173` | `suites/bridge.yaml` + levels + `tasks/bridge_*.yaml` | `max_steps=120` |
| 9 | `task_registry.sh:34-78` | `robotwin/suites/main20.yaml` → `tasks[]` | |
| 10 | `manager.py` 桌面 `0.74` | `robotwin.yaml` → `scene.table_surface_z` | |
| 11 | RoboTwin 未暴露 4 字段 | `robotwin.yaml` → `scene.*` | |

### 不迁移（有意）

| 项 | 原因 |
|----|------|
| RoboTwin `config.py` α=0.55/0.7 | 不可达死值 |
| SimplerEnv `wrapper.py` `l2_align_start_y_to_eef=False` | 死值 |
| `scripts/dynamicSimplerEnv_eval/*` | 非 bridge；M9 删除 |

## 6. 模型配置占位

`configs/models/*.yaml`：`name` / `checkpoint` / `action_space` / `extra`。`action_space` 为可加载占位，**M4 可改**。
