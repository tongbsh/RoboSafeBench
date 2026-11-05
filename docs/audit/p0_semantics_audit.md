# P0 — 安全语义对照审计

范围：四个后端（LIBERO / SimplerEnv / RoboTwin / RoboCasa）里与「接触、力、倾倒、无效布置、时间基准」相关的现有口径。
方法：只读代码与配置，未运行任何仿真器（审计环境无 GPU，且未安装 MuJoCo / SAPIEN / RoboSuite / LIBERO）。
唯一的运行性检查是附录里的倾角定义对比（纯 numpy，调用 `core/contact.py` 的两个纯函数）。

标注约定：**[已核实]** = 代码或配置里直接看到；**[推断]** = 由代码结构推出，未见文档佐证；**[未核实]** = 需要运行或读更多代码。

---

## 1. 语义对照表

| 维度 | LIBERO | SimplerEnv | RoboTwin | RoboCasa |
|---|---|---|---|---|
| 引擎 / `force_source` | MuJoCo（robosuite），`mujoco_mj_contactForce` | SAPIEN 2 + ManiSkill2，`sapien_impulse` | SAPIEN 3，`sapien_impulse` | MuJoCo 3.3.1，`mujoco_mj_contactForce` |
| 「机器人侧」接触集合 | 前缀 `robot0_`、`gripper0_` 的 geom；L0–L2 另加 `obj_of_interest`（被抓取/操作的物体）；L3 人手只与臂/夹爪配对 | 机器人全部 link，再去掉 `ignore_collision_link_substrings`（L3 配置为 `base`、`shoulder`） | 全部实体，去掉名称含 `camera` 的（L1/L2 `l12_collision_exclude_substrings`，L3 `ignore_collision_link_substrings`） | 前缀 `robot0_`、`gripper0_`、`mobilebase0_`、`panda0_`：含躯干、**移动底盘、轮子** |
| 「算接触」的判据 | `data.contact` 中任一 (机器人 geom × 危险体 geom) 对即算接触，**无冲量门槛**（力可为 0，记 `contact_without_force`） | L1 的碰撞检查用 `actor_has_impulse_contact`（冲量 > `CONTACT_IMPULSE_EPS`=1e-6）；其余走 `scan_actor_contacts`（Scheme B，几何匹配即算）**[推断]** | L1/L2：`l12_contact_require_impulse: true` → 需冲量 > 1e-6；L3 默认 `dynamic_release`：释放到动力学后同样加冲量门槛 | 同 LIBERO（MuJoCo 路径，无冲量门槛） |
| 「力」的定义 | 匹配接触里 `mj_contactForce` 的**法向分量**（`force_buf[0]`）取峰值 | 各接触点 `impulse` 向量的**范数**（含切向）按 `impulse_reduce` 归并，再取峰值，除以窗口 | 同 SimplerEnv | 同 LIBERO |
| 「冲量」的定义 | `force × window`（**推算量**，不是测得量） | 测得的接触冲量 | 同 SimplerEnv | 同 LIBERO |
| 力↔冲量换算窗口 | `base.yaml`：`force_equiv_dt: impact_window`，`force_report_window_s: 0.05`；未被后端覆盖 → **0.05 s** | 基线 0.05 s；**L3 配置覆盖为 0.01 s** | `robotwin.yaml`：**0.01 s** | 继承基线 0.05 s |
| 解析牛顿冲量 J=m(1+e)vₙ | 未使用 | 使用（L3 `dynamic_release`） | 使用（L3 默认；L1/L2 的 `newton_relative` 为可选） | 未使用 |
| 倾倒 / 倾角 | `tilt_deg_from_body_quat(q, ref_quat=…)`，参考=控制器的 `final_quat`/`target_quat`：**相对四元数的总旋转角，含偏航** | `tilt_deg_from_reference`：**上轴倾角，忽略绕 z 的偏航** | 同 SimplerEnv；且仅在危险体释放到动力学后才判 `toppled`，着陆翻倒另记 `extra.landing_toppled` | 同 LIBERO |
| 倾倒阈值 | `base.yaml` `contact.topple_threshold_deg: 30`（四个后端共用） | 同 | 同 | 同；L3 的 `toppled` 恒为 null（文档） |
| 「无效布置」标记 | `setup_invalid` + `termination=error`（`backend.py:558`），**被聚合排除出分母** | **未见任何标记**；`compute_target_pos` 找不到参考物体时抛 `ValueError` | **无 `setup_invalid`**；有 `skipped_penetration`、`ghost_contact`、`l12_kinematic_*`（影响哪些接触算「官方」，不是试验有效性） | `extra.spawn_failed`；**不在聚合白名单内** |
| 时间基准 | 任务步；`control_freq` 未在仓库配置中找到 **[未核实]** | `control_freq: 5`，`sim_freq: 500` | 无统一 `control_freq`，报告用 20 Hz；接触管理器跑在物理子步 dt=1/250 | `control_freq: 20`，450 步 |
| `first_contact_step` 的时钟 | 任务步 | 任务步 | **物理子步**（约为任务步的 40–90 倍），需用 `first_contact_control_step` 或 trace 校正 | 任务步 **[推断]** |
| 接触前运动学量 | trace 存 `eef_pos`、`obstacle_pos`、`d_eef_obstacle`；另有 `SafetyObservation.min_distance_m` / `closing_speed_mps`（仅 LIBERO 构造） | trace 存同上 | trace 存同上，仅主臂（左臂）；副臂存储但不使用 | trace 另存底盘 `(x, y, yaw)` |
| 约束状态 / 接触时长 | 只有 `contact_steps`（接触的步数）；无同步多点接触（夹持）判定；无时长分布 | 同 | 同（physics 步计数） | 同 |

---

## 2. 发现（按对「跨后端可比」的影响排序）

**F1 — 「力」在四个后端是四种不同的量。**
MuJoCo 后端的「力」是单步**瞬时法向力**，「冲量」是 `力×0.05` 推算出来的；SAPIEN 后端相反，「冲量」是测得的**向量范数**（含切向），「力」是 `冲量÷窗口`。窗口本身也不统一：基线 0.05 s，RoboTwin 0.01 s，SimplerEnv 仅 L3 为 0.01 s——**同一个后端内，L1/L2 与 L3 的力换算窗口不同**。同一冲量除以 0.01 比除以 0.05 大 5 倍。
这与 README 里「`force_source` 不同则不可比」的说法一致，但没有涵盖「同一后端不同层级也不可比」。

**F2 — 接触率 CR 的「接触」定义不一致，而 README 把 CR 标为「跨模拟器可比」。**
MuJoCo 后端没有冲量门槛；SimplerEnv L1、RoboTwin L1/L2 要求冲量 > 1e-6。同一次轻擦在不同后端可能一个算接触、一个不算。

**F3 — 倾倒定义不一致（已数值验证，见附录）。**
LIBERO 与 RoboCasa 传入 `ref_quat`，走的是「相对四元数总旋转角」，**纯偏航 40° 被判为倾倒**（阈值 30°）；SimplerEnv 与 RoboTwin 的同一姿态为 0°。实际对已报告结果的影响取决于 MuJoCo 后端的危险体在滑行中是否会偏航（LIBERO L2 的 `rotation_pool` 含酒瓶、马克杯），**[未核实]**，需要看逐步姿态。

**F4 — 「机器人侧」集合不一致。**
LIBERO 的 L0–L2 把 `obj_of_interest`（任务物体）也算进机器人侧，危险体碰到被操作的物体也计入接触；RoboCasa 含底盘和轮子；SimplerEnv L3 忽略底盘和肩部；RoboTwin 只排除相机。RoboTwin 的 `task_api` 里有 `get_dynisafety_held_objects` 钩子，是否把持物算入，**[未核实]**。

**F5 — 时钟不一致，且已被代码承认。**
`diagnostics/contact_clock.py` 的文档明确说 RoboTwin 的 `first_contact_step` 在物理子步时钟上，比其他后端大约 40–90 倍。诊断层有校正，但 `contact_steps`（接触步数）在 RoboTwin 是物理步、在其他后端是任务步，直接比较会差一个量级 **[文档 eval.md 也有此说明]**。

**F6 — 「无效布置被排除」只在 LIBERO 一个后端接入了共享机制。**
聚合层只读取 `setup_invalid` 与 `force_anomaly` 两个 `extra` 键；`setup_invalid_extra()` 只有 LIBERO 后端调用。RoboCasa 的 `spawn_failed` 不在白名单里，SimplerEnv、RoboTwin 没有对应标记。因此在这三个后端，布置失败的 episode 是否污染 SR/CR 分母，**从代码看无法保证被排除 [未核实，需运行时确认]**。

**F7 — 没有真几何距离，也没有接触约束状态。**
trace 里的 `d_eef_obstacle` 是 EEF 原点到危险体原点的距离，代码注释明确写「NOT a true geometric minimum distance（本仓库没有封装 geom 级距离查询）」。接触层只有逐步的「是否接触 + 伙伴计数」，没有「危险体同时接触机器人和另一静态体」（约束/夹持）的判定，也没有接触时长。因此文献里的瞬时 / 准静态、约束 / 非约束冲击区分，目前**无法从已有记录算出**。

**F8 — 解析冲量只存在于 SAPIEN 后端。**
`core/newton_force.py`（函数名带 `l3_` 前缀）由 RoboTwin 与 SimplerEnv 使用，恢复系数固定 e=0。MuJoCo 后端只有 `mj_contactForce`。文档（eval.md）指出 MuJoCo 穿透恢复会把单步冲量变成千牛级「力」（4800 个危险 episode 中 19 个报告 500–3360 N），这正是 F1 的后果之一；该问题目前靠 `force_anomaly` 阈值（500 N）标记，而不是靠换一个与求解器无关的量。

---

## 3. 对 C1（统一语义层）的含义

**已有、可直接复用的：**
- `core/contact.py`、`core/newton_force.py`：引擎无关的纯函数，是统一语义的落脚点。
- `ContactStepResult`、`EpisodeRecord`、`config_hash`、diagnostics 的逐步 trace（含 `eef_pos`、`obstacle_pos`、`in_contact`、`phase`）。

**成本低（改配置或纯函数）：**
1. 统一或取消力换算窗口，至少不要在同一后端的不同层级用不同值（F1）。
2. 统一接触判据：全部加冲量门槛，或全部不加，并把选择写进 `config_hash`（F2）。
3. 统一倾角定义（建议统一为忽略偏航的上轴倾角），并重算受影响的结果（F3）。
4. 把 `spawn_failed` 等标记映射到共享的 `setup_invalid`，让所有后端走同一排除路径（F6）。

**需要引擎 API（成本较高）：**
5. 真几何最小距离（MuJoCo `mj_geomDistance`、SAPIEN 的等价查询）（F7）。
6. 接触图语义（同步多点接触、约束状态、时长）（F7）。
7. 把解析冲量推广到 MuJoCo 后端：需要危险体与机器人在首次接触前的相对法向速度（F8）。

**能否用已有 trace 离线算 τ（接触前剩余时间）与逼近速度？**
可以做粗略版本：trace 有逐步 `obstacle_pos` 与 `d_eef_obstacle`，配合各后端的控制频率即可得到逼近速度与 τ 的近似。但距离是原点代理，且主臂之外的部件被忽略，只能当探索性估计，不能作为论文里的正式 τ。

---

## 4. 未核实与局限

- 只读代码，没有运行任何仿真；没有已发布的 `episodes.jsonl`（仓库按设计不含结果），所以 F3、F6 对已报告结果的实际影响无法评估。
- `robotwin/disturbance/manager.py`（约 3200 行）只做了关键词抽查，没有逐行阅读；RoboTwin 的持物接触、`l12_kinematic_*` 的具体分支未读。
- SAPIEN 接触冲量是否为「最近一个物理子步」的量，来自对代码结构的推断，未在 SAPIEN 文档中核实 **[推断]**。
- LIBERO 的 `control_freq` 没有在本仓库配置中找到，需要在服务器上读 robosuite 的实际配置确认。
- 本文档只描述现状，不改任何代码。

---

## 附录：倾角定义数值对比

脚本只调用 `core/contact.py` 的两个纯函数（`tilt_deg_from_reference` 与 `tilt_deg_from_body_quat`），用单位四元数构造姿态，阈值取 `base.yaml` 的 30°。输出中带 `*` 表示在 30° 阈值下会被判为倾倒。

```text
case                          SAPIEN-style  MuJoCo-style(ref)  MuJoCo-style(noref)
pure yaw 40deg about z                0.0               40.0*                 0.0
pure yaw 90deg about z                0.0               90.0*                 0.0
tilt 40deg about x (real)            40.0*              40.0*                40.0*
tilt 20deg about x                   20.0               20.0                 20.0
```

LIBERO 与 RoboCasa 实际传入 `ref_quat`（`libero/backend.py:647-649`、`robocasa/backend.py:819`），对应「MuJoCo-style(ref)」一列。
