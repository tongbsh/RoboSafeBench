# 六个基准的接入档案

对象：原始 LIBERO（D18）、SimplerEnv、RoboTwin、RoboCasa365、RoboDojo、CALVIN。前置：读 `00_paradigm.md` 与 `01_integration_rules.md`。
证据来源：前四个来自本仓库代码与 P0 审计；RoboDojo、RoboCasa 来自 `docs/audit/r1_benchmark_capability.md`；CALVIN 来自对 `mees/calvin`（`fa03f01`）与 `mees/calvin_env`（`797142c`）的源码阅读。
标注：**[已核实]** 源码里直接看到；**[推断]**；**[待探针]**。所有 `[待探针]` 项必须由对应适配器任务在服务器上消除，不得凭背景知识当事实使用。

---

## 1. 总览

| 基准 | 引擎 | 环境 / Python | 时基 | 评测循环归属 | 预期 Tier | 预期接入难度 |
|---|---|---|---|---|---|---|
| 原始 LIBERO | MuJoCo（robosuite） | `dynisafety-libero`；Python 3.10 | 20 Hz；物理步 0.002 s（H3 实测） | 我们自己的脚本 | A（C1–C13 已验收） | 内存 MJCF 注入，保留原始任务 |
| SimplerEnv（bridge） | SAPIEN 2.2.2 + ManiSkill2 real2sim | 既有 SimplerEnv 环境 | `control_freq=5`，`sim_freq=500`（`bridge.yaml`）**[已核实]** | 我们自己的脚本 | A | 低到中（控制步长 0.2 s，须子步挂钩；真实背景叠加） |
| RoboTwin（main20） | SAPIEN 3.0.0b1 | 既有 RoboTwin 环境 | 物理 1/250 s；无统一 `control_freq` **[已核实]** | 我们自己的脚本 | A | 中到高（逐任务钩子与主臂规则 → 注解预算） |
| RoboCasa365 | MuJoCo 3.3.1，`numpy==2.2.5` | `dynisafety-robocasa`，Python 3.11 **[已核实]** | `control_freq=20`（`kitchen.py:381`）；物理步长 **[待探针]** | robosuite 环境，我们自己的脚本 | A | 低到中 |
| RoboDojo | Isaac Sim 5.1 + Isaac Lab 2.3（PhysX） | conda `RoboDojo`，Python 3.11，`numpy==1.26.0`，`torch==2.7.0` cu128 **[已核实]** | `dt=0.004`，`decimation=1`，一个动作展开为多个物理步 **[已核实]**，步数 **[待探针]** | **策略侧**（XPolicyLab 的 `eval_one_episode`）| A **[待探针]** | 高 |
| CALVIN | PyBullet | `calvin_venv`，**Python 3.8** **[已核实 README]** | 物理 240 Hz，控制 30 Hz，每动作 8 个子步 **[已核实]** | 用户脚本（`evaluate_policy.py`），官方留有环境包装器挂钩 **[已核实]** | A | 低 |

五种引擎家族：MuJoCo（LIBERO、RoboCasa）、SAPIEN 2（SimplerEnv）、SAPIEN 3（RoboTwin）、PhysX/Isaac Sim（RoboDojo）、PyBullet（CALVIN）。

**建议接入顺序：** CALVIN → RoboCasa → LIBERO → SimplerEnv → RoboTwin；RoboDojo 的环境搭建与 R5 探针并行启动，适配器最后做。理由：CALVIN 引擎最简单（运行时生成、真几何距离、可精确恢复），最适合第一个走通完整契约。

---

## 2. CALVIN

**结构（已读源码）。**
- `PlayTableSimEnv`（`calvin_env/envs/play_table_env.py`）：`bullet_time_step=240`，`control_freq=30`，`action_repeat = int(240 // 30) = 8`；`step()` 先 `robot.apply_action`，再循环 `p.stepSimulation` 8 次，然后 `scene.step()`、`get_obs()`、`get_info()`，返回 `(obs, 0, False, info)`——**没有奖励与终止**，成功由外部判定器给出。
- 评测（`calvin_models/calvin_agent/evaluation/evaluate_policy.py`）：`EP_LEN=360` 步（12 s）每个子任务，`NUM_SEQUENCES=1000` 个序列，每序列至多 5 个子任务；`make_env` 里有注释“insert your own env wrapper”；环境在序列内**不**逐子任务重置，只重置模型。初始状态通过 `env.reset(robot_obs, scene_obs)` 从验证集给定，确定性。`get_env` 构造环境时设 `use_scene_info=True`。
- 任务判定：`Tasks`（`envs/tasks.py`）是**声明式**的——`conf/tasks/new_playtable_tasks.yaml` 里每个任务是对 `start_info` / `end_info` 的状态差分函数调用（如 `rotate_object 'block_red' -60`、`move_door_rel 'base__drawer' 0.12`、`lift_object 'block_red' 0.05 'table' 'base_link'`）。→ **任务相关实体可从判定器配置静态读出**。
- 场景：`conf/scene/calvin_scene_*.yaml` 声明 `surfaces`（`table`、`slider_left`、`slider_right`，各为 AABB 的两个角点）、固定物体（桌子，带抽屉 / 滑门 / 按钮 / 开关 / 灯）与可移动物体（三个方块，`initial_pos: any`）。
- 机器人：Franka Panda，7 维相对位姿动作；`use_target_pose: true`，`relative_to_absolute` 对**累计目标位姿**积分，所以零动作不会立刻刹停，会收敛到累计目标。
- 状态：`scene.get_obs()` 只拼接已注册的门 / 按钮 / 开关 / 灯与 `movable_objects` 的位姿，`get_info()` 只遍历固定与可移动物体 → **未注册的额外 Bullet 刚体不会进入 `scene_obs` 与任务判定**（R5 天然满足，仍需 C7 验证）。机器人观测里的 `contacts` 来自 `p.getContactPoints(bodyA=robot)`，会包含与危险体的接触，需要确认任务判定器不受其影响。
- 可恢复：`serialize()` / `reset_from_storage()` 对机器人与场景可精确恢复 **[已核实]**；`setPhysicsEngineParameter(deterministicOverlappingPairs=1)` 已设 **[已核实]**。

**HazardHost 映射。**

| 方法 | 计划 |
|---|---|
| `alloc_hazard` / `free_hazard` | 运行时 `loadURDF` / `createMultiBody` 随时可建（`play_table_scene.load` 中已大量使用 `loadURDF`）**[已核实]**；危险体不注册进 `scene` 的物体列表 |
| `set_hazard_pose` | `resetBasePositionAndOrientation`；运动学化通过质量置零或约束 **[待探针]** |
| `set_hazard_visible` | 停放到工作区与相机视野外 |
| `release_to_physics` | 恢复质量 + `resetBaseVelocity` **[待探针]** |
| `hazard_contacts` | `getContactPoints(hazard, robot)`（该 API 在仓库内已被使用）**[已核实]**；相对速度由核心用运动学补算，机器人连杆速度用 `getLinkState(computeLinkVelocity=1)` **[待探针]** |
| `min_distance` | `getClosestPoints` 真几何距离 **[待探针]** |
| `snapshot` / `restore` | 用 `serialize` / `reset_from_storage`，危险体由运行时重建 |
| `robot_state` | `robot.get_observation()`（TCP 位姿、夹爪开度）**[已核实]** |
| `object_poses` | 场景物体 `get_state()`；`surfaces` 来自场景 YAML **[已核实]** |
| `hold_action` | 7 维零相对动作 + 当前夹爪指令；需单独标定 T_stop，因目标位姿积分 |
| `task_success` | `Tasks.get_task_info_for_set(start_info, current_info, {subtask})` |
| `install_hooks` | 子类 / 包装 `PlayTableSimEnv.step`：在 8 次 `stepSimulation` 前后调用运行时；复制了上游 6 行代码，故**固定 `calvin_env@797142c` 并在 import 时断言**（R13） |

**特别注意。**
1. **Python 3.8**：`hazardlib` 必须在 3.8 下安装并通过测试（R0）。
2. 危险体生命周期按**子任务**管理：序列内环境不重置，上一个子任务的危险体必须在子任务边界清除。
3. 探针来源：CALVIN 演示数据（`dataset/download_data.sh debug` 约 1.3 GB）、验证集的初始状态与语言标注；TaskContext 的角色来自判定器配置。
4. 参考策略 / G2：无现成特权避障，G1 先用演示回放 + Oracle；G2 暂标 `uncertified`。
5. 许可：MIT（README 徽章）。
6. **重力为 −9.8**（`calvin_env/envs/play_table_env.py:133`，`setGravity(0, 0, -9.8)`）**[已核实]**，与其他基准常用的 9.81 不同；fall 族的 τ 反解必须用宿主实测值（`00_paradigm.md` D2）。PyBullet 动态体默认阻尼 **[待探针]**。

---

## 3. RoboCasa365

**结构（已读源码，`robocasa/robocasa@3443cdf`，v1.0.1）。** `Kitchen(ManipulationEnv)` 基类；`control_freq=20`、`horizon=1000` 默认；任务类按 `_get_obj_cfgs`（声明放置）、`_setup_kitchen_references`（`register_fixture_ref`，如 `self.cab`、`self.counter`）、`get_ep_meta`（语言）、`_check_success` 定义；`self.fixtures`（`kitchen.py:632`）与 `self.objects`（`:853`）；`edit_model_xml`（`:1216`）加载前改 MJCF；`_post_action`（`:1476`）；原子任务分布在 15 个模块、组合任务约 362 个文件；依赖 `numpy==2.2.5`、`mujoco==3.3.1`。

| 方法 | 计划 |
|---|---|
| `alloc_hazard` | 在 `edit_model_xml` 里预置危险体池（本仓库导航后端已用此法：`backends/robocasa/env.py:797-854`）**[已核实]** |
| `set_hazard_pose` / `release_to_physics` | 写 free-joint 的 `qpos` / `qvel`（本仓库已实现位姿写入）|
| `hazard_contacts` | 复用 `backends/_engine/mujoco_contact.py`（需 Python ≥3.10 且已装 `dynisafety`，RoboCasa 环境满足）**[已核实]** |
| `min_distance` | `mujoco.mj_geomDistance` **[推断]** |
| `snapshot` / `restore` | `sim.get_state()` / `set_state()` **[推断]** |
| `robot_state` | EEF 位姿、底盘 `(x, y, yaw)`（本仓库已有）**[已核实]**；`robot_contact_bodies` 含底盘与轮子（前缀 `robot0_`、`gripper0_`、`mobilebase0_`、`panda0_`）**[已核实]** |
| `object_poses` | `self.objects` + `get_fixture(...)`；台面高度需从固定装置读取 **[待探针]** |
| `task_success` | `_check_success()` **[已核实]** |
| `hold_action` | PandaOmron 的 12 维混合动作（`hybrid_mobile_12d`），零臂增量、零底盘速度、保持夹爪 **[待探针]** |
| `install_hooks` | robosuite 的 `step` 在每个物理子步调用 `_pre_action` **[推断，待在已装包里核实]**；备选为包装 `sim.step` |

**特别注意。** 场景由种子程序化生成，TaskContext 要**按 (任务, 布局种子)** 探针；支撑面（台面、柜门、抽屉）要从固定装置读出而不是假设桌面；移动底盘的暴露区取「臂 ∪ 底盘」扫掠体；资产约 10 GB；与现有导航后端共存（导航后端不得被破坏）。探针来源：RoboCasa 演示 / MimicGen 数据（搜索摘要，**[待探针]** 下载与格式）；策略可得性 **[待调研 X1]**。

---

## 4. RoboDojo

结构、挂钩点、限制见 `docs/audit/r1_benchmark_capability.md`，此处只列接入决定。

| 项 | 决定 / 计划 |
|---|---|
| 挂钩 | 自备启动脚本：在 `create_eval_env` 之前给 `task_registry.load_task_class` 套混入（`eval_env.py:43-52`、`main.py:346`）；危险体推进放进 `EvalEnv.step(env_idx_list, decimation)`，每个物理步（0.004 s）**[已核实]** |
| 回合循环 | 在策略侧（`XPolicyLab.policy.<name>.deploy.eval_one_episode[_batch]`）；仿真侧只暴露 `take_action[_batch]`、`get_obs[_batch]`、`is_episode_end`。适配器**禁止**依赖回合循环 |
| 危险体池 | 在场景配置里追加额外物体并用 `relocate_offscreen()` 停放；配置在 OmegaConf 中于内存里补丁，不改上游文件 **[待探针 E1]** |
| 位姿 | `set_local_pose`（逐环境）；传送式运动在 PhysX 下是否异常 **[待探针 E2]** |
| 速度 / 释放 | 仓库内无速度写入先例；Isaac 刚体视图速度接口 **[待探针 E3]** |
| 接触 | 物体物理配置的 `prepare_contact_sensor` + `contact_filter_prim_paths_expr`（默认关）；机器人侧 `activate_contact_sensors=False` **[已核实]**；读数与相对速度 **[待探针 E4]** |
| 真几何距离 | **[待探针 E5]**；无则 `surrogate` |
| 确定性 | 论文称跨 GPU 微小非确定性 → 声明 `statistical`，配对只用统计配对 **[待探针 E6]** |
| 无效样本 | `layout_valid` / `unstable_envs` 映射到 `setup_invalid` / `sim_unstable`；危险体引起的不稳定单列 `hazard_induced_unstable` |
| 任务相关实体 | `run_reward()` 按标签引用物体，YAML 声明标签 → 静态读出 **[已核实]** |
| 质量 | `RigidObject` 把质量截断到 ≤0.5 kg（`rigid.py`）。危险体若沿用则动能受限；**[决定待定]**：另起不截断的类，并在汇报中说明偏离 |
| G2 | `env/planner_manager/curobo_planner.py` 仅见机器人碰撞球与自碰撞配置，grep 未见世界障碍物；能否加世界碰撞 **[待探针]**，否则标 `uncertified` |
| 批量 | 同进程默认 10 个环境，危险体状态逐环境；C12 必测 |
| 许可 | MIT（`LICENSE`）；本地衍生结果不上官方榜单 |
| 环境 | 用户自行安装 Isaac Sim 5.1（RTX PRO 6000；驱动 580.65.06；Ubuntu 22.04 / 24.04，以官方页为准）；`doctor` 通过后开始 |

---

## 5. 原始 LIBERO（D18，2026-11-10）

- 四个套件：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，各 10 任务；保持官方顺序，任务 id = `<suite>/<官方任务名>`。
- 上游 `Lifelong-Robot-Learning/LIBERO@8f1084e`；与旧 SafeLIBERO 代码分开，不把套件改名为 `safelibero_*`。
- 原始 BDDL / 物体 / 相机 / 成功谓词保持；在内存 MJCF 的 worldbody 末尾追加独立危险体，自由关节状态按名字映射，停放时关碰撞并补偿重力。
- 时钟、接触与距离：每个 `MjSim.step` 前后挂钩；20 Hz 控制、0.002 s 物理步；scratch MjData 查询几何接触；`mj_geomDistance` 真几何间距。
- 机器人侧只含机器人，抓持物单列；BDDL 的 `obj_of_interest` 与 `goal_state` 可静态读出，不把全部任务物体当作机器人。
- 探针为官方 HDF5 演示的开环回放，布局取同集初始状态（D16）；正式双来源要求仍未满足，试点显式 `min_source_kinds: 1`。
- G1 用参考演示的停等包装；G2 尚无该新宿主的完整 AEGIS 桥接，失败的 G1 标 `uncertified`。
- C1–C13：12 pass / 1 skip（C12 单环境）；证据见 H3 汇报。旧 SafeLIBERO 手工 L1–L3 与本协议无直接一一对照，历史结论不重算。

## 6. SimplerEnv（bridge）

- 危险体：预置的隐藏 actor（`HIDDEN_POS`），`set_actor_pose` 摆放，运动学转动力学释放（`needs_physics_release`）**[已核实]**。
- 接触：`backends/_engine/sapien_contact.py`（`scan_actor_contacts`、`first_actor_contact_kinematics` 含法向与接触位置）**[已核实]**。
- **子步挂钩是关键风险：** 控制 5 Hz = 每动作 0.2 s，仿真 500 Hz = 100 子步。危险体以 1 m/s 运动，每个控制步位移 20 cm，远超 R1 的位移上限。必须挂到 ManiSkill2 `step` 内部的子步循环 **[待探针：具体位置]**。
- 实体：`get_reference_object_actor` **[已核实]**；`target_pos: null` 已是自动模式。
- 视觉：用真实背景图叠加（`rgb_overlay`）→ C9 / R16 必须验证危险体被正确合成 **[待探针]**。
- 探针：无演示；只能用脚本 / 参考策略的 L0 回放，TaskContext 质量受限，需在汇报中如实说明。
- 成功：ManiSkill2 的 `evaluate()` 中的 `success` **[推断]**。

## 7. RoboTwin（main20）

- 现状（违反 R2）：20 个任务各有 `target_mode`、`primary_arm`（覆盖文件）；6 个可选钩子 `get_dynisafety_*`（`task_api.py`），由补丁 `patches/robotwin/dynisafety_hooks.patch` 注入 **[已核实]**；`task.object` 约定为默认路径。
- 新范式处理：把 `primary_arm` / 角色转成 `configs/hazard/annotations/robotwin.yaml`（声明式），统计行数作为注解预算；约定路径不满足的任务标 `unsupported`，不写特例。
- 时钟：接触管理器跑在物理子步（1/250 s），`first_contact_step` 与他处不同钟（`diagnostics/contact_clock.py`）→ 新记录统一用秒。
- 探针：专家数据 / 脚本专家（仓库有 `expert_cache.py`，**内容未读**）。
- 既有 `manager.py`（约 3200 行）包含 `kinematic_first_hit` / `newton_relative` / `dynamic_release` 等方案，是统一语义的主要来源；新适配器**不得**复制它，只实现 HazardHost 原语。

---

## 8. 探针与参考控制器可得性（用于 G1 / G2 / 探针集）

| 基准 | 演示 | 脚本专家 | 参考策略 | G1（Oracle 包装） | G2（特权避障） |
|---|---|---|---|---|---|
| LIBERO | 有 **[推断]** | 否 | openvla_oft、pi05 等（本仓库已有适配）| 有 | AEGIS + 桥接 |
| SimplerEnv | 否 | 否 | starvla、xvla_official、internvla_m1 | 需包装 | 暂无 |
| RoboTwin | 有（专家）| 有 | internvla_a1、fastwam、lingbot_va、starvla、xvla_official | 需包装 | 暂无 |
| RoboCasa | 有（HF 数据）**[待探针]** | 否 | 暂无（`random_mobile`）| 需包装 | 暂无 |
| RoboDojo | 有（HF 数据，约 3,500 条）| 数据生成用 cuRobo 规划 **[待探针]** | XPolicyLab 约 40 个策略，检查点可得性 **[待调研 X1]** | 需包装 | 待探针 |
| CALVIN | 有（play 数据）| 否 | 多个公开方法，检查点可得性 **[待调研 X1]** | 需包装 | 暂无 |

---

## 9. 开放问题（须在汇报中回答，不得默认）

1. 各基准的 `hold_action` 与 T_stop 能否稳定测出（C10）。
2. SimplerEnv 的子步挂钩位置与叠加合成。
3. RoboDojo E1–E8 的结果，特别是 E2（传送式运动是否触发 PhysX 异常监控）。
4. 在 CALVIN 与 RoboCasa 上，机器人侧碰撞体集合与“抓持物”判定是否能用声明式方式给出。
5. 哪个基准作为合约冻结后的**盲留出**（`00_paradigm.md` §11）。
