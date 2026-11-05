# R1 — 基准能力对照（Benchmark Descriptor 草稿）

对象：RoboDojo（`f57e163`，2026-10-21）、RoboCasa365 / robocasa（`3443cdf`，v1.0.1，2026-10-30），以及本仓库已有的后端。
方法：浅克隆后读源码（RoboDojo 的 `XPolicyLab`、`IsaacLab`、`curobo` 是空子模块，**未读**）。未运行任何仿真。
标注：**[已核实]** 在源码里直接看到；**[推断]** 由结构推出；**[待探针]** 必须在装好环境的服务器上实测（见第 4 节）。

---

## 1. 能力对照表

| 描述符字段 | 本仓库 LIBERO 后端（参照） | RoboCasa365（操作任务） | RoboDojo |
|---|---|---|---|
| 引擎 / 版本 | MuJoCo（robosuite） | MuJoCo 3.3.1，`numpy==2.2.5`（`setup.py`）**[已核实]** | Isaac Sim 5.1 + Isaac Lab 2.3，Python 3.11（README 徽章）**[已核实]** |
| 物理步长 | — | robosuite 模型步长，每个控制步内多个子步（`kitchen.py:1146` 附近）**[已核实]** | `dt: 0.004`，`decimation: 1`（`env_cfg/sim/sim_config.yml`）**[已核实]** |
| 控制频率 | 见 P0 | `control_freq=20`（`kitchen.py:381`）**[已核实]** | 一个策略动作展开为一串 `ControlSeq`，循环 `step()` 直到队列清空（`eval_env.py:446-452`），每个 `step` 一个物理步 **[已核实]**；动作对应的物理步数 **[待探针]** |
| 并行 | 单环境 | 单环境（robosuite） | 同进程多环境，默认 `num_envs: 10`、`env_spacing: 7`，每个环境可有不同物体与布局 **[已核实]** |
| 场景表示 | BDDL 加手工障碍物 | 程序化厨房：`self.fixtures`（`kitchen.py:632`）、`self.objects`（`:853`），固定装置按语义类型注册（`FixtureType.COUNTER` 等）**[已核实]** | 场景 YAML（物体类别、`xlim/ylim`、相对平面 `Table`）加 Python 任务类；USD 资产 **[已核实]** |
| 任务相关实体来源 | 靠探针推断 | 任务类里的命名引用（如 `self.cab`、`self.counter`、`obj`），`_get_obj_cfgs` 声明放置 **[已核实]** | YAML 的标签（`bowl0..2`），`run_reward()` 里按标签引用 **[已核实]**——可静态解析 |
| 支撑面 / 工作区边界 | 桌面常量 | `get_fixture(FixtureType.COUNTER, ref=…)`，台面高度需从固定装置读取 **[待探针]** | YAML 的 `xlim/ylim` 加 `Table` 的位置与尺寸（`default.yml`）**[已核实]** |
| 危险体生成方式 | 预置隐藏障碍物（SafeLIBERO） | **加载前改 MJCF**：`kitchen.py:1216` 的 `edit_model_xml`；本仓库导航后端已用此法注入隐藏障碍物（`backends/robocasa/env.py:797-854`）**[已核实]** | **预置并移到屏外**：场景对象有 `relocate_offscreen()`，把位置设到约 (100000, 100000, 100000)（带 ±1000 抖动）的远处（`rigid.py` / `geometry.py` / `dynamic.py` / `table.py`）**[已核实]**；运行时新建 prim 的可行性 **[待探针]** |
| 运动学摆放 | 写 free-joint qpos | 同左（本仓库已实现）**[已核实]** | `set_local_pose`（逐环境）**[已核实]**；传送式运动在 PhysX 下是否产生异常冲量 **[待探针]** |
| 释放到动力学（设初速度） | 已实现 | 写 qvel **[推断]** | 仓库内**没有**调用速度写入接口（`set_velocities` 等零命中）**[已核实]**；Isaac Sim 刚体视图的速度接口能否用 **[待探针]** |
| 接触信息 | `mj_contactForce`（P0） | 本仓库 `_engine/mujoco_contact.py` 可复用 **[已核实]** | 物体物理配置里有 `prepare_contact_sensor`、`contact_filter_prim_paths_expr` 开关，**默认关**（`geometry.py:73-102`）；机器人侧 `activate_contact_sensors=False`（`robot_config/franka.py:12`）**[已核实]**；接触读数与相对速度 **[待探针]** |
| 真几何距离 | 无（P0 F7） | `mj_geomDistance` 可用 **[推断]** | Isaac 内的等价查询 **[待探针]** |
| 机器人状态读取 | EEF 位姿 | EEF 位姿，底盘 `(x, y, yaw)` **[已核实]** | `robot_manager` / `obs_manager`；双臂 ARX X5，重力关闭（`x5.py:13`）**[已核实]** |
| 成功判定 | Python | `_check_success()`（`kitchen.py:1542`），按任务子类实现 **[已核实]** | `run_reward()` 调 `reward_manager.check([...])`；`is_episode_end` 判定，`step_lim` 在任务类（如 `stack_bowls`=800）**[已核实]** |
| 复位与种子 | 种子加 episode 下标 | 场景由种子程序化生成 **[推断]** | 种子 == 布局编号（预收集的 `Eval_Layout`）；跨 GPU 有微小非确定性（论文）**[已核实]** |
| 无效样本机制 | `setup_invalid`（P0） | 无共享机制 **[推断]** | `layout_valid` → 失败并 `episode_nums -= 1`；`unstable_envs` → 计入 `unstable_nums`、不进评测总数 **[已核实]**；PhysX 异常监控与重启恢复（`physx_warning_monitor.py`）**[已核实]** |
| 策略接入与回合循环 | 本仓库 adapter | LeRobot 集成（搜索摘要）**[未读]** | 回合循环在策略侧：`XPolicyLab.policy.<name>.deploy.eval_one_episode[_batch](TASK_ENV, model_client)`，仿真侧暴露 `take_action[_batch]`、`get_obs[_batch]`、`is_episode_end`；WebSocket 传输 **[已核实]** |
| 不改上游的挂钩点 | — | 子类化或包装 `_post_action` / `edit_model_xml` **[已核实]** | `create_eval_env` 经 `task_registry.load_task_class` 取任务类再 `class EvalEnv(task_class)`（`eval_env.py:43-52`，`main.py:346`）：自备启动脚本，在加载前给任务类套混入，并在 `step()` 内每物理步调用 `advance` **[已核实]** |
| 许可与治理 | MIT | 未核实 | **MIT**（`LICENSE`，README）；仿真侧完整开源、允许本地调试；**官方榜单**需走云评测，本地衍生结果不上榜 **[已核实]** |
| 环境互斥 | numpy<2 | numpy 2.2.5 + mujoco 3.3.1 | `scripts/install.sh`：conda 环境 `RoboDojo`、`numpy==1.26.0`、`torch==2.7.0` cu128 **[已核实]** |

> 更正：此前基于检索摘要，我写过 RoboDojo 仓库页显示「非商业」许可。读 `LICENSE` 与 README 后确认是 **MIT**，该说法不成立，此项风险撤销。

---

## 2. 对 HazardHost 契约与危险规格的影响

1. **规格里的时间必须用秒，不用步数。** RoboDojo 的最小推进粒度是 0.004 s 的物理步，RoboCasa 是 0.05 s 的控制步；同一危险（如 τ=0.5 s）在两边要由适配器采样成不同步数。这与 P0 的 F5（时钟不一致）一致，也是 τ 作为强度轴的前提。
2. **`advance(t)` 的粒度要取引擎能提供的最细粒度。** 快速运动的危险体在 0.05 s 的粒度下可能穿透（隧穿），需要在适配器里按子步插值或做连续碰撞检查；这要在探针里实测。
3. **必须支持批量与逐环境状态。** RoboDojo 同进程多环境，每个环境有各自的 `end_flag`、`take_action_cnt`、危险时间表。HazardHost 的方法应以环境下标为参数。
4. **无效样本可以映射到现有机制。** `setup_invalid` ↔ `layout_valid` / `unstable_envs`；**但**危险体引起 PhysX 异常会被监控判为「不稳定」而从评测总数中剔除，这会静默地偏置结果，必须单独计数并报告。
5. **RoboDojo 的 `RigidObject` 把质量截断到 ≤0.5 kg**（`rigid.py`：`min(mass, 0.5)`）。危险体若沿用这个类，质量与动能会被压低；另起一个类则偏离其资产管线。这需要作为设计取舍写明。
6. **任务相关实体可以更多地「静态读出」。** RoboDojo 的 `run_reward()` 按标签引用物体，RoboCasa 的任务类里有命名的固定装置引用；探针回放仍需要（用于占用分布和阶段），但实体识别不必全靠回放。
7. **策略回合循环不在仿真侧。** 所以危险体不能挂在「回合开始 / 结束」的循环里，只能挂在 `step()`（物理步）、`reset()`、`is_episode_end()` 上。

---

## 3. 适配成本的初步判断（待探针校正）

| 目标 | 预期难度 | 主要依据 |
|---|---|---|
| RoboCasa365 操作任务 | 低到中 | 同引擎家族；本仓库已有 MJCF 注入、free-joint 摆放、MuJoCo 接触封装；难点是台面与柜体等支撑面的识别 |
| RoboDojo | 中到高 | 新引擎、批量接口、接触传感器默认关、速度写入接口在仓库内无先例；但有「屏外预置」先例，挂钩点干净，许可友好 |

---

## 4. R5 引擎探针清单（需在服务器上实测，Isaac Sim 装好后进行）

每条给出「问题 / 通过标准」。脚本等环境装好、`robodojo.sh doctor` 通过后再写，避免对着没验证过的接口写盲代码。

| # | 问题 | 通过标准 |
|---|---|---|
| E1 | 能否在 `reset` 后向某个环境放置一个预置的危险体（从屏外移入）并逐物理步移动 | 位姿按预期变化，其余环境不受影响 |
| E2 | 传送式运动经过机器人时，PhysX 是否产生异常冲量、是否触发 `physx_warning_monitor` | 不触发「不稳定」判定；记录冲量量级 |
| E3 | 能否给刚体设初速度并释放到动力学 | 危险体按设定速度运动，落地后行为合理 |
| E4 | 开启 `prepare_contact_sensor` 与 `contact_filter_prim_paths_expr` 后，能否读到危险体与机器人的接触及法向相对速度 | 逐步读数与几何事实一致；对吞吐的影响可接受 |
| E5 | 有无危险体到机器人的真几何最小距离 | 有则记录接口；无则标记「仅近似」 |
| E6 | 同一种子、同一布局编号重复复位，初始状态是否一致（含跨进程） | 记录差异量级，决定配对设计是否只能用统计配对 |
| E7 | 一个动作展开的物理步数是多少（不同机器人 / 动作类型） | 得到秒与步的换算表 |
| E8 | 具身 T_stop：发保持动作后 EEF 速度降到阈值以下所需时间 | 得到可复现的数值，用于 Σ=τ/T_stop |

RoboCasa 侧对应探针（E1–E8 的 MuJoCo 版本）可在 `dynisafety-robocasa` 环境里做；其中 E4、E5 基本可由现有封装直接完成。

---

## 5. 环境安装提示（RTX PRO 6000）

- Isaac Sim 5.1 官方要求页把 RTX PRO 6000 Blackwell 列为支持的 GPU；该页给出的 Linux 驱动为 **580.65.06**，系统为 Ubuntu 22.04 / 24.04（检索结果，**以官方页为准**）。
- RoboDojo 的 `scripts/install.sh` 固定 `torch==2.7.0`（cu128 索引）、`numpy==1.26.0`、conda 环境名 `RoboDojo`（Python 3.11）。cu128 与 Blackwell 的常见要求一致，但这是我的背景知识，**待在你们的机器上确认**。
- 资产来自 Hugging Face 数据集 `RoboDojo-Benchmark/RoboDojo`（`Robots`、`Object`、`Material`、`Eval_Layout`），`scripts/init_assets.sh` 下载。
- `IsaacLab`、`curobo`、`XPolicyLab` 是子模块，需 `git submodule update --init --recursive`。IsaacLab 与 curobo 指向的是 `yuechen0614` 的 fork。
- 与本仓库的环境关系：RoboDojo 需要 numpy 1.26，本仓库主包要求 `numpy<2.0`，**可兼容**；RoboCasa 需要 numpy 2.2.5，仍须单独环境（与现有约定一致）。

---

## 6. 局限

- 只读了 RoboDojo 与 RoboCasa 的部分文件（环境骨架、评测客户端、配置、一个示例任务）；`reward_manager`、`robot_manager`、`scene_manager` 的主体、XPolicyLab 均未读。
- RoboCasa365 的策略接入（LeRobot）未读。
- 「Isaac Sim 的接口能否做到 X」全部归入待探针，没有依据背景知识下结论。
