# H2 — RoboCasa365 适配器（`BENCH=robocasa`）

**先读 `H00_host_template.md`，本文件只写 RoboCasa 特有部分。**
起点分支 `hazard/integration`；环境 `dynisafety-robocasa`（Python 3.11，`numpy==2.2.5`，`mujoco==3.3.1`）；上游固定：`robocasa/robocasa@3443cdf`（v1.0.1）。

## 已核实的事实（`02_benchmark_profiles.md` §3）
- `Kitchen(ManipulationEnv)`，`control_freq=20`；`edit_model_xml`（`kitchen.py:1216`）可在加载前改 MJCF；`_post_action`（`:1476`）；`self.fixtures`（`:632`）、`self.objects`（`:853`）；`register_fixture_ref`（`:1747`）；`_check_success`（`:1542`）。
- 本仓库导航后端已用 `edit_model_xml` 包装注入隐藏障碍物：`src/dynisafety/backends/robocasa/env.py:797-854`。
- 机器人 PandaOmron（轮式底盘 + 臂），动作 12 维混合（`hybrid_mobile_12d`）。

## 特有要求

1. **任务范围**：先做 **3 个原子操作任务**（建议含 `PickPlaceCounterToCabinet`，以及抽屉类、门类各一个；类名以源码为准），**不是**导航。现有导航后端不得被破坏（`pytest tests/test_backends_robocasa.py` 前后一致）。
2. **危险体池**：在 `edit_model_xml` 包装里预置（参考 `env.py:797-854` 的做法），停放在场景与相机视野外；碰撞过滤保证停放体不交互。
3. **挂钩**：先在已装的 robosuite 源码里**核实**子步钩子位置（`_pre_action` 是否在每个物理子步被调用；记录文件与行号），再决定包装 `_pre_action` 还是 `sim.step`。物理步长（`model_timestep`）要实测并写入 BD。
4. **支撑面与固定装置**：`support_surfaces` 从 `get_fixture(...)` 读出台面、柜门、抽屉的 AABB / 高度；不得假设桌面平坦或高度为常数。结果写进 BD 的 `state_access` 并在汇报里描述读取方式。
5. **机器人侧碰撞体**：前缀 `robot0_`、`gripper0_`、`mobilebase0_`、`panda0_`（含底盘与轮子）；抓持物判定（`held_object`）如何声明式给出，写入汇报。
6. **接触**：可复用 `dynisafety.backends._engine.mujoco_contact`（本环境 Python 3.11，且已装 `dynisafety` 时可 import；否则自行用 `hazardlib` 内的最小实现并在汇报中说明）。**真几何距离**用 `mujoco.mj_geomDistance`，实测。
7. **`hold_action`**：零臂增量、零底盘速度、保持夹爪；需实测机器人是否漂移。
8. **按 (任务, 布局种子) 探针**：布局由种子程序化生成；探针运行必须按种子；至少 5 个种子 × 3 个任务。
9. **探针素材**：RoboCasa 演示 / MimicGen 数据的获取方式与大小先在 `env_notes` 里调研并**先问用户**再下载；若暂时拿不到，用脚本化的零 / 保持运行 + 少量手写探针轨迹（明确标注为临时，不得当正式探针集）。
10. **资产约 10 GB**：已由 E00 处理；本任务不再重复下载。

## 额外验收
- 现有 `tests/test_backends_robocasa.py` 的非仿真部分前后结果一致。
- 汇报中说明：3 个任务的支撑面读取方法是否通用于其他原子任务（有证据则列出，否则标未核实）。
