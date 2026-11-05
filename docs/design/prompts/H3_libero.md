# H3 — LIBERO（SafeLIBERO）适配器（`BENCH=libero`）

**先读 `H00_host_template.md`，本文件只写 LIBERO 特有部分。**
起点分支 `hazard/integration`；环境 `openvla-oft`（README 约定）；本任务有**最多可复用的既有代码**，也是**手工 L1–L3 对照基线**所在，所以回归安全最重要。

## 已核实的事实（`02_benchmark_profiles.md` §5）
- 危险体来源：SafeLIBERO 场景里手工放置的障碍物，名字含 `obstacle`，停放在 |x|、|y|>5 之外（`backends/libero/disturbance/_support.py: detect_visible_obstacles`）。
- 接触：`backends/_engine/mujoco_contact.py`；`backends/libero/contact.py` 的机器人侧集合含 `obj_of_interest`（任务物体）。
- G1 / G2 的既有部件：`policies/safety/oracle.py`、`policies/safety/aegis/*`、`backends/libero/safety/*`。
- 手工 L1–L3 的落点来自 SafeLIBERO 作者放置，时机来自 `onset_schedule.py`（阶段分层）与反应式触发；`phase_anchors` 的生成脚本不在仓库中。

## 特有要求

1. **回归保护**：本任务开始前与结束后各运行一次 `pytest tests -q -m "not sim"`（以实际标记为准），两次结果必须一致；`git diff --stat origin/hazard/integration -- src/dynisafety` 必须为空。
2. **危险体池**：把 SafeLIBERO 的预置障碍物当作**池**，不使用其作者放置的落点与时机——位置与时机全部由 `hazardlib` 实例化；`alloc_hazard` 认领一个池成员，`set_hazard_visible(False)` 恢复停放。同一任务里池成员的形状 / 质量要在 BD 中声明（不能假设可自定义尺寸；若不可自定义，实例化的 `agent` 需约束到池成员，并如实记录这一限制）。
3. **接触语义改动**：新宿主的 `robot_contact_bodies` 不含任务物体（只含机器人；抓持物标 `held_object`）。这与旧后端不同，**必须在汇报中列出受影响的旧结论方向**，但不重算旧结果。
4. **实体**：任务物体名来自既有配置的 `task_object_names`；同时核实 BDDL 目标谓词能否静态读出任务相关实体（给出读取方式或结论“不行”）。**不得**用 `phase_anchors` 当探针结果；应自行做无危险回放得到阶段边界。
5. **控制频率与子步**：实测 `control_dt`、`physics_dt`、每动作子步数，写入 BD；确认在 robosuite 的哪个函数里可以挂子步钩子（记录文件与行号）。
6. **范围**：SafeLIBERO 的 `spatial` 套件 4 个任务即可，另加 `goal` 与 `long` 各 1 个任务，用来暴露旧代码里逐任务调参（`l1_home_workspace_onset` 等）的位置；汇报里列出这些旧调参在新范式下由什么机制取代（或无法取代）。
7. **旧手工 L1–L3 的对照**：在汇报里给出一张映射表：旧 L1 / L2 / L3 ↔ 新的 `intrusion.fall / slide / reach` 在强度轴上的近似位置（用实测的 τ 估计旧设计的名义可用反应时间）。这张表是 H1 假设（强度轴能否复现手工排序）的前置输入，**只估计，不下结论**。
8. **G1 / G2**：G1 用 `OraclePolicyWrapper` 包装演示回放（或一个参考策略）；G2 评估 AEGIS 能否作为 `PrivilegedAvoider` 接入（只做接口可行性评估，不做大规模运行）。

## 额外验收
- 回归检查前后一致（贴两次输出的差异）。
- 汇报里的“旧 L1–L3 → 新强度轴”映射表附原始数据路径。
