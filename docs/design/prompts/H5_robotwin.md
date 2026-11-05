# H5 — RoboTwin 适配器（`BENCH=robotwin`）

**先读 `H00_host_template.md`，本文件只写 RoboTwin 特有部分。**
起点分支 `hazard/integration`；环境：既有 RoboTwin 环境（`sapien==3.0.0b1`）。

## 已核实的事实（`02_benchmark_profiles.md` §7）
- 既有实现违反 R2：20 个任务各有 `target_mode`、`primary_arm`（`configs/backends/robotwin/tasks/*.yaml`）；`task_api.py` 提供 6 个可选钩子 `get_dynisafety_*`，由补丁 `patches/robotwin/dynisafety_hooks.patch` 注入；默认走 `task.object` 约定。
- 接触管理器跑在物理子步（1/250 s），`first_contact_step` 与他处不同钟（`diagnostics/contact_clock.py`）。
- `backends/robotwin/disturbance/manager.py`（约 3200 行）包含 `kinematic_first_hit / newton_relative / dynamic_release` 等方案，是统一语义的主要来源。
- 仓库有 `backends/robotwin/expert_cache.py`（**内容未读**）。

## 特有要求

1. **新适配器不得复制 `manager.py`**：只实现 HazardHost 原语，危险逻辑由运行时负责。读 `manager.py` 与 `contact.py` 的目的是理解引擎行为（如 SAPIEN 接触读取、实体名称），不是搬运方案。
2. **R2 与注解预算**：把 `primary_arm` / 目标角色转成 `configs/hazard/annotations/robotwin.yaml`（声明式字段），逐任务统计行数；`task.object` 约定不成立的任务标 `unsupported` 并写明原因，**不写特例分支**。先做 **5 个任务**，主臂规则要覆盖“固定左 / 固定右 / 运行时按物体位置决定”三类（旧覆盖文件里有 `primary_arm_rule` 的例子）。
3. **补丁**：不要依赖 `dynisafety_hooks.patch` 注入的任务类钩子。若新方案确实需要某个钩子，先评估能否用启动脚本包装替代；确需补丁则新建 `patches/robotwin/hazard_hooks.patch` 并在汇报中列出每处修改的必要性。
4. **时钟**：BD 的 `timebase` 以物理子步为最细粒度（实测 `physics_dt`）；新事件一律用秒记录，**不再产生以子步计数的 `first_contact_step`**。汇报里验证新旧时钟换算（用旧后端同一初始状态做对照，只比较时间，不比较结果）。
5. **接触**：`read_pair_body_entity`（`bodies[*].entity`）；机器人侧 = 全部机器人实体去掉名称含 `camera` 者；抓持物判定声明式给出（评估能否用 `get_dynisafety_held_objects` 的默认路径）。
6. **探针素材**：RoboTwin 专家（脚本化规划器）无危险回放；先读 `expert_cache.py` 弄清其缓存格式与失效条件，再用。
7. **G1**：Oracle 包装专家回放；**G2** 暂无，标 `uncertified`。

## 额外验收
- 注解预算表：任务 × 注解字段 × 行数，及 `unsupported` 任务清单与原因。
- 旧 `manager.py` 中被新范式取代的方案 → 新机制的对应表（只列，不删旧代码）。
