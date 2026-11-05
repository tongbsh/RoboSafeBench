# H4 — SimplerEnv 适配器（`BENCH=simplerenv`）

**先读 `H00_host_template.md`，本文件只写 SimplerEnv 特有部分。**
起点分支 `hazard/integration`；环境：既有 SimplerEnv 环境（`sapien==2.2.2`，ManiSkill2 real2sim）。

## 已核实的事实（`02_benchmark_profiles.md` §6）
- `bridge.yaml`：`control_freq: 5`，`sim_freq: 500` → 每动作 100 个物理子步，每个控制步 0.2 s。
- 既有危险体：隐藏 actor（`HIDDEN_POS`）、`set_actor_pose` 摆放、运动学转动力学释放（`backends/simplerenv/disturbance/l1.py`）。
- 接触：`backends/_engine/sapien_contact.py` 的 `scan_actor_contacts`、`first_actor_contact_kinematics`（含法向与接触位置）。
- 目标：`targets.py: compute_target_pos`，`target_pos: null` 时由场景自省（`get_reference_object_actor`）。
- 视觉：real2sim 评测把仿真前景叠加到真实背景图（`rgb_overlay`）。

## 特有要求

1. **子步挂钩是头号风险**：以 1 m/s 运动的危险体，每个控制步位移 20 cm，远超 R1 的位移上限。必须在 ManiSkill2 的 `step` 内部**子步循环**里挂钩。先在已装的 `ManiSkill2_real2sim` 源码中**找到并记录**该循环的位置（文件与行号）及可挂钩方式；若只能在控制步粒度挂钩，则 `BD.timebase` 如实写 0.2 s，并让运行时用 R1 的“子步细分 + 连续检测”策略——细分在宿主侧不可行时，把该基准降为 Tier B 并说明，**不要假装可行**。
2. **视觉合成（C9 / R16）**：危险体在叠加后的策略图像里必须可见。验证办法：同一初始状态下，危险体停放与置于相机视野内两次渲染的策略输入图像做差，并目检保存的样图（放仓库外，路径写进汇报）。若危险体因不在分割前景中而被背景覆盖，如实报告，并评估是否可通过把危险体加入前景分割 ID 解决（不改上游，用包装）。
3. **接触**：SAPIEN 冲量是否为最近一个物理子步的量需实测（P0 里这是推断）；这决定 R9 里“引擎冲量仅作辅助”的辅助量含义，写入汇报。
4. **危险体释放**：沿用既有“运动学 → 动力学”切换，但速度由运行时给定、不加下限（R7）；验证 `set_linear_velocity` 一类调用的生效时机（C5）。
5. **探针素材**：SimplerEnv 无演示。用无危险的参考策略 L0 回放（先用 E00 能跑的最小策略；**学习策略的下载由 X1 负责**，若此时没有可用策略，用保持动作 + 少量手写轨迹作为临时素材并标注）；`TaskContext` 的质量受限，要在汇报里如实评估（例如名义路径是否可信）。
6. **范围**：bridge 的 3 个任务（carrot、eggplant、spoon 中任取，或含 stack），每个 ≥5 个初始状态。
7. **无注解目标**：`target_pos: null` 已是自动模式，注解预算期望为 0；若某任务必须设注解，列出原因。

## 额外验收
- 汇报里附“子步挂钩位置”的源码证据（文件与行号）与最终采用的方案。
- 视觉合成验证的差分图统计（像素差占比）与样图路径。
