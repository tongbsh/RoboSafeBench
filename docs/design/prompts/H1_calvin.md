# H1 — CALVIN 适配器（`BENCH=calvin`）

**先读 `H00_host_template.md`，本文件只写 CALVIN 特有部分。** 建议这是**第一个**走通的适配器：引擎最简单。
起点分支 `hazard/integration`；环境 `calvin_venv`（**Python 3.8**）；上游固定：`mees/calvin@fa03f01`、`mees/calvin_env@797142c`。

## 已核实的事实（`02_benchmark_profiles.md` §2，详见源码）
- 240 Hz 物理、30 Hz 控制、每动作 8 个 `stepSimulation`；`env.step` 返回 `(obs, 0, False, info)`，无奖励与终止。
- 成功由 `Tasks.get_task_info_for_set(start_info, current_info, {subtask})` 判定（状态差分，声明式配置 `conf/tasks/new_playtable_tasks.yaml`）。
- `scene.get_obs()` / `get_info()` 只涉及已注册物体 → 未注册的 Bullet 刚体不进入状态向量；但 `robot.get_observation()` 的 `contacts` 会包含与危险体的接触。
- 动作为 7 维相对位姿，`use_target_pose: true` → 对累计目标位姿积分，零动作不会立刻刹停。
- `serialize()` / `reset_from_storage()` 可精确恢复；`deterministicOverlappingPairs=1`。
- 序列内不重置环境，只重置模型；`EP_LEN=360`。

## 特有要求

1. **挂钩**：子类化 / 包装 `PlayTableSimEnv.step`，在 8 次 `stepSimulation` 的每一次前后调用 `runtime.pre_step/post_step`（复制了上游 `step` 的几行，**必须固定 `calvin_env@797142c` 并在 import 时断言**）。评测脚本里用官方预留的 `make_env` 包装点。
2. **危险体生命周期按子任务管理**：子任务开始时 `begin_episode`，结束（成功或 360 步到）时 `end_episode` 并停放；序列内跨子任务不残留。
3. **隔离（C7）**：危险体**不得**注册进 `scene` 的物体列表；验证 `scene_obs`、`get_info()['scene_info']` 与任务判定结果在“危险体存在但不接触”时与无危险运行**逐位一致**；并专门检查 `robot_info['contacts']` 里出现危险体接触时任务判定器是否受影响，结果写进汇报。
4. **机器人侧碰撞体**：Panda 臂 + 夹爪；抓持物按 `held_object` 分类（可由夹爪开度与方块位姿声明式推断；若不能可靠判定，C4 的该子项标注并说明）。
5. **`hold_action`**：7 维零相对动作 + 当前夹爪指令。因目标位姿积分，**T_stop 协议需额外记录目标–实际位姿滞后**，并在汇报里讨论它对 Σ 定义的影响（不要自行改定义，提出建议即可）。
6. **真几何距离**：`getClosestPoints`，`distance: geometric`；若实测不可用则如实标记。
7. **探针素材**：用 debug 数据集（约 1.3 GB）里的验证初始状态与演示做无危险回放；角色从判定器配置静态读出（列出 `new_playtable_tasks.yaml` 中被你覆盖的任务，≥10 个，涵盖推 / 旋转 / 抽屉 / 举起类各至少一个），并统计注解行数（期望 0 或极少）。
8. **Python 3.8**：核心与适配器代码不得出现 3.9+ 语法与标准库特性。
9. **渲染**：C9 要在静态相机与夹爪相机两路图像上验证危险体可见。
10. **重力与阻尼**：`load()` 设重力为 −9.8（`calvin_env/envs/play_table_env.py:133`），与其他基准不同；用自由落体测试得到 `gravity_mps2` 写入 BD。PyBullet 动态体的默认线 / 角阻尼是否非零 **[待探针]**；危险体必须用 `changeDynamics` 置 0（置不了 0 就不满足 Tier A，D13 h），并用 C5 验证落地时间。
11. **GPU 与渲染**：只用 **GPU 2**（用户指定；GPU 0 / 1 是别的任务）。GPU 2 上常驻用户的占卡脚本，它在 NVML 计算进程列表里看到本用户的进程会让出，但纯 EGL 渲染的进程可能只是图形进程、不触发让出，此时 GPU 2 约有 5 GiB 空闲，够 200×200 的渲染用。E00 的日志是 `EGL device choice: -1 of 7`（没指定设备，实际在哪张卡 [未核实]）：先查清 EGL 设备号与 `nvidia-smi` 编号的对应（CALVIN 自带 `set_egl_device` 的做法可参考），用 `EGL_VISIBLE_DEVICES` 固定到 GPU 2；tacto 的 pyrender 也走 EGL，同样核实它落在哪张卡。用 `nvidia-smi` 的进程列表确认后把命令与输出贴进汇报。`calvin_venv` 里的 torch 不能用 GPU（E00 已知坑 3），本任务也不需要。
12. **测试依赖**：`calvin_venv` 没有 `pytest`。为了“额外验收”第一条，可以在 `calvin_venv` 里装一个支持 Python 3.8 的 `pytest`（小下载，不需要 sudo），版本写进汇报与 `env_notes/calvin.md`；不要装或升级别的包。

## 额外验收
- 在 `hz38` 之外，`calvin_venv` 里 `pytest hazardlib/tests -q` 也全绿（`PYTHONPATH=<worktree>/hazardlib/src`）。
- 特权运行（T_stop、Oracle）用 D14 的试点目录 `configs/hazard/pilot/calvin/`；预注册建议值表附在汇报末尾。
- 汇报中单列“CALVIN 上最可能被审稿人质疑的两点”，各给出你的判断与证据。
