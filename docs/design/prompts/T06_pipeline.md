# T06 — 流水线：探针 / 实例化 / 门控 / 校准 / 冻结

起点分支：`hazard/integration`（T04 已合并）　依赖：T04　需要仿真器：否（在 `FakeHost` 上验证）　先读：T00，`00_paradigm.md` §3、§5、§7、§8。可与 T05 并行。

## 目标
实现 `hazardctl` 流水线的**引擎无关**部分，并在 `FakeHost` 上端到端验证。真实基准的接入在 H 类任务里。

## 范围与产出

0. **前置（单独一次提交）**：T01 的 `spec/certificate.py` 中 `CERT_LABELS` 增加 `baseline_unusable`（§8 的 G1 前提），同步 T01 测试；`configs/hazard/gates.yaml` 的 `g0` 下新增 `early_contact_tol_m: PLACEHOLDER`（用于下文第 3 项的首次接触校验）。
1. **探针**（`hazardlib/probe/`）：
   - 输入：一组探针运行的 trace（格式沿用仓库诊断 trace 约定：逐步 `eef_pos`、`gripper`、`object_poses`、`in_contact`、`phase`、`t_s`）与成功事件；
   - 输出 `TaskContext`：阶段时间线（由夹爪闭合 / 张开事件与成功时刻切分；阈值走 YAML）、任务相关实体（**顺序**：静态声明 → 被抓取或被移动的物体 → 注解；每个实体记录来源）、名义 EEF 路径（跨探针运行按阶段归一化时间取逐点中位）、占用分布、工作区边界；
   - 探针集描述（来源类别、运行数、参考策略名）进入 `provenance` 并参与哈希；**拒绝**任何标记为“被测策略”的来源（R4）。
2. **实例化**（`hazardlib/instantiate/`）：`instantiate(spec, task_context, seed, bd) -> HazardInstance`：按 `00_paradigm.md` §5.2 选命中点与命中时刻，调用 T02 的 `build_hit_path`；是 `(spec, TC, 任务编号, 布局, 实例种子)` 的**纯函数**（用独立的随机数流，不共享全局随机状态）。
   - **换算与记录**：调用 `build_hit_path` 之前，先按 `standoff = r_hazard(n̂) + eef_contact_radius_m` 把接触点换算成质心命中点，并把 `standoff_m` 写入 `provenance`（D1）；释放窗口参数传 `runtime.yaml` 的 `release_velocity_window_s`，**不是**引擎钩子步长（D4）；
   - **关键帧**：把 T02 的解析 `Path` 采样成 `HazardInstance.keyframes`，间隔 `keyframe_dt_s`（YAML），并**测试**线性插值回放与解析路径的位置误差 ≤ `keyframe_interp_tol_m`（YAML）；危险体姿态取 `HazardSpec.agent` 指定的固定姿态（默认单位四元数），写入 `provenance`；
   - **拒绝**：带严重度的 `reach` 实例 `overshoot_m` 必须 > 0，否则拒绝（`overshoot_m = 0` 属于 `occupancy.dwell`，D6）；
   - **实例自描述（D11 b）**：写入 `spec_id`、`family`、`agent`（取自 HazardSpec）；
   - **强度 → 原语参数的映射**：从 HazardSpec 的强度向量（τ、v 等）到原语参数（`reach` 的起点距离与 `overshoot_m`、`slide` 的起点距离、`fall` 的释放高度）的映射由本任务定义，写进模块文档与汇报；其中需要的常数放进 `intensity.yaml`（写 `PLACEHOLDER`），不得硬编码；
   - **`slide` 的支撑面约束**：`slide` 实例的命中点高度 = 支撑面顶 + 危险体沿 z 的半尺寸，路径在支撑面上水平运动（z 速度为 0），释放后由支撑面托住，不是悬空平移；命中时刻只在名义 EEF 距该支撑面顶的高度 ≤ 危险体高度 + `eef_contact_radius_m` 的阶段里选，找不到就拒绝，原因记 `no_support_surface`。支撑面取宿主的 `support_surfaces(env)`，缺失时用 TaskContext 估计并记录来源；
   - **`fall` 的释放高度**：`fall` 的 τ 由释放高度决定，`h = g·τ²/2`（`g` 用 BD 实测的 `gravity_mps2`）：τ = 0.4 s 约需 0.78 m，0.5 s 约 1.23 m，1.0 s 约 4.9 m。释放点超出工作区上界（BD / TaskContext 的工作区边界）时由 G0 拒绝，原因记 `release_height_out_of_workspace`。汇报里列出 FakeHost 以及 `02_benchmark_profiles.md` 中有工作区数据的基准上 `fall` 族的可行 τ 区间。“前导运动”（例如先从架边滑出再下落）以扩大 τ 是 **[开放]** 项，本任务不实现；
3. **门控**（`hazardlib/gates/`）：
   - G0：几何有效（工作区内、与静态场景体不穿透、停放位置合法）；拒绝 `diagnostics["onset_margin_s"] < 0` 的实例；记录 `τ_vis`（宿主提供 `render_check` 或可由相机几何估计时；否则为 `None` 并在证书备注里写明），**只记录、不过滤**（D5）。另加三项检查，违反即拒绝并记原因：
     - **首次接触时刻**：`build_hit_path` 只保证 `t_hit` 时刚好接触，不保证之前没碰上。实测反例：EEF 以 0.5 m/s 前进、危险体同向 0.3 m/s 追赶时仍被接受，onset 时质心距 0.04 m 已小于接触余量 0.06 m，`t_hit` 时闭合速度为 −0.2 m/s。因此要求 (i) `t_hit` 时危险体相对名义 EEF 沿接触法向的闭合速度 > 0，否则记 `non_closing`；(ii) `[onset, t_hit)` 内危险体质心到名义机器人占用（EEF，加上 trace 中有的机器人碰撞体位置）的最小距离 ≥ `standoff − g0.early_contact_tol_m`，否则记 `early_contact`。trace 只有 EEF 时在 `provenance` 记 `early_contact_check: eef_only`，并在汇报中说明 `fall` 落向夹爪时腕部 / 前臂可能先被砸到；
     - **R1（D11 d）**：用 T02 的 `needs_substeps`，按钩子间隔（BD 声明的钩子粒度对应的步长）检查运动学阶段路径；动力学阶段用预测最大速度 × 钩子间隔检查（`fall` 为 `g·(t_hit − onset)`，`slide` / `reach` 为释放速度）。位移超过 `max_step_ratio × agent 最小半尺寸` 记 `r1_step_ratio`；BD `hazard_ccd: true` 时跳过此项并在 `provenance` 写明；
     - **`fall` 释放高度**：见上面实例化一节，记 `release_height_out_of_workspace`；
   - G1：`run_g1(host, instance, reference_policy)` 编排“Oracle 停等恢复”运行；`ReferencePolicy` 抽象 = `{demo_replay, scripted, external_policy}`；先要求参考策略在**无危险**时成功，否则标 `baseline_unusable`；
   - G2：`PrivilegedAvoider` 抽象接口（`filter_action(action, robot_state, hazard_state)`），无实现时证书标 `uncertified`，**不得假称可解**；
   - G1 / G2 的回合用 `EpisodeDriver` 驱动（D11 a）：`arm` → `reset_episode` → `step` 循环，参考策略读 `policy_observation`；
   - **门结果 → 标签的决策表**（写成一个纯函数并逐行测试），按顺序判定：G0 不通过 → 实例被拒绝，不进 Manifest，按原因计数；参考策略在无危险时不成功 → `baseline_unusable`；G1 通过 → `wait_solvable`；G1 不通过且有 G2 实现：G2 通过 → `evade_solvable`，否则 → `unsolvable_by_probes`；G1 不通过且无 G2 实现 → `uncertified`。**主分集合 = {`wait_solvable`, `evade_solvable`}**；`uncertified` 可以运行，但结果单列（主表脚注给出数量与结果）；`unsolvable_by_probes` 单报“不可避免”桶；`baseline_unusable` 不运行，只计数报告。标签与 `00_paradigm.md` §8 一致。
4. **校准**（`hazardlib/calibrate/`）：`run_t_stop_protocol(host, ...)` 按 §7 流程在宿主上测 T_stop（名义速度取自 TaskContext 的 EEF 速度中位数）；输出 `calibration.yaml`；T_stop 有删失（`n_censored > 0`，机器人未停下）时标记为**下界**，Σ 的计算必须拒绝或显式带标记（D6）。
5. **冻结**（`hazardlib/manifest/`）：生成 `Manifest`，含各哈希；`verify_manifest` 校验未被篡改。按 D11 c 填写 `config_hashes`（所用 `configs/hazard/*.yaml` 的逐文件哈希）；冻结前对这些配置做 strict 加载，含 `PLACEHOLDER` 即拒绝冻结。FakeHost 端到端测试使用一套测试专用的完整配置目录（放在测试夹具里），**不改**仓库里的 `PLACEHOLDER`。
6. **CLI**：`hazardctl probe|instantiate|gate|calibrate|freeze`，输入输出为文件（YAML / JSONL），便于在不同环境的进程之间传递。

## 验收标准（贴输出）

1. `pytest hazardlib/tests/pipeline -q` 在 `hz38`、`hz311` 全绿。
2. `FakeHost` 上端到端：探针 → TaskContext → 实例化 → 门控 → 校准 → 冻结，一条命令链跑通。
3. **确定性**：相同输入两次运行得到相同的实例与相同的 Manifest 哈希；改任一输入（种子、探针集、BD、任一配置文件）哈希变化。
4. **名义必碰**：在 `FakeHost` 上，无反应的机器人必被撞；Oracle 停等运行无接触且成功 → `wait_solvable`。
5. **不可解**：构造 τ 小于 `FakeHost` 的 T_stop 的实例，证书必须为 `unsolvable_by_probes`（或在无 G2 实现时 `uncertified`），且不进入“主分”集合（给出过滤函数及测试）。
6. **暴露率**：被测机器人轨迹远离命中点时，实例被标 `exposure_low`；贴近时不标。判定**调用 T03 的 `is_exposure_low`**（D9 e），窗口取 `[t_onset, t_hit]`，半径取自 `semantics.yaml`；不得在流水线里重写。
7. **探针拒绝**：把“被测策略”作为探针来源会抛错。
8. 流水线代码不 import 仿真库；Python 3.8 兼容。
9. **standoff**：`FakeHost` 上按含 standoff 的实例，首次官方接触与名义 `t_hit` 之差 < 容差；忽略 standoff 时失败；`standoff_m` 出现在 `provenance`。
10. **关键帧回放**：实例关键帧经线性插值回放，与解析路径的位置误差 ≤ `keyframe_interp_tol_m`；由关键帧得到的释放速度与 T02 的固定窗口后向差分一致。
11. **拒绝**：`onset_margin_s < 0` 的实例、`overshoot_m = 0` 的带严重度 `reach` 实例，都被拒绝并有测试。
12. **T_stop 删失与 Σ 的受保护入口（D6、D8）**：`calibrate.margin_sigma(tau_s, t_stop_s)` 保持为纯数学（T02 的设计，只收一个数，看不到删失）。在 T06 新增受保护入口 `sigma_from_summary(tau_s, summary, allow_censored)`，返回 `(sigma, is_upper_bound)`：`summary.n_censored > 0` 时，T_stop 只是**下界**（机器人有重复没停下，中位数偏乐观），因此 Σ = τ/T_stop 是**上界**——默认（`allow_censored=False`）直接拒绝并抛错，显式允许时返回 `is_upper_bound=True`。校准输出（`calibration.yaml`）在有删失时带“T_stop 为下界”标记。**卫生测试**：用 AST 检查，除 `calibrate/` 内的受保护入口与测试文件外，任何模块不得直接调用 `margin_sigma`。
13. **G0 新增拒绝**：上面的追赶式反例被拒（`non_closing` 或 `early_contact`）；onset 时已在 standoff 内的实例被拒；`fall` 释放高度超出工作区被拒；R1 超限被拒，同一实例在 `hazard_ccd: true` 时不拒。每种拒绝原因有测试，且原因计数出现在门控输出里。
14. **`slide` 支撑面**：`slide` 实例路径的 z 恒等于支撑面顶 + 危险体半高；名义 EEF 始终高于阈值的任务被拒，原因 `no_support_surface`。
15. **决策表**：逐行测试；`baseline_unusable` 在 `CERT_LABELS` 中；主分过滤函数只放行 `wait_solvable` 与 `evade_solvable`。
16. **冻结**：配置含 `PLACEHOLDER` 时 `freeze` 拒绝；`config_hashes` 随任一配置文件内容变化。

## 不要做
不要实现任何真实基准的宿主；不要写 G2 的具体避障算法；不要做实验。

## 汇报
`docs/design/reports/T06.md`。列出 TaskContext 估计中你做的假设（如名义路径的聚合方式）及其对真实基准可能的失效情形；列出强度 → 原语参数的映射；列出各基准 `fall` 族的可行 τ 区间。
