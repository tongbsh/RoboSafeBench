# H3 — 原始 LIBERO 适配器汇报

状态：本地实现与适配验收完成；代码尚未提交 / 合并。R1-L 的最终四任务策略运行见 [R1-L 汇报](R1-L.md)。
日期：2026-11-10。基线：hazard/integration @ bd2cfa2。D18 依据用户更正，使用原始 LIBERO，四个 suite 各 10 个任务。

## 做了什么

- 原始 `Lifelong-Robot-Learning/LIBERO@8f1084e`，四套件按官方 task_order_index=0 读取；task_id 为 `<suite>/<官方任务名>`，不映射 safelibero_*。
- 在原始模型 **source MJCF** 的 worldbody 末尾追加独立危险体；原始 BDDL、物体、相机、控制器与成功谓词保持。显式质量与 box 惯性，保留原始模型统计量（裁剪、阴影和缩放）。
- 复用了旧 H3 worktree 的布局、几何与宿主草稿，并移除其 SafeLIBERO 场景 / 预置障碍物池假设。未修改该 worktree。
- HazardHost + EpisodeDriver：25 个 2 ms 子步构成一个 50 ms 控制步；每个 MjSim.step 前后挂钩；scratch MjData 取接触、几何距离与机器人运动学，避免观测查询改变主仿真状态。
- 机器人侧包含机器人刚体与单列 held_object；BDDL obj_of_interest 作为任务实体声明，不再把所有任务物体计入机器人。
- 表面样本来自编译后的真实碰撞几何。间距 5 mm，含刚体朝向；三族 C13 使用几何 standoff。几何体素合并的覆盖误差界在 geometry.py 内说明。
- 每次 reset 重建模型与渲染上下文，消除接触探针后的一灰度级渲染历史差异；未放宽 C8。
- 支持演示初始状态、演示中间状态和原始 init_files 状态；按关节名映射，追加危险体 DOF 单独停放。

## 验收：命令与真实输出

公共启动环境：`LIBERO_CONFIG_PATH=<本机 upstream 配置目录> MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=hazardlib/src NUMBA_CACHE_DIR=/tmp/robosafebench-numba`。环境为 dynisafety-libero（Python 3.10.22、numpy 1.26.4、mujoco 3.6.0、robosuite 1.4.0）。

### C1–C13

命令：`python -m hazardhosts.libero.probe --out _logs/h3_original/scene --conformance`。最终 adapter_version = libero-original-h3-2。

```
C1 PASS 
C2 PASS 
C3 PASS 
C4 PASS 
C5 PASS 
C6 PASS 
C7 PASS 
C8 PASS 
C9 PASS 
C10 PASS 
C11 PASS 
C12 SKIP single-environment host (parallel.num_envs = 1): no other environment to observe
C13 PASS 
CONFORMANCE {"error": 0, "fail": 0, "pass": 12, "skip": 1, "unverified": 0}
```

报告：`_logs/h3_original/scene/conformance.json`。BD strict=True 加载成功，Tier A；正式配置尚未预注册，本报告使用了开发探针回退值，不能用于正式冻结。

### 原始场景回归

命令：`python -m hazardhosts.libero.probe --out _logs/h3_original/scene --parity --tasks 0,1,2,3`。四任务各 demo_0 的完整动作回放，原始场景 vs 危险体停放：

| task | steps | max EEF diff (m) | max joint diff | max RGB diff | success timeline identical |
|---|---:|---:|---:|---:|---|
| 0 | 98 | 0.0 | 0.0 | 0 | True |
| 1 | 152 | 0.0 | 0.0 | 0 | True |
| 2 | 103 | 0.0 | 0.0 | 0 | True |
| 3 | 99 | 0.0 | 0.0 | 0 | True |

证据在 `_logs/h3_original/scene/parity/`。上游 git status 无改动，`git diff -- src/dynisafety` 为空。

### 原生评测循环（item 14）

运行原始 `libero/lifelong/metric.py:evaluate_one_task_success` 的未改写函数体（AST 加载，检查固定提交与文件未修改），只注入单环境向量代理、保持策略和观测转换；不是自行重写一个评测循环。

命令：`python -m hazardhosts.libero.probe --out _logs/h3_original/scene --native --pilot-root _logs/libero_original_pilot_v2`。
输出：`ITEM 14 PASS`，4 begun / 4 ended / 4 logs。保持策略的成功率为 0，仅验收回合生命周期，不是策略性能结果。证据：`_logs/h3_original/scene/native/summary.json`。

### 参考回放、T_stop 与特权门控

最终原始场景产物：`_logs/libero_original_pilot_v2/`。spatial 0–3 各回放前 10 集，失败完整计数；只用成功演示构造上下文。每任务一个选定布局，T_stop 重复 5 次。数字汇总见 R1-L 与各 task_XX/{baselines.json,calibration.yaml,preparation.yaml,instances.jsonl}。

额外 smoke：goal 的 open_the_middle_drawer_of_the_cabinet 为 9/10；libero_10 首任务为 9/10；object 首任务为 1/1。证据：`_logs/h3_original/scene/libero_{goal,10,object}/replays.json`。这不是四套件 40 任务的完整评测。

### 演示布局与官方 init_files

命令：`python -m hazardhosts.libero.probe --out _logs/h3_original/scene --initial-compare --tasks 0,1,2,3`，每任务 3 对初始状态，用同一集动作回放，记录物体初始位置差异、EEF 路径、首次持物和成功时刻。

12 对诊断中，EEF 路径 RMSE 为 0.01339–0.05977 m，单物体初始位置差最大 0.16573 m；部分相同动作在官方 init 布局上不成功。证据：`_logs/h3_original/scene/layouts/initial_layout_comparison.json`。因此虽然任务和场景来自原始 LIBERO，本次演示初始状态试点的 SR 不能当作官方 init_files 协议的公开 SR。

## 偏离设计文档之处与理由

- D18 替换旧 SafeLIBERO 假设；旧手工 L1–L3 映射要求撤销。旧后端和旧结果未改。
- 只有演示一种探针来源，显式 min_source_kinds=1；这是 D14 试点偏离，不满足正式 R4 双来源要求。
- 4 cm、50 g 橙色 box 为运动族的几何代理；reach 不宣称是真实人体模型。
- 当前 slide 候选均以 no_support_surface 被拒；没有将失败构造伪装成可用实例。fall 的可见提前量在部分实例为 0，单独保留诊断。
- G2 接口可行性：可由 host.robot_state、runtime.hazard_state 和 min_distance 构造真值观测；现有 AEGIS 桥接依赖旧 LiberoEnv，不能直接接到本宿主，尚未实现 PrivilegedAvoider。G1 失败一律 uncertified，未称不可解。
- pipeline_io 使用可用时的 CSafeLoader / CSafeDumper（无则 SafeLoader / SafeDumper），加速含几十万几何点的 TaskContext；安全类型与格式不变。
- 卫生测试允许 Python 自动附加的已导入子模块，但仍要求包 __init__.py 为空，且不得导出其他符号；未放宽科学测试容差。

## 测试与未解决项

最终核心测试：Python 3.11 为 **1564 passed**（`_logs/h3_original/pytest_hz311_complete.log`），Python 3.8 为 **1563 passed, 1 skipped**（`pytest_hz38_complete.log`），包括新增的 RNG 协议测试。
旧仓库 tests 的基线与最终均为 **25 failed, 1071 passed, 11 skipped, 4 deselected, 6 errors**（含 socket 权限、缺失的历史脚本、现有 CLI 卫生问题），31 个失败 / 错误项的集合相同；日志为 `_logs/h3_original/regression_libero_before.log` 与 `regression_libero_final.log`。收尾另跑源码卫生检查 **313 passed**，核验当前 README、hazardlib README、进度与两份汇报中的 **42 个本地链接均有效**，`git diff --check` 通过；没有重跑实验或全量测试。
正式预注册、第二来源、G2、slide 覆盖、可见提前量过滤策略、跨基准正式实验与 R1-F 均未完成。

## 适配成本账本

引擎适配与布局 / 几何 / 路径 / 注册 / MJCF 注入：1,494 行（`wc -l`：host 965 + demos 150 + geometry 236 + paths 65 + scene 42 + target 36）；不含探针、native 验证、pilot、策略桥接与视频。逐任务注解 0 行，上游补丁 0。首次 C1–C13 为 6 pass / 6 fail / 1 skip；最终 12 pass / 1 skip。工作在当前工作区，未提交合并。

## 下一步建议

R1-L 已完成 π₀.₅ 四任务试点。下一步先评估低可见落物、slide 实例化几何与 uncertified 占比，再扩展策略和布局；保持 pilot 与正式结果分离。
