# T01 — 核心接口与规格对象（`hazardlib` 骨架）

起点分支：`claude/sweet-newton-2clmpw`　依赖：无　需要仿真器：否　先读：T00。
这是**最先完成**的任务，其他核心任务都依赖它的接口；务必小而稳，宁可少做，不要提前实现 T02–T06 的逻辑。

## 目标
建立独立包 `hazardlib` 的骨架，定义全部**数据结构与接口**，不含运动学、语义、运行时的实现。

## 范围与产出

1. `hazardlib/pyproject.toml`：包名 `hazardlib`，`requires-python = ">=3.8"`，依赖仅 `numpy`、`pyyaml`；开发依赖 `pytest`。`src/` 布局，包含两个顶层包：`hazardlib`（核心）与 `hazardhosts`（先放空的 `__init__.py` 与每个基准一个空子包 `libero simplerenv robotwin robocasa robodojo calvin`）。
2. `hazardlib/spec/`（数据类 + `to_dict` / `from_dict` + 校验）：
   - `HazardSpec`：`spec_version, id, family, agent(kind, shape, half_extents_m, mass_kg), motion(primitive, params), exposure(anchor, anchor_phase{phase, frac}, offset_m), timing(mode ∈ {nominal_time, phase_event}), intensity(dict), release(mode)`；
   - `BenchmarkDescriptor`：字段严格对应 `01_integration_rules.md` §2，并提供 `tier()` 方法按 §2 的规则推导 Tier；`strict=True` 时任何 `unknown` 都抛错；
   - `TaskContext`：`task_id, layout_id, entities(role→{name, source}), phases(list of {name, t_start_s, t_end_s}), nominal_ee_path(时间参数化的位姿序列), occupancy(点集+时间), workspace_bounds, support_surfaces, provenance`；
   - `HazardInstance`：`instance_id, keyframes(time_s, pos, quat), release_time_s, release_velocity, nominal_hit_point, nominal_hit_time_s, onset_time_s, certificate, provenance`；
   - `Certificate`：`g0, g1, g2, label ∈ {wait_solvable, evade_solvable, unsolvable_by_probes, uncertified}, notes`；
   - `Manifest`：实例清单 + BD 哈希 + 探针集描述哈希 + 规格库哈希 + 适配器版本 + 基准提交号 + 种子。
3. `hazardlib/host.py`：`HazardHost`（`typing.Protocol`，方法与签名严格对应 `01_integration_rules.md` §3.1–3.2）、`Tier` 枚举、`CapabilityError`、`HazardHandle`、`RobotState`、`ContactObs`、`TimeState`、`Pose`、`Surface`、`Action`（别名）。
4. `hazardlib/hashing.py`：规范化 JSON（键排序、浮点固定精度的表示规则写进文档字符串）→ `sha256`；同内容不同键序必须同哈希。
5. `hazardlib/config.py`：YAML 加载器，支持 `configs/hazard/` 下的文件；值为字符串 `PLACEHOLDER` 的字段在 `strict=True` 下抛错。
6. `configs/hazard/`：`bd/`（先放 `calvin.yaml` 草稿，按 `02_benchmark_profiles.md` §2 填写已核实的字段，其余写 `unknown`）、`annotations/`（空目录 + `.gitkeep`）、`families.yaml`、`intensity.yaml`、`gates.yaml`、`semantics.yaml`、`conformance.yaml`（先放结构与 `PLACEHOLDER` 值，**不要编造数值**）。
7. `hazardlib/README.md`：一页说明目录、安装、测试命令、Python 兼容规则。
8. `hazardlib/tests/`：覆盖上述全部。

## 验收标准（逐条在汇报里贴输出）

1. 建两个干净环境 `hz38`（Python 3.8）与 `hz311`（Python 3.11），各自执行 `pip install -e hazardlib --no-deps` 后 `pytest hazardlib/tests -q` **全绿**。
2. 在 numpy 1.x 与 numpy 2.x 下各跑一次测试，**全绿**（可用同一环境换 numpy 版本）。
3. 哈希测试：同内容不同键序 → 同哈希；改任一字段 → 哈希改变。
4. `BenchmarkDescriptor.tier()` 对 `calvin.yaml` 草稿给出的结果与你推导的一致，并有“缺少某项能力时降为 B / C”的单元测试。
5. `strict=True` 加载含 `unknown` 或 `PLACEHOLDER` 的文件会抛出可读错误；`strict=False` 可加载。
6. `grep -rnE "import (torch|scipy|mujoco|pybullet|sapien|isaac|robosuite|dynisafety)" hazardlib/src/hazardlib` **无输出**。
7. 3.8 兼容检查：`python3.8 -m compileall hazardlib/src` 无错误；代码中不出现 `match` 语句、`dataclass(slots=True)`、`zip(strict=`、运行时求值的 `X | Y`。

## 不要做
不要实现运动学、语义判定、运行时、探针、一致性测试；不要写任何阈值数值；不要修改 `src/dynisafety/`。

## 汇报
`docs/design/reports/T01.md`。另外在汇报里单独列出：你对接口做出的、设计文档没写明的决定（例如字段类型、可选性）。
