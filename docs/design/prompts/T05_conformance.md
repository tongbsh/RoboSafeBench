# T05 — 一致性测试套件 C1–C13

起点分支：`hazard/integration`（T04 已合并）　依赖：T04　需要仿真器：否（套件本身）　先读：T00，以及 `01_integration_rules.md` §5。可与 T06 并行。

## 目标
把 `01_integration_rules.md` §5 的 C1–C13 实现为**可对任意 `HazardHost` 运行**的 pytest 套件，并用“变异体”证明每个测试都能抓到它声称要抓的缺陷。这是后面所有适配器的验收标准，必须严格。

## 范围与产出

1. `hazardlib/src/hazardlib/conformance/`：
   - `tests/test_c01_*.py … test_c13_*.py`，一个测试一个文件；
   - `conftest.py`：`--host=<name>` 选项，通过注册表解析宿主工厂；`--host=fake` 使用 `FakeHost`；注册表 `register_host(name, factory)`，适配器包在 import 时注册；
   - `python -m hazardlib.conformance --host <name> --out report.json`：运行并输出 JSON 报告（每个测试的通过 / 失败 / 跳过及原因、测得的数值、BD 哈希、适配器版本）。
2. `configs/hazard/conformance.yaml`：所有容差与阈值。**不要编造看似合理的数字然后当作标准**——先用 `FakeHost` 的解析值定出能区分“正确”与“变异体”的最宽容差，把推导过程写进 YAML 的注释和你的汇报；真实仿真器上的容差由各适配器任务在汇报里提出调整建议，**不得**在适配器任务里悄悄改。
3. 变异体验证 `hazardlib/tests/test_mutants.py`：对 `FakeHost` 的每个变异开关（`tunneling`、`residual_velocity`、`state_leak`、`wrong_dt`、`no_rollback`、`park_interacts`、`oracle_depends_on_hazard`、`bad_standoff`、`hook_dependent_release`、`post_impact_vn`）断言**至少一个**对应的 C 测试失败；同时断言无缺陷的 `FakeHost` 通过全部 C 测试。给出“变异开关 → 抓到它的测试编号”对照表。
   回合由 `EpisodeDriver`（D11 a）驱动：C2、C7、C9 用 `policy_observation` 与无危险对照回合（`arm(env, None, control_id=...)`）逐键比较；C3 在高速穿越时同时检查接触事件与 `r1_violation_count`（D11 d）。
4. C 测试的跳过规则：C6 在 Tier B 允许跳过；C12 仅批量基准；C8 在 BD 声明为 `statistical` 时改为测差异量级；其余不得跳过。跳过必须在报告里写明原因，且**不能因为适配器没实现而跳过**。
5. **C13（名义命中时机，D1）**：在 `FakeHost` 上实现完整版本——用 `build_hit_path` 构造含 `standoff` 的实例，令机器人沿已知名义路径运动，断言首次官方接触时刻与名义 `t_hit` 之差 < `tol_hit_time_s`；`bad_standoff` 变异体必须使它失败。容差键是 `conformance.yaml` 的 `c13_nominal_hit.hit_time_abs_tol_s`（T04-F 已建，PLACEHOLDER）：名义释放提前一个物理步（D12 b），reach 的命中误差是 O(dt)（FakeHost 上 dt = 0.002 时实测 1.4e-4 s），因此推导出的容差必须至少覆盖一个物理步，同时仍要抓住 `bad_standoff`。真实宿主上若没有可重复的名义回放，C13 报告为 `unverified`（**不是**通过），并在汇报里写明原因。C5 同时要求：先实测 `gravity_mps2` 再比较落地时间，并覆盖“FakeHost 取不同重力”的情形。
6. **D12 的口径**：“首次官方接触”指伙伴为机器人侧的 `dynamic` 事件（`EventKind.DYNAMIC`），与 `scene` 的 `dynamic` 事件不算（D12 f）。`01_integration_rules.md` §3.3 的 15 条宿主钩子语义（D12 e）是适配器契约；其中能由 C 测试检验的（如停放期间的接触、运动学重叠同步上报、释放后速度立即可读、碰撞体名一致），在对照表里注明由哪个测试覆盖，检验不了的列为“未覆盖”并说明原因。

## 验收标准（贴输出）

1. `pytest hazardlib/tests -q` 与 `python -m hazardlib.conformance --host fake` 在 `hz38`、`hz311` 全绿。
2. 变异体对照表完整：10 个变异开关各被至少一个测试抓住；若某个缺陷没有对应测试能抓到，**补测试**，不要降低要求。
3. 报告 JSON 的结构稳定（含 schema 版本号），能被 T06 与汇报脚本读取。
4. 套件不 import 任何仿真库。

## 不要做
不要为任何真实基准写测试逻辑；不要在套件中放基准特例；不要修改 `FakeHost` 的正确行为来迁就测试（缺陷开关除外，需在汇报说明）。

## 汇报
`docs/design/reports/T05.md`。包含：C1–C13 的“声明要抓的缺陷 / 实际抓到它的变异体 / 容差推导”表；你认为 §5 中过松或过严的条目。
