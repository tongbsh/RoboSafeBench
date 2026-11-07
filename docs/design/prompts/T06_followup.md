# T06-F — T06 收尾（同一分支继续）

起点分支：`hazard/T06`（**同一分支继续提交，不要新开分支**）　依赖：T05、T06 已合并进 `hazard/integration`　需要仿真器：否　先读：T00，你自己的 `reports/T06.md`，`00_paradigm.md` §8 与 §13 的 **D13（本任务的依据，尤其是 a–d）**，`01_integration_rules.md` §3.3 第 16 条、R14、§5 开头。先把 `origin/hazard/integration` 合并进 `hazard/T06`，保证能看到 D13 与 T05 的一致性套件。

与 T05-F 并行：T05-F 改 `hazardlib/src/hazardlib/conformance/`、`configs/hazard/conformance.yaml`、`configs/hazard/fakehost.yaml`，**这些文件你都不要碰**（只通过公开接口 `hazardlib.conformance.read_report`、`run_suite`、`make_fake_target` 使用套件）。T05-F 会改 `conformance.yaml` 的键，所以**不要把 `conformance.yaml` 复制进仓库里的任何测试配置目录**；测试需要时在 `tmp_path` 里临时拼装。

本任务**允许**改动 T01 的 `spec/manifest.py` 与 T04 的 `runtime.py`（D13 批准的跨任务修改），只做下面列出的改动。

## 目标
把 D13 a–d 落到代码里。每一项一个小提交；既有测试保持绿，不放宽任何断言。

## 范围

1. **Manifest 新字段（D13 a、b）**。`Manifest` 增加必填的 `conformance_report_hash: str` 与 `calibration_hash: str`，同步 T01 的记录测试（只增加用例）。`manifest_version` 加一（尚无已发布的 Manifest，但格式变了）。两个哈希都用 `hashing.sha256_hex` 对解析后的内容计算（与 `config_hashes` 同法：格式与注释不计，值计）。
2. **冻结绑定一致性报告（D13 a）**。
   - 新增纯函数，例如 `check_conformance_report(report, bd, adapter_version, config_hashes) -> List[str]`，**一次列出全部问题**：`verdict` 不是 `pass`；`bd_hash` 与 `bd.hash()` 不同；`adapter_version` 不同；报告的 `config_hashes` 中 `runtime`、`semantics`、`intensity` 缺失或与本次冻结的不同；`runtime_config` 中任何字段的 `source` 来自 `conformance.yaml`（`probe_runtime` / `probe_calibration` 回退值）。
   - `freeze(...)` 必须接收一致性报告（用 `read_report` 读入并校验格式）与校准文件；有任何问题即拒绝，错误信息列出全部问题。
   - `write_frozen` 把报告与 `calibration.yaml` 一起写进冻结目录；`verify_manifest` 校验两者的哈希与 Manifest 一致，被改动、缺失时一次列出。`freeze_report.yaml` 里原来的 `calibration_hash` 改为引用 Manifest 字段，不再是唯一出处。
   - CLI：`hazardctl freeze` 增加必填的 `--conformance <report.json>` 与 `--calibration <calibration.yaml>`；被拒时打印全部问题并返回 2（沿用现有约定）。
3. **运行时特权访问器（D13 d）**。`HazardRuntime.hazard_state(env)`：返回该环境当前危险体的 `host.hazard_state(handle)`；没有分配危险体时返回 `None`。文档字符串写明它是特权接口，只供 Oracle、G2、门控使用，不得进入策略观测（R5），并写明“后构造的运行时接管钩子，旧的随即作废”（01 §3.3 第 16 条，**不加**拒绝第二个运行时的守卫，一致性套件依赖顺序接管）。`driving.py` 去掉截获 `alloc_hazard` 的 `_HostTap`，Oracle / G2 改用这个访问器。运行时测试只增加用例（含停放后、回合结束后、多环境各自独立）。
4. **测试**。
   - `check_conformance_report` 的每条拒绝原因各一个用例，外加“多个问题同时出现时全部列出”。用真实报告做底：在 `tmp_path` 里拼一个配置目录（仓库的 `conformance.yaml`、`fakehost.yaml` 加测试夹具里已赋值的 `runtime.yaml` / `semantics.yaml` / `intensity.yaml`），用 `make_fake_target(config_dir=...)` 与 `run_suite` 生成一次报告，再逐项改字段；仓库配置下（`runtime.yaml` 仍是 PLACEHOLDER）生成的报告应因回退值被拒，也测一例。
   - 端到端：`record → probe → instantiate → gate → calibrate → freeze → verify` 至少一条带真实一致性报告。流水线与一致性测试要用**同一个宿主**，BD 哈希才一致：可以让端到端改用 `make_fake_target` 的 FakeHost 世界，或用它的 `world_overrides` / `host_cls` 对齐 `pipeline_world` 的宿主；选法写进汇报。
   - `verify` 能发现：报告被改、校准文件被改、两者缺失。
   - Oracle 经访问器读到的状态与宿主真值一致；策略观测里没有危险体状态（已有测试保持）。
5. **汇报**：在 `docs/design/reports/T06.md` 末尾追加 “T06-F” 一节：每项的验证命令与真实输出、Manifest 新旧字段、CLI 新参数、端到端里两边 BD 对齐的方式、偏离与理由。

## 验收标准（贴输出）
1. `pytest hazardlib -q` 在 `hz38`、`hz311` 全绿（贴总数，对比合并后集成分支的 1257 + 1 skip / 1258）。
2. 端到端命令链的真实输出（贴每条命令的一行摘要），包括一次因报告不合格被拒的输出。
3. `python -m hazardlib.conformance --host fake` 仍是 13 项 `pass`（确认没动到套件行为）。
4. 无数值默认值、无仿真库 import、Python 3.8 兼容（既有卫生测试通过）。
5. `git status` 干净；只改了上面列出的范围（贴 `git diff --stat origin/hazard/integration`）。

## 不要做
不改 `docs/design/00–03`、`src/dynisafety/`、T05-F 的文件（见上）；不实现 G2 避障算法、真实宿主或实验；不给 PLACEHOLDER 赋值；不为通过测试放宽任何阈值。
