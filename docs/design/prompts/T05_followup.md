# T05-F — T05 收尾（同一分支继续）

起点分支：`hazard/T05`（**同一分支继续提交，不要新开分支**）　依赖：T05、T06 已合并进 `hazard/integration`　需要仿真器：否　先读：T00，你自己的 `reports/T05.md`，`00_paradigm.md` §13 的 **D13（本任务的依据，尤其是 h、i）**，`01_integration_rules.md` §5。先把 `origin/hazard/integration` 合并进 `hazard/T05`，保证能看到 D13 与 T06 的代码。

与 T06-F 并行：T06-F 改 `spec/manifest.py`、`runtime.py`、`driving.py`、`manifest/`、`cli.py` 与 `tests/pipeline/`，**这些文件你都不要碰**；你的范围是 `hazardlib/src/hazardlib/conformance/`、`configs/hazard/conformance.yaml`、`configs/hazard/fakehost.yaml` 与对应测试。

## 目标
把 D13 h 中属于一致性套件的部分落到代码与配置里。每一项一个小提交；既有测试保持绿，`CAUGHT_BY` 与 `LOCAL_DEFECTS` 的抓取表**只能增加、不能减少**。

## 范围

1. **C5 落地时间容差以物理步为单位**（D13 h）。把 `c5_release_gravity.free_fall_time_rel_tol` 换成以物理步计的键（例如 `free_fall_time_tol_physics_steps`）；判据为 \(|t_{测} - \sqrt{2h/g_{测}}| \le n \cdot dt\)，`dt` 取 BD 的 `physics_dt_s`。
   - 推导写进 YAML 注释：场景接触在步末才检测的引擎，落地时刻的量化误差至多一个物理步；FakeHost 有 CCD，e_ok 是浮点量级。
   - `gravity_rel_tol` 与落地时间容差的耦合（\(t \propto g^{-1/2}\)）按新单位重新推导，注释写清公式；`test_conformance_config.py` 的耦合检查同步。
   - **约束**：`DampedFall`、`GravityMismatch` 等既有变异体必须仍被抓到。若容差改成一步后抓不到，按“相对量化误差 \(n\,dt/t_{land}\) 随下落时间变小、阻尼效应随下落时间变大”的关系加大探针的下落高度（`probe` 段或 `fakehost.yaml` 的场景量），不得反过来缩小 n 去迁就变异体；推导与实测数字写进汇报。
2. **statistical 宿主的 C2 / C7 / C12 判据与按观测种类的 C8 上限**（D13 h）。
   - `c8_determinism.statistical_max_abs_diff` 拆成按观测种类的上限，至少分“状态向量”与“图像”两类（图像对每个相机分别比较，单位是灰度级）；值保持 `PLACEHOLDER`，注释写明由 statistical 宿主的 H 任务实测后定。
   - BD 为 `same_seed_repro: statistical` 时，C2 / C7 / C12 不再要求逐值一致：把“有危险体对照”与“无危险运行”的差异按同一种类与 C8 的上限比较。上限仍是 `PLACEHOLDER` 时这几项记 `unverified`（原因写“测了，未判”，并报告实测量级），不记 `fail`、不跳过。
   - 成功判定：C8 的同种子重复里成功判定一致时，C2 / C7 / C12 仍要求成功判定一致；不一致时对成功判定记 `unverified` 并写明原因。
   - `exact` 宿主的行为一字不变（逐值一致）。
   - 测试：在 `test_mutants.py` 现有 `NoisyReset` 的基础上，覆盖“statistical 宿主，上限为 PLACEHOLDER → unverified；上限为实测量级 2 倍 → pass；0.5 倍 → fail”，三项各一组；再加一个 statistical 宿主上真的泄漏状态（差异远超上限）的缺陷，确认 C2 / C7 判 `fail`。
3. **报告不含本机绝对路径**。`config_source` 现在写的是配置目录的绝对路径。改为：配置目录在仓库内时写相对仓库根的路径（如 `configs/hazard`）；在仓库外时只写 `external:<目录名>`。`config_hashes` 已足以确认内容。同步 `report.py` 的格式说明与 `read_report` 的校验，加一个测试断言报告 JSON 里不出现 `os.path.expanduser("~")` 与仓库根的绝对路径。
4. **文档字符串与 README** 同步 D13：C3 的穿越速度定义、C5 阻尼为 0、C11 的 `inject_engine_fault`、C12 适用于 `num_envs > 1`、`CapabilityError` 记 `fail`（这些代码已按此实现，只核对措辞与 01 §5 一致）。
5. **汇报**：在 `docs/design/reports/T05.md` 末尾追加 “T05-F” 一节：每项的验证命令与真实输出、新旧键与推导、抓取表前后对比（应只增不减）、偏离与理由。

## 验收标准（贴输出）
1. `pytest hazardlib -q` 在 `hz38`、`hz311` 全绿（贴总数，对比合并后集成分支的 1257 + 1 skip / 1258）。
2. `python -m hazardlib.conformance --host fake --out /tmp/t05f.json` 在两个环境都是 13 项 `pass`；报告里没有本机绝对路径（贴 `grep` 输出）。
3. 抓取表：10 个内置变异体与全部测试本地缺陷的失败检查与 T05 一致或更多（贴对比）。
4. `conformance.yaml` 里 `PLACEHOLDER` 只剩 C8 的 statistical 上限；每个容差都有推导注释（`test_conformance_config.py` 通过）。
5. `git status` 干净；只改了上面列出的范围（贴 `git diff --stat origin/hazard/integration`）。

## 不要做
不改 `docs/design/00–03`、`src/dynisafety/`、T06-F 的文件（见上）；不收紧或放宽 C1–C13 的其它容差（统一收紧在 H1、H2 之后由协调者做，D13 i）；不为通过测试删减变异体或断言。
