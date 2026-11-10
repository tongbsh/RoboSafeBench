# H3 — 原始 LIBERO 适配器（BENCH=libero；D18）

2026-11-10 用户更正覆盖本文件原来的 SafeLIBERO 方案。以原始 LIBERO 为宿主，四套件 `libero_spatial` / `libero_object` / `libero_goal` / `libero_10`，每套件 10 任务。

当前实现与验收见 `docs/design/reports/H-libero.md`；代码在当前仓库 `hazardlib/src/hazardhosts/libero/`。旧 H3 worktree 的四个草稿已复用并改造，其 SafeLIBERO 假设不再适用。

要求：
1. 原始 BDDL、任务物体、相机、原生成功谓词、官方任务顺序保持；不 import SafeLIBERO，不映射 `safelibero_*` 套件。
2. 危险体通过内存 MJCF 追加独立自由刚体；按名字映射演示与官方 init 状态。停放隔离要同时有 C2 与原始场景回放对照。
3. 遵守 H00 的物理步钩子、几何、接触语义、严格 BD、C1–C13、原生循环 item 14、T_stop、门控与成本账本要求。
4. spatial 前四个官方任务，每任务前 10 集演示；另验证 goal 与 long 各一任务。回放失败计数，不从成功率分母里静默删除。
5. 仅成功参考回放生成 TaskContext；试点单来源偏离明确记录。被测 π₀.₅ 不作探针来源。
6. G1 停等与 G2 分开；G2 未实现时记 uncertified。滑入构造失败、不可见落物、低暴露都如实报告。
7. 旧 SafeLIBERO 手工 L1–L3 映射要求已撤销，历史报告保留。
8. R1-L 在相同布局、种子与名义时机上配对危险 / 无危险回合；复用现有模型适配器，输出实际策略相机 MP4、时间戳、JSONL 与试点安全表。

环境：使用 `env_notes/libero.md` 的 **upstream** LIBERO_CONFIG_PATH；其余 GPU 与版本约束不变。不得改上游文件或旧 `src/dynisafety` 后端。
