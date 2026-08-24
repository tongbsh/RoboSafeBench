# Migration: old commands → new commands

DyniSafety’s supported entry is the **`dynisafety` CLI** (`pip install -e .`
with a backend extra — see [installation.md](installation.md)).
Shell wrappers under the old root `scripts/` tree were removed or archived in M9.
Do **not** resurrect text-log → regex aggregation (iron rule 2); use JSONL +
`dynisafety report generate|validate`.

## EpisodeRecord 1.x → 2.0

`SCHEMA_VERSION` 从 `"1.0"` / `"1.1"` 升到 `"2.0"`。这是**破坏性**变更：`EpisodeRecord.from_json_dict` **拒绝读取** 1.x 文件，而不是尽力兼容。错误信息指向本文。

**为什么拒绝而不是改写字段。** 2.0 删掉了三个 legacy 反应性字段：

| 已移除（1.x） | 旧语义 |
|---------------|--------|
| `reaction_latency_steps` | 扰动 onset 后首次速度突变的步数；`None` 表示无反应 |
| `velocity_ratio` | onset 后 / 前窗口平均速度比 |
| `behavior_class` | 四类词表：`collide` / `freeze` / `cautious` / `ignore` |

它们的替代物在 `EpisodeRecord.diagnostics`（`BehaviorDiagnostics.to_json_dict()`，见 [behavior_diagnostics.md](behavior_diagnostics.md)）以及 run 目录下的 `traces/`（逐步 EEF / 接触 / 动作）。新旧 **不是同名同义**：

- 旧 `behavior_class` 的 `ignore` / `cautious` / `freeze` / `collide` 与新的 9 类（`collide_unreacted` / `collide_reacted` / `evade_success` / `evade_fail` / `stop_recover` / `stop_no_recover` / `partial_response` / `no_response` / `not_triggered`）**词表与判定都不兼容**。
- 旧 `ReactivityAnalyzer` 会把**碰撞本身造成的速度突变**判成「反应」；新分析器把接触后的运动学变化排除在反应之外。同一数字、同一标签，含义已经变了。

因此把 1.x JSONL 里的 `behavior_class="ignore"` 读进 2.0 会静默误报。正确做法是**重跑**旧结果，而不是改写磁盘上的字段。

1.x 产物（`artifacts/` 下已落盘的 `episodes.jsonl` / `summary.json`）作废；`dynisafety report` 不会把它们当 2.0 读入。重跑后行为诊断在：

- 每条 `EpisodeRecord.diagnostics`（正式报表字段；`None` 时表格渲染 n/a，不填 0）
- `{results}/l{level}/{run_id}/traces/`（`epXXXX.jsonl.gz` + `index.json`）

`ReactivityAnalyzer` 已从 `core/metrics.py` 删除，职责由 `diagnostics/analyze.py` 承接。`BackendCapabilities.has_reactivity` 一并删除：四个 backend 都产出 `diagnostics`，不再用该 flag 做门控。

## Eval / sweep

| Old | New |
|-----|-----|
| `bash scripts/eval_{spatial,object,goal,long}_all.sh …` | `dynisafety sweep --suite paper_main` (or `smoke`); single cell: `dynisafety eval …` |
| `bash scripts/eval_dynisafety.sh <SUITE> <TASK> <LEVEL> …` (file was already missing) | `dynisafety eval --backend libero --suite <SUITE> --task <id> --level <LEVEL> --model <m> …` |
| `bash scripts/dynamicLibero/eval_dynisafety.sh <SUITE> <TASK> <LEVEL>` | same `dynisafety eval --backend libero …` (all of `--backend` / `--model` / `--suite` / `--level` required; no business defaults) |
| `bash scripts/dynamicLibero/eval_level{1,2,3}_all_suites.sh` / `eval_all.sh` | `dynisafety sweep --suite …` |
| `cd dynamic_SimplerEnv && bash scripts/run_dynisafety.sh …` | `dynisafety eval --backend simplerenv --model … --suite bridge --level …` (or `sweep`) |
| Backend-local `dynisafety_libero/scripts/*.sh` / `dynamic_SimplerEnv/scripts/*.sh` (may still be on disk; C2 cutover pending) | Prefer `dynisafety eval` / `sweep`; do not use for paper reproduction |

### CLI shape (reference only)

```bash
dynisafety eval \
  --backend {libero,simplerenv,robotwin,robocasa} \
  --model <registry_or_yaml_stem> \
  --suite <suite_id> \
  --level <0..3> \
  [--task <int>]   # omit = all tasks from Backend.list_tasks
  [--dry-run]

dynisafety sweep --suite smoke|--suite smoke_robocasa|--suite paper_main [--gpus 0,1] [--resume] [--retry-failed] [--dry-run]
```

Suite ids are **backend-specific** (e.g. `safelibero_spatial`, `bridge`,
`main20`, `robocasa_nav`). See README Quick Start table. `robocasa` is not in
`paper_main`.

RoboCasa kitchen-nav (numpy 2 env `dynisafety-robocasa`; `random_mobile`):

```bash
dynisafety eval --backend robocasa --model random_mobile --suite robocasa_nav --level 0 --dry-run
dynisafety sweep --suite smoke_robocasa --dry-run
```

## Policy / OpenPI server

| Old | New |
|-----|-----|
| `bash scripts/simplerEnv/start_openpi_server.sh` (**deleted**) | Prefer `dynisafety serve --model pi0` (or the model YAML that defines the serve spec). Fallback: run upstream OpenPI from your checkout (no new DyniSafety shell wrapper). |

## Checkpoints / assets

| Old | New |
|-----|-----|
| Root `download_hf.py` (hard-coded machine paths) | `python scripts/install/download_assets.py` (`--dest` / `$DYNISAFETY_CHECKPOINTS`). `download_hf.py` is deprecated (wrapper in docs pass); do not use old absolute paths. |
| Manual MANO copy into tracked tree | User-obtained MANO under `<workspace>/assets/mano_v1_2/` (gitignored); `python scripts/install/download_assets.py --mano-notice` |

## RLBench / BridgeVLA

| Old | New |
|-----|-----|
| `bash scripts/dynamicRLBench/level{1,2,3}_*.sh` | **Archived** at `old_cache/legacy_rlbench/dynamicRLBench/` (local only). Backend **not** restored; do not use for paper runs. |

## Aggregation / outputs

| Old | New |
|-----|-----|
| `python …/aggregate_logs.py` (regex over text logs) | `dynisafety report generate` / `validate` over `results/**/episodes.jsonl` (see [contracts.md](contracts.md) §6). Required flags include `--results`, `--out`, `--grid`, `--report-yaml` (generate also needs `--metrics-yaml`). See [reproducing_paper.md](reproducing_paper.md). |
| `**/outputs*/`, `outputs_report/`, `outputs_past*` | Canonical layout: `results/l{level}/<run_id>/` (`resolved_config.yaml`, `episodes.jsonl`, `summary.json`, …). Legacy flat `results/<run_id>/` still resume-compatible. |

## Docs / license

| Old | New |
|-----|-----|
| `docs/core_metrics_design.md` | Metrics: [metrics.md](metrics.md); level design: [benchmark_design.md](benchmark_design.md). Historical copy (local only): `old_cache/docs_internal/archive/` |
| `docs/dynisafety_libero_dynamic_levels_1_3.md` | [benchmark_design.md](benchmark_design.md) + `configs/backends/*/levels/` |
| Root / SimplerEnv LICENSE | [LICENSE](../LICENSE) (MIT); third-party notices: [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) |
| Scattered install notes | [installation.md](installation.md) (canonical); pins in [third_party.md](third_party.md) |

## Offline archives (developer machines only)

These paths are **not** part of the published protocol. They may exist on
machines that ran the pre-refactor archive pass (§0.2). Reviewers should ignore
them; substitute your own backup location if needed.

| Path (example local archive root) | Contents |
|-----------------------------------|----------|
| `<archive_root>/outputs/` | Former in-repo `outputs*` trees |
| `<archive_root>/golden_logs_*.tar.gz` | Golden logs for M12 (do not delete if present) |
| `<archive_root>/past_files_cache/` | Former repo `past_files_cache/` |
| `<archive_root>/paper_report/` | Former `paper_report/` demos / figure clips |
