# 仓库目录

用户向导览：每个顶层目录干什么。更细的内部盘点在 [`../paper/仓库导览.md`](../paper/仓库导览.md)（过程稿，默认不入库）。

评测主路径：

```text
dynisafety eval / sweep
  → config/loader.py 合并 YAML
  → backends/* 建环境、注入 L0–L3、rollout
  → policies/adapters 出动作
  → runner/loop.py 写入
       {out}/eval/{backend}/{suite}/l{level}/{run_id}/
  → cli/report 只读 JSONL 出表
```

`{out}` 见 `configs/paths.yaml`（`DYNI_OUT` / `{workspace}/out`）。`results/` 是旧树，resume / 只读报告还会扫。

默认值只许写在 `configs/` 的 YAML 里。Python / CLI 不写业务超参。

---

## 顶层

| 路径 | 一句话 |
|------|--------|
| `README.md` | 对外入口 |
| `src/dynisafety/` | 评测框架 |
| `configs/` | **唯一**超参默认；`paths.yaml` 是 I/O |
| `tests/` | 契约 / 接线 / 黄金快照 |
| `docs/user/` | 本手册（从 [README.md](README.md) 进） |
| `docs/paper/` | 过程稿 |
| `scripts/` | 安装、一键评测（包装 CLI） |
| `tools/` | 模型审计、环境名解析 |
| `environment/` `requirements/` `docker/` | conda 配方、短 pin、LIBERO 镜像 |
| `patches/` | 上游最小 diff |
| `examples/` `templates/` | 接入示范 |
| `third_party/` | 仿真器树（submodule 或 install） |
| `external/` | 本机大文件：权重、模型仓、数据集（gitignore） |
| `out/` | 规范 I/O（gitignore） |
| `results/` | 旧评测树 |

---

## `src/dynisafety/`

| 目录 | 作用 |
|------|------|
| `cli/` | `eval` / `sweep` / `serve` / `report` / `doctor` / `config` |
| `config/` | YAML 加载、合并、路径 |
| `backends/` | `libero` / `simplerenv` / `robotwin` / `robocasa` |
| `policies/adapters/` | 每个模型一个 adapter |
| `runner/` | rollout 循环、调度、conda/端口 |
| `core/` | EpisodeRecord、指标、写盘 |
| `report/` `diagnostics/` | 基础指标表图、行为诊断 |

并行旁路：`dynisafety parallel-eval`（RoboTwin + X-VLA Infer/Env 拆进程），不是论文主入口。

---

## `configs/`

| 路径 | 作用 |
|------|------|
| `base.yaml` | 全局默认 |
| `suites/paper_main.yaml` | 论文矩阵 |
| `suites/smoke*.yaml` | 冒烟 |
| `backends/<b>/` | 该仿真器的 level / suite / task |
| `models/<stem>.yaml` | 一个模型一份；含 `eval_env` / `server.env` |
| `paths.yaml` | `out/` 布局 |
| `report/` | grid 与 report 设置 |

---

## `scripts/`

| 目录 | 给谁 | 入口 |
|------|------|------|
| `setup/` | 装机器 | [install.md](install.md)、`install_conda_env.sh` |
| `eval/` | 正式评测 | [eval.md](eval.md)、`00_smoke.sh` → `02_report.sh` |
| `extra/` | LingBot-VA serve / GSO overlay | `scripts/extra/README.md` |
| `lib/env.sh` | 被 source | 切 conda、`dyni_cli` |

`tools/model_env.py`：从 YAML 解析 eval/serve conda 名（`scripts/lib/env.sh` 调用）。

---

## 加一个模型

一份 adapter + 一份 YAML，不要新写用户向 shell。步骤：[adding_a_model.md](repo.md)。仓外示例：[../../examples/example_agent/README.md](../../examples/example_agent/README.md)。

旧命令对照：[migration.md](repo.md)。


---

# 接入新模型（原 adding_a_model.md）

# Adding a VLA model to DyniSafety

Goal: add a new model by writing **one adapter file** and **one YAML**, then
verify with `dynisafety models` and a smoke eval (no new shell entrypoints).

A complete out-of-tree example (adapter + YAML + tests, no `src/` edits) is
[examples/example_agent/README.md](../../examples/example_agent/README.md).

## Prerequisites

- Editable install for one backend: `pip install -e ".[<backend>]"` — see
  [install.md](install.md) (extras are mutually exclusive).
- Read `docs/user/eval.md` §4 (Policy protocol) — do not invent a second signature.
- Iron law 1: every hyperparameter default lives in YAML once. Adapters and CLI
  must not hardcode ports, chunk sizes, or timeouts.
- Confirm the eval loop / backend with `dynisafety doctor --backend <b>` before
  live runs.

## Steps

### 1. Copy the template

```bash
cp templates/model_adapter_template.py \
   src/dynisafety/policies/adapters/my_model.py
```

Rename `TemplateAdapter` → `MyModelAdapter`. Implement:

| Method | Responsibility |
|--------|----------------|
| `from_config(cfg)` | Validate `cfg.transport`; connect remote via `connect_transport(cfg)` |
| `reset(ctx)` | Clear episode state; optional `transport.reset` |
| `act(obs)` | Read `Observation.images` / `state`; return `ActionChunk` |
| `close()` | Release sockets / GPU |

**Forbidden in adapters**

- `requests.post` / `websockets.connect` / `zmq.Context`
- `task_env.take_action(...)`
- `conda activate` / hardcoded ports

Remote models must call `dynisafety.policies.adapters._common.connect_transport`.

### 2. Write `configs/models/my_model.yaml`

Use the frozen layout (see `configs/models/starvla.yaml` or `random.yaml`):

```yaml
name: my_model
transport: local          # or http | websocket | zmq
adapter: dynisafety.policies.adapters.my_model:MyModelAdapter
action_space: delta_ee_7d # or abs_ee_7d | abs_ee_16d | qpos_14d | hybrid_mobile_12d
chunk_size: 8
observation:
  images: [primary, wrist]
  needs_state: true
# server:                 # required when transport != local
#   env: myenv
#   root: ${env:MY_ROOT|${workspace}/third_party/my_model}
#   launch: serve.py
#   default_port: 8000
#   health_path: /health
#   startup_timeout_s: 600
#   args: ["--ckpt", "{checkpoint}", "--port", "{port}"]
checkpoint:
  default: "${env:MY_CKPT|${workspace}/checkpoints/my_model}"
resources:
  max_jobs_per_gpu: 2
backends: [libero]        # subset of libero | simplerenv | robotwin | robocasa
extra:
  # model-specific required keys — no silent defaults in Python
  seed: 0                 # example for RandomAdapter-style local models
  eval_env: openvla-oft   # conda that runs dynisafety eval (must have the sim)
  # serve_env: myenv      # or server.env — conda that runs the inference server
```

HTTP transports also need under `extra`: `act_timeout_s`, `reset_path`,
`act_path`, `server_host`. WebSocket needs `ws_codec` (`openpi`|`starvla`|`lingbot`)
and codec-specific keys (see existing YAMLs).

`extra.eval_env` / `server.env` are literal conda names. Reuse an existing
paper env when the simulator matches; do not invent a parallel `dynisafety-*`
name on a machine that already works. Matrix: [install.md](install.md).

`server.args` is the argv tail handed to `python <root>/<launch>`; the process runs
with `cwd` set to `root`. Always available: `{port}` `{gpu}` `{checkpoint}` `{model}`
`{suite}`. Any other placeholder must be supplied per suite:

```yaml
server:
  args: ["--ckpt", "{checkpoint}", "--unnorm_key", "{unnorm_key}", "--port", "{port}"]
  per_suite:
    safelibero_object: {unnorm_key: libero_object_no_noops}
    bridge:            {unnorm_key: bridge_oxe, args: [...]}   # `args` overrides the tail
```

Use this whenever the server loads suite-specific weights: the checkpoint is read at
startup, so `dynisafety serve` requires `--suite` and a sweep starts one server
generation per suite. Unresolved placeholders are reported at config-load time, not
after the model has loaded.

### 3. List models

```bash
dynisafety models
```

Your stem must appear. States:

| state | Meaning |
|-------|---------|
| `ready` | Adapter imports and checkpoint path looks OK |
| `missing_dependency` | Import failed (package not installed) |
| `missing_checkpoint` | Local checkpoint path missing |
| `load_error` | YAML/config/adapter error (detail string) |

Listing always prints every declared YAML (including failures). There is no
`memoryvla` entry by design.

### 4. Smoke with the random reference model

Before wiring heavy weights, confirm the loop accepts a Policy:

```bash
dynisafety eval \
  --backend libero \
  --model random \
  --suite safelibero_spatial \
  --level 0 \
  --dry-run
```

(`--dry-run` prints the plan without starting the simulator.)

### 5. End-to-end example (recommended)

**A — Dry-run (no simulator):** after adding `configs/models/my_model.yaml` with
`backends` including `libero` (or your target):

```bash
dynisafety eval \
  --backend libero \
  --model my_model \
  --suite safelibero_spatial \
  --level 0 \
  --dry-run

dynisafety sweep --suite smoke --only model=my_model --dry-run
```

**B — Live smoke (needs install + checkpoint):** backends are registered for
`libero` / `simplerenv` / `robotwin` / `robocasa`. Prefer a local
`transport: local` fake or `random` / `random_mobile` first, then your weights:

```bash
dynisafety doctor --backend libero
dynisafety sweep --suite smoke
# RoboCasa kitchen-nav (numpy 2 env; not paper_main):
# dynisafety doctor --backend robocasa
# dynisafety eval --backend robocasa --model random_mobile --suite robocasa_nav --level 0 --dry-run
# dynisafety sweep --suite smoke_robocasa --dry-run
# or only your model, once it is listed in a suite matrix / --only filter:
# dynisafety sweep --suite smoke --only model=my_model
```

If live smoke fails on scheduler / env naming issues, fix the adapter or model
YAML — do not add shell wrappers as a workaround.

### 6. Checklist before PR

- [ ] `configs/models/<id>.yaml` uses the frozen keys only
- [ ] `adapter:` import path resolves
- [ ] `grep -rn "requests.post\\|websockets.connect\\|zmq.Context" src/dynisafety/policies/adapters/` is empty for your file
- [ ] No new shell user-entry script
- [ ] `pytest tests/test_policies.py -q` passes
- [ ] LingBot-like servers: `extra.dump_artifacts: false` (never leave debug dumps on)

## Reference adapters

| Need | Example |
|------|---------|
| Local OFT | `adapters/openvla_oft.py` + `configs/models/openvla_oft.yaml` |
| HTTP | `adapters/univla.py` + `univla.yaml` |
| HTTP (examples payload) | `adapters/turbovla.py` + `turbovla.yaml` |
| HTTP (VLA-Adapter-Pro) | `adapters/vla_adapter.py` + `vla_adapter.yaml`（chunk 8 / `delta_ee_7d` / 夹爪 invert 在 serve） |
| WebSocket + codec | `adapters/pi0.py` / `starvla.py` / `lingbot_va.py` |
| ZMQ (GR00T) | `adapters/gr00t.py` |
| Shared class + variant | `adapters/xvla.py` (`extra.variant`) |
| Smoke / tutorial | `adapters/random.py` (`random`, `random_mobile`) |
| Out-of-tree example | `examples/example_agent/` ([README](../../examples/example_agent/README.md)) |


---

# 旧命令迁移（原 migration.md）

# Migration: old commands → new commands

DyniSafety’s supported entry is the **`dynisafety` CLI** (`pip install -e .`
with a backend extra — see [install.md](install.md)).
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

它们的替代物在 `EpisodeRecord.diagnostics`（`BehaviorDiagnostics.to_json_dict()`，见 [behavior_diagnostics.md](eval.md)）以及 run 目录下的 `traces/`（逐步 EEF / 接触 / 动作）。新旧 **不是同名同义**：

- 旧 `behavior_class` 的 `ignore` / `cautious` / `freeze` / `collide` 与新的 9 类（`collide_unreacted` / `collide_reacted` / `evade_success` / `evade_fail` / `stop_recover` / `stop_no_recover` / `partial_response` / `no_response` / `not_triggered`）**词表与判定都不兼容**。
- 旧 `ReactivityAnalyzer` 会把**碰撞本身造成的速度突变**判成「反应」；新分析器把接触后的运动学变化排除在反应之外。同一数字、同一标签，含义已经变了。
- 2026-08-29 起默认 `reacted` / `reaction_rate` 是**闭环**（接触前运动学偏离 **且** `latency_chunks ≥ min_latency_chunks`，默认 1 个 action chunk）。15° 运动学位改叫 `reacted_kinematic`。旧 JSONL 没有该字段时，报表把 `reaction_rate` 写成 n/a，避免把运动学率当成闭环率。论文数字须 `bash scripts/eval/recompute_diagnostics.sh`（`dynisafety diagnostics recompute`）从 `traces/` 重写 `diagnostics` 后再出报表。

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
| Root `download_hf.py` (hard-coded machine paths) | `python scripts/setup/download_assets.py` (`--dest` / `$DYNISAFETY_CHECKPOINTS`). `download_hf.py` is deprecated (wrapper in docs pass); do not use old absolute paths. |
| Manual MANO copy into tracked tree | User-obtained MANO under `<workspace>/assets/mano_v1_2/` (gitignored); `python scripts/setup/download_assets.py --mano-notice` |

## RLBench / BridgeVLA

| Old | New |
|-----|-----|
| `bash scripts/extra/dynamicRLBench/level{1,2,3}_*.sh` | **Archived**. Backend **not** restored; do not use for paper runs. |

## Aggregation / outputs

| Old | New |
|-----|-----|
| `python …/aggregate_logs.py` (regex over text logs) | `dynisafety report generate` / `validate` over `results/**/episodes.jsonl` (see [contracts.md](eval.md) §6). Required flags include `--results`, `--out`, `--grid`, `--report-yaml` (generate also needs `--metrics-yaml`). See [eval.md](eval.md). |
| `**/outputs*/`, `outputs_report/`, `outputs_past*` | Canonical layout: `results/l{level}/<run_id>/` (`resolved_config.yaml`, `episodes.jsonl`, `summary.json`, …). Legacy flat `results/<run_id>/` still resume-compatible. |

## Docs / license

| Old | New |
|-----|-----|
| `docs/core_metrics_design.md` | Metrics: [metrics.md](eval.md); level design: [benchmark_design.md](eval.md) |
| `docs/dynisafety_libero_dynamic_levels_1_3.md` | [benchmark_design.md](eval.md) + `configs/backends/*/levels/` |
| Root / SimplerEnv LICENSE | [LICENSE](../../LICENSE) (MIT); third-party notices: [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) |
| Scattered install notes | [install.md](install.md); pins in [third_party.md](install.md) |

## Offline archives (developer machines only)

These paths are **not** part of the published protocol. They may exist on
machines that ran an older layout; use your own backup location if needed.

| Path (example local archive root) | Contents |
|-----------------------------------|----------|
| `<archive_root>/outputs/` | Former in-repo `outputs*` trees |
| `<archive_root>/golden_logs_*.tar.gz` | Archived golden logs (optional; set `DYNI_GOLDEN_TAR` to use them) |
| `<archive_root>/past_files_cache/` | Former repo `past_files_cache/` |
| `<archive_root>/paper_report/` | Former `paper_report/` demos / figure clips |
