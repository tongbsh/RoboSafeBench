# Reproducing the DyniSafety paper tables

End-to-end path from a clean checkout to the main paper tables/figures.
Authoritative experiment matrix: **`configs/suites/paper_main.yaml`** (do not
fork a second matrix in this doc).

Anonymous / double-blind reviews: omit author metadata; use public clone URL
and local checkpoint caches only.

---

## 1. Hardware and wall-clock

| Item | Value |
|------|--------|
| GPU | **TBD** (fill with the paper’s measured setup) |
| CPU / RAM | **TBD** |
| Wall-clock for full `paper_main` | **TBD** |
| Approx. GPU-hours | **TBD** |

Until filled, size clusters from the job count below and your per-episode
latency. Prefer `--resume` on long sweeps.

---

## 2. Install

Follow **[installation.md](installation.md)** for **each** backend you will run
(separate conda envs; extras are mutually exclusive):

```bash
make install-libero        # and/or install-simplerenv / install-robotwin
# RoboCasa kitchen-nav is a separate numpy-2 env; not in paper_main:
# make install-robocasa && dynisafety doctor --backend robocasa
dynisafety doctor --backend libero
```

Sanity (no heavy load):

```bash
dynisafety sweep --suite smoke --dry-run
```

---

## 3. Checkpoints and assets

```bash
python scripts/install/download_assets.py --help
# Prefer --dest or $DYNISAFETY_CHECKPOINTS; never commit weights.
python scripts/install/download_assets.py --mano-notice   # L3 hand; not redistributed
```

Model stems in `paper_main` must resolve via `configs/models/<stem>.yaml`
(checkpoint paths / server specs). Remote models need `dynisafety serve` (or
upstream servers) as described in [adding_a_model.md](adding_a_model.md) /
model YAML.

---

## 4. Paper-main matrix (from YAML comments)

From `configs/suites/paper_main.yaml` (seeds `[7]` only):

| Backend | Expansion (jobs × episodes) |
|---------|-----------------------------|
| `libero` | 4 suites × 4 tasks × 4 levels × 7 models × 1 seed = **448** jobs × **50** ep |
| `robotwin` | 1 suite × 20 tasks × 4 levels × 5 models × 1 seed = **400** jobs × **20** ep |
| `simplerenv` | 1 suite × 4 tasks × 4 levels × 3 models × 1 seed = **48** jobs × **50** ep |
| **Total** | **896** jobs |

Models / suites are exactly those listed in the YAML `matrix:` block (see also
README model table).

**Protocol footnotes (required when publishing numbers):**

- SimplerEnv bridge: `runtime.max_steps: 120` in
  `configs/backends/simplerenv/suites/bridge.yaml` — not comparable to 60-step
  literature baselines without disclosure ([benchmark_design.md](benchmark_design.md)).
- Force columns: not comparable across `force_source`
  ([metrics.md](metrics.md)).
- RoboTwin episode seeds use `runtime.seed_stride` from config
  ([configuration.md](configuration.md)).
- **RoboCasa is not in `paper_main`.** Kitchen wheeled-nav (`robocasa_nav`) is a
  separate protocol (`configs/suites/smoke_robocasa.yaml`). Do not mix its 450-step
  / 20 Hz clock or kitchen-floor forces with the tabletop matrix.

---

## 5. Run the sweep

Plan only:

```bash
dynisafety sweep --suite paper_main --dry-run
# or: --suite configs/suites/paper_main.yaml
```

Live (example multi-GPU):

```bash
dynisafety sweep --suite paper_main --gpus 0,1,2,3
dynisafety sweep --suite paper_main --gpus 0,1,2,3 --resume
dynisafety sweep --suite paper_main --retry-failed
```

Filter a slice while debugging:

```bash
dynisafety sweep --suite paper_main --only backend=libero,level=0 --dry-run
```

Outputs land under `results/` per [contracts.md](contracts.md) §6
(`episodes.jsonl`, `resolved_config.yaml`, `summary.json`, …). **Do not**
aggregate from text logs.

Single-cell debug (all of `--backend/--model/--suite/--level` required):

```bash
dynisafety eval \
  --backend libero \
  --model openvla_oft \
  --suite safelibero_spatial \
  --level 0 \
  --task 0
```

---

## 6. Report / validate

CLI requires explicit paths and settings YAMLs (no business defaults on the
CLI). `--metrics-yaml` is typically `configs/base.yaml`. `--grid` should encode
the same coverage as `paper_main` (write a grid YAML that lists expected
backend/suite/task/level/model/seed cells, or reuse a grid you keep next to the
run). `--report-yaml` must define bootstrap / confidence / `max_error_rate`
keys expected by `dynisafety report` (see `dynisafety report generate --help`
and [metrics.md](metrics.md)).

Example shape (adjust paths):

```bash
dynisafety report validate \
  --results results/ \
  --out reports/ \
  --grid path/to/paper_main_grid.yaml \
  --report-yaml path/to/report_settings.yaml

dynisafety report generate \
  --results results/ \
  --out reports/ \
  --grid path/to/paper_main_grid.yaml \
  --report-yaml path/to/report_settings.yaml \
  --metrics-yaml configs/base.yaml
```

Incomplete coverage refuses tables unless `--allow-incomplete`.

### Table / figure ↔ command

| Paper artifact | Producer |
|----------------|----------|
| Main metric table | `reports/table_main.{md,tex,csv}` from `report generate` |
| Per-task breakdown | `reports/table_per_task.csv` |
| Ablation-style table | `reports/table_ablation.csv` |
| Coverage gaps | `reports/coverage.csv` (`validate` or `generate`) |
| SR/CR vs level plot | `reports/plots/level_sr_cr.png` |
| Force distribution | `reports/plots/force_box.png` |
| Run-level tidy metrics | `reports/runs.csv` (+ optional `episodes.parquet`) |

Display names (`SR`, `FPeakMean(N)`, …) are render-only; canonical IDs are in
[metrics.md](metrics.md) / [contracts.md](contracts.md).

---

## 7. Checklist before claiming reproduction

- [ ] Three backend envs installed for `paper_main`; `doctor` reviewed  
  (RoboCasa is a fourth env and **not** in `paper_main`)  
- [ ] Checkpoints match `configs/models/*.yaml`  
- [ ] Full `paper_main` finished or coverage explained with `--allow-incomplete`  
- [ ] Tables footnote **120-step** SimplerEnv budget and **force non-comparability**  
- [ ] `config_hash` / git dirty state recorded from run manifests  

Optional submission format draft: [leaderboard.md](leaderboard.md).
