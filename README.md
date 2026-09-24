# DyniSafety Benchmark

DyniSafety is a dynamic physical safety benchmark for Vision-Language-Action
(VLA) policies: it measures whether a policy keeps task success while responding
safely to moving obstacles and human-like disturbances across four simulators
(LIBERO / MuJoCo tabletop, SimplerEnv / SAPIEN 2, RoboTwin / SAPIEN 3,
RoboCasa / MuJoCo kitchen **wheeled** navigation).

RoboCasa evaluates **dynamic obstacles on the kitchen floor** during PandaOmron
wheeled navigation (`NavigateKitchen`). It is **not** bipedal locomotion and
**not** countertop manipulation. It is **not** in
`configs/suites/paper_main.yaml`.

Reported scores are **simulator-internal safety indices**. They are **not**
deployment-grade physical safety certification. Absolute contact-force values
must **not** be compared across simulators or scenes (`force_source` differs;
kitchen-floor contacts are not tabletop newtons). Step clocks (`max_steps`,
`move_*_step`) must **not** be compared across backends. See
[docs/user/eval.md](docs/user/eval.md) and
[docs/user/eval.md](docs/user/eval.md).

> **Anonymous submission.** Author names and affiliations are withheld
> (`TODO` in the citation block). Do not treat placeholder metadata as final.

---

## Teaser

_TODO: teaser figure (levels L0–L3 or multi-backend collage). Image asset not
shipped in this revision — replace this paragraph with a markdown image when
available._

---

## Quick Start (5-minute smoke)

User guides: **[docs/user/README.md](docs/user/README.md)**.
Install one backend first ([docs/user/install.md](docs/user/install.md)).
Activate the YAML eval env; do not rename an env that already works:

```bash
conda activate openvla-oft   # libero paper eval name
make install-libero          # or: scripts/setup/install_backend.sh libero
pip install -e ".[libero]"   # extras are mutually exclusive
dynisafety doctor --backend libero
```

RoboCasa uses a **separate NumPy 2** env (`dynisafety-robocasa`); do not install
it into the LIBERO / SimplerEnv / RoboTwin envs:

```bash
conda activate dynisafety-robocasa
export MUJOCO_GL=egl
make install-robocasa
dynisafety doctor --backend robocasa
```

**Recommended first command** (matrix in `configs/suites/smoke.yaml`:
libero × `openvla_oft` × L0 × 1 episode):

```bash
dynisafety sweep --suite smoke
```

RoboCasa kitchen-nav smoke (not `paper_main`; uses `random_mobile`):

```bash
dynisafety sweep --suite smoke_robocasa --dry-run
```

Dry-run without launching the simulator:

```bash
dynisafety sweep --suite smoke --dry-run
```

### Single-cell `eval` (same command shape; only `--backend` / suite change)

All of `--backend`, `--model`, `--suite`, and `--level` are **required** (no
CLI business defaults). Use the same flags every time; pick the row for your
backend. `--suite` is backend-specific (there is no shared suite id across
simulators).

| `--backend`   | example `--suite`   | example `--model` |
|---------------|---------------------|-------------------|
| `libero`      | `safelibero_spatial`| `random`          |
| `simplerenv`  | `bridge`            | `random`          |
| `robotwin`    | `main20`            | `random`          |
| `robocasa`    | `robocasa_nav`      | `random_mobile`   |

```bash
dynisafety eval --backend libero     --model random        --suite safelibero_spatial --level 0 --dry-run
dynisafety eval --backend simplerenv --model random        --suite bridge             --level 0 --dry-run
dynisafety eval --backend robotwin   --model random        --suite main20            --level 0 --dry-run
dynisafety eval --backend robocasa   --model random_mobile --suite robocasa_nav      --level 0 --dry-run
```

Inspect the fully merged config before a live run:

```bash
dynisafety config print --backend libero --level 0 --model random --suite safelibero_spatial
```

Aggregate only from JSONL (never from text-log regex). Full flag set is required
(`--grid`, `--report-yaml`, `--metrics-yaml`, …) — see `dynisafety report
generate --help` and [docs/user/eval.md](docs/user/eval.md); paper command sequence
in [docs/user/eval.md](docs/user/eval.md).

---

## Benchmark definition (disturbance levels)

| Level | Name | Intent |
|------:|------|--------|
| L0 | Baseline | Task execution with **no** injected dynamic disturbance |
| L1 | Sudden Drop | An obstacle suddenly enters the workspace (drop / spawn) |
| L2 | Sliding Obstacle | An obstacle moves through the workspace on a planned path |
| L3 | Shared Reaching | A simulated human hand reaches into the workspace (contact risk) |

Geometry, triggers, and physical parameters are defined in YAML under
`configs/backends/<backend>/levels/` (defaults live only there). Narrative and
comparability notes: [docs/user/eval.md](docs/user/eval.md).

The L1–L3 **display names above are for tabletop backends** (LIBERO / SimplerEnv
/ RoboTwin). RoboCasa uses a **parallel floor-nav ladder** (Frontal Sudden Appear
/ Base-Directed Approach / Approaching Human) — see the RoboCasa section in
[docs/user/eval.md](docs/user/eval.md). Do not treat RoboCasa L1–L3
as tabletop drop / arc-transit / shared reaching.

**SimplerEnv step budget:** all models use `runtime.max_steps: 120` on the
bridge suite (`configs/backends/simplerenv/suites/bridge.yaml`). L0 baselines
may be **higher** than published SimplerEnv numbers taken under a 60-step
budget — do not mix those protocols in one table.

**RoboCasa step budget:** `runtime.max_steps: 450` at `control_freq: 20` on
`robocasa_nav`. Do not compare those step indices to LIBERO / SimplerEnv /
RoboTwin clocks.

---

## Metrics

Episode fields are written as machine-readable `EpisodeRecord` lines in
`out/eval/{backend}/{suite}/l{level}/<run_id>/episodes.jsonl`. Aggregation and tables read JSONL only.

| Display | ID | Meaning | Cross-simulator |
|---------|-----|---------|-----------------|
| SR | `sr` | Task success rate | Comparable |
| CR | `cr` | Fraction of episodes with disturbance contact | Comparable |
| Topple | `topple_rate` | Object topple rate (L1/L2) | Comparable when defined; else n/a |
| FPeakMean(N) | `f_peak_mean_all` | Mean peak contact force; no-contact episodes count as **0 N** | **Not** comparable across `force_source` |
| SASR | `sasr` | Safety-adjusted success (`success * φ` from YAML) | Comparable as a rate |

Full definitions, injury bins, and report artifacts:
[docs/user/eval.md](docs/user/eval.md). Contracts:
[docs/user/eval.md](docs/user/eval.md).

---

## Supported models

Policy adapters live under `src/dynisafety/policies/adapters/`; each model has
one YAML stem in `configs/models/`. Paper-main intersection (authoritative
matrix: `configs/suites/paper_main.yaml`):

| Backend | Suites | Models (`configs/models/*.yaml` stems) |
|---------|--------|------------------------------------------|
| `libero` | `safelibero_{spatial,object,goal,long}` | `openvla_oft`, `xvla`, `starvla`, `pi0`, `pi05`, `openvla_oft_rlinf` |
| `simplerenv` | `bridge` | `starvla`, `internvla_m1`, `xvla_official` |
| `robotwin` | `main20` | `internvla_a1`, `fastwam`, `lingbot_va`, `starvla`, `xvla_official` |

`robocasa` is **not** in `paper_main`. First suite: `robocasa_nav` (NavigateKitchen
multi-destination). Smoke model: `random_mobile` (`hybrid_mobile_12d`).

Additional registered stems (smoke / servers / extras) include `random`,
`random_mobile`, `openvla`, `gr00t`, `dreamvla`, `octo-*`, `spatialvla*`,
and others under `configs/models/`. How to add a model:
[docs/user/repo.md](docs/user/repo.md).

---

## Installation

One backend per environment. Preferred entrypoints:

```bash
make install-libero        # or install-simplerenv / install-robotwin / install-robocasa
# equivalent: scripts/setup/install_backend.sh libero|simplerenv|robotwin|robocasa
```

Full guide: **[docs/user/install.md](docs/user/install.md)**
(index: [docs/user/README.md](docs/user/README.md)).
Pins: [docs/user/install.md](docs/user/install.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Reproducing the paper

End-to-end command sequence, job counts, and report mapping:
[docs/user/eval.md](docs/user/eval.md)
(hardware / wall-clock marked TBD until filled).

Sweep entry:

```bash
dynisafety sweep --suite paper_main --gpus 0,1,2,3
# resume / retry: see --help
```

---

## Adding a model

One adapter + one YAML — no new shell user entrypoints:
[docs/user/repo.md](docs/user/repo.md).
Runnable out-of-tree walkthrough: [examples/example_agent/README.md](examples/example_agent/README.md).

---

## Citation

Machine-readable stub: [CITATION.cff](CITATION.cff) (authors / venue = `TODO`
for anonymous review). BibTeX placeholder (do not invent names):

```bibtex
@article{dynisafety2025,
  title   = {DyniSafety: Benchmarking VLA Safety under Dynamic Physical Disturbances},
  author  = {TODO},
  journal = {TODO},
  year    = {2025}
}
```

Optional result-submission format: [docs/user/eval.md](docs/user/eval.md).

---

## License

This project is released under the **MIT** License — see [LICENSE](LICENSE).
Third-party simulators and assets retain their upstream terms; MANO / SMPL+H
models are **not** redistributed (obtain them yourself). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Repository layout

```text
.
├── src/dynisafety/          # Package: core, backends, policies, CLI
├── configs/                 # Sole source of hyperparameter defaults (YAML)
├── docs/
│   ├── user/                # Open-source user docs
│   └── paper/               # Paper/benchmark construction (not shipped)
├── examples/example_agent/  # Runnable out-of-tree agent integration walkthrough
├── scripts/                 # setup / eval / debug / extra
├── environment/             # optional conda recipes (names: docs/user/install.md)
├── requirements/            # Mutually exclusive backend pins
├── third_party/             # Simulators (submodules / install trees)
├── templates/               # Model adapter template
├── results/                 # Local runs (gitignored) → episodes.jsonl
└── LICENSE
```

Old shell entrypoints and layouts: [docs/user/repo.md](docs/user/repo.md).
Download checkpoints with `scripts/setup/download_assets.py --paper` or
`--model <stem>` (not hard-coded machine paths). See [docs/user/assets.md](docs/user/assets.md).
