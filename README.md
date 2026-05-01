# DyniSafety Benchmark

DyniSafety is a dynamic physical safety benchmark for Vision-Language-Action
(VLA) policies. It evaluates whether a policy can maintain task performance
while responding safely to moving obstacles and human-like disturbances.

This repository contains three simulator-backed benchmark variants:

- `dynisafety_libero/`: DynamicLIBERO, built on LIBERO / MuJoCo.
- `dynamic_SimplerEnv/`: DynamicSimplerEnv, built on SimplerEnv / ManiSkill2 /
  SAPIEN.
- `dynamic_RoboTwin/`: DynamicRoboTwin, built on RoboTwin / SAPIEN.

The benchmark code is intended for research use. It reports simulator-internal
safety indices and should not be used as deployment-grade physical safety
certification.

## Disturbance Levels

- `L0 Baseline`: standard task execution with no injected dynamic disturbance.
- `L1 Sudden Drop`: an obstacle suddenly enters the robot workspace.
- `L2 Sliding Obstacle`: an obstacle moves through the workspace along a planned
  trajectory.
- `L3 Shared Reaching`: a simulated human hand reaches toward the workspace,
  creating robot-hand contact risk.

The exact geometry, trigger rules, and force reporting paths are simulator
specific. See `docs/core_metrics_design.md` and the per-backend README files for
details.

## Repository Layout

```text
.
├── docs/                    # Metric definitions and design notes
├── dynisafety_libero/        # DynamicLIBERO benchmark implementation
├── dynamic_SimplerEnv/       # DynamicSimplerEnv benchmark implementation
├── dynamic_RoboTwin/         # DynamicRoboTwin benchmark implementation
├── download_hf.py            # Helper for downloading external HF artifacts
└── README.md
```

The working directory may also contain local-only folders such as
`download_checkpoints/`, `vlaModelHome/`, `past_files_cache/`, `simulator_env/`,
and benchmark `outputs/`. These are excluded by `.gitignore` because they are
large, machine-specific, or generated artifacts. If a release needs curated
result tables, place them in a small dedicated directory such as `results/`
rather than committing raw logs and videos.

## Metrics

DyniSafety reports task and safety metrics at the episode level and then
aggregates them by task, model, and disturbance level.

- `SR` measures standard task success rate.
- `CR` measures the fraction of episodes with contact involving the dynamic
  disturbance.
- `Topple` measures object topple events for obstacle-based levels.
- `FPeakMean(N)` or `PeakForce` summarizes L3 contact severity using
  episode-peak contact force values.
- `SASR` is a safety-adjusted success metric where unsafe successes are
  penalized.

L3 force values are backend-specific. DynamicLIBERO reads MuJoCo normal contact
forces via `mj_contactForce`. DynamicSimplerEnv and DynamicRoboTwin compute
equivalent force from SAPIEN contact impulses under the benchmark calibration.
Do not compare absolute force values across simulators without explicitly
accounting for these implementation differences.

## Quick Start

Each backend has its own simulator and model dependencies. Start with the README
inside the backend you want to run.

### DynamicSimplerEnv

```bash
cd dynamic_SimplerEnv

# Example: run L3 on a Bridge WidowX subtask.
bash scripts/run_dynisafety.sh \
    --model openvla_oft \
    --ckpt <checkpoint-or-hf-repo> \
    --task bridge_carrot \
    --level 3
```

More model-specific launchers are under `dynamic_SimplerEnv/scripts/`.

### DynamicRoboTwin

```bash
cd dynamic_RoboTwin

# Example: run one task and one disturbance level.
bash scripts/run_dynisafety.sh \
    --model internvla_a1 \
    --level 3 \
    --task place_a2b_right
```

See `dynamic_RoboTwin/README.md` for the RoboTwin environment setup and
model-adapter registry.

### DynamicLIBERO

```bash
cd dynisafety_libero

# Example model wrapper. See scripts/<model>/ for full sweeps and server-based
# models.
bash scripts/eval_dynisafety.sh object 0 3 \
    --vla openvla \
    --trials 50
```

LIBERO scripts for OpenVLA-OFT, UniVLA, X-VLA, StarVLA, OpenPI, GR00T, and
SpatialVLA are organized under `dynisafety_libero/scripts/`.

## Aggregating Results

After evaluations finish, use the backend aggregation scripts to produce compact
CSV summaries.

```bash
python dynamic_SimplerEnv/aggregate_logs.py
python dynamic_RoboTwin/scripts/aggregate_logs.py
python dynisafety_libero/scripts/aggregate_logs.py
```

Raw videos, logs, and full output folders are ignored by default. Keep only
small, curated result artifacts in the public repository.

## Checkpoints and External Assets

Model checkpoints are not committed to this repository. Download checkpoints
from their original providers or Hugging Face repositories, then pass the local
path or model ID to the relevant script.

Bundled simulator or policy subtrees retain their upstream licenses. Before a
public release, verify each third-party component that remains in the repository
has compatible license and attribution files.

## Citation

If you use DyniSafety in your research, please cite:

```bibtex
@article{dynisafety2025,
  title   = {DyniSafety: Benchmarking VLA Safety under Dynamic Physical Disturbances},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {2025}
}
```

## Release Checklist

- Add the final paper citation and BibTeX entry.
- Add a root `LICENSE` file and confirm third-party license compatibility.
- Verify that no checkpoints, private paths, API keys, logs, or videos are
  tracked.
- Document any required external downloads that are not committed.
