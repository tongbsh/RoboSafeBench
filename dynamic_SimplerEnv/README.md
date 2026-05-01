# DyniSafety-SimplerEnv

**DyniSafety-SimplerEnv** is the [SimplerEnv](https://github.com/simpler-env/SimplerEnv) variant of the **DyniSafety** benchmark — a framework for evaluating the safety-awareness of Vision-Language-Action (VLA) models under dynamic physical disturbances.

The benchmark can run a **Level 0 baseline** (standard SimplerEnv, no injected disturbance) or inject three real-world-inspired perturbation levels into the SAPIEN / ManiSkill2 simulator, measuring task success and collision-avoidance behaviour where applicable.

---

## Dynamic Disturbance Levels

| Level | Name | Description |
|-------|------|-------------|
| 0 | **Baseline** | No dynamic disturbance — same as original SimplerEnv / ManiSkill2 evaluation (`--dynamic-level 0`) |
| 1 | **Sudden Drop** | An obstacle suddenly appears on the EEF-to-target line when the end-effector approaches |
| 2 | **Sliding Obstacle** | An obstacle slides along a parabolic arc trajectory into the robot's workspace |
| 3 | **Shared Reaching** | A simulated human hand reaches toward the workspace, competing for the same target region |

`--level all` in `run_dynisafety.sh` runs levels **1, 2, and 3** only; use `--level 0` explicitly for the baseline.

---

## Repository Layout

```
dynamic_SimplerEnv/
├── main_inference.py               # Entry point — runs DyniSafety evaluation loop
├── dynisafety/                     # Core benchmark module
│   ├── wrapper.py                  # DyniSafetyWrapper + DyniSafetyConfig
│   ├── evaluator.py                # Episode loop with dynamic disturbance injection
│   ├── controllers.py              # Level 1/2/3 disturbance controllers
│   ├── output_layout.py            # Logging / video path layout
│   └── assets/                     # Level 3 hand mesh assets
├── scripts/
│   ├── run_dynisafety.sh           # Main orchestration script (model-agnostic)
│   ├── eval_dynisafety.sh          # VLA registry — auto-selects checkpoint by model name
│   ├── pick_coke_can_dynisafety.sh # Task wrapper: GraspSingleOpenedCokeCanInScene-v0
│   ├── move_near_dynisafety.sh     # Task wrapper: MoveNearGoogleInScene-v0
│   ├── drawer_dynisafety.sh        # Task wrapper: Open/CloseTop/Middle/BottomDrawer-v0
│   ├── bridge_widowx_dynisafety.sh # WidowX Bridge 4 tasks (bridge.sh parity, --policy-setup widowx_bridge)
│   ├── dynamicSimplerEnv_eval/     # Fine-grained per-task × per-level scripts
│   │   ├── drawer_level1_sudden_drop.sh
│   │   ├── drawer_level2_sliding_obstacle.sh
│   │   ├── drawer_level3_shared_reaching.sh
│   │   ├── run_level{1,2,3}_*.sh   # pick_coke_can variants
│   │   └── move_near_level{1,2,3}_*.sh
│   ├── univla/                     # UniVLA server + eval scripts
│   ├── memoryvla/                  # MemoryVLA server + eval scripts
│   ├── openvla-oft/                # OpenVLA-OFT eval script (in-process)
│   ├── gr00t/                      # GR00T eval script (in-process, no server needed)
│   ├── openpi/                     # π₀ / π₀.₅ server + eval scripts
│   ├── spatialVLA/                 # SpatialVLA server + eval scripts
│   ├── x-vla/                      # X-VLA server + eval scripts
│   └── starvla/                    # StarVLA server + eval scripts
└── SimplerEnv-OpenVLA/             # Bundled SimplerEnv base (submodule)
    └── simpler_env/
        └── policies/               # Policy adapters for each VLA model
```

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/your-org/DyniSafety-SimplerEnv.git dynamic_SimplerEnv
cd dynamic_SimplerEnv
```

### 2. Set up the base SimplerEnv environment

Follow the installation guide in `SimplerEnv-OpenVLA/README.md`. The recommended conda environment is `openvla-oft`.

```bash
conda create -n openvla-oft python=3.10
conda activate openvla-oft
pip install -e SimplerEnv-OpenVLA/
pip install -e SimplerEnv-OpenVLA/ManiSkill2_real2sim/
```

### 3. Model-specific environments

Each VLA model that runs as a separate server requires its own conda environment. Refer to the individual `scripts/<model>/start_*.sh` header comments for the required setup.

---

## Quick Start

### Option A — Per-task / per-level scripts (recommended for getting started)

These scripts directly call `main_inference.py` with all parameters preset. The `drawer_level1_sudden_drop.sh` script has been validated end-to-end:

```bash
cd dynamic_SimplerEnv

# Default: OpenVLA, Level 1 Sudden Drop, Drawer task
bash scripts/dynamicSimplerEnv_eval/drawer_level1_sudden_drop.sh

# Custom model and checkpoint
bash scripts/dynamicSimplerEnv_eval/drawer_level1_sudden_drop.sh \
    --model cogact --ckpt CogACT/CogACT-Large
```

Available scripts: `drawer_level{1,2,3}_*.sh`, `run_level{1,2,3}_sudden_drop.sh`, `move_near_level{1,2,3}_*.sh`.

### Option B — Generic orchestration script

```bash
cd dynamic_SimplerEnv

# OpenVLA, all three levels, pick_coke_can task
bash scripts/run_dynisafety.sh \
    --model openvla \
    --ckpt openvla/openvla-7b \
    --task pick_coke_can \
    --level all

# OpenVLA-OFT, Level 2, drawer task
bash scripts/run_dynisafety.sh \
    --model openvla_oft \
    --ckpt moojink/openvla-7b-oft-finetuned-libero-goal \
    --task drawer \
    --level 2

# Level 0 baseline (no DyniSafety perturbation)
bash scripts/run_dynisafety.sh \
    --model openvla \
    --ckpt openvla/openvla-7b \
    --task pick_coke_can \
    --level 0

# WidowX Bridge (same envs as SimplerEnv-OpenVLA/scripts/bridge.sh): use a Bridge-trained checkpoint
# and widowx_bridge policy setup (the task script passes --policy-setup widowx_bridge).
# RGB overlay paths resolve under SIMPLERENV_ROOT (default: bundled SimplerEnv-OpenVLA).
bash scripts/run_dynisafety.sh \
    --model openvla_oft \
    --ckpt moojink/openvla-7b-oft-finetuned-bridge \
    --task bridge \
    --level 1

# Single Bridge subtask: bridge_carrot | bridge_stack | bridge_spoon | bridge_eggplant
bash scripts/run_dynisafety.sh \
    --model openvla_oft \
    --ckpt moojink/openvla-7b-oft-finetuned-bridge \
    --task bridge_spoon \
    --level 0
```

### WidowX Bridge tasks

| `--task` | ManiSkill2 environments |
|----------|-------------------------|
| `bridge` | All four Bridge eval envs in `bridge.sh` order |
| `bridge_carrot` | `PutCarrotOnPlateInScene-v0` on `bridge_table_1_v1`, robot `widowx` |
| `bridge_stack` | `StackGreenCubeOnYellowCubeBakedTexInScene-v0`, same table |
| `bridge_spoon` | `PutSpoonOnTableClothInScene-v0`, same table |
| `bridge_eggplant` | `PutEggplantInBasketScene-v0` on `bridge_table_1_v2`, robot `widowx_sink_camera_setup` |

Filter subtasks when invoking [`scripts/bridge_widowx_dynisafety.sh`](scripts/bridge_widowx_dynisafety.sh) directly: `BRIDGE_SUBTASK=carrot|stack|spoon|eggplant|all`.

### Option C — VLA registry wrapper

```bash
# Evaluates the model using preset checkpoint defaults
bash scripts/eval_dynisafety.sh --vla openvla_oft --task move_near --level 1
```

---

## Supported VLA Models

### In-process models (no server required)

| Model key | Script | Notes |
|-----------|--------|-------|
| `openvla` | `run_dynisafety.sh --model openvla` | OpenVLA-7B |
| `openvla_oft` | `run_dynisafety.sh --model openvla_oft` | OpenVLA-OFT fine-tuned |
| `cogact` | `run_dynisafety.sh --model cogact` | CogACT |
| `gr00t` | `scripts/gr00t/run_dynisafety_eval.sh` | GR00T (in-process via `--ckpt-path`) |
| `rt1` | `run_dynisafety.sh --model rt1` | RT-1 |
| `octo-base` / `octo-small` | `run_dynisafety.sh --model octo-base` | Octo |

### Server-based models (start server first, then run client)

| Model key | Start server | Run eval |
|-----------|-------------|----------|
| `univla` | `bash scripts/univla/start_univla_server.sh` | `bash scripts/univla/run_dynisafety_eval.sh` |
| `memoryvla` | `bash scripts/memoryvla/start_memoryvla_server.sh --model-path <ckpt>` | `bash scripts/memoryvla/run_dynisafety_eval.sh` |
| `pi0` / `pi05` | `bash scripts/openpi/start_pi0_server.sh --checkpoint-dir <dir>` | `bash scripts/openpi/run_dynisafety_eval.sh` |
| `spatialvla_server` | `bash scripts/spatialVLA/start_spatialvla_server.sh` | `bash scripts/spatialVLA/run_dynisafety_eval.sh` |
| `xvla` | `bash scripts/x-vla/start_xvla_server.sh --model-path <ckpt>` | `bash scripts/x-vla/run_dynisafety_eval.sh` |
| `starvla` | `bash scripts/starvla/start_starvla_server.sh --ckpt-path <file>` | `bash scripts/starvla/run_dynisafety_eval.sh` |

**Two-terminal workflow for server-based models:**

```bash
# Terminal 1 — start the server
bash scripts/memoryvla/start_memoryvla_server.sh \
    --model-path shihao1895/memvla-bridge \
    --unnorm-key bridge_orig

# Terminal 2 — run evaluation once server is ready
MEMORYVLA_SERVER_HOST=127.0.0.1 MEMORYVLA_SERVER_PORT=8111 \
bash scripts/run_dynisafety.sh \
    --model memoryvla --ckpt ignored \
    --task pick_coke_can --level 1
```

### X-VLA official dual-root setup

`xvla_official` can use a separate SimplerEnv checkout without changing the default bundled benchmark root used by other models.

- Default bundled path: `dynamic_SimplerEnv/SimplerEnv-OpenVLA`
- Official override path: pass `--official-simplerenv-root /path/to/SimplerEnv-OpenVLA` or export `SIMPLERENV_ROOT_OFFICIAL=/path/to/SimplerEnv-OpenVLA`
- The override only applies when `scripts/x-vla/run_bridge_xvla_eval.sh` is called with `--official`
- Bundled root -> `XVLA_EXEC_MODE=simpler_delta`
- Alternate root -> `XVLA_EXEC_MODE=upstream_literal`

Example:

```bash
# Keep all other models on the bundled delta-EE benchmark root.
# Only xvla_official switches to a separate X-VLA-compatible SimplerEnv checkout.
bash dynamic_SimplerEnv/scripts/x-vla/run_bridge_xvla_eval.sh \
    --official \
    --official-simplerenv-root /path/to/SimplerEnv-XVLA-Official \
    --task bridge_carrot \
    --level 0
```

The alternate root should be compatible with the current DyniSafety entrypoints and asset layout, including `simpler_env`, `ManiSkill2_real2sim`, and the Bridge overlay paths used by `scripts/bridge_widowx_dynisafety.sh`.

---

## Outputs

By default (`--output-layout bench`), results are written to:

```
dynamic_SimplerEnv/
└── outputs/
    └── <policy_model>/
        ├── logs/
        │   └── EVAL-dynisafety-<task>-level<N>-<env>-<timestamp>.txt
        └── videos/
            └── <task>_level<N>/
                └── <env>/
                    └── ...mp4
```

Use `--output-layout legacy` together with `--log-root results` to produce the nested directory structure expected by `SimplerEnv-OpenVLA/tools/calc_metrics_evaluation_videos.py`.

---

## Registering a New VLA Model

1. Add a policy factory in `SimplerEnv-OpenVLA/simpler_env/policies/__init__.py` using `register_policy()`.
2. Create `SimplerEnv-OpenVLA/simpler_env/policies/<your_model>/` with a model class following the existing pattern (e.g. `openvla/openvla_model.py`).
3. Run with `--model <your_model>` — no other changes needed.

A template is provided at `SimplerEnv-OpenVLA/simpler_env/policies/your_model_here/`.

---

## Citation

If you use DyniSafety-SimplerEnv in your research, please cite:

```bibtex
@article{dynisafety2025,
  title   = {DyniSafety: Benchmarking VLA Safety under Dynamic Physical Disturbances},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {2025},
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

The bundled `SimplerEnv-OpenVLA/` submodule retains its original license (Apache 2.0). `ManiSkill2_real2sim/` retains its original license.
