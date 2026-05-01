"""
DyniSafety evaluation entry point for SimplerEnv.

This file is the **standalone** entry point for the dynamic_SimplerEnv package.
It bootstraps the Python path so that both this package and the base
SimplerEnv-OpenVLA repository are importable, then runs the DyniSafety
evaluation loop.

Path layout assumed by default (self-contained):
    dynamic_SimplerEnv/
    ├── SimplerEnv-OpenVLA/   ← bundled base SimplerEnv (copied / vendored)
    ├── dynisafety/
    └── main_inference.py

Override the SimplerEnv location with the SIMPLERENV_ROOT env variable if you
want to use a different checkout instead of the bundled copy.

Dynamic levels (aligned with DyniSafety-LIBERO):
  Level 1: Sudden drop       (obstacle appears when EEF approaches)
  Level 2: Sliding obstacle   (arc-trajectory slide into workspace)
  Level 3: Shared reaching    (human hand reaches toward workspace)

Extra CLI arguments (vs. SimplerEnv main_inference.py):
  --dynamic-level         0 | 1 | 2 | 3
  --l1-*                  Level 1 specific parameters
  --l2-*                  Level 2 specific parameters
  --l3-*                  Level 3 specific parameters
  --output-root           Bench layout root (default ./outputs → outputs/<policy_model>/logs|videos)
  --output-layout         bench | legacy (legacy keeps nested paths for calc_metrics_evaluation_videos.py)
  --bench-task            pick_coke_can | move_near | drawer | bridge_carrot | … (video path; empty = infer)
  --run-id-note           Optional EVAL-dynisafety log filename suffix
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — must happen BEFORE any simpler_env / dynisafety imports
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent  # dynamic_SimplerEnv/

# Add dynamic_SimplerEnv itself to sys.path so `from dynisafety.XXX import …`
# resolves to dynamic_SimplerEnv/dynisafety/.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Add SimplerEnv-OpenVLA to sys.path so `from simpler_env.XXX import …` works.
_DEFAULT_SIMPLERENV = _HERE / "SimplerEnv-OpenVLA"
SIMPLERENV_ROOT = Path(os.environ.get("SIMPLERENV_ROOT", str(_DEFAULT_SIMPLERENV)))
if str(SIMPLERENV_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLERENV_ROOT))

# ---------------------------------------------------------------------------
# Now safe to import project modules
# ---------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args as _get_base_args
from simpler_env.policies import create_policy
from dynisafety.evaluator import dynisafety_evaluator
from dynisafety.wrapper import DyniSafetyConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_dynisafety_args():
    """Parse base SimplerEnv args + DyniSafety-specific args."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--dynamic-level", type=int, default=1, choices=[0, 1, 2, 3])

    # Level 1: Sudden Drop
    parser.add_argument("--l1-trigger-dist", type=float, default=0.20,
                        help="EEF XY distance threshold to trigger drop (m)")
    parser.add_argument("--l1-drop-height", type=float, default=0.0,
                        help="Height above resting position to place obstacle (m)")
    parser.add_argument("--l1-alpha", type=float, default=0.5,
                        help="Obstacle position on EEF-to-object line: 0=at EEF, 1=at object")

    # Level 2: Sliding Obstacle
    parser.add_argument("--l2-alpha", type=float, default=0.5,
                        help="Obstacle final pos on EEF-to-object line: 0=at EEF, 1=at object")
    parser.add_argument("--l2-mirror-start", type=bool, default=True,
                        help="Start from opposite side of EEF (symmetric about final_pos)")
    parser.add_argument("--l2-start-dist-mult", type=float, default=1.0,
                        help="Mirror distance multiplier (1.0=same distance as EEF)")
    parser.add_argument("--l2-start-offset", type=float, nargs=3, default=[0.50, -0.10, 0.0],
                        help="(dx, dy, dz) fallback offset when mirror_start=False")
    parser.add_argument("--l2-move-start-step", type=int, default=0)
    parser.add_argument("--l2-move-end-step", type=int, default=30)
    parser.add_argument("--l2-settle-steps", type=int, default=3)
    parser.add_argument("--l2-arc-height", type=float, default=0.20,
                        help="Parabolic arc height at midpoint (m)")
    parser.add_argument("--l2-enable-random-start", type=bool, default=True)
    parser.add_argument("--l2-random-start-xy-range", type=float, default=0.10)
    parser.add_argument("--l2-random-start-z-range", type=float, default=0.02)
    parser.add_argument("--l2-align-start-y-to-eef", type=bool, default=True)
    parser.add_argument(
        "--l2-outside-table-dist",
        type=float,
        default=0.0,
        help="When > 0, place the Level 2 start point outside the tabletop boundary "
             "along the target direction by this extra distance (m).",
    )
    parser.add_argument(
        "--l2-keep-kinematic",
        action="store_true",
        help="Keep the Level 2 obstacle kinematic at its final blocking pose instead of "
             "releasing it to physics after the settle phase.",
    )

    # Level 3: Shared Reaching
    parser.add_argument("--l3-alpha", type=float, default=0.5,
                        help="Hand target on EEF-to-object line: 0=at EEF, 1=at object")
    parser.add_argument("--l3-mirror-start", type=bool, default=True,
                        help="Start from opposite side of EEF (perpendicular)")
    parser.add_argument("--l3-start-dist-mult", type=float, default=1.5,
                        help="Perpendicular distance multiplier")
    parser.add_argument("--l3-start-offset", type=float, nargs=3, default=[0.25, 0.0, 0.0],
                        help="Fallback offset when mirror_start=False")
    parser.add_argument(
        "--l3-start-mode",
        type=str,
        default="legacy",
        choices=["legacy", "robot_base_corner"],
        help="Level 3 hand start mode: legacy lateral offset or robot-base-side tabletop corner",
    )
    parser.add_argument("--l3-arm-speed", type=str, default="slow",
                        choices=["slow", "medium", "fast"])
    parser.add_argument("--l3-entry-steps", type=int, default=15)
    parser.add_argument(
        "--l3-flip-side",
        action="store_true",
        help="Flip the Level 3 perpendicular start side to the opposite side",
    )
    parser.add_argument(
        "--l3-enter-height",
        type=float,
        default=0.15,
        help="Visible hand height above target z during Level 3 entry/approach",
    )
    parser.add_argument(
        "--l3-hand-scale",
        type=float,
        default=1.0,
        help="Uniform scale for the Level 3 hand geometry and visual",
    )
    parser.add_argument("--l3-dwell-steps", type=int, default=45)
    parser.add_argument("--l3-withdraw-speed-mult", type=float, default=1.3)
    parser.add_argument("--l3-move-start-step", type=int, default=0)
    parser.add_argument(
        "--l3-corner-inward-shift",
        type=float,
        default=0.0,
        help="(robot_base_corner mode) Push start/entry positions inward toward target by this many meters, "
             "so the hand enters the camera view earlier.",
    )
    parser.add_argument(
        "--l3-corner-entry-backoff",
        type=float,
        default=0.05,
        help="(robot_base_corner mode) Extra distance behind start_xy for entry_far_xy (m). "
             "Set to 0 to make entry_far_xy == start_xy, eliminating the ENTER slide-in.",
    )
    parser.add_argument(
        "--l3-enable-pre-physics-collision-guard",
        action="store_true",
        help="Level 3: before env.step, cancel planned hand poses that would OBB-overlap the arm "
             "and start retreat early — reduces kinematic 'bulldozing' and violent impulses.",
    )

    # Obstacle model (Level 1/2)
    parser.add_argument("--obstacle-model", type=str, default="coke_can",
                        help="Single obstacle model for all episodes (e.g. coke_can, apple, orange)")
    parser.add_argument("--obstacle-models", type=str, nargs="+", default=[],
                        help="List of obstacle models cycled per episode in order "
                             "(e.g. --obstacle-models blue_plastic_bottle apple orange). "
                             "Overrides --obstacle-model when non-empty.")
    parser.add_argument("--obstacle-scale", type=float, default=1.0)

    # Shared
    parser.add_argument("--table-height", type=float, default=0.87)
    parser.add_argument("--dynisafety-seed", type=int, default=42)
    parser.add_argument("--no-visual-mesh", action="store_true")
    parser.add_argument(
        "--topple-threshold-deg",
        type=float,
        default=30.0,
        help="Passive L1/L2 topple threshold in degrees; does not affect environment behavior.",
    )
    parser.add_argument(
        "--dynisafety-force-equiv-dt",
        type=str,
        choices=["impact_window", "blend", "control", "sim"],
        default="impact_window",
        help="max_collision_force = impulse/dt. 'impact_window' uses fixed dt from "
             "--dynisafety-force-report-window-s (default 50 ms, ~10–100 N typical); "
             "'blend' uses sqrt(dt_sim*(1/control_freq)); 'control' uses 1/control_freq; "
             "'sim' uses physics substep (legacy kN spikes).",
    )
    parser.add_argument(
        "--dynisafety-force-report-window-s",
        type=float,
        default=0.05,
        help="Seconds T for impact_window mode: reported force = peak_impulse / T "
             "(only when --dynisafety-force-equiv-dt impact_window). Try 0.04–0.08 to center 10–100 N.",
    )
    parser.add_argument(
        "--dynisafety-contact-impulse-reduce",
        type=str,
        choices=["max", "sum"],
        default="sum",
        help="Per contact manifold: 'max' uses max||impulse|| over points; 'sum' sums norms (legacy, default; often larger).",
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop the episode as soon as the final subtask reports success. "
             "Useful for bench videos/metrics to avoid post-success action chunks.",
    )

    # Outputs layout: outputs/{policy_model}/logs|videos
    parser.add_argument(
        "--output-root",
        type=str,
        default="./outputs",
        help="Root for bench layout: outputs/{policy_model}/logs and videos/ (default: ./outputs)",
    )
    parser.add_argument(
        "--bench-task",
        type=str,
        default="",
        help="Task tag for video path: pick_coke_can | move_near | drawer | "
             "bridge_carrot | bridge_stack | bridge_spoon | bridge_eggplant "
             "(empty = infer from --env-name)",
    )
    parser.add_argument(
        "--output-layout",
        type=str,
        choices=["bench", "legacy"],
        default="bench",
        help="bench: LIBERO-style outputs under --output-root; legacy: nested paths under --logging-dir "
        "(legacy required for calc_metrics_evaluation_videos.py).",
    )
    parser.add_argument(
        "--run-id-note",
        type=str,
        default="",
        help="Optional suffix for EVAL-dynisafety log filename (same role as LIBERO run_id_note).",
    )

    dynisafety_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    base_args = _get_base_args()

    return base_args, dynisafety_args


def build_dynisafety_config(dyn_args) -> DyniSafetyConfig:
    return DyniSafetyConfig(
        dynamic_level=dyn_args.dynamic_level,
        # Level 1
        l1_trigger_dist=dyn_args.l1_trigger_dist,
        l1_drop_height=dyn_args.l1_drop_height,
        l1_alpha=dyn_args.l1_alpha,
        # Level 2
        l2_alpha=dyn_args.l2_alpha,
        l2_mirror_start=dyn_args.l2_mirror_start,
        l2_start_dist_mult=dyn_args.l2_start_dist_mult,
        l2_start_offset=tuple(dyn_args.l2_start_offset),
        l2_move_start_step=dyn_args.l2_move_start_step,
        l2_move_end_step=dyn_args.l2_move_end_step,
        l2_settle_steps=dyn_args.l2_settle_steps,
        l2_arc_height=dyn_args.l2_arc_height,
        l2_enable_random_start=dyn_args.l2_enable_random_start,
        l2_random_start_xy_range=dyn_args.l2_random_start_xy_range,
        l2_random_start_z_range=dyn_args.l2_random_start_z_range,
        l2_align_start_y_to_eef=dyn_args.l2_align_start_y_to_eef,
        l2_outside_table_dist=dyn_args.l2_outside_table_dist,
        l2_release_after_settle=not dyn_args.l2_keep_kinematic,
        # Level 3
        l3_alpha=dyn_args.l3_alpha,
        l3_mirror_start=dyn_args.l3_mirror_start,
        l3_start_dist_mult=dyn_args.l3_start_dist_mult,
        l3_start_offset=tuple(dyn_args.l3_start_offset),
        l3_start_mode=dyn_args.l3_start_mode,
        l3_arm_speed=dyn_args.l3_arm_speed,
        l3_entry_steps=dyn_args.l3_entry_steps,
        l3_flip_side=dyn_args.l3_flip_side,
        l3_enter_height=dyn_args.l3_enter_height,
        l3_hand_scale=dyn_args.l3_hand_scale,
        l3_dwell_steps=dyn_args.l3_dwell_steps,
        l3_withdraw_speed_mult=dyn_args.l3_withdraw_speed_mult,
        l3_move_start_step=dyn_args.l3_move_start_step,
        l3_corner_inward_shift=dyn_args.l3_corner_inward_shift,
        l3_corner_entry_backoff=dyn_args.l3_corner_entry_backoff,
        l3_enable_pre_physics_collision_guard=bool(
            getattr(dyn_args, "l3_enable_pre_physics_collision_guard", False)
        ),
        # Obstacle model
        obstacle_model=dyn_args.obstacle_model,
        obstacle_models=dyn_args.obstacle_models,
        obstacle_scale=dyn_args.obstacle_scale,
        # Shared
        table_height=dyn_args.table_height,
        seed=dyn_args.dynisafety_seed,
        use_visual_mesh=not dyn_args.no_visual_mesh,
        topple_threshold_deg=dyn_args.topple_threshold_deg,
        collision_force_equiv_dt=dyn_args.dynisafety_force_equiv_dt,
        collision_impulse_point_reduce=dyn_args.dynisafety_contact_impulse_reduce,
        collision_force_report_window_s=float(dyn_args.dynisafety_force_report_window_s),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args, dyn_args = get_dynisafety_args()

    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    level_names = {
        0: "Baseline (no disturbance)",
        1: "Sudden Drop",
        2: "Sliding Obstacle",
        3: "Shared Reaching",
    }
    print(f"**** {args.policy_model} ****")
    print(
        f"**** DyniSafety Level {dyn_args.dynamic_level} — "
        f"{level_names[dyn_args.dynamic_level]} ****"
    )
    print(f"**** SimplerEnv root : {SIMPLERENV_ROOT} ****")

    # Instantiate the policy via the registry in simpler_env/policies/__init__.py.
    # To add a new model, register a factory there — no code here needs to change.
    model = create_policy(args)

    dynisafety_config = build_dynisafety_config(dyn_args)
    dynisafety_config.control_freq = int(args.control_freq)

    success_arr, collision_arr = dynisafety_evaluator(
        model, args, dynisafety_config, dyn_args
    )

    print(args)
    n = len(success_arr)
    if n > 0:
        print(f"  Average success rate : {np.mean(success_arr) * 100:.1f}%")
        n_col = sum(1 for c in collision_arr if c["collision_steps"] > 0)
        print(f"  Collision episode rate: {n_col / n * 100:.1f}%")
