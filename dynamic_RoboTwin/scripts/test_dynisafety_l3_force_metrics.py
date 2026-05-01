#!/usr/bin/env python3
"""Unit checks for RoboTwin DyniSafety L3 collision force metrics (SimplerEnv-aligned).

Uses only ``numpy`` plus two small modules under ``envs/dynisafety/`` (no SAPIEN).

Run from repository root::

    python dynamic_RoboTwin/scripts/test_dynisafety_l3_force_metrics.py

For one full Level-3 eval episode (requires RoboTwin + SAPIEN + policy), see the
docstring at the bottom of this file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]  # dynamic_safty_bench/
_DYN_ROOT = _REPO_ROOT / "dynamic_RoboTwin" / "RoboTwin" / "envs" / "dynisafety"


def _load(name: str, rel: str):
    path = _DYN_ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_cfg = _load("rt_dynisafety_config", "config.py")
_cm = _load("rt_dynisafety_contact_metrics", "contact_metrics.py")

RoboTwinDyniSafetyConfig = _cfg.RoboTwinDyniSafetyConfig
force_equiv_timestep = _cm.force_equiv_timestep
contact_point_impulse_scalar = _cm.contact_point_impulse_scalar


def main() -> None:
    raw = {
        "collision_impulse_point_reduce": "SUM",
        "collision_force_equiv_dt": "SIM",
        "collision_force_report_window_s": 0.02,
        "collision_force_control_freq_hz": 10.0,
    }
    c = RoboTwinDyniSafetyConfig.from_mapping(raw)
    assert c.collision_impulse_point_reduce == "sum"
    assert c.collision_force_equiv_dt == "sim"
    assert abs(c.collision_force_report_window_s - 0.02) < 1e-9
    assert abs(c.collision_force_control_freq_hz - 10.0) < 1e-9

    dt_sim = 0.004
    assert (
        abs(
            force_equiv_timestep(
                dt_sim,
                collision_force_equiv_dt="impact_window",
                collision_force_report_window_s=0.05,
                collision_force_control_freq_hz=20.0,
            )
            - 0.05
        )
        < 1e-9
    )
    assert (
        abs(
            force_equiv_timestep(
                dt_sim,
                collision_force_equiv_dt="sim",
                collision_force_report_window_s=0.05,
                collision_force_control_freq_hz=20.0,
            )
            - 0.004
        )
        < 1e-9
    )
    assert (
        abs(
            force_equiv_timestep(
                dt_sim,
                collision_force_equiv_dt="control",
                collision_force_report_window_s=0.05,
                collision_force_control_freq_hz=20.0,
            )
            - 0.05
        )
        < 1e-9
    )
    expected_blend = float(np.sqrt(0.004 / 25.0))
    assert (
        abs(
            force_equiv_timestep(
                dt_sim,
                collision_force_equiv_dt="blend",
                collision_force_report_window_s=0.05,
                collision_force_control_freq_hz=25.0,
            )
            - expected_blend
        )
        < 1e-9
    )

    pts = [
        SimpleNamespace(impulse=np.array([3.0, 0.0, 0.0])),
        SimpleNamespace(impulse=np.array([0.0, 4.0, 0.0])),
    ]
    assert abs(contact_point_impulse_scalar(pts, "max") - 4.0) < 1e-9
    assert abs(contact_point_impulse_scalar(pts, "sum") - 7.0) < 1e-9
    assert contact_point_impulse_scalar([], "max") == 0.0

    print("test_dynisafety_l3_force_metrics: OK")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Full sim: single L3 episode (run yourself when RoboTwin env is ready)
# ---------------------------------------------------------------------------
#   cd dynamic_RoboTwin/RoboTwin
#   export PYTHONPATH=.
#   python ../main_inference.py \
#       --policy_name internvla_a1 \
#       --policy_kwargs '{}' \
#       --dynamic_level 3 \
#       --test_num 1 \
#       --task_idx 0 \
#       --task_config demo_clean \
#       --collision_force_equiv_dt impact_window \
#       --collision_impulse_point_reduce max
#
# Legacy RoboTwin force scale (substep dt + summed point impulses):
#       --collision_force_equiv_dt sim \
#       --collision_impulse_point_reduce sum
