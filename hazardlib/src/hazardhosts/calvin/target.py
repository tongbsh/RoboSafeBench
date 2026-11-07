"""The CALVIN conformance target (H1): ``register_host("calvin", make_calvin_target)``.

``hazardhosts/calvin/__init__.py`` stays empty (the hazardhosts packages import nothing at package level), so
``python -m hazardlib.conformance --host calvin`` needs ``--host-module hazardhosts.calvin.target``.

* host: :class:`~hazardhosts.calvin.host.CalvinHost` (dataset directory and EGL device from the environment).
* scenario: ``configs/hazard/calvin.yaml: scenario``; every number there names the probe that measured it.
* motion: the recorded play data replayed from the motion layout's frame as relative actions
  (``policies.relative_actions``); the action of the control step that starts at episode time ``t`` is the one of
  frame ``round(t / control_dt)``. No nominal EEF path is declared: C13 records it by replay.
* ``build_hit_instance``: not provided. CALVIN has no instance builder of its own; the pipeline uses the generic
  ``hazardlib.instantiate``, whose D1 conversion (``p_ref + standoff * n``) is the suite builder's.
* ``inject_engine_fault``: ``CalvinHost.inject_engine_fault`` (a real speed blow-up the host's detector catches).
"""
from typing import Any, Mapping, Optional

import numpy as np

from hazardlib.config import ConfigError
from hazardlib.conformance.target import ConformanceTarget, EpisodeRef, Scenario, StrikeTarget, register_host
from hazardlib.spec.hazard_spec import AgentSpec

from .host import CalvinHost
from .policies import _scales, relative_actions


def _strike(data: Optional[Mapping[str, Any]]) -> Optional[StrikeTarget]:
    return None if data is None else StrikeTarget(data["body"], tuple(data["direction"]), data["radius_m"])


def calvin_scenario(data: Mapping[str, Any]) -> Scenario:
    ref = lambda d: EpisodeRef(d["task_id"], d["layout_id"], d["seed"])  # noqa: E731
    return Scenario(still=ref(data["still"]), motion=ref(data["motion"]),
                    probe_agent=AgentSpec.from_dict(dict(data["probe_agent"])),
                    free_point_m=tuple(data["free_point_m"]), free_direction=tuple(data["free_direction"]),
                    strike=_strike(data["strike"]), held_strike=_strike(data.get("held_strike")),
                    drop_point_m=tuple(data["drop_point_m"]), drop_surface_height_m=data["drop_surface_height_m"],
                    motion_hit_time_s=data["motion_hit_time_s"],
                    motion_hit_direction=tuple(data["motion_hit_direction"]),
                    motion_hold_time_s=data["motion_hold_time_s"], motion_duration_s=data["motion_duration_s"])


def make_calvin_target(dataset_dir: Optional[str] = None, egl_device: Optional[int] = None,
                       config_dir: Optional[str] = None) -> ConformanceTarget:
    host = CalvinHost(dataset_dir=dataset_dir, egl_device=egl_device, config_dir=config_dir)
    if "scenario" not in host.config:
        raise ConfigError("calvin.yaml needs the top-level key 'scenario'")
    scenario = calvin_scenario(host.config["scenario"])
    layout = host.dataset.layout(scenario.motion.layout_id)
    n_steps = int(round(scenario.motion_duration_s / host.control_dt)) + 1
    actions = relative_actions(host.dataset, layout.split, layout.frame, n_steps, _scales(host.env))

    def motion_action(t: float) -> np.ndarray:
        k = int(round(t / host.control_dt))
        if not 0 <= k < len(actions):
            raise ValueError("motion_action(%r): outside the replayed %d control steps" % (t, len(actions)))
        return actions[k]

    return ConformanceTarget(name="calvin", host=host, adapter_version=host.config["adapter_version"],
                             scenario=scenario, motion_action=motion_action,
                             inject_engine_fault=host.inject_engine_fault)


register_host("calvin", make_calvin_target)
