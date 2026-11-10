"""Conformance target for the original LIBERO adapter (configuration-driven)."""
import numpy as np

from hazardlib.conformance.target import ConformanceTarget, EpisodeRef, Scenario, StrikeTarget, register_host
from hazardlib.spec.hazard_spec import AgentSpec

from .host import LiberoHost
from .demos import parse_layout


def make_libero_target(config_dir=None):
    host = LiberoHost(config_dir=config_dir)
    cfg = host.config["scenario"]
    def ref(data):
        return EpisodeRef(data["task_id"], data["layout_id"], data["seed"])
    scenario = Scenario(still=ref(cfg["still"]), motion=ref(cfg["motion"]),
                        probe_agent=AgentSpec.from_dict(cfg["probe_agent"]),
                        free_point_m=tuple(cfg["free_point_m"]), free_direction=tuple(cfg["free_direction"]),
                        strike=StrikeTarget(**cfg["strike"]), held_strike=None,
                        drop_point_m=tuple(cfg["drop_point_m"]),
                        drop_surface_height_m=cfg["drop_surface_height_m"],
                        motion_hit_time_s=cfg["motion_hit_time_s"],
                        motion_hit_direction=tuple(cfg["motion_hit_direction"]),
                        motion_hold_time_s=cfg["motion_hold_time_s"], motion_duration_s=cfg["motion_duration_s"])
    host.reset_episode(0, scenario.motion.task_id, scenario.motion.layout_id, scenario.motion.seed)
    layout = parse_layout(scenario.motion.layout_id)
    actions = host._demo(scenario.motion.task_id, layout.demo_set).actions(layout.episode)[layout.start:]
    def motion_action(t):
        index = int(round(t / host.control_dt))
        return actions[index] if index < len(actions) else host.hold_action(0)
    return ConformanceTarget(name="libero", host=host, adapter_version=host.config["adapter_version"],
                             scenario=scenario, motion_action=motion_action,
                             inject_engine_fault=host.inject_engine_fault)


register_host("libero", make_libero_target)
