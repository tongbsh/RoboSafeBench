"""The built-in conformance host ``fake``: :class:`~hazardlib.conformance.fakehost.FakeHost` in the world of
``configs/hazard/fakehost.yaml`` (T05).

``make_fake_target`` builds it with optional mutations (the injected defects of ``fakehost.MUTATIONS``), an
honest ``ccd=False``, world overrides (e.g. another gravity) or a FakeHost subclass (test-local defects in
``hazardlib/tests/test_mutants.py``). The registry entry ``fake`` is the clean host.

FakeHost has no engine faults, so ``inject_engine_fault`` plays the host's fault detector: it reports the fault
through the installed hooks (``report_instability``) at the position of a scene body, or of the given hazard. On
FakeHost, C11 therefore checks the mapping path (hook forwarding, attribution, denominator), not detection.
"""
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from ..config import ConfigError, load_config
from ..spec.hazard_spec import AgentSpec
from .fakehost import FakeAction, FakeHost, FakeSphere, FakeTask, FakeTaskObject, FakeWorldConfig
from .harness import PROBE_SPEC_ID
from .target import ConformanceTarget, EpisodeRef, Scenario, StrikeTarget, register_host

FAKEHOST_CONFIG = "fakehost"
INJECTED_FAULT_KIND = "fakehost_injected_fault"


def _sphere(data: Mapping[str, Any]) -> FakeSphere:
    return FakeSphere(data["name"], tuple(data["center_m"]), data["radius_m"])


def fake_world_config(data: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None) -> FakeWorldConfig:
    """``FakeWorldConfig`` from the ``world`` block of fakehost.yaml; ``overrides`` replaces top-level fields."""
    world = dict(data)
    tasks = {}
    for task_id, task in world.pop("tasks").items():
        layouts = {layout_id: tuple(FakeTaskObject(_sphere(o), o["on_hit_rotation_wxyz"]) for o in objects)
                   for layout_id, objects in task["layouts"].items()}
        tasks[task_id] = FakeTask(tuple(task["path_times_s"]), tuple(tuple(p) for p in task["path_pos_m"]),
                                  tuple(task["goal_pos_m"]), task["success_radius_m"], layouts)
    values = dict(world, tasks=tasks)
    for key in ("robot_bodies", "held_bodies"):
        values[key] = tuple(_sphere(b) for b in world[key])
    for key in ("table_min_xy_m", "table_max_xy_m", "image_min_xy_m", "image_max_xy_m", "park_pos_m",
                "robot_include", "robot_exclude"):
        values[key] = tuple(world[key])
    values.update(dict(overrides or {}))
    return FakeWorldConfig(**values)


def _strike(data: Optional[Mapping[str, Any]]) -> Optional[StrikeTarget]:
    return None if data is None else StrikeTarget(data["body"], tuple(data["direction"]), data["radius_m"])


def fake_scenario(data: Mapping[str, Any]) -> Scenario:
    ref = lambda d: EpisodeRef(d["task_id"], d["layout_id"], d["seed"])  # noqa: E731
    return Scenario(still=ref(data["still"]), motion=ref(data["motion"]),
                    probe_agent=AgentSpec.from_dict(dict(data["probe_agent"])),
                    free_point_m=tuple(data["free_point_m"]), free_direction=tuple(data["free_direction"]),
                    strike=_strike(data["strike"]), held_strike=_strike(data.get("held_strike")),
                    drop_point_m=tuple(data["drop_point_m"]), drop_surface_height_m=data["drop_surface_height_m"],
                    motion_hit_time_s=data["motion_hit_time_s"],
                    motion_hit_direction=tuple(data["motion_hit_direction"]),
                    motion_hold_time_s=data["motion_hold_time_s"], motion_duration_s=data["motion_duration_s"])


def load_fakehost_config(config_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    data = load_config(FAKEHOST_CONFIG, strict=True, config_dir=config_dir)
    for key in ("adapter_version", "world", "scenario"):
        if not isinstance(data, Mapping) or key not in data:
            raise ConfigError("fakehost.yaml needs the top-level key %r" % key)
    return dict(data)


def make_fake_target(mutations: Sequence[str] = (), ccd: bool = True,
                     world_overrides: Optional[Mapping[str, Any]] = None, host_cls: Optional[type] = None,
                     host_kwargs: Optional[Mapping[str, Any]] = None, name: str = "fake",
                     config_dir: Optional[Union[str, Path]] = None) -> ConformanceTarget:
    data = load_fakehost_config(config_dir)
    world = fake_world_config(data["world"], world_overrides)
    cls = FakeHost if host_cls is None else host_cls
    if not (isinstance(cls, type) and issubclass(cls, FakeHost)):
        raise TypeError("host_cls must be FakeHost or a subclass, got %r" % (cls,))
    host = cls(world, mutations=tuple(mutations), ccd=ccd, **dict(host_kwargs or {}))
    scenario = fake_scenario(data["scenario"])
    motion_task = scenario.motion.task_id

    def build_hit_instance(**kwargs: Any) -> Any:
        return host.nominal_hit_instance(motion_task, kwargs["instance_id"], PROBE_SPEC_ID, kwargs["family"],
                                         kwargs["agent"], kwargs["t_hit_s"], kwargs["tau_s"], kwargs["params"],
                                         kwargs["keyframe_dt_s"])

    def inject_engine_fault(env: int, hazard: Any) -> None:
        hooks = host._runtime  # what install_hooks received (the runtime, or a recording tap around it)
        if hooks is None:
            raise RuntimeError("no hooks installed on the FakeHost")
        if hazard is None:
            objects = host.object_poses(env)
            pos = objects[sorted(objects)[0]].pos if objects else host.robot_state(env).eef_pose.pos
            hooks.report_instability(env, INJECTED_FAULT_KIND, entity_is_hazard=False, entity_pos=pos)
        else:
            hooks.report_instability(env, INJECTED_FAULT_KIND, entity_is_hazard=True,
                                     entity_pos=host.hazard_state(hazard).pos)

    tag = "+".join(sorted(mutations)) if mutations else "clean"
    return ConformanceTarget(name=name, host=host, adapter_version="%s/%s" % (data["adapter_version"], tag),
                             scenario=scenario, motion_action=lambda t: FakeAction.follow(),
                             nominal_eef_path=host.nominal_eef_path(motion_task),
                             build_hit_instance=build_hit_instance, inject_engine_fault=inject_engine_fault)


register_host("fake", make_fake_target)
