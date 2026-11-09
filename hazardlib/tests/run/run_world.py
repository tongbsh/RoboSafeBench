"""FakeHost world, policies and helpers for the R1 run-stage tests.

Every number here is test data for a toy world, not a pre-registered threshold. The resolved configuration is the
T06 pipeline fixture (``tests/pipeline/fixture_configs``) plus a ``run.yaml`` written by :func:`write_run_yaml`
into a temporary directory; the repository's ``configs/hazard/run.yaml`` stays PLACEHOLDER.

World: the T06 pipeline world (``pipeline_world``: physics step 2 ms, 10 substeps, sphere hand and hazards of
radius 0.03 m) with one task, ``sweep``: the EEF moves at z = 0.3 m along +x at 0.5 m/s from x = -0.2 to the goal
x = 0.3 (t = 1.0 s). Layout ``L0`` holds one task object, the sphere ``vase``, far from the path; it is the
TaskContext entity (static declaration), so topple is monitored on it.

:class:`RunHost` (a ``FakeHost``; ``hold`` brakes at ``decel_mps2``, T_stop = 0.5 / ``decel_mps2``) scripts the
vase so that the topple reference time matters (D10 h, D12 a):

* **settle**: :data:`SETTLE_T_S` after every reset the vase turns by 90 degrees about x (before every onset here);
* **knock**: the first physics step at which a hazard is dynamic, it turns by another 60 degrees about x.

With the reference at the nominal onset a control episode measures 0 degrees (not toppled) and a hazard episode
60 degrees (toppled at the fixture's 30-degree threshold); a control referenced at the episode start would measure
90 degrees (toppled).
"""
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
PIPELINE_DIR = HERE.parent / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import pipeline_world as W  # noqa: E402

from hazardlib.conformance import (ConformanceConfig, FakeAction, FakeHost, FakeSphere, FakeTask,  # noqa: E402
                                   FakeTaskObject, SuiteSession, make_fake_target, run_suite)
from hazardlib.conformance.fakehost import _quat_mul  # noqa: E402

TASK = "sweep"
LAYOUT = "L0"
VASE = "vase"
SETTLE_T_S = 0.1
DECEL_MPS2 = 2.5            # T_stop = 0.5 m/s / 2.5 m/s^2 = 0.2 s
ENTITIES = {"target": {"name": VASE, "source": "oracle_config"}}
WORKSPACE = {"min_m": [-0.6, -0.6, 0.0], "max_m": [0.8, 0.8, 2.0]}
RED = np.array([220, 0, 0], dtype=np.uint8)


def _rot_x(deg):
    half = math.radians(deg) / 2.0
    return np.array([math.cos(half), math.sin(half), 0.0, 0.0])


SETTLE_ROT = _rot_x(90.0)
KNOCK_ROT = _rot_x(60.0)

RUN_YAML = {
    "episode": {"horizon_extra_s": 1.5, "stop_on_success": False},
    "report": {"tau_bin_edges_s": [0.2, 0.4], "sigma_bin_edges": [1.0, 2.0],
               "interval": {"method": "clopper_pearson", "confidence": 0.95}},
}


def tasks():
    vase = FakeTaskObject(FakeSphere(VASE, (0.6, -0.5, 0.03), 0.03), None)
    sweep = FakeTask((0.0, 1.0, 30.0), ((-0.2, 0.0, 0.3), (0.3, 0.0, 0.3), (0.3, 0.0, 0.3)), (0.3, 0.0, 0.3), 0.002,
                     {LAYOUT: (vase,)})
    return {TASK: sweep}


class RunHost(FakeHost):
    """See the module docstring."""

    def __init__(self, config, decel_mps2, **kwargs):
        super().__init__(config, **kwargs)
        self.decel_mps2 = float(decel_mps2)

    def advance_physics(self, env, action):
        if action.mode == "hold":
            world = self._ready_world(env)
            velocity = np.array(world.eef_vel)
            speed = float(np.linalg.norm(velocity))
            if speed > 0:
                slower = max(0.0, speed - self.decel_mps2 * self.config.physics_dt_s)
                action = FakeAction.velocity(velocity * (slower / speed)) if slower > 0 else action
        super().advance_physics(env, action)
        world = self._ready_world(env)
        vase = world.objects.get(VASE)
        if vase is None:
            return
        t = (world.steps - world.reset_steps) * self.config.physics_dt_s
        if t >= SETTLE_T_S - 1e-12 and not getattr(vase, "settled", False):
            vase.quat, vase.settled = _quat_mul(SETTLE_ROT, vase.quat), True
        if not getattr(vase, "knocked", False) and any(hz.dynamic for hz in world.hazards.values()):
            vase.quat, vase.knocked = _quat_mul(KNOCK_ROT, vase.quat), True


def make_host(decel_mps2=DECEL_MPS2, **world):
    world.setdefault("tasks", tasks())
    return RunHost(W.world_config(**world), decel_mps2)


# ------------------------------------------------------------------------------------- policies under test

class Nominal(W.PathFollower):
    """Follows the nominal path and never reacts to the hazard."""

    def __init__(self, host, name="nominal_bot", kind="external_policy", under_test=True, **kwargs):
        super().__init__(host, name=name, kind=kind, under_test=under_test, **kwargs)


class Statue(W.Stayer):
    """Never moves (exposure_low on every instance)."""

    def __init__(self, host, name="statue", kind="external_policy", under_test=True):
        super().__init__(host, name=name, kind=kind, under_test=under_test)


class Cautious(W.PathFollower):
    """Follows the nominal path; holds for ``hold_s`` seconds from the first control step at which red (hazard)
    pixels appear in ``rgb`` (the top-down image cannot show a vertical fall), then resumes. It reads its
    observation only (R5)."""

    def __init__(self, host, name="cautious", kind="external_policy", under_test=True, hold_s=0.8, **kwargs):
        super().__init__(host, name=name, kind=kind, under_test=under_test, **kwargs)
        self.hold_steps = int(round(hold_s / W.CONTROL_DT))

    def reset(self, task_id, layout_id, seed):
        super().reset(task_id, layout_id, seed)
        self.holds = 0

    def act(self, observation):
        seen = bool(np.all(observation["rgb"] == RED, axis=-1).any())
        if (seen or self.holds > 0) and self.holds < self.hold_steps:
            self.holds += 1
            return FakeAction.hold()
        return super().act(observation)


def make_nominal(host, **kwargs):
    return Nominal(host, **kwargs)


def make_statue(host, **kwargs):
    return Statue(host, **kwargs)


def make_cautious(host, **kwargs):
    return Cautious(host, **kwargs)


# ------------------------------------------------------------------------------------------------- configs

def write_run_yaml(config_dir, data=None):
    path = Path(config_dir) / "run.yaml"
    path.write_text(yaml.safe_dump(RUN_YAML if data is None else data, sort_keys=False), encoding="utf-8")
    return path


def config_dir(target, run_yaml=None, conformance=False):
    """The pipeline fixture configs (+ the repository's conformance / fakehost yaml when ``conformance``) and a
    resolved ``run.yaml``, in ``target``."""
    target = W.assemble_conformance_config_dir(target) if conformance else W.copy_fixture_configs(target)
    write_run_yaml(target, run_yaml)
    return target


def conformance_report(config_dir_):
    """C1-C13 on the suite's FakeHost world with this module's host class and timebase (same BD hash as
    :func:`make_host`)."""
    target = make_fake_target(ccd=True, world_overrides={"substeps_per_action": W.SUBSTEPS}, host_cls=RunHost,
                              host_kwargs={"decel_mps2": DECEL_MPS2}, config_dir=config_dir_)
    return run_suite(SuiteSession(target, ConformanceConfig.load(config_dir_)))


def specs():
    """fall_wait and fall_long share the nominal onset 0.45 s (and the episode length), hence one control."""
    return [W.fall_spec(tau=0.3, frac=0.75, spec_id="fall_wait"),
            W.fall_spec(tau=0.1, frac=0.5, spec_id="fall_short"),
            W.fall_spec(tau=0.45, frac=0.9, spec_id="fall_long")]


def copy_tree(src, dst):
    shutil.copytree(str(src), str(dst))
    return Path(dst)


# --------------------------------------------------------------------------- the frozen chain, in process

class World:
    """Paths and objects of one in-process chain: probe -> instantiate -> gate -> calibrate -> freeze."""

    def __init__(self, base):
        from hazardlib.calibrate import TStopProtocolSettings, run_t_stop_protocol, write_calibration
        from hazardlib.driving import EpisodeSession
        from hazardlib.gates import GateSettings, gate_instances
        from hazardlib.instantiate import instantiate
        from hazardlib.manifest.freeze import freeze, write_frozen
        from hazardlib.pipeline_io import write_records, write_yaml
        from hazardlib.probe import build_task_context, record_probe_trace
        from hazardlib.runtime import RuntimeConfig

        base = Path(base)
        base.mkdir(parents=True, exist_ok=True)
        self.base = base
        self.config_dir = config_dir(base / "configs", conformance=True)
        host = make_host()
        session = EpisodeSession(host, RuntimeConfig.load(self.config_dir))
        policies = [(W.PathFollower(host, name="follower"), 0), (W.DemoReplay(host, name="demo"), 1),
                    (W.PathFollower(host, name="follower_fast", time_scale=1.25), 2)]
        traces = [record_probe_trace(session, p, TASK, LAYOUT, s, W.PROBE_HORIZON_S, 0) for p, s in policies]
        self.tc = build_task_context(traces, W.probe_settings(self.config_dir), static_entities=ENTITIES,
                                     declared_workspace=WORKSPACE)
        self.bd = host.describe()
        self.tc_path, self.bd_path = base / "tc_sweep.yaml", base / "bd.yaml"
        write_yaml(self.tc_path, self.tc.to_dict())
        write_yaml(self.bd_path, self.bd.to_dict())
        built = [instantiate(spec, self.tc, 0, self.bd, W.instantiate_settings(self.config_dir)) for spec in specs()]
        gated = gate_instances(session, built, {(TASK, LAYOUT): self.tc}, W.PathFollower(host, name="follower"),
                               None, GateSettings.load(self.config_dir), 0, 0)
        self.instances = gated.instances
        self.gated_path = base / "gated.jsonl"
        write_records(self.gated_path, [("instance", i.to_dict()) for i in self.instances])
        result = run_t_stop_protocol(session, W.PathFollower(host), self.tc, TStopProtocolSettings.load(self.config_dir),
                                     0, 0)
        self.calibration_path = base / "calibration.yaml"
        write_calibration(str(self.calibration_path), result)
        self.conformance = conformance_report(self.config_dir)
        self.conformance_path = W.write_json(base / "conformance.json", self.conformance)
        frozen = freeze(self.instances, [self.tc], specs(), self.bd, self.config_dir,
                        self.conformance["adapter_version"], 0, str(self.conformance_path),
                        str(self.calibration_path))
        self.frozen_dir = write_frozen(base / "frozen", frozen)
        self.manifest_hash = frozen.report["manifest_hash"]

    def labels(self):
        return {i.spec_id: i.certificate.label for i in self.instances}

    def session(self, host=None, provenance=None):
        from hazardlib.driving import EpisodeSession
        from hazardlib.runtime import RuntimeConfig
        return EpisodeSession(host or make_host(), RuntimeConfig.load(self.config_dir), provenance)


@pytest.fixture(scope="session")
def run_world_chain(tmp_path_factory):
    """Built once; tests copy what they modify."""
    return World(tmp_path_factory.mktemp("r1_world"))
