"""Robot geometry through the pipeline (T07; D15 a, b): probe trace -> TaskContext -> builders -> G0, and the chain.

* Trace (schema 2): ``robot_bodies`` are the robot's own bodies, ``robot_body_quats`` their orientations,
  ``held_object_bodies_pos`` the held object, ``robot_geometry`` once per trace.
* TaskContext: body paths survive a held object entering and leaving the collision-body list (H1: 28 of 54 CALVIN
  contexts lost their body paths that way); bodies missing from some step are dropped by name; geometry must agree.
* Builders: ``hazardlib.instantiate``, the suite's C13 builder and ``FakeHost.nominal_hit_instance`` give the same
  standoff on the same input and record ``standoff_source``; without geometry the outputs are bit-identical to the
  integration branch (``SCALAR_DIGEST``).
* G0: with geometry a hazard that meets a wrist before ``t_hit`` is rejected; the held object counts as points.
* ``record -> probe -> instantiate -> gate -> calibrate -> freeze -> verify`` on a FakeHost with geometry.
"""
import hashlib
import json

import numpy as np
import pytest
import yaml

import pipeline_world as W
from pipeline_world import conformance_dir, conformance_reports  # noqa: F401 (fixtures)
from hazardlib.conformance import FakeHost, FakeSphere
from hazardlib.conformance.harness import RobotAtHit, build_nominal_hit_instance
from hazardlib.gates import run_g0
from hazardlib.host import CapabilityError
from hazardlib.instantiate import REJECT_REASONS, InstanceRejected, instantiate
from hazardlib.kinematics import SampledPath
from hazardlib.kinematics.standoff import BodyPosePaths, RobotGeometry
from hazardlib.probe import PROBE_TRACE_SCHEMA_VERSION, ProbeTrace, build_task_context, read_probe_trace, \
    write_probe_trace
from hazardlib.spec import AgentSpec, TaskContext

import test_pipeline_cli as CLI
import test_pipeline_probe as P

SPACING = 0.005
WRIST = FakeSphere("robot0_wrist", (0.0, 0.062, 0.12), 0.04)   # off the fall axis: its origin is 62 mm away
MUG = FakeSphere("mug_body", (0.0, 0.0, -0.08), 0.02)


def geo_host(**world):
    return W.make_host(robot_point_spacing_m=SPACING, **world)


def wrist_bodies():
    return W.world_config().robot_bodies + (WRIST,)


def tc_of(host, task):
    return build_task_context(W.record_traces(host, task), W.probe_settings(), static_entities=W.STATIC_ENTITIES,
                              declared_workspace=W.WORKSPACE)


_TC = {}


def geo_tc(task, wrist=False):
    key = (task, wrist)
    if key not in _TC:
        _TC[key] = tc_of(geo_host(**({"robot_bodies": wrist_bodies()} if wrist else {})), task)
    return _TC[key]


def without_geometry(tc):
    data = tc.to_dict()
    return TaskContext.from_dict(dict(data, provenance=dict(data["provenance"], robot_geometry=None,
                                                            robot_body_quat_paths=None)))


class HeldBodiesHost(FakeHost):
    """Reports the mug's collision body (and ``held_object``) only while the EEF is past the grasp point, the way
    an adapter puts the grasped object into its collision-body list (H1, contract feedback 4)."""

    def robot_state(self, env):
        state = super().robot_state(env)
        if float(state.eef_pose.pos[0]) >= W.GRASP_X - 1e-3:
            return state
        keep = [i for i, n in enumerate(state.collision_body_names) if n != MUG.name]
        quat = None if state.collision_body_quat is None else state.collision_body_quat[keep]
        return type(state)(eef_pose=state.eef_pose, eef_lin_vel_mps=state.eef_lin_vel_mps,
                           collision_body_pos=state.collision_body_pos[keep], gripper_opening_m=None,
                           held_object=None, collision_body_names=[state.collision_body_names[i] for i in keep],
                           collision_body_lin_vel_mps=state.collision_body_lin_vel_mps[keep],
                           held_object_bodies=None, collision_body_quat=quat)


def held_host(spacing=SPACING):
    return HeldBodiesHost(W.world_config(held_object="mug", held_bodies=(MUG,), robot_point_spacing_m=spacing))


# ------------------------------------------------------------------------------------------------- trace

def test_trace_records_robot_bodies_orientations_and_geometry():
    traces = W.record_traces(geo_host(), "pick")
    header = traces[0].header
    assert header.schema_version == PROBE_TRACE_SCHEMA_VERSION == "2"
    assert header.robot_geometry.spacing_m == SPACING
    assert sorted(header.robot_geometry.points) == ["robot0_hand", "robot0_link"]   # the camera is not robot_link
    for step in traces[0].steps:
        assert sorted(step.robot_bodies) == sorted(step.robot_body_quats) == ["robot0_hand", "robot0_link"]
        assert step.robot_body_quats["robot0_hand"] == [1.0, 0.0, 0.0, 0.0]
        # GripperHost holds the cube (no collision body) after the grasp
        assert step.held_object_bodies_pos == ({} if step.held_object == W.HELD else None)
    plain = W.record_traces(W.make_host(), "pick")[0]
    assert plain.header.robot_geometry is None and all(s.robot_body_quats is None for s in plain.steps)


def test_trace_schema_1_is_refused(tmp_path):
    trace = W.record_traces(W.make_host(), "low")[0]
    path = tmp_path / "t.jsonl"
    write_probe_trace(path, trace)
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header["probe_trace"]["schema_version"] = "1"
    path.write_text("\n".join([json.dumps(header)] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="unsupported schema_version '1'"):
        read_probe_trace(path)


def test_held_object_bodies_are_never_robot_bodies():
    traces = W.record_traces(held_host(), "pick")
    for step in traces[0].steps:
        assert MUG.name not in step.robot_bodies
        if step.held_object is None:
            assert step.held_object_bodies_pos is None
        else:
            assert list(step.held_object_bodies_pos) == [MUG.name]
    assert MUG.name not in traces[0].header.robot_geometry.points


def test_a_geometry_host_without_body_orientations_is_refused():
    class NoQuat(FakeHost):
        def robot_state(self, env):
            state = super().robot_state(env)
            state.collision_body_quat = None
            return state

    with pytest.raises(CapabilityError, match="collision_body_quat"):
        W.record_traces(NoQuat(W.world_config(robot_point_spacing_m=SPACING)), "low")


# ------------------------------------------------------------------------------------------- TaskContext

def test_body_paths_survive_a_held_object_entering_and_leaving():
    """H1's degeneration: the held object's body joins the collision-body list at the grasp. The robot body
    paths are kept, the held object gets its own path from the grasp on."""
    tc = tc_of(held_host(), "pick")
    prov = tc.provenance
    assert sorted(prov["robot_body_paths"]) == sorted(prov["robot_body_quat_paths"]) == ["robot0_hand", "robot0_link"]
    assert prov["robot_body_paths_dropped"] == []
    assert sorted(prov["robot_geometry"]["points"]) == ["robot0_hand", "robot0_link"]
    held = prov["held_object_path"]
    assert held["time_s"] and all(list(b) == [MUG.name] for b in held["bodies"])
    nominal = SampledPath.from_pose_sequence(tc.nominal_ee_path)
    assert min(held["time_s"]) >= 0.6 - 0.05  # held only around and after the grasp (x >= 0.299 at t ~ 0.6 s)
    t = held["time_s"][-1]
    np.testing.assert_allclose(held["bodies"][-1][MUG.name], nominal.pos(t) + np.array(MUG.center_m), atol=1e-9)
    assert prov["occupancy_content"] == "nominal_eef_and_robot_bodies"
    assert len(tc.occupancy.points) == 3 * len(tc.nominal_ee_path.time_s)   # EEF + 2 robot bodies, no mug


def _with_bodies(trace, extra_until_s=None, extra="robot0_wrist"):
    data = trace.to_dict()
    for step in data["steps"]:
        if extra_until_s is not None and step["t_s"] < extra_until_s:
            step["robot_bodies"][extra] = [0.0, 0.0, 1.0]
    return ProbeTrace.from_dict(data)


def test_bodies_missing_from_some_step_are_dropped_by_name():
    runs = P.three()
    runs[1] = _with_bodies(runs[1], extra_until_s=0.5)
    tc = build_task_context(runs, P.settings())
    assert sorted(tc.provenance["robot_body_paths"]) == ["robot0_hand"]
    assert tc.provenance["robot_body_paths_dropped"] == ["robot0_wrist"]
    assert tc.provenance["robot_geometry"] is None and tc.provenance["robot_body_quat_paths"] is None


def _with_geometry(trace, spacing=SPACING, quats=True, points=None):
    data = trace.to_dict()
    data["header"]["robot_geometry"] = {"spacing_m": spacing,
                                        "points": points or {"robot0_hand": [[0.0, 0.0, 0.03], [0.03, 0.0, 0.0]]}}
    for step in data["steps"]:
        step["robot_body_quats"] = {"robot0_hand": [1.0, 0.0, 0.0, 0.0]} if quats else None
    return ProbeTrace.from_dict(data)


def test_robot_geometry_must_agree_across_runs():
    runs = [_with_geometry(r) for r in P.three()]
    tc = build_task_context(runs, P.settings())
    assert tc.provenance["robot_geometry"]["spacing_m"] == SPACING
    assert sorted(tc.provenance["robot_body_quat_paths"]) == ["robot0_hand"]
    for broken in ([runs[0], runs[1], _with_geometry(P.three()[2], spacing=0.004)],
                   [runs[0], runs[1], P.three()[2]],
                   [runs[0], runs[1], _with_geometry(P.three()[2], points={"robot0_hand": [[0.0, 0.0, 0.02]]})]):
        with pytest.raises(ValueError, match="different robot geometry"):
            build_task_context(broken, P.settings())


def test_robot_geometry_needs_orientations_and_samples_of_every_kept_body():
    with pytest.raises(ValueError, match="robot_body_quats"):
        build_task_context([_with_geometry(r, quats=False) for r in P.three()], P.settings())
    with pytest.raises(ValueError, match="no samples for the robot bodies"):
        build_task_context([_with_geometry(r, points={"robot0_link": [[0.0, 0.0, 0.0]]}) for r in P.three()],
                           P.settings())


# ------------------------------------------------------------------------------------------------ builders

SCALAR_DIGEST = "7822e54637116ed9a886936495b99155afd93d90bf1f8798b2fae8fceffc0e45"
_KEYS = ("standoff_m", "contact_normal", "contact_point", "hazard_support_radius_m", "anchor_point",
         "hit_velocity_mps", "hit_diagnostics", "mapping", "keyframes", "support_surface", "eef_contact_radius_m",
         "intensity", "tau_s", "direction_source", "keyframe_release_velocity_mps", "release_window_s")


def _exact(obj):
    if isinstance(obj, dict):
        return {k: _exact(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_exact(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj).hex()
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def test_without_geometry_builder_and_g0_are_bit_identical_to_the_integration_branch():
    """``SCALAR_DIGEST`` was computed with the code of ``hazard/integration @ 6449277`` (hz38 and hz311 agree) on
    these eight specs: every keyframe, timing, release and standoff number and every G0 check, as exact float hex."""
    specs = [(W.fall_spec(), "pick"), (W.fall_spec(tau=0.3, frac=0.75), "pick"), (W.slide_spec(), "low"),
             (W.slide_spec(direction=(0.6, 0.8, 0.0)), "low"), (W.reach_spec(), "pick"),
             (W.reach_spec(tau=0.25, speed=0.2, direction=(1.0, 0.0, 0.0), phase="carry", frac=1.0 / 30.0,
                           profile="linear"), "pick"),
             (W.slide_spec(tau=0.1, speed=0.3, direction=(1.0, 0.0, 0.0)), "low"),
             (W.fall_spec(tau=0.5, phase="carry", frac=0.5), "pick")]
    bd = W.make_host().describe()
    out = []
    for spec, task in specs:
        inst = W.build(spec, task)
        assert inst.provenance["standoff_source"] == "eef_contact_radius"
        assert inst.provenance["standoff_body"] is None and inst.provenance["robot_point_spacing_m"] is None
        d = inst.to_dict()
        out.append({"spec": spec.id, "task": task,
                    "instance": _exact({k: d[k] for k in ("keyframes", "nominal_hit_point", "nominal_hit_time_s",
                                                         "onset_time_s", "release_time_s", "release_velocity")}),
                    "provenance": _exact({k: d["provenance"][k] for k in _KEYS if k in d["provenance"]}),
                    "g0": _exact(run_g0(inst, W.task_context(task), bd, W.gate_settings().g0).to_dict())})
    assert hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest() == SCALAR_DIGEST


def _robot_at_hit(tc):
    prov = tc.provenance
    return RobotAtHit(RobotGeometry.from_dict(prov["robot_geometry"]),
                      BodyPosePaths(tc.nominal_ee_path.time_s, prov["robot_body_paths"], prov["robot_body_quat_paths"]))


AGENT = AgentSpec(kind="object", shape="sphere", half_extents_m=[W.HAZARD_R], mass_kg=0.2)


@pytest.mark.parametrize("family,make_spec,params", [
    ("intrusion.fall", lambda: W.fall_spec(), lambda inst: {"g": W.GRAVITY}),
    ("intrusion.reach", lambda: W.reach_spec(direction=(0.6, 0.8, 0.0)), W.hit_params),
], ids=["fall", "reach_oblique"])
def test_the_three_builders_give_the_same_standoff(family, make_spec, params):
    tc = geo_tc("pick")
    host = geo_host()
    inst = W.build(make_spec(), "pick", host=host, tc=tc)
    prov = inst.provenance
    assert prov["standoff_source"] == "robot_geometry" and prov["standoff_body"] == "robot0_hand"
    assert prov["robot_point_spacing_m"] == SPACING
    t_hit, tau = inst.nominal_hit_time_s, prov["tau_s"]
    nominal = SampledPath.from_pose_sequence(tc.nominal_ee_path)
    kw = dict(family=family, agent=AGENT, t_hit_s=t_hit, tau_s=tau, params=params(inst), keyframe_dt_s=0.01)
    suite = build_nominal_hit_instance("c13", nominal, eef_contact_radius_m=W.HAND_R, robot=_robot_at_hit(tc), **kw)
    fake = host.nominal_hit_instance("pick", "fake", "fake", **kw)
    assert suite.provenance["standoff_m"] == prov["standoff_m"]           # same inputs, same function
    assert fake.provenance["standoff_m"] == pytest.approx(prov["standoff_m"], abs=1e-9)   # its own nominal path
    for other in (suite, fake):
        assert other.provenance["standoff_source"] == "robot_geometry"
        assert other.provenance["standoff_body"] == "robot0_hand"
        assert other.provenance["robot_point_spacing_m"] == SPACING
    print("%s standoff: builder %.9f m, C13 builder %.9f m, FakeHost %.9f m (scalar %.3f m)"
          % (family, prov["standoff_m"], suite.provenance["standoff_m"], fake.provenance["standoff_m"], W.STANDOFF))
    assert 0.0 <= W.STANDOFF - prov["standoff_m"] <= 1e-3   # sphere on sphere: within the sampling error


def test_scalar_builders_record_their_source():
    host = W.make_host()
    inst = W.build(W.fall_spec(), "pick", host=host)
    nominal = W.task_context("pick").nominal_ee_path
    kw = dict(family="intrusion.fall", agent=AGENT, t_hit_s=inst.nominal_hit_time_s, tau_s=0.2,
              params={"g": W.GRAVITY}, keyframe_dt_s=0.01)
    suite = build_nominal_hit_instance("c13", SampledPath.from_pose_sequence(nominal), eef_contact_radius_m=W.HAND_R,
                                       **kw)
    fake = host.nominal_hit_instance("pick", "fake", "fake", **kw)
    for i in (inst, suite, fake):
        assert i.provenance["standoff_source"] == "eef_contact_radius" and i.provenance["standoff_body"] is None
        assert i.provenance["standoff_m"] == W.STANDOFF


def test_geometry_finds_the_wrist_above_the_hand():
    tc = geo_tc("pick", wrist=True)
    inst = W.build(W.fall_spec(), "pick", host=geo_host(robot_bodies=wrist_bodies()), tc=tc)
    assert inst.provenance["standoff_body"] == "robot0_wrist"
    # wrist 0.12 m up, 0.062 m off the axis, r 0.04, hazard r 0.03: 0.12 + sqrt(0.07^2 - 0.062^2)
    assert inst.provenance["standoff_m"] == pytest.approx(0.12 + np.sqrt(0.07 ** 2 - 0.062 ** 2), abs=1e-3)


def test_no_contact_along_the_normal_is_a_rejection():
    assert "no_robot_contact_along_normal" in REJECT_REASONS and "slide_normal_unsolved" in REJECT_REASONS
    tc = geo_tc("pick")
    with pytest.raises(InstanceRejected) as info:
        W.build(W.fall_spec(offset=(0.2, 0.0, 0.0)), "pick", host=geo_host(), tc=tc)   # 0.2 m beside the hand
    assert info.value.reason == "no_robot_contact_along_normal"


def test_the_builder_needs_the_bd_the_probe_ran_with():
    with pytest.raises(ValueError, match="robot_point_spacing_m"):
        instantiate(W.fall_spec(), geo_tc("pick"), 0, W.make_host().describe(), W.instantiate_settings())
    with pytest.raises(ValueError, match="robot_point_spacing_m"):
        instantiate(W.fall_spec(), geo_tc("pick"), 0, W.make_host(robot_point_spacing_m=0.004).describe(),
                    W.instantiate_settings())


def test_geometric_slide_tilts_onto_the_robot_or_rejects():
    tc = geo_tc("low")
    inst = W.build(W.slide_spec(), "low", host=geo_host(), tc=tc)
    prov = inst.provenance
    assert prov["standoff_source"] == "robot_geometry"
    normal = np.array(prov["contact_normal"])
    centroid = np.array(prov["anchor_point"]) + prov["standoff_m"] * normal
    assert centroid[2] == pytest.approx(prov["support_surface"]["top_m"] + W.HAZARD_R, abs=1e-12)
    # the tilted geometric standoff stays within the sampling error of the scalar one
    scalar = W.build(W.slide_spec(), "low")
    assert prov["standoff_m"] == pytest.approx(scalar.provenance["standoff_m"], abs=1e-3)


# ---------------------------------------------------------------------------------------------------- G0

def test_g0_with_geometry_rejects_a_hazard_that_meets_the_wrist_first():
    """The scalar instance aims 0.06 m above the hand; falling, it passes the wrist (62 mm off its axis, r 0.04):
    the body-origin check (threshold 0.06 - 0.001 m) misses it, the geometry does not."""
    tc_geo = geo_tc("pick", wrist=True)
    tc_scalar = without_geometry(tc_geo)
    host = geo_host(robot_bodies=wrist_bodies())
    scalar_inst = instantiate(W.fall_spec(), tc_scalar, 0, W.make_host(robot_bodies=wrist_bodies()).describe(),
                              W.instantiate_settings())
    settings = W.gate_settings().g0
    old = run_g0(scalar_inst, tc_scalar, host.describe(), settings)
    assert old.passed and old.checks["early_contact_check"] == "eef_and_bodies"
    new = run_g0(scalar_inst, tc_geo, host.describe(), settings)
    assert new.reasons == ["early_contact"] and new.checks["early_contact_check"] == "robot_geometry"
    early = new.checks["early_contact"]
    assert early["body"] == "robot0_wrist" and early["max_penetration_m"] > settings.early_contact_tol_m
    assert early["time_s"] < scalar_inst.nominal_hit_time_s
    geo_inst = W.build(W.fall_spec(), "pick", host=host, tc=tc_geo)
    ok = run_g0(geo_inst, tc_geo, host.describe(), settings)
    assert ok.passed, ok.reasons
    assert ok.checks["early_contact"]["max_penetration_m"] <= settings.early_contact_tol_m


def test_g0_counts_the_held_object_as_points():
    """A reach from below aimed at the hand passes the held mug (8 cm under the hand) first."""
    host = held_host()
    tc = tc_of(host, "pick")
    spec = W.reach_spec(direction=(0.0, 0.0, 1.0), phase="approach", frac=0.99)
    inst = instantiate(spec, tc, 0, host.describe(), W.instantiate_settings())
    assert inst.provenance["standoff_body"] == "robot0_hand"
    result = run_g0(inst, tc, host.describe(), W.gate_settings().g0)
    assert "early_contact" in result.reasons
    assert result.checks["early_contact"]["body"] == MUG.name and result.checks["early_contact"]["held_object_points"]


def test_g0_without_geometry_keeps_its_checks():
    result = run_g0(W.build(W.fall_spec(), "pick"), W.task_context("pick"), W.make_host().describe(),
                    W.gate_settings().g0)
    assert result.checks["early_contact_check"] == "eef_and_bodies" and "min_distance_m" in result.checks["early_contact"]


# ------------------------------------------------------------------------------------------- end to end

def test_end_to_end_chain_with_robot_geometry(tmp_path, conformance_reports, monkeypatch):
    report = conformance_reports(decel_mps2=W.DECEL_MPS2, robot_point_spacing_m=SPACING)
    c13 = [c for c in report["checks"] if c["id"] == "C13"][0]
    assert report["verdict"] == "pass" and c13["measured"]["standoff_source"] == "robot_geometry"
    # the T06 chain of test_pipeline_cli on the same braking host, with robot geometry switched on
    monkeypatch.setattr(CLI, "HOST", CLI.HOST + ["--host-arg", "robot_point_spacing_m=%r" % SPACING])
    probe, inst, gate, cal, frz, ver = CLI.chain(tmp_path / "geo", report)
    work = tmp_path / "geo"
    tc = yaml.safe_load((work / "tc_pick.yaml").read_text())
    assert tc["provenance"]["robot_geometry"]["spacing_m"] == SPACING
    assert yaml.safe_load((work / "bd.yaml").read_text())["robot_point_spacing_m"] == SPACING
    sources = set()
    for line in (work / "instances.jsonl").read_text().splitlines():
        record = json.loads(line)
        if "instance" in record:
            sources.add(record["instance"]["provenance"]["standoff_source"])
    assert sources == {"robot_geometry"} and inst["instances"] >= 1
    assert gate["labels"] and frz["entries"] >= 1 and ver["ok"] is True
    print("geometry chain: instances %s rejected %s gate %s freeze %s entries" % (
        inst["instances"], inst["rejected_by_reason"], gate["labels"], frz["entries"]))
