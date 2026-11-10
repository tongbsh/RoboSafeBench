"""Original LIBERO HazardHost and EpisodeDriver.

Uses four canonical original suites, 10 tasks each. A configured hazard pool is
appended to the original MJCF in memory; no LIBERO imports or assets.
Physics hooks wrap each MjSim.step. Contacts use scratch MjData.
"""
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from hazardlib.config import load_bd, load_config
from hazardlib.host import (CapabilityError, ContactObs, HazardHandle, HazardState, PartnerClass, Pose,
                            RobotState, Surface, TimeState)
from hazardlib.spec.hazard_spec import AgentSpec

from .demos import DemoFile, map_plain_state, parse_layout
from .geometry import RigidGroups, group_points, is_collision_geom, mat_to_quat, quat_mul, quat_to_mat
from .paths import datasets_root, import_libero

HOST_CONFIG = "libero"
INSTABILITY_NONFINITE = "libero_nonfinite_state"
INSTABILITY_SPEED = "libero_speed_limit"
INSTABILITY_WARNING = "libero_mujoco_warning"


def _vec3(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError("%s must be 3 finite numbers, got %r" % (name, value))
    return arr


def _unit_quat(value: Any) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or float(np.linalg.norm(q)) == 0.0:
        raise ValueError("quat must be 4 finite numbers (wxyz), not all zero, got %r" % (value,))
    return q / np.linalg.norm(q)


def quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """``dynisafety.backends.libero.env.quat2axisangle`` (robosuite's xyzw convention), reproduced so that the
    observation matches without importing the old backend."""
    q = np.asarray(quat_xyzw, dtype=np.float64).copy()
    q[3] = min(1.0, max(-1.0, q[3]))
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float64)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


class _Member(object):
    """One pool member (a LIBERO obstacle)."""

    def __init__(self, name: str, index: int) -> None:
        self.name = name
        self.index = index
        self.body = -1
        self.qadr = -1
        self.dadr = -1
        self.geoms = []  # type: List[int]
        self.contype = []  # type: List[int]
        self.conaffinity = []  # type: List[int]
        self.com = np.zeros(3)
        self.mass = 0.0
        self.agent = None  # type: Optional[AgentSpec]
        self.unavailable = None  # type: Optional[str]
        self.park_pos = np.zeros(3)
        self.key = name
        self.allocated = False
        self.parked = True
        self.dynamic = False
        self.contacts = []  # type: List[ContactObs]
        self.touching = set()  # type: Set[str]
        self.pos_before = None  # type: Optional[np.ndarray]


class LiberoHost(object):
    """The LIBERO adapter. ``config_dir`` defaults to hazardlib's; ``settle_steps`` (hold steps between the layout
    state and ``begin_episode``, dynisafety's ``num_steps_wait``) and ``camera_hw`` default to ``libero.yaml``."""

    def __init__(self, config_dir: Optional[str] = None, settle_steps: Optional[int] = None,
                 camera_hw: Optional[int] = None) -> None:
        self.config = load_config(HOST_CONFIG, strict=True, config_dir=config_dir)
        self._bd = load_bd("libero", strict=True, config_dir=config_dir)
        cfg = self.config
        self._settle = int(cfg["settle_steps"] if settle_steps is None else settle_steps)
        self._hw = int(cfg["camera_hw"] if camera_hw is None else camera_hw)
        self._cameras = dict(cfg["cameras"])
        self._tasks = {}
        self._pool_cfg = dict(cfg["pool"])
        self._touch = float(cfg["contacts"]["touch_distance_m"])
        self._search = float(cfg["min_distance_search_m"])
        limits = cfg["instability"]
        self._max_lin = float(limits["max_lin_speed_mps"])
        self._max_ang = float(limits["max_ang_speed_radps"])
        self._max_joint = float(limits["max_joint_speed"])
        self._libero = import_libero()
        from libero.libero import benchmark
        for suite_name in cfg["suites"]:
            if suite_name not in benchmark.get_benchmark_dict():
                raise CapabilityError("configured LIBERO tree lacks original suite %r; SafeLIBERO is not accepted"
                                      % suite_name)
            suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
            for i in range(suite.n_tasks):
                task = suite.get_task(i)
                self._tasks[suite_name + "/" + task.name] = {
                    "suite": suite_name, "demo_set": suite_name, "name": task.name}
        import mujoco
        import robosuite
        self.mj = mujoco
        versions = {"mujoco": mujoco.__version__, "robosuite": robosuite.__version__}
        if versions["mujoco"] != self._bd.engine.version:
            raise CapabilityError("bd/libero.yaml engine.version %r, installed mujoco %r"
                                  % (self._bd.engine.version, versions["mujoco"]))
        if versions["robosuite"] != self._bd.benchmark.commits.get("robosuite"):
            raise CapabilityError("bd/libero.yaml benchmark.commits.robosuite %r, installed robosuite %r"
                                  % (self._bd.benchmark.commits.get("robosuite"), versions["robosuite"]))
        self.versions = versions
        self._hooks = None  # type: Any
        self._env = None  # type: Any
        self._task = None  # type: Optional[str]
        self._steps = 0
        self._allocations = 0
        self._obs = None  # type: Any
        self._gripper_cmd = -1.0
        self._demo_files = {}  # type: Dict[str, DemoFile]
        self._init_states = {}  # type: Dict[str, np.ndarray]
        self._fresh_at = -1
        self._points_cache = {}  # type: Dict[str, Dict[str, np.ndarray]]

    # ------------------------------------------------------------------------------------------ scene

    @property
    def env(self) -> Any:
        if self._env is None:
            raise RuntimeError("no LIBERO environment yet: call reset_episode first")
        return self._env

    def task_ids(self) -> List[str]:
        return list(self._tasks)

    def _task_cfg(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self._tasks:
            raise ValueError("task_id %r is not one of the configured LIBERO tasks %s" % (task_id, sorted(self._tasks)))
        return self._tasks[task_id]

    def _suite_task(self, task_id: str) -> Tuple[Any, int]:
        from libero.libero import benchmark
        suite_name = self._task_cfg(task_id)["suite"]
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        names = [suite.get_task(i).name for i in range(suite.n_tasks)]
        return suite, names.index(self._task_cfg(task_id)["name"])

    def _build(self, task_id: str) -> None:
        """Build the LIBERO environment of ``task_id`` (original BDDL) and set up the host's view
        of it. ``hard_reset=False`` keeps the injected scene during the layout
        reset; rebuild_on_reset recreates it for the next episode."""
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        if self._env is not None:
            self.close()
        suite, index = self._suite_task(task_id)
        task = suite.get_task(index)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=self._hw, camera_widths=self._hw,
                                 camera_names=[self._cameras["primary"], self._cameras["wrist"]], hard_reset=False)
        self._env = env
        self._task = task_id
        self._language = str(task.language)
        self._suite, self._suite_index = suite, index
        env.seed(0)
        env.reset()
        # Extend the original task's compiled MJCF in memory. No LIBERO BDDL,
        # assets, task definitions or upstream files participate in this host.
        from .scene import add_hazard_pool, joint_layout
        self._original_layout = joint_layout(env.sim.model._model)
        original_stat = {key: np.array(getattr(env.sim.model._model.stat, key), copy=True)
                         for key in ("center", "extent", "meansize", "meanmass", "meaninertia")}
        # Use the source MJCF, not MjModel.get_xml(): the latter calls
        # mj_saveLastXML and changes numeric precision / compiled defaults.
        env.reset_from_xml_string(add_hazard_pool(env.env.model.get_xml(), self._pool_cfg))
        # Offscreen bodies must not enlarge the scene's automatic clipping /
        # shadow bounds or change the original model's scaling statistics.
        for key, value in original_stat.items():
            setattr(env.sim.model._model.stat, key, value)
        self._setup_scene()

    def _setup_scene(self) -> None:
        mj = self.mj
        inner = self._env.env
        self.sim = inner.sim
        self.m = self.sim.model._model
        self.d = self.sim.data._data
        m = self.m
        self.physics_dt = float(m.opt.timestep)
        self.control_dt = float(inner.control_timestep)
        self._check_timebase(inner)
        self.body_name = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        self.groups = RigidGroups(m, self.body_name)
        object_root = {}
        for name, obj in list(inner.objects_dict.items()) + list(inner.fixtures_dict.items()):
            object_root[m.body(obj.root_body).id] = name
        self.objects = {name: m.body(obj.root_body).id for name, obj in inner.objects_dict.items()}
        self.fixtures = {name: m.body(obj.root_body).id for name, obj in inner.fixtures_dict.items()}
        for item in self._pool_cfg["members"]:
            self.objects[item["name"]] = m.body(item["name"]).id
        self.label = {}  # type: Dict[str, str]
        for gname, frame in self.groups.frame_body.items():
            self.label[gname] = object_root.get(frame, gname)
        self.label_of_body = [self.label.get(self.groups.group_of[b], self.body_name[b]) if b in self.groups.group_of else self.body_name[b]
                              for b in range(m.nbody)]
        prefixes = tuple(self.config["robot_group_prefixes"])
        collides = {}  # type: Dict[str, bool]
        for g in range(m.ngeom):
            if is_collision_geom(m, g):
                collides[self.groups.group_of[int(m.geom_bodyid[g])]] = True
        self.robot_groups = [g for g in sorted(self.groups.frame_body, key=lambda n: self.groups.frame_body[n])
                             if g.startswith(prefixes) and collides.get(g)]
        self.robot_frame = [self.groups.frame_body[g] for g in self.robot_groups]
        self.robot_body_set = set()
        for g in self.robot_groups:
            self.robot_body_set.update(self.groups.members[g])
        fingers = list(self.config["finger_groups"])
        missing = [f for f in fingers if f not in self.robot_groups]
        if missing:
            raise CapabilityError("finger groups %s not among the robot groups %s" % (missing, self.robot_groups))
        self.finger_bodies = [set(self.groups.members[f]) for f in fingers]
        robot = inner.robots[0]
        self.eef_site = int(robot.eef_site_id)
        self.finger_qadr = [int(m.jnt_qposadr[m.joint(j).id]) for j in self.config["finger_joints"]]
        self.base_body = m.body(self.config["robot_base_body"]).id
        self.free_objects = {}  # type: Dict[str, Tuple[int, int, int]]
        for name, body in self.objects.items():
            for j in range(m.njnt):
                if int(m.jnt_bodyid[j]) == body and int(m.jnt_type[j]) == int(mj.mjtJoint.mjJNT_FREE):
                    self.free_objects[name] = (body, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
        self.joints = {}  # type: Dict[str, Tuple[int, int, int, int]]
        sizes = {int(mj.mjtJoint.mjJNT_FREE): (7, 6), int(mj.mjtJoint.mjJNT_BALL): (4, 3),
                 int(mj.mjtJoint.mjJNT_HINGE): (1, 1), int(mj.mjtJoint.mjJNT_SLIDE): (1, 1)}
        for j in range(m.njnt):
            nq, nv = sizes[int(m.jnt_type[j])]
            self.joints[mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j)] = (int(m.jnt_qposadr[j]), nq,
                                                                     int(m.jnt_dofadr[j]), nv)
        self._setup_pool()
        self._scratch = mj.MjData(m)
        self._fresh_at = -1
        self._warnings = [int(w.number) for w in self.d.warning]
        self._wrap_step()

    def _check_timebase(self, inner: Any) -> None:
        tb = self._bd.timebase
        substeps = int(round(inner.control_timestep / inner.model_timestep))
        bad = []
        if not math.isclose(tb.physics_dt_s, self.physics_dt, rel_tol=1e-9):
            bad.append("physics_dt_s %r vs opt.timestep %r" % (tb.physics_dt_s, self.physics_dt))
        if not math.isclose(tb.control_dt_s, self.control_dt, rel_tol=1e-9):
            bad.append("control_dt_s %r vs control_timestep %r" % (tb.control_dt_s, self.control_dt))
        if tb.substeps_per_action != substeps:
            bad.append("substeps_per_action %r vs %r" % (tb.substeps_per_action, substeps))
        if not math.isclose(float(inner.model_timestep), self.physics_dt, rel_tol=1e-12):
            bad.append("robosuite model_timestep %r vs opt.timestep %r" % (inner.model_timestep, self.physics_dt))
        if bad:
            raise CapabilityError("bd/libero.yaml timebase disagrees with the env: %s" % "; ".join(bad))

    def _setup_pool(self) -> None:
        m, mj, pool = self.m, self.mj, self._pool_cfg
        names = [item["name"] for item in pool["members"]]
        self.extras = sorted(n for n in self.free_objects
                             if any(s in n for s in pool["extras_substrings"]) and n not in names)
        self.members = []  # type: List[_Member]
        origin = np.asarray(pool["park_origin_m"], dtype=np.float64)
        spacing = np.asarray(pool["park_spacing_m"], dtype=np.float64)
        for i, name in enumerate(names + self.extras):
            if name not in self.free_objects:
                raise CapabilityError("LIBERO object %r has no free joint" % name)
            mem = _Member(name, i)
            mem.body, mem.qadr, mem.dadr = self.free_objects[name]
            mem.geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == mem.body and is_collision_geom(m, g)]
            mem.contype = [int(m.geom_contype[g]) for g in mem.geoms]
            mem.conaffinity = [int(m.geom_conaffinity[g]) for g in mem.geoms]
            mem.com = np.array(m.body_ipos[mem.body], dtype=np.float64)
            mem.mass = float(m.body_mass[mem.body])
            mem.park_pos = origin + i * spacing
            if name in self.extras:
                mem.unavailable = "LIBERO pedestal / extra (pool.extras_substrings), not an obstacle"
            else:
                self._describe_member(mem)
            m.dof_damping[mem.dadr:mem.dadr + 6] = 0.0
            self.members.append(mem)
        self.pool = [mem for mem in self.members if mem.name in names]
        self.hazard_geoms = {}  # type: Dict[int, _Member]
        for mem in self.members:
            for g in mem.geoms:
                self.hazard_geoms[g] = mem
            self._collision(mem, False)
            m.body_gravcomp[mem.body] = 1.0

    def _describe_member(self, mem: _Member) -> None:
        """``mem.agent`` if the member is exactly one box aligned with its body frame, centred on its centre of mass
        (see the module docstring); else ``mem.unavailable`` says why."""
        m, pool = self.m, self._pool_cfg
        if len(mem.geoms) != 1 or int(m.geom_type[mem.geoms[0]]) != int(self.mj.mjtGeom.mjGEOM_BOX):
            mem.unavailable = ("collision geometry is %d geom(s) of type(s) %s, not one box: no AgentSpec shape "
                               "describes it" % (len(mem.geoms), sorted(set(int(m.geom_type[g]) for g in mem.geoms))))
            return
        g = mem.geoms[0]
        rot = quat_to_mat(m.geom_quat[g])
        if not np.allclose(np.abs(rot), np.round(np.abs(rot)), atol=1e-6):
            mem.unavailable = "its box is not aligned with the body frame (quat %s)" % (list(m.geom_quat[g]),)
            return
        offset = float(np.linalg.norm(np.asarray(m.geom_pos[g]) - mem.com))
        if offset > float(pool["max_center_offset_m"]):
            mem.unavailable = ("box centre %.4f m from the centre of mass (> pool.max_center_offset_m %.4f)"
                               % (offset, float(pool["max_center_offset_m"])))
            return
        half = np.abs(rot).dot(np.asarray(m.geom_size[g], dtype=np.float64))
        mem.agent = AgentSpec(kind=str(pool["agent_kind"]), shape="box",
                              half_extents_m=[round(float(h), 6) for h in half], mass_kg=round(mem.mass, 6))

    def pool_table(self) -> List[Dict[str, Any]]:
        """Every pool member and extra: geometry summary and whether / why it is offered as a hazard."""
        m = self.m
        rows = []
        for mem in self.members:
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
            for g in mem.geoms:
                rot = quat_to_mat(m.geom_quat[g])
                ext = np.abs(rot).dot(np.asarray(m.geom_size[g], dtype=np.float64))
                lo = np.minimum(lo, np.asarray(m.geom_pos[g]) - ext)
                hi = np.maximum(hi, np.asarray(m.geom_pos[g]) + ext)
            rows.append({"name": mem.name, "geoms": len(mem.geoms), "mass_kg": mem.mass,
                         "com_m": mem.com.tolist(), "aabb_half_m": ((hi - lo) / 2).tolist(),
                         "aabb_centre_minus_com_m": ((hi + lo) / 2 - mem.com).tolist(),
                         "agent": None if mem.agent is None else mem.agent.to_dict(),
                         "unavailable": mem.unavailable})
        return rows

    def _collision(self, mem: _Member, on: bool) -> None:
        for g, ct, ca in zip(mem.geoms, mem.contype, mem.conaffinity):
            self.m.geom_contype[g] = ct if on else 0
            self.m.geom_conaffinity[g] = ca if on else 0

    def _wrap_step(self) -> None:
        sim = self.sim
        if getattr(sim, "_hazard_host", None) is self:
            return
        original = type(sim).step

        def step(with_udd: bool = True) -> None:
            self._before_physics_step()
            original(sim, with_udd)
            self._after_physics_step()

        sim.step = step
        sim._hazard_host = self

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        for f in self._demo_files.values():
            f.close()
        self._demo_files = {}

    # -------------------------------------------------------------------------------- HazardHost: setup

    def describe(self) -> Any:
        return self._bd

    def install_hooks(self, runtime: Any) -> None:
        self._hooks = runtime

    def alloc_hazard(self, env: int, agent: AgentSpec) -> HazardHandle:
        self._check_env(env)
        tol = float(self._pool_cfg["agent_match_abs_tol_m"])
        rel = float(self._pool_cfg["agent_match_rel_tol"])
        offered = []
        for mem in self.pool:
            if mem.agent is None:
                continue
            offered.append(mem.agent.to_dict())
            a = mem.agent
            if (agent.shape == a.shape and len(agent.half_extents_m) == len(a.half_extents_m)
                    and all(abs(float(x) - float(y)) <= tol for x, y in zip(agent.half_extents_m, a.half_extents_m))
                    and abs(float(agent.mass_kg) - a.mass_kg) <= rel * a.mass_kg and not mem.allocated):
                self._park(mem)
                self._allocations += 1
                mem.key = "%s.%d" % (mem.name, self._allocations)
                mem.allocated = True
                return HazardHandle(env=0, key=mem.key, native=mem)
        raise CapabilityError("no free LIBERO pool member matches the agent %s; offered agents (fixed size and "
                              "mass, H3 item 2): %s" % (agent.to_dict(), offered))

    def free_hazard(self, h: HazardHandle) -> None:
        mem = self._member(h)
        self._park(mem)
        mem.allocated = False
        mem.key = mem.name

    # -------------------------------------------------------------------------------- actuation

    def _set_pose(self, mem: _Member, com_pos: np.ndarray, quat: np.ndarray) -> None:
        d = self.d
        body_pos = com_pos - quat_to_mat(quat).dot(mem.com)
        d.qpos[mem.qadr:mem.qadr + 3] = body_pos
        d.qpos[mem.qadr + 3:mem.qadr + 7] = quat
        d.qvel[mem.dadr:mem.dadr + 6] = 0.0
        d.qfrc_applied[mem.dadr:mem.dadr + 6] = 0.0
        self._fresh_at = -1

    def _make_static(self, mem: _Member) -> None:
        self._collision(mem, False)
        self.m.body_gravcomp[mem.body] = 1.0
        mem.dynamic = False

    def _park(self, mem: _Member) -> None:
        self._make_static(mem)
        self._set_pose(mem, mem.park_pos, np.array([1.0, 0.0, 0.0, 0.0]))
        mem.parked = True
        mem.contacts = []
        mem.touching = set()
        mem.pos_before = None

    def set_hazard_pose(self, h: HazardHandle, pos: Sequence[float], quat: Sequence[float]) -> None:
        mem = self._member(h)
        self._make_static(mem)
        self._set_pose(mem, _vec3("pos", pos), _unit_quat(quat))

    def set_hazard_visible(self, h: HazardHandle, visible: bool) -> None:
        mem = self._member(h)
        if visible:
            mem.parked = False
        else:
            self._park(mem)

    def release_to_physics(self, h: HazardHandle, lin_vel: Sequence[float], ang_vel: Sequence[float]) -> None:
        mem = self._member(h)
        if mem.parked:
            raise RuntimeError("release_to_physics on parked hazard %r" % mem.key)
        lin, ang = _vec3("lin_vel", lin_vel), _vec3("ang_vel", ang_vel)
        self._collision(mem, True)
        self.m.body_gravcomp[mem.body] = 0.0
        quat = np.array(self.d.qpos[mem.qadr + 3:mem.qadr + 7], dtype=np.float64)
        rot = quat_to_mat(quat)
        r = rot.dot(mem.com)
        # free joint: qvel[0:3] = velocity of the body origin (world), qvel[3:6] = angular velocity (body frame)
        self.d.qvel[mem.dadr:mem.dadr + 3] = lin - np.cross(ang, r)
        self.d.qvel[mem.dadr + 3:mem.dadr + 6] = rot.T.dot(ang)
        mem.dynamic = True
        self._fresh_at = -1

    # ------------------------------------------------------------------------------------ state

    def hazard_state(self, h: HazardHandle) -> HazardState:
        mem = self._member(h)
        d = self.d
        body_pos = np.array(d.qpos[mem.qadr:mem.qadr + 3], dtype=np.float64)
        quat = np.array(d.qpos[mem.qadr + 3:mem.qadr + 7], dtype=np.float64)
        rot = quat_to_mat(quat)
        pos = body_pos + rot.dot(mem.com)
        if not mem.dynamic:
            return HazardState(pos=pos, quat=quat, lin_vel_mps=np.zeros(3), ang_vel_radps=np.zeros(3))
        v0 = np.array(d.qvel[mem.dadr:mem.dadr + 3], dtype=np.float64)
        w = rot.dot(np.array(d.qvel[mem.dadr + 3:mem.dadr + 6], dtype=np.float64))
        return HazardState(pos=pos, quat=quat, lin_vel_mps=v0 + np.cross(w, rot.dot(mem.com)), ang_vel_radps=w)

    def hazard_contacts(self, h: HazardHandle) -> List[ContactObs]:
        mem = self._member(h)
        out, mem.contacts = mem.contacts, []
        return out

    def _fresh(self) -> Any:
        """Scratch ``MjData`` with the current (end-of-step) state's kinematics and velocities. The main data's
        derived quantities describe the start of the last physics step (``mj_step`` integrates after computing
        them), the scratch copy describes its end; the main data is left alone."""
        if self._fresh_at == self._steps:
            return self._scratch
        mj, m, d, s = self.mj, self.m, self.d, self._scratch
        s.time = d.time
        s.qpos[:] = d.qpos
        s.qvel[:] = d.qvel
        if m.na:
            s.act[:] = d.act
        if m.nmocap:
            s.mocap_pos[:] = d.mocap_pos
            s.mocap_quat[:] = d.mocap_quat
        mj.mj_kinematics(m, s)
        mj.mj_comPos(m, s)
        mj.mj_comVel(m, s)
        self._fresh_at = self._steps
        return s

    def _velocity(self, data: Any, objtype: Any, objid: int) -> np.ndarray:
        res = np.zeros(6)
        self.mj.mj_objectVelocity(self.m, data, objtype, objid, res, 0)
        return res

    def robot_state(self, env: int) -> RobotState:
        self._check_env(env)
        mj, s = self.mj, self._fresh()
        eef_pos = np.array(s.site_xpos[self.eef_site], dtype=np.float64)
        eef_quat = mat_to_quat(np.asarray(s.site_xmat[self.eef_site]).reshape(3, 3))
        eef_vel = self._velocity(s, mj.mjtObj.mjOBJ_SITE, self.eef_site)[3:]
        names, pos, quats, vels = list(self.robot_groups), [], [], []
        for b in self.robot_frame:
            pos.append(np.array(s.xpos[b]))
            quats.append(np.array(s.xquat[b]))
            vels.append(self._velocity(s, mj.mjtObj.mjOBJ_XBODY, b)[3:])
        held = self._held_object()
        if held is not None:
            b = self.objects[held]
            names.append(held)
            pos.append(np.array(s.xpos[b]))
            quats.append(np.array(s.xquat[b]))
            vels.append(self._velocity(s, mj.mjtObj.mjOBJ_XBODY, b)[3:])
        q = self.d.qpos
        opening = float(q[self.finger_qadr[0]] - q[self.finger_qadr[1]])
        return RobotState(eef_pose=Pose(eef_pos, eef_quat), eef_lin_vel_mps=eef_vel, collision_body_pos=pos,
                          gripper_opening_m=opening,
                          base_pose=Pose(np.array(s.xpos[self.base_body]), np.array(s.xquat[self.base_body])),
                          held_object=held, collision_body_names=names, collision_body_lin_vel_mps=vels,
                          held_object_bodies=None if held is None else [held], collision_body_quat=quats)

    def _held_object(self) -> Optional[str]:
        """The free-joint object touching both finger groups in the engine's last contact list."""
        d, m = self.d, self.m
        sides = {}  # type: Dict[str, set]
        parked = set(mem.body for mem in self.members)
        for i in range(int(d.ncon)):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for fb, ob in ((b1, b2), (b2, b1)):
                for side, bodies in enumerate(self.finger_bodies):
                    if fb in bodies:
                        label = self.label_of_body[ob]
                        if label in self.free_objects and ob not in parked:
                            sides.setdefault(label, set()).add(side)
        held = sorted(name for name, s in sides.items() if len(s) == 2)
        return held[0] if held else None

    def object_poses(self, env: int) -> Dict[str, Pose]:
        """Task objects (not pool members or extras) and fixtures by name, plus the moving parts of fixtures
        (drawers, buttons: jointed rigid groups) by group name; world-frame body poses (wxyz)."""
        self._check_env(env)
        s = self._fresh()
        hidden = set(mem.name for mem in self.members)
        out = {}
        for name, b in sorted(self.objects.items()):
            if name not in hidden:
                out[name] = Pose(np.array(s.xpos[b]), np.array(s.xquat[b]))
        for name, b in sorted(self.fixtures.items()):
            out[name] = Pose(np.array(s.xpos[b]), np.array(s.xquat[b]))
            for gname, members in self.groups.members.items():
                frame = self.groups.frame_body[gname]
                if gname not in out and self._descends(frame, b) and frame != b and self._jointed(frame):
                    out[gname] = Pose(np.array(s.xpos[frame]), np.array(s.xquat[frame]))
        return out

    def _descends(self, b: int, root: int) -> bool:
        while b != 0:
            if b == root:
                return True
            b = int(self.m.body_parentid[b])
        return False

    def _jointed(self, b: int) -> bool:
        return any(int(self.m.jnt_bodyid[j]) == b for j in range(int(self.m.njnt)))

    def time_state(self, env: int) -> TimeState:
        self._check_env(env)
        return TimeState(sim_time_s=self._steps * self.physics_dt, physics_dt_s=self.physics_dt,
                         control_dt_s=self.control_dt)

    def task_success(self, env: int) -> bool:
        """LIBERO's own goal test (``BDDLBaseDomain._check_success``: the BDDL goal predicates), read only."""
        self._check_env(env)
        return bool(self.env.env._check_success())

    def hold_action(self, env: int) -> np.ndarray:
        """OSC_POSE delta action with zero motion (the controller's goal becomes the current pose) and the last
        gripper command (a held object stays held)."""
        self._check_env(env)
        controller = self.env.env.robots[0].controller
        controller.update(force=True)
        controller.reset_goal()
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._gripper_cmd])

    # --------------------------------------------------------------------------- optional methods

    def min_distance(self, h: HazardHandle) -> float:
        """Signed geometric distance (``mj_geomDistance``, end-of-step state) from the hazard's collision geoms
        to the robot's and the held object's, clipped at 0 and at ``min_distance_search_m``."""
        mem = self._member(h)
        s, m = self._fresh(), self.m
        targets = set(self.robot_body_set)
        held = self._held_object()
        if held is not None:
            targets.add(self.objects[held])
        best = self._search
        fromto = np.zeros(6)
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in targets or not is_collision_geom(m, g):
                continue
            for hg in mem.geoms:
                gap = float(np.linalg.norm(s.geom_xpos[g] - s.geom_xpos[hg])) - m.geom_rbound[g] - m.geom_rbound[hg]
                if gap >= best:
                    continue
                best = min(best, float(self.mj.mj_geomDistance(m, s, hg, g, best, fromto)))
        return max(0.0, best)

    def support_surfaces(self, env: int) -> List[Surface]:
        """Top faces of the collision boxes of ``support_bodies`` (the table), from the compiled model. A scene
        without any of them (libero_object: the objects stand on the floor) uses ``support_fallback_bodies``,
        whose plane geoms count with their declared half extents (the plane itself is infinite)."""
        self._check_env(env)
        out = self._support_surfaces_of(self.config["support_bodies"], False)
        return out or self._support_surfaces_of(self.config["support_fallback_bodies"], True)

    def _support_surfaces_of(self, body_names: List[str], planes: bool) -> List[Surface]:
        s, m = self._fresh(), self.m
        wanted = [int(self.mj.mjtGeom.mjGEOM_BOX)] + ([int(self.mj.mjtGeom.mjGEOM_PLANE)] if planes else [])
        out = []
        for body_name in body_names:
            if body_name not in self.body_name:
                continue
            b = m.body(body_name).id
            for g in range(m.ngeom):
                if not self._descends(int(m.geom_bodyid[g]), b) or not is_collision_geom(m, g):
                    continue
                if int(m.geom_type[g]) not in wanted:
                    continue
                rot = np.asarray(s.geom_xmat[g]).reshape(3, 3)
                size = np.asarray(m.geom_size[g], dtype=np.float64)
                is_plane = int(m.geom_type[g]) == int(self.mj.mjtGeom.mjGEOM_PLANE)
                if is_plane:
                    size = np.array([size[0], size[1], 0.0])
                ext = np.abs(rot).dot(size)
                c = np.asarray(s.geom_xpos[g], dtype=np.float64)
                top = float(c[2] + ext[2])
                name = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_GEOM, g) or "geom%d" % g
                out.append(Surface(name=name, min_m=[float(c[0] - ext[0]), float(c[1] - ext[1]), top],
                                   max_m=[float(c[0] + ext[0]), float(c[1] + ext[1]), top],
                                   source="model: collision %s %s of body %s" % (
                                       "plane" if is_plane else "box", name, body_name)))
        return out

    def render_check(self, h: HazardHandle) -> bool:
        """Whether a hazard geom (collision or visual) owns a pixel of a policy camera (segmentation render at
        ``camera_hw``; the main data is the start-of-step state of the last physics step, as the policy images)."""
        mem = self._member(h)
        geoms = set(g for g in range(self.m.ngeom) if int(self.m.geom_bodyid[g]) == mem.body)
        for camera in (self._cameras["primary"], self._cameras["wrist"]):
            seg = self.sim.render(self._hw, self._hw, camera_name=camera, segmentation=True)
            seg = np.asarray(seg)
            ids = seg[..., 1][seg[..., 0] == int(self.mj.mjtObj.mjOBJ_GEOM)]
            if any(int(g) in geoms for g in np.unique(ids)):
                return True
        return False

    def robot_collision_points(self, env: int) -> Dict[str, np.ndarray]:
        """Surface samples (``geometry.group_points``) of every robot group, in its frame (D15 a)."""
        self._check_env(env)
        spacing = float(self._bd.robot_point_spacing_m)
        key = "%s:%r" % (self._task, spacing)
        if key not in self._points_cache:
            self._points_cache[key] = {g: group_points(self.m, self.groups, g, spacing) for g in self.robot_groups}
        return {k: v.copy() for k, v in self._points_cache[key].items()}

    # ------------------------------------------------------------------------------- EpisodeDriver

    def _demo(self, task_id: str, demo_set: str) -> DemoFile:
        key = "%s/%s" % (demo_set, task_id)
        if key not in self._demo_files:
            name = self._task_cfg(task_id)["name"]
            self._demo_files[key] = DemoFile(datasets_root() / demo_set / ("%s_demo.hdf5" % name))
        return self._demo_files[key]

    def layout_state(self, task_id: str, layout_id: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """``(state, gripper_command, info)`` of a layout in the current scene (the scene of ``task_id`` must be
        built): the LIBERO ``[time, qpos, qvel]`` before parking, the gripper command the layout's history
        ends with (-1 open at a demonstration start), and the provenance."""
        layout = parse_layout(layout_id)
        width = 1 + int(self.m.nq) + int(self.m.nv)
        if layout.kind == "init":
            states = self._init_file(task_id)
            if not 0 <= layout.index < len(states):
                raise ValueError("%s: init file has %d states" % (layout_id, len(states)))
            state = np.asarray(states[layout.index], dtype=np.float64)
            state, unfilled = map_plain_state(state, self._original_layout, self.joints, width)
            return state, -1.0, {"layout_kind": "init", "unfilled_joints": unfilled}
        expected = self._task_cfg(task_id)["demo_set"]
        if layout.demo_set != expected:
            raise ValueError("layout %r: task %r has its demonstrations in %r" % (layout_id, task_id, expected))
        demo = self._demo(task_id, layout.demo_set)
        states = demo.states(layout.episode)
        if not 0 <= layout.start < len(states):
            raise ValueError("%s: episode has %d states" % (layout_id, len(states)))
        state, unfilled = map_plain_state(states[layout.start], demo.joint_layout(layout.episode), self.joints, width)
        grip = -1.0
        if layout.start > 0:
            grip = float(np.sign(demo.actions(layout.episode)[layout.start - 1][-1])) or -1.0
        return state, grip, {"layout_kind": "demo", "unfilled_joints": unfilled}

    def _init_file(self, task_id: str) -> np.ndarray:
        if task_id not in self._init_states:
            suite, index = self._suite_task(task_id)
            self._init_states[task_id] = np.asarray(suite.get_task_init_states(index), dtype=np.float64)
        return self._init_states[task_id]

    def _gripper_history(self, task_id: str, layout_id: str) -> Optional[np.ndarray]:
        """For a mid-demonstration layout (``@j``, j > 0): the robosuite gripper's internal ramp state
        (``PandaGripper.current_action``) after the demonstration's first j actions, starting from its reset value;
        ``None`` for layouts that start an episode."""
        layout = parse_layout(layout_id)
        if layout.kind != "demo" or layout.start == 0:
            return None
        gripper = self.env.env.robots[0].gripper
        current = np.zeros_like(gripper.current_action, dtype=np.float64)
        substeps = int(self._bd.timebase.substeps_per_action)
        for a in self._demo(task_id, layout.demo_set).actions(layout.episode)[:layout.start]:
            for _ in range(substeps):
                current = np.clip(current + np.array([-1.0, 1.0]) * float(gripper.speed) * np.sign(a[-1]), -1.0, 1.0)
        return current

    def reset_episode(self, env: int, task_id: str, layout_id: str, seed: int) -> None:
        """Build a fresh task scene when configured, seed and reset it, then set the layout's state with
        every pool member and extra parked (``set_init_state``), run ``settle_steps`` hold steps, then
        ``begin_episode``. The seed only seeds robosuite's reset; the layout state overwrites what it samples."""
        self._check_env(env)
        if self._env is None or self._task != task_id or self.config["rebuild_on_reset"]:
            self._task_cfg(task_id)
            self._build(task_id)
        self._env.seed(int(seed))
        self._env.reset()
        state, grip, _info = self.layout_state(task_id, layout_id)
        nq = int(self.m.nq)
        for mem in self.members:
            if mem.allocated:
                raise RuntimeError("reset_episode with hazard %r still allocated" % mem.key)
            self._make_static(mem)
            mem.parked, mem.contacts, mem.touching, mem.pos_before = True, [], set(), None
            state[1 + mem.qadr:1 + mem.qadr + 7] = np.concatenate([mem.park_pos - mem.com, [1.0, 0.0, 0.0, 0.0]])
            state[1 + nq + mem.dadr:1 + nq + mem.dadr + 6] = 0.0
        self.d.qfrc_applied[:] = 0.0
        gripper = self._env.env.robots[0].gripper
        gripper.current_action = np.zeros_like(gripper.current_action)
        ramp = self._gripper_history(task_id, layout_id)
        if ramp is not None:
            self._env.env.robots[0].gripper.current_action = ramp
        self._obs = self._env.set_init_state(state)
        controller = self._env.env.robots[0].controller
        controller.update(force=True)
        controller.reset_goal()
        self._gripper_cmd = grip
        self._fresh_at = -1
        for _ in range(self._settle):
            self._obs, _, _, _ = self._env.step(self.hold_action(0))
        if self._hooks is not None:
            self._hooks.begin_episode(0)

    def step(self, env: int, action: Any) -> None:
        self._check_env(env)
        action = np.asarray(action, dtype=np.float64)
        self._gripper_cmd = float(action[-1])
        self._obs, _, _, _ = self.env.step(action)

    def policy_observation(self, env: int) -> Dict[str, Any]:
        """What dynisafety's LIBERO backend gives a policy (module docstring); arrays are copies."""
        self._check_env(env)
        if self._obs is None:
            raise RuntimeError("policy_observation before reset_episode")
        obs = self._obs
        primary = np.ascontiguousarray(obs[self._cameras["primary"] + "_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs[self._cameras["wrist"] + "_image"][::-1, ::-1])
        state = np.concatenate([np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3),
                                quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
                                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)])
        controller = self.env.env.robots[0].controller
        rot = np.asarray(controller.ee_ori_mat, dtype=np.float64)
        ee = np.concatenate([np.asarray(controller.ee_pos, dtype=np.float32).reshape(3),
                             np.concatenate([rot[:3, 0], rot[:3, 1]]).astype(np.float32),
                             np.zeros(1, dtype=np.float32), np.zeros(10, dtype=np.float32)])
        return {"primary": primary, "wrist": wrist, "state": state.astype(np.float32), "ee_proprio_20d": ee}

    def language_instruction(self, env: int) -> str:
        """The task's language instruction (BDDL ``:language`` via the LIBERO benchmark), for policy bridges."""
        self._check_env(env)
        return self._language

    # ---------------------------------------------------------------------- physics-step listener

    def _before_physics_step(self) -> None:
        self._check_stability()
        active = [mem for mem in self.pool if mem.allocated and not mem.parked]
        for mem in active:
            mem.pos_before = self._com_world(mem)
        if self._hooks is not None:
            self._hooks.pre_step(0)
        dynamic = [mem for mem in self.pool if mem.allocated and mem.dynamic]
        for mem in self.pool:
            self.d.qfrc_applied[mem.dadr:mem.dadr + 6] = 0.0
        if dynamic:
            self.mj.mj_forward(self.m, self.d)
            for mem in dynamic:
                self.d.qfrc_applied[mem.dadr:mem.dadr + 6] = -self.d.qfrc_fluid[mem.dadr:mem.dadr + 6]

    def _after_physics_step(self) -> None:
        self._steps += 1
        self._fresh_at = -1
        active = [mem for mem in self.pool if mem.allocated and not mem.parked]
        if active:
            self._collect_contacts(active)
        self._check_stability()
        if self._hooks is not None:
            self._hooks.post_step(0)

    def _com_world(self, mem: _Member) -> np.ndarray:
        q = self.d.qpos
        return np.array(q[mem.qadr:mem.qadr + 3]) + quat_to_mat(q[mem.qadr + 3:mem.qadr + 7]).dot(mem.com)

    def _collect_contacts(self, active: List[_Member]) -> None:
        mj, m = self.mj, self.m
        s = self._fresh()
        switched = [mem for mem in active if not mem.dynamic]
        for mem in switched:
            self._collision(mem, True)
        try:
            mj.mj_collision(m, s)
        finally:
            for mem in switched:
                self._collision(mem, False)
        t1 = self._steps * self.physics_dt
        t0 = t1 - self.physics_dt
        held = self._held_object()
        deepest = {}  # type: Dict[int, Dict[str, Tuple[float, np.ndarray, np.ndarray, int]]]
        for i in range(int(s.ncon)):
            c = s.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            n = np.array(c.frame[:3], dtype=np.float64)
            for hz, other, normal in ((g1, g2, -n), (g2, g1, n)):
                mem = self.hazard_geoms.get(hz)
                if mem is None or mem not in active:
                    continue
                omem = self.hazard_geoms.get(other)
                if omem is not None:
                    if omem is mem or omem not in active:
                        continue
                    name = omem.key
                else:
                    name = self.label_of_body[int(m.geom_bodyid[other])]
                dist = float(c.dist)
                slot = deepest.setdefault(id(mem), {})
                if name not in slot or dist < slot[name][0]:
                    slot[name] = (dist, np.array(c.pos, dtype=np.float64), normal, other)
        for mem in active:
            slot = deepest.get(id(mem), {})
            if not slot:
                mem.touching = set()
                continue
            pos_now = self._com_world(mem)
            for name in sorted(slot):
                dist, position, normal, other = slot[name]
                stamp = t1 if name in mem.touching else self._onset(mem, other, normal, dist, pos_now, t0, t1)
                mem.contacts.append(ContactObs(partner_name=name, partner_class=self._partner_class(name, other, held),
                                               position=position, normal=normal, time_s=stamp))
            mem.touching = set(slot)

    def _partner_class(self, name: str, geom: int, held: Optional[str]) -> PartnerClass:
        if geom in self.hazard_geoms:
            return PartnerClass.HAZARD
        if int(self.m.geom_bodyid[geom]) in self.robot_body_set:
            return PartnerClass.ROBOT_LINK
        if held is not None and name == held:
            return PartnerClass.HELD_OBJECT
        return PartnerClass.SCENE

    def _onset(self, mem: _Member, geom: int, normal: np.ndarray, dist: float, pos_now: np.ndarray, t0: float,
               t1: float) -> float:
        """Time in [t0, t1] at which the closing motion over the step makes the overlap ``-dist`` begin."""
        if mem.pos_before is None:
            return t0
        depth = self._touch - dist
        hazard_vel = (pos_now - mem.pos_before) / (t1 - t0)
        partner_vel = self._velocity(self._fresh(), self.mj.mjtObj.mjOBJ_XBODY, int(self.m.geom_bodyid[geom]))[3:]
        closing = float(np.dot(hazard_vel - partner_vel, -normal))
        if closing <= 0.0:
            return t0
        return max(t0, t1 - depth / closing)

    def _check_stability(self) -> None:
        if self._hooks is None:
            return
        d, m = self.d, self.m
        found = None  # type: Optional[Tuple[str, bool, Optional[List[float]]]]
        for i, w in enumerate(d.warning):
            if int(w.number) > self._warnings[i]:
                self._warnings = [int(x.number) for x in d.warning]
                found = (INSTABILITY_WARNING, False, None)
                break
        if found is None:
            for mem in self.pool:
                if mem.allocated and not mem.parked:
                    kind = self._unstable(d.qpos[mem.qadr:mem.qadr + 3], d.qvel[mem.dadr:mem.dadr + 3],
                                          d.qvel[mem.dadr + 3:mem.dadr + 6])
                    if kind is not None:
                        found = (kind, True, self._finite(d.qpos[mem.qadr:mem.qadr + 3]))
                        break
        if found is None:
            hidden = set(mem.name for mem in self.members)
            for name, (b, qadr, dadr) in sorted(self.free_objects.items()):
                if name in hidden:
                    continue
                kind = self._unstable(d.qpos[qadr:qadr + 3], d.qvel[dadr:dadr + 3], d.qvel[dadr + 3:dadr + 6])
                if kind is not None:
                    found = (kind, False, self._finite(d.qpos[qadr:qadr + 3]))
                    break
        if found is None:
            if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
                found = (INSTABILITY_NONFINITE, False, None)
            elif m.nv and float(np.max(np.abs(d.qvel))) > max(self._max_joint, self._max_lin, self._max_ang):
                found = (INSTABILITY_SPEED, False, None)
        if found is None:
            return
        kind, is_hazard, pos = found
        try:
            self._hooks.report_instability(0, kind, entity_is_hazard=is_hazard, entity_pos=pos)
        except RuntimeError as exc:
            if "outside an episode" not in str(exc):
                raise

    def _unstable(self, pos: Sequence[float], lin: Sequence[float], ang: Sequence[float]) -> Optional[str]:
        values = np.concatenate([np.asarray(pos), np.asarray(lin), np.asarray(ang)])
        if not np.all(np.isfinite(values)):
            return INSTABILITY_NONFINITE
        if np.linalg.norm(lin) > self._max_lin or np.linalg.norm(ang) > self._max_ang:
            return INSTABILITY_SPEED
        return None

    @staticmethod
    def _finite(pos: Sequence[float]) -> Optional[List[float]]:
        values = [float(v) for v in pos]
        return values if all(math.isfinite(v) for v in values) else None

    # ------------------------------------------------------------------------------- internals

    def _check_env(self, env: Any) -> None:
        if isinstance(env, bool) or not isinstance(env, (int, np.integer)) or int(env) != 0:
            raise ValueError("LIBERO has one environment (env 0), got %r" % (env,))

    def _member(self, h: HazardHandle) -> _Member:
        mem = h.native
        if not isinstance(mem, _Member) or mem not in self.pool or mem.key != h.key:
            raise ValueError("unknown hazard handle %r" % (h,))
        if not mem.allocated:
            raise RuntimeError("hazard %r is not allocated" % mem.key)
        return mem

    def inject_engine_fault(self, env: int, h: Optional[HazardHandle]) -> None:
        """Conformance C11: give a body a speed of ``fault_injection_speed_mps`` straight up behind the runtime's
        back (the released hazard if ``h`` is given, made dynamic first; else the first visible free object). The
        instability detector reports it at the next physics step."""
        self._check_env(env)
        speed = float(self.config["fault_injection_speed_mps"])
        if h is None:
            hidden = set(mem.name for mem in self.members)
            name = sorted(n for n in self.free_objects if n not in hidden)[0]
            _, _, dadr = self.free_objects[name]
        else:
            mem = self._member(h)
            self._collision(mem, True)
            self.m.body_gravcomp[mem.body] = 0.0
            mem.dynamic = True
            dadr = mem.dadr
        self.d.qvel[dadr:dadr + 3] = [0.0, 0.0, speed]
        self._fresh_at = -1
