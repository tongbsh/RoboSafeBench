"""``CalvinHost``: the CALVIN adapter (HazardHost + EpisodeDriver, 01 §3.1 / §3.1a; task H1).

Engine primitives only; every hazard decision is the runtime's. One PyBullet world (``num_envs`` 1, ``env`` 0).

**Clock and hooks.** ``HookedPlayTableEnv`` (``env.py``) calls :meth:`before_physics_step` /
:meth:`after_physics_step` around each of the 8 ``stepSimulation`` calls of a control step; they forward to the
installed runtime's ``pre_step`` / ``post_step``. ``sim_time_s`` is the number of physics steps since the env was
built (the reset's own ``stepSimulation`` included) times ``1/240`` s. ``reset_episode`` rebuilds the PyBullet
world (the only way to make an episode independent of the previous ones, see :meth:`CalvinHost._rebuild_world`),
resets the env to a dataset frame and then calls ``begin_episode``.

**Hazard bodies.** Spheres and boxes made with ``createMultiBody(useMaximalCoordinates=True)``: a free rigid body
whose linear / angular damping is 0 (set explicitly, probes/physics.py) and that never sleeps. Bodies are pooled
by (shape, half extents, mass) and named ``hazard_<n>`` (their contact partner name); ``free_hazard`` parks a
body for reuse, and every allocation gets a new handle key ``hazard_<n>.<allocation count>``. *Parked*:
mass 0 (static), collision filter mask 0 (collides with nothing), at its park pose far outside every camera.
*Kinematic* (``set_hazard_pose``): the same static, filtered body at the commanded pose with zero velocity, so
it neither pushes nor is pushed by anything. *Released*: mass ``agent.mass_kg``, the configured collision
filter, the commanded velocities (``resetBaseVelocity`` reads back exactly, probes/physics.py). The hazard is
not one of the scene's objects, so it never enters ``scene_obs`` or the scene info read by the oracle.

**Contacts.** After every physics step, for every unparked hazard, ``getClosestPoints`` against every other body
with a collision shape gives the pairs within ``contacts.touch_distance_m``; the deepest point per partner is one
``ContactObs`` (normal ``contactNormalOnB``: from the partner toward the hazard). This sees an overlap in the step
in which it appears, also while the hazard is kinematic and filtered (the engine's own contact points are
computed from the poses before a step, so they arrive one step later, probes/physics.py). A partner newly in
contact is time-stamped where the closing motion over the step makes the overlap begin (clamped to the step);
a continuing contact at the end of the step. Partner names: ``<body>`` for single-link bodies (blocks, the floor
plane), ``<body>/<link>`` for articulated ones, where links joined by fixed joints form one rigid body, named after
the base link if it contains the base, else after its heaviest link (``robot/panda_hand`` covers link 7, the hand
and the camera mounts; ``robot/panda_leftfinger`` its fingertip; ``table/base_link`` the plank and the lamps);
hazards by their key.

**Robot.** EEF = the ``tcp`` link (what ``robot_obs`` reports). Collision bodies = the robot's rigid bodies that
have collision shapes, positions and velocities of their named link's centre of mass, followed by the held
object's (when there is one). ``gripper_opening_m`` is the measured finger-joint sum (``robot_obs[6]``).
``held_object`` is the movable object that touches both fingers.

**Oracle.** ``task_success`` forwards ``Tasks.get_task_info_for_set(start_info, env.get_info(), {task_id})``
exactly like ``evaluate_policy.rollout``; ``start_info`` is the info right after the reset (or of the layout's
oracle start frame).
"""
import copy
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hazardlib.config import ConfigError, load_bd, load_config
from hazardlib.host import (CapabilityError, ContactObs, HazardHandle, HazardState, PartnerClass, Pose,
                            RobotState, Surface, TimeState)
from hazardlib.spec.hazard_spec import AgentSpec

from .dataset import CalvinDataset
from .env import calvin_env_root, make_env, pin_egl_device

HOST_CONFIG = "calvin"
DATASET_ENV = "CALVIN_DATASET_DIR"
EGL_ENV = "CALVIN_EGL_DEVICE"
SHAPES = ("sphere", "box")
FINGER_ROOTS = ("panda_leftfinger", "panda_rightfinger")
INSTABILITY_NONFINITE = "calvin_nonfinite_state"
INSTABILITY_SPEED = "calvin_speed_limit"


def _wxyz(quat_xyzw: Sequence[float]) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def _xyzw(quat_wxyz: Sequence[float]) -> List[float]:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quat must be 4 finite numbers (wxyz), got %r" % (quat_wxyz,))
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("quat must not be zero")
    q = q / norm
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _vec3(name: str, value: Any) -> List[float]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError("%s must be 3 finite numbers, got %r" % (name, value))
    return [float(v) for v in arr]


class _Hazard:
    """One pooled hazard body: ``name`` is the body's (contact partner name), ``key`` the current allocation's."""

    def __init__(self, name: str, uid: int, signature: Tuple[Any, ...], mass: float, park_pos: List[float]) -> None:
        self.name = name
        self.key = name
        self.uid = uid
        self.signature = signature
        self.mass = mass
        self.park_pos = park_pos
        self.allocated = False
        self.parked = True
        self.dynamic = False
        self.contacts = []  # type: List[ContactObs]
        self.touching = set()  # type: set
        self.pos_before = None  # type: Optional[np.ndarray]


class _Bodies:
    """Names of the bodies and rigid link groups of the CALVIN world (built once; hazards are added later)."""

    def __init__(self, env: Any) -> None:
        p, cid = env.p, env.cid
        self.robot_uid = env.robot.robot_uid
        names = {self.robot_uid: "robot"}
        for obj in env.scene.get_objects():
            names[obj.uid] = obj.name
        for i in range(p.getNumBodies(physicsClientId=cid)):
            uid = p.getBodyUniqueId(i, physicsClientId=cid)
            if uid not in names:
                names[uid] = p.getBodyInfo(uid, physicsClientId=cid)[1].decode() or "body%d" % uid
        self.body_name = names
        self.movable = {obj.uid: obj.name for obj in env.scene.movable_objects}
        self.link_name = {}  # type: Dict[Tuple[int, int], str]
        self.partner = {}  # type: Dict[Tuple[int, int], str]
        self.representative = {}  # type: Dict[str, Tuple[int, int]]
        self.collision_uids = []  # type: List[int]
        for uid, name in sorted(names.items()):
            n = p.getNumJoints(uid, physicsClientId=cid)
            links = [-1] + list(range(n))
            parent, fixed = {}, {}
            self.link_name[(uid, -1)] = p.getBodyInfo(uid, physicsClientId=cid)[0].decode()
            for j in range(n):
                info = p.getJointInfo(uid, j, physicsClientId=cid)
                self.link_name[(uid, j)] = info[12].decode()
                parent[j] = info[16]
                fixed[j] = info[2] == p.JOINT_FIXED
            groups = {}  # type: Dict[int, List[int]]
            for link in links:
                root = link
                while root != -1 and fixed[root]:
                    root = parent[root]
                groups.setdefault(root, []).append(link)
            has_collision = False
            for root, members in sorted(groups.items()):
                if root == -1:
                    heaviest = -1
                else:
                    heaviest = max(members, key=lambda l: (p.getDynamicsInfo(uid, l, physicsClientId=cid)[0], -l))
                label = name if n == 0 else "%s/%s" % (name, self.link_name[(uid, heaviest)])
                collides = False
                for link in members:
                    self.partner[(uid, link)] = label
                    if p.getCollisionShapeData(uid, link, physicsClientId=cid):
                        collides = True
                if collides:
                    self.representative[label] = (uid, heaviest)
                    has_collision = True
            if has_collision:
                self.collision_uids.append(uid)
        self.robot_bodies = sorted((label for label, (uid, _) in self.representative.items()
                                    if uid == self.robot_uid),
                                   key=lambda label: self.representative[label][1])
        fingers = {}
        for label in self.robot_bodies:
            link = self.link_name[self.representative[label]]
            if link in FINGER_ROOTS:
                fingers[link] = self.representative[label][1]
        if sorted(fingers) != sorted(FINGER_ROOTS):
            raise CapabilityError("the CALVIN robot has no collision bodies %s (found %s)"
                                  % (list(FINGER_ROOTS), self.robot_bodies))
        self.finger_links = {}  # type: Dict[int, int]
        for side, root in enumerate(FINGER_ROOTS):
            for (uid, link), label in self.partner.items():
                if uid == self.robot_uid and label.endswith("/" + root):
                    self.finger_links[link] = side


class CalvinHost:
    """The CALVIN adapter. ``dataset_dir`` defaults to ``$CALVIN_DATASET_DIR``, ``egl_device`` (index in
    ``eglQueryDevicesEXT``, see probes/egl_devices.py) to ``$CALVIN_EGL_DEVICE``; ``config_dir`` to hazardlib's
    default. The BD is ``bd/calvin.yaml`` loaded strictly (no ``unknown`` left)."""

    def __init__(self, dataset_dir: Optional[str] = None, egl_device: Optional[int] = None,
                 config_dir: Optional[str] = None) -> None:
        dataset_dir = dataset_dir if dataset_dir is not None else os.environ.get(DATASET_ENV)
        if not dataset_dir:
            raise CapabilityError("CalvinHost needs the dataset directory (argument or $%s)" % DATASET_ENV)
        if egl_device is None:
            if os.environ.get(EGL_ENV) is None:
                raise CapabilityError("CalvinHost needs the EGL device index (argument or $%s)" % EGL_ENV)
            egl_device = int(os.environ[EGL_ENV])
        self.config = load_config(HOST_CONFIG, strict=True, config_dir=config_dir)
        self._bd = load_bd("calvin", strict=True, config_dir=config_dir)
        cfg = self.config
        self.dataset = CalvinDataset(dataset_dir, cfg["dataset"]["splits"])
        pin_egl_device(int(egl_device))
        env_split = self.dataset.splits[cfg["dataset"]["env_split"]]
        self.env = make_env(os.path.join(dataset_dir, env_split), list(cfg["cameras"]))
        self.env.listener = self
        self.p, self.cid = self.env.p, self.env.cid
        self.oracle = self._load_oracle(cfg["oracle_tasks_config"])
        self.physics_dt = float(self.p.getPhysicsEngineParameters(physicsClientId=self.cid)["fixedTimeStep"])
        self.control_dt = 1.0 / float(self.env.control_freq)
        self._check_timebase()
        self._rng_state = copy.deepcopy(self.env.np_random.bit_generator.state)
        self._world = self._world_signature()
        self.bodies = _Bodies(self.env)
        self._hazards = []  # type: List[_Hazard]
        self._allocations = 0
        self._hooks = None  # type: Any
        self._task = None  # type: Optional[str]
        self._start_info = None  # type: Any
        self._obs = None  # type: Any
        contacts = cfg["contacts"]
        self._touch = float(contacts["touch_distance_m"])
        self._search = float(cfg["min_distance_search_m"])
        limits = cfg["instability"]
        self._max_lin = float(limits["max_lin_speed_mps"])
        self._max_ang = float(limits["max_ang_speed_radps"])
        self._max_joint = float(limits["max_joint_speed"])
        self._table_joints = [(d.uid, d.joint_index) for d in self.env.scene.doors]
        for obj in list(self.env.scene.buttons) + list(self.env.scene.switches):
            self._table_joints.append((obj.uid, obj.joint_index))
        self._robot_joints = list(range(self.p.getNumJoints(self.bodies.robot_uid, physicsClientId=self.cid)))

    # ------------------------------------------------------------------------------------------- set-up

    def _load_oracle(self, relative: str) -> Any:
        import hydra
        from omegaconf import OmegaConf
        path = calvin_env_root().parent / relative
        if not path.is_file():
            raise ConfigError("oracle_tasks_config %s not found (calvin checkout = parent of calvin_env)" % path)
        return hydra.utils.instantiate(OmegaConf.load(str(path)))

    def _check_timebase(self) -> None:
        tb = self._bd.timebase
        repeat = int(self.env.action_repeat)
        mismatches = []
        if not math.isclose(tb.physics_dt_s, self.physics_dt, rel_tol=1e-9):
            mismatches.append("physics_dt_s %r vs engine %r" % (tb.physics_dt_s, self.physics_dt))
        if not math.isclose(tb.control_dt_s, self.control_dt, rel_tol=1e-9):
            mismatches.append("control_dt_s %r vs env %r" % (tb.control_dt_s, self.control_dt))
        if tb.substeps_per_action != repeat:
            mismatches.append("substeps_per_action %r vs action_repeat %r" % (tb.substeps_per_action, repeat))
        if mismatches:
            raise CapabilityError("bd/calvin.yaml timebase disagrees with the env: %s" % "; ".join(mismatches))

    def _world_signature(self) -> List[Tuple[int, str]]:
        p, cid = self.p, self.cid
        return [(p.getBodyUniqueId(i, physicsClientId=cid),
                 p.getBodyInfo(p.getBodyUniqueId(i, physicsClientId=cid), physicsClientId=cid)[1].decode())
                for i in range(p.getNumBodies(physicsClientId=cid))]

    def _rebuild_world(self) -> None:
        """``PlayTableSimEnv.load()`` (resetSimulation, then the robot and the scene loaded again) with the env's
        random state as it was after construction, then the hazard pool re-created parked in its original order,
        so every body keeps its unique id. After ``env.reset``, ``restoreState`` or CALVIN's ``reset_from_storage``
        an episode still depends on the previous ones (probes/determinism.py; presumably Bullet's contact manifolds
        and pair cache, not verified); only a new world makes an episode a function of its reset."""
        if self.env.scene.np_random is not self.env.np_random:
            raise RuntimeError("the CALVIN scene no longer samples from env.np_random")
        self.env.np_random.bit_generator.state = copy.deepcopy(self._rng_state)
        self.env.load()
        if self._world_signature() != self._world:
            raise RuntimeError("rebuilding the CALVIN world changed its bodies: %s -> %s"
                               % (self._world, self._world_signature()))
        for hazard in self._hazards:
            uid = self._new_body(hazard.signature, hazard.park_pos)
            if uid != hazard.uid:
                raise RuntimeError("hazard %s re-created with uid %d instead of %d" % (hazard.name, uid, hazard.uid))
            hazard.dynamic = True
            self._park(hazard)

    def close(self) -> None:
        if self.env is not None and self.env.cid >= 0:
            self.env.close()
            self.env.cid = -1

    # -------------------------------------------------------------------------------- HazardHost: setup

    def describe(self) -> Any:
        return self._bd

    def install_hooks(self, runtime: Any) -> None:
        self._hooks = runtime

    def alloc_hazard(self, env: int, agent: AgentSpec) -> HazardHandle:
        self._check_env(env)
        if agent.shape not in SHAPES:
            raise CapabilityError("CalvinHost builds %s hazards, not %r" % (list(SHAPES), agent.shape))
        half = [float(h) for h in agent.half_extents_m]
        if agent.shape == "sphere" and len(set(half)) != 1:
            raise CapabilityError("a sphere hazard needs equal half extents, got %r" % (half,))
        if agent.shape == "box" and len(half) != 3:
            raise CapabilityError("a box hazard needs 3 half extents, got %r" % (half,))
        signature = (agent.shape, tuple(half), float(agent.mass_kg))
        for hazard in self._hazards:
            if not hazard.allocated and hazard.signature == signature:
                break
        else:
            hazard = self._create(signature)
        self._park(hazard)
        self._allocations += 1
        hazard.key = "%s.%d" % (hazard.name, self._allocations)
        hazard.allocated = True
        return HazardHandle(env=0, key=hazard.key, native=hazard)

    def _create(self, signature: Tuple[Any, ...]) -> _Hazard:
        body = self.config["hazard_body"]
        k = len(self._hazards)
        park = [float(o + k * s) for o, s in zip(body["park_origin_m"], body["park_spacing_m"])]
        hazard = _Hazard("hazard_%d" % k, self._new_body(signature, park), signature, signature[2], park)
        self._hazards.append(hazard)
        self.bodies.body_name[hazard.uid] = hazard.name
        return hazard

    def _new_body(self, signature: Tuple[Any, ...], park: List[float]) -> int:
        p, cid, body = self.p, self.cid, self.config["hazard_body"]
        shape, half, mass = signature
        if shape == "sphere":
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=half[0], physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=half[0], rgbaColor=body["rgba"], physicsClientId=cid)
        else:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half), physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=body["rgba"],
                                      physicsClientId=cid)
        uid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                basePosition=park, useMaximalCoordinates=True, physicsClientId=cid)
        p.changeDynamics(uid, -1, linearDamping=0.0, angularDamping=0.0, lateralFriction=body["lateral_friction"],
                         spinningFriction=body["spinning_friction"], rollingFriction=body["rolling_friction"],
                         restitution=body["restitution"], activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
                         physicsClientId=cid)
        return uid

    def free_hazard(self, h: HazardHandle) -> None:
        hazard = self._hazard(h)
        self._park(hazard)
        hazard.allocated = False

    # ----------------------------------------------------------------------------- HazardHost: actuation

    def _make_static(self, hazard: _Hazard) -> None:
        p, cid = self.p, self.cid
        if hazard.dynamic:
            p.changeDynamics(hazard.uid, -1, mass=0.0, physicsClientId=cid)
            hazard.dynamic = False
        p.setCollisionFilterGroupMask(hazard.uid, -1, 0, 0, physicsClientId=cid)

    def _park(self, hazard: _Hazard) -> None:
        p, cid = self.p, self.cid
        self._make_static(hazard)
        p.changeDynamics(hazard.uid, -1, mass=0.0, physicsClientId=cid)
        p.resetBasePositionAndOrientation(hazard.uid, hazard.park_pos, [0.0, 0.0, 0.0, 1.0], physicsClientId=cid)
        p.resetBaseVelocity(hazard.uid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=cid)
        hazard.parked = True
        hazard.contacts = []
        hazard.touching = set()
        hazard.pos_before = None

    def set_hazard_pose(self, h: HazardHandle, pos: Sequence[float], quat: Sequence[float]) -> None:
        hazard = self._hazard(h)
        p, cid = self.p, self.cid
        self._make_static(hazard)
        p.resetBasePositionAndOrientation(hazard.uid, _vec3("pos", pos), _xyzw(quat), physicsClientId=cid)
        p.resetBaseVelocity(hazard.uid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=cid)

    def set_hazard_visible(self, h: HazardHandle, visible: bool) -> None:
        hazard = self._hazard(h)
        if visible:
            hazard.parked = False
        else:
            self._park(hazard)

    def release_to_physics(self, h: HazardHandle, lin_vel: Sequence[float], ang_vel: Sequence[float]) -> None:
        hazard = self._hazard(h)
        if hazard.parked:
            raise RuntimeError("release_to_physics on parked hazard %r" % hazard.key)
        p, cid, body = self.p, self.cid, self.config["hazard_body"]
        lin, ang = _vec3("lin_vel", lin_vel), _vec3("ang_vel", ang_vel)
        p.changeDynamics(hazard.uid, -1, mass=hazard.mass, linearDamping=0.0, angularDamping=0.0,
                         physicsClientId=cid)
        p.setCollisionFilterGroupMask(hazard.uid, -1, body["collision_group"], body["collision_mask"],
                                      physicsClientId=cid)
        p.resetBaseVelocity(hazard.uid, lin, ang, physicsClientId=cid)
        hazard.dynamic = True

    # ------------------------------------------------------------------------------- HazardHost: state

    def hazard_state(self, h: HazardHandle) -> HazardState:
        hazard = self._hazard(h)
        pos, quat = self.p.getBasePositionAndOrientation(hazard.uid, physicsClientId=self.cid)
        lin, ang = self.p.getBaseVelocity(hazard.uid, physicsClientId=self.cid)
        if not hazard.dynamic:
            lin, ang = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        return HazardState(pos=pos, quat=_wxyz(quat), lin_vel_mps=lin, ang_vel_radps=ang)

    def hazard_contacts(self, h: HazardHandle) -> List[ContactObs]:
        hazard = self._hazard(h)
        out, hazard.contacts = hazard.contacts, []
        return out

    def robot_state(self, env: int) -> RobotState:
        self._check_env(env)
        p, cid, bodies = self.p, self.cid, self.bodies
        robot = bodies.robot_uid
        tcp = p.getLinkState(robot, self.env.robot.tcp_link_id, computeLinkVelocity=1,
                             computeForwardKinematics=1, physicsClientId=cid)
        names, positions, velocities = list(bodies.robot_bodies), [], []
        for label in bodies.robot_bodies:
            pos, vel = self._link_com(*bodies.representative[label])
            positions.append(pos)
            velocities.append(vel)
        held_uid = self._held_uid()
        held = None if held_uid is None else bodies.movable[held_uid]
        if held_uid is not None:
            pos, vel = self._link_com(held_uid, -1)
            names.append(held)
            positions.append(pos)
            velocities.append(vel)
        base_pos, base_quat = p.getBasePositionAndOrientation(robot, physicsClientId=cid)
        joints = self.env.robot.gripper_joint_ids
        opening = sum(p.getJointState(robot, j, physicsClientId=cid)[0] for j in joints)
        return RobotState(eef_pose=Pose(tcp[0], _wxyz(tcp[1])), eef_lin_vel_mps=tcp[6],
                          collision_body_pos=positions, gripper_opening_m=opening,
                          base_pose=Pose(base_pos, _wxyz(base_quat)), held_object=held,
                          collision_body_names=names, collision_body_lin_vel_mps=velocities,
                          held_object_bodies=None if held is None else [held])

    def object_poses(self, env: int) -> Dict[str, Pose]:
        self._check_env(env)
        p, cid = self.p, self.cid
        out = {}
        for obj in self.env.scene.movable_objects:
            pos, quat = p.getBasePositionAndOrientation(obj.uid, physicsClientId=cid)
            out[obj.name] = Pose(pos, _wxyz(quat))
        for obj in self.env.scene.fixed_objects:
            pos, quat = p.getBasePositionAndOrientation(obj.uid, physicsClientId=cid)
            out["%s/%s" % (obj.name, self.bodies.link_name[(obj.uid, -1)])] = Pose(pos, _wxyz(quat))
            for j in range(p.getNumJoints(obj.uid, physicsClientId=cid)):
                state = p.getLinkState(obj.uid, j, computeForwardKinematics=1, physicsClientId=cid)
                out["%s/%s" % (obj.name, self.bodies.link_name[(obj.uid, j)])] = Pose(state[4], _wxyz(state[5]))
        return out

    def time_state(self, env: int) -> TimeState:
        self._check_env(env)
        return TimeState(sim_time_s=self.env.physics_steps * self.physics_dt, physics_dt_s=self.physics_dt,
                         control_dt_s=self.control_dt)

    def task_success(self, env: int) -> bool:
        self._check_env(env)
        if self._task is None:
            raise RuntimeError("task_success before reset_episode")
        done = self.oracle.get_task_info_for_set(self._start_info, self.env.get_info(), {self._task})
        return len(done) > 0

    def hold_action(self, env: int) -> Any:
        self._check_env(env)
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(self.env.robot.gripper_action)])

    # ---------------------------------------------------------------------------------- optional methods

    def min_distance(self, h: HazardHandle) -> float:
        hazard = self._hazard(h)
        others = [self.bodies.robot_uid]
        held = self._held_uid()
        if held is not None:
            others.append(held)
        best = self._search
        for uid in others:
            for c in self.p.getClosestPoints(hazard.uid, uid, self._search, physicsClientId=self.cid):
                best = min(best, float(c[8]))
        return max(0.0, best)

    def support_surfaces(self, env: int) -> List[Surface]:
        self._check_env(env)
        return [Surface(name=s["name"], min_m=list(s["min_m"]), max_m=list(s["max_m"]), source=s["source"])
                for s in self.config["support_surfaces"]]

    def render_check(self, h: HazardHandle) -> bool:
        hazard = self._hazard(h)
        for camera in self.env.cameras:
            view, proj = self._camera_matrices(camera)
            image = self.p.getCameraImage(width=camera.width, height=camera.height, viewMatrix=view,
                                          projectionMatrix=proj, physicsClientId=self.cid)
            mask = np.reshape(np.asarray(image[4]), (camera.height, camera.width))
            if np.any(mask == hazard.uid):
                return True
        return False

    def _camera_matrices(self, camera: Any) -> Tuple[Any, Any]:
        if hasattr(camera, "viewMatrix"):
            return camera.viewMatrix, camera.projectionMatrix
        # GripperCamera.render (calvin_env/camera/gripper_camera.py:27-37 at the pinned commit)
        state = self.p.getLinkState(camera.robot_uid, camera.gripper_cam_link, physicsClientId=self.cid)
        pos, orn = state[:2]
        rot = np.array(self.p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        view = self.p.computeViewMatrix(pos, pos + rot[:, 1], -rot[:, 2])
        proj = self.p.computeProjectionMatrixFOV(fov=camera.fov, aspect=camera.aspect, nearVal=camera.nearval,
                                                 farVal=camera.farval)
        return view, proj

    # ------------------------------------------------------------------------------------ EpisodeDriver

    def reset_episode(self, env: int, task_id: str, layout_id: str, seed: int) -> None:
        """Reset to the layout's dataset frame (``seed`` is recorded by the caller only: CALVIN resets to a stored
        state, nothing is sampled). The world is rebuilt first (:meth:`_rebuild_world`), then
        ``PlayTableSimEnv.reset(robot_obs, scene_obs)``, with the gripper command set to the frame's
        ``robot_obs[14]`` (``Robot.reset`` does not touch ``gripper_action``)."""
        self._check_env(env)
        if task_id not in self.oracle.tasks:
            raise ValueError("task_id %r is not a task of the oracle (%s)" % (task_id, sorted(self.oracle.tasks)))
        layout = self.dataset.layout(layout_id)
        self._rebuild_world()
        if layout.oracle_start_frame != layout.frame:
            self._reset_to(layout.split, layout.oracle_start_frame)
            self._start_info = self.env.get_info()
            self._obs = self._reset_to(layout.split, layout.frame)
        else:
            self._obs = self._reset_to(layout.split, layout.frame)
            self._start_info = self.env.get_info()
        self._task = task_id
        if self._hooks is not None:
            self._hooks.begin_episode(0)

    def _reset_to(self, split: str, frame: int) -> Any:
        robot_obs, scene_obs = self.dataset.state(split, frame)
        self.env.robot.gripper_action = int(robot_obs[-1])
        for obj in self.env.scene.movable_objects:
            self.p.resetBaseVelocity(obj.uid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=self.cid)
        return self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    def begin_native_episode(self, task_id: str, start_info: Any) -> None:
        """For a native evaluation loop that does not reset between subtasks (evaluate_policy.rollout): the
        current state starts an episode of ``task_id`` judged against ``start_info``; fires ``begin_episode``."""
        if task_id not in self.oracle.tasks:
            raise ValueError("task_id %r is not a task of the oracle" % (task_id,))
        self._task = task_id
        self._start_info = start_info
        if self._hooks is not None:
            self._hooks.begin_episode(0)

    def step(self, env: int, action: Any) -> None:
        self._check_env(env)
        self._obs, _, _, _ = self.env.step(action)

    def policy_observation(self, env: int) -> Dict[str, Any]:
        """What the CALVIN evaluation hands the policy, minus the depth maps (the probe harness accepts H x W x 3
        uint8 images and 1-D vectors only): ``rgb_static``, ``rgb_gripper``, ``robot_obs``, ``scene_obs``."""
        self._check_env(env)
        if self._obs is None:
            raise RuntimeError("policy_observation before reset_episode")
        out = dict(self._obs["rgb_obs"])
        out["robot_obs"] = self._obs["robot_obs"]
        out["scene_obs"] = self._obs["scene_obs"]
        return out

    # ---------------------------------------------------------------------------- physics-step listener

    def before_physics_step(self, env: Any) -> None:
        self._check_stability()
        for hazard in self._hazards:
            if hazard.allocated and not hazard.parked:
                hazard.pos_before = np.array(self.p.getBasePositionAndOrientation(hazard.uid,
                                                                                  physicsClientId=self.cid)[0])
            else:
                hazard.pos_before = None
        if self._hooks is not None:
            self._hooks.pre_step(0)

    def after_physics_step(self, env: Any) -> None:
        t1 = self.env.physics_steps * self.physics_dt
        for hazard in self._hazards:
            if hazard.allocated and not hazard.parked:
                self._collect_contacts(hazard, t1 - self.physics_dt, t1)
        self._check_stability()
        if self._hooks is not None:
            self._hooks.post_step(0)

    # -------------------------------------------------------------------------------------- internals

    def _check_env(self, env: Any) -> None:
        if isinstance(env, bool) or not isinstance(env, (int, np.integer)) or int(env) != 0:
            raise ValueError("CALVIN has one environment (env 0), got %r" % (env,))

    def _hazard(self, h: HazardHandle) -> _Hazard:
        hazard = h.native
        if not isinstance(hazard, _Hazard) or hazard not in self._hazards or hazard.key != h.key:
            raise ValueError("unknown hazard handle %r" % (h,))
        if not hazard.allocated:
            raise RuntimeError("hazard %r is not allocated" % hazard.key)
        return hazard

    def _link_com(self, uid: int, link: int) -> Tuple[np.ndarray, np.ndarray]:
        p, cid = self.p, self.cid
        if link == -1:
            pos = p.getBasePositionAndOrientation(uid, physicsClientId=cid)[0]
            vel = p.getBaseVelocity(uid, physicsClientId=cid)[0]
        else:
            state = p.getLinkState(uid, link, computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId=cid)
            pos, vel = state[0], state[6]
        return np.array(pos, dtype=np.float64), np.array(vel, dtype=np.float64)

    def _held_uid(self) -> Optional[int]:
        sides = {}  # type: Dict[int, set]
        robot = self.bodies.robot_uid
        for c in self.p.getContactPoints(bodyA=robot, physicsClientId=self.cid):
            if c[2] in self.bodies.movable and c[3] in self.bodies.finger_links:
                sides.setdefault(c[2], set()).add(self.bodies.finger_links[c[3]])
        held = sorted(uid for uid, s in sides.items() if len(s) == 2)
        return held[0] if held else None

    def _held_object(self) -> Optional[str]:
        uid = self._held_uid()
        return None if uid is None else self.bodies.movable[uid]

    def _partner_class(self, uid: int, held: Optional[int]) -> PartnerClass:
        if uid == self.bodies.robot_uid:
            return PartnerClass.ROBOT_LINK
        if held is not None and uid == held:
            return PartnerClass.HELD_OBJECT
        if any(h.uid == uid for h in self._hazards):
            return PartnerClass.HAZARD
        return PartnerClass.SCENE

    def _collect_contacts(self, hazard: _Hazard, t0: float, t1: float) -> None:
        p, cid = self.p, self.cid
        partners = list(self.bodies.collision_uids)
        partners += [h.uid for h in self._hazards if h is not hazard and h.allocated and not h.parked]
        deepest = {}  # type: Dict[str, Any]
        for uid in partners:
            for c in p.getClosestPoints(hazard.uid, uid, self._touch, physicsClientId=cid):
                name = self._partner_name(c[2], c[4])
                if name not in deepest or c[8] < deepest[name][8]:
                    deepest[name] = c
        if not deepest:
            hazard.touching = set()
            return
        held = self._held_uid()
        pos_now = np.array(p.getBasePositionAndOrientation(hazard.uid, physicsClientId=cid)[0])
        for name in sorted(deepest):
            c = deepest[name]
            normal = np.array(c[7], dtype=np.float64)
            if name in hazard.touching:
                stamp = t1
            else:
                stamp = self._onset(hazard, c, normal, pos_now, t0, t1)
            position = 0.5 * (np.array(c[5]) + np.array(c[6]))
            hazard.contacts.append(ContactObs(partner_name=name, partner_class=self._partner_class(c[2], held),
                                              position=position, normal=normal, time_s=stamp))
        hazard.touching = set(deepest)

    def _partner_name(self, uid: int, link: int) -> str:
        for hazard in self._hazards:
            if hazard.uid == uid:
                return hazard.key
        return self.bodies.partner[(uid, link)]

    def _onset(self, hazard: _Hazard, c: Any, normal: np.ndarray, pos_now: np.ndarray, t0: float,
               t1: float) -> float:
        """Time in [t0, t1] at which the closing motion over the step makes the overlap begin."""
        if hazard.pos_before is None:
            return t0
        depth = self._touch - float(c[8])
        hazard_vel = (pos_now - hazard.pos_before) / (t1 - t0)
        if any(h.uid == c[2] for h in self._hazards):
            partner_vel = np.array(self.p.getBaseVelocity(c[2], physicsClientId=self.cid)[0])
        else:
            partner_vel = self._link_com(c[2], c[4])[1]
        closing = float(np.dot(hazard_vel - partner_vel, -normal))
        if closing <= 0.0:
            return t0
        return max(t0, t1 - depth / closing)

    def _check_stability(self) -> None:
        if self._hooks is None:
            return
        p, cid = self.p, self.cid
        for hazard in self._hazards:
            if hazard.allocated and not hazard.parked:
                pos = p.getBasePositionAndOrientation(hazard.uid, physicsClientId=cid)[0]
                lin, ang = p.getBaseVelocity(hazard.uid, physicsClientId=cid)
                kind = self._unstable(pos, lin, ang)
                if kind is not None:
                    self._hooks.report_instability(0, kind, entity_is_hazard=True, entity_pos=self._finite(pos))
                    return
        for obj in self.env.scene.movable_objects:
            pos = p.getBasePositionAndOrientation(obj.uid, physicsClientId=cid)[0]
            lin, ang = p.getBaseVelocity(obj.uid, physicsClientId=cid)
            kind = self._unstable(pos, lin, ang)
            if kind is not None:
                self._hooks.report_instability(0, kind, entity_is_hazard=False, entity_pos=self._finite(pos))
                return
        for uid, joints in [(self.bodies.robot_uid, self._robot_joints)] + [(u, [j]) for u, j in self._table_joints]:
            for j, state in zip(joints, p.getJointStates(uid, joints, physicsClientId=cid)):
                q, qd = state[0], state[1]
                if not (math.isfinite(q) and math.isfinite(qd)) or abs(qd) > self._max_joint:
                    kind = INSTABILITY_NONFINITE if not (math.isfinite(q) and math.isfinite(qd)) else INSTABILITY_SPEED
                    pos = p.getLinkState(uid, j, physicsClientId=cid)[0]
                    self._hooks.report_instability(0, kind, entity_is_hazard=False, entity_pos=self._finite(pos))
                    return

    def _unstable(self, pos: Sequence[float], lin: Sequence[float], ang: Sequence[float]) -> Optional[str]:
        values = list(pos) + list(lin) + list(ang)
        if not all(math.isfinite(v) for v in values):
            return INSTABILITY_NONFINITE
        if np.linalg.norm(lin) > self._max_lin or np.linalg.norm(ang) > self._max_ang:
            return INSTABILITY_SPEED
        return None

    @staticmethod
    def _finite(pos: Sequence[float]) -> Optional[List[float]]:
        values = [float(v) for v in pos]
        return values if all(math.isfinite(v) for v in values) else None

    # ------------------------------------------------------------------------------- adapter extras

    def hazard_uid(self, h: HazardHandle) -> int:
        return self._hazard(h).uid

    def inject_engine_fault(self, env: int, h: Optional[HazardHandle]) -> None:
        """Conformance C11: give a body a speed of ``fault_injection_speed_mps`` straight up, behind the
        runtime's back (a released hazard if ``h`` is given, made dynamic first; else the first movable object).
        The instability detector reports it at the next physics step."""
        self._check_env(env)
        speed = float(self.config["fault_injection_speed_mps"])
        if h is None:
            uid = self.env.scene.movable_objects[0].uid
        else:
            hazard = self._hazard(h)
            self.p.changeDynamics(hazard.uid, -1, mass=hazard.mass, physicsClientId=self.cid)
            hazard.dynamic = True
            uid = hazard.uid
        self.p.resetBaseVelocity(uid, [0.0, 0.0, speed], [0.0, 0.0, 0.0], physicsClientId=self.cid)
