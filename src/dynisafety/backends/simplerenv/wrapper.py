"""Thin Gym wrapper: orchestrate actor / targets / disturbance / contact."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.backends.simplerenv.actor import (
    ActorLifecycle,
    current_obstacle_model,
    hand_rgba,
)
from dynisafety.backends.simplerenv.contact import (
    actor_has_geometric_contact,
    filter_collision_link_names,
    first_hand_contact_kinematics,
    force_equiv_dt_s,
    is_toppled,
    measure_contact_step,
    measure_tilt_deg,
    require_contact_params,
    sapien2_linear_velocity,
    scene_timestep,
)
from dynisafety.backends.simplerenv.disturbance.assemble import build_controller
from dynisafety.backends.simplerenv.env import (
    default_asset_dir,
    models_dir_for_root,
)
from dynisafety.backends.simplerenv.targets import SceneAccess
from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.contact import CONTACT_IMPULSE_EPS, is_force_anomaly
from dynisafety.core.newton_force import (
    L3_FORCE_SCHEME_DYNAMIC_RELEASE,
    L3_RELEASE_MIN_SPEED_MPS,
    is_ghost_overlap,
    l3_closing_speed_mps,
    l3_newton_normal_impulse_ns,
    l3_release_linear_velocity,
    normalize_l3_force_scheme,
    official_l3_contact,
)


class DyniSafetyWrapper:
    """L1–L3 disturbance injection; level 0 must not construct this class.

    Duck-types as a Gymnasium wrapper (``.env`` / ``.unwrapped`` / reset/step)
    without importing ``gymnasium`` at module load (unit tests stay sim-free).
    """

    def __init__(
        self,
        env: Any,
        *,
        level: int,
        run_cfg: RunConfig,
        episode_index: int,
        seed: int,
        task_id: int = 0,
        simplerenv_root: Path,
        workspace: Path,
        resolved_extra: Mapping[str, Any],
        task_extra: Mapping[str, Any],
    ) -> None:
        if level not in (1, 2, 3):
            raise ValueError(
                f"DyniSafetyWrapper rejects level={level}; use bare env for L0 "
                "(factory branch, defect #16)"
            )
        self.env = env
        self._level = int(level)
        self._run_cfg = run_cfg
        self._episode_index = int(episode_index)
        self._seed = int(seed)
        self._task_id = int(task_id)
        self._simplerenv_root = Path(simplerenv_root)
        self._workspace = Path(workspace)
        self._resolved_extra = dict(resolved_extra)
        self._task_extra = dict(task_extra)

        self._lifecycle: ActorLifecycle | None = None
        self._controller: Any | None = None
        self._task_step = 0
        self._model_name = ""
        self._actor_scale = 1.0
        self._actor_type = "obstacle"
        self._hand_scale = 1.0
        self._use_visual_mesh = True
        self._pre_physics_prevents = 0
        self._force_anomaly_any = False
        self._last_step_info: dict[str, Any] = {}
        self._tabletop_xy_bounds_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._force_dt: float | None = None
        self._hand_mass_kg: float | None = None
        self._l3_force_linear_vel: np.ndarray | None = None
        self._l3_release_linear_vel: np.ndarray | None = None
        self._l3_official_force_done = False
        self._l3_extra: dict[str, Any] = {}

        impulse_reduce, fanom, ianom, topple = require_contact_params(run_cfg)
        self._impulse_reduce = impulse_reduce
        self._force_anomaly_n = fanom
        self._impulse_anomaly_ns = ianom
        self._topple_threshold_deg = topple

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    @property
    def inner_env(self) -> Any:
        return self.unwrapped

    def __getattr__(self, name: str) -> Any:
        # Forward ManiSkill2 helpers (get_language_instruction, is_final_subtask, …).
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def scene(self) -> Any:
        return self.inner_env._scene

    @property
    def robot(self) -> Any:
        return self.inner_env.agent.robot

    @property
    def dynisafety_step_info(self) -> Mapping[str, Any]:
        return dict(self._last_step_info)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _get_eef_pos(self) -> np.ndarray:
        return np.asarray(self.inner_env.agent.ee_link.get_pose().p, dtype=np.float64)

    def _get_robot_link_names(self) -> set[str]:
        return {link.name for link in self.robot.get_links()}

    def _get_robot_link_positions(self) -> np.ndarray:
        return np.asarray([link.get_pose().p for link in self.robot.get_links()], dtype=np.float64)

    def _scene_access(self) -> SceneAccess:
        scene_cfg = self._run_cfg.scene
        if scene_cfg is MISSING or scene_cfg.table_height is MISSING:
            raise ValueError("cfg.scene.table_height required (YAML)")
        return SceneAccess(
            get_eef_pos=self._get_eef_pos,
            get_scene=lambda: self.scene,
            get_robot=lambda: self.robot,
            get_inner_env=lambda: self.inner_env,
            get_dynamic_actor=lambda: None if self._lifecycle is None else self._lifecycle.actor,
            table_height=float(scene_cfg.table_height),
            tabletop_xy_bounds_cache=self._tabletop_xy_bounds_cache,
        )

    def _resolve_actor_params(self) -> None:
        disturbance = self._run_cfg.disturbance
        if disturbance is MISSING:
            raise ValueError("cfg.disturbance is MISSING")
        obstacle = disturbance.obstacle
        if obstacle is MISSING:
            raise ValueError("cfg.disturbance.obstacle is MISSING")
        models = obstacle.models
        if models is MISSING or models is None:
            raise ValueError("disturbance.obstacle.models required (YAML)")
        scale = obstacle.scale
        if scale is MISSING:
            raise ValueError("disturbance.obstacle.scale required (YAML)")
        density = self._resolved_extra.get("obstacle_density")
        if density is None:
            raise ValueError("extra.obstacle_density required (YAML)")
        use_visual = self._resolved_extra.get("use_visual_mesh")
        if use_visual is None:
            raise ValueError("extra.use_visual_mesh required (YAML)")
        se_extra = self._resolved_extra.get("simplerenv")
        if not isinstance(se_extra, Mapping) or "hand_color" not in se_extra:
            raise ValueError("extra.simplerenv.hand_color required (YAML)")
        self._hand_color = hand_rgba(se_extra["hand_color"])

        hand = self._run_cfg.hand
        if hand is MISSING:
            raise ValueError("cfg.hand is MISSING")
        half = hand.half_size
        col_off = hand.collision_offset
        if half is MISSING or col_off is MISSING:
            raise ValueError("hand.half_size and hand.collision_offset required")

        if self._level == 3:
            self._actor_type = "hand"
            self._model_name = "hand"
            hs = hand.scale
            if hs is MISSING:
                raise ValueError("hand.scale required for L3")
            self._hand_scale = float(hs)
            mass = hand.mass_kg
            if mass is MISSING:
                raise ValueError("hand.mass_kg required for L3")
            self._hand_mass_kg = float(mass)
        else:
            self._actor_type = "obstacle"
            self._model_name = current_obstacle_model(
                models=list(models),
                fallback_model=str(models[0]),
                episode_index=self._episode_index,
            )
            self._hand_scale = 1.0

        self._actor_scale = float(scale)
        self._use_visual_mesh = bool(use_visual)
        self._density = float(density)
        self._hand_half_size = list(half)
        self._hand_collision_offset = list(col_off)

    def _ensure_lifecycle(self) -> ActorLifecycle:
        if self._lifecycle is not None:
            return self._lifecycle
        self._lifecycle = ActorLifecycle(
            self.scene,
            models_dir=models_dir_for_root(self._simplerenv_root),
            asset_dir=default_asset_dir(),
            hand_half_size=self._hand_half_size,
            hand_collision_offset=self._hand_collision_offset,
            density=self._density,
            hand_color=self._hand_color,
        )
        return self._lifecycle

    def _l3_force_scheme(self) -> str:
        if self._level != 3:
            return "kinematic_first_hit"
        se = self._resolved_extra.get("simplerenv")
        if not isinstance(se, Mapping) or "l3_force_scheme" not in se:
            raise ValueError("extra.simplerenv.l3_force_scheme missing from resolved config")
        return normalize_l3_force_scheme(
            se["l3_force_scheme"],
            field_name="extra.simplerenv.l3_force_scheme",
        )

    def _l3_uses_dynamic_release(self) -> bool:
        return (
            self._level == 3
            and self._l3_force_scheme() == L3_FORCE_SCHEME_DYNAMIC_RELEASE
        )

    def _official_target_names(self) -> set[str]:
        return filter_collision_link_names(
            self._get_robot_link_names(),
            ignore_substrings=self._ignore_substrings(),
        )

    def _control_dt_s(self) -> float:
        runtime = self._run_cfg.runtime
        if runtime is MISSING or runtime.control_freq is MISSING:
            raise ValueError("runtime.control_freq required")
        hz = float(runtime.control_freq)
        if hz <= 0.0:
            raise ValueError(f"runtime.control_freq must be > 0, got {runtime.control_freq!r}")
        return 1.0 / hz

    def _snapshot_l3_inbound_velocity(self, overlap_pos: np.ndarray) -> None:
        ctrl = self._controller
        last_free = None if ctrl is None else getattr(ctrl, "last_free_pos", None)
        path = (
            np.zeros(3, dtype=np.float64)
            if ctrl is None or not hasattr(ctrl, "path_velocity")
            else np.asarray(ctrl.path_velocity(), dtype=np.float64)
        )
        kwargs = {
            "overlap_pos": overlap_pos,
            "last_free_pos": last_free,
            "path_velocity": path,
            "sim_dt": self._control_dt_s(),
        }
        if self._l3_force_linear_vel is None:
            self._l3_force_linear_vel = l3_release_linear_velocity(
                **kwargs, min_speed_mps=0.0
            )
        if self._l3_release_linear_vel is None:
            self._l3_release_linear_vel = l3_release_linear_velocity(
                **kwargs, min_speed_mps=L3_RELEASE_MIN_SPEED_MPS
            )

    def _l3_relative_contact_frame(self) -> tuple[np.ndarray, np.ndarray]:
        inbound = (
            np.zeros(3, dtype=np.float64)
            if self._l3_force_linear_vel is None
            else np.asarray(self._l3_force_linear_vel, dtype=np.float64).reshape(3)
        )
        normal = inbound.copy()
        v_partner = np.zeros(3, dtype=np.float64)
        life = self._lifecycle
        actor = None if life is None else life.actor
        if actor is None:
            return normal, v_partner
        try:
            actor_pos = np.asarray(actor.get_pose().p, dtype=np.float64).reshape(3)
        except Exception:
            return normal, v_partner
        frame = first_hand_contact_kinematics(
            self.scene,
            actor_name=str(actor.name),
            actor_pos=actor_pos,
            target_names=self._official_target_names(),
        )
        if frame is None:
            return normal, v_partner
        if float(np.linalg.norm(frame.normal_from_actor)) > 1e-12:
            normal = np.asarray(frame.normal_from_actor, dtype=np.float64).reshape(3)
        if frame.partner_entity is not None:
            got = sapien2_linear_velocity(frame.partner_entity)
            if got is not None:
                v_partner = got
        return normal, v_partner

    def _maybe_inject_l3_newton(self, geometric: bool, impulse: float) -> float:
        if (
            not geometric
            or not self._l3_uses_dynamic_release()
            or self._lifecycle is None
            or self._lifecycle.is_kinematic
            or self._l3_force_linear_vel is None
            or float(impulse) > CONTACT_IMPULSE_EPS
            or self._hand_mass_kg is None
        ):
            return float(impulse)
        v_hand = np.asarray(self._l3_force_linear_vel, dtype=np.float64).reshape(3)
        normal, v_partner = self._l3_relative_contact_frame()
        closing = l3_closing_speed_mps(v_hand, v_partner, normal)
        injected = l3_newton_normal_impulse_ns(
            float(self._hand_mass_kg), v_hand, v_partner, normal
        )
        self._l3_force_linear_vel = None
        self._l3_extra["l3_force_from_release_momentum"] = True
        self._l3_extra["l3_force_inbound_speed_mps"] = float(np.linalg.norm(v_hand))
        self._l3_extra["l3_partner_speed_mps"] = float(np.linalg.norm(v_partner))
        self._l3_extra["l3_closing_speed_mps"] = float(closing)
        return float(injected)

    def _apply_l3_release(self) -> None:
        life = self._lifecycle
        ctrl = self._controller
        if life is None or not life.is_kinematic:
            return
        if ctrl is not None:
            ctrl.restore_last_free_pose()
        vel = (
            np.zeros(3, dtype=np.float64)
            if self._l3_release_linear_vel is None
            else self._l3_release_linear_vel
        )
        actor = life.release_to_physics(
            model_name=self._model_name,
            scale=self._actor_scale,
            use_visual_mesh=self._use_visual_mesh,
            actor_type=self._actor_type,
            hand_scale=self._hand_scale,
            mass_kg=self._hand_mass_kg,
            linear_velocity=vel,
        )
        if ctrl is not None:
            ctrl.actor = actor
            if hasattr(ctrl, "on_released_to_physics"):
                ctrl.on_released_to_physics()
        self._l3_extra["released_to_physics"] = True
        self._l3_extra["l3_release_speed_mps"] = float(np.linalg.norm(vel))

    def _vanish_l3_hand(self) -> None:
        ctrl = self._controller
        if ctrl is not None and hasattr(ctrl, "vanish"):
            ctrl.vanish()
        life = self._lifecycle
        if life is not None:
            life.destroy()
        self._l3_official_force_done = True

    def _seal_released_l3_hand(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Remove the palm after the official snapshot so it cannot fall.

        SimplerEnv's gym step is 0.2 s. Last-free rollback often misses
        re-contact, leaving a dynamic body that lands on the table. A miss
        still records Newton-from-inbound once, then destroys the actor.
        """
        if not self._l3_uses_dynamic_release() or self._l3_official_force_done:
            return metrics
        life = self._lifecycle
        official = bool(metrics.get("in_contact", False))
        if official:
            self._vanish_l3_hand()
            return metrics
        released = life is not None and not life.is_kinematic
        if not released:
            return metrics
        impulse = self._maybe_inject_l3_newton(True, 0.0)
        dt = float(self._force_dt or self._ensure_force_dt())
        if float(impulse) > CONTACT_IMPULSE_EPS:
            force_n = float(impulse / dt) if dt > 0 else 0.0
            metrics = {
                **metrics,
                "in_contact": True,
                "force_n": force_n,
                "impulse_ns": float(impulse),
                "contact_without_force": False,
            }
        self._vanish_l3_hand()
        return metrics

    def _record_l3_kinematic_overlap(self, impulse: float) -> None:
        dt = float(self._force_dt or self._ensure_force_dt())
        force = float(impulse / dt) if dt > 0 else 0.0
        self._l3_extra["l3_kinematic_overlap_sim_steps"] = int(
            self._l3_extra.get("l3_kinematic_overlap_sim_steps", 0) or 0
        ) + 1
        self._l3_extra["ghost_contact_sim_steps"] = int(
            self._l3_extra.get("ghost_contact_sim_steps", 0) or 0
        ) + 1
        peak_j = float(self._l3_extra.get("l3_kinematic_peak_impulse_ns", 0.0) or 0.0)
        peak_f = float(self._l3_extra.get("l3_kinematic_peak_force_n", 0.0) or 0.0)
        if impulse > peak_j:
            self._l3_extra["l3_kinematic_peak_impulse_ns"] = float(impulse)
        if force > peak_f:
            self._l3_extra["l3_kinematic_peak_force_n"] = float(force)

    def _release_to_physics(self) -> None:
        if self._l3_uses_dynamic_release():
            self._apply_l3_release()
            return
        life = self._lifecycle
        if life is None or not life.is_kinematic:
            return
        actor = life.release_to_physics(
            model_name=self._model_name,
            scale=self._actor_scale,
            use_visual_mesh=self._use_visual_mesh,
            actor_type=self._actor_type,
            hand_scale=self._hand_scale,
        )
        if self._controller is not None:
            self._controller.actor = actor

    def _needs_physics_release(self) -> bool:
        ctrl = self._controller
        if ctrl is None:
            return False
        return bool(getattr(ctrl, "needs_physics_release", False))

    def _ensure_force_dt(self) -> float:
        if self._force_dt is not None:
            return float(self._force_dt)
        runtime = self._run_cfg.runtime
        if runtime is MISSING or runtime.control_freq is MISSING:
            raise ValueError("runtime.control_freq required")
        cf = int(runtime.control_freq)
        self._force_dt = force_equiv_dt_s(
            self._run_cfg,
            scene_timestep_s=scene_timestep(self.scene, control_freq=cf),
            control_freq=cf,
        )
        return float(self._force_dt)

    def _ignore_substrings(self) -> list[str]:
        if self._level != 3:
            return []
        l3 = self._run_cfg.disturbance.l3
        if l3 is MISSING:
            return []
        raw = l3.ignore_collision_link_substrings
        if raw is MISSING or raw is None:
            return []
        return list(raw)

    def _measure_step_contact(self) -> dict[str, Any]:
        life = self._lifecycle
        actor = None if life is None else life.actor
        empty = {
            "in_contact": False,
            "force_n": 0.0,
            "impulse_ns": 0.0,
            "partners": {},
            "force_anomaly": False,
            "contact_without_force": False,
            "tilt_deg": 0.0,
            "toppled": False,
        }
        if actor is None:
            return empty
        dt = self._ensure_force_dt()
        contact = measure_contact_step(
            self.scene,
            actor_name=str(actor.name),
            robot_link_names=self._get_robot_link_names(),
            ignore_link_substrings=self._ignore_substrings(),
            force_equiv_dt=float(dt),
            impulse_reduce=self._impulse_reduce,
            force_anomaly_threshold_n=self._force_anomaly_n,
            impulse_anomaly_threshold_ns=self._impulse_anomaly_ns,
        )
        geometric = bool(contact.in_contact)
        impulse = float(contact.impulse_ns)
        if self._l3_uses_dynamic_release():
            released = life is not None and not life.is_kinematic
            impulse = self._maybe_inject_l3_newton(geometric, impulse)
            official = official_l3_contact(
                geometric=geometric,
                impulse=impulse,
                force_scheme=self._l3_force_scheme(),
                released_to_physics=released,
                require_impulse=True,
                impulse_eps=CONTACT_IMPULSE_EPS,
            )
            if is_ghost_overlap(geometric=geometric, official=official):
                self._record_l3_kinematic_overlap(impulse)
            force_n = float(impulse / dt) if official and dt > 0 else 0.0
            in_contact = official
            partners = dict(contact.partners) if official else {}
            without_force = bool(official and impulse <= CONTACT_IMPULSE_EPS)
        else:
            official = geometric
            force_n = float(contact.force_n)
            in_contact = geometric
            partners = dict(contact.partners)
            without_force = bool(contact.contact_without_force)
            impulse = float(contact.impulse_ns)

        tilt = 0.0
        toppled = False
        if self._level in (1, 2) and self._controller is not None:
            try:
                ref = np.asarray(getattr(self._controller, "_arm_quat"), dtype=np.float64)
                cur = np.asarray(actor.get_pose().q, dtype=np.float64)
                tilt = measure_tilt_deg(ref_quat=ref, current_quat=cur)
                toppled = is_toppled(tilt, threshold_deg=self._topple_threshold_deg)
            except Exception:
                tilt = 0.0
                toppled = False
        force_anomaly = is_force_anomaly(
            force_n=force_n,
            impulse_ns=impulse if official else 0.0,
            force_threshold_n=self._force_anomaly_n,
            impulse_threshold_ns=self._impulse_anomaly_ns,
        )
        if force_anomaly:
            self._force_anomaly_any = True
        return {
            "in_contact": in_contact,
            "force_n": force_n,
            "impulse_ns": impulse if official else 0.0,
            "partners": partners,
            "force_anomaly": force_anomaly,
            "contact_without_force": without_force,
            "tilt_deg": tilt,
            "toppled": toppled,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        self._task_step = 0
        self._pre_physics_prevents = 0
        self._force_anomaly_any = False
        self._last_step_info = {}
        self._force_dt = None
        self._hand_mass_kg = None
        self._l3_force_linear_vel = None
        self._l3_release_linear_vel = None
        self._l3_official_force_done = False
        self._l3_extra = {}
        if self._level == 3:
            self._l3_extra["l3_force_scheme"] = self._l3_force_scheme()
            self._l3_extra["released_to_physics"] = False
            self._l3_extra["l3_force_from_release_momentum"] = False
            self._l3_extra["ghost_contact_sim_steps"] = 0
            self._l3_extra["l3_kinematic_overlap_sim_steps"] = 0
            self._l3_extra["l3_kinematic_peak_impulse_ns"] = 0.0
            self._l3_extra["l3_kinematic_peak_force_n"] = 0.0

        opts = options or {}
        obj_init = opts.get("obj_init_options") or {}
        episode_id = obj_init.get("episode_id")
        if episode_id is not None:
            try:
                self._episode_index = int(episode_id)
            except (TypeError, ValueError):
                pass

        self._resolve_actor_params()
        life = self._ensure_lifecycle()
        life.create(
            model_name=self._model_name,
            scale=self._actor_scale,
            kinematic=True,
            use_visual_mesh=self._use_visual_mesh,
            actor_type=self._actor_type,
            hand_scale=self._hand_scale,
        )

        access = self._scene_access()
        # Persist cache mutations from geometry helpers.
        self._tabletop_xy_bounds_cache = access.tabletop_xy_bounds_cache

        episode_seed = self._seed + self._episode_index
        self._controller = build_controller(
            self._level,
            life.actor,
            self._run_cfg,
            access=access,
            reset_options=opts,
            get_eef_pos_fn=self._get_eef_pos,
            seed=episode_seed,
            resolved_extra=self._resolved_extra,
            episode_index=int(self._episode_index),
            task_id=self._task_id,
            mode_seed=int(self._seed),
        )
        self._tabletop_xy_bounds_cache = access.tabletop_xy_bounds_cache
        return obs, info

    def step(self, action):
        pre_step_collided = False
        ctrl = self._controller
        if (
            ctrl is not None
            and self._level == 3
            and hasattr(ctrl, "prevent_commanded_collision")
            and bool(getattr(ctrl, "_enable_pre_physics_collision_guard", False))
        ):
            pre_step_collided = bool(
                ctrl.prevent_commanded_collision(
                    self._get_robot_link_positions(),
                    self._task_step,
                )
            )
            if pre_step_collided:
                self._pre_physics_prevents += 1

        if ctrl is not None and not pre_step_collided:
            ctrl.update_before_step(self._task_step)

        life = self._lifecycle
        if life is not None and life.is_kinematic and self._needs_physics_release():
            if self._l3_uses_dynamic_release() and self._l3_force_linear_vel is None:
                try:
                    overlap = np.asarray(life.actor.get_pose().p, dtype=np.float64)
                    self._snapshot_l3_inbound_velocity(overlap)
                except Exception:
                    pass
            self._release_to_physics()

        obs, reward, done, truncated, info = self.env.step(action)

        if ctrl is not None and not self._l3_uses_dynamic_release():
            names = self._get_robot_link_names()
            positions = self._get_robot_link_positions()
            if not ctrl.blocked:
                ctrl.check_robot_collision(self.scene, names, positions)

        if (
            self._l3_uses_dynamic_release()
            and not self._l3_official_force_done
            and life is not None
            and life.actor is not None
            and life.is_kinematic
        ):
            names = self._official_target_names()
            geometric = actor_has_geometric_contact(
                self.scene,
                actor_name=str(life.actor.name),
                robot_link_names=names,
            )
            if geometric:
                try:
                    overlap = np.asarray(life.actor.get_pose().p, dtype=np.float64)
                    self._snapshot_l3_inbound_velocity(overlap)
                except Exception:
                    pass
                if ctrl is not None:
                    ctrl.request_dynamic_release()
            elif ctrl is not None:
                ctrl.snapshot_last_free()

        metrics = self._measure_step_contact()
        if life is not None and life.is_kinematic and self._needs_physics_release():
            self._release_to_physics()
        metrics = self._seal_released_l3_hand(metrics)

        onset = None
        if ctrl is not None:
            onset = getattr(ctrl, "disturbance_onset_step", None)

        self._last_step_info = {
            **metrics,
            "pre_physics_prevents": self._pre_physics_prevents,
            "force_anomaly_any": self._force_anomaly_any,
            "disturbance_onset_step": onset,
            "level": self._level,
            "force_equiv_dt_s": self._force_dt,
            **self._l3_extra,
        }
        info = dict(info) if info is not None else {}
        info["dynisafety"] = dict(self._last_step_info)
        self._task_step += 1
        return obs, reward, done, truncated, info
