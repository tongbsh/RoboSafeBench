"""Thin Gym wrapper: orchestrate actor / targets / disturbance / contact."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.backends.simplerenv.actor import (
    ActorLifecycle,
    current_obstacle_model,
)
from dynisafety.backends.simplerenv.contact import (
    force_equiv_dt_s,
    is_toppled,
    measure_contact_step,
    measure_tilt_deg,
    require_contact_params,
    scene_timestep,
)
from dynisafety.backends.simplerenv.disturbance.assemble import build_controller
from dynisafety.backends.simplerenv.env import (
    default_asset_dir,
    models_dir_for_root,
)
from dynisafety.backends.simplerenv.targets import SceneAccess
from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.contact import is_force_anomaly


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
        )
        return self._lifecycle

    def _release_to_physics(self) -> None:
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
        if actor is None:
            return {
                "in_contact": False,
                "force_n": 0.0,
                "impulse_ns": 0.0,
                "partners": {},
                "force_anomaly": False,
                "contact_without_force": False,
                "tilt_deg": 0.0,
                "toppled": False,
            }
        if self._force_dt is None:
            runtime = self._run_cfg.runtime
            if runtime is MISSING or runtime.control_freq is MISSING:
                raise ValueError("runtime.control_freq required")
            cf = int(runtime.control_freq)
            self._force_dt = force_equiv_dt_s(
                self._run_cfg,
                scene_timestep_s=scene_timestep(self.scene, control_freq=cf),
                control_freq=cf,
            )
        contact = measure_contact_step(
            self.scene,
            actor_name=str(actor.name),
            robot_link_names=self._get_robot_link_names(),
            ignore_link_substrings=self._ignore_substrings(),
            force_equiv_dt=float(self._force_dt),
            impulse_reduce=self._impulse_reduce,
            force_anomaly_threshold_n=self._force_anomaly_n,
            impulse_anomaly_threshold_ns=self._impulse_anomaly_ns,
        )
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
            force_n=contact.force_n,
            impulse_ns=contact.impulse_ns,
            force_threshold_n=self._force_anomaly_n,
            impulse_threshold_ns=self._impulse_anomaly_ns,
        )
        if force_anomaly:
            self._force_anomaly_any = True
        return {
            "in_contact": contact.in_contact,
            "force_n": contact.force_n,
            "impulse_ns": contact.impulse_ns,
            "partners": dict(contact.partners),
            "force_anomaly": force_anomaly,
            "contact_without_force": bool(contact.contact_without_force),
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
            self._release_to_physics()

        obs, reward, done, truncated, info = self.env.step(action)

        if ctrl is not None:
            names = self._get_robot_link_names()
            positions = self._get_robot_link_positions()
            if not ctrl.blocked:
                ctrl.check_robot_collision(self.scene, names, positions)

        metrics = self._measure_step_contact()
        if life is not None and life.is_kinematic and self._needs_physics_release():
            self._release_to_physics()

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
        }
        info = dict(info) if info is not None else {}
        info["dynisafety"] = dict(self._last_step_info)
        self._task_step += 1
        return obs, reward, done, truncated, info
