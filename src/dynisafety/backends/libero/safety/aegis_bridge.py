"""LIBERO bridge: obstacle/EEF ellipsoids → AEGIS CBF-QP filter (Prompt A1/A3a).

Mount (when ``extra.libero.aegis.enabled``):
  LiberoOracleSession → Policy.act(nominal) → [AegisActionFilter] → env.step

Obstacle geometry:
  obstacle_source=sim_gt  — MuJoCo body pose (A1)
  obstacle_source=vision  — GroundingDINO RGB-D ellipsoid (A3c); fallback sim_gt

Default OFF. When disabled, callers must skip this module (zero overhead).
Does not modify disturbance trajectories or the Oracle state machine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.backends.libero.safety.phases import (
    LiberoOraclePhase,
    probe_controller_phase,
)
from dynisafety.config.schema import RunConfig

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_AEGIS_YAML = _REPO_ROOT / "configs" / "safety" / "aegis.yaml"
_ACTIVATE_AFTER_SETTLED = "settled_or_done"
_ACTIVATE_AFTER_ALWAYS = "always"
_VALID_ACTIVATE_AFTER = frozenset({_ACTIVATE_AFTER_SETTLED, _ACTIVATE_AFTER_ALWAYS})
_SOURCE_SIM_GT = "sim_gt"
_SOURCE_VISION = "vision"
_VALID_OBSTACLE_SOURCES = frozenset({_SOURCE_SIM_GT, _SOURCE_VISION})


def aegis_enabled_from_extra(extra_libero: Mapping[str, Any] | None) -> bool:
    """Require explicit ``extra.libero.aegis.enabled`` (default false if absent)."""
    if not extra_libero:
        return False
    block = extra_libero.get("aegis")
    if not isinstance(block, Mapping):
        return False
    return bool(block.get("enabled", False))


def _aegis_block(extra_libero: Mapping[str, Any]) -> Mapping[str, Any]:
    block = extra_libero.get("aegis")
    if not isinstance(block, Mapping):
        return {}
    return block


def _require_float_vec3(block: Mapping[str, Any], key: str) -> np.ndarray:
    raw = block.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        raise ValueError(f"extra.libero.aegis.{key} must be a length-3 list")
    out = np.asarray([float(x) for x in raw], dtype=np.float64)
    return out


def _parse_eef_axes_overrides(block: Mapping[str, Any]) -> tuple[tuple[str, np.ndarray], ...]:
    """Upstream ``if 'orange juice'/'milk'/'alphabet soup' in task_description``."""
    raw = block.get("eef_axes_diag_by_instruction_substr")
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ValueError("extra.libero.aegis.eef_axes_diag_by_instruction_substr must be a mapping")
    out: list[tuple[str, np.ndarray]] = []
    for key, value in raw.items():
        substr = str(key).strip().lower()
        if not substr:
            raise ValueError(
                "extra.libero.aegis.eef_axes_diag_by_instruction_substr keys must be non-empty"
            )
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
            raise ValueError(
                f"extra.libero.aegis.eef_axes_diag_by_instruction_substr[{key!r}] "
                "must be a length-3 list"
            )
        out.append((substr, np.asarray([float(x) for x in value], dtype=np.float64)))
    return tuple(out)


def resolve_eef_axes_diag(
    default: np.ndarray,
    overrides: Sequence[tuple[str, np.ndarray]],
    instruction: str | None,
) -> np.ndarray:
    """Pick EEF semi-axes from instruction substrings (official main_aegis.py)."""
    text = (instruction or "").lower()
    if not text:
        return np.asarray(default, dtype=np.float64)
    for substr, axes in overrides:
        if substr in text:
            return np.asarray(axes, dtype=np.float64)
    return np.asarray(default, dtype=np.float64)


def _aegis_config_path(block: Mapping[str, Any]) -> Path:
    raw = block.get("config_path")
    if raw is None or raw == "":
        return _DEFAULT_AEGIS_YAML
    path = Path(str(raw))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _import_aegis_api() -> dict[str, Any]:
    """Import A0 AEGIS API; raise with clear message if Prompt A0 is missing."""
    try:
        from dynisafety.policies.safety.aegis import (
            AegisActionFilter,
            EllipsoidState,
            aegis_config_with_enabled,
            load_aegis_config,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "LIBERO aegis enabled but dynisafety.policies.safety.aegis is unavailable. "
            "Merge Prompt A0 (policies/safety/aegis) or disable extra.libero.aegis.enabled."
        ) from exc
    return {
        "AegisActionFilter": AegisActionFilter,
        "EllipsoidState": EllipsoidState,
        "aegis_config_with_enabled": aegis_config_with_enabled,
        "load_aegis_config": load_aegis_config,
    }


def _rotation_from_scipy_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.asarray(R.from_quat(q).as_matrix(), dtype=np.float64)


def _rotation_from_mujoco_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    # MuJoCo [w,x,y,z] → SciPy [x,y,z,w]
    return _rotation_from_scipy_xyzw(np.array([q[1], q[2], q[3], q[0]], dtype=np.float64))


def build_eef_ellipsoid(
    obs: Mapping[str, Any],
    *,
    axes_diag: np.ndarray,
    offset_local: np.ndarray,
    ellipsoid_cls: Any,
) -> Any:
    """EEF safety ellipsoid (upstream ball_pos = eef + R @ offset_local)."""
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
    rot = _rotation_from_scipy_xyzw(eef_quat)
    center = eef_pos + rot @ np.asarray(offset_local, dtype=np.float64).reshape(3)
    return ellipsoid_cls(
        position=center,
        axes_diag=np.asarray(axes_diag, dtype=np.float64).reshape(3),
        rotation=rot,
    )


def _obstacle_axes_from_body(
    sim: Any,
    body_id: int,
    *,
    fallback: np.ndarray,
    use_geom_size: bool,
) -> np.ndarray:
    if not use_geom_size:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    from dynisafety.backends._engine.mujoco_contact import (
        collect_geom_ids_for_bodies,
        expand_body_descendants,
    )

    model = sim.model
    data = sim.data
    bodies = expand_body_descendants(model, [int(body_id)])
    gids = sorted(collect_geom_ids_for_bodies(model, bodies))
    collision = [
        gid
        for gid in gids
        if not (
            int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0
        )
    ]
    use = collision or gids
    if not use:
        return np.asarray(fallback, dtype=np.float64).reshape(3)

    mesh_type = 7  # mujoco.mjtGeom.mjGEOM_MESH
    has_mesh = any(int(model.geom_type[gid]) == mesh_type for gid in use)
    if has_mesh:
        axes = _union_aabb_axes_in_body_frame(sim, int(body_id), use)
        if axes is not None:
            return axes

    best: np.ndarray | None = None
    best_vol = -1.0
    for gid in use:
        size = np.asarray(model.geom_size[gid], dtype=np.float64).reshape(-1)
        if size.size >= 3:
            axes = np.maximum(size[:3], 1e-3)
        elif size.size == 1:
            axes = np.full(3, max(float(size[0]), 1e-3), dtype=np.float64)
        else:
            continue
        vol = float(np.prod(axes))
        if vol > best_vol:
            best_vol = vol
            best = axes
    if best is None:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return best


def _union_aabb_axes_in_body_frame(
    sim: Any,
    body_id: int,
    geom_ids: Sequence[int],
) -> np.ndarray | None:
    """Half-extents of the union AABB of *geom_ids*, in the root body frame."""
    model = sim.model
    data = sim.data
    try:
        root_r = np.asarray(data.body_xmat[int(body_id)], dtype=np.float64).reshape(3, 3)
        root_p = np.asarray(data.body_xpos[int(body_id)], dtype=np.float64).reshape(3)
    except Exception:
        return None
    mn = np.full(3, np.inf, dtype=np.float64)
    mx = np.full(3, -np.inf, dtype=np.float64)
    found = False
    for gid in geom_ids:
        try:
            gpos = np.asarray(data.geom_xpos[int(gid)], dtype=np.float64).reshape(3)
            g_r = np.asarray(data.geom_xmat[int(gid)], dtype=np.float64).reshape(3, 3)
            size = np.asarray(model.geom_size[int(gid)], dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if size.size < 3:
            rb = float(model.geom_rbound[int(gid)]) if hasattr(model, "geom_rbound") else 0.0
            half = np.full(3, max(rb, 1e-3), dtype=np.float64)
        else:
            half = np.maximum(size[:3], 0.0)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corner = gpos + g_r @ (half * np.array([sx, sy, sz]))
                    local = root_r.T @ (corner - root_p)
                    mn = np.minimum(mn, local)
                    mx = np.maximum(mx, local)
                    found = True
    if not found:
        return None
    return np.maximum(0.5 * (mx - mn), 1e-3)


def build_obstacle_ellipsoid_sim_gt(
    *,
    env: Any,
    controller: Any | None,
    axes_diag_default: np.ndarray,
    use_geom_size: bool,
    ellipsoid_cls: Any,
) -> tuple[Any | None, str | None, str | None]:
    """Build obstacle ellipsoid from MuJoCo body pose (``obstacle_source=sim_gt``).

    Returns
    -------
    ellipsoid, name, error
        ``error`` is a short diagnostic when the body cannot be resolved; the
        caller must account it separately from a clean "no obstacle" miss.
    """
    name = None
    if controller is not None:
        name = getattr(controller, "active_obstacle_name", None)
    if not name:
        return None, None, None
    try:
        inner = env.env if hasattr(env, "env") else env
        obj = inner.objects_dict[str(name)]
        body_id = inner.sim.model.body_name2id(obj.root_body)
        pos = np.asarray(inner.sim.data.body_xpos[body_id], dtype=np.float64).reshape(3)
        quat = np.asarray(inner.sim.data.body_xquat[body_id], dtype=np.float64).reshape(4)
        rot = _rotation_from_mujoco_wxyz(quat)
        axes = _obstacle_axes_from_body(
            inner.sim,
            body_id,
            fallback=axes_diag_default,
            use_geom_size=use_geom_size,
        )
        return (
            ellipsoid_cls(position=pos, axes_diag=axes, rotation=rot),
            str(name),
            None,
        )
    except Exception as exc:  # noqa: BLE001 — missing body / dict → skip this step
        return None, str(name) if name else None, f"{type(exc).__name__}: {exc}"


def _eef_state_7d_from_obs(obs: Mapping[str, Any]) -> np.ndarray:
    """LIBERO obs → 7D EEF state (pos + axis-angle + gripper) for abs↔delta."""
    from dynisafety.backends.libero.env import quat2axisangle

    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    aa = quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
    g = float(grip[0]) if grip.size else 0.0
    return np.concatenate([pos, aa, [g]])


def _to_delta_ee_7d(action_7d: np.ndarray, eef_7d: np.ndarray) -> np.ndarray:
    out = np.asarray(action_7d, dtype=np.float64).copy()
    out[:6] = out[:6] - np.asarray(eef_7d, dtype=np.float64).reshape(7)[:6]
    return out


def _to_abs_ee_7d(delta_7d: np.ndarray, eef_7d: np.ndarray) -> np.ndarray:
    out = np.asarray(delta_7d, dtype=np.float64).copy()
    out[:6] = np.asarray(eef_7d, dtype=np.float64).reshape(7)[:6] + out[:6]
    return out


# Phases where hazard motion has finished (static residual may remain on L1/L2).
_POST_HAZARD_PHASES = frozenset(
    {
        LiberoOraclePhase.SETTLED_OR_DONE,
        LiberoOraclePhase.WITHDRAWN,
    }
)
_FILTERABLE_ACTION_SPACES = frozenset({"delta_ee_7d", "abs_ee_7d"})


@dataclass
class LiberoAegisSession:
    """Per-episode AEGIS bookkeeping for LIBERO rollout."""

    enabled: bool
    activate_after: str
    obstacle_source: str
    oracle_enabled: bool
    level: int
    settle_steps: int
    suite: str = ""
    filter: Any | None = None
    eef_offset_local: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -0.08], dtype=np.float64)
    )
    eef_axes_diag: np.ndarray = field(
        default_factory=lambda: np.array([0.06, 0.12, 0.11], dtype=np.float64)
    )
    eef_axes_overrides: tuple[tuple[str, np.ndarray], ...] = ()
    instruction: str = ""
    obstacle_axes_diag_default: np.ndarray = field(
        default_factory=lambda: np.array([0.06, 0.06, 0.08], dtype=np.float64)
    )
    use_geom_size_for_obstacle: bool = True
    intervened_steps: int = 0
    qp_fail_count: int = 0
    filter_calls: int = 0
    skipped_no_obstacle: int = 0
    skipped_wrong_dim: int = 0
    skipped_unsupported_space: int = 0
    skipped_ellipsoid_build_error: int = 0
    skipped_withdrawn_cleared: int = 0
    last_phase: str | None = None
    last_obstacle_name: str | None = None
    last_ellipsoid_build_error: str | None = None
    last_effective_obstacle_source: str | None = None
    last_vision_prompt: str | None = None
    last_vision_reason: str | None = None
    vision_failed: bool = False
    vision_calls: int = 0
    vision_cache_hits: int = 0
    vision_fail_count: int = 0
    vision_fallback_count: int = 0
    skipped_vision_failure: int = 0
    vision_config: Any | None = None
    _ellipsoid_cls: Any | None = None
    _vision_estimator: Any | None = None
    _vision_cache: Any | None = None
    _vision_cache_step: int = -1
    _vision_cache_name: str | None = None

    @classmethod
    def create(
        cls,
        *,
        cfg: RunConfig,
        extra_libero: Mapping[str, Any],
        oracle_enabled: bool,
        vision_estimator: Any | None = None,
        instruction: str | None = None,
    ) -> LiberoAegisSession:
        from dynisafety.backends.libero.safety.observation import settle_steps_for_level

        level = int(cfg.level)
        enabled = aegis_enabled_from_extra(extra_libero)
        block = _aegis_block(extra_libero)
        activate_after = str(block.get("activate_after", _ACTIVATE_AFTER_SETTLED)).strip().lower()
        if activate_after not in _VALID_ACTIVATE_AFTER:
            raise ValueError(
                f"extra.libero.aegis.activate_after must be one of "
                f"{sorted(_VALID_ACTIVATE_AFTER)}, got {activate_after!r}"
            )
        obstacle_source = str(block.get("obstacle_source", _SOURCE_SIM_GT)).strip().lower()
        if obstacle_source not in _VALID_OBSTACLE_SOURCES:
            raise ValueError(
                f"extra.libero.aegis.obstacle_source must be one of "
                f"{sorted(_VALID_OBSTACLE_SOURCES)}, got {obstacle_source!r}"
            )

        suite = ""
        raw_suite = getattr(cfg, "suite", None)
        if isinstance(raw_suite, str) and raw_suite.strip():
            suite = raw_suite.strip()
        instr = str(instruction).strip() if instruction else ""

        if not enabled:
            return cls(
                enabled=False,
                activate_after=activate_after,
                obstacle_source=obstacle_source,
                oracle_enabled=bool(oracle_enabled),
                level=level,
                settle_steps=settle_steps_for_level(cfg, level),
                suite=suite,
                instruction=instr,
            )

        api = _import_aegis_api()
        cfg_path = _aegis_config_path(block)
        aegis_cfg = api["aegis_config_with_enabled"](api["load_aegis_config"](cfg_path), True)
        filt = api["AegisActionFilter"](aegis_cfg)

        vision_cfg = None
        estimator = None
        if obstacle_source == _SOURCE_VISION:
            from dynisafety.backends.libero.safety.aegis_vision import (
                make_default_vision_estimator,
                vision_config_from_mapping,
            )

            vision_block = block.get("vision")
            if not isinstance(vision_block, Mapping):
                raise ValueError(
                    "extra.libero.aegis.vision mapping is required when "
                    "obstacle_source='vision' (use configs/backends/libero/aegis/vision.yaml)"
                )
            vision_cfg = vision_config_from_mapping(vision_block)
            estimator = (
                vision_estimator
                if vision_estimator is not None
                else make_default_vision_estimator(vision_cfg)
            )

        eef_overrides = _parse_eef_axes_overrides(block)
        return cls(
            enabled=True,
            activate_after=activate_after,
            obstacle_source=obstacle_source,
            oracle_enabled=bool(oracle_enabled),
            level=level,
            settle_steps=settle_steps_for_level(cfg, level),
            suite=suite,
            filter=filt,
            eef_offset_local=_require_float_vec3(block, "eef_offset_local"),
            eef_axes_diag=resolve_eef_axes_diag(
                _require_float_vec3(block, "eef_axes_diag"),
                eef_overrides,
                instr,
            ),
            eef_axes_overrides=eef_overrides,
            instruction=instr,
            obstacle_axes_diag_default=_require_float_vec3(block, "obstacle_axes_diag_default"),
            use_geom_size_for_obstacle=bool(block.get("use_geom_size_for_obstacle", True)),
            vision_config=vision_cfg,
            _ellipsoid_cls=api["EllipsoidState"],
            _vision_estimator=estimator,
        )

    def reset(self) -> None:
        if self.filter is not None:
            self.filter.reset()
        self.intervened_steps = 0
        self.qp_fail_count = 0
        self.filter_calls = 0
        self.skipped_no_obstacle = 0
        self.skipped_wrong_dim = 0
        self.skipped_unsupported_space = 0
        self.skipped_ellipsoid_build_error = 0
        self.skipped_withdrawn_cleared = 0
        self.last_phase = None
        self.last_obstacle_name = None
        self.last_ellipsoid_build_error = None
        self.last_effective_obstacle_source = None
        self.last_vision_prompt = None
        self.last_vision_reason = None
        self.vision_failed = False
        self.vision_calls = 0
        self.vision_cache_hits = 0
        self.vision_fail_count = 0
        self.vision_fallback_count = 0
        self.skipped_vision_failure = 0
        self._vision_cache = None
        self._vision_cache_step = -1
        self._vision_cache_name = None

    def should_filter_phase(self, phase: LiberoOraclePhase) -> bool:
        """Whether the current phase allows AEGIS filtering."""
        if not self.enabled:
            return False
        if self.activate_after == _ACTIVATE_AFTER_ALWAYS:
            return True
        # settled_or_done: with oracle off → every step (static对照);
        # with oracle on → only after hazard motion finishes
        # (L1/L2 SETTLED_OR_DONE static residual, or L3 WITHDRAWN clearance).
        if not self.oracle_enabled:
            return True
        return phase in _POST_HAZARD_PHASES

    def resolve_phase(
        self,
        *,
        controller: Any | None,
        task_step: int,
        probe_phase: LiberoOraclePhase | None = None,
    ) -> LiberoOraclePhase:
        if probe_phase is not None:
            return probe_phase
        result = probe_controller_phase(
            controller,
            level=self.level,
            task_step=task_step,
            settle_steps=self.settle_steps,
        )
        return result.phase

    def _controller_obstacle_name(self, controller: Any | None) -> str | None:
        if controller is None:
            return None
        name = getattr(controller, "active_obstacle_name", None)
        if name is None or str(name).strip() == "":
            return None
        return str(name)

    def _vision_cache_valid(self, *, task_step: int, name: str | None) -> bool:
        from dynisafety.backends.libero.safety.aegis_vision import VisionRefreshMode

        if self._vision_cache is None or self.vision_config is None:
            return False
        if self._vision_cache_name != name:
            return False
        mode = self.vision_config.refresh
        if mode is VisionRefreshMode.ALWAYS:
            return False
        if mode is VisionRefreshMode.ON_SETTLE:
            return True
        n = self.vision_config.refresh_every_n
        if n is None:
            return False
        return (int(task_step) - int(self._vision_cache_step)) < int(n)

    def _estimate_vision(
        self,
        *,
        obs: Mapping[str, Any],
        env: Any,
        controller: Any | None,
        task_step: int,
        name: str | None,
    ) -> Any:
        from dynisafety.backends.libero.safety.aegis_vision import (
            VisionEstimate,
            VisionEstimateFailure,
            VisionEstimateRequest,
            resolve_vision_text_prompt,
        )

        assert self.vision_config is not None
        assert self._vision_estimator is not None
        if self._vision_cache_valid(task_step=task_step, name=name):
            self.vision_cache_hits += 1
            return self._vision_cache

        prompt = resolve_vision_text_prompt(self.vision_config, obstacle_name=name)
        self.last_vision_prompt = prompt
        if prompt is None:
            estimate = VisionEstimate.failure(VisionEstimateFailure.MISSING_PROMPT, prompt=None)
        else:
            request = VisionEstimateRequest(
                obs=obs,
                env=env,
                controller=controller,
                task_step=int(task_step),
                obstacle_name=name,
                text_prompt=prompt,
                prompt_mode=self.vision_config.text_prompt_mode,
                views=self.vision_config.views,
                suite=self.suite,
            )
            estimate = self._vision_estimator.estimate(request)
        self.vision_calls += 1
        if estimate.failed:
            self.vision_failed = True
            self.vision_fail_count += 1
        self._vision_cache = estimate
        self._vision_cache_step = int(task_step)
        self._vision_cache_name = name
        return estimate

    def _sim_gt_obstacle(
        self, *, env: Any, controller: Any | None
    ) -> tuple[Any | None, str | None, str | None]:
        return build_obstacle_ellipsoid_sim_gt(
            env=env,
            controller=controller,
            axes_diag_default=self.obstacle_axes_diag_default,
            use_geom_size=self.use_geom_size_for_obstacle,
            ellipsoid_cls=self._ellipsoid_cls,
        )

    def _resolve_obstacle(
        self,
        *,
        obs: Mapping[str, Any],
        env: Any,
        controller: Any | None,
        task_step: int,
    ) -> tuple[Any | None, str | None, str | None, bool]:
        """Return ellipsoid, name, build_err, skip_filter."""
        from dynisafety.backends.libero.safety.aegis_vision import VisionFailurePolicy

        name = self._controller_obstacle_name(controller)
        if self.obstacle_source != _SOURCE_VISION:
            obstacle, gt_name, build_err = self._sim_gt_obstacle(env=env, controller=controller)
            if obstacle is not None:
                self.last_effective_obstacle_source = _SOURCE_SIM_GT
            return obstacle, gt_name if gt_name is not None else name, build_err, False

        estimate = self._estimate_vision(
            obs=obs,
            env=env,
            controller=controller,
            task_step=task_step,
            name=name,
        )
        self.last_vision_reason = estimate.reason
        if not estimate.failed and estimate.ellipsoid is not None:
            self.last_effective_obstacle_source = _SOURCE_VISION
            return estimate.ellipsoid, name, None, False

        self.vision_failed = True
        policy = self.vision_config.on_failure if self.vision_config is not None else None
        reason = estimate.reason or "vision_failed"
        if policy is VisionFailurePolicy.RAISE:
            raise RuntimeError(f"AEGIS vision obstacle estimate failed: {reason}")
        if policy is VisionFailurePolicy.SKIP_FILTER:
            self.skipped_vision_failure += 1
            return None, name, None, True

        self.vision_fallback_count += 1
        obstacle, gt_name, build_err = self._sim_gt_obstacle(env=env, controller=controller)
        if obstacle is not None:
            self.last_effective_obstacle_source = _SOURCE_SIM_GT
        return obstacle, gt_name if gt_name is not None else name, build_err, False

    def _apply_filter(
        self,
        arr: np.ndarray,
        *,
        obs: Mapping[str, Any],
        eef: Any,
        obstacle: Any,
        action_space: str,
    ) -> np.ndarray:
        if action_space == "abs_ee_7d":
            eef_7d = _eef_state_7d_from_obs(obs)
            nominal_delta = _to_delta_ee_7d(arr, eef_7d)
            filtered_delta, debug = self.filter.filter(nominal_delta, eef, obstacle)
            filtered = _to_abs_ee_7d(filtered_delta, eef_7d)
        else:
            filtered, debug = self.filter.filter(arr, eef, obstacle)
        self.filter_calls += 1
        if debug.intervened:
            self.intervened_steps += 1
        if debug.qp_failed:
            self.qp_fail_count += 1
        return np.asarray(filtered, dtype=np.float64)

    def maybe_filter_action(
        self,
        action: np.ndarray,
        *,
        obs: Mapping[str, Any],
        env: Any,
        controller: Any | None,
        task_step: int,
        probe_phase: LiberoOraclePhase | None = None,
        action_space: str = "delta_ee_7d",
    ) -> np.ndarray:
        """Filter a stepped 7D action when enabled and phase allows; else return input.

        CBF-QP expects **delta/velocity-like** 7D commands. For ``abs_ee_7d`` the
        bridge converts abs→delta with the current EEF, filters, then delta→abs
        so X-VLA absolute targets are not misread as world velocities.
        """
        if not self.enabled or self.filter is None or self._ellipsoid_cls is None:
            return action

        phase = self.resolve_phase(
            controller=controller, task_step=task_step, probe_phase=probe_phase
        )
        self.last_phase = phase.value
        if not self.should_filter_phase(phase):
            return action

        # L3 hand has left the scene (≈ L0): no residual static obstacle to shield.
        if phase is LiberoOraclePhase.WITHDRAWN:
            self.skipped_withdrawn_cleared += 1
            return action

        arr = np.asarray(action)
        if arr.shape != (7,):
            self.skipped_wrong_dim += 1
            return action

        space = str(action_space)
        if space not in _FILTERABLE_ACTION_SPACES:
            self.skipped_unsupported_space += 1
            return action

        eef = build_eef_ellipsoid(
            obs,
            axes_diag=self.eef_axes_diag,
            offset_local=self.eef_offset_local,
            ellipsoid_cls=self._ellipsoid_cls,
        )
        obstacle, name, build_err, skip_filter = self._resolve_obstacle(
            obs=obs,
            env=env,
            controller=controller,
            task_step=task_step,
        )
        self.last_obstacle_name = name
        if skip_filter:
            return action
        if build_err is not None:
            self.skipped_ellipsoid_build_error += 1
            self.last_ellipsoid_build_error = build_err
            return action
        if obstacle is None:
            self.skipped_no_obstacle += 1
            return action

        return self._apply_filter(arr, obs=obs, eef=eef, obstacle=obstacle, action_space=space)

    def episode_extra(self) -> dict[str, Any]:
        """Fields for ``EpisodeRecord.extra`` (no SCHEMA_VERSION bump)."""
        if not self.enabled:
            source: str | None = None
        elif self.last_effective_obstacle_source is not None:
            source = self.last_effective_obstacle_source
        elif (
            self.obstacle_source == _SOURCE_VISION and self.vision_failed and self.filter_calls == 0
        ):
            source = None
        else:
            source = self.obstacle_source
        out: dict[str, Any] = {
            "aegis_enabled": bool(self.enabled),
            "aegis_intervened_steps": int(self.intervened_steps),
            "aegis_qp_fail_count": int(self.qp_fail_count),
            "aegis_obstacle_source": source,
            "aegis_obstacle_source_requested": self.obstacle_source if self.enabled else None,
            "aegis_activate_after": self.activate_after,
            "aegis_filter_calls": int(self.filter_calls),
            "aegis_skipped_no_obstacle": int(self.skipped_no_obstacle),
            "aegis_skipped_ellipsoid_build_error": int(self.skipped_ellipsoid_build_error),
            "aegis_skipped_unsupported_space": int(self.skipped_unsupported_space),
            "aegis_skipped_withdrawn_cleared": int(self.skipped_withdrawn_cleared),
            "aegis_skipped_vision_failure": int(self.skipped_vision_failure),
            "aegis_last_phase": self.last_phase,
            "aegis_last_obstacle_name": self.last_obstacle_name,
            "aegis_last_ellipsoid_build_error": self.last_ellipsoid_build_error,
            "aegis_vision_failed": bool(self.vision_failed),
            "aegis_vision_calls": int(self.vision_calls),
            "aegis_vision_cache_hits": int(self.vision_cache_hits),
            "aegis_vision_fail_count": int(self.vision_fail_count),
            "aegis_vision_fallback_count": int(self.vision_fallback_count),
            "aegis_vision_last_reason": self.last_vision_reason,
            "aegis_vision_last_prompt": self.last_vision_prompt,
        }
        if self.level == 3 and self.activate_after == _ACTIVATE_AFTER_SETTLED:
            out["aegis_note"] = (
                "L3 + activate_after=settled_or_done: Oracle covers the live-hand "
                "window; after WITHDRAWN the scene is cleared (≈ L0) so AEGIS does "
                "not filter residual geometry. Use activate_after=always (and "
                "typically oracle off) if you need CBF during hand presence."
            )
        return out
