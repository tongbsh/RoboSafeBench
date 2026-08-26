"""X-VLA HTTP adapter — shared by ``xvla`` and ``xvla_official`` (extra.variant).

Wire format matches the old DyniSafety clients and upstream X-VLA evals:

* LIBERO (``extra.variant=default``):
  ``old_cache/dynisafety_libero/policies/xvla_policy.py`` /
  ``external/model_repos/X-VLA/evaluation/libero/libero_client.py``
* RoboTwin (``extra.variant=official`` + left/right cams):
  ``policies/_robotwin_impl/xvla_policy.py`` image0/1/2 contract
* SimplerEnv Bridge (``extra.variant=official`` + primary only):
  ``old_cache/.../xvla_official_inference.py`` image0-only + 20-D proprio memory

The FastAPI ``/act`` handler calls ``json_numpy.loads`` on image/proprio fields,
so those values **must** be json_numpy-encoded strings — not nested Python lists
from ``as_jsonable``.
"""

from __future__ import annotations

import base64
import json
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from dynisafety.policies.action_space import effective_action_space
from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    connect_transport,
    require_extra,
    require_image,
    require_state,
)
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)
from dynisafety.policies.transports import Transport

_JSON_NUMPY_DUMPS: Callable[[np.ndarray], str] | None = None


def _fallback_json_numpy_dumps(arr: np.ndarray) -> str:
    return json.dumps(
        {
            "__numpy__": base64.b64encode(arr.tobytes()).decode("ascii"),
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
        }
    )


def json_numpy_dumps(array: np.ndarray) -> str:
    """Encode an ndarray in the json_numpy wire format expected by X-VLA ``/act``.

    Prefer the real ``json_numpy`` package when present (LIBERO / SimplerEnv eval
    envs); otherwise emit the same schema used by the RoboTwin client so
    ``internvla_a1`` (no json_numpy install) still works.
    """
    global _JSON_NUMPY_DUMPS
    arr = np.asarray(array)
    fn = _JSON_NUMPY_DUMPS
    if fn is None:
        try:
            import json_numpy  # noqa: PLC0415

            fn = json_numpy.dumps
        except ImportError:
            fn = _fallback_json_numpy_dumps
        _JSON_NUMPY_DUMPS = fn
    return fn(arr)


def _axangle_to_rot6d(axangle: np.ndarray) -> np.ndarray:
    """Axis-angle → 6D rotation (Zhou et al.): first two *columns* concatenated.

    Must NOT use ``mat[:, :2].reshape(-1)`` (C-order interleaves rows and
    scrambles the proprio rotation fed to X-VLA).
    """
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    mat = Rotation.from_rotvec(np.asarray(axangle, dtype=np.float64)).as_matrix()
    return np.concatenate([mat[:, 0], mat[:, 1]]).astype(np.float32)


def _obs_state_to_proprio_20d(state: np.ndarray) -> np.ndarray:
    pos = state[:3]
    axangle = state[3:6]
    rot6d = _axangle_to_rot6d(axangle)
    proprio_10d = np.concatenate([pos, rot6d, [0.0]]).astype(np.float32)
    return np.concatenate([proprio_10d, np.zeros(10, dtype=np.float32)])


def _rot6d_to_axangle(rot6d: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    mat = np.stack([b1, b2, b3], axis=1)
    return Rotation.from_matrix(mat).as_rotvec().astype(np.float32)


def _binarize_gripper(g: float, *, threshold: float = 0.5) -> float:
    """Match libero_client / old xvla_policy: raw grip > threshold → +1 else -1."""
    return 1.0 if float(g) > float(threshold) else -1.0


def _binarize_gripper_official(g: float, *, threshold: float = 0.5) -> float:
    """Match WidowX / RoboTwin official clients: raw grip < threshold → +1 else -1."""
    return 1.0 if float(g) < float(threshold) else -1.0


# Official WidowX Bridge proprio: pos(3) + fixed identity-ish rot6d/grip tail(7) + zeros(10).
_OFFICIAL_PROPRIO_TAIL = np.array(
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
)
_EULER_BIAS = np.array([0.0, np.pi / 2.0, 0.0], dtype=np.float64)


def _official_proprio_20_from_eef(eef: np.ndarray) -> np.ndarray:
    """Bridge client init proprio from EEF xyz (orientation ignored on reset)."""
    pos = np.asarray(eef[:3], dtype=np.float32)
    proprio_10 = np.concatenate([pos, _OFFICIAL_PROPRIO_TAIL])
    return np.concatenate([proprio_10, np.zeros(10, dtype=np.float32)])


def _quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """Match RoboTwin ``_quat_to_rotate6d`` (matrix[:, :2].reshape(6))."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return Rotation.from_quat(quat).as_matrix()[:, :2].reshape(6)


def _rot6d_to_quat_xyzw_official(v6: np.ndarray) -> np.ndarray:
    """Match RoboTwin ``_rotate6d_to_quat`` (interleaved column packing)."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    v6 = np.asarray(v6, dtype=np.float64).reshape(6)
    a1 = v6[0:5:2]
    a2 = v6[1:6:2]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    mat = np.stack([b1, b2, b3], axis=-1)
    quat = Rotation.from_matrix(mat).as_quat()
    return quat / (np.linalg.norm(quat) + 1e-8)


def _bool_gripper_to_official(gripper_value: float) -> float:
    """RoboTwin gripper [0,1] → {+1 open, -1 close}."""
    return 1.0 - float(gripper_value) * 2.0


def _pred_gripper_to_ee(gripper_pred: float, threshold: float) -> float:
    return 1.0 - 2.0 * float(float(gripper_pred) > float(threshold))


def _official_robotwin_row_to_ee16(
    row20: np.ndarray, *, gripper_threshold: float
) -> np.ndarray:
    """Absolute 20-D X-VLA row → dual-arm EE 16-D for ``take_action(ee)``."""
    row = np.asarray(row20, dtype=np.float64).reshape(-1)
    if row.shape[0] < 20:
        raise ValueError(f"X-VLA RoboTwin action row too short: {row.shape}")
    left_xyz = row[0:3]
    left_quat = _rot6d_to_quat_xyzw_official(row[3:9])
    left_grip = np.array([_pred_gripper_to_ee(row[9], gripper_threshold)], dtype=np.float64)
    right_xyz = row[10:13]
    right_quat = _rot6d_to_quat_xyzw_official(row[13:19])
    right_grip = np.array(
        [_pred_gripper_to_ee(row[19], gripper_threshold)], dtype=np.float64
    )
    return np.concatenate(
        [left_xyz, left_quat, left_grip, right_xyz, right_quat, right_grip],
        axis=0,
    ).astype(np.float32)


def _rot6d_to_euler_xyz_official(v6: np.ndarray) -> np.ndarray:
    """Match archived Bridge ``rotate6d_to_euler_xyz`` (interleaved column packing)."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    v6 = np.asarray(v6, dtype=np.float64).reshape(6)
    a1 = v6[0:5:2]
    a2 = v6[1:6:2]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    mat = np.stack([b1, b2, b3], axis=-1)
    return Rotation.from_matrix(mat).as_euler("xyz")


def _bridge_delta_ee_7d(
    row10: np.ndarray,
    eef: np.ndarray | None,
    *,
    gripper_threshold: float,
) -> np.ndarray:
    """Absolute 10-D X-VLA row → delta EE 7-D (archived ``simpler_delta``)."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    row = np.asarray(row10, dtype=np.float64).reshape(-1)
    if row.shape[0] < 10:
        raise ValueError(f"X-VLA Bridge action row too short: {row.shape}")
    pred_p = row[:3]
    rot6 = row[3:9]
    grip = _binarize_gripper_official(float(row[9]), threshold=gripper_threshold)

    target_rot = Rotation.from_euler("xyz", _rot6d_to_euler_xyz_official(rot6) + _EULER_BIAS)
    if eef is None or len(eef) < 3:
        delta_p = pred_p
        delta_rv = target_rot.as_rotvec()
    else:
        eef_arr = np.asarray(eef, dtype=np.float64).reshape(-1)
        delta_p = pred_p - eef_arr[:3]
        if eef_arr.shape[0] >= 7:
            # Prefer wxyz (ManiSkill) → xyzw for scipy.
            quat_xyzw = np.roll(eef_arr[3:7], -1)
            r_cur = Rotation.from_quat(quat_xyzw)
            delta_rv = (target_rot * r_cur.inv()).as_rotvec()
        else:
            delta_rv = target_rot.as_rotvec()
    return np.concatenate([delta_p, delta_rv, [grip]]).astype(np.float32)


class XVLAAdapter:
    """HTTP X-VLA client.

    ``extra.variant``: ``default`` | ``official`` (S2 D7=A).

    LIBERO paper evals should set ``action_space: abs_ee_7d`` (no
    ``warmup_steps``). Env settle is ``num_steps_wait`` in the backend;
    ``LiberoBackend._apply_action_chunk`` sets ``use_delta=False`` from
    ``abs_ee_7d`` (not a hard-coded ``task_step==0`` flip).
    Optional ``extra.warmup_steps`` / ``post_warmup_action_space`` only apply to
    *policy* steps after wait — mistaking wait length for warmup applies absolute
    targets as deltas.
    """

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport
        variant = require_extra(cfg, "variant")
        if variant not in ("default", "official"):
            raise ValueError(
                f"extra.variant must be 'default' or 'official', got {variant!r}"
            )
        self._variant = str(variant)
        self._domain_id = require_extra(cfg, "domain_id")
        if isinstance(self._domain_id, bool) or not isinstance(self._domain_id, int):
            raise ValueError(f"extra.domain_id must be int, got {self._domain_id!r}")
        self._steps = require_extra(cfg, "denoising_steps")
        if isinstance(self._steps, bool) or not isinstance(self._steps, int):
            raise ValueError(f"extra.denoising_steps must be int, got {self._steps!r}")
        # Optional: official paper clients set this; libero default path omits it → 0.5.
        grip_th = cfg.extra.get("gripper_threshold", 0.5)
        if isinstance(grip_th, bool) or not isinstance(grip_th, (int, float)):
            raise ValueError(
                f"extra.gripper_threshold must be a number, got {grip_th!r}"
            )
        self._gripper_threshold = float(grip_th)
        by_task = cfg.extra.get("gripper_threshold_by_task")
        if by_task is None:
            self._gripper_threshold_by_task: dict[str, float] = {}
        elif not isinstance(by_task, dict):
            raise ValueError(
                f"extra.gripper_threshold_by_task must be a mapping, got {by_task!r}"
            )
        else:
            self._gripper_threshold_by_task = {
                str(k): float(v) for k, v in by_task.items()
            }
        self._proprio: np.ndarray | None = None
        # Bridge (simpler_delta): abs action plan drained one step per act(),
        # matching archived XVLAOfficialBridgeInference._action_plan.
        self._bridge_plan: deque[np.ndarray] = deque()
        # RoboTwin official: recurrent commanded EE (xyz+quat) for next proprio.
        self._left_endpose_override: np.ndarray | None = None
        self._right_endpose_override: np.ndarray | None = None
        self._warmup_steps = cfg.extra.get("warmup_steps")
        self._post_warmup_space = cfg.extra.get("post_warmup_action_space")
        self._bridge_task_name: str | None = None

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> XVLAAdapter:
        if cfg.transport != "http":
            raise ValueError("XVLAAdapter requires transport='http'")
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        del ctx
        self._proprio = None
        self._bridge_plan.clear()
        self._left_endpose_override = None
        self._right_endpose_override = None
        self._bridge_task_name = None
        self._transport.reset({})

    def needs_visual_observation(self) -> bool:
        """Whether the next ``act`` must include camera images.

        Bridge drains a local plan with current EEF only; images are required
        solely on the HTTP fill. Other variants query once per empty chunk.
        """
        if self._variant == "official" and self._bridge_plan:
            return False
        return True

    def _bridge_gripper_threshold(self, obs: Observation) -> float:
        """Prefer upstream per-task WidowX thresholds when task_name is known."""
        name = None
        if obs.extra and obs.extra.get("task_name") is not None:
            name = str(obs.extra["task_name"])
        elif self._bridge_task_name:
            name = self._bridge_task_name
        if name and name in self._gripper_threshold_by_task:
            return float(self._gripper_threshold_by_task[name])
        return float(self._gripper_threshold)

    def act(self, obs: Observation) -> ActionChunk:
        if self._variant == "official":
            return self._act_official(obs)
        return self._act_default(obs)

    def _space_for_step(self, step: int) -> str:
        warmup = self._warmup_steps
        if warmup is not None and (
            isinstance(warmup, bool) or not isinstance(warmup, int)
        ):
            raise ValueError(f"extra.warmup_steps must be int|null, got {warmup!r}")
        post = self._post_warmup_space
        if post is not None and not isinstance(post, str):
            raise ValueError(
                f"extra.post_warmup_action_space must be str|null, got {post!r}"
            )
        return effective_action_space(
            configured_space=self._cfg.action_space,
            step=step,
            warmup_steps=warmup if isinstance(warmup, int) or warmup is None else None,
            post_warmup_space=post if isinstance(post, str) or post is None else None,
        )

    def _act_default(self, obs: Observation) -> ActionChunk:
        # Match old_cache/.../xvla_source_client.py (controller proprio) when
        # backend supplies extra['ee_proprio_20d']; else fall back to obs state.
        if self._proprio is None:
            ee20 = obs.extra.get("ee_proprio_20d") if obs.extra else None
            if ee20 is not None:
                self._proprio = np.asarray(ee20, dtype=np.float32).reshape(20)
            else:
                self._proprio = _obs_state_to_proprio_20d(
                    require_state(obs).astype(np.float32)
                )
        primary = np.asarray(require_image(obs, "primary"), dtype=np.uint8)
        wrist = np.asarray(require_image(obs, "wrist"), dtype=np.uint8)
        # Backend flips both cameras; X-VLA libero client only flips agentview.
        wrist = wrist[::-1, ::-1].copy()
        payload: dict[str, Any] = {
            "image0": json_numpy_dumps(primary),
            "image1": json_numpy_dumps(wrist),
            "proprio": json_numpy_dumps(self._proprio),
            "language_instruction": obs.instruction,
            "domain_id": self._domain_id,
            "steps": self._steps,
        }
        result = self._transport.infer(payload)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"X-VLA server error: {result['error']}")
        raw = np.asarray(result.get("action", result.get("actions")), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        # Recurrent proprio: only overwrite pos+rot6d (dims 0:9), keep grip/zeros.
        if self._proprio is None or self._proprio.shape[0] < 9:
            raise RuntimeError("X-VLA proprio state missing after infer")
        self._proprio[:9] = raw[-1, :9].copy()
        actions = np.stack([self._row_to_7d(row) for row in raw], axis=0)
        space = self._space_for_step(obs.step)
        return actions_to_chunk(actions, space=space)

    def _act_official(self, obs: Observation) -> ActionChunk:
        # RoboTwin: image0/1/2. Bridge WidowX: image0 only (archived client).
        # Drain steps may omit cameras; HTTP fill still requires primary.
        if "left" in obs.images and "right" in obs.images:
            images = [
                np.asarray(require_image(obs, "primary"), dtype=np.uint8),
                np.asarray(require_image(obs, "left"), dtype=np.uint8),
                np.asarray(require_image(obs, "right"), dtype=np.uint8),
            ]
        elif "wrist" in obs.images:
            images = [
                np.asarray(require_image(obs, "primary"), dtype=np.uint8),
                np.asarray(require_image(obs, "wrist"), dtype=np.uint8),
            ]
        else:
            primary = None
            if "primary" in obs.images:
                primary = np.asarray(require_image(obs, "primary"), dtype=np.uint8)
            return self._act_official_bridge(obs, primary)

        # Dual-arm RoboTwin official protocol (abs EE 16-D).
        if self._cfg.action_space == "abs_ee_16d":
            return self._act_official_robotwin(obs, images)

        state = np.asarray(require_state(obs), dtype=np.float32).reshape(-1)
        proprio = state
        payload: dict[str, Any] = {
            "domain_id": self._domain_id,
            "proprio": json_numpy_dumps(proprio),
            "language_instruction": obs.instruction,
            "steps": self._steps,
        }
        for i, img in enumerate(images):
            payload[f"image{i}"] = json_numpy_dumps(img)

        result = self._transport.infer(payload)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"X-VLA server error: {result['error']}")
        raw = np.asarray(result.get("action", result.get("actions")), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        space = self._space_for_step(obs.step)
        if raw.shape[1] == 7:
            return actions_to_chunk(raw, space=space)
        if space in ("delta_ee_7d", "abs_ee_7d") and raw.shape[1] >= 10:
            actions = np.stack(
                [self._row_to_7d_official(row[:10]) for row in raw], axis=0
            )
            return actions_to_chunk(actions, space=space)
        return actions_to_chunk(raw, space=space)

    def _act_official_robotwin(
        self, obs: Observation, images: list[np.ndarray]
    ) -> ActionChunk:
        """Match ``_robotwin_impl/xvla_policy.py`` / official RoboTwin-2.0 client."""
        endpose = (obs.extra or {}).get("endpose")
        if not isinstance(endpose, dict):
            raise KeyError(
                "official X-VLA RoboTwin path needs Observation.extra['endpose'] "
                "(left/right_endpose + grippers)"
            )
        left_ee = (
            self._left_endpose_override
            if self._left_endpose_override is not None
            else np.asarray(endpose["left_endpose"], dtype=np.float32)
        )
        right_ee = (
            self._right_endpose_override
            if self._right_endpose_override is not None
            else np.asarray(endpose["right_endpose"], dtype=np.float32)
        )
        left_grip = _bool_gripper_to_official(float(endpose["left_gripper"]))
        right_grip = _bool_gripper_to_official(float(endpose["right_gripper"]))
        abs_eef = np.concatenate(
            [
                left_ee[:3],
                _quat_xyzw_to_rot6d(left_ee[3:7]),
                np.array([left_grip], dtype=np.float64),
                right_ee[:3],
                _quat_xyzw_to_rot6d(right_ee[3:7]),
                np.array([right_grip], dtype=np.float64),
            ],
            axis=0,
        ).astype(np.float32)

        payload: dict[str, Any] = {
            "domain_id": self._domain_id,
            "proprio": json_numpy_dumps(abs_eef),
            "language_instruction": obs.instruction,
            "steps": self._steps,
        }
        for i, img in enumerate(images):
            payload[f"image{i}"] = json_numpy_dumps(np.asarray(img, dtype=np.uint8))

        result = self._transport.infer(payload)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"X-VLA server error: {result['error']}")
        raw = np.asarray(result.get("action", result.get("actions")), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] < 20:
            raise ValueError(f"RoboTwin X-VLA action dim < 20: {raw.shape}")

        actions = np.stack(
            [
                _official_robotwin_row_to_ee16(row, gripper_threshold=self._gripper_threshold)
                for row in raw
            ],
            axis=0,
        )
        # Recurrent proprio uses last commanded EE (xyz+quat), like archive client.
        last = actions[-1]
        self._left_endpose_override = np.asarray(last[:7], dtype=np.float32).copy()
        self._right_endpose_override = np.asarray(last[8:15], dtype=np.float32).copy()
        return actions_to_chunk(actions, space="abs_ee_16d")

    def _act_official_bridge(
        self, obs: Observation, primary: np.ndarray | None = None
    ) -> ActionChunk:
        """Archived ``simpler_delta``: one HTTP chunk, one env step per ``act``.

        Abs→delta must use the *current* 8D ``eef_pos`` each step (not the eef
        frozen at query time), matching
        ``old_cache/.../xvla_official_inference.py``.
        """
        if obs.extra and obs.extra.get("task_name") is not None:
            self._bridge_task_name = str(obs.extra["task_name"])
        grip_th = self._bridge_gripper_threshold(obs)

        eef = None
        if obs.extra and obs.extra.get("eef_pos") is not None:
            eef = np.asarray(obs.extra["eef_pos"], dtype=np.float64).reshape(-1)
        elif obs.state is not None and np.asarray(obs.state).size >= 8:
            eef = np.asarray(obs.state, dtype=np.float64).reshape(-1)

        if self._proprio is None:
            if eef is not None and eef.shape[0] >= 3:
                self._proprio = _official_proprio_20_from_eef(eef)
            else:
                self._proprio = np.zeros(20, dtype=np.float32)

        if not self._bridge_plan:
            if primary is None:
                primary = np.asarray(require_image(obs, "primary"), dtype=np.uint8)
            payload: dict[str, Any] = {
                "proprio": json_numpy_dumps(self._proprio),
                "language_instruction": obs.instruction,
                "image0": json_numpy_dumps(np.asarray(primary, dtype=np.uint8)),
                "domain_id": self._domain_id,
                "steps": self._steps,
            }
            result = self._transport.infer(payload)
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(f"X-VLA server error: {result['error']}")
            raw = np.asarray(result.get("action", result.get("actions")), dtype=np.float32)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.shape[1] < 10:
                raise ValueError(f"Bridge X-VLA action dim < 10: {raw.shape}")
            for row in raw:
                self._bridge_plan.append(np.asarray(row[:10], dtype=np.float64))

        if not self._bridge_plan:
            raise RuntimeError("X-VLA Bridge action plan empty after infer")

        action_pred = self._bridge_plan.popleft()
        assert self._proprio is not None
        self._proprio = np.asarray(self._proprio, dtype=np.float32)
        self._proprio[:10] = action_pred[:10].astype(np.float32)

        # Old client: len(eef)<8 → treat pred as delta (warned as likely wrong).
        if eef is None or eef.shape[0] < 8:
            delta = _bridge_delta_ee_7d(
                action_pred,
                None,
                gripper_threshold=grip_th,
            )
        else:
            delta = _bridge_delta_ee_7d(
                action_pred,
                eef,
                gripper_threshold=grip_th,
            )
        return actions_to_chunk(delta.reshape(1, -1), space=self._space_for_step(obs.step))

    def _row_to_7d(self, row: np.ndarray) -> np.ndarray:
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if row.shape[0] == 7:
            return row
        if row.shape[0] < 10:
            raise ValueError(f"X-VLA action row too short: {row.shape}")
        pos = row[:3]
        rot = _rot6d_to_axangle(row[3:9])
        grip = _binarize_gripper(
            float(row[9]), threshold=self._gripper_threshold
        )
        return np.concatenate([pos, rot, [grip]]).astype(np.float32)

    def _row_to_7d_official(self, row: np.ndarray) -> np.ndarray:
        """Official gripper polarity (< thresh → open/+1)."""
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if row.shape[0] == 7:
            return row
        if row.shape[0] < 10:
            raise ValueError(f"X-VLA action row too short: {row.shape}")
        pos = row[:3]
        rot = _rot6d_to_axangle(row[3:9])
        grip = _binarize_gripper_official(
            float(row[9]), threshold=self._gripper_threshold
        )
        return np.concatenate([pos, rot, [grip]]).astype(np.float32)

    def close(self) -> None:
        self._transport.close()
