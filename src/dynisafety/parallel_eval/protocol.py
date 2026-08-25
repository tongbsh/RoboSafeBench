"""Cross-process message contract for Infer (torch 2.1) ↔ Env (torch 2.7).

Wire payloads are pickle'd across those two conda stacks. A torch.Tensor or a
custom class instance is not safe to unpickle across that version gap, so every
outbound tree is reduced to a plain dict of numpy arrays / Python scalars /
strings (and list/dict of those) before it touches the socket.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np

# Official X-VLA proprio / action width and RoboTwin camera order (plan §合同).
# These are wire-shape invariants, not tunable hyperparameters.
PROPRIO_DIM = 20
ACTION_DIM = 20
OBS_IMAGE_COUNT = 3

MSG_OBS = "obs"
MSG_ACT = "act"
MSG_CTRL = "ctrl"

CTRL_WORKER_READY = "worker_ready"
CTRL_EPISODE_BEGIN = "episode_begin"
CTRL_EPISODE_END = "episode_end"
CTRL_SHUTDOWN = "shutdown"
CTRL_ERROR = "error"

CTRL_KINDS = frozenset(
    {
        CTRL_WORKER_READY,
        CTRL_EPISODE_BEGIN,
        CTRL_EPISODE_END,
        CTRL_SHUTDOWN,
        CTRL_ERROR,
    }
)

# Infer prints this; Driver scans conda/transformers noise for the bound port.
LISTEN_LINE_PREFIX = "parallel_eval.infer listening"


def mix_rng_key(*, rng_seed: int, episode_uid: str, step: int) -> int:
    """Stable 64-bit key for ``ObsMsg.rng_key``. Not ``hash()`` — that is salted.

    This is wire-contract code: Env fills ``ObsMsg.rng_key`` and Infer
    recomputes the same value when ``deterministic_rng`` is on. Both sides
    must share this exact blake2b field order
    (``rng_seed``, ``episode_uid``, ``step``). Do not move the hash back
    into Infer-only or Env-only modules.
    """
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
        raise TypeError(f"rng_seed must be int, got {type(rng_seed).__name__}")
    if not isinstance(episode_uid, str) or not episode_uid:
        raise ValueError(f"episode_uid must be a non-empty str, got {episode_uid!r}")
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(f"step must be int, got {type(step).__name__}")
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(str(rng_seed).encode("ascii"))
    hasher.update(b"\x00")
    hasher.update(episode_uid.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(str(step).encode("ascii"))
    return int.from_bytes(hasher.digest(), byteorder="little", signed=False)

_OBS_FIELDS = (
    "slot_id",
    "episode_uid",
    "step",
    "images",
    "proprio",
    "instruction",
    "domain_id",
    "denoising_steps",
    "rng_key",
)
_ACT_FIELDS = ("slot_id", "episode_uid", "step", "actions")
_CTRL_FIELDS = ("kind", "slot_id", "episode_uid", "reason", "traceback")

_ALLOWED_SCALAR_TYPES = (bool, int, float, str, type(None))
_ROOT_PATH = "<root>"


class _HasSlotId(Protocol):
    slot_id: int


TSlot = TypeVar("TSlot", bound=_HasSlotId)


@dataclass(frozen=True)
class ObsMsg:
    """One env slot's observation, already converted to wire-safe numpy.

    ``images`` is length-3 in official order (primary, left, right). ``proprio``
    is the 20-D EE/gripper vector assembled on the Env side so Infer stays
    stateless. ``rng_key`` exists because ``generate_actions`` draws
    ``torch.randn`` — P0 bit-alignment needs an explicit seed, not "hope the
    two paths match".
    """

    slot_id: int
    episode_uid: str
    step: int
    images: list[np.ndarray]
    proprio: np.ndarray
    instruction: str
    domain_id: int
    denoising_steps: int
    rng_key: int

    def __post_init__(self) -> None:
        _check_obs(self)


@dataclass(frozen=True)
class ActMsg:
    """One slot's action chunk: ``[T, 20]`` float32 in official X-VLA space."""

    slot_id: int
    episode_uid: str
    step: int
    actions: np.ndarray

    def __post_init__(self) -> None:
        _check_act(self)


@dataclass(frozen=True)
class CtrlMsg:
    """Driver orchestration (ready / episode boundaries / shutdown / errors).

    Optional fields stay present and ``None`` so the wire key set cannot drift
    between Infer and Env class definitions.
    """

    kind: str
    slot_id: int | None
    episode_uid: str | None
    reason: str | None
    traceback: str | None

    def __post_init__(self) -> None:
        _check_ctrl(self)


WireMsg = ObsMsg | ActMsg | CtrlMsg


def validate_payload(obj: Any) -> None:
    """Reject anything that is not pickle-safe across torch 2.1 ↔ 2.7.

    Allowed: ``None``, ``bool``, ``int``, ``float``, ``str``, ``numpy.ndarray``,
    ``numpy.generic``, and list/dict of those (dict keys must be ``str``).
    Anything else — including ``torch.Tensor`` and user class instances —
    raises with a dotted/bracket field path.
    """
    _validate_wire_value(obj, _ROOT_PATH)


def to_wire(msg: WireMsg) -> dict[str, Any]:
    """Encode a message as a plain dict. Never pickle the dataclass itself."""
    if isinstance(msg, ObsMsg):
        payload: dict[str, Any] = {
            "type": MSG_OBS,
            "slot_id": msg.slot_id,
            "episode_uid": msg.episode_uid,
            "step": msg.step,
            "images": [np.array(image, copy=True) for image in msg.images],
            "proprio": np.array(msg.proprio, copy=True),
            "instruction": msg.instruction,
            "domain_id": msg.domain_id,
            "denoising_steps": msg.denoising_steps,
            "rng_key": msg.rng_key,
        }
    elif isinstance(msg, ActMsg):
        payload = {
            "type": MSG_ACT,
            "slot_id": msg.slot_id,
            "episode_uid": msg.episode_uid,
            "step": msg.step,
            "actions": np.array(msg.actions, copy=True),
        }
    elif isinstance(msg, CtrlMsg):
        payload = {
            "type": MSG_CTRL,
            "kind": msg.kind,
            "slot_id": msg.slot_id,
            "episode_uid": msg.episode_uid,
            "reason": msg.reason,
            "traceback": msg.traceback,
        }
    else:
        raise TypeError(f"to_wire: unsupported message type {type(msg)!r}")
    validate_payload(payload)
    return payload


def from_wire(data: Mapping[str, Any]) -> WireMsg:
    """Decode a dict produced by ``to_wire``. Unknown/missing keys fail-loud."""
    if not isinstance(data, Mapping):
        raise TypeError(f"from_wire expected a mapping, got {type(data)!r}")
    validate_payload(data)
    if "type" not in data:
        raise ValueError("wire payload missing required key 'type'")
    kind = data["type"]
    if kind == MSG_OBS:
        fields = _require_exact_keys(data, _OBS_FIELDS, label="obs")
        return ObsMsg(
            slot_id=_require_int(fields["slot_id"], "slot_id"),
            episode_uid=_require_str(fields["episode_uid"], "episode_uid"),
            step=_require_int(fields["step"], "step"),
            images=_as_image_list(fields["images"]),
            proprio=_as_float32_vec(fields["proprio"], "proprio", PROPRIO_DIM),
            instruction=_require_str(fields["instruction"], "instruction"),
            domain_id=_require_int(fields["domain_id"], "domain_id"),
            denoising_steps=_require_int(fields["denoising_steps"], "denoising_steps"),
            rng_key=_require_int(fields["rng_key"], "rng_key"),
        )
    if kind == MSG_ACT:
        fields = _require_exact_keys(data, _ACT_FIELDS, label="act")
        return ActMsg(
            slot_id=_require_int(fields["slot_id"], "slot_id"),
            episode_uid=_require_str(fields["episode_uid"], "episode_uid"),
            step=_require_int(fields["step"], "step"),
            actions=_as_float32_matrix(fields["actions"], "actions", ACTION_DIM),
        )
    if kind == MSG_CTRL:
        fields = _require_exact_keys(data, _CTRL_FIELDS, label="ctrl")
        return CtrlMsg(
            kind=_require_str(fields["kind"], "kind"),
            slot_id=_optional_int(fields["slot_id"], "slot_id"),
            episode_uid=_optional_str(fields["episode_uid"], "episode_uid"),
            reason=_optional_str(fields["reason"], "reason"),
            traceback=_optional_str(fields["traceback"], "traceback"),
        )
    raise ValueError(f"wire payload type must be obs/act/ctrl, got {kind!r}")


def group_by_slot(messages: Sequence[TSlot]) -> dict[int, list[TSlot]]:
    """Bucket messages by ``slot_id``, preserving per-slot arrival order.

    Infer uses this when a window holds more than one step from the same env
    (pipeline) and must not reorder a slot's own timeline.
    """
    grouped: dict[int, list[TSlot]] = {}
    for msg in messages:
        slot = _require_int(msg.slot_id, "slot_id")
        grouped.setdefault(slot, []).append(msg)
    return grouped


def ungroup_by_slot(grouped: Mapping[int, Sequence[TSlot]]) -> list[TSlot]:
    """Flatten ``group_by_slot`` in mapping order, then within-slot order."""
    out: list[TSlot] = []
    for msgs in grouped.values():
        out.extend(msgs)
    return out


def index_by_slot(messages: Sequence[TSlot]) -> dict[int, TSlot]:
    """One message per ``slot_id``. Duplicates mean the batch window is wrong."""
    indexed: dict[int, TSlot] = {}
    for msg in messages:
        slot = _require_int(msg.slot_id, "slot_id")
        if slot in indexed:
            raise ValueError(f"duplicate slot_id {slot} in batch window")
        indexed[slot] = msg
    return indexed


def gather_by_slot_ids(indexed: Mapping[int, TSlot], slot_ids: Sequence[int]) -> list[TSlot]:
    """Rebuild a batch in the caller's slot order after ``generate_actions``."""
    out: list[TSlot] = []
    for slot in slot_ids:
        if slot not in indexed:
            raise KeyError(f"slot_id {slot} missing from indexed batch")
        out.append(indexed[slot])
    return out


def ctrl_worker_ready(*, slot_id: int) -> CtrlMsg:
    return CtrlMsg(
        kind=CTRL_WORKER_READY,
        slot_id=slot_id,
        episode_uid=None,
        reason=None,
        traceback=None,
    )


def ctrl_episode_begin(*, slot_id: int, episode_uid: str) -> CtrlMsg:
    return CtrlMsg(
        kind=CTRL_EPISODE_BEGIN,
        slot_id=slot_id,
        episode_uid=episode_uid,
        reason=None,
        traceback=None,
    )


def ctrl_episode_end(*, slot_id: int, episode_uid: str) -> CtrlMsg:
    return CtrlMsg(
        kind=CTRL_EPISODE_END,
        slot_id=slot_id,
        episode_uid=episode_uid,
        reason=None,
        traceback=None,
    )


def ctrl_shutdown(*, reason: str) -> CtrlMsg:
    return CtrlMsg(
        kind=CTRL_SHUTDOWN,
        slot_id=None,
        episode_uid=None,
        reason=reason,
        traceback=None,
    )


def ctrl_error(
    *,
    reason: str,
    traceback: str,
    slot_id: int | None,
    episode_uid: str | None,
) -> CtrlMsg:
    return CtrlMsg(
        kind=CTRL_ERROR,
        slot_id=slot_id,
        episode_uid=episode_uid,
        reason=reason,
        traceback=traceback,
    )


def _validate_wire_value(obj: Any, path: str) -> None:
    if isinstance(obj, _ALLOWED_SCALAR_TYPES):
        return
    if isinstance(obj, (np.ndarray, np.generic)):
        return
    if isinstance(obj, list):
        for index, item in enumerate(obj):
            _validate_wire_value(item, f"{path}[{index}]")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path}: dict keys must be str, got {type(key).__name__}"
                )
            child = key if path == _ROOT_PATH else f"{path}.{key}"
            _validate_wire_value(value, child)
        return
    raise TypeError(_disallowed_type_message(obj, path))


def _disallowed_type_message(obj: Any, path: str) -> str:
    cls = type(obj)
    module = getattr(cls, "__module__", "")
    name = getattr(cls, "__name__", cls.__qualname__)
    extra = ""
    if module == "torch" or module.startswith("torch."):
        extra = (
            " (torch tensors are not pickle-safe across Infer torch 2.1 and "
            "Env torch 2.7)"
        )
    return (
        f"{path}: disallowed type {module}.{name}{extra}; "
        "wire payload may only contain numpy arrays, Python scalars/strings, "
        "and list/dict of those"
    )


def _require_exact_keys(
    data: Mapping[str, Any],
    fields: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    expected = set(fields)
    actual = set(data) - {"type"}
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ValueError(f"{label} wire missing keys {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} wire unknown keys {sorted(unknown)}")
    return {key: data[key] for key in fields}


def _require_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}: expected int, got {type(value).__name__}")
    return value


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: expected non-empty str, got {value!r}")
    return value


def _optional_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, path)


def _optional_str(value: Any, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{path}: expected str or None, got {type(value).__name__}")
    return value


def _as_image_list(value: Any) -> list[np.ndarray]:
    if not isinstance(value, list):
        raise TypeError(f"images: expected list, got {type(value).__name__}")
    if len(value) != OBS_IMAGE_COUNT:
        raise ValueError(
            f"images: expected {OBS_IMAGE_COUNT} views (primary/left/right), "
            f"got {len(value)}"
        )
    images: list[np.ndarray] = []
    for index, item in enumerate(value):
        path = f"images[{index}]"
        if not isinstance(item, np.ndarray):
            raise TypeError(f"{path}: expected numpy.ndarray, got {type(item).__name__}")
        if item.dtype != np.uint8:
            raise TypeError(f"{path}: expected dtype uint8, got {item.dtype}")
        images.append(np.array(item, copy=True))
    return images


def _as_float32_vec(value: Any, path: str, dim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{path}: expected numpy.ndarray, got {type(value).__name__}")
    if value.dtype != np.float32:
        raise TypeError(f"{path}: expected dtype float32, got {value.dtype}")
    if value.shape != (dim,):
        raise ValueError(f"{path}: expected shape ({dim},), got {value.shape}")
    return np.array(value, copy=True)


def _as_float32_matrix(value: Any, path: str, dim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{path}: expected numpy.ndarray, got {type(value).__name__}")
    if value.dtype != np.float32:
        raise TypeError(f"{path}: expected dtype float32, got {value.dtype}")
    if value.ndim != 2 or value.shape[1] != dim:
        raise ValueError(f"{path}: expected shape (T, {dim}), got {value.shape}")
    return np.array(value, copy=True)


def _check_obs(msg: ObsMsg) -> None:
    _require_int(msg.slot_id, "slot_id")
    _require_str(msg.episode_uid, "episode_uid")
    _require_int(msg.step, "step")
    _require_str(msg.instruction, "instruction")
    _require_int(msg.domain_id, "domain_id")
    _require_int(msg.denoising_steps, "denoising_steps")
    _require_int(msg.rng_key, "rng_key")
    _as_image_list(msg.images)
    _as_float32_vec(msg.proprio, "proprio", PROPRIO_DIM)


def _check_act(msg: ActMsg) -> None:
    _require_int(msg.slot_id, "slot_id")
    _require_str(msg.episode_uid, "episode_uid")
    _require_int(msg.step, "step")
    _as_float32_matrix(msg.actions, "actions", ACTION_DIM)


def _check_ctrl(msg: CtrlMsg) -> None:
    if msg.kind not in CTRL_KINDS:
        raise ValueError(
            f"CtrlMsg.kind must be one of {sorted(CTRL_KINDS)}, got {msg.kind!r}"
        )
    if msg.kind == CTRL_WORKER_READY:
        _require_int(msg.slot_id, "slot_id")
        return
    if msg.kind in (CTRL_EPISODE_BEGIN, CTRL_EPISODE_END):
        _require_int(msg.slot_id, "slot_id")
        _require_str(msg.episode_uid, "episode_uid")
        return
    if msg.kind == CTRL_SHUTDOWN:
        _require_str(msg.reason, "reason")
        return
    _require_str(msg.reason, "reason")
    _require_str(msg.traceback, "traceback")
    _optional_int(msg.slot_id, "slot_id")
    _optional_str(msg.episode_uid, "episode_uid")
