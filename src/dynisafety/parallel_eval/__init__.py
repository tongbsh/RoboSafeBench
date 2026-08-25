"""Opt-in parallel eval (RoboTwin + X-VLA). HTTP ``/act`` stays the default path.

Wire contract, Infer/Env workers, seed-warm, and the process driver live in
this package. The HTTP ``dynisafety eval`` path stays the default.

Heavy modules (``driver``, ``seed_warm``, Env/Infer workers) are lazy. Infer
runs in the ``XVLA`` conda env, which has no ``imageio`` / sapien / mplib;
eager-importing ``seed_warm`` would pull ``backends.robotwin`` and crash
bootstrap (``from dynisafety.parallel_eval.infer_worker import serve_inference``
executes this package ``__init__`` first).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from dynisafety.parallel_eval.channel import (
    Channel,
    ChannelClosed,
    ChannelError,
    ChannelListener,
    ChannelTimeout,
)
from dynisafety.parallel_eval.config import ParallelEvalConfig, load_parallel_eval_config
from dynisafety.parallel_eval.protocol import (
    ACTION_DIM,
    CTRL_EPISODE_BEGIN,
    CTRL_EPISODE_END,
    CTRL_ERROR,
    CTRL_SHUTDOWN,
    CTRL_WORKER_READY,
    LISTEN_LINE_PREFIX,
    MSG_ACT,
    MSG_CTRL,
    MSG_OBS,
    OBS_IMAGE_COUNT,
    PROPRIO_DIM,
    ActMsg,
    CtrlMsg,
    ObsMsg,
    ctrl_episode_begin,
    ctrl_episode_end,
    ctrl_error,
    ctrl_shutdown,
    ctrl_worker_ready,
    from_wire,
    gather_by_slot_ids,
    group_by_slot,
    index_by_slot,
    mix_rng_key,
    to_wire,
    ungroup_by_slot,
    validate_payload,
)

if TYPE_CHECKING:
    from dynisafety.parallel_eval.driver import (
        ChildProcessGroup,
        drive_parallel_eval,
        parse_infer_listen_endpoint,
        terminate_child_processes,
    )
    from dynisafety.parallel_eval.seed_warm import (
        WARM_SCHEMA_VERSION,
        WarmDocument,
        WarmEpisode,
        jobs_from_warm_document,
        jobs_from_warm_path,
        load_warm_document,
        make_episode_uid,
        write_warm_document,
    )

__all__ = [
    "ACTION_DIM",
    "CTRL_EPISODE_BEGIN",
    "CTRL_EPISODE_END",
    "CTRL_ERROR",
    "CTRL_SHUTDOWN",
    "CTRL_WORKER_READY",
    "LISTEN_LINE_PREFIX",
    "MSG_ACT",
    "MSG_CTRL",
    "MSG_OBS",
    "OBS_IMAGE_COUNT",
    "PROPRIO_DIM",
    "ActMsg",
    "Channel",
    "ChannelClosed",
    "ChannelError",
    "ChannelListener",
    "ChannelTimeout",
    "CtrlMsg",
    "ObsMsg",
    "ParallelEvalConfig",
    "ctrl_episode_begin",
    "ctrl_episode_end",
    "ctrl_error",
    "ctrl_shutdown",
    "ctrl_worker_ready",
    "from_wire",
    "gather_by_slot_ids",
    "group_by_slot",
    "index_by_slot",
    "load_parallel_eval_config",
    "load_warm_document",
    "mix_rng_key",
    "make_episode_uid",
    "parse_infer_listen_endpoint",
    "terminate_child_processes",
    "to_wire",
    "ungroup_by_slot",
    "validate_payload",
    "write_warm_document",
    "WARM_SCHEMA_VERSION",
    "WarmDocument",
    "WarmEpisode",
    "drive_parallel_eval",
    "ChildProcessGroup",
    "jobs_from_warm_document",
    "jobs_from_warm_path",
]

# Env / parent process only. Infer must never import these at package load.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ChildProcessGroup": (".driver", "ChildProcessGroup"),
    "drive_parallel_eval": (".driver", "drive_parallel_eval"),
    "parse_infer_listen_endpoint": (".driver", "parse_infer_listen_endpoint"),
    "terminate_child_processes": (".driver", "terminate_child_processes"),
    "WARM_SCHEMA_VERSION": (".seed_warm", "WARM_SCHEMA_VERSION"),
    "WarmDocument": (".seed_warm", "WarmDocument"),
    "WarmEpisode": (".seed_warm", "WarmEpisode"),
    "jobs_from_warm_document": (".seed_warm", "jobs_from_warm_document"),
    "jobs_from_warm_path": (".seed_warm", "jobs_from_warm_path"),
    "load_warm_document": (".seed_warm", "load_warm_document"),
    "make_episode_uid": (".seed_warm", "make_episode_uid"),
    "write_warm_document": (".seed_warm", "write_warm_document"),
}


def __getattr__(name: str) -> Any:
    """PEP 562: load driver / seed-warm only when the parent process asks."""
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
