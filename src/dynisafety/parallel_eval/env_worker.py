"""One process, one RoboTwin env, one Channel client.

``make_env`` chdirs, ``_init_task_env_`` reseeds global numpy/torch, and
``sapien_clear_cache`` is process-wide. A second env in this process would
share that state. The entry is ``serve_env_slot`` / ``drive_slot`` — not
``run_episode(s)`` (hygiene R1 / H12).
"""

from __future__ import annotations

import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.backends.robotwin.env import RoboTwinEnvHandle
from dynisafety.config.schema import RunConfig
from dynisafety.core.episode import EpisodeRecord
from dynisafety.core.rollout import BoundRun
from dynisafety.parallel_eval.channel import Channel, ChannelError
from dynisafety.parallel_eval.episode import (
    EpisodeSlot,
    apply_chunk,
    begin,
    chunk_queue_empty,
    close_env_once,
    collect_obs,
    control_steps_remaining,
    finish,
    mark_error,
    step_once,
)
from dynisafety.parallel_eval.protocol import (
    CTRL_SHUTDOWN,
    OBS_IMAGE_COUNT,
    ActMsg,
    CtrlMsg,
    ObsMsg,
    ctrl_episode_begin,
    ctrl_episode_end,
    ctrl_error,
    ctrl_worker_ready,
    from_wire,
    mix_rng_key,
    to_wire,
)
from dynisafety.parallel_eval.slot_state import SlotState
from dynisafety.policies.base import Observation

_OBS_IMAGE_KEYS = ("primary", "left", "right")


@dataclass(frozen=True)
class SlotEpisodeJob:
    """One episode assigned to this slot.

    ``start_seed`` is the expert-check *search start* (the cache key), not the
    accepted demo seed. ``begin`` hands it to ``expert_check_setup``; a cache
    hit returns the warmed accepted seed. Passing the accepted seed here
    misses the cache and can silently drift the sequence (plan 0.7).
    """

    episode_uid: str
    episode_index: int
    start_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.episode_uid, str) or not self.episode_uid:
            raise ValueError(f"episode_uid must be a non-empty str, got {self.episode_uid!r}")
        if isinstance(self.episode_index, bool) or not isinstance(self.episode_index, int):
            raise TypeError(f"episode_index must be int, got {self.episode_index!r}")
        if self.episode_index < 0:
            raise ValueError(f"episode_index must be >= 0, got {self.episode_index}")
        if isinstance(self.start_seed, bool) or not isinstance(self.start_seed, int):
            raise TypeError(f"start_seed must be int, got {self.start_seed!r}")


def serve_env_slot(
    *,
    slot_id: int,
    host: str,
    port: int,
    timeout_s: float,
    handle: RoboTwinEnvHandle,
    cfg: RunConfig,
    bound: BoundRun,
    resolved_data: Mapping[str, Any],
    workspace: Path,
    observation_needs_state: bool,
    warmup_steps: int | None,
    post_warmup_space: str | None,
    gripper_threshold: float,
    domain_id: int,
    denoising_steps: int,
    rng_seed: int,
    jobs: Sequence[SlotEpisodeJob],
) -> list[EpisodeRecord]:
    """Process entry: connect, drive assigned jobs, always close the Channel.

    ``rng_seed`` is ``extra.parallel_eval.rng_seed`` (the x1 key), not
    ``cfg.seed``. Missing it is a TypeError — no Python default (R4).
    """
    channel = Channel.connect(host=host, port=port, timeout_s=timeout_s)
    try:
        return drive_slot(
            channel=channel,
            slot_id=slot_id,
            handle=handle,
            cfg=cfg,
            bound=bound,
            resolved_data=resolved_data,
            workspace=workspace,
            observation_needs_state=observation_needs_state,
            warmup_steps=warmup_steps,
            post_warmup_space=post_warmup_space,
            gripper_threshold=gripper_threshold,
            domain_id=domain_id,
            denoising_steps=denoising_steps,
            rng_seed=rng_seed,
            jobs=jobs,
        )
    finally:
        channel.close()


def drive_slot(
    *,
    channel: Channel,
    slot_id: int,
    handle: RoboTwinEnvHandle,
    cfg: RunConfig,
    bound: BoundRun,
    resolved_data: Mapping[str, Any],
    workspace: Path,
    observation_needs_state: bool,
    warmup_steps: int | None,
    post_warmup_space: str | None,
    gripper_threshold: float,
    domain_id: int,
    denoising_steps: int,
    rng_seed: int,
    jobs: Sequence[SlotEpisodeJob],
) -> list[EpisodeRecord]:
    """Already-connected loop. Infer stays unblocked via ``ctrl_error`` on failure."""
    _require_int(slot_id, "slot_id")
    _require_int(domain_id, "domain_id")
    _require_int(denoising_steps, "denoising_steps")
    _require_int(rng_seed, "rng_seed")
    slot_state = SlotState(gripper_threshold=gripper_threshold)
    episode_uid: str | None = None
    open_slot: EpisodeSlot | None = None
    closed_by_finish = False
    records: list[EpisodeRecord] = []
    try:
        channel.put(to_wire(ctrl_worker_ready(slot_id=slot_id)))
        for job in jobs:
            episode_uid = job.episode_uid
            closed_by_finish = False
            open_slot = None
            channel.put(
                to_wire(ctrl_episode_begin(slot_id=slot_id, episode_uid=job.episode_uid))
            )
            slot = begin(
                handle=handle,
                cfg=cfg,
                bound=bound,
                slot_state=slot_state,
                episode_index=job.episode_index,
                start_seed=job.start_seed,
                resolved_data=resolved_data,
                workspace=workspace,
                observation_needs_state=observation_needs_state,
                warmup_steps=warmup_steps,
                post_warmup_space=post_warmup_space,
            )
            open_slot = slot
            stop_jobs = False
            try:
                while control_steps_remaining(slot):
                    observation = collect_obs(slot)
                    if chunk_queue_empty(slot):
                        channel.put(
                            to_wire(
                                _obs_msg(
                                    observation,
                                    slot_id=slot_id,
                                    episode_uid=job.episode_uid,
                                    proprio=slot_state.proprio_20d(observation),
                                    domain_id=domain_id,
                                    denoising_steps=denoising_steps,
                                    rng_seed=rng_seed,
                                )
                            )
                        )
                        incoming = from_wire(channel.get())
                        if _is_shutdown(incoming):
                            stop_jobs = True
                            break
                        actions_20 = _require_act(
                            incoming,
                            slot_id=slot_id,
                            episode_uid=job.episode_uid,
                        )
                        apply_chunk(slot, actions_20)
                    if not step_once(slot):
                        break
            except ChannelError:
                raise
            except Exception as exc:  # noqa: BLE001
                mark_error(slot, exc)
            records.append(finish(slot))
            closed_by_finish = True
            open_slot = None
            channel.put(
                to_wire(ctrl_episode_end(slot_id=slot_id, episode_uid=job.episode_uid))
            )
            if stop_jobs:
                break
        return records
    except Exception as exc:
        if open_slot is not None and not open_slot.env_closed:
            close_env_once(open_slot)
        elif open_slot is None and not closed_by_finish:
            handle.close()
        try:
            channel.put(
                to_wire(
                    ctrl_error(
                        reason=str(exc),
                        traceback=traceback.format_exc(),
                        slot_id=slot_id,
                        episode_uid=episode_uid,
                    )
                )
            )
        except Exception:
            pass
        raise


def _obs_msg(
    observation: Observation,
    *,
    slot_id: int,
    episode_uid: str,
    proprio: np.ndarray,
    domain_id: int,
    denoising_steps: int,
    rng_seed: int,
) -> ObsMsg:
    return ObsMsg(
        slot_id=slot_id,
        episode_uid=episode_uid,
        step=int(observation.step),
        images=_obs_images(observation),
        proprio=np.asarray(proprio, dtype=np.float32),
        instruction=observation.instruction,
        domain_id=int(domain_id),
        denoising_steps=int(denoising_steps),
        rng_key=mix_rng_key(
            rng_seed=rng_seed,
            episode_uid=episode_uid,
            step=int(observation.step),
        ),
    )


def _obs_images(observation: Observation) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for key in _OBS_IMAGE_KEYS:
        if key not in observation.images:
            raise KeyError(
                f"Observation.images missing {key!r}; have {sorted(observation.images)}"
            )
        images.append(np.asarray(observation.images[key], dtype=np.uint8))
    if len(images) != OBS_IMAGE_COUNT:
        raise ValueError(f"expected {OBS_IMAGE_COUNT} views, got {len(images)}")
    return images


def _require_act(msg: object, *, slot_id: int, episode_uid: str) -> np.ndarray:
    if not isinstance(msg, ActMsg):
        raise TypeError(f"expected ActMsg from Infer, got {type(msg)!r}")
    if msg.slot_id != slot_id:
        raise ValueError(f"ActMsg.slot_id {msg.slot_id} != worker slot {slot_id}")
    if msg.episode_uid != episode_uid:
        raise ValueError(
            f"ActMsg.episode_uid {msg.episode_uid!r} != current {episode_uid!r}"
        )
    return msg.actions


def _is_shutdown(msg: object) -> bool:
    return isinstance(msg, CtrlMsg) and msg.kind == CTRL_SHUTDOWN


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {value!r}")
    return value


__all__ = [
    "SlotEpisodeJob",
    "drive_slot",
    "serve_env_slot",
]
