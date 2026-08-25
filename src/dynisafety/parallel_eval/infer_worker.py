"""Infer-process loop: window-aggregate obs, batch-forward, reply per slot.

Infer is stateless (EE override lives on Env). The loop must not wait for a
full ``num_envs`` barrier — slot ``take_action`` times vary 0.25–0.87 s, so a
fixed B is a straggler tax (plan 0.5). ``poll_ready`` + ``batch_window_ms``
fires with whatever arrived.

YAML must include ``extra.parallel_eval.rng_seed`` (no Python default). Env
fills ``ObsMsg.rng_key`` via ``mix_rng_key``; Infer only consumes it.

Start this process with ``conda run --no-capture-output -n <infer_conda_env>``.
After bind it prints ``parallel_eval.infer listening <host>:<port>`` so the
driver can hand the real port to Env workers when ``channel_port`` is 0.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable, Sequence

from dynisafety.parallel_eval.channel import (
    Channel,
    ChannelClosed,
    ChannelListener,
    poll_ready,
)
from dynisafety.parallel_eval.config import ParallelEvalConfig
from dynisafety.parallel_eval.protocol import (
    CTRL_EPISODE_BEGIN,
    CTRL_EPISODE_END,
    CTRL_ERROR,
    CTRL_SHUTDOWN,
    CTRL_WORKER_READY,
    LISTEN_LINE_PREFIX,  # re-exported: Driver used to import this from here
    ActMsg,
    CtrlMsg,
    ObsMsg,
    WireMsg,
    ctrl_error,
    from_wire,
    index_by_slot,
    mix_rng_key,
    to_wire,
)
from dynisafety.parallel_eval.xvla_local import (
    XvlaRuntime,
    forward_obs_batch,
    load_xvla,
    stack_slot_noise,
)

logger = logging.getLogger(__name__)


def serve_inference(
    *,
    config: ParallelEvalConfig,
    model_path: str,
    repo_root: str,
    processor_path: str | None,
    runtime: XvlaRuntime | None,
    listener: ChannelListener | None,
    on_bound: Callable[[ChannelListener], None] | None,
) -> None:
    """Bind, accept ``num_envs`` Env connections, then window-batch forever.

    ``runtime`` / ``listener`` / ``on_bound`` are injection points (pass
    ``None`` in production). Fourth-package driver: spawn this in the Infer
    conda env, parse the listen line for the OS-assigned port, then start Env
    workers pointing at that port.
    """
    _validate_serve_config(config)
    bound_here = listener is None
    if listener is None:
        listener = Channel.bind(
            host=config.channel_host,
            port=config.channel_port,
            timeout_s=config.channel_timeout_s,
        )
    print(f"{LISTEN_LINE_PREFIX} {listener.host}:{listener.port}", flush=True)
    if on_bound is not None:
        on_bound(listener)

    if runtime is None:
        runtime = load_xvla(
            repo_root=repo_root,
            model_path=model_path,
            processor_path=processor_path,
            device=config.infer_device,
            torch_dtype=config.torch_dtype,
        )

    channels: list[Channel] = []
    try:
        for _ in range(config.num_envs):
            channels.append(listener.accept())
        if bound_here:
            listener.close()
        _infer_loop(channels, config=config, runtime=runtime)
    except Exception as exc:
        _notify_peers(channels, exc)
        raise
    finally:
        for channel in channels:
            channel.close()
        if bound_here:
            listener.close()


def _validate_serve_config(config: ParallelEvalConfig) -> None:
    if config.num_envs < 1:
        raise ValueError(
            f"extra.parallel_eval.num_envs must be >= 1, got {config.num_envs}"
        )
    if config.batch_window_ms < 1:
        raise ValueError(
            f"extra.parallel_eval.batch_window_ms must be >= 1, got {config.batch_window_ms}"
        )
    if config.torch_dtype != "float32":
        raise ValueError(
            f"extra.parallel_eval.torch_dtype must be 'float32', got {config.torch_dtype!r}"
        )


def _infer_loop(
    channels: list[Channel],
    *,
    config: ParallelEvalConfig,
    runtime: XvlaRuntime,
) -> None:
    """Serve until shutdown or every Env socket is gone.

    An Env process exits after its last job and closes TCP. That is normal
    slot completion (P1), not a fatal Infer error. Drop the dead channel and
    keep the remaining slots.
    """
    window_s = config.batch_window_ms / 1000.0
    shutdown = False
    while not shutdown and channels:
        batch, shutdown = _collect_window(
            channels,
            window_s=window_s,
            max_obs=len(channels),
            config=config,
        )
        if batch:
            _dispatch_batch(batch, runtime=runtime)


def _drop_closed_channel(
    live: list[Channel],
    channel: Channel,
    waiting: list[Channel],
) -> None:
    logger.info("infer peer closed; dropping finished Env channel")
    if channel in waiting:
        waiting.remove(channel)
    if channel in live:
        live.remove(channel)
    try:
        channel.close()
    except Exception:
        logger.exception("failed to close dropped infer channel")


def _collect_window(
    channels: list[Channel],
    *,
    window_s: float,
    max_obs: int,
    config: ParallelEvalConfig,
) -> tuple[list[tuple[Channel, ObsMsg]], bool]:
    if not channels:
        return [], True
    deadline = _monotonic() + window_s
    batch: list[tuple[Channel, ObsMsg]] = []
    waiting = list(channels)
    while waiting and len(batch) < max_obs:
        remaining = deadline - _monotonic()
        if remaining <= 0.0:
            break
        ready = poll_ready(waiting, timeout_s=remaining)
        if not ready:
            break
        for channel in ready:
            try:
                msg = from_wire(channel.get())
            except ChannelClosed:
                _drop_closed_channel(channels, channel, waiting)
                continue
            if _handle_message(channel, msg, batch=batch, waiting=waiting, config=config):
                return batch, True
    if not channels:
        return batch, True
    return batch, False


def _handle_message(
    channel: Channel,
    msg: WireMsg,
    *,
    batch: list[tuple[Channel, ObsMsg]],
    waiting: list[Channel],
    config: ParallelEvalConfig,
) -> bool:
    """Return True when Infer should exit after this window."""
    if isinstance(msg, ObsMsg):
        _check_obs_rng(msg, config)
        batch.append((channel, msg))
        waiting.remove(channel)
        return False
    if isinstance(msg, CtrlMsg):
        return _handle_ctrl(msg)
    if isinstance(msg, ActMsg):
        raise ValueError(
            f"Infer received ActMsg on slot {msg.slot_id}; only Env sends obs/ctrl"
        )
    raise TypeError(f"Infer received unsupported message type {type(msg)!r}")


def _handle_ctrl(msg: CtrlMsg) -> bool:
    if msg.kind == CTRL_SHUTDOWN:
        logger.info("infer shutdown requested: %s", msg.reason)
        return True
    if msg.kind == CTRL_ERROR:
        logger.error(
            "peer error slot=%s episode=%s reason=%s\n%s",
            msg.slot_id,
            msg.episode_uid,
            msg.reason,
            msg.traceback,
        )
        raise RuntimeError(f"peer sent ctrl_error: {msg.reason}\n{msg.traceback}")
    if msg.kind == CTRL_WORKER_READY:
        logger.info("worker_ready slot=%s", msg.slot_id)
        return False
    if msg.kind == CTRL_EPISODE_BEGIN:
        logger.info("episode_begin slot=%s uid=%s", msg.slot_id, msg.episode_uid)
        return False
    if msg.kind == CTRL_EPISODE_END:
        logger.info("episode_end slot=%s uid=%s", msg.slot_id, msg.episode_uid)
        return False
    raise ValueError(f"unknown ctrl kind {msg.kind!r}")


def _check_obs_rng(msg: ObsMsg, config: ParallelEvalConfig) -> None:
    if not config.deterministic_rng:
        return
    expected = mix_rng_key(
        rng_seed=config.rng_seed,
        episode_uid=msg.episode_uid,
        step=msg.step,
    )
    if msg.rng_key != expected:
        raise ValueError(
            f"ObsMsg.rng_key {msg.rng_key} != mix_rng_key("
            f"rng_seed={config.rng_seed}, episode_uid={msg.episode_uid!r}, "
            f"step={msg.step}) = {expected}"
        )


def _dispatch_batch(
    batch: Sequence[tuple[Channel, ObsMsg]],
    *,
    runtime: XvlaRuntime,
) -> None:
    messages = [obs for _, obs in batch]
    index_by_slot(messages)
    num_actions = int(runtime.model.num_actions)
    dim_action = int(runtime.model.action_space.dim_action)
    x1 = stack_slot_noise(messages, num_actions=num_actions, dim_action=dim_action)
    rows = forward_obs_batch(runtime, messages, x1=x1)
    logger.info(
        "infer batch size %s slots=%s",
        len(batch),
        [obs.slot_id for obs in messages],
    )
    for (channel, obs), actions in zip(batch, rows, strict=True):
        channel.put(
            to_wire(
                ActMsg(
                    slot_id=obs.slot_id,
                    episode_uid=obs.episode_uid,
                    step=obs.step,
                    actions=actions,
                )
            )
        )


def _notify_peers(channels: Sequence[Channel], exc: BaseException) -> None:
    tb = traceback.format_exc()
    if not tb.strip():
        tb = f"{type(exc).__name__}: {exc}"
    payload = to_wire(
        ctrl_error(
            reason=f"{type(exc).__name__}: {exc}",
            traceback=tb,
            slot_id=None,
            episode_uid=None,
        )
    )
    for channel in channels:
        try:
            channel.put(payload)
        except Exception:
            logger.exception("failed to notify peer of infer error")


def _monotonic() -> float:
    return time.monotonic()
