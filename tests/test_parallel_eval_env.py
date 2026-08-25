"""Env-side parallel_eval tests (no sapien, no weights)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dynisafety.backends.robotwin.env import RoboTwinEnvHandle
from dynisafety.config.schema import ContactConfig, MetricsConfig, OutputConfig, RunConfig
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.core.protocols import TaskSpec
from dynisafety.core.rollout import ActionChunkQueue, BoundRun
from dynisafety.parallel_eval.channel import Channel
from dynisafety.parallel_eval.env_worker import (
    SlotEpisodeJob,
    drive_slot,
    serve_env_slot,
)
from dynisafety.parallel_eval.episode import (
    apply_chunk,
    begin,
    chunk_queue_empty,
    collect_obs,
    control_steps_remaining,
    finish,
    step_once,
)
from dynisafety.parallel_eval.protocol import (
    ACTION_DIM,
    CTRL_EPISODE_BEGIN,
    CTRL_EPISODE_END,
    CTRL_ERROR,
    CTRL_WORKER_READY,
    ActMsg,
    ObsMsg,
    from_wire,
    to_wire,
)
from dynisafety.parallel_eval.slot_state import (
    SlotState,
    bool_gripper_to_official,
    official_robotwin_row_to_ee16,
    pred_gripper_to_ee,
    quat_xyzw_to_rot6d,
)
from dynisafety.policies.adapters import xvla as xvla_mod
from dynisafety.policies.base import Observation

_GRIPPER_THRESHOLD = 0.7
_RNG_SEED = 20260825
_SIM_FIRST_CONTACT = 87
_SIM_CONTACT_STEPS = 12
_SIM_CONTACT_EVENTS = 3


def test_slot_state_reexports_adapter_formulas() -> None:
    """A local rewrite would make P0 alignment meaningless."""
    assert quat_xyzw_to_rot6d is xvla_mod._quat_xyzw_to_rot6d
    assert bool_gripper_to_official is xvla_mod._bool_gripper_to_official
    assert official_robotwin_row_to_ee16 is xvla_mod._official_robotwin_row_to_ee16
    assert pred_gripper_to_ee is xvla_mod._pred_gripper_to_ee


def test_slot_state_proprio_matches_adapter_bit_for_bit() -> None:
    left = np.array([0.11, -0.22, 0.33, 0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    right = np.array([0.44, 0.55, 0.66, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    obs = _obs_with_endpose(left=left, right=right, left_grip=0.0, right_grip=1.0)
    slot = SlotState(gripper_threshold=_GRIPPER_THRESHOLD)
    slot.reset()

    ours = slot.proprio_20d(obs)
    theirs = _adapter_proprio_20d(
        left_ee=left,
        right_ee=right,
        left_grip=0.0,
        right_grip=1.0,
    )
    np.testing.assert_array_equal(ours, theirs)
    assert ours.dtype == np.float32
    assert ours.shape == (20,)


def test_slot_state_reset_then_override_from_last_ee16_row() -> None:
    left = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    right = np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    obs = _obs_with_endpose(left=left, right=right, left_grip=0.2, right_grip=0.8)
    slot = SlotState(gripper_threshold=_GRIPPER_THRESHOLD)
    slot.reset()
    assert slot.left_endpose_override is None
    assert slot.right_endpose_override is None

    first = slot.proprio_20d(obs)
    np.testing.assert_array_equal(
        first,
        _adapter_proprio_20d(left_ee=left, right_ee=right, left_grip=0.2, right_grip=0.8),
    )

    actions_20 = np.linspace(-0.5, 0.5, 2 * ACTION_DIM, dtype=np.float32).reshape(2, ACTION_DIM)
    ee16 = slot.actions_20_to_ee16(actions_20)
    ref = np.stack(
        [
            xvla_mod._official_robotwin_row_to_ee16(row, gripper_threshold=_GRIPPER_THRESHOLD)
            for row in actions_20
        ],
        axis=0,
    )
    np.testing.assert_array_equal(ee16, ref)
    np.testing.assert_array_equal(slot.left_endpose_override, ee16[-1, :7])
    np.testing.assert_array_equal(slot.right_endpose_override, ee16[-1, 8:15])

    later = _obs_with_endpose(left=left, right=right, left_grip=0.9, right_grip=0.1)
    second = slot.proprio_20d(later)
    expected = _adapter_proprio_20d(
        left_ee=ee16[-1, :7],
        right_ee=ee16[-1, 8:15],
        left_grip=0.9,
        right_grip=0.1,
    )
    np.testing.assert_array_equal(second, expected)


def test_episode_module_does_not_own_seed_cursor_or_r1_names() -> None:
    root = Path(__file__).resolve().parents[1]
    episode_src = (root / "src/dynisafety/parallel_eval/episode.py").read_text(encoding="utf-8")
    worker_src = (root / "src/dynisafety/parallel_eval/env_worker.py").read_text(
        encoding="utf-8"
    )
    assert "next_candidate_seed" not in episode_src
    assert "episode_search_start" not in episode_src
    assert "def run_episode(" not in episode_src
    assert "def run_episodes(" not in episode_src
    assert "def run_episode(" not in worker_src
    assert "def run_episodes(" not in worker_src


def test_action_chunk_queue_reused_refill_only_when_empty(tmp_path: Path) -> None:
    env = _FakeTaskEnv()
    slot = begin(**_begin_kwargs(tmp_path, env, chunk_size=3, max_steps=8))
    assert isinstance(slot.action_queue, ActionChunkQueue)
    collect_obs(slot)
    assert chunk_queue_empty(slot)

    actions = _valid_actions_20(10)
    n_take = apply_chunk(slot, actions)
    assert n_take == 3
    assert len(slot.action_queue) == 3
    with pytest.raises(RuntimeError, match="empty queue"):
        apply_chunk(slot, actions)

    slot.action_queue.popleft()
    slot.action_queue.popleft()
    slot.action_queue.popleft()
    assert chunk_queue_empty(slot)
    short = _valid_actions_20(2)
    assert apply_chunk(slot, short) == 2
    assert len(slot.action_queue) == 2


def test_step_once_advances_one_control_step(tmp_path: Path) -> None:
    env = _FakeTaskEnv()
    slot = begin(**_begin_kwargs(tmp_path, env, chunk_size=4, max_steps=8))
    collect_obs(slot)
    apply_chunk(slot, _valid_actions_20(4))
    assert env.take_action_cnt == 0

    assert step_once(slot) is True
    assert env.take_action_cnt == 1
    assert len(env.actions) == 1
    assert env.action_types == ["ee"]

    assert step_once(slot) is True
    assert env.take_action_cnt == 2
    assert len(env.actions) == 2


def test_contact_clocks_three_fields_and_finish_schema(tmp_path: Path) -> None:
    env = _FakeTaskEnv(contact_on_control_step=1)
    slot = begin(**_begin_kwargs(tmp_path, env, chunk_size=3, max_steps=3))
    while control_steps_remaining(slot):
        collect_obs(slot)
        if chunk_queue_empty(slot):
            apply_chunk(slot, _valid_actions_20(5))
        if not step_once(slot):
            break

    assert env.take_action_cnt == 3
    record = finish(slot)
    assert record.schema_version == SCHEMA_VERSION
    assert record.schema_version == "2.0"
    assert record.first_contact_step == _SIM_FIRST_CONTACT
    assert record.extra["first_contact_control_step"] == 1
    assert record.extra["contact_control_steps"] == 1
    assert record.contact_steps == _SIM_CONTACT_STEPS
    assert record.contact_events == _SIM_CONTACT_EVENTS
    assert record.extra["contact_control_events"] == 1
    assert record.termination == "max_steps"
    assert record.success is False
    assert record.extra["accepted_seed"] == 11
    assert record.extra["candidate_seed_start"] == 11
    restored = EpisodeRecord.from_json_dict(record.to_json_dict())
    assert restored.schema_version == "2.0"
    assert restored.first_contact_step == _SIM_FIRST_CONTACT
    assert restored.extra["first_contact_control_step"] == 1


def test_finish_closes_env_exactly_once(tmp_path: Path) -> None:
    env = _FakeTaskEnv()
    slot = begin(**_begin_kwargs(tmp_path, env, chunk_size=1, max_steps=1))
    collect_obs(slot)
    apply_chunk(slot, _valid_actions_20(1))
    step_once(slot)
    assert env.close_calls == 0
    finish(slot)
    assert env.close_calls == 1
    finish(slot)
    assert env.close_calls == 1


def test_serve_env_slot_channel_roundtrip_and_single_close(tmp_path: Path) -> None:
    env = _FakeTaskEnv()
    handle = _handle(tmp_path, env)
    cfg, bound, resolved = _cfg_bound_resolved(tmp_path, chunk_size=2, max_steps=2)
    job = SlotEpisodeJob(episode_uid="ep-slot0", episode_index=0, start_seed=11)
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=5.0)
    err: list[BaseException] = []
    records_box: list[list[Any]] = []

    def _worker() -> None:
        try:
            records_box.append(
                serve_env_slot(
                    slot_id=0,
                    host=listener.host,
                    port=listener.port,
                    timeout_s=5.0,
                    handle=handle,
                    cfg=cfg,
                    bound=bound,
                    resolved_data=resolved,
                    workspace=tmp_path,
                    observation_needs_state=False,
                    warmup_steps=None,
                    post_warmup_space=None,
                    gripper_threshold=_GRIPPER_THRESHOLD,
                    domain_id=6,
                    denoising_steps=10,
                    rng_seed=_RNG_SEED,
                    jobs=[job],
                )
            )
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    try:
        with listener.accept() as peer:
            ready = from_wire(peer.get())
            assert ready.kind == CTRL_WORKER_READY
            assert ready.slot_id == 0
            begin_msg = from_wire(peer.get())
            assert begin_msg.kind == CTRL_EPISODE_BEGIN
            obs = from_wire(peer.get())
            assert isinstance(obs, ObsMsg)
            assert obs.step == 0
            assert obs.proprio.shape == (20,)
            assert obs.domain_id == 6
            peer.put(
                to_wire(
                    ActMsg(
                        slot_id=0,
                        episode_uid="ep-slot0",
                        step=0,
                        actions=_valid_actions_20(4),
                    )
                )
            )
            end = from_wire(peer.get())
            assert end.kind == CTRL_EPISODE_END
    finally:
        thread.join(timeout=5.0)
        listener.close()

    assert err == []
    assert thread.is_alive() is False
    assert env.take_action_cnt == 2
    assert env.close_calls == 1
    assert len(records_box[0]) == 1
    assert records_box[0][0].schema_version == "2.0"


def test_drive_slot_reports_ctrl_error_and_closes_once(tmp_path: Path) -> None:
    env = _FakeTaskEnv()
    handle = _handle(tmp_path, env)
    cfg, bound, _resolved = _cfg_bound_resolved(tmp_path, chunk_size=1, max_steps=1)
    job = SlotEpisodeJob(episode_uid="ep-bad", episode_index=0, start_seed=3)
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=5.0)
    err: list[BaseException] = []

    def _worker() -> None:
        try:
            client = Channel.connect(
                host=listener.host, port=listener.port, timeout_s=5.0
            )
            try:
                drive_slot(
                    channel=client,
                    slot_id=1,
                    handle=handle,
                    cfg=cfg,
                    bound=bound,
                    resolved_data={"extra": {}},
                    workspace=tmp_path,
                    observation_needs_state=False,
                    warmup_steps=None,
                    post_warmup_space=None,
                    gripper_threshold=_GRIPPER_THRESHOLD,
                    domain_id=6,
                    denoising_steps=10,
                    rng_seed=_RNG_SEED,
                    jobs=[job],
                )
            finally:
                client.close()
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    try:
        with listener.accept() as peer:
            ready = from_wire(peer.get())
            assert ready.kind == CTRL_WORKER_READY
            begin_msg = from_wire(peer.get())
            assert begin_msg.kind == CTRL_EPISODE_BEGIN
            error = from_wire(peer.get())
            assert error.kind == CTRL_ERROR
            assert error.traceback
            assert "robotwin" in (error.reason or "")
    finally:
        thread.join(timeout=5.0)
        listener.close()

    assert err
    assert env.close_calls == 1


def test_slot_state_requires_gripper_threshold() -> None:
    with pytest.raises(TypeError):
        SlotState()  # type: ignore[call-arg]


def test_serve_env_slot_requires_rng_seed() -> None:
    import inspect

    sig = inspect.signature(serve_env_slot)
    assert "rng_seed" in sig.parameters
    assert sig.parameters["rng_seed"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="rng_seed"):
        serve_env_slot(  # type: ignore[call-arg]
            slot_id=0,
            host="127.0.0.1",
            port=1,
            timeout_s=1.0,
            handle=None,  # type: ignore[arg-type]
            cfg=None,  # type: ignore[arg-type]
            bound=None,  # type: ignore[arg-type]
            resolved_data={},
            workspace=Path("."),
            observation_needs_state=False,
            warmup_steps=None,
            post_warmup_space=None,
            gripper_threshold=_GRIPPER_THRESHOLD,
            domain_id=6,
            denoising_steps=10,
            jobs=[],
        )


class _FakeTaskEnv:
    """Record ``take_action`` instead of running TOPP / ``scene.step``."""

    def __init__(self, *, contact_on_control_step: int | None = None) -> None:
        self.take_action_cnt = 0
        self.eval_success = False
        self.step_lim = 32
        self.actions: list[np.ndarray] = []
        self.action_types: list[str] = []
        self.close_calls = 0
        self.setup_calls = 0
        self.instruction: str | None = None
        self.contact_on_control_step = contact_on_control_step
        self.info: dict[str, Any] = {"dynisafety": _idle_dyn()}
        self._left = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._right = np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def setup_demo(self, **_kwargs: Any) -> None:
        self.setup_calls += 1
        self.take_action_cnt = 0
        self.eval_success = False

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    def get_obs(self) -> dict[str, Any]:
        return {
            "observation": {
                "head_camera": {"rgb": np.full((4, 4, 3), 10, dtype=np.uint8)},
                "left_camera": {"rgb": np.full((4, 4, 3), 20, dtype=np.uint8)},
                "right_camera": {"rgb": np.full((4, 4, 3), 30, dtype=np.uint8)},
            },
            "endpose": {
                "left_endpose": self._left.copy(),
                "right_endpose": self._right.copy(),
                "left_gripper": 0.0,
                "right_gripper": 1.0,
            },
        }

    def take_action(self, action: Any, action_type: str = "qpos") -> None:
        self.take_action_cnt += 1
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.action_types.append(str(action_type))
        collided = self.contact_on_control_step == (self.take_action_cnt - 1)
        self.info["dynisafety"] = _idle_dyn(
            step_collided=collided,
            step_force_n=2.5 if collided else 0.0,
            step_impulse_ns=0.4 if collided else 0.0,
            first_collision_step=_SIM_FIRST_CONTACT,
            collision_steps=_SIM_CONTACT_STEPS,
            collision_events=_SIM_CONTACT_EVENTS,
        )

    def close_env(self, clear_cache: bool = False) -> None:
        del clear_cache
        self.close_calls += 1


def _idle_dyn(**overrides: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "step_collided": False,
        "step_force_n": 0.0,
        "step_impulse_ns": 0.0,
        "contact_without_force": False,
        "max_collision_force": 1.25,
        "max_collision_impulse": 0.08,
        "cumulative_impulse_ns": 0.16,
        "collision_steps": 0,
        "collision_events": 0,
        "first_collision_step": None,
        "contact_without_force_steps": 0,
    }
    info.update(overrides)
    return info


def _obs_with_endpose(
    *,
    left: np.ndarray,
    right: np.ndarray,
    left_grip: float,
    right_grip: float,
) -> Observation:
    return Observation(
        images={"primary": np.zeros((2, 2, 3), dtype=np.uint8)},
        state=None,
        instruction="place the cup",
        step=0,
        extra={
            "endpose": {
                "left_endpose": np.asarray(left, dtype=np.float32),
                "right_endpose": np.asarray(right, dtype=np.float32),
                "left_gripper": float(left_grip),
                "right_gripper": float(right_grip),
            }
        },
    )


def _valid_actions_20(n_rows: int) -> np.ndarray:
    """Official 20-D rows with a valid rot6d so decode does not blow up."""
    rot6d = xvla_mod._quat_xyzw_to_rot6d(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
    row = np.zeros(ACTION_DIM, dtype=np.float32)
    row[3:9] = rot6d
    row[13:19] = rot6d
    row[0:3] = (0.1, 0.2, 0.3)
    row[10:13] = (0.4, 0.5, 0.6)
    rows = []
    for index in range(n_rows):
        item = row.copy()
        item[0:3] += 0.01 * index
        item[10:13] += 0.01 * index
        rows.append(item)
    return np.stack(rows, axis=0)


def _adapter_proprio_20d(
    *,
    left_ee: np.ndarray,
    right_ee: np.ndarray,
    left_grip: float,
    right_grip: float,
) -> np.ndarray:
    left_g = xvla_mod._bool_gripper_to_official(float(left_grip))
    right_g = xvla_mod._bool_gripper_to_official(float(right_grip))
    return np.concatenate(
        [
            left_ee[:3],
            xvla_mod._quat_xyzw_to_rot6d(left_ee[3:7]),
            np.array([left_g], dtype=np.float64),
            right_ee[:3],
            xvla_mod._quat_xyzw_to_rot6d(right_ee[3:7]),
            np.array([right_g], dtype=np.float64),
        ],
        axis=0,
    ).astype(np.float32)


def _handle(tmp_path: Path, env: _FakeTaskEnv) -> RoboTwinEnvHandle:
    return RoboTwinEnvHandle(
        task_env=env,
        task=TaskSpec(
            task_id=1,
            task_name="place_empty_cup",
            suite="main20",
            instruction="place the empty cup",
            max_steps=32,
        ),
        instruction="place the empty cup",
        task_args={},
        clear_cache_freq=8,
        robotwin_root=tmp_path,
    )


def _cfg_bound_resolved(
    tmp_path: Path, *, chunk_size: int, max_steps: int
) -> tuple[RunConfig, BoundRun, dict[str, Any]]:
    cfg = RunConfig(
        backend="robotwin",
        suite="main20",
        level=0,
        seed=7,
        model="xvla_official",
        checkpoint="/ckpt",
        max_steps=max_steps,
        run_id="pe-env-test",
        metrics=MetricsConfig(),
        contact=ContactConfig(
            topple_threshold_deg=30.0,
            force_equiv_dt="impact_window",
            force_report_window_s=0.05,
            impulse_reduce="sum",
        ),
        output=OutputConfig(results_dir=str(tmp_path), save_video=False),
    )
    bound = BoundRun(
        config_hash="hash-pe-env",
        action_space="abs_ee_16d",
        chunk_size=chunk_size,
    )
    resolved = {
        "extra": {
            "robotwin": {
                "expert_check": False,
                "expert_check_cache": False,
                "expert_check_max_attempts": 1,
                "expert_check_timeout_s": 1.0,
                "expert_check_log_every": 1,
                "prompt_mode": "task_name",
                "instruction_type": "unseen",
            }
        }
    }
    return cfg, bound, resolved


def _begin_kwargs(
    tmp_path: Path,
    env: _FakeTaskEnv,
    *,
    chunk_size: int,
    max_steps: int,
) -> dict[str, Any]:
    cfg, bound, resolved = _cfg_bound_resolved(tmp_path, chunk_size=chunk_size, max_steps=max_steps)
    return {
        "handle": _handle(tmp_path, env),
        "cfg": cfg,
        "bound": bound,
        "slot_state": SlotState(gripper_threshold=_GRIPPER_THRESHOLD),
        "episode_index": 0,
        "start_seed": 11,
        "resolved_data": resolved,
        "workspace": tmp_path,
        "observation_needs_state": False,
        "warmup_steps": None,
        "post_warmup_space": None,
    }
