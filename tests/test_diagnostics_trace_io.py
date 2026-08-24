"""Trace recorder / writer / reader I/O — no simulator."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dynisafety.config.schema import DiagnosticsConfig, DiagnosticsTraceConfig
from dynisafety.diagnostics.reader import (
    TraceSchemaVersionError,
    TruncatedTraceError,
    iter_traces,
    read_index,
    read_trace,
)
from dynisafety.diagnostics.recorder import (
    GRIPPER_INDICES,
    StepTraceRecorder,
    extract_gripper_cmd,
)
from dynisafety.diagnostics.schema import (
    TRACE_SCHEMA_VERSION,
    DisturbancePhase,
    Trace,
    TraceHeader,
)
from dynisafety.diagnostics.writer import TraceWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDER_SRC = REPO_ROOT / "src" / "dynisafety" / "diagnostics" / "recorder.py"


def _trace_cfg(
    *,
    save: bool = True,
    fmt: str = "jsonl_gz",
    stride: int = 1,
) -> DiagnosticsTraceConfig:
    return DiagnosticsTraceConfig(save=save, format=fmt, stride=stride)


def _header(**kw: object) -> TraceHeader:
    base: dict[str, object] = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_id": "run-test",
        "episode_index": 0,
        "backend": "libero",
        "level": 1,
        "seed": 7,
        "disturbance_triggered": False,
        "t_onset": 10,
        "t_hazard_end": None,
        "hazard_persists": None,
        "action_chunk_len": 8,
        "action_space": "delta_ee_7d",
        "control_freq_hz": 20.0,
        "stride": 1,
        "n_steps": 0,
    }
    base.update(kw)
    return TraceHeader(**base)  # type: ignore[arg-type]


def _record_step(
    rec: StepTraceRecorder,
    t: int,
    *,
    eef_pos: object = (0.1, 0.2, 0.3),
    action: object | None = None,
    **kwargs: object,
) -> None:
    if action is None:
        action = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    rec.record(
        t=t,
        eef_pos=eef_pos,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
        in_contact=bool(kwargs.pop("in_contact", False)),
        force_n=float(kwargs.pop("force_n", 0.0)),  # type: ignore[arg-type]
        impulse_ns=float(kwargs.pop("impulse_ns", 0.0)),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _finalize(rec: StepTraceRecorder) -> Trace:
    return rec.finalize(
        t_hazard_end=40,
        disturbance_triggered=True,
        hazard_persists=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# recorder → writer → reader roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_equal(tmp_path: Path) -> None:
    header = _header(episode_index=0)
    rec = StepTraceRecorder(header, _trace_cfg())
    eef = np.array([0.4, 0.0, 0.2], dtype=np.float64)
    action = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)
    rec.record(
        t=0,
        eef_pos=eef,
        eef_quat=(0.0, 0.0, 0.0, 1.0),
        base_xy_yaw=None,
        action=action,
        obstacle_pos=(1.0, 0.0, 0.2),
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
        phase=DisturbancePhase.INACTIVE,
    )
    rec.record(
        t=1,
        eef_pos=(0.41, 0.0, 0.2),
        action=(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        obstacle_pos=None,
        in_contact=True,
        force_n=12.5,
        impulse_ns=0.4,
        phase="active",
    )
    original = _finalize(rec)
    writer = TraceWriter(tmp_path, _trace_cfg())
    rel = writer.write(0, original)
    assert rel == "traces/ep0000.jsonl.gz"
    loaded = read_trace(tmp_path / rel)
    assert loaded == original
    assert loaded.header == original.header
    assert loaded.steps == original.steps


def test_eef_pos_secondary_roundtrip(tmp_path: Path) -> None:
    """Dual-arm right EEF is copied and survives writer → reader."""
    rec = StepTraceRecorder(_header(backend="robotwin", action_space="qpos_14d"), _trace_cfg())
    live_right = np.array([0.41, 0.52, 0.63], dtype=np.float64)
    rec.record(
        t=0,
        eef_pos=(0.11, 0.22, 0.33),
        eef_pos_secondary=live_right,
        action=(0.0,) * 14,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    live_right[0] = 99.0
    original = _finalize(rec)
    assert original.steps[0].eef_pos_secondary == (0.41, 0.52, 0.63)
    assert original.steps[0].eef_pos == (0.11, 0.22, 0.33)

    writer = TraceWriter(tmp_path, _trace_cfg())
    rel = writer.write(0, original)
    assert rel is not None
    loaded = read_trace(tmp_path / rel)
    assert loaded == original
    assert loaded.steps[0].eef_pos_secondary == (0.41, 0.52, 0.63)
    assert loaded.steps[0].eef_pos_secondary is not live_right


def test_roundtrip_jsonl_uncompressed(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(episode_index=3), _trace_cfg(fmt="jsonl"))
    _record_step(rec, 0, eef_pos=(0.0, 0.0, 0.0), obstacle_pos=(3.0, 4.0, 0.0))
    original = rec.finalize(
        t_hazard_end=None,
        disturbance_triggered=False,
        hazard_persists=None,
        t_onset=None,
    )
    writer = TraceWriter(tmp_path, _trace_cfg(fmt="jsonl"))
    rel = writer.write(3, original)
    assert rel == "traces/ep0003.jsonl"
    assert (tmp_path / rel).read_bytes()[:1] != b"\x1f"
    assert read_trace(tmp_path / rel) == original


# ---------------------------------------------------------------------------
# defensive copy — the load-bearing test
# ---------------------------------------------------------------------------


def test_defensive_copy_numpy_buffer_reuse() -> None:
    """Rollout obs buffers are mutated in place; recorded steps must not follow."""
    rec = StepTraceRecorder(_header(), _trace_cfg())
    eef = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    obstacle = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    for t in range(3):
        eef[0] = float(t)
        action[0] = float(t) * 0.1
        obstacle[1] = float(t)
        quat[0] = float(t) * 0.01
        rec.record(
            t=t,
            eef_pos=eef,
            eef_quat=quat,
            action=action,
            obstacle_pos=obstacle,
            in_contact=False,
            force_n=0.0,
            impulse_ns=0.0,
        )
    steps = _finalize(rec).steps
    assert steps[0].eef_pos == (0.0, 0.0, 0.0)
    assert steps[1].eef_pos == (1.0, 0.0, 0.0)
    assert steps[2].eef_pos == (2.0, 0.0, 0.0)
    assert steps[0].action[0] == pytest.approx(0.0)
    assert steps[1].action[0] == pytest.approx(0.1)
    assert steps[2].action[0] == pytest.approx(0.2)
    assert steps[0].obstacle_pos == (1.0, 0.0, 0.0)
    assert steps[1].obstacle_pos == (1.0, 1.0, 0.0)
    assert steps[2].eef_quat is not None and steps[2].eef_quat[0] == pytest.approx(0.02)


def test_recorder_does_not_hold_caller_references() -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg())
    eef = [0.1, 0.2, 0.3]
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    obstacle = [1.0, 2.0, 3.0]
    quat = [0.0, 0.0, 0.0, 1.0]
    base = [0.4, 0.5, 0.6]
    rec.record(
        t=0,
        eef_pos=eef,
        eef_quat=quat,
        base_xy_yaw=base,
        action=action,
        obstacle_pos=obstacle,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    eef[0] = 99.0
    action[-1] = 99.0
    obstacle[0] = 99.0
    quat[0] = 99.0
    base[0] = 99.0
    step = _finalize(rec).steps[0]
    assert step.eef_pos is not eef
    assert step.action is not action
    assert step.obstacle_pos is not obstacle
    assert step.eef_quat is not quat
    assert step.base_xy_yaw is not base
    assert step.eef_pos == (0.1, 0.2, 0.3)
    assert step.action[-1] == -1.0
    assert step.obstacle_pos == (1.0, 2.0, 3.0)
    assert step.eef_quat == (0.0, 0.0, 0.0, 1.0)
    assert step.base_xy_yaw == (0.4, 0.5, 0.6)


def test_clearance_from_copied_positions() -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg())
    rec.record(
        t=0,
        eef_pos=(0.0, 0.0, 0.0),
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        obstacle_pos=(3.0, 4.0, 0.0),
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    rec.record(
        t=1,
        eef_pos=(0.0, 0.0, 0.0),
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        obstacle_pos=None,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    steps = _finalize(rec).steps
    assert steps[0].d_eef_obstacle == pytest.approx(5.0)
    assert steps[1].d_eef_obstacle is None


# ---------------------------------------------------------------------------
# stride
# ---------------------------------------------------------------------------


def test_stride_records_every_nth_step() -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg(stride=2))
    for t in range(6):
        _record_step(rec, t, eef_pos=(float(t), 0.0, 0.0))
    trace = _finalize(rec)
    assert [s.t for s in trace.steps] == [0, 2, 4]
    assert trace.header.n_steps == 3
    assert trace.header.stride == 2


def test_stride_override_wins_over_cfg() -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg(stride=1), stride=3)
    for t in range(6):
        _record_step(rec, t)
    assert [s.t for s in _finalize(rec).steps] == [0, 3]


def test_finalize_header_stride_matches_records() -> None:
    rec = StepTraceRecorder(_header(stride=99), _trace_cfg(stride=3))
    for t in range(10):
        _record_step(rec, t)
    trace = _finalize(rec)
    assert [s.t for s in trace.steps] == [0, 3, 6, 9]
    assert trace.header.stride == 3
    assert trace.header.n_steps == len(trace.steps) == 4


def test_finalize_writes_constructor_stride_override() -> None:
    rec = StepTraceRecorder(_header(stride=99), _trace_cfg(stride=1), stride=3)
    for t in range(9):
        _record_step(rec, t)
    trace = _finalize(rec)
    assert trace.header.stride == 3
    assert trace.header.n_steps == len(trace.steps) == 3
    assert [s.t for s in trace.steps] == [0, 3, 6]


def test_stride_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="stride"):
        StepTraceRecorder(_header(), stride=0)


def test_diagnostics_config_nested_trace_is_accepted() -> None:
    parent = DiagnosticsConfig(enabled=True, trace=_trace_cfg(stride=2))
    rec = StepTraceRecorder(_header(), parent)
    assert rec.stride == 2


# ---------------------------------------------------------------------------
# gripper_cmd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("space", "action", "expected"),
    [
        ("delta_ee_7d", [0, 0, 0, 0, 0, 0, -0.5], -0.5),
        ("abs_ee_7d", [0, 0, 0, 0, 0, 0, 0.9], 0.9),
        (
            "abs_ee_16d",
            [0] * 7 + [0.11] + [0] * 7 + [0.22],
            0.11,
        ),
        ("qpos_14d", [0] * 6 + [0.3] + [0] * 6 + [0.4], 0.3),
        ("hybrid_mobile_12d", [0] * 6 + [0.7] + [0] * 5, 0.7),
    ],
)
def test_extract_gripper_cmd_known_spaces(
    space: str, action: list[float], expected: float
) -> None:
    assert extract_gripper_cmd(action, space) == pytest.approx(expected)
    rec = StepTraceRecorder(_header(action_space=space), _trace_cfg())
    rec.record(
        t=0,
        eef_pos=(0.0, 0.0, 0.0),
        action=action,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    assert _finalize(rec).steps[0].gripper_cmd == pytest.approx(expected)


def test_dual_arm_gripper_indices_match_action_space_layout() -> None:
    assert GRIPPER_INDICES["abs_ee_16d"] == (7, 15)
    assert GRIPPER_INDICES["qpos_14d"] == (6, 13)
    assert GRIPPER_INDICES["hybrid_mobile_12d"] == (6,)
    assert GRIPPER_INDICES["delta_ee_7d"] == (6,)
    assert GRIPPER_INDICES["abs_ee_7d"] == (6,)


def test_extract_gripper_cmd_unknown_space_returns_none() -> None:
    assert extract_gripper_cmd([0.0] * 7, "not_a_space") is None
    rec = StepTraceRecorder(_header(action_space="mystery_9d"), _trace_cfg())
    rec.record(
        t=0,
        eef_pos=(0.0, 0.0, 0.0),
        action=[0.0] * 9,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    assert _finalize(rec).steps[0].gripper_cmd is None


def test_extract_gripper_cmd_short_action_returns_none() -> None:
    assert extract_gripper_cmd([0.0, 0.0], "delta_ee_7d") is None


def test_explicit_gripper_cmd_overrides_slice() -> None:
    rec = StepTraceRecorder(_header(action_space="delta_ee_7d"), _trace_cfg())
    rec.record(
        t=0,
        eef_pos=(0.0, 0.0, 0.0),
        action=[0.0] * 6 + [0.5],
        gripper_cmd=0.25,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    assert _finalize(rec).steps[0].gripper_cmd == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# writer index / sha256 / incremental
# ---------------------------------------------------------------------------


def test_index_json_and_sha256(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(episode_index=0), _trace_cfg())
    _record_step(rec, 0)
    _record_step(rec, 1)
    trace = _finalize(rec)
    rel = TraceWriter(tmp_path, _trace_cfg()).write(0, trace)
    assert rel is not None
    index = read_index(tmp_path)
    assert set(index) == {0}
    entry = index[0]
    assert entry["filename"] == "ep0000.jsonl.gz"
    assert entry["n_steps"] == 2
    on_disk = tmp_path / "traces" / entry["filename"]
    assert entry["sha256"] == _sha256(on_disk)
    assert len(entry["sha256"]) == 64


def test_incremental_write_keeps_prior_episodes(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path, _trace_cfg(fmt="jsonl"))
    hashes = []
    for ep in range(3):
        rec = StepTraceRecorder(_header(episode_index=ep, seed=ep), _trace_cfg())
        for t in range(ep + 1):
            _record_step(rec, t)
        trace = rec.finalize(
            t_hazard_end=5,
            disturbance_triggered=True,
            hazard_persists=False,
        )
        writer.write(ep, trace)
        hashes.append(_sha256(tmp_path / "traces" / f"ep{ep:04d}.jsonl"))
    index = read_index(tmp_path)
    assert sorted(index) == [0, 1, 2]
    assert index[0]["n_steps"] == 1
    assert index[1]["n_steps"] == 2
    assert index[2]["n_steps"] == 3
    for ep, digest in enumerate(hashes):
        assert index[ep]["sha256"] == digest
        assert _sha256(tmp_path / "traces" / f"ep{ep:04d}.jsonl") == digest
    loaded = list(iter_traces(tmp_path))
    assert [tr.header.episode_index for tr in loaded] == [0, 1, 2]
    assert [tr.header.n_steps for tr in loaded] == [1, 2, 3]


def test_save_false_is_noop(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg())
    _record_step(rec, 0)
    trace = _finalize(rec)
    writer = TraceWriter(tmp_path, _trace_cfg(save=False))
    assert writer.save is False
    assert writer.write(0, trace) is None
    assert not (tmp_path / "traces").exists()


def test_write_rejects_episode_index_mismatch(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(episode_index=1), _trace_cfg())
    _record_step(rec, 0)
    trace = _finalize(rec)
    with pytest.raises(ValueError, match="episode_index"):
        TraceWriter(tmp_path, _trace_cfg()).write(0, trace)


def test_gzip_and_jsonl_roundtrip_equal(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(episode_index=0), _trace_cfg())
    _record_step(rec, 0, eef_pos=(0.2, 0.1, 0.4), action=(0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0))
    original = _finalize(rec)
    gz_dir = tmp_path / "gz"
    raw_dir = tmp_path / "raw"
    TraceWriter(gz_dir, _trace_cfg(fmt="jsonl_gz")).write(0, original)
    TraceWriter(raw_dir, _trace_cfg(fmt="jsonl")).write(0, original)
    assert read_trace(gz_dir / "traces" / "ep0000.jsonl.gz") == original
    assert read_trace(raw_dir / "traces" / "ep0000.jsonl") == original
    assert read_trace(gz_dir / "traces" / "ep0000.jsonl.gz") == read_trace(
        raw_dir / "traces" / "ep0000.jsonl"
    )


# ---------------------------------------------------------------------------
# reader errors
# ---------------------------------------------------------------------------


def test_schema_version_mismatch(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg())
    _record_step(rec, 0)
    trace = _finalize(rec)
    rel = TraceWriter(tmp_path, _trace_cfg(fmt="jsonl")).write(0, trace)
    assert rel is not None
    path = tmp_path / rel
    header_line, rest = path.read_text(encoding="utf-8").split("\n", 1)
    payload = json.loads(header_line)
    payload["trace_schema_version"] = "0.9"
    path.write_text(json.dumps(payload) + "\n" + rest, encoding="utf-8")
    with pytest.raises(TraceSchemaVersionError, match="docs/behavior_diagnostics.md") as exc:
        read_trace(path)
    assert exc.value.found == "0.9"


def test_old_trace_missing_stride_is_readable_schema_error(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg())
    _record_step(rec, 0)
    trace = _finalize(rec)
    rel = TraceWriter(tmp_path, _trace_cfg(fmt="jsonl")).write(0, trace)
    assert rel is not None
    path = tmp_path / rel
    header_line, rest = path.read_text(encoding="utf-8").split("\n", 1)
    payload = json.loads(header_line)
    del payload["stride"]
    path.write_text(json.dumps(payload) + "\n" + rest, encoding="utf-8")
    with pytest.raises(TraceSchemaVersionError, match="docs/behavior_diagnostics.md") as exc:
        read_trace(path)
    message = str(exc.value)
    assert "stride" in message
    lowered = message.lower()
    assert "schema" in lowered or "contract" in lowered or "version" in lowered
    assert not isinstance(exc.value, KeyError)


def test_schema_version_checked_before_missing_fields(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg())
    _record_step(rec, 0)
    trace = _finalize(rec)
    rel = TraceWriter(tmp_path, _trace_cfg(fmt="jsonl")).write(0, trace)
    assert rel is not None
    path = tmp_path / rel
    header_line, rest = path.read_text(encoding="utf-8").split("\n", 1)
    payload = json.loads(header_line)
    payload["trace_schema_version"] = "0.9"
    del payload["stride"]
    path.write_text(json.dumps(payload) + "\n" + rest, encoding="utf-8")
    with pytest.raises(TraceSchemaVersionError, match="docs/behavior_diagnostics.md") as exc:
        read_trace(path)
    assert exc.value.found == "0.9"
    assert "does not match" in str(exc.value)


def test_truncated_jsonl_raises(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg())
    for t in range(3):
        _record_step(rec, t)
    trace = _finalize(rec)
    rel = TraceWriter(tmp_path, _trace_cfg(fmt="jsonl")).write(0, trace)
    assert rel is not None
    path = tmp_path / rel
    data = path.read_bytes()
    path.write_bytes(data[:-8])
    with pytest.raises(TruncatedTraceError, match="truncated"):
        read_trace(path)


def test_truncated_jsonl_strict_false_keeps_prefix(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg())
    for t in range(3):
        _record_step(rec, t)
    trace = _finalize(rec)
    rel = TraceWriter(tmp_path, _trace_cfg(fmt="jsonl")).write(0, trace)
    assert rel is not None
    path = tmp_path / rel
    lines = path.read_text(encoding="utf-8").splitlines()
    # keep header + two complete steps + a torn last line
    torn = "\n".join(lines[:3]) + "\n" + lines[3][:12]
    path.write_text(torn, encoding="utf-8")
    recovered = read_trace(path, strict=False)
    assert recovered.header.n_steps == 2
    assert [s.t for s in recovered.steps] == [0, 1]


def test_truncated_gzip_raises(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg())
    for t in range(4):
        _record_step(rec, t)
    trace = _finalize(rec)
    rel = TraceWriter(tmp_path, _trace_cfg(fmt="jsonl_gz")).write(0, trace)
    assert rel is not None
    path = tmp_path / rel
    blob = path.read_bytes()
    path.write_bytes(blob[: max(10, len(blob) // 2)])
    with pytest.raises(TruncatedTraceError):
        read_trace(path)


def test_iter_traces_empty_run(tmp_path: Path) -> None:
    assert list(iter_traces(tmp_path)) == []


def test_read_index_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="index"):
        read_index(tmp_path)


# ---------------------------------------------------------------------------
# finalize / header
# ---------------------------------------------------------------------------


def test_finalize_fills_header_fields() -> None:
    rec = StepTraceRecorder(_header(t_onset=None, n_steps=0), _trace_cfg())
    _record_step(rec, 0)
    _record_step(rec, 1)
    trace = rec.finalize(
        t_hazard_end=33,
        disturbance_triggered=True,
        hazard_persists=False,
        t_onset=12,
    )
    assert trace.header.n_steps == 2
    assert trace.header.t_hazard_end == 33
    assert trace.header.disturbance_triggered is True
    assert trace.header.hazard_persists is False
    assert trace.header.t_onset == 12
    assert trace.header.stride == 1
    assert trace.header.trace_schema_version == TRACE_SCHEMA_VERSION


def test_finalize_twice_raises() -> None:
    rec = StepTraceRecorder(_header(), _trace_cfg())
    _record_step(rec, 0)
    _finalize(rec)
    with pytest.raises(RuntimeError, match="finalize"):
        _finalize(rec)


# ---------------------------------------------------------------------------
# zero-side-effect source constraints
# ---------------------------------------------------------------------------


def test_recorder_source_has_no_rng_or_backend_imports() -> None:
    tree = ast.parse(RECORDER_SRC.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
                imported.add(node.module)
    assert "random" not in imported
    assert "numpy" not in imported
    assert "np" not in imported
    assert not any(name.startswith("dynisafety.backends") for name in imported)
    src = RECORDER_SRC.read_text(encoding="utf-8")
    assert "np.random" not in src
    assert "random." not in src


# ---------------------------------------------------------------------------
# volume (used by the handoff report; not a pass/fail threshold)
# ---------------------------------------------------------------------------


def test_five_hundred_step_trace_size(tmp_path: Path) -> None:
    rec = StepTraceRecorder(_header(episode_index=0), _trace_cfg())
    action = [0.01, -0.02, 0.0, 0.0, 0.0, 0.1, -1.0]
    for t in range(500):
        rec.record(
            t=t,
            eef_pos=(0.4 + 0.001 * t, 0.01 * (t % 7), 0.2),
            eef_quat=(0.0, 0.0, 0.0, 1.0),
            action=action,
            obstacle_pos=(0.5, 0.0, 0.15) if t >= 20 else None,
            in_contact=t % 17 == 0,
            force_n=float(t % 17),
            impulse_ns=0.01 * (t % 17),
            phase=DisturbancePhase.ACTIVE if t >= 20 else DisturbancePhase.INACTIVE,
        )
    trace = rec.finalize(
        t_hazard_end=80,
        disturbance_triggered=True,
        hazard_persists=True,
    )
    raw_dir = tmp_path / "raw"
    gz_dir = tmp_path / "gz"
    raw_rel = TraceWriter(raw_dir, _trace_cfg(fmt="jsonl")).write(0, trace)
    gz_rel = TraceWriter(gz_dir, _trace_cfg(fmt="jsonl_gz")).write(0, trace)
    assert raw_rel is not None and gz_rel is not None
    raw_n = (raw_dir / raw_rel).stat().st_size
    gz_n = (gz_dir / gz_rel).stat().st_size
    sizes = tmp_path / "sizes.json"
    sizes.write_text(json.dumps({"raw": raw_n, "gzip": gz_n, "n_steps": 500}), encoding="utf-8")
    assert raw_n > gz_n
    assert gz_n > 0
    assert read_trace(gz_dir / gz_rel) == trace
