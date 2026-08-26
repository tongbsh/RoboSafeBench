"""Contract tests for M0 skeleton (no simulator)."""

from __future__ import annotations

import json
import warnings
from dataclasses import fields
from pathlib import Path

import pytest

from dynisafety.config.schema import (
    MISSING,
    ContactConfig,
    MetricsConfig,
    OutputConfig,
    RunConfig,
    assert_no_embedded_defaults,
    iter_config_dataclasses,
)
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.core.registry import Registry, RegistryLoadError
from dynisafety.core.writer import RunWriter, build_manifest


def _sample_record(episode_index: int = 0) -> EpisodeRecord:
    return EpisodeRecord(
        schema_version=SCHEMA_VERSION,
        run_id="libero-dummy-smoke-l1-s0-20260101T000000",
        config_hash="abc123def456",
        backend="libero",
        suite="smoke",
        task_id=0,
        task_name="task0",
        instruction="pick up the object",
        model="dummy",
        checkpoint="/ckpt",
        level=1,
        seed=0,
        episode_index=episode_index,
        steps=10,
        max_steps=50,
        termination="success",
        wall_time_s=1.25,
        success=True,
        success_step=8,
        contact_steps=2,
        contact_events=1,
        first_contact_step=3,
        contact_partners={"link_a": 2},
        peak_force_n=1.5,
        peak_impulse_ns=0.01,
        cumulative_impulse_ns=0.02,
        force_equiv_dt_s=0.05,
        max_tilt_deg=12.0,
        toppled=False,
        video_path=None,
        extra={"note": "unit"},
    )


def test_episode_record_round_trip() -> None:
    rec = _sample_record()
    restored = EpisodeRecord.from_json_dict(rec.to_json_dict())
    assert restored == rec


def test_episode_record_rejects_schema_1x() -> None:
    data = _sample_record().to_json_dict()
    data["schema_version"] = "1.0"
    with pytest.raises(ValueError, match="docs/user/migration.md"):
        EpisodeRecord.from_json_dict(data)


def test_episode_record_rejects_unknown_and_missing_keys() -> None:
    data = _sample_record().to_json_dict()
    with pytest.raises(KeyError, match="unknown"):
        EpisodeRecord.from_json_dict({**data, "nope": 1})
    data.pop("success")
    with pytest.raises(KeyError, match="missing"):
        EpisodeRecord.from_json_dict(data)


def test_registry_register_get_available() -> None:
    reg: Registry[object] = Registry("policy")

    @reg.register("dummy")
    class Dummy:
        pass

    assert reg.get("dummy") is Dummy
    assert reg.available() == ["dummy"]
    with pytest.raises(KeyError, match="already registered"):
        reg.register("dummy")(Dummy)
    with pytest.raises(KeyError, match="available"):
        reg.get("missing")


def test_registry_load_entry_fails_loudly_without_warning() -> None:
    reg: Registry[object] = Registry("backend")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RegistryLoadError) as ei:
            reg.load_entry("x", "totally_missing_mod_for_dynisafety:Cls")
    assert not caught
    assert "totally_missing_mod_for_dynisafety" in ei.value.missing_dependencies


def test_run_writer_crash_safe_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    writer = RunWriter(run_dir)
    writer.create()
    for i in range(36):
        writer.append_episode(_sample_record(i))
    writer.close()

    lines = (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 36
    for i, line in enumerate(lines):
        rec = EpisodeRecord.from_json_dict(json.loads(line))
        assert rec.episode_index == i
    assert not (run_dir / "videos").exists()


def test_build_manifest_has_frozen_keys() -> None:
    manifest = build_manifest(
        config_hash="deadbeefcafe",
        schema_version=SCHEMA_VERSION,
        git_sha="abc",
        git_dirty=False,
        submodules={},
        gpu_name=None,
    )
    required = {
        "git_sha",
        "git_dirty",
        "submodules",
        "python_version",
        "packages",
        "gpu_name",
        "config_hash",
        "schema_version",
    }
    assert set(manifest) == required


def test_schema_fields_use_missing_sentinel_only() -> None:
    # Reflection guard for iron law 1.
    assert_no_embedded_defaults()
    for cls in iter_config_dataclasses():
        for f in fields(cls):
            assert f.default is MISSING, f"{cls.__name__}.{f.name}"


def test_schema_classes_construct_with_missing() -> None:
    run = RunConfig()
    assert run.backend is MISSING
    assert run.contact is MISSING
    assert isinstance(ContactConfig(), ContactConfig)
    assert isinstance(MetricsConfig(), MetricsConfig)
    assert isinstance(OutputConfig(), OutputConfig)
