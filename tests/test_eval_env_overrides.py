"""Env-injected eval overrides for debug sweeps (video / diversity / pack)."""

from __future__ import annotations

from argparse import Namespace

from dynisafety.cli.eval import build_eval_overrides


def _args(**kwargs) -> Namespace:
    base = dict(
        overrides=None,
        seed=None,
        episodes=None,
        no_video=False,
        output_root=None,
        backend="libero",
        level=2,
    )
    base.update(kwargs)
    return Namespace(**base)


def test_dyni_diversity_and_pack_libero(monkeypatch) -> None:
    monkeypatch.setenv("DYNI_DIVERSITY", "1")
    monkeypatch.setenv("DYNI_OBSTACLE_PACK", "1")
    monkeypatch.delenv("DYNI_SAVE_VIDEO", raising=False)
    items = build_eval_overrides(_args(backend="libero", level=1))
    assert "extra.libero.l2.spawn_diversity.enabled=true" in items
    assert "extra.libero.l3.reach_diversity.enabled=true" in items
    assert "extra.libero.obstacle_pack.enabled=true" in items


def test_dyni_diversity_simplerenv_disables_align(monkeypatch) -> None:
    monkeypatch.setenv("DYNI_DIVERSITY", "1")
    monkeypatch.delenv("DYNI_OBSTACLE_PACK", raising=False)
    items = build_eval_overrides(_args(backend="simplerenv", level=2))
    assert "extra.simplerenv.l2.spawn_diversity.enabled=true" in items
    assert "disturbance.l2.align_start_y_to_eef=false" in items
    assert "extra.libero.obstacle_pack.enabled=true" not in items


def test_dyni_flags_unset_do_not_inject(monkeypatch) -> None:
    monkeypatch.delenv("DYNI_DIVERSITY", raising=False)
    monkeypatch.delenv("DYNI_OBSTACLE_PACK", raising=False)
    monkeypatch.delenv("DYNI_SAVE_VIDEO", raising=False)
    monkeypatch.delenv("DYNI_CAPTURE", raising=False)
    monkeypatch.delenv("DYNI_CAPTURE_ROOT", raising=False)
    items = build_eval_overrides(_args(backend="libero", level=2))
    assert not any("spawn_diversity.enabled=" in x for x in items)
    assert not any("obstacle_pack.enabled=" in x for x in items)
    assert not any(x.startswith("capture.enabled=") for x in items)


def test_dyni_diversity_and_pack_off_libero(monkeypatch) -> None:
    monkeypatch.setenv("DYNI_DIVERSITY", "0")
    monkeypatch.setenv("DYNI_OBSTACLE_PACK", "0")
    monkeypatch.delenv("DYNI_SAVE_VIDEO", raising=False)
    items = build_eval_overrides(_args(backend="libero", level=2))
    assert "extra.libero.l2.spawn_diversity.enabled=false" in items
    assert "extra.libero.l3.reach_diversity.enabled=false" in items
    assert "extra.libero.obstacle_pack.enabled=false" in items


def test_dyni_capture_enables_and_optional_root(monkeypatch) -> None:
    monkeypatch.setenv("DYNI_CAPTURE", "1")
    monkeypatch.setenv("DYNI_CAPTURE_ROOT", "/tmp/wrm-capture")
    monkeypatch.delenv("DYNI_SAVE_VIDEO", raising=False)
    items = build_eval_overrides(_args(backend="libero", level=0))
    assert "capture.enabled=true" in items
    assert "capture.dataset_root=/tmp/wrm-capture" in items
