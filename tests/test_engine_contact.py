"""Fake-object tests for MuJoCo / SAPIEN engine contact adapters."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dynisafety.backends._engine import mujoco_contact as mj
from dynisafety.backends._engine import sapien_contact as sap
from dynisafety.core.contact import CONTACT_IMPULSE_EPS


class _FakePoint:
    def __init__(self, impulse: np.ndarray | None) -> None:
        self.impulse = impulse


def _actor_scene(*contacts: object) -> SimpleNamespace:
    return SimpleNamespace(get_contacts=lambda: list(contacts))


def test_sapien_pair_reader_actor01() -> None:
    scene = _actor_scene(
        SimpleNamespace(
            actor0=SimpleNamespace(name="obs"),
            actor1=SimpleNamespace(name="link_0"),
            points=[_FakePoint(np.array([3.0, 0.0, 0.0])), _FakePoint(np.array([0.0, 4.0, 0.0]))],
        )
    )
    result = sap.measure_actor_contact(
        scene,
        actor_name="obs",
        target_names={"link_0"},
        pair_reader=sap.read_pair_actor01,
        force_equiv_dt_s=0.05,
        impulse_reduce="sum",
    )
    assert result.in_contact is True
    assert result.contact_without_force is False
    assert result.impulse_ns == pytest.approx(7.0)
    assert result.force_n == pytest.approx(7.0 / 0.05)
    assert result.partners == {"link_0": 1}
    assert result.peak_partner == "link_0"
    assert result.force_equiv_dt_s == pytest.approx(0.05)


def test_sapien_pair_reader_body_entity() -> None:
    body0 = SimpleNamespace(entity=SimpleNamespace(name="hand"))
    body1 = SimpleNamespace(entity=SimpleNamespace(name="link_a"))
    scene = _actor_scene(
        SimpleNamespace(bodies=[body0, body1], points=[_FakePoint(np.array([2.0, 0.0, 0.0]))])
    )
    result = sap.measure_actor_contact(
        scene,
        actor_name="hand",
        target_names={"link_a"},
        pair_reader=sap.read_pair_body_entity,
        force_equiv_dt_s=0.05,
        impulse_reduce="sum",
    )
    assert result.in_contact is True
    assert result.partners == {"link_a": 1}
    assert result.impulse_ns == pytest.approx(2.0)
    assert sap.read_pair_actor01(SimpleNamespace(bodies=[body0, body1])) is None
    assert sap.read_pair_body_entity(SimpleNamespace(actor0=None, actor1=None)) is None


def test_sapien_scheme_b_zero_impulse_is_contact_without_force() -> None:
    scene = _actor_scene(
        SimpleNamespace(
            actor0=SimpleNamespace(name="obs"),
            actor1=SimpleNamespace(name="link_0"),
            points=[_FakePoint(np.array([0.0, 0.0, 0.0]))],
        )
    )
    result = sap.measure_actor_contact(
        scene,
        actor_name="obs",
        target_names={"link_0"},
        pair_reader=sap.read_pair_actor01,
        force_equiv_dt_s=0.05,
        impulse_reduce="sum",
    )
    assert result.in_contact is True
    assert result.contact_without_force is True
    assert result.force_n == pytest.approx(0.0)
    assert result.impulse_ns == pytest.approx(0.0)
    assert result.peak_partner is None


def test_sapien_peak_partner_is_max_impulse_side() -> None:
    scene = _actor_scene(
        SimpleNamespace(
            actor0=SimpleNamespace(name="obs"),
            actor1=SimpleNamespace(name="link_weak"),
            points=[_FakePoint(np.array([1.0, 0.0, 0.0]))],
        ),
        SimpleNamespace(
            actor0=SimpleNamespace(name="link_strong"),
            actor1=SimpleNamespace(name="obs"),
            points=[_FakePoint(np.array([5.0, 0.0, 0.0]))],
        ),
    )
    result = sap.measure_actor_contact(
        scene,
        actor_name="obs",
        target_names={"link_weak", "link_strong"},
        pair_reader=sap.read_pair_actor01,
        force_equiv_dt_s=0.1,
        impulse_reduce="sum",
    )
    assert result.partners == {"link_weak": 1, "link_strong": 1}
    assert result.peak_partner == "link_strong"
    assert result.impulse_ns == pytest.approx(5.0)


def test_sapien_actor_has_contact_require_impulse_modes() -> None:
    zero = _actor_scene(
        SimpleNamespace(
            actor0=SimpleNamespace(name="obs"),
            actor1=SimpleNamespace(name="link_0"),
            points=[_FakePoint(np.array([0.0, 0.0, 0.0]))],
        )
    )
    kicked = _actor_scene(
        SimpleNamespace(
            actor0=SimpleNamespace(name="obs"),
            actor1=SimpleNamespace(name="link_0"),
            points=[
                _FakePoint(np.array([3.0, 0.0, 0.0])),
                _FakePoint(np.array([0.0, 4.0, 0.0])),
            ],
        )
    )
    kwargs = {
        "actor_name": "obs",
        "target_names": {"link_0"},
        "pair_reader": sap.read_pair_actor01,
    }
    assert sap.actor_has_contact(zero, require_impulse=False, **kwargs) is True
    assert sap.actor_has_contact(zero, require_impulse=True, **kwargs) is False
    assert sap.actor_has_contact(kicked, require_impulse=True, **kwargs) is True
    assert CONTACT_IMPULSE_EPS < 7.0


def test_sapien_scene_timestep_probe_and_fallback() -> None:
    scene = SimpleNamespace(get_timestep=lambda: 0.004)
    assert sap.scene_timestep(scene) == pytest.approx(0.004)
    empty = SimpleNamespace()
    assert sap.scene_timestep(empty, control_freq=20) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="control_freq"):
        sap.scene_timestep(empty, control_freq=0)
    with pytest.raises(ValueError, match="timestep"):
        sap.scene_timestep(empty)


# ---------------------------------------------------------------------------
# MuJoCo (fake sim + monkeypatched mj_contactForce)
# ---------------------------------------------------------------------------


class _FakeMjContact:
    def __init__(self, geom1: int, geom2: int) -> None:
        self.geom1 = geom1
        self.geom2 = geom2


class _FakeMjModel:
    def __init__(
        self,
        *,
        nbody: int,
        ngeom: int,
        body_parentid: list[int],
        geom_bodyid: list[int],
        body_names: dict[int, str | None],
        name_to_id: dict[str, int] | None = None,
        geom_contype: list[int] | None = None,
        geom_conaffinity: list[int] | None = None,
    ) -> None:
        self.nbody = nbody
        self.ngeom = ngeom
        self.body_parentid = body_parentid
        self.geom_bodyid = geom_bodyid
        self._body_names = body_names
        self._name_to_id = name_to_id or {}
        if geom_contype is not None:
            self.geom_contype = geom_contype
        if geom_conaffinity is not None:
            self.geom_conaffinity = geom_conaffinity
        self._model = self

    def body_id2name(self, body_id: int) -> str | None:
        return self._body_names.get(int(body_id))

    def body_name2id(self, name: str) -> int:
        if name not in self._name_to_id:
            raise KeyError(name)
        return self._name_to_id[name]


class _FakeMjData:
    def __init__(self, contacts: list[_FakeMjContact], body_xquat: dict[int, np.ndarray]) -> None:
        self.contact = contacts
        self.ncon = len(contacts)
        self.body_xquat = body_xquat
        self._data = self


class _FakeSim:
    def __init__(self, model: _FakeMjModel, data: _FakeMjData) -> None:
        self.model = model
        self.data = data


def _patch_mj_force(monkeypatch: pytest.MonkeyPatch, forces: dict[int, float]) -> None:
    def _fake(raw_model: object, raw_data: object, index: int, force_buf: np.ndarray) -> None:
        force_buf[:] = 0.0
        force_buf[0] = float(forces.get(index, 0.0))

    monkeypatch.setattr(mj, "_mj_contact_force", _fake)


def test_mujoco_pairing_partners_and_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeMjModel(
        nbody=4,
        ngeom=4,
        body_parentid=[-1, 0, 0, 1],
        geom_bodyid=[0, 1, 2, 3],
        body_names={0: "robot0_link", 1: "robot0_hand", 2: "box", 3: "lid"},
    )
    data = _FakeMjData(
        [
            _FakeMjContact(0, 2),
            _FakeMjContact(3, 1),
            _FakeMjContact(0, 1),
        ],
        body_xquat={},
    )
    _patch_mj_force(monkeypatch, {0: 3.0, 1: 10.0, 2: 99.0})
    result = mj.measure_geom_pair_contact(
        _FakeSim(model, data),
        side_a_geom_ids={0, 1},
        side_b_geom_ids={2, 3},
        force_equiv_dt_s=0.05,
        impulse_reduce="sum",
    )
    assert result.in_contact is True
    assert result.partners == {"box": 1, "lid": 1}
    assert result.peak_partner == "lid"
    assert result.force_n == pytest.approx(10.0)
    assert result.impulse_ns == pytest.approx(0.5)
    assert result.contact_without_force is False
    assert result.force_equiv_dt_s == pytest.approx(0.05)


def test_mujoco_unnamed_body_uses_geom_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeMjModel(
        nbody=2,
        ngeom=2,
        body_parentid=[-1, 0],
        geom_bodyid=[0, 1],
        body_names={0: "robot0_link", 1: None},
    )
    data = _FakeMjData([_FakeMjContact(0, 1)], body_xquat={})
    _patch_mj_force(monkeypatch, {0: 2.0})
    result = mj.measure_geom_pair_contact(
        _FakeSim(model, data),
        side_a_geom_ids={0},
        side_b_geom_ids={1},
        force_equiv_dt_s=0.1,
        impulse_reduce="max",
    )
    assert result.partners == {"geom:1": 1}
    assert result.peak_partner == "geom:1"


def test_mujoco_empty_side_b_early_return(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    def _boom(*_a: object, **_k: object) -> None:
        called["n"] += 1
        raise AssertionError("mj_contactForce must not run when side_b is empty")

    monkeypatch.setattr(mj, "_mj_contact_force", _boom)
    model = _FakeMjModel(
        nbody=1,
        ngeom=1,
        body_parentid=[-1],
        geom_bodyid=[0],
        body_names={0: "robot0_link"},
    )
    data = _FakeMjData([_FakeMjContact(0, 0)], body_xquat={})
    result = mj.measure_geom_pair_contact(
        _FakeSim(model, data),
        side_a_geom_ids={0},
        side_b_geom_ids=set(),
        force_equiv_dt_s=0.05,
        impulse_reduce="sum",
    )
    assert result.in_contact is False
    assert result.partners == {}
    assert result.force_n == 0.0
    assert result.peak_partner is None
    assert called["n"] == 0


def test_mujoco_zero_force_contact_without_force(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeMjModel(
        nbody=2,
        ngeom=2,
        body_parentid=[-1, 0],
        geom_bodyid=[0, 1],
        body_names={0: "arm", 1: "obs"},
    )
    data = _FakeMjData([_FakeMjContact(0, 1)], body_xquat={})
    _patch_mj_force(monkeypatch, {0: 0.0})
    result = mj.measure_geom_pair_contact(
        _FakeSim(model, data),
        side_a_geom_ids={0},
        side_b_geom_ids={1},
        force_equiv_dt_s=0.05,
        impulse_reduce="sum",
    )
    assert result.in_contact is True
    assert result.contact_without_force is True
    assert result.peak_partner is None


def test_mujoco_expand_body_subtree_direct_children_only() -> None:
    model = _FakeMjModel(
        nbody=4,
        ngeom=0,
        body_parentid=[-1, 0, 1, 1],
        geom_bodyid=[],
        body_names={},
    )
    assert mj.expand_body_subtree(model, [0]) == {0, 1}
    assert mj.expand_body_subtree(model, [1]) == {1, 2, 3}


def test_mujoco_expand_body_descendants_includes_grandchildren() -> None:
    model = _FakeMjModel(
        nbody=4,
        ngeom=0,
        body_parentid=[-1, 0, 1, 1],
        geom_bodyid=[],
        body_names={},
    )
    assert mj.expand_body_descendants(model, [0]) == {0, 1, 2, 3}
    assert mj.expand_body_descendants(model, [1]) == {1, 2, 3}


def test_mujoco_collect_geom_helpers() -> None:
    model = _FakeMjModel(
        nbody=3,
        ngeom=3,
        body_parentid=[-1, 0, 0],
        geom_bodyid=[0, 1, 2],
        body_names={0: "robot0_arm", 1: "robot0_hand", 2: "object"},
        geom_contype=[1, 0, 1],
        geom_conaffinity=[0, 0, 1],
    )
    assert mj.collect_geom_ids_for_bodies(model, {0, 1}) == {0, 1}
    assert mj.collect_geom_ids_by_body_prefix(model, prefixes=("robot0_",)) == {0}
    assert mj.collect_geom_ids_by_body_prefix(
        model, prefixes=("robot0_",), require_collision=False
    ) == {0, 1}


def test_mujoco_body_quat() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    model = _FakeMjModel(
        nbody=1,
        ngeom=0,
        body_parentid=[-1],
        geom_bodyid=[],
        body_names={0: "box"},
        name_to_id={"box": 0},
    )
    data = _FakeMjData([], body_xquat={0: q})
    sim = _FakeSim(model, data)
    assert mj.body_quat(sim, "box") == pytest.approx(q)
    with pytest.raises(KeyError, match="unknown body"):
        mj.body_quat(sim, "missing")
