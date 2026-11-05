import copy
import json

import numpy as np
import pytest

from hazardlib import hashing
from hazardlib.hashing import canonical_json, sha256_hex
from hazardlib.spec import BenchmarkDescriptor, HazardInstance, HazardSpec, Manifest, TaskContext

import conftest


def leaf_paths(obj, path=()):
    """Paths (tuples of keys/indexes) of every leaf of a nested dict/list."""
    if isinstance(obj, dict):
        return [p for k in obj for p in leaf_paths(obj[k], path + (k,))]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in leaf_paths(v, path + (i,))]
    return [path]


def _set(doc, path, value):
    doc = copy.deepcopy(doc)
    cur = doc
    for part in path[:-1]:
        cur = cur[part]
    cur[path[-1]] = value
    return doc


def _get(doc, path):
    cur = doc
    for part in path:
        cur = cur[part]
    return cur


def _perturb(value):
    if value is None:
        return 1.0
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.5 + 1.0
    if isinstance(value, str):
        return value + "x"
    raise AssertionError("unexpected leaf %r" % (value,))


def reverse_keys(obj):
    if isinstance(obj, dict):
        return {k: reverse_keys(obj[k]) for k in reversed(list(obj))}
    if isinstance(obj, list):
        return [reverse_keys(v) for v in obj]
    return obj


DOCS = {
    "hazard_spec": conftest.hazard_spec_dict,
    "task_context": conftest.task_context_dict,
    "instance": conftest.hazard_instance_dict,
    "manifest": conftest.manifest_dict,
    "bd": conftest.complete_bd_dict,
}


@pytest.mark.parametrize("name", sorted(DOCS))
def test_key_order_does_not_change_hash(name):
    doc = DOCS[name]()
    assert sha256_hex(doc) == sha256_hex(reverse_keys(doc))
    assert canonical_json(doc) == canonical_json(reverse_keys(doc))


@pytest.mark.parametrize("name", sorted(DOCS))
def test_changing_any_single_leaf_changes_hash(name):
    doc = DOCS[name]()
    base = sha256_hex(doc)
    paths = leaf_paths(doc)
    assert paths, "document has no leaves?"
    for path in paths:
        mutated = _set(doc, path, _perturb(_get(doc, path)))
        assert sha256_hex(mutated) != base, "hash unchanged after editing %s" % (path,)


@pytest.mark.parametrize("name", sorted(DOCS))
def test_adding_or_removing_a_key_changes_hash(name):
    doc = DOCS[name]()
    base = sha256_hex(doc)
    extra = copy.deepcopy(doc)
    extra["zz_extra"] = 1
    assert sha256_hex(extra) != base
    shrunk = copy.deepcopy(doc)
    shrunk.pop(sorted(shrunk)[0])
    assert sha256_hex(shrunk) != base


def test_list_order_matters():
    assert sha256_hex({"a": [1, 2]}) != sha256_hex({"a": [2, 1]})


def test_records_hash_equals_hash_of_their_dict():
    spec = HazardSpec.from_dict(conftest.hazard_spec_dict())
    assert spec.hash() == sha256_hex(spec.to_dict()) == sha256_hex(conftest.hazard_spec_dict())
    for cls, factory in ((TaskContext, conftest.task_context_dict),
                         (HazardInstance, conftest.hazard_instance_dict),
                         (Manifest, conftest.manifest_dict),
                         (BenchmarkDescriptor, conftest.complete_bd_dict)):
        record = cls.from_dict(factory())
        assert record.hash() == sha256_hex(factory())


def test_record_hash_changes_when_a_field_changes():
    spec = HazardSpec.from_dict(conftest.hazard_spec_dict())
    before = spec.hash()
    spec.agent.mass_kg = spec.agent.mass_kg + 1.0
    assert spec.hash() != before


def test_numpy_values_hash_like_python_values():
    plain = {"v": [1.5, 2.5], "n": 3, "b": True}
    numpy_version = {"v": np.array([1.5, 2.5]), "n": np.int64(3), "b": np.bool_(True)}
    assert sha256_hex(plain) == sha256_hex(numpy_version)
    assert canonical_json({"f": np.float32(0.5)}) == canonical_json({"f": 0.5})


def test_tuple_equals_list():
    assert sha256_hex({"a": (1, 2)}) == sha256_hex({"a": [1, 2]})


def test_int_and_integral_float_hash_the_same():
    assert sha256_hex({"a": 3}) == sha256_hex({"a": 3.0})


def test_negative_zero_is_normalised():
    assert sha256_hex({"a": -0.0}) == sha256_hex({"a": 0.0})


def test_changes_below_float_resolution_are_ignored_and_above_are_not():
    base = {"a": 0.1}
    assert sha256_hex({"a": 0.1 + 1e-15}) == sha256_hex(base)
    assert sha256_hex({"a": 0.1 * (1 + 1e-6)}) != sha256_hex(base)


def test_bool_is_not_confused_with_int():
    assert sha256_hex({"a": True}) != sha256_hex({"a": 1})


def test_unicode_is_escaped_and_stable():
    text = canonical_json({"ké": "中"})
    assert text == '{"k\\u00e9":"\\u4e2d"}'
    assert sha256_hex({"ké": "中"}) == sha256_hex({"ké": "中"})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_numbers_are_rejected(bad):
    with pytest.raises(ValueError, match="non-finite"):
        sha256_hex({"a": [bad]})


@pytest.mark.parametrize("bad", [{1, 2}, b"bytes", object(), {1: "a"}])
def test_unsupported_values_are_rejected(bad):
    with pytest.raises(TypeError):
        sha256_hex({"a": bad})


def test_canonical_json_is_valid_json_without_whitespace():
    doc = conftest.hazard_spec_dict()
    text = canonical_json(doc)
    assert " " not in text and "\n" not in text  # sample has no spaces inside strings
    assert json.loads(text) == json.loads(json.dumps(doc))


def test_golden_digest_pins_the_hash_format():
    """If this fails the canonical form changed and every stored hash is invalid."""
    doc = {"b": [1, 2.5, None, True, "x"], "a": {"z": 0.1, "y": -0.0}}
    assert canonical_json(doc) == '{"a":{"y":0,"z":0.1},"b":[1,2.5,null,true,"x"]}'
    assert sha256_hex(doc) == "1b5798c8bc5761e3611e57049ee2a023f346a1a0323b433ad79d785e15faeb60"
    assert hashing.HASH_FORMAT_VERSION == 1
