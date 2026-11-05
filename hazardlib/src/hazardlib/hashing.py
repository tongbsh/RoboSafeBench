"""Canonical JSON and sha256 hashing for specs, descriptors and manifests.

Canonical form (``HASH_FORMAT_VERSION`` 1). Two values that are equal under
these rules produce byte-identical text and therefore the same hash:

1. Text is JSON with no whitespace; ``,`` and ``:`` are the only separators.
2. Object keys must be ``str`` and are emitted sorted by Unicode code point, so
   the insertion order of a mapping never matters.
3. Strings are escaped ASCII-only (``json.dumps(..., ensure_ascii=True)``).
4. ``True``/``False``/``None`` -> ``true``/``false``/``null``.
5. Integers (``int``, ``numpy.integer``) are emitted exactly, in decimal.
6. Floats (``float``, ``numpy.floating``, widened to float64 first) must be
   finite (NaN/inf raise ``ValueError``). They are emitted as
   ``'%.{FLOAT_SIG_DIGITS}g' % x`` with ``-0.0`` normalised to ``0.0``. Only the
   first ``FLOAT_SIG_DIGITS`` significant digits matter: a change below that
   resolution does NOT change the hash. ``%g`` drops trailing zeros, so
   ``3.0`` and ``3`` hash identically (for magnitudes below 10**FLOAT_SIG_DIGITS).
7. ``list``, ``tuple`` and ``numpy.ndarray`` are emitted as JSON arrays, in order.
8. Objects with a ``to_dict()`` method (records) are hashed via that dict.
9. Anything else (sets, bytes, arbitrary objects) raises ``TypeError``.

``FLOAT_SIG_DIGITS`` and ``HASH_FORMAT_VERSION`` define the hash *format*, like
a physical constant; they are not tunable thresholds. Changing either changes
every hash ever produced, so ``tests/test_hashing.py`` pins a golden digest.
"""
import hashlib
import json
import math
import numbers
from typing import Any

import numpy as np

HASH_FORMAT_VERSION = 1
FLOAT_SIG_DIGITS = 12
_FLOAT_FORMAT = "%." + str(FLOAT_SIG_DIGITS) + "g"


def _emit(obj: Any, path: str) -> str:
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _emit(obj.to_dict(), path)
    if obj is None:
        return "null"
    if isinstance(obj, (bool, np.bool_)):
        return "true" if obj else "false"
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=True)
    if isinstance(obj, numbers.Integral):
        return str(int(obj))
    if isinstance(obj, numbers.Real):
        number = float(obj)
        if not math.isfinite(number):
            raise ValueError("cannot hash non-finite number %r at %s" % (number, path or "<root>"))
        if number == 0.0:
            number = 0.0  # normalise -0.0
        return _FLOAT_FORMAT % number
    if isinstance(obj, np.ndarray):
        return _emit(obj.tolist(), path)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_emit(v, "%s[%d]" % (path, i)) for i, v in enumerate(obj)) + "]"
    if isinstance(obj, dict):
        parts = []
        for key in obj:
            if not isinstance(key, str):
                raise TypeError("cannot hash mapping with non-string key %r at %s"
                                % (key, path or "<root>"))
        for key in sorted(obj):
            parts.append(json.dumps(key, ensure_ascii=True) + ":"
                         + _emit(obj[key], "%s.%s" % (path, key) if path else key))
        return "{" + ",".join(parts) + "}"
    raise TypeError("cannot hash value of type %s at %s" % (type(obj).__name__, path or "<root>"))


def canonical_json(obj: Any) -> str:
    """Return the canonical JSON text of ``obj`` (rules in the module docstring)."""
    return _emit(obj, "")


def sha256_hex(obj: Any) -> str:
    """Lower-case hex sha256 of the UTF-8 canonical JSON of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
