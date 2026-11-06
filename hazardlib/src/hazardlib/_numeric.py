"""Argument validation for the numeric modules (kinematics, calibrate).

Every helper raises ``ValueError`` with the argument name in the message.
Nothing here supplies a default value: callers always pass every number.
"""
import numbers
from typing import Any

import numpy as np
import math


def finite_float(name: str, value: Any) -> float:
    """``value`` as a finite Python float. Rejects bool, str, bytes and non-scalar arrays."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("%s: expected a number, got bool %r" % (name, value))
    if isinstance(value, (str, bytes)):
        raise ValueError("%s: expected a number, got %r" % (name, value))
    if isinstance(value, np.ndarray) and value.ndim != 0:
        raise ValueError("%s: expected a scalar, got an array of shape %s" % (name, value.shape))
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s: expected a finite number, got %r" % (name, value)) from None
    if not math.isfinite(number):
        raise ValueError("%s: expected a finite number, got %r" % (name, number))
    return number


def positive_float(name: str, value: Any) -> float:
    number = finite_float(name, value)
    if not number > 0:
        raise ValueError("%s must be > 0, got %r" % (name, number))
    return number


def non_negative_float(name: str, value: Any) -> float:
    number = finite_float(name, value)
    if not number >= 0:
        raise ValueError("%s must be >= 0, got %r" % (name, number))
    return number


def positive_int(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError("%s: expected an integer, got %r" % (name, value))
    number = int(value)
    if number < 1:
        raise ValueError("%s must be >= 1, got %r" % (name, number))
    return number


def vec3(name: str, value: Any) -> np.ndarray:
    """A fresh finite float64 array of shape (3,)."""
    try:
        arr = np.array(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("%s: cannot convert %r to a float vector" % (name, value)) from None
    if arr.shape != (3,):
        raise ValueError("%s: expected shape (3,), got %s" % (name, arr.shape))
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s: contains non-finite values %s" % (name, arr))
    return arr


def unit_vec3(name: str, value: Any) -> np.ndarray:
    """``value`` normalised to unit length; a zero vector is an error."""
    arr = vec3(name, value)
    norm = float(np.linalg.norm(arr))
    if not norm > 0:
        raise ValueError("%s must be a non-zero vector, got %s" % (name, arr))
    return arr / norm


def finite_array(name: str, value: Any) -> np.ndarray:
    """A float64 array (any shape) with only finite entries."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("%s: cannot convert %r to a float array" % (name, value)) from None
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s: contains non-finite values" % name)
    return arr
