"""Small statistics shared by the generator and its hosts (numpy only)."""
import numpy as np


def distribution(values, quantiles):
    """``{"n", "min", "max", "quantiles": {str(q): value}}`` of the finite values (``n == 0``: empty)."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "min": None, "max": None, "quantiles": {}}
    return {"n": int(len(values)), "min": float(values.min()), "max": float(values.max()),
            "quantiles": {str(q): float(np.quantile(values, q)) for q in quantiles}}
