"""Generate the committed toy checkpoint used by the example agent.

``.pt`` / ``.pth`` / ``.bin`` / ``.safetensors`` are gitignored in this repo;
``.npz`` is not — that is why the toy weights use ``.npz``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Fixed seed so re-running this script is a no-op for git.
_SEED = 20260820
_PROJ_IN = 64
_PROJ_OUT = 16  # wide enough for every DyniSafety action space; adapters slice.
_FORMAT_VERSION = 1


def main() -> None:
    out = Path(__file__).resolve().parent / "mock_vla_weights.npz"
    rng = np.random.default_rng(_SEED)
    proj = rng.standard_normal((_PROJ_IN, _PROJ_OUT)).astype(np.float32) * np.float32(0.01)
    scale = rng.uniform(0.25, 0.75, size=_PROJ_OUT).astype(np.float32)
    np.savez(
        out,
        proj=proj,
        scale=scale,
        format_version=np.int32(_FORMAT_VERSION),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
