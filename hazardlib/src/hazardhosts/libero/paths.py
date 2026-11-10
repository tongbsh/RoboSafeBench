"""Where the original LIBERO tree and the LIBERO demonstrations are (task H3).

No machine path lives in the repository. Both locations come from the LIBERO configuration file that the LIBERO
package itself reads at import time (``$LIBERO_CONFIG_PATH/config.yaml``, env_notes/libero.md): its
``benchmark_root`` is ``<tree>/libero/libero`` and its ``datasets`` the directory holding ``libero_spatial/``,
``libero_goal/``, ``libero_10/`` ... with the official ``*_demo.hdf5`` files. The original LIBERO tree and upstream LIBERO
are both packaged as ``libero``; :func:`import_libero` puts the configured tree first on ``sys.path`` and checks that
the imported package is that tree (a second ``libero`` in the same process is an error, env_notes "已知坑" 4).
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from hazardlib.host import CapabilityError

CONFIG_ENV = "LIBERO_CONFIG_PATH"


def libero_config() -> Dict[str, Any]:
    """The LIBERO configuration (``config.yaml`` under ``$LIBERO_CONFIG_PATH``); a missing file is an error here,
    not an interactive prompt inside ``import libero``."""
    root = os.environ.get(CONFIG_ENV)
    if not root:
        raise CapabilityError("the LIBERO host needs $%s (a directory with LIBERO's config.yaml, env_notes/libero.md)"
                              % CONFIG_ENV)
    path = Path(root) / "config.yaml"
    if not path.is_file():
        raise CapabilityError("%s does not exist: LIBERO would ask for it interactively at import time" % path)
    with open(str(path)) as fh:
        data = yaml.safe_load(fh)
    for key in ("benchmark_root", "datasets"):
        if not isinstance(data, dict) or not data.get(key):
            raise CapabilityError("%s has no %r entry" % (path, key))
    return data


def tree_root() -> Path:
    """The original LIBERO checkout: the directory that holds the ``libero`` package (``benchmark_root/../..``)."""
    root = Path(libero_config()["benchmark_root"]).resolve().parent.parent
    if not (root / "libero" / "libero" / "__init__.py").is_file():
        raise CapabilityError("benchmark_root of the LIBERO config does not point into a LIBERO tree: %s" % root)
    return root


def datasets_root() -> Path:
    path = Path(libero_config()["datasets"])
    if not path.is_dir():
        raise CapabilityError("LIBERO datasets directory %s does not exist" % path)
    return path


def import_libero() -> Any:
    """Import the configured original LIBERO ``libero`` package; it must come from :func:`tree_root`."""
    root = tree_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import libero.libero as pkg  # noqa: E402  (after the sys.path insertion)
    got = Path(pkg.__file__).resolve().parent
    if got != (root / "libero" / "libero").resolve():
        raise CapabilityError("`import libero` resolved to %s, not the configured original LIBERO tree %s (two libero "
                              "packages in one process?)" % (got, root))
    return pkg
