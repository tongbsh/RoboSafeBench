"""VLA policy registry for DyniSafety-LIBERO benchmark.

The registry decouples the benchmark runner from any specific VLA model.
To plug in a new model, register a :class:`VLAPolicy` subclass and pass
``--model_family <name>`` on the command line — no benchmark code needs to
change.

Quick-start
-----------
Create a file (e.g. ``my_policy.py``) alongside the evaluation script::

    import numpy as np
    from dynisafety_libero.policies import register_policy, VLAPolicy

    class MyPolicy(VLAPolicy):
        def __init__(self, cfg):
            # load your model here
            ...

        def predict_actions(self, obs, task_description):
            # obs has keys: full_image, wrist_image, state
            # return a list of shape-(7,) numpy arrays
            ...

        @property
        def action_chunk_size(self):
            return 16  # or however many steps your model outputs

    register_policy("my_model", MyPolicy)

Then import your file before calling ``eval_dynisafety`` (or add an import at
the top of ``run_dynisafety_eval.py``) and run with
``--model_family my_model``.

Built-in policies
-----------------
* ``"openvla"`` / ``"openvla_oft"`` – :class:`~.openvla_policy.OpenVLAPolicy`
  (same class; ``openvla_oft`` is an alias for dedicated eval outputs).
  Loaded lazily on first :func:`create_policy` call so heavy deps are only
  imported when actually needed.
* ``"pi0"`` / ``"pi05"`` – :class:`~.pi0_policy.Pi0Policy`
  WebSocket client that connects to a running openpi ``serve_policy.py``
  server.  Start the server separately in the openpi conda/uv environment
  before launching the benchmark (see ``scripts/start_pi0_server.sh``).
* ``"memoryvla"`` – :class:`~.memoryvla_policy.MemoryVLAPolicy`
  HTTP client to MemoryVLA FastAPI server (``vlaModelHome/MemoryVLA/server.py``).
  Start the server with ``scripts/start_memoryvla_server.sh`` in the memvla env.
* ``"spatialvla"`` – :class:`~.spatialvla_policy.SpatialVLAPolicy`
  HTTP client to SpatialVLA FastAPI server (``vlaModelHome/SpatialVLA/server.py``).
  Start the server with ``scripts/start_spatialvla_server.sh`` in the spatialvla env.
* ``"xvla"`` – :class:`~.xvla_policy.XVLAPolicy`
  HTTP client to X-VLA FastAPI server (``vlaModelHome/X-VLA/deploy.py``).
  Start the server with ``scripts/start_xvla_server.sh`` in the XVLA env.
* ``"starvla"`` – :class:`~.starvla_policy.StarVLAPolicy`
  WebSocket client to StarVLA ``server_policy`` (``vlaModelHome/starVLA/deployment/``).
  Start the server with ``scripts/starvla/start_starvla_server.sh`` in the StarVLA env.
* ``"gr00t"`` – :class:`~.gr00t_policy.Gr00tPolicy`
  ZeroMQ client to Isaac-GR00T ``PolicyServer`` (``vlaModelHome/Isaac-GR00T/gr00t/eval/run_gr00t_server.py``).
  Start the server with ``scripts/gr00t/start_gr00t_server.sh`` in the GR00T env (``LIBERO_PANDA`` + sim wrapper).
* ``"univla_local"`` – :class:`~.univla_local_policy.UniVLALocalPolicy`
  In-process UniVLA (latent action + decoder). Repo default ``vlaModelHome/UniVLA``; set ``UNIVLA_ROOT`` if needed.
  See ``scripts/openvla-oft/run_dynisafety_univla_eval.py``.
"""

from .base_policy import VLAPolicy

_REGISTRY: dict = {}

# Model families that are served via the openpi WebSocket server.
_PI0_FAMILIES = frozenset({"pi0", "pi05"})


def register_policy(model_family: str, cls: type) -> None:
    """Register *cls* as the policy for *model_family*.

    Can be called multiple times with the same key to override a registration
    (e.g. for testing).
    """
    _REGISTRY[model_family] = cls


def create_policy(cfg) -> VLAPolicy:
    """Instantiate and return the :class:`VLAPolicy` for ``cfg.model_family``.

    Built-in policies are loaded lazily on the first call so their heavyweight
    dependencies (``torch``, ``prismatic``, …) are only imported when needed.

    Parameters
    ----------
    cfg:
        Configuration dataclass with at least a ``model_family`` field.

    Raises
    ------
    ValueError
        If ``cfg.model_family`` has not been registered.
    """
    if not _REGISTRY:
        # Lazily register all built-in policies.
        # Each import is guarded so that a missing optional dependency in one
        # policy does not prevent other policies from being registered.
        try:
            from .openvla_policy import OpenVLAPolicy  # noqa: PLC0415
            _REGISTRY["openvla"] = OpenVLAPolicy
            _REGISTRY["openvla_oft"] = OpenVLAPolicy
        except Exception as _e:  # noqa: BLE001
            import warnings; warnings.warn(f"openvla/openvla_oft policy unavailable: {_e}")

        try:
            from .pi0_policy import Pi0Policy  # noqa: PLC0415
            for name in _PI0_FAMILIES:
                _REGISTRY[name] = Pi0Policy
        except Exception as _e:  # noqa: BLE001
            import warnings; warnings.warn(f"pi0/pi05 policy unavailable: {_e}")

        try:
            from .memoryvla_policy import MemoryVLAPolicy  # noqa: PLC0415
            _REGISTRY["memoryvla"] = MemoryVLAPolicy
        except Exception as _e:  # noqa: BLE001
            import warnings; warnings.warn(f"memoryvla policy unavailable: {_e}")

        try:
            from .spatialvla_policy import SpatialVLAPolicy  # noqa: PLC0415
            _REGISTRY["spatialvla"] = SpatialVLAPolicy
        except Exception as _e:  # noqa: BLE001
            import warnings; warnings.warn(f"spatialvla policy unavailable: {_e}")

        try:
            from .xvla_policy import XVLAPolicy  # noqa: PLC0415
            _REGISTRY["xvla"] = XVLAPolicy
        except Exception as _e:  # noqa: BLE001
            import warnings; warnings.warn(f"xvla policy unavailable: {_e}")

        try:
            from .starvla_policy import StarVLAPolicy  # noqa: PLC0415
            _REGISTRY["starvla"] = StarVLAPolicy
        except Exception as _e:  # noqa: BLE001
            import warnings; warnings.warn(f"starvla policy unavailable: {_e}")

        try:
            from .gr00t_policy import Gr00tPolicy  # noqa: PLC0415
            _REGISTRY["gr00t"] = Gr00tPolicy
        except Exception as _e:  # noqa: BLE001
            import warnings; warnings.warn(f"gr00t policy unavailable: {_e}")

        try:
            from .univla_local_policy import UniVLALocalPolicy  # noqa: PLC0415
            _REGISTRY["univla_local"] = UniVLALocalPolicy
        except Exception as _e:  # noqa: BLE001
            import warnings; warnings.warn(f"univla_local policy unavailable: {_e}")

    if cfg.model_family not in _REGISTRY:
        raise ValueError(
            f"Unknown model_family '{cfg.model_family}'. "
            f"Registered policies: {sorted(_REGISTRY)}. "
            "Call register_policy() before create_policy() to add a new model."
        )
    return _REGISTRY[cfg.model_family](cfg)


__all__ = ["VLAPolicy", "register_policy", "create_policy"]
