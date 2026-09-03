"""Optional WRM episode-package sidecar for eval rollouts.

``CaptureSession.create`` is the only constructor backends should call.

* ``cfg.capture`` missing or ``enabled=False`` → disabled no-op; no directories
  are created and ``wants_images`` is False.
* Enabled → same ``env.step`` loop copies extra fields and writes a
  ``robosafe_capture_1.0`` package at ``finalize``. Fail-soft: a capture
  error never changes obs / action / control flow and never aborts eval.
* Packages are **not** written into ``EpisodeRecord`` / ``episodes.jsonl``.

See ``configs/base.yaml → capture.*``. Default is off.
"""

from __future__ import annotations

from dynisafety.capture.session import CaptureSession

__all__ = ["CaptureSession"]
