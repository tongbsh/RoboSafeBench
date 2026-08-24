"""Shared HTTP transport for remote VLA servers (M4)."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

# Infrastructure poll cadence (not a model hyperparameter; not taken from YAML).
_POLL_INTERVAL_S = 1.0


class HttpTransport:
    """JSON HTTP client: health / reset / act.

    All timeouts and paths are constructor-required (iron law 1 — no business
    defaults in code). Callers obtain values from resolved model YAML.
    """

    def __init__(
        self,
        *,
        base_url: str,
        health_path: str,
        startup_timeout_s: int,
        act_timeout_s: int,
        reset_path: str,
        act_path: str,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(f"base_url must be a non-empty str, got {base_url!r}")
        if not isinstance(health_path, str) or not health_path.strip():
            raise ValueError(f"health_path must be a non-empty str, got {health_path!r}")
        if not isinstance(reset_path, str) or not reset_path.strip():
            raise ValueError(f"reset_path must be a non-empty str, got {reset_path!r}")
        if not isinstance(act_path, str) or not act_path.strip():
            raise ValueError(f"act_path must be a non-empty str, got {act_path!r}")
        if isinstance(startup_timeout_s, bool) or not isinstance(startup_timeout_s, int):
            raise ValueError(f"startup_timeout_s must be int, got {startup_timeout_s!r}")
        if startup_timeout_s < 1:
            raise ValueError(f"startup_timeout_s must be >= 1, got {startup_timeout_s}")
        if isinstance(act_timeout_s, bool) or not isinstance(act_timeout_s, int):
            raise ValueError(f"act_timeout_s must be int, got {act_timeout_s!r}")
        if act_timeout_s < 1:
            raise ValueError(f"act_timeout_s must be >= 1, got {act_timeout_s}")

        self._base_url = base_url.rstrip("/") + "/"
        self._health_url = urljoin(self._base_url, health_path.lstrip("/"))
        self._reset_url = urljoin(self._base_url, reset_path.lstrip("/"))
        self._act_url = urljoin(self._base_url, act_path.lstrip("/"))
        self._startup_timeout_s = startup_timeout_s
        self._act_timeout_s = act_timeout_s
        self._closed = False

    def wait_until_ready(self) -> None:
        """Block until ``health`` returns HTTP 200 or *startup_timeout_s* elapses."""
        self._ensure_open()
        deadline = time.monotonic() + self._startup_timeout_s
        logger.info("HttpTransport: waiting for %s …", self._health_url)
        while time.monotonic() < deadline:
            if self.health():
                logger.info("HttpTransport: ready at %s", self._base_url)
                return
            time.sleep(_POLL_INTERVAL_S)
        raise RuntimeError(
            f"HTTP server at {self._base_url} not ready within "
            f"{self._startup_timeout_s}s (health={self._health_url})"
        )

    def health(self) -> bool:
        self._ensure_open()
        try:
            # Probe timeout reuses act_timeout_s (no separate default literal).
            resp = requests.get(self._health_url, timeout=self._act_timeout_s)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def reset(self, payload: Mapping[str, Any] | None = None) -> None:
        """POST reset; treat 404/405 as no-op (some servers e.g. X-VLA have no /reset)."""
        self._ensure_open()
        body: Mapping[str, Any] = {} if payload is None else payload
        resp = requests.post(self._reset_url, json=dict(body), timeout=self._act_timeout_s)
        if resp.status_code in (404, 405):
            logger.info(
                "HttpTransport: reset endpoint %s returned %s; treating as no-op",
                self._reset_url,
                resp.status_code,
            )
            return
        resp.raise_for_status()

    def infer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self._ensure_open()
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload)!r}")
        resp = requests.post(self._act_url, json=dict(payload), timeout=self._act_timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, Mapping):
            raise TypeError(
                f"act response must be a JSON object, got {type(data).__name__}"
            )
        return dict(data)

    def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("HttpTransport is closed")
