"""What the policy saw (R1 item 4): policy-side wrappers and the frame sink interface.

``FrameRecorder`` wraps a ``ReferencePolicy``: every ``act(observation)`` hands the images under the chosen keys
to a :class:`FrameSink`, together with the simulation time of that control step (seconds from the episode start,
read from the host's ``time_state``) and the hazard phase (``parked`` / ``kinematic`` / ``released`` /
``setup_failed``; ``None`` in a control episode). The frames are the observation the policy was given, not a
separate render, and the hazard phase is never passed to the policy (R5).

``FrameSink`` protocol: ``open(episode_id, meta)``, ``add(key, frame, t_s, phase)``, ``close() -> paths`` (a
mapping name -> file path; empty when nothing is written). The core ships an in-memory sink for tests and a
numpy-only ``.npy`` sink; video encoding (mp4) lives outside the core (R1-L, under ``hazardhosts/``) and is loaded
with ``hazardctl run --frame-sink module:factory``. A sink factory is called as ``factory(frames_dir, **args)``.

``TimedPolicy`` measures the wall-clock time of every ``act`` call (``time.perf_counter``). The simulation does
not advance during the call (``EpisodeSession.run_episode`` is synchronous); the times are diagnostics only and
never enter a metric.
"""
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ..driving import describe_policy

FRAME_INDEX_FILE = "frames.json"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


@runtime_checkable
class FrameSink(Protocol):
    def open(self, episode_id: str, meta: Mapping[str, Any]) -> None:
        ...

    def add(self, key: str, frame: Any, t_s: float, phase: Optional[str]) -> None:
        ...

    def close(self) -> Dict[str, str]:
        ...


class _Buffer:
    """Open / add / close bookkeeping shared by the core sinks."""

    def __init__(self) -> None:
        self.episode_id = None  # type: Optional[str]
        self.meta = {}  # type: Dict[str, Any]
        self.frames = {}  # type: Dict[str, List[np.ndarray]]
        self.times = {}  # type: Dict[str, List[float]]
        self.phases = {}  # type: Dict[str, List[Optional[str]]]

    def open(self, episode_id: str, meta: Mapping[str, Any]) -> None:
        if self.episode_id is not None:
            raise RuntimeError("frame sink: open(%r) while episode %r is open" % (episode_id, self.episode_id))
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a non-empty string, got %r" % (episode_id,))
        self.episode_id, self.meta = episode_id, dict(meta)
        self.frames, self.times, self.phases = {}, {}, {}

    def add(self, key: str, frame: Any, t_s: float, phase: Optional[str]) -> None:
        if self.episode_id is None:
            raise RuntimeError("frame sink: add(%r) without an open episode" % (key,))
        self.frames.setdefault(key, []).append(np.array(frame, copy=True))
        self.times.setdefault(key, []).append(float(t_s))
        self.phases.setdefault(key, []).append(phase)

    def _closing(self) -> str:
        if self.episode_id is None:
            raise RuntimeError("frame sink: close() without an open episode")
        episode_id, self.episode_id = self.episode_id, None
        return episode_id


class MemoryFrameSink(_Buffer):
    """Keeps every closed episode in ``episodes[episode_id] = {"meta", "frames", "t_s", "phase"}`` (tests)."""

    def __init__(self) -> None:
        super().__init__()
        self.episodes = {}  # type: Dict[str, Dict[str, Any]]

    def close(self) -> Dict[str, str]:
        episode_id = self._closing()
        self.episodes[episode_id] = {"meta": self.meta, "frames": self.frames, "t_s": self.times,
                                     "phase": self.phases}
        return {}


def safe_name(text: str) -> str:
    """``text`` with every character outside ``[A-Za-z0-9._-]`` replaced by ``_`` (file names)."""
    return _UNSAFE.sub("_", text)


class NpyFrameSink(_Buffer):
    """Writes ``<frames_dir>/<episode>/<key>.npy`` (frames stacked on axis 0, the dtype the policy received) and
    ``<frames_dir>/<episode>/frames.json`` (episode id, meta, and per key the times and hazard phases)."""

    def __init__(self, frames_dir: Any) -> None:
        super().__init__()
        self.frames_dir = Path(frames_dir)

    def close(self) -> Dict[str, str]:
        episode_id = self._closing()
        out = self.frames_dir / safe_name(episode_id)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}  # type: Dict[str, str]
        for key in sorted(self.frames):
            path = out / ("%s.npy" % safe_name(key))
            np.save(str(path), np.stack(self.frames[key]))
            paths[key] = str(path)
        index = {"episode_id": episode_id, "meta": self.meta,
                 "keys": {key: {"file": Path(paths[key]).name, "n_frames": len(self.frames[key]),
                                "t_s": self.times[key], "phase": self.phases[key]} for key in sorted(self.frames)}}
        index_path = out / FRAME_INDEX_FILE
        index_path.write_text(json.dumps(index, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        paths["index"] = str(index_path)
        return paths


def make_npy_sink(frames_dir: Any) -> NpyFrameSink:
    """``--frame-sink hazardlib.run.frames:make_npy_sink``."""
    return NpyFrameSink(frames_dir)


class _PolicyWrapper:
    """Forwards the ``ReferencePolicy`` attributes of the wrapped policy."""

    def __init__(self, policy: Any) -> None:
        info = describe_policy(policy)
        self.inner = policy
        self.kind, self.name, self.under_test = info["kind"], info["name"], info["under_test"]

    def reset(self, task_id: str, layout_id: str, seed: int) -> None:
        self.inner.reset(task_id, layout_id, seed)


class TimedPolicy(_PolicyWrapper):
    """Wall-clock seconds of every ``act`` of the current episode in ``step_times_s`` (reset by ``reset``)."""

    def __init__(self, policy: Any) -> None:
        super().__init__(policy)
        self.step_times_s = []  # type: List[float]

    def reset(self, task_id: str, layout_id: str, seed: int) -> None:
        self.step_times_s = []
        super().reset(task_id, layout_id, seed)

    def act(self, observation: Dict[str, Any]) -> Any:
        start = time.perf_counter()
        action = self.inner.act(observation)
        self.step_times_s.append(time.perf_counter() - start)
        return action

    def summary(self) -> Dict[str, Any]:
        """``n_steps``, ``total_s``, ``mean_s``, ``median_s``, ``max_s`` and ``per_step_s`` (diagnostic only)."""
        times = np.array(self.step_times_s, dtype=np.float64)
        if times.size == 0:
            return {"n_steps": 0, "total_s": 0.0, "mean_s": None, "median_s": None, "max_s": None, "per_step_s": []}
        return {"n_steps": int(times.size), "total_s": float(times.sum()), "mean_s": float(times.mean()),
                "median_s": float(np.median(times)), "max_s": float(times.max()),
                "per_step_s": [float(t) for t in times]}


class FrameRecorder(_PolicyWrapper):
    """See the module docstring. ``session`` is the ``EpisodeSession`` that drives the episodes (its host's
    ``time_state`` and its runtime's episode start and hazard phase are read), ``env`` the environment index.
    Call :meth:`begin` before ``run_episode`` and :meth:`end` after it; ``n_frames`` counts the ``act`` calls of
    the current episode."""

    def __init__(self, policy: Any, sink: Any, keys: Sequence[str], session: Any, env: int) -> None:
        super().__init__(policy)
        for method in ("open", "add", "close"):
            if not callable(getattr(sink, method, None)):
                raise TypeError("%s is not a FrameSink: missing %s()" % (type(sink).__name__, method))
        keys = list(keys)
        if not keys or any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != len(keys):
            raise ValueError("frame keys must be a non-empty list of distinct names, got %r" % (keys,))
        self.sink, self.keys, self.session, self.env = sink, keys, session, env
        self.n_frames = 0
        self._open = False

    def begin(self, episode_id: str, meta: Mapping[str, Any]) -> None:
        self.sink.open(episode_id, meta)
        self._open = True
        self.n_frames = 0

    def end(self) -> Dict[str, str]:
        if not self._open:
            raise RuntimeError("FrameRecorder.end() without begin()")
        self._open = False
        return dict(self.sink.close())

    def _clock(self) -> Any:
        now = self.session.host.time_state(self.env).sim_time_s
        info = self.session.runtime.episode_info(self.env)
        if info is None:
            return float(now), None
        phase = info["phase"] if info["family"] is not None else None
        return float(now - info["t_start_s"]), phase

    def act(self, observation: Dict[str, Any]) -> Any:
        if not self._open:
            raise RuntimeError("FrameRecorder.act() outside begin() / end()")
        t_s, phase = self._clock()
        for key in self.keys:
            if key not in observation:
                raise KeyError("frame key %r is not in the policy observation (keys: %s)" % (key, sorted(observation)))
            self.sink.add(key, observation[key], t_s, phase)
        self.n_frames += 1
        return self.inner.act(observation)
