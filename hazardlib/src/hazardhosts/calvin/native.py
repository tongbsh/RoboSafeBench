"""CALVIN's native evaluation loop with the hazard runtime (H1; 01 §3.3 item 14, D13 j).

The loop is upstream's own code: ``evaluate_sequence`` and ``rollout`` of ``calvin_agent/evaluation/
evaluate_policy.py`` with ``get_env_state_for_initial_condition`` / ``temp_seed`` / ``collect_plan`` of
``evaluation/utils.py``, at ``calvin@fa03f01``. Those modules cannot be imported in ``calvin_venv``: they import
pytorch_lightning, whose fsspec uses Python 3.9 syntax (``fsspec/caching.py:40``, ``list[[int, int]]``).
:func:`load_upstream` therefore compiles the named top-level statements of the pinned, unmodified files (checked
against the commit, R13) into one namespace, with the module-level names they read (numpy, ``time``, ``colored``,
``pyhash``...) imported as the files do. Nothing of upstream is edited or copied into this repository.

The hazard hooks wrap the namespace's ``rollout`` (one call = one CALVIN subtask = one hazard episode), so the
upstream ``evaluate_sequence`` calls the wrapper:

    arm(0, instance or control) -> host.begin_native_episode(subtask, env.get_info())   (begin_episode)
    upstream rollout(...)                                                                (EP_LEN = 360 steps)
    runtime.end_episode(0) -> append_jsonl(<log>)                                        (exactly once)

``begin_native_episode`` takes the same ``env.get_info()`` the upstream rollout reads as its ``start_info``
(nothing moves in between), so the hazard log's ``task_success`` and upstream's verdict use the same oracle input.
"""
import ast
import collections
import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from hazardlib.semantics import append_jsonl

from .env import UpstreamMismatchError, _git, calvin_env_root

CALVIN_COMMIT = "fa03f01f19c65920e18cf37398a9ce859274af76"
UPSTREAM = (
    ("calvin_models/calvin_agent/evaluation/utils.py",
     ("hasher", "temp_seed", "get_env_state_for_initial_condition", "collect_plan")),
    ("calvin_models/calvin_agent/evaluation/evaluate_policy.py", ("EP_LEN", "evaluate_sequence", "rollout")),
)


def calvin_root() -> Path:
    return calvin_env_root().parent


def check_calvin_commit() -> str:
    root = calvin_root()
    head = _git(root, "rev-parse", "HEAD")
    if head != CALVIN_COMMIT:
        raise UpstreamMismatchError("calvin is at %s, the native loop is pinned to %s" % (head, CALVIN_COMMIT))
    dirty = _git(root, "status", "--porcelain", "--", *[path for path, _ in UPSTREAM])
    if dirty:
        raise UpstreamMismatchError("calvin has local changes in %s" % dirty)
    return head


def _names(node: ast.stmt) -> List[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    return []


def load_upstream() -> Dict[str, Any]:
    """The namespace holding the upstream definitions named in :data:`UPSTREAM` (see the module docstring)."""
    check_calvin_commit()
    import pyhash
    from termcolor import colored
    namespace = {"__name__": "calvin_native_upstream", "np": np, "pi": np.pi, "time": time, "logging": logging,
                 "contextlib": contextlib, "pyhash": pyhash, "colored": colored}  # type: Dict[str, Any]
    for relative, wanted in UPSTREAM:
        path = calvin_root() / relative
        tree = ast.parse(path.read_text(), filename=str(path))
        body = [node for node in tree.body if set(_names(node)) & set(wanted)]
        found = {name for node in body for name in _names(node)}
        missing = sorted(set(wanted) - found)
        if missing:
            raise UpstreamMismatchError("%s has no top-level %s" % (path, missing))
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class NativeLoop:
    """Upstream's evaluation loop on ``host.env`` with ``session.runtime`` hooked in (module docstring).

    ``arm_next(subtask)`` returns what to arm for the next subtask: ``(instance, task_entities)`` or ``None`` for
    a control episode. Logs go to ``log_path`` (one line per subtask episode); ``episodes`` counts the episodes
    begun and ``ended`` the ``end_episode`` calls."""

    def __init__(self, host: Any, runtime: Any, log_path: str,
                 arm_next: Optional[Callable[[str], Any]] = None) -> None:
        self.host, self.runtime, self.log_path = host, runtime, log_path
        self.arm_next = arm_next or (lambda subtask: None)
        self.upstream = load_upstream()
        self._rollout = self.upstream["rollout"]
        self.upstream["rollout"] = self.rollout
        self.episodes = 0
        self.ended = 0
        self.results = []  # type: List[Dict[str, Any]]

    def rollout(self, env: Any, model: Any, task_oracle: Any, subtask: str, val_annotations: Any, plans: Any,
                debug: bool) -> bool:
        if env is not self.host.env:
            raise ValueError("the native loop must run on the host's env")
        armed = self.arm_next(subtask)
        if armed is None:
            self.runtime.arm(0, None, control_id="native:%s:%d" % (subtask, self.episodes))
        else:
            instance, entities = armed
            self.runtime.arm(0, instance, task_entities=entities)
        self.host.begin_native_episode(subtask, env.get_info())
        self.episodes += 1
        t0 = self.host.time_state(0).sim_time_s
        try:
            success = self._rollout(env, model, task_oracle, subtask, val_annotations, plans, debug)
        finally:
            log = self.runtime.end_episode(0)
            self.ended += 1
            append_jsonl(self.log_path, log)
        self.results.append({"subtask": subtask, "upstream_success": bool(success),
                             "log_task_success": log.task_success, "log_id": log.instance_id,
                             "steps": int(round((self.host.time_state(0).sim_time_s - t0) / self.host.control_dt))})
        return success

    def evaluate_sequence(self, model: Any, task_checker: Any, initial_state: Any, eval_sequence: Iterable[str],
                          val_annotations: Any) -> int:
        """Upstream ``evaluate_sequence`` (resets the env to the initial condition, then chains the subtasks);
        ``plans`` is a ``defaultdict(list)`` as in upstream ``evaluate_policy``."""
        return self.upstream["evaluate_sequence"](self.host.env, model, task_checker, initial_state,
                                                  list(eval_sequence), val_annotations,
                                                  collections.defaultdict(list), False)
