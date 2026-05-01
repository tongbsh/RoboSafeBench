"""Optional JSON metadata for sudden-drop eval (dynamic level 1): obstacle name per episode.

The bundled ``l2_active_obstacle_metadata.json`` is generated from SafeLIBERO BDDL
``(:init)`` obstacle line order and ``libero_suite_task_map`` task indices (same rule as
``simulator_env/safelibero/libero/libero/benchmark/safelibero_obstacle_defaults.py``).
Regenerate without logs::

  PYTHONPATH=../simulator_env/dyniSafety_libero python \\
    ../simulator_env/safelibero/scripts/regenerate_l2_obstacle_metadata_from_bddls.py \\
    --out experiments/robot/dynisafety_libero/l2_active_obstacle_metadata.json

File format (see ``l2_active_obstacle_metadata.example.json``):

* Top-level keys: task suite names, e.g. ``safelibero_object``.
* Second-level keys: BDDL task stems (``Task.name``), e.g.
  ``pick_up_the_orange_juice_and_place_it_in_the_basket``.
* Values: either

  * A list of obstacle MuJoCo object names, one per episode index (wrap with modulo
    if the list is shorter than ``num_trials_per_task``), or
  * A dict mapping safety level ``\"I\"`` / ``\"II\"`` to such a list.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def load_l2_active_obstacle_metadata(path: str) -> Dict[str, Any]:
    if not path or not str(path).strip():
        raise ValueError("metadata path is empty")
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"L2 obstacle metadata not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("L2 obstacle metadata root must be a JSON object")
    return data


def _episode_list_for_task_entry(task_entry: Any, safety_level: str) -> Optional[List[str]]:
    if task_entry is None:
        return None
    if isinstance(task_entry, list):
        return task_entry
    if isinstance(task_entry, dict):
        seq = task_entry.get(safety_level)
        if seq is None:
            seq = task_entry.get(str(safety_level))
        if seq is None and safety_level != "I" and "I" in task_entry:
            seq = task_entry.get("I")
        if seq is None:
            return None
        if not isinstance(seq, list):
            return None
        return seq
    return None


def resolve_l2_active_obstacle_name(
    meta: Dict[str, Any],
    task_suite_name: str,
    task_bddl_stem: str,
    safety_level: str,
    episode_idx: int,
) -> Optional[str]:
    """Return MuJoCo obstacle object name for this episode, or None if not configured."""
    suite_block = meta.get(task_suite_name)
    if not isinstance(suite_block, dict):
        return None
    task_entry = suite_block.get(task_bddl_stem)
    seq = _episode_list_for_task_entry(task_entry, safety_level)
    if not seq:
        return None
    if not all(isinstance(x, str) and x for x in seq):
        return None
    return seq[int(episode_idx) % len(seq)]
