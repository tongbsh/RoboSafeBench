"""Layout selection and the global episode queue of the generated-cell evaluation (L5; no simulator).

The generation batch (``trajectory_batch``) leaves, per LIBERO task, ``trajectories/<demo>/`` folders holding
``validation.json`` (cells ``family@elevation`` x Sigma target, each with its selected instance, evidence, Sigma
record and G1 diagnostic), ``instances.jsonl``, ``context_metadata.yaml`` and ``calibration.yaml``. This module reads
them (never writes) and decides what to run:

* :func:`select_task`: the first N layouts (coverage order) that have at least one admitted instance. A folder is
  complete when ``instances.jsonl`` exists (it is written after ``validation.json``); a layout that is not complete
  yet and precedes the N-th selected one makes the task *incomplete* (selection would change when it finishes),
  unless ``allow_incomplete``;
* :func:`plan_layout_episodes`: the hazard and control episodes of one layout, by the core ``plan_episodes`` (same
  control sharing rule), from the nominal success time of the context alone (no TaskContext hashing: the
  orchestrator plans ~800 layouts);
* :func:`build_queue`: the global, layout-major queue of ``Work`` items minus the episodes that already have a record.
"""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Set, Tuple

from hazardlib.pipeline_io import read_records, read_yaml
from hazardlib.run.plan import CONTROL, HAZARD, plan_episodes

from .trajectory_batch import parse_tasks, usable_demos

INSTANCES_FILE = "instances.jsonl"
VALIDATION_FILE = "validation.json"
CONTEXT_FILE = "context_metadata.yaml"
CALIBRATION_FILE = "calibration.yaml"


def task_name(suite: str, index: int) -> str:
    return "%s_task%d" % (suite, index)


class IncompleteTask(Exception):
    """A layout before the N-th selected one has no ``instances.jsonl`` yet (the batch is still working on it)."""


@dataclasses.dataclass
class Work:
    task: str            # "libero_spatial_task0"
    demo: str            # "demo_3"
    episode_id: str
    kind: str            # "hazard" | "control"
    attempt: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"task": self.task, "demo": self.demo, "episode_id": self.episode_id, "kind": self.kind}


@dataclasses.dataclass
class LayoutSelection:
    task: str
    suite: str
    demo: str
    folder: Path
    cells: Dict[str, Dict[str, Any]]      # instance_id -> cell metadata
    cell_layout_counts: Dict[str, bool]   # "<cell>|<target>" -> True when this layout admitted an instance
    episodes: List[Tuple[str, str]]       # (episode_id, kind) in plan order

    @property
    def n_hazard(self) -> int:
        return sum(1 for _, kind in self.episodes if kind == HAZARD)

    @property
    def n_control(self) -> int:
        return sum(1 for _, kind in self.episodes if kind == CONTROL)


def split_cell(cell: str) -> Tuple[str, str]:
    family, _, elevation = cell.partition("@")
    return family, elevation


def layout_cells(validation: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, bool]]:
    """``(instances, coverage)``: every selected instance of a ``validation.json`` keyed by instance id (cell,
    Sigma target, G1 diagnostic, achieved Sigma / tau), and ``"<cell>|<target>" -> admitted`` for every cell."""
    instances, coverage = {}, {}  # type: Dict[str, Dict[str, Any]], Dict[str, bool]
    for cell, targets in sorted(validation.get("families", {}).items()):
        family, elevation = split_cell(cell)
        for target, result in sorted(targets.items(), key=lambda kv: float(kv[0])):
            selected = result.get("selected")
            coverage["%s|%s" % (cell, target)] = selected is not None
            if selected is None:
                continue
            instance = selected["instance"]
            g1 = selected.get("g1") or {}
            sigma = selected.get("sigma") or {}
            certificate = instance.get("certificate") or {}
            instances[instance["instance_id"]] = {
                "cell": cell, "family": family, "elevation": elevation, "target": target,
                "g1_passed": g1.get("passed"), "label": certificate.get("label"),
                "sigma_achieved": sigma.get("sigma_achieved"), "tau_s": sigma.get("tau_s"),
                "t_stop_s": sigma.get("t_stop_s")}
    return instances, coverage


def plan_layout_episodes(instances: Sequence[Any], t_end_s: float, horizon_extra_s: float,
                         seed: int) -> List[Tuple[str, str]]:
    """``[(episode_id, kind)]`` of one layout, from the core planner (controls shared exactly as in the run
    stage); the TaskContext is replaced by its nominal success time, the only field the planner reads."""
    contexts = {}
    for instance in instances:
        prov = instance.provenance
        contexts[(prov["task_id"], prov["layout_id"])] = SimpleNamespace(phases=[SimpleNamespace(t_end_s=t_end_s)])
    return [(p.episode_id, p.kind) for p in plan_episodes(list(instances), contexts, [seed], horizon_extra_s, None)]


def select_task(gen_root: Any, suite: str, index: int, n_layouts: int, horizon_extra_s: float, seed: int,
                allow_incomplete: bool = False) -> List[LayoutSelection]:
    """The first ``n_layouts`` layouts (coverage order) with an admitted instance. Raises :class:`IncompleteTask`
    (see module docstring) and ``FileNotFoundError`` for a task that has no generation output at all."""
    name = task_name(suite, index)
    folder = Path(gen_root) / name
    if not (folder / "coverage.json").is_file():
        raise FileNotFoundError("%s has no coverage.json" % folder)
    chosen = []  # type: List[LayoutSelection]
    for demo in usable_demos(folder, None):
        if len(chosen) >= n_layouts:
            break
        layout = folder / "trajectories" / demo
        if not (layout / INSTANCES_FILE).is_file():
            if allow_incomplete:
                continue
            raise IncompleteTask("%s/%s is not validated yet" % (name, demo))
        instances, _ = read_records(layout / INSTANCES_FILE)
        if not instances:
            continue
        validation = json.loads((layout / VALIDATION_FILE).read_text())
        cells, coverage = layout_cells(validation)
        ids = {i.instance_id for i in instances}
        if ids != set(cells):
            raise ValueError("%s: instances.jsonl and validation.json disagree (%d vs %d instances)"
                             % (layout, len(ids), len(cells)))
        context = read_yaml(layout / CONTEXT_FILE)
        episodes = plan_layout_episodes(instances, float(context["phases"][-1]["t_end_s"]), horizon_extra_s, seed)
        chosen.append(LayoutSelection(task=name, suite=suite, demo=demo, folder=layout, cells=cells,
                                      cell_layout_counts=coverage, episodes=episodes))
    return chosen


def select_all(gen_root: Any, tasks: Sequence[Tuple[str, int]], n_layouts: int, horizon_extra_s: float, seed: int,
               allow_incomplete: bool = False) -> Tuple[Dict[str, List[LayoutSelection]], Dict[str, str]]:
    """``(selected, skipped)``: layouts per task name, and the reason for every task left out."""
    selected, skipped = {}, {}  # type: Dict[str, List[LayoutSelection]], Dict[str, str]
    for suite, index in tasks:
        name = task_name(suite, index)
        try:
            layouts = select_task(gen_root, suite, index, n_layouts, horizon_extra_s, seed, allow_incomplete)
        except (IncompleteTask, FileNotFoundError) as exc:
            skipped[name] = str(exc)
            continue
        if not layouts:
            skipped[name] = "no layout with an admitted instance"
            continue
        selected[name] = layouts
    return selected, skipped


def build_queue(selected: Dict[str, List[LayoutSelection]], done: Dict[str, Set[str]]) -> List[Work]:
    """Layout-major global queue (tasks in the given order, layouts in coverage order, plan order within a layout,
    a control right before the first hazard that needs it). ``done[task]``: episode ids that already have a record
    (never queued again)."""
    queue = []  # type: List[Work]
    for name, layouts in selected.items():
        finished = done.get(name, set())
        for layout in layouts:
            for episode_id, kind in layout.episodes:
                if episode_id not in finished:
                    queue.append(Work(name, layout.demo, episode_id, kind))
    return queue


def expected_counts(selected: Dict[str, List[LayoutSelection]]) -> Dict[str, Dict[str, int]]:
    return {name: {"layouts": len(layouts), "hazard": sum(l.n_hazard for l in layouts),
                   "control": sum(l.n_control for l in layouts)} for name, layouts in selected.items()}


def selection_document(selected: Dict[str, List[LayoutSelection]], skipped: Dict[str, str],
                       arguments: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON stored as ``selection.json``: what the report needs to attribute every record to a cell."""
    tasks = {}
    for name, layouts in selected.items():
        tasks[name] = {"suite": layouts[0].suite, "layouts": [
            {"demo": l.demo, "folder": str(l.folder), "n_hazard": l.n_hazard, "n_control": l.n_control,
             "instances": l.cells, "cell_admitted": l.cell_layout_counts} for l in layouts]}
    return {"arguments": arguments, "tasks": tasks, "skipped": skipped, "expected": expected_counts(selected)}


def parse_task_spec(text: str) -> List[Tuple[str, int]]:
    return parse_tasks(text)
