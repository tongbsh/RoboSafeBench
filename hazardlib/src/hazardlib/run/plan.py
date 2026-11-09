"""Which episodes a run drives (R1 items 2, 3): hazard episodes and their paired hazard-free controls.

For every selected instance and every environment seed there is one **hazard episode** (``episode_id`` = the
instance id) and one **control episode**: the same ``task_id``, ``layout_id`` and seed with ``instance=None``,
the control's ``control_onset_s`` = the instance's nominal onset (the reference time of the topple measurement,
D10 h, D12 a) and the instance's episode length. Instances whose controls would be identical (same task, layout,
seed, nominal onset and episode length) share one control episode, run once; its ``control_id`` is derived from
the smallest of their instance ids (``control:<instance_id>``) and the control records list all of them
(``paired_instance_ids``).

Episode length (00 §8, as for the G1 episodes): ``max(TaskContext nominal success time, nominal_hit_time_s) +
run.yaml: episode.horizon_extra_s``.

Order: seed by seed, instances in input order (Manifest entry order in a formal run); a control comes right
before the first hazard episode that needs it.
"""
import dataclasses
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..gates.decision import label_of
from ..gates.episodes import nominal_success_time_s
from ..spec.certificate import CERT_LABELS
from ..spec.instance import HazardInstance
from ..spec.task_context import TaskContext

HAZARD = "hazard"
CONTROL = "control"
EPISODE_KINDS = (HAZARD, CONTROL)
CONTROL_PREFIX = "control:"


@dataclasses.dataclass(frozen=True)
class PlannedEpisode:
    kind: str
    episode_id: str
    seed: int
    task_id: str
    layout_id: str
    horizon_s: float
    instance: Optional[HazardInstance]
    control_id: str
    control_onset_s: Optional[float]
    paired_instance_ids: Tuple[str, ...]


def episode_horizon_s(task_context: TaskContext, instance: HazardInstance, horizon_extra_s: float) -> float:
    return max(nominal_success_time_s(task_context), float(instance.nominal_hit_time_s)) + float(horizon_extra_s)


def tau_s(instance: HazardInstance) -> float:
    """tau_nom = t_hit - t_onset (00 §5.1), independent of the tested policy."""
    return float(instance.nominal_hit_time_s) - float(instance.onset_time_s)


def control_id_for(instance_ids: Sequence[str]) -> str:
    return CONTROL_PREFIX + min(instance_ids)


def check_labels(labels: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    """``None`` (every label) or the validated tuple of certificate labels to run."""
    if labels is None:
        return None
    unknown = sorted(set(labels) - set(CERT_LABELS))
    if unknown or not labels:
        raise ValueError("--labels must name certificate labels %s, got %r" % (list(CERT_LABELS), list(labels)))
    return tuple(labels)


def plan_episodes(instances: Sequence[HazardInstance], contexts: Mapping[Tuple[str, str], TaskContext],
                  seeds: Sequence[int], horizon_extra_s: float,
                  labels: Optional[Sequence[str]]) -> List[PlannedEpisode]:
    labels = check_labels(labels)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of distinct ints, got %r" % (list(seeds),))
    selected = [i for i in instances if labels is None or label_of(i) in labels]
    horizon = {}  # type: Dict[str, float]
    groups = {}  # type: Dict[Tuple[str, str, float, float], List[str]]
    for instance in selected:
        prov = instance.provenance
        tc = contexts[(prov["task_id"], prov["layout_id"])]
        horizon[instance.instance_id] = episode_horizon_s(tc, instance, horizon_extra_s)
        key = (prov["task_id"], prov["layout_id"], float(instance.onset_time_s), horizon[instance.instance_id])
        groups.setdefault(key, []).append(instance.instance_id)
    out = []  # type: List[PlannedEpisode]
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seeds must be ints, got %r" % (seed,))
        emitted = set()
        for instance in selected:
            prov = instance.provenance
            task_id, layout_id = prov["task_id"], prov["layout_id"]
            length = horizon[instance.instance_id]
            paired = tuple(sorted(groups[(task_id, layout_id, float(instance.onset_time_s), length)]))
            control_id = control_id_for(paired)
            if control_id not in emitted:
                emitted.add(control_id)
                out.append(PlannedEpisode(kind=CONTROL, episode_id=control_id, seed=seed, task_id=task_id,
                                          layout_id=layout_id, horizon_s=length, instance=None,
                                          control_id=control_id, control_onset_s=float(instance.onset_time_s),
                                          paired_instance_ids=paired))
            out.append(PlannedEpisode(kind=HAZARD, episode_id=instance.instance_id, seed=seed, task_id=task_id,
                                      layout_id=layout_id, horizon_s=length, instance=instance,
                                      control_id=control_id, control_onset_s=None,
                                      paired_instance_ids=(instance.instance_id,)))
    return out
