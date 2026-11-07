"""The gate stage: G0 -> baseline -> nominal hit -> G1 -> G2 -> label for a batch of instances.

G0-rejected instances leave the batch (``GateOutput.rejections``, counted by reason; an instance with several
reasons counts once per reason and once in ``n_rejected``). Every other instance gets a certificate, and the
gate details go into ``provenance.gates``. ``baseline_unusable`` instances are kept with their label so that
the freeze step can count them; they are never run further. The baseline is evaluated once per
(task, layout).
"""
import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..driving import EpisodeSession, describe_policy
from ..spec.instance import HazardInstance
from ..spec.task_context import TaskContext
from .decision import decide_label, in_main_score, make_certificate
from .episodes import BaselineResult, run_baseline, run_g1, run_g2, run_nominal_hit
from .g0 import G0_REASONS, run_g0
from .settings import GateSettings


@dataclasses.dataclass
class GateOutput:
    instances: List[HazardInstance]
    rejections: List[Dict[str, Any]]
    summary: Dict[str, Any]


def _key(instance: HazardInstance) -> Tuple[str, str]:
    prov = instance.provenance
    if "task_id" not in prov or "layout_id" not in prov:
        raise ValueError("instance %r has no provenance task_id / layout_id" % (instance.instance_id,))
    return prov["task_id"], prov["layout_id"]


def gate_instances(session: EpisodeSession, instances: Sequence[HazardInstance],
                   task_contexts: Mapping[Tuple[str, str], TaskContext], reference_policy: Any,
                   avoider: Optional[Any], settings: GateSettings, seed: int, env: int) -> GateOutput:
    policy_info = describe_policy(reference_policy)
    if policy_info["under_test"]:
        raise ValueError("the G1/G2 reference policy %r is marked under_test" % (policy_info["name"],))
    baselines = {}  # type: Dict[Tuple[str, str], BaselineResult]
    kept, rejections = [], []  # type: List[HazardInstance], List[Dict[str, Any]]
    by_reason = {reason: 0 for reason in G0_REASONS}
    labels = {}  # type: Dict[str, int]
    for instance in instances:
        key = _key(instance)
        if key not in task_contexts:
            raise ValueError("no TaskContext for %s (instance %r)" % (key, instance.instance_id))
        tc = task_contexts[key]
        if instance.provenance.get("task_context_hash") != tc.hash():
            raise ValueError("instance %r was built from another TaskContext than the one given for %s"
                             % (instance.instance_id, key))
        g0 = run_g0(instance, tc, session.bd, settings.g0)
        if not g0.passed:
            for reason in g0.reasons:
                by_reason[reason] += 1
            prov = instance.provenance
            rejections.append({"stage": "g0", "instance_id": instance.instance_id, "spec_id": instance.spec_id,
                               "task_id": key[0], "layout_id": key[1], "instance_seed": prov.get("instance_seed"),
                               "reasons": list(g0.reasons), "checks": g0.checks})
            continue
        if key not in baselines:
            baselines[key] = run_baseline(session, reference_policy, tc, settings, seed, env)
        baseline = baselines[key]
        gates = {"g0": g0.to_dict(), "baseline": {"usable": baseline.usable, "successes": baseline.successes,
                                                  "runs": baseline.runs}, "reference_policy": policy_info}
        notes = []
        g1 = g2 = None
        if baseline.usable:
            nominal = run_nominal_hit(session, instance, reference_policy, tc, settings, seed, env)
            gates["nominal_hit"] = nominal
            gates["tau_vis_s"] = nominal["tau_vis_s"]
            notes.append("tau_vis_s=%s (%s)" % (nominal["tau_vis_s"], nominal["tau_vis_note"]))
            g1 = run_g1(session, instance, reference_policy, tc, settings, seed, env)
            gates["g1"] = g1.to_dict()
            if not g1.passed:
                g2 = run_g2(session, instance, reference_policy, avoider, tc, settings, seed, env)
                gates["g2"] = g2.to_dict()
                if g2.passed is None:
                    notes.append(g2.note)
        else:
            notes.append("baseline_unusable: reference %r succeeded in %d/%d hazard-free runs"
                         % (policy_info["name"], baseline.successes, baseline.runs))
        label = decide_label(True, baseline.usable, None if g1 is None else g1.passed, avoider is not None,
                             None if g2 is None else g2.passed)
        certificate = make_certificate(label, None if g1 is None else g1.passed,
                                       None if g2 is None else g2.passed, "; ".join(notes))
        provenance = dict(instance.provenance, gates=gates)
        gated = HazardInstance.from_dict(dict(instance.to_dict(), certificate=certificate.to_dict(),
                                              provenance=provenance))
        labels[label] = labels.get(label, 0) + 1
        kept.append(gated)
    summary = {
        "n_instances": len(instances), "n_rejected": len(rejections),
        "rejections_by_reason": {k: v for k, v in by_reason.items() if v},
        "labels": dict(sorted(labels.items())),
        "n_main_score": sum(1 for i in kept if in_main_score(i)),
        "baselines": {"%s/%s" % k: v.to_dict() for k, v in sorted(baselines.items())},
        "reference_policy": policy_info, "g2_available": avoider is not None,
    }
    return GateOutput(instances=kept, rejections=rejections, summary=summary)
