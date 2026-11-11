"""``hazardctl run``: drive the planned episodes with the policy under test and append one record each (R1 item 3).

Output directory:

* ``run_meta.yaml``: what identifies the run (mode, ``manifest_hash``, policy name and ``describe_policy``, BD
  hash, calibration hash, the ``run.yaml`` episode settings, topple monitoring) plus the ``run.yaml`` hash and the
  input paths of the first invocation. A later invocation on the same directory must have the same identity (else
  refused): it resumes the run, e.g. with more seeds. The ``report`` section of ``run.yaml`` may change in between
  (every record keeps the ``run_config_hash`` it was run with).
* ``episodes.jsonl``: one JSON record per finished episode, appended and flushed (``fsync``) right after the
  episode, so an interruption loses at most the running episode. On resume, a trailing line without its newline
  (an interrupted write) is dropped, and every (episode id, policy name, seed) that already has a record is
  skipped.
* ``frames/``: what a ``FrameSink`` writes (only with ``--frame-sink``); records keep the paths relative to the
  output directory.

Refusals before any episode: the policy name (``--policy-name``, and the policy's own ``name``) appears among the
probe sources of a TaskContext in use (00 §5.4; the error names the TaskContext); the host's BD is not the
inputs' BD; the calibration's T_stop is censored and ``allow_censored_t_stop`` was not given (D8); the plan is
empty.

Each episode goes through ``EpisodeSession.run_episode`` (synchronous: the simulation does not advance while the
policy computes its action). Hazard episodes are annotated with ``exposure_low`` (``annotate_exposure``, radius
``semantics.yaml: exposure.low_exposure_radius_m``). Topple is monitored on every TaskContext entity (D10 h)
unless ``monitor_topple`` is false; a control passes ``control_onset_s`` = its instance's nominal onset (D12 a).
With ``session.runtime.skip_unreported_entities`` the entities the host does not report a pose for (e.g. a
region of an articulated fixture) are left out of the monitoring instead of refusing the episode; the record's
``task_entities`` then holds the monitored names and ``task_entities_skipped`` the others.
The wall-clock time of every ``act`` is recorded (``TimedPolicy``); only its summary is stored, as a diagnostic.
"""
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from ..calibrate.protocol import sigma_from_summary
from ..driving import describe_policy
from ..gates.decision import in_main_score, label_of
from ..gates.exposure import annotate_exposure
from ..spec.task_context import TaskContext
from .frames import FrameRecorder, TimedPolicy
from .inputs import RunInputs
from .plan import CONTROL, HAZARD, PlannedEpisode, plan_episodes, tau_s
from .settings import RunSettings

EPISODES_FILE = "episodes.jsonl"
RUN_META_FILE = "run_meta.yaml"
FRAMES_DIR = "frames"
RECORD_VERSION = "1"
META_VERSION = "1"
IDENTITY_FIELDS = ("meta_version", "pilot", "manifest_hash", "benchmark", "bd_hash", "calibration_hash",
                   "policy_name", "policy", "episode_settings", "monitor_topple")


class PolicyInProbeSetError(ValueError):
    """The policy under test is one of the probe sources of a TaskContext (00 §5.4, R4)."""


class RunDirectoryError(ValueError):
    """The output directory belongs to another run, or its records cannot be read."""


def probe_source_names(task_context: TaskContext) -> List[str]:
    """Names of every probe source of the TaskContext (``provenance.probe_set``: sources and reference policies)."""
    probe_set = task_context.provenance.get("probe_set") or {}
    names = {source.get("name") for source in probe_set.get("sources") or [] if isinstance(source, Mapping)}
    names.update(probe_set.get("reference_policies") or [])
    return sorted(n for n in names if isinstance(n, str))


def check_not_in_probe_set(policy_names: Sequence[str], contexts: Mapping[Tuple[str, str], TaskContext]) -> None:
    problems = []  # type: List[str]
    for key in sorted(contexts):
        tc = contexts[key]
        sources = probe_source_names(tc)
        for name in sorted(set(policy_names)):
            if name in sources:
                problems.append("policy %r is a probe source of TaskContext %s/%s (hash %s; probe sources %s)"
                                % (name, tc.task_id, tc.layout_id, tc.hash(), sources))
    if problems:
        raise PolicyInProbeSetError("00 §5.4: a policy under test may not be evaluated on instances generated from "
                                    "its own probes:\n  %s" % "\n  ".join(problems))


def read_episode_records(path: Any, repair: bool) -> List[Dict[str, Any]]:
    """Records of ``episodes.jsonl`` (``[]`` if absent). A last line without its newline is an interrupted write:
    it is ignored, and with ``repair`` removed from the file. Any other unreadable line is an error."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    partial = lines.pop()
    if partial and repair:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(line + "\n" for line in lines))
    out = []  # type: List[Dict[str, Any]]
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise RunDirectoryError("%s line %d is not JSON: %s" % (path, number, exc)) from None
        if not isinstance(record, dict) or record.get("record_version") != RECORD_VERSION:
            raise RunDirectoryError("%s line %d is not a run record (record_version %r)"
                                    % (path, number, RECORD_VERSION))
        out.append(record)
    return out


def append_record(path: Any, record: Mapping[str, Any]) -> None:
    line = json.dumps(record, allow_nan=False, sort_keys=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def record_key(record: Mapping[str, Any]) -> Tuple[str, str, int]:
    return record["episode_id"], record["policy_name"], record["seed"]


def _write_or_check_meta(out: Path, meta: Dict[str, Any]) -> None:
    path = out / RUN_META_FILE
    if not path.exists():
        path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        return
    existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    diffs = ["%s: directory has %r, this run %r" % (name, existing.get(name), meta.get(name))
             for name in IDENTITY_FIELDS if existing.get(name) != meta.get(name)]
    if diffs:
        raise RunDirectoryError("%s belongs to another run (%d difference(s)); use a new --out-dir:\n  %s"
                                % (out, len(diffs), "\n  ".join(diffs)))


def _relative(paths: Mapping[str, str], out: Path) -> Dict[str, str]:
    result = {}
    for key, value in paths.items():
        path = Path(value)
        try:
            result[key] = path.resolve().relative_to(out.resolve()).as_posix()
        except ValueError:
            result[key] = str(value)
    return result


def _inference_summary(timed: TimedPolicy) -> Dict[str, Any]:
    summary = timed.summary()
    summary.pop("per_step_s")
    return summary


@dataclasses.dataclass
class RunResult:
    out_dir: str
    planned: int
    ran: int
    skipped: int
    ran_by_kind: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def run_stage(session: Any, inputs: RunInputs, policy: Any, policy_name: str, settings: RunSettings, out_dir: Any,
              seeds: Sequence[int], env: int, exposure_radius_m: float, labels: Optional[Sequence[str]],
              frame_sink: Optional[Any], frame_keys: Optional[Sequence[str]], monitor_topple: bool,
              allow_censored_t_stop: bool, only_episode_ids: Optional[Collection[str]] = None) -> RunResult:
    """Run every planned episode that has no record yet (module docstring). ``frame_sink`` and ``frame_keys`` go
    together (both or neither). ``only_episode_ids``: drive just the planned episodes with these ids (the others
    are neither run nor counted as skipped); a scheduler that spreads one plan over several processes uses it."""
    if not isinstance(policy_name, str) or not policy_name:
        raise ValueError("a run needs the policy name (--policy-name), got %r" % (policy_name,))
    info = describe_policy(policy)
    check_not_in_probe_set([policy_name, info["name"]], inputs.contexts)
    if session.bd.hash() != inputs.bd.hash():
        raise ValueError("the host's BD %r (%s) is not the run's BD %r (%s)"
                         % (session.bd.name, session.bd.hash(), inputs.bd.name, inputs.bd.hash()))
    if (frame_sink is None) != (not frame_keys):
        raise ValueError("--frame-sink and --frame-keys go together (got sink %r, keys %r)" % (frame_sink, frame_keys))
    t_stop = inputs.t_stop
    if t_stop is not None:
        sigma_from_summary(0.0, t_stop, allow_censored_t_stop)  # refuses a censored T_stop unless allowed (D8)
    plan = plan_episodes(inputs.instances, inputs.contexts, list(seeds), settings.episode.horizon_extra_s, labels)
    if not plan:
        raise ValueError("nothing to run: no instance has one of the labels %s" % (list(labels or []),))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"meta_version": META_VERSION, "pilot": inputs.pilot, "manifest_hash": inputs.manifest_hash,
            "benchmark": inputs.bd.name, "bd_hash": inputs.bd.hash(), "calibration_hash": inputs.calibration_hash,
            "policy_name": policy_name, "policy": info, "run_config_hash": settings.config_hash,
            "episode_settings": dataclasses.asdict(settings.episode), "monitor_topple": bool(monitor_topple),
            "sources": dict(inputs.sources)}
    _write_or_check_meta(out, meta)
    episodes_path = out / EPISODES_FILE
    done = {record_key(r) for r in read_episode_records(episodes_path, repair=True)}

    timed = TimedPolicy(policy)
    recorder = None if frame_sink is None else FrameRecorder(timed, frame_sink, list(frame_keys or []), session, env)
    driver = timed if recorder is None else recorder
    ran = {HAZARD: 0, CONTROL: 0}
    skipped = 0
    for planned in plan:
        if only_episode_ids is not None and planned.episode_id not in only_episode_ids:
            continue
        if (planned.episode_id, policy_name, planned.seed) in done:
            skipped += 1
            continue
        record = _run_one(session, inputs, planned, driver, timed, recorder, policy_name, info, settings, env,
                          exposure_radius_m, monitor_topple, allow_censored_t_stop, out)
        append_record(episodes_path, record)
        done.add(record_key(record))
        ran[planned.kind] += 1
    return RunResult(out_dir=str(out), planned=len(plan), ran=sum(ran.values()), skipped=skipped, ran_by_kind=ran)


def _run_one(session: Any, inputs: RunInputs, planned: PlannedEpisode, driver: Any, timed: TimedPolicy,
             recorder: Optional[FrameRecorder], policy_name: str, info: Dict[str, Any], settings: RunSettings,
             env: int, exposure_radius_m: float, monitor_topple: bool, allow_censored_t_stop: bool,
             out: Path) -> Dict[str, Any]:
    tc = inputs.contexts[(planned.task_id, planned.layout_id)]
    entities = sorted({ref.name for ref in tc.entities.values()}) if monitor_topple else []
    if recorder is not None:
        recorder.begin("%s__%s__s%d" % (planned.episode_id, policy_name, planned.seed),
                       {"kind": planned.kind, "episode_id": planned.episode_id, "policy_name": policy_name,
                        "seed": planned.seed, "task_id": planned.task_id, "layout_id": planned.layout_id})
    result = session.run_episode(env, planned.instance, driver, planned.task_id, planned.layout_id, planned.seed,
                                 planned.horizon_s,
                                 control_id=planned.control_id if planned.kind == CONTROL else None,
                                 stop_on_success=settings.episode.stop_on_success,
                                 measure_visibility=planned.kind == HAZARD, task_entities=entities or None,
                                 control_onset_s=planned.control_onset_s)
    frames = None if recorder is None else _relative(recorder.end(), out)
    info_run = session.runtime.episode_info(env)
    skipped = [] if info_run is None else list(info_run.get("task_entities_skipped") or [])
    monitored = [name for name in entities if name not in skipped]
    log = result.log
    instance = planned.instance
    hazard_fields = {"instance_id": None, "spec_id": None, "family": None, "tau_s": None, "sigma": None,
                     "sigma_is_upper_bound": None, "label": None, "in_main_score": None}
    if instance is not None:
        log = annotate_exposure(log, instance, result.eef_pos, result.times_s, exposure_radius_m)
        tau = tau_s(instance)
        sigma, upper = (None, None)
        if inputs.t_stop is not None:
            sigma, upper = sigma_from_summary(tau, inputs.t_stop, allow_censored_t_stop)
        hazard_fields = {"instance_id": instance.instance_id, "spec_id": instance.spec_id, "family": instance.family,
                         "tau_s": tau, "sigma": sigma, "sigma_is_upper_bound": upper,
                         "label": label_of(instance), "in_main_score": in_main_score(instance)}
    record = {"record_version": RECORD_VERSION, "kind": planned.kind, "episode_id": planned.episode_id,
              "control_id": planned.control_id, "paired_instance_ids": list(planned.paired_instance_ids),
              "control_onset_s": planned.control_onset_s, "task_id": planned.task_id,
              "layout_id": planned.layout_id, "seed": planned.seed}
    record.update(hazard_fields)
    record.update({"horizon_s": planned.horizon_s, "stop_on_success": settings.episode.stop_on_success,
                   "n_control_steps": int(len(result.times_s) - 1), "success_t_s": result.success_t_s,
                   "official_contact_count": result.official_contact_count,
                   "first_visible_t_s": result.first_visible_t_s, "task_entities": monitored or None,
                   "task_entities_skipped": skipped or None,
                   "policy_name": policy_name, "policy": dict(info), "inference_time_s": _inference_summary(timed),
                   "frames": frames, "pilot": inputs.pilot, "manifest_hash": inputs.manifest_hash,
                   "run_config_hash": settings.config_hash, "log": log.to_dict()})
    return record
