"""Aggregate episode rows into Base vs Oracle feasibility cells."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dynisafety.core.episode import EpisodeRecord
from dynisafety.core.joint_outcomes import classify_joint_outcome
from dynisafety.feasibility.classify import (
    FeasibilityThresholds,
    classify_cell,
    classify_task_level,
    geometry_invalid_from_extra,
    oracle_controller_failed,
)


@dataclass
class CellMetrics:
    backend: str
    model: str
    suite: str
    task_id: int
    task_name: str
    level: int
    condition: str  # base | oracle_swr | oracle_swr_aegis
    n: int
    sr: float
    cr: float
    ssr: float
    ss: int
    us: int
    cff: int
    cif: int
    oracle_pause_steps_mean: float
    policy_infer_steps_mean: float
    aegis_intervened_steps_mean: float
    stop_rate: float | None
    resume_rate: float | None
    geometry_invalid: bool
    oracle_failure_reason: str | None
    feasibility_class: str
    feasibility_reason: str
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs)) / float(len(xs))


def _quadrant_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, int, int, int]:
    ss = us = cff = cif = 0
    for r in rows:
        label = classify_joint_outcome(
            success=bool(r["success"]), contact_steps=int(r["contact_steps"])
        )
        if label == "SS":
            ss += 1
        elif label == "US":
            us += 1
        elif label == "CFF":
            cff += 1
        else:
            cif += 1
    return len(rows), ss, us, cff, cif


def _condition_from_record(ep: EpisodeRecord | Mapping[str, Any]) -> str:
    if isinstance(ep, EpisodeRecord):
        extra = ep.extra if isinstance(ep.extra, dict) else {}
        level = int(ep.level)
        policy_kind = None
    else:
        extra = ep.get("extra") if isinstance(ep.get("extra"), dict) else {}
        extra = extra or {}
        level = int(ep.get("level", -1))
        policy_kind = ep.get("policy_kind") or ep.get("condition") or ep.get(
            "feasibility_condition"
        )

    for key in ("feasibility_condition", "condition", "policy_kind"):
        raw = extra.get(key)
        if raw is None and key == "policy_kind":
            raw = policy_kind
        if raw is None and isinstance(ep, Mapping) and key != "policy_kind":
            raw = ep.get(key)
        if raw is None:
            continue
        s = str(raw).strip().lower()
        if s in {"base", "default"}:
            return "base"
        if s in {"oracle_swr_aegis", "oracle+aegis", "oracle_aegis"}:
            return "oracle_swr_aegis"
        if s in {"oracle", "oracle_swr", "oracle_stop_wait_resume", "swr"}:
            return "oracle_swr"
    # Infer from session flags when feasibility_condition was not stamped.
    aegis_on = extra.get("aegis_enabled") is True
    oracle_on = extra.get("oracle_enabled") is True
    if oracle_on and aegis_on:
        return "oracle_swr_aegis"
    if oracle_on:
        return "oracle_swr"
    if level == 0:
        return "base"
    return "base"


def _as_episode_view(ep: EpisodeRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(ep, EpisodeRecord):
        return {
            "backend": ep.backend,
            "model": ep.model,
            "suite": ep.suite,
            "task_id": ep.task_id,
            "task_name": ep.task_name,
            "level": ep.level,
            "success": ep.success,
            "contact_steps": ep.contact_steps,
            "extra": dict(ep.extra) if isinstance(ep.extra, dict) else {},
            "condition": _condition_from_record(ep),
        }
    extra = ep.get("extra") if isinstance(ep.get("extra"), dict) else {}
    return {
        "backend": str(ep.get("backend", "")),
        "model": str(ep.get("model", "")),
        "suite": str(ep.get("suite", "")),
        "task_id": int(ep.get("task_id", -1)),
        "task_name": str(ep.get("task_name", "")),
        "level": int(ep.get("level", -1)),
        "success": bool(ep.get("success", False)),
        "contact_steps": int(ep.get("contact_steps", 0)),
        "extra": dict(extra or {}),
        "condition": _condition_from_record(ep),
    }


def load_episode_dicts_from_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load episodes as dicts; supports full EpisodeRecord JSONL or slim fixtures."""
    out: list[dict[str, Any]] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            if "schema_version" in data:
                try:
                    rec = EpisodeRecord.from_json_dict(data)
                    out.append(_as_episode_view(rec))
                    continue
                except Exception:
                    pass
            required = ("success", "contact_steps", "level", "task_id")
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing keys {missing}")
            out.append(_as_episode_view(data))
    return out


def aggregate_episodes_to_cells(
    episodes: Sequence[EpisodeRecord | Mapping[str, Any]],
    *,
    thresholds: FeasibilityThresholds,
) -> list[CellMetrics]:
    """Group by backend/model/suite/task/level/condition → CellMetrics."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for raw in episodes:
        view = _as_episode_view(raw)
        key = (
            view["backend"],
            view["model"],
            view["suite"],
            view["task_id"],
            view["level"],
            view["condition"],
        )
        groups[key].append(view)

    cells: list[CellMetrics] = []
    for key, rows in sorted(groups.items(), key=lambda kv: kv[0]):
        backend, model, suite, task_id, level, condition = key
        n, ss, us, cff, cif = _quadrant_counts(rows)
        sr = float(ss + us) / float(n) if n else 0.0
        cr = float(us + cif) / float(n) if n else 0.0
        ssr = float(ss) / float(n) if n else 0.0

        pause_mean = _mean(
            [float(r["extra"].get("oracle_pause_steps", 0) or 0) for r in rows]
        )
        infer_mean = _mean(
            [float(r["extra"].get("policy_infer_steps", 0) or 0) for r in rows]
        )
        aegis_mean = _mean(
            [float(r["extra"].get("aegis_intervened_steps", 0) or 0) for r in rows]
        )
        resume_rate = _mean(
            [1.0 for r in rows if r["extra"].get("oracle_resume_occurred") is True]
        )
        stop_flags = [
            1.0 for r in rows if float(r["extra"].get("oracle_pause_steps", 0) or 0) > 0
        ]
        if condition in {"oracle_swr", "oracle_swr_aegis"}:
            stop_rate = _mean(stop_flags) if rows else 0.0
        else:
            stop_rate = (
                float(sum(1 for r in rows if int(r["contact_steps"]) == 0)) / float(n)
                if n
                else 0.0
            )

        geom_invalid = any(geometry_invalid_from_extra(r["extra"]) for r in rows)
        fail_reasons = [oracle_controller_failed(r["extra"]) for r in rows]
        fail_reasons = [x for x in fail_reasons if x]
        oracle_fail = fail_reasons[0] if fail_reasons else None

        cls, reason = classify_cell(
            n=n,
            ssr=ssr,
            cr=cr,
            ss=ss,
            oracle_pause_steps_mean=pause_mean,
            stop_rate=stop_rate,
            resume_rate=(
                resume_rate if condition in {"oracle_swr", "oracle_swr_aegis"} else None
            ),
            geometry_invalid=geom_invalid,
            oracle_failure_reason=oracle_fail,
            thresholds=thresholds,
            condition=str(condition),
        )

        cells.append(
            CellMetrics(
                backend=str(backend),
                model=str(model),
                suite=str(suite),
                task_id=int(task_id),
                task_name=str(rows[0]["task_name"]),
                level=int(level),
                condition=str(condition),
                n=n,
                sr=sr,
                cr=cr,
                ssr=ssr,
                ss=ss,
                us=us,
                cff=cff,
                cif=cif,
                oracle_pause_steps_mean=pause_mean,
                policy_infer_steps_mean=infer_mean,
                aegis_intervened_steps_mean=aegis_mean,
                stop_rate=stop_rate,
                resume_rate=(
                    resume_rate if condition in {"oracle_swr", "oracle_swr_aegis"} else None
                ),
                geometry_invalid=geom_invalid,
                oracle_failure_reason=oracle_fail,
                feasibility_class=cls.value,
                feasibility_reason=reason,
            )
        )
    return cells


def build_base_vs_oracle_table(cells: Sequence[CellMetrics]) -> list[dict[str, Any]]:
    """Wide table: one row per task×level with Base / Oracle / Aegis metrics."""
    by_key: dict[tuple[Any, ...], dict[str, CellMetrics]] = defaultdict(dict)
    for cell in cells:
        key = (cell.backend, cell.model, cell.suite, cell.task_id, cell.level)
        by_key[key][cell.condition] = cell

    rows: list[dict[str, Any]] = []
    for key in sorted(by_key.keys()):
        backend, model, suite, task_id, level = key
        bucket = by_key[key]
        base = bucket.get("base")
        ora = bucket.get("oracle_swr")
        aeg = bucket.get("oracle_swr_aegis")
        name = (base or ora or aeg).task_name if (base or ora or aeg) else ""
        rows.append(
            {
                "backend": backend,
                "model": model,
                "suite": suite,
                "task_id": task_id,
                "task_name": name,
                "level": level,
                "N_base": base.n if base else 0,
                "N_oracle": ora.n if ora else 0,
                "N_aegis": aeg.n if aeg else 0,
                "Base_SR": base.sr if base else None,
                "Base_CR": base.cr if base else None,
                "Base_SSR": base.ssr if base else None,
                "Base_SS": base.ss if base else None,
                "Base_US": base.us if base else None,
                "Base_CFF": base.cff if base else None,
                "Base_CIF": base.cif if base else None,
                "Oracle_SR": ora.sr if ora else None,
                "Oracle_CR": ora.cr if ora else None,
                "Oracle_SSR": ora.ssr if ora else None,
                "Oracle_SS": ora.ss if ora else None,
                "Oracle_US": ora.us if ora else None,
                "Oracle_CFF": ora.cff if ora else None,
                "Oracle_CIF": ora.cif if ora else None,
                "Oracle_pause_steps_mean": ora.oracle_pause_steps_mean if ora else None,
                "Oracle_policy_infer_steps_mean": (
                    ora.policy_infer_steps_mean if ora else None
                ),
                "Aegis_SR": aeg.sr if aeg else None,
                "Aegis_CR": aeg.cr if aeg else None,
                "Aegis_SSR": aeg.ssr if aeg else None,
                "Aegis_SS": aeg.ss if aeg else None,
                "Aegis_US": aeg.us if aeg else None,
                "Aegis_CFF": aeg.cff if aeg else None,
                "Aegis_CIF": aeg.cif if aeg else None,
                "Aegis_pause_steps_mean": (
                    aeg.oracle_pause_steps_mean if aeg else None
                ),
                "Aegis_intervened_steps_mean": (
                    aeg.aegis_intervened_steps_mean if aeg else None
                ),
                "delta_CR_oracle_vs_base": (ora.cr - base.cr) if (ora and base) else None,
                "delta_SSR_oracle_vs_base": (
                    (ora.ssr - base.ssr) if (ora and base) else None
                ),
                "delta_CR_aegis_vs_oracle": (aeg.cr - ora.cr) if (aeg and ora) else None,
                "delta_SSR_aegis_vs_oracle": (
                    (aeg.ssr - ora.ssr) if (aeg and ora) else None
                ),
                # Backward-compatible aliases used by Prompt 14 fixtures/tests.
                "delta_CR": (ora.cr - base.cr) if (ora and base) else None,
                "delta_SSR": (ora.ssr - base.ssr) if (ora and base) else None,
            }
        )
    return rows


def build_feasibility_matrix(
    cells: Sequence[CellMetrics],
    *,
    thresholds: FeasibilityThresholds,
) -> list[dict[str, Any]]:
    """Task×level feasibility class using base / oracle / aegis evidence."""
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for cell in cells:
        key = (cell.backend, cell.model, cell.suite, cell.task_id, cell.level)
        by_key[key][cell.condition] = cell.as_dict()

    rows: list[dict[str, Any]] = []
    for key in sorted(by_key.keys()):
        backend, model, suite, task_id, level = key
        bucket = by_key[key]
        base = bucket.get("base")
        ora = bucket.get("oracle_swr")
        aeg = bucket.get("oracle_swr_aegis")
        cls, reason = classify_task_level(
            base=base, oracle=ora, aegis=aeg, thresholds=thresholds
        )
        name = (base or ora or aeg or {}).get("task_name", "")
        rows.append(
            {
                "backend": backend,
                "model": model,
                "suite": suite,
                "task_id": task_id,
                "task_name": name,
                "level": level,
                "feasibility_class": cls.value,
                "feasibility_reason": reason,
                "base_ssr": None if base is None else base.get("ssr"),
                "oracle_ssr": None if ora is None else ora.get("ssr"),
                "aegis_ssr": None if aeg is None else aeg.get("ssr"),
                "base_cr": None if base is None else base.get("cr"),
                "oracle_cr": None if ora is None else ora.get("cr"),
                "aegis_cr": None if aeg is None else aeg.get("cr"),
                "oracle_pause_steps_mean": None
                if ora is None
                else ora.get("oracle_pause_steps_mean"),
                "aegis_intervened_steps_mean": None
                if aeg is None
                else aeg.get("aegis_intervened_steps_mean"),
                "oracle_failure_reason": None
                if ora is None
                else ora.get("oracle_failure_reason"),
                "geometry_invalid": bool(
                    (base or {}).get("geometry_invalid")
                    or (ora or {}).get("geometry_invalid")
                    or (aeg or {}).get("geometry_invalid")
                ),
            }
        )
    return rows
