"""Matrix loading, job expansion, and resource estimates (no GPU runs here)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dynisafety.feasibility.classify import FeasibilityThresholds

VALID_CONDITIONS = frozenset({"base", "oracle_swr", "oracle_swr_aegis"})


@dataclass(frozen=True)
class FeasibilityJob:
    backend: str
    model: str
    suite: str
    task_id: int
    level: int
    condition: str  # base | oracle_swr | oracle_swr_aegis
    episodes: int
    seed: int
    oracle_config: str | None
    aegis_config: str | None = None


@dataclass(frozen=True)
class ResourceEstimate:
    n_jobs: int
    n_episodes_total: int
    gpu_hours_low: float
    gpu_hours_high: float
    wall_hours_one_gpu_low: float
    wall_hours_one_gpu_high: float
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_jobs": self.n_jobs,
            "n_episodes_total": self.n_episodes_total,
            "gpu_hours_low": self.gpu_hours_low,
            "gpu_hours_high": self.gpu_hours_high,
            "wall_hours_one_gpu_low": self.wall_hours_one_gpu_low,
            "wall_hours_one_gpu_high": self.wall_hours_one_gpu_high,
            "notes": list(self.notes),
        }


def load_feasibility_matrix(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"feasibility matrix root must be mapping: {path}")
    return data


def thresholds_from_matrix(data: Mapping[str, Any]) -> FeasibilityThresholds:
    raw = data.get("thresholds")
    if not isinstance(raw, Mapping):
        raise ValueError("feasibility matrix missing thresholds: block")
    for key in (
        "min_episodes",
        "ssr_feasible",
        "cr_stop_only_max",
        "pause_steps_stop_evidence",
    ):
        if key not in raw:
            raise ValueError(f"thresholds.{key} required in matrix YAML")
    return FeasibilityThresholds(
        min_episodes=int(raw["min_episodes"]),
        ssr_feasible=float(raw["ssr_feasible"]),
        cr_stop_only_max=float(raw["cr_stop_only_max"]),
        pause_steps_stop_evidence=float(raw["pause_steps_stop_evidence"]),
        require_geometry_flags_for_invalid=bool(
            raw.get("require_geometry_flags_for_invalid", True)
        ),
    )


def _config_path(block: Mapping[str, Any] | None, key: str = "config_path") -> str | None:
    if not isinstance(block, Mapping):
        return None
    raw = block.get(key)
    if raw is None or raw == "":
        return None
    return str(raw)


def expand_matrix_jobs(data: Mapping[str, Any]) -> list[FeasibilityJob]:
    """Expand matrix YAML into concrete eval jobs (base / oracle_swr / oracle_swr_aegis)."""
    episodes = int(data["episodes_per_task"])
    seeds = [int(s) for s in data["seeds"]]
    default_conditions = [str(c) for c in data["conditions"]]
    for c in default_conditions:
        if c not in VALID_CONDITIONS:
            raise ValueError(
                f"unknown condition {c!r}; expected one of {sorted(VALID_CONDITIONS)}"
            )

    oracle_cfg = _config_path(data.get("oracle") if isinstance(data.get("oracle"), Mapping) else None)
    aegis_cfg = _config_path(data.get("aegis") if isinstance(data.get("aegis"), Mapping) else None)

    jobs: list[FeasibilityJob] = []
    for entry in data.get("matrix") or []:
        backend = str(entry["backend"])
        model = str(entry["model"])
        suite = str(entry["suite"])
        levels = [int(x) for x in entry["levels"]]
        tasks = [int(x) for x in entry["tasks"]]
        eps = int(entry.get("episodes_per_task", episodes))
        conditions = [str(c) for c in entry.get("conditions", default_conditions)]
        for c in conditions:
            if c not in VALID_CONDITIONS:
                raise ValueError(
                    f"unknown condition {c!r}; expected one of {sorted(VALID_CONDITIONS)}"
                )
        for level in levels:
            for task_id in tasks:
                for condition in conditions:
                    for seed in seeds:
                        jobs.append(
                            FeasibilityJob(
                                backend=backend,
                                model=model,
                                suite=suite,
                                task_id=task_id,
                                level=level,
                                condition=condition,
                                episodes=eps,
                                seed=seed,
                                oracle_config=(
                                    oracle_cfg
                                    if condition in {"oracle_swr", "oracle_swr_aegis"}
                                    else None
                                ),
                                aegis_config=(
                                    aegis_cfg if condition == "oracle_swr_aegis" else None
                                ),
                            )
                        )
    return jobs


def estimate_matrix_resources(
    jobs: list[FeasibilityJob],
    *,
    seconds_per_episode_low: float,
    seconds_per_episode_high: float,
) -> ResourceEstimate:
    """Rough GPU-hour estimate; does not start any run."""
    n_eps = sum(j.episodes for j in jobs)
    low_h = n_eps * float(seconds_per_episode_low) / 3600.0
    high_h = n_eps * float(seconds_per_episode_high) / 3600.0
    notes = (
        "Estimate only — actual wall time depends on VLA forward, sim, and queueing.",
        "Formal GPU runs require explicit authorization (BLOCKED_UNTIL_AUTHORIZED).",
        "Oracle_swr: pause VLA during dynamic disturbance; aegis must stay off.",
        "Oracle_swr_aegis: same pause, then AEGIS CBF-QP after settle (static obstacle).",
        "Closing aegis must not change base / oracle_swr regression semantics.",
        "L1/L2 low SSR after CR drop is often static-obstacle residual without avoidance.",
    )
    return ResourceEstimate(
        n_jobs=len(jobs),
        n_episodes_total=n_eps,
        gpu_hours_low=round(low_h, 2),
        gpu_hours_high=round(high_h, 2),
        wall_hours_one_gpu_low=round(low_h, 2),
        wall_hours_one_gpu_high=round(high_h, 2),
        notes=notes,
    )


def job_to_eval_command(job: FeasibilityJob, *, max_steps: int | None = None) -> str:
    """Render a ``dynisafety eval`` command line for one cell.

    Always sets both oracle and aegis switches explicitly so ``aegis.enabled=false``
    cannot silently differ across base / oracle_swr jobs.
    """
    parts = [
        "dynisafety eval",
        f"--backend {job.backend}",
        f"--model {job.model}",
        f"--suite {job.suite}",
        f"--task {job.task_id}",
        f"--level {job.level}",
        f"--seed {job.seed}",
        f"--episodes {job.episodes}",
    ]
    if job.condition == "base":
        parts.append("--set extra.libero.oracle.enabled=false")
        parts.append("--set extra.libero.aegis.enabled=false")
    elif job.condition == "oracle_swr":
        cfg = job.oracle_config or "configs/backends/libero/oracle/stop_wait_resume.yaml"
        parts.append(f"--config {cfg}")
        parts.append("--set extra.libero.oracle.enabled=true")
        parts.append("--set extra.libero.aegis.enabled=false")
    elif job.condition == "oracle_swr_aegis":
        ocfg = job.oracle_config or "configs/backends/libero/oracle/stop_wait_resume.yaml"
        acfg = job.aegis_config or "configs/backends/libero/aegis/enable.yaml"
        parts.append(f"--config {ocfg}")
        parts.append(f"--config {acfg}")
        parts.append("--set extra.libero.oracle.enabled=true")
        parts.append("--set extra.libero.aegis.enabled=true")
    else:
        raise ValueError(f"unsupported condition {job.condition!r}")
    if max_steps is not None:
        parts.append(f"--set runtime.max_steps={max_steps}")
    parts.append(f"--set extra.feasibility_condition={job.condition}")
    return " ".join(parts)
