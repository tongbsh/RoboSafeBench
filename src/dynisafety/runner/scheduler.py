"""Sweep suite expansion, GPU scheduling, server lifecycle, resume / failures."""

from __future__ import annotations

import functools
import json
import os
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dynisafety.config.resolve import (
    ConfigResolveError,
    default_third_party_root,
    find_workspace_root,
    resolve_placeholders,
)
from dynisafety.policies.base import resolve_checkpoint

# Loopback the scheduler itself polls / the eval worker dials. This is a
# transport detail of the local process pair, not a model hyperparameter, so it
# is not subject to iron law 1's "no defaults in Python".
_LOOPBACK_HOST = "127.0.0.1"

# BLAS / OpenMP pool size inside eval workers. Like _LOOPBACK_HOST this is a
# process-runtime detail rather than a model hyperparameter.
#
# A rollout is a serial loop over 7-DoF actions and 128x128 frames; the heavy
# work is MuJoCo / SAPIEN stepping and offscreen rendering, none of which goes
# through BLAS. A wide pool therefore only spins on barriers. Measured on
# LIBERO with an inherited OMP_NUM_THREADS=60: three helper threads burned an
# extra 1.55 cores per worker at identical wall-clock per step, so the box
# saturated at 6 workers while 92% of 60 cores sat idle. Pinning to 1 left
# ms/step unchanged (198 vs 199 on long-t2-L0) at 0.90 cores per worker.
_EVAL_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def eval_thread_count() -> str:
    """Threads per eval worker; ``DYNI_EVAL_THREADS`` overrides the pinned 1."""
    raw = os.environ.get("DYNI_EVAL_THREADS", "").strip()
    if not raw:
        return "1"
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"DYNI_EVAL_THREADS must be a positive int, got {raw!r}")
    return raw


def pin_eval_threads(env: dict[str, str]) -> None:
    """Override any inherited BLAS/OpenMP pool size in an eval worker's env."""
    threads = eval_thread_count()
    for key in _EVAL_THREAD_ENV_KEYS:
        env[key] = threads


@functools.lru_cache(maxsize=1)
def _conda_env_prefixes() -> dict[str, Path]:
    """env name → prefix from ``conda env list`` (honours ``envs_dirs``)."""
    conda = shutil.which("conda")
    if conda is None:
        return {}
    try:
        proc = subprocess.run(
            [conda, "env", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {}
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    out: dict[str, Path] = {}
    for raw in payload.get("envs") or []:
        prefix = Path(raw)
        if prefix.is_dir():
            out[prefix.name] = prefix
    return out


def _conda_env_lib_dir(env_name: str) -> Path | None:
    """Return ``…/envs/<name>/lib`` when present (prefer autodl conda root).

    Sweep tool env often prepends *base* ``libstdc++`` via ``LD_LIBRARY_PATH``.
    ``conda run -n <eval_env>`` inherits that and breaks tokenizers/ICU
    (``CXXABI_… not found``). Callers should prepend this path for eval/serve.

    Named envs may live on a data disk (``envs_dirs``), not under
    ``~/miniconda3/envs``. Fall back to conda's registered prefixes.
    """
    if not isinstance(env_name, str) or not env_name.strip() or env_name == "base":
        return None
    roots: list[Path] = []
    for key in ("DYNI_CONDA_ROOT", "CONDA_ROOT", "CONDA_PREFIX"):
        raw = os.environ.get(key)
        if not raw:
            continue
        p = Path(raw)
        # base prefix has envs/; activated env prefix's parent may be envs/.
        if (p / "envs").is_dir():
            roots.append(p)
        elif p.parent.name == "envs":
            roots.append(p.parent.parent)
    roots.extend(
        [
            Path.home() / "miniconda3",
            Path.home() / "anaconda3",
        ]
    )
    seen: set[Path] = set()
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        if root in seen:
            continue
        seen.add(root)
        lib = root / "envs" / env_name / "lib"
        if lib.is_dir():
            return lib
    registered = _conda_env_prefixes().get(env_name)
    if registered is not None:
        lib = registered / "lib"
        if lib.is_dir():
            return lib
    return None


def _prefer_conda_env_libs(env: dict[str, str], env_name: str | None) -> None:
    if not env_name:
        return
    lib = _conda_env_lib_dir(env_name)
    if lib is None:
        return
    prev = env.get("LD_LIBRARY_PATH", "")
    prefix = str(lib)
    if prev:
        # Drop duplicate leading entries of the same path.
        parts = [p for p in prev.split(os.pathsep) if p and p != prefix]
        env["LD_LIBRARY_PATH"] = os.pathsep.join([prefix, *parts])
    else:
        env["LD_LIBRARY_PATH"] = prefix


# ---------------------------------------------------------------------------
# Suite / job matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixBlock:
    backend: str
    suites: tuple[str, ...]
    tasks: tuple[int, ...] | None
    levels: tuple[int, ...]
    models: tuple[str, ...]
    episodes_per_task: int | None
    seeds: tuple[int, ...] | None


@dataclass(frozen=True)
class SweepSuite:
    name: str
    episodes_per_task: int
    seeds: tuple[int, ...]
    matrix: tuple[MatrixBlock, ...]
    path: Path


@dataclass(frozen=True)
class JobSpec:
    backend: str
    suite: str
    task_id: int
    task_name: str
    level: int
    model: str
    seed: int
    episodes: int

    @property
    def job_key(self) -> str:
        return (
            f"{self.backend}|{self.model}|{self.suite}|"
            f"{self.task_id}|{self.level}|{self.seed}"
        )


@dataclass(frozen=True)
class OnlyFilter:
    clauses: tuple[tuple[str, str], ...]

    def match(self, job: JobSpec) -> bool:
        if not self.clauses:
            return True
        fields = {
            "backend": str(job.backend),
            "model": str(job.model),
            "suite": str(job.suite),
            "task": str(job.task_id),
            "task_id": str(job.task_id),
            "level": str(job.level),
            "seed": str(job.seed),
        }
        for key, value in self.clauses:
            if key not in fields:
                return False
            allowed = {item for item in value.split("|") if item}
            if not allowed or fields[key] not in allowed:
                return False
        return True


def parse_only_filter(spec: str | None) -> OnlyFilter:
    """Parse ``model=X,level=3`` into equality clauses."""
    if spec is None or not str(spec).strip():
        return OnlyFilter(clauses=())
    clauses: list[tuple[str, str]] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid --only clause {part!r}; expected key=value")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"invalid --only clause {part!r}")
        clauses.append((key, value))
    return OnlyFilter(clauses=tuple(clauses))


def load_sweep_suite(path: Path) -> SweepSuite:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"sweep suite root must be mapping: {path}")
    name = raw.get("name")
    if not name:
        raise ValueError(f"sweep suite missing name: {path}")
    episodes = raw.get("episodes_per_task")
    seeds = raw.get("seeds")
    matrix_raw = raw.get("matrix")
    if not isinstance(episodes, int):
        raise ValueError(f"{path}: episodes_per_task must be int")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{path}: seeds must be non-empty list")
    if not isinstance(matrix_raw, list) or not matrix_raw:
        raise ValueError(f"{path}: matrix must be non-empty list")
    blocks: list[MatrixBlock] = []
    for i, block in enumerate(matrix_raw):
        if not isinstance(block, dict):
            raise ValueError(f"{path}: matrix[{i}] must be mapping")
        tasks_raw = block.get("tasks")
        tasks: tuple[int, ...] | None
        if tasks_raw is None:
            tasks = None
        else:
            if not isinstance(tasks_raw, list) or not tasks_raw:
                raise ValueError(f"{path}: matrix[{i}].tasks must be non-empty list of ints")
            tasks = tuple(int(t) for t in tasks_raw)
        block_eps = block.get("episodes_per_task")
        block_seeds = block.get("seeds")
        blocks.append(
            MatrixBlock(
                backend=str(block["backend"]),
                suites=tuple(str(s) for s in block["suites"]),
                tasks=tasks,
                levels=tuple(int(x) for x in block["levels"]),
                models=tuple(str(m) for m in block["models"]),
                episodes_per_task=int(block_eps) if block_eps is not None else None,
                seeds=tuple(int(s) for s in block_seeds) if block_seeds is not None else None,
            )
        )
    return SweepSuite(
        name=str(name),
        episodes_per_task=episodes,
        seeds=tuple(int(s) for s in seeds),
        matrix=tuple(blocks),
        path=path.resolve(),
    )


def _tasks_from_backend_suite_yaml(
    configs_root: Path,
    backend: str,
    suite: str,
) -> list[tuple[int, str]]:
    """Resolve (task_id, task_name) when matrix.tasks is omitted."""
    suite_path = configs_root / "backends" / backend / "suites" / f"{suite}.yaml"
    if suite_path.is_file():
        data = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
        tasks = data.get("tasks")
        if isinstance(tasks, list) and tasks:
            out: list[tuple[int, str]] = []
            for item in tasks:
                if isinstance(item, dict):
                    out.append((int(item["task_id"]), str(item["name"])))
                else:
                    raise ValueError(f"{suite_path}: task entries must be mappings with task_id/name")
            return out
    # Fallback: task YAML stems under backends/{b}/tasks/ matching suite prefix.
    tasks_dir = configs_root / "backends" / backend / "tasks"
    if tasks_dir.is_dir():
        stems = sorted(p.stem for p in tasks_dir.glob("*.yaml"))
        if suite != "bridge":
            stems = [s for s in stems if s.startswith(suite) or suite in s]
        if stems:
            return [(i, name) for i, name in enumerate(stems)]
    raise ValueError(
        f"cannot resolve tasks for backend={backend!r} suite={suite!r}; "
        f"set matrix.tasks or add tasks to {suite_path}"
    )


def expand_jobs(suite: SweepSuite, *, configs_root: Path) -> list[JobSpec]:
    """Cartesian expand sweep suite → concrete JobSpec list."""
    configs_root = Path(configs_root)
    jobs: list[JobSpec] = []
    for block in suite.matrix:
        episodes = (
            block.episodes_per_task
            if block.episodes_per_task is not None
            else suite.episodes_per_task
        )
        seeds = block.seeds if block.seeds is not None else suite.seeds
        for suite_name in block.suites:
            if block.tasks is not None:
                task_pairs = [(tid, f"task_{tid}") for tid in block.tasks]
                # Prefer names from suite YAML when present.
                try:
                    named = {
                        tid: name
                        for tid, name in _tasks_from_backend_suite_yaml(
                            configs_root, block.backend, suite_name
                        )
                    }
                    task_pairs = [(tid, named.get(tid, f"task_{tid}")) for tid in block.tasks]
                except ValueError:
                    pass
            else:
                task_pairs = _tasks_from_backend_suite_yaml(
                    configs_root, block.backend, suite_name
                )
            for task_id, task_name in task_pairs:
                for level in block.levels:
                    for model in block.models:
                        for seed in seeds:
                            jobs.append(
                                JobSpec(
                                    backend=block.backend,
                                    suite=suite_name,
                                    task_id=task_id,
                                    task_name=task_name,
                                    level=level,
                                    model=model,
                                    seed=seed,
                                    episodes=episodes,
                                )
                            )
    return jobs


def estimate_totals(jobs: Sequence[JobSpec]) -> dict[str, Any]:
    per_backend: dict[str, int] = {}
    for job in jobs:
        per_backend[job.backend] = per_backend.get(job.backend, 0) + 1
    return {
        "n_jobs": len(jobs),
        "n_episodes_total": sum(j.episodes for j in jobs),
        "per_backend_jobs": per_backend,
    }


# ---------------------------------------------------------------------------
# Model serve / schedule spec (fields temporarily in model YAML; M4 owns schema)
# ---------------------------------------------------------------------------


# Format vars every serve argv can use; anything else must come from
# server.per_suite.<suite>.
_RUNTIME_SERVE_VARS = frozenset({"port", "gpu", "checkpoint", "model", "suite"})


def _argv_placeholders(argv: Sequence[str]) -> list[str]:
    out: list[str] = []
    for part in argv:
        for _, field, _, _ in string.Formatter().parse(part):
            if field:
                out.append(field.split(".")[0].split("[")[0])
    return out


@dataclass(frozen=True)
class ModelServeSpec:
    name: str
    model_id: str
    suite: str | None
    checkpoint: str
    checkpoint_error: str | None
    action_space: str
    extra: dict[str, Any]
    max_jobs_per_gpu: int | None
    eval_env: str | None
    serve_env: str | None
    needs_server: bool
    serve_command: tuple[str, ...]
    serve_cwd: str | None
    serve_vars: dict[str, str]
    serve_process_env: dict[str, str]
    health_url_template: str | None
    missing_fields: tuple[str, ...]


def _empty_serve_spec(model: str, missing: tuple[str, ...]) -> ModelServeSpec:
    return ModelServeSpec(
        name=model,
        model_id=model,
        suite=None,
        checkpoint="",
        checkpoint_error=None,
        action_space="",
        extra={},
        max_jobs_per_gpu=None,
        eval_env=None,
        serve_env=None,
        needs_server=False,
        serve_command=(),
        serve_cwd=None,
        serve_vars={},
        serve_process_env={},
        health_url_template=None,
        missing_fields=missing,
    )


def _build_serve_argv(
    server: Mapping[str, Any],
    *,
    path: Path,
    suite: str | None,
) -> tuple[tuple[str, ...], str | None, dict[str, str], list[str]]:
    """Assemble the server argv from the M4 ``server:`` block.

    Returns ``(argv_template, cwd, format_vars, missing)``. ``argv_template``
    still carries ``{port}`` / ``{gpu}`` / ``{checkpoint}`` placeholders; those
    are only known when a server is actually started.

    ``server.per_suite.<suite>`` supplies extra format variables (e.g. UniVLA's
    per-suite ``unnorm_key``) and may override ``args`` wholesale for suites
    that need a different launch mode.
    """
    missing: list[str] = []
    root = server.get("root")
    launch = server.get("launch")
    if not isinstance(root, str) or not root.strip():
        missing.append("server.root")
    if not isinstance(launch, str) or not launch.strip():
        missing.append("server.launch")

    args = server.get("args")
    fmt_vars: dict[str, str] = {}
    per_suite = server.get("per_suite")
    if per_suite is not None:
        if not isinstance(per_suite, Mapping):
            raise ValueError(f"{path}: server.per_suite must be a mapping")
        if suite is not None:
            overrides = per_suite.get(suite)
            if overrides is None:
                # Loud: a per-suite server config that does not cover this suite
                # would otherwise silently serve the wrong weights.
                missing.append(
                    f"server.per_suite[{suite}] (known: {sorted(per_suite)})"
                )
            elif not isinstance(overrides, Mapping):
                raise ValueError(
                    f"{path}: server.per_suite[{suite}] must be a mapping"
                )
            else:
                if "args" in overrides:
                    args = overrides["args"]
                fmt_vars = {
                    str(k): str(v) for k, v in overrides.items() if k != "args"
                }

    if args is None:
        missing.append("server.args")
        args = []
    if not isinstance(args, list):
        raise ValueError(f"{path}: server.args must be a list of strings")

    if missing:
        return (), None, fmt_vars, missing

    script = str(Path(str(root)) / str(launch))
    argv = ("python", script, *(str(a) for a in args))

    # Catch `{unnorm_key}` style holes now rather than after conda + model load.
    unresolved = sorted(
        set(_argv_placeholders(argv)) - _RUNTIME_SERVE_VARS - set(fmt_vars)
    )
    if unresolved:
        missing.append(
            "server.args placeholders " + ", ".join(f"{{{n}}}" for n in unresolved)
        )
    return argv, str(root), fmt_vars, missing


def load_model_serve_spec(
    model: str,
    *,
    configs_root: Path,
    suite: str | None = None,
    backend: str | None = None,
) -> ModelServeSpec:
    path = Path(configs_root) / "models" / f"{model}.yaml"
    missing: list[str] = []
    if not path.is_file():
        return _empty_serve_spec(model, ("model_yaml", "max_jobs_per_gpu"))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"model YAML root must be mapping: {path}")

    # Same placeholder pass the eval path runs, so serve argv and checkpoints
    # come out as real filesystem paths instead of `${workspace}/...` literals.
    workspace = find_workspace_root()
    try:
        data = resolve_placeholders(
            data,
            workspace=workspace,
            third_party=default_third_party_root(workspace),
        )
    except ConfigResolveError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    # Shared models (starvla) override eval_env / action_space per backend.
    from dynisafety.policies.base import apply_per_backend  # noqa: PLC0415

    data = apply_per_backend(data, backend)

    # Scheduling keys may live under extra: so M1 loader (unknown-key reject) still works.
    # max_jobs_per_gpu canonical home is resources.* (M4); keep top-level / extra as fallbacks.
    extra = dict(data.get("extra") or {})

    def _get(key: str) -> Any:
        if key in data and data[key] is not None:
            return data[key]
        return extra.get(key)

    mj = _resolve_max_jobs_per_gpu(data, extra)
    if mj is None:
        missing.append("max_jobs_per_gpu")

    if _get("needs_server") is None and "needs_server" not in data and "needs_server" not in extra:
        missing.append("needs_server")
        needs = False
    else:
        needs = bool(_get("needs_server"))

    checkpoint = ""
    checkpoint_error: str | None = None
    if "checkpoint" in data:
        try:
            checkpoint = resolve_checkpoint(data, suite=suite)
        except ValueError as exc:
            checkpoint_error = str(exc)

    # Flat serve_command / health_url_template stay supported as an escape hatch;
    # the canonical source is the M4 `server:` block.
    flat_cmd = _get("serve_command")
    if flat_cmd is not None and not isinstance(flat_cmd, list):
        raise ValueError(f"{path}: serve_command must be a list")
    health = _get("health_url_template")

    argv: tuple[str, ...] = ()
    serve_cwd: str | None = None
    serve_vars: dict[str, str] = {}
    argv_missing: list[str] = []
    server = data.get("server")
    if flat_cmd:
        argv = tuple(str(x) for x in flat_cmd)
    elif isinstance(server, Mapping):
        argv, serve_cwd, serve_vars, argv_missing = _build_serve_argv(
            server, path=path, suite=suite
        )
        if needs:
            missing.extend(argv_missing)
        if health is None:
            health_path = server.get("health_path")
            if isinstance(health_path, str) and health_path.strip():
                host = str(extra.get("server_host") or _LOOPBACK_HOST)
                health = f"http://{host}:{{port}}{health_path}"
    elif needs:
        missing.append("server")

    # Serve argv/health required only for server models (and for `dynisafety serve`).
    if needs:
        if not argv and "server" not in missing and not argv_missing:
            missing.append("serve_command")
        if not health:
            missing.append("health_url_template")

    eval_env = _get("eval_env")
    serve_env = _get("serve_env")
    if serve_env is None and isinstance(server, Mapping):
        serve_env = server.get("env")

    serve_process_env: dict[str, str] = {}
    if isinstance(server, Mapping) and server.get("process_env") is not None:
        raw_penv = server.get("process_env")
        if not isinstance(raw_penv, Mapping):
            raise ValueError(f"{path}: server.process_env must be a mapping")
        serve_process_env = {str(k): str(v) for k, v in raw_penv.items()}

    return ModelServeSpec(
        name=str(data.get("name", model)),
        model_id=model,
        suite=suite,
        checkpoint=checkpoint,
        checkpoint_error=checkpoint_error,
        action_space=str(data.get("action_space", "")),
        extra=extra,
        max_jobs_per_gpu=int(mj) if mj is not None else None,
        eval_env=str(eval_env) if eval_env is not None else None,
        serve_env=str(serve_env) if serve_env is not None else None,
        needs_server=needs,
        serve_command=argv,
        serve_cwd=serve_cwd,
        serve_vars=serve_vars,
        serve_process_env=serve_process_env,
        health_url_template=str(health) if health is not None else None,
        missing_fields=tuple(missing),
    )


def _resolve_max_jobs_per_gpu(
    data: Mapping[str, Any], extra: Mapping[str, Any] | None = None
) -> Any:
    """Locate ``max_jobs_per_gpu`` without inventing defaults (铁律 1).

    Lookup order matches ``policies.base._parse_max_jobs_per_gpu`` plus the M3
    transitional top-level key: ``resources`` → top-level → ``extra``.
    """
    resources = data.get("resources")
    if isinstance(resources, Mapping) and resources.get("max_jobs_per_gpu") is not None:
        return resources["max_jobs_per_gpu"]
    if data.get("max_jobs_per_gpu") is not None:
        return data["max_jobs_per_gpu"]
    if extra is None:
        raw_extra = data.get("extra")
        extra = raw_extra if isinstance(raw_extra, Mapping) else {}
    return extra.get("max_jobs_per_gpu")


def servers_per_gpu_count(max_jobs_per_gpu: int) -> int:
    """How many serve processes to start on each GPU.

    Default 1: all eval slots on that GPU share one loaded model.
    ``DYNI_SERVERS_PER_GPU`` may raise this up to ``max_jobs_per_gpu`` so each
    slot gets its own weights (needs enough VRAM).
    """
    raw = os.environ.get("DYNI_SERVERS_PER_GPU", "").strip()
    if not raw:
        return 1
    value = int(raw)
    if value < 1:
        raise ValueError(f"DYNI_SERVERS_PER_GPU must be >= 1, got {value}")
    return min(value, max_jobs_per_gpu)


def gpu_slot_order(gpu_ids: Sequence[int], max_jobs_per_gpu: int) -> list[int]:
    """Round-robin GPU slots so a short batch spreads across cards.

    ``max_jobs_per_gpu=2``, gpus=(0, 1, 2) → ``[0, 1, 2, 0, 1, 2]``.
    The old GPU-major pack ``[0, 0, 1, 1, 2, 2]`` left later cards idle
    whenever the batch was smaller than the full slot count.
    """
    if max_jobs_per_gpu < 1:
        raise ValueError(f"max_jobs_per_gpu must be >= 1, got {max_jobs_per_gpu}")
    slots: list[int] = []
    for _ in range(max_jobs_per_gpu):
        slots.extend(int(g) for g in gpu_ids)
    return slots


def gpus_needed_for_jobs(slot_gpus: Sequence[int], n_jobs: int) -> tuple[int, ...]:
    """GPUs that receive at least one job in this group (first-batch occupancy).

    Later batches reuse the same slot prefix, so unused trailing cards never
    get work and must not load a server.
    """
    if not slot_gpus or n_jobs <= 0:
        return ()
    seen: list[int] = []
    for gpu in slot_gpus[: min(int(n_jobs), len(slot_gpus))]:
        if gpu not in seen:
            seen.append(gpu)
    return tuple(seen)


def effective_max_jobs_per_gpu(
    model: str,
    model_yaml: Mapping[str, Any],
    cap: int | None,
) -> int:
    """Read ``max_jobs_per_gpu`` from model YAML; optional global cap."""
    del model  # identity for error messages via caller
    raw = _resolve_max_jobs_per_gpu(model_yaml)
    if raw is None:
        raise ValueError("max_jobs_per_gpu missing from model YAML")
    value = int(raw)
    if value < 1:
        raise ValueError(f"max_jobs_per_gpu must be >= 1, got {value}")
    if cap is not None:
        if cap < 1:
            raise ValueError(f"jobs_per_gpu cap must be >= 1, got {cap}")
        value = min(value, cap)
    return value


# ---------------------------------------------------------------------------
# Ports / servers
# ---------------------------------------------------------------------------


@dataclass
class ServerHandle:
    model: str
    gpu: int
    port: int
    pid: int
    log_path: Path
    process: subprocess.Popen[Any] | None = None


def allocate_port() -> int:
    """Allocate a free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(url: str, timeout_s: float, pid: int) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        if pid > 0 and _pid_dead(pid):
            raise RuntimeError(f"server pid {pid} exited before becoming healthy ({url})")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if 200 <= int(resp.status) < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
        # Also accept plain TCP if URL is http://host:port/...
        try:
            host, port = _host_port_from_url(url)
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_err = exc
        time.sleep(0.5)
    raise TimeoutError(f"health check timed out for {url}: {last_err}")


def _host_port_from_url(url: str) -> tuple[str, int]:
    # http://127.0.0.1:1234/health
    rest = url.split("://", 1)[-1]
    hostport = rest.split("/", 1)[0]
    host, port_s = hostport.rsplit(":", 1)
    return host, int(port_s)


def _pid_dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _format_serve_arg(part: str, fmt: Mapping[str, str], *, model: str) -> str:
    try:
        return part.format(**fmt)
    except KeyError as exc:
        raise ValueError(
            f"model {model}: serve argv {part!r} references unknown placeholder "
            f"{exc.args[0]!r}; available: {sorted(fmt)}. Declare it under "
            "server.per_suite.<suite> in the model YAML."
        ) from exc


def start_model_server(
    spec: ModelServeSpec,
    *,
    gpu: int,
    port: int,
    log_path: Path,
    health_timeout_s: float,
) -> ServerHandle:
    if not spec.needs_server:
        raise ValueError(f"model {spec.name} does not need a server")
    if not spec.serve_command:
        raise ValueError(f"model {spec.name} missing serve_command")
    if spec.checkpoint_error:
        raise ValueError(f"model {spec.name}: {spec.checkpoint_error}")
    fmt = {
        "port": str(port),
        "gpu": str(gpu),
        "checkpoint": spec.checkpoint,
        "model": spec.name,
        "suite": spec.suite or "",
        **spec.serve_vars,
    }
    argv = [_format_serve_arg(part, fmt, model=spec.name) for part in spec.serve_command]
    if spec.serve_env:
        argv = ["conda", "run", "--no-capture-output", "-n", spec.serve_env, *argv]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PORT"] = str(port)
    # Model YAML `server.process_env` (e.g. openpi → disable JAX 75% prealloc).
    # Caller-exported values win over YAML defaults.
    for key, value in spec.serve_process_env.items():
        env.setdefault(key, value)
    _prefer_conda_env_libs(env, spec.serve_env)
    if spec.serve_cwd:
        # Upstream launchers are written to run from their repo root (the old
        # start_*_server.sh scripts all `cd` there first).
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (spec.serve_cwd, env.get("PYTHONPATH")) if p
        )
    with log_path.open("w", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            argv,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=spec.serve_cwd,
            start_new_session=True,
        )
    handle = ServerHandle(
        model=spec.name,
        gpu=gpu,
        port=port,
        pid=int(proc.pid),
        log_path=log_path,
        process=proc,
    )
    if spec.health_url_template:
        url = spec.health_url_template.format(**fmt)
        try:
            wait_for_health(url, health_timeout_s, handle.pid)
        except Exception:
            stop_server(handle)
            raise
    return handle


def stop_server(handle: ServerHandle) -> None:
    proc = handle.process
    if proc is None:
        if handle.pid > 0 and not _pid_dead(handle.pid):
            try:
                os.kill(handle.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Resume / failures
# ---------------------------------------------------------------------------


def _run_id_prefix(job: JobSpec) -> str:
    return f"{job.backend}-{job.model}-{job.suite}-t{job.task_id}-l{job.level}-s{job.seed}-"


def _legacy_run_id_prefix(job: JobSpec) -> str:
    """Pre-task-scoped dirs: ``{backend}-{model}-{suite}-l{level}-s{seed}-``."""
    return f"{job.backend}-{job.model}-{job.suite}-l{job.level}-s{job.seed}-"


def _is_level_bucket_name(name: str) -> bool:
    """True for ``l0`` / ``l12`` style level buckets under results/."""
    return len(name) >= 2 and name[0] == "l" and name[1:].isdigit()


def _iter_run_dir_candidates(results_root: Path, job: JobSpec) -> list[Path]:
    """Match new ``eval/{backend}/{suite}/l{level}/`` plus legacy ``l{level}/`` / flat."""
    prefixes = (_run_id_prefix(job), _legacy_run_id_prefix(job))
    search_roots: list[Path] = []
    try:
        from dynisafety.config.paths import load_paths

        search_roots.append(
            load_paths()
            .with_out(results_root)
            .eval_parent(backend=job.backend, suite=job.suite, level=int(job.level))
        )
    except (OSError, ValueError, KeyError):
        pass
    level_dir = results_root / f"l{int(job.level)}"
    if level_dir.is_dir():
        search_roots.append(level_dir)
    # Flat layout (pre level-bucket): keep scanning so in-flight / old runs resume.
    search_roots.append(results_root)

    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            # When scanning the flat root, skip level buckets themselves.
            if root == results_root and _is_level_bucket_name(path.name):
                continue
            if not any(path.name.startswith(pref) for pref in prefixes):
                continue
            if path in seen:
                continue
            seen.add(path)
            candidates.append(path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def is_job_complete(results_root: Path, job: JobSpec) -> bool:
    """True iff a matching run has summary.status==completed and episode count OK."""
    results_root = Path(results_root)
    if not results_root.is_dir():
        return False
    for run_dir in _iter_run_dir_candidates(results_root, job):
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if summary.get("status") != "completed":
            continue
        if int(summary.get("n_episodes_written", -1)) < job.episodes:
            continue
        # task_id distinguishes legacy runs that share the old run_id prefix.
        if "task_id" in summary and int(summary["task_id"]) != job.task_id:
            continue
        return True
    return False


def append_failure(
    sweep_dir: Path,
    job: JobSpec,
    *,
    exit_code: int,
    log_path: Path,
    error: str,
) -> None:
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    path = sweep_dir / "failures.jsonl"
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "job_key": job.job_key,
        "backend": job.backend,
        "model": job.model,
        "suite": job.suite,
        "task": job.task_id,
        "task_id": job.task_id,
        "task_name": job.task_name,
        "level": job.level,
        "seed": job.seed,
        "episodes": job.episodes,
        "exit_code": exit_code,
        "log": str(log_path),
        "error": error,
    }
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_failed_jobs(sweep_dir: Path) -> list[JobSpec]:
    path = Path(sweep_dir) / "failures.jsonl"
    if not path.is_file():
        return []
    by_key: dict[str, JobSpec] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        job = JobSpec(
            backend=str(row["backend"]),
            suite=str(row["suite"]),
            task_id=int(row.get("task_id", row.get("task"))),
            task_name=str(row.get("task_name", f"task_{row.get('task_id', row.get('task'))}")),
            level=int(row["level"]),
            model=str(row["model"]),
            seed=int(row["seed"]),
            episodes=int(row.get("episodes", 0)),
        )
        by_key[job.job_key] = job
    return list(by_key.values())


def detect_gpu_ids(explicit: Sequence[int] | None) -> tuple[int, ...]:
    """Runtime GPU discovery (not a YAML hyperparameter default)."""
    if explicit is not None:
        return tuple(int(x) for x in explicit)
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip() != "":
        parts = [p.strip() for p in env.split(",") if p.strip() != ""]
        if parts:
            return tuple(int(p) for p in parts)
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        out = None
    if out is not None and out.returncode == 0 and out.stdout.strip():
        ids = [int(line.strip()) for line in out.stdout.splitlines() if line.strip()]
        if ids:
            return tuple(ids)
    return ()


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerConfig:
    gpu_ids: tuple[int, ...]
    jobs_per_gpu_cap: int | None
    resume: bool
    retry_failed: bool
    dry_run: bool
    results_root: Path
    sweep_dir: Path
    configs_root: Path
    health_timeout_s: float
    only: OnlyFilter


class SweepScheduler:
    def __init__(self, cfg: SchedulerConfig, jobs: Sequence[JobSpec]) -> None:
        self.cfg = cfg
        self._all_jobs = list(jobs)
        self._servers: list[ServerHandle] = []

    def plan(self) -> list[JobSpec]:
        jobs = [j for j in self._all_jobs if self.cfg.only.match(j)]
        if self.cfg.retry_failed:
            failed = load_failed_jobs(self.cfg.sweep_dir)
            if failed:
                expanded = {j.job_key: j for j in self._all_jobs}
                jobs = [expanded.get(fj.job_key, fj) for fj in failed]
                jobs = [j for j in jobs if self.cfg.only.match(j)]
        if self.cfg.resume:
            jobs = [j for j in jobs if not is_job_complete(self.cfg.results_root, j)]
        return jobs

    def _suite_serve_warnings(
        self,
        model: str,
        jobs: Sequence[JobSpec],
    ) -> list[str]:
        out: list[str] = []
        suites = sorted({j.suite for j in jobs if j.model == model})
        for suite in suites:
            suite_spec = load_model_serve_spec(
                model, configs_root=self.cfg.configs_root, suite=suite
            )
            if suite_spec.checkpoint_error:
                out.append(f"model {model} suite {suite}: {suite_spec.checkpoint_error}")
            if suite_spec.missing_fields:
                out.append(
                    f"model {model} suite {suite}: missing "
                    f"{', '.join(suite_spec.missing_fields)}"
                )
        return out

    def run(self) -> int:
        jobs = self.plan()
        totals = estimate_totals(jobs)
        warnings: list[str] = []
        specs: dict[str, ModelServeSpec] = {}
        for model in sorted({j.model for j in self._all_jobs}):
            spec = load_model_serve_spec(model, configs_root=self.cfg.configs_root)
            specs[model] = spec
            base_missing = spec.missing_fields
            if spec.needs_server:
                # Each suite gets its own server generation, so validate every
                # (model, suite) pair up front instead of failing hours in. That
                # supersedes the suite-less argv check, which cannot resolve
                # per-suite placeholders and would only ever be a false alarm.
                base_missing = tuple(
                    f for f in base_missing if not f.startswith("server")
                )
                warnings.extend(self._suite_serve_warnings(model, jobs))
            if base_missing:
                warnings.append(
                    f"model {model}: missing {', '.join(base_missing)}"
                )

        if self.cfg.dry_run:
            self._print_dry_run(jobs, totals, warnings, specs)
            return 0

        if warnings:
            for w in warnings:
                print(f"[sweep] WARN: {w}", file=sys.stderr)
        if not self.cfg.gpu_ids:
            print("[sweep] ERROR: no GPUs available; pass --gpus", file=sys.stderr)
            return 2
        for model, spec in specs.items():
            if any(j.model == model for j in jobs) and spec.max_jobs_per_gpu is None:
                print(
                    f"[sweep] ERROR: model {model} missing max_jobs_per_gpu in YAML",
                    file=sys.stderr,
                )
                return 2

        self.cfg.sweep_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.results_root.mkdir(parents=True, exist_ok=True)
        failures = 0
        try:
            # Schedule per model so server lifecycle is scoped.
            models_order: list[str] = []
            for j in jobs:
                if j.model not in models_order:
                    models_order.append(j.model)
            for model in models_order:
                model_jobs = [j for j in jobs if j.model == model]
                if not model_jobs:
                    continue
                # Concurrency is a property of (model, backend), not of the
                # model alone: a served model is a thin client on LIBERO but
                # may hold GPU state on a SAPIEN backend. Resolve the spec per
                # backend so per_backend.<backend>.resources applies.
                backends_order: list[str] = []
                for j in model_jobs:
                    if j.backend not in backends_order:
                        backends_order.append(j.backend)
                for backend in backends_order:
                    backend_jobs = [j for j in model_jobs if j.backend == backend]
                    spec = load_model_serve_spec(
                        model, configs_root=self.cfg.configs_root, backend=backend
                    )
                    mj = effective_max_jobs_per_gpu(
                        model,
                        {"max_jobs_per_gpu": spec.max_jobs_per_gpu},
                        self.cfg.jobs_per_gpu_cap,
                    )
                    failures += self._run_model_jobs(
                        backend_jobs,
                        spec=spec,
                        max_jobs_per_gpu=mj,
                        backend=backend,
                    )
        finally:
            self._cleanup_servers()

        return 1 if failures else 0

    def _print_dry_run(
        self,
        jobs: Sequence[JobSpec],
        totals: Mapping[str, Any],
        warnings: Sequence[str],
        specs: Mapping[str, ModelServeSpec],
    ) -> None:
        print(f"[sweep] suite_jobs_planned = {totals['n_jobs']}")
        print(f"[sweep] n_episodes_total   = {totals['n_episodes_total']}")
        print(f"[sweep] per_backend_jobs   = {totals['per_backend_jobs']}")
        print(f"[sweep] gpus               = {list(self.cfg.gpu_ids) or '(none detected)'}")
        print(f"[sweep] jobs_per_gpu_cap   = {self.cfg.jobs_per_gpu_cap}")
        print(f"[sweep] resume             = {self.cfg.resume}")
        print(f"[sweep] retry_failed       = {self.cfg.retry_failed}")
        print(f"[sweep] results_root       = {self.cfg.results_root}")
        print(f"[sweep] sweep_dir          = {self.cfg.sweep_dir}")
        for w in warnings:
            print(f"[sweep] WARN: {w}")
        # Concurrency is resolved per (model, backend), so print it that way —
        # a shared model can legitimately differ between LIBERO and RoboTwin.
        print(f"[sweep] eval_threads       = {eval_thread_count()}")
        for model, backend in sorted({(j.model, j.backend) for j in jobs}):
            spec = load_model_serve_spec(
                model, configs_root=self.cfg.configs_root, backend=backend
            )
            label = f"model {model} on {backend}"
            if spec.max_jobs_per_gpu is None:
                print(f"[sweep] {label}: max_jobs_per_gpu=<missing>")
                continue
            try:
                eff = effective_max_jobs_per_gpu(
                    model,
                    {"max_jobs_per_gpu": spec.max_jobs_per_gpu},
                    self.cfg.jobs_per_gpu_cap,
                )
            except ValueError as exc:
                print(f"[sweep] {label}: {exc}")
            else:
                slots = len(self.cfg.gpu_ids) * eff if self.cfg.gpu_ids else 0
                n_serve = servers_per_gpu_count(eff) if spec.needs_server else 0
                print(
                    f"[sweep] {label}: max_jobs_per_gpu={eff} "
                    f"needs_server={spec.needs_server} worker_slots={slots} "
                    f"servers_per_gpu={n_serve}"
                )
        max_print = 40
        for i, job in enumerate(jobs):
            if i >= max_print:
                print(f"[sweep] ... ({len(jobs) - max_print} more jobs)")
                break
            print(
                f"[dry-run] {job.backend} | {job.model} | {job.suite} | "
                f"task={job.task_id} | level={job.level} | seed={job.seed} | "
                f"episodes={job.episodes}"
            )
        print("[dry-run] no processes will be launched.")

    def _run_model_jobs(
        self,
        jobs: Sequence[JobSpec],
        *,
        spec: ModelServeSpec,
        max_jobs_per_gpu: int,
        backend: str | None = None,
    ) -> int:
        if not spec.needs_server:
            return self._run_job_group(jobs, spec=spec, max_jobs_per_gpu=max_jobs_per_gpu)

        # Server models load their weights at startup, so a per-suite checkpoint
        # (UniVLA) means one server generation per suite — sharing one server
        # across suites would silently serve the wrong fine-tune.
        suites: list[str] = []
        for job in jobs:
            if job.suite not in suites:
                suites.append(job.suite)
        failures = 0
        for suite in suites:
            suite_jobs = [j for j in jobs if j.suite == suite]
            suite_spec = load_model_serve_spec(
                spec.model_id,
                configs_root=self.cfg.configs_root,
                suite=suite,
                backend=backend,
            )
            if suite_spec.missing_fields or suite_spec.checkpoint_error:
                reason = suite_spec.checkpoint_error or (
                    "missing " + ", ".join(suite_spec.missing_fields)
                )
                print(
                    f"[sweep] ERROR: model {spec.model_id} suite {suite}: {reason}",
                    file=sys.stderr,
                )
                for job in suite_jobs:
                    append_failure(
                        self.cfg.sweep_dir,
                        job,
                        exit_code=2,
                        log_path=self.cfg.sweep_dir,
                        error=f"serve config: {reason}",
                    )
                failures += len(suite_jobs)
                continue
            failures += self._run_job_group(
                suite_jobs, spec=suite_spec, max_jobs_per_gpu=max_jobs_per_gpu
            )
        return failures

    def _run_job_group(
        self,
        jobs: Sequence[JobSpec],
        *,
        spec: ModelServeSpec,
        max_jobs_per_gpu: int,
    ) -> int:
        gpu_ids = self.cfg.gpu_ids
        slot_gpus = gpu_slot_order(gpu_ids, max_jobs_per_gpu)
        if not slot_gpus:
            return len(jobs)
        used_gpus = set(gpus_needed_for_jobs(slot_gpus, len(jobs)))

        failures = 0
        started_servers: list[ServerHandle] = []
        try:
            servers_by_gpu: dict[int, list[ServerHandle]] = {}
            if spec.needs_server:
                n_serve = servers_per_gpu_count(max_jobs_per_gpu)
                tag = f"{spec.name}-{spec.suite}" if spec.suite else spec.name
                try:
                    for gpu in sorted(used_gpus):
                        handles: list[ServerHandle] = []
                        for k in range(n_serve):
                            port = allocate_port()
                            suffix = f"gpu{gpu}" if n_serve == 1 else f"gpu{gpu}_s{k}"
                            log_path = self.cfg.sweep_dir / "servers" / tag / f"{suffix}.log"
                            handle = start_model_server(
                                spec,
                                gpu=gpu,
                                port=port,
                                log_path=log_path,
                                health_timeout_s=self.cfg.health_timeout_s,
                            )
                            handles.append(handle)
                            started_servers.append(handle)
                            self._servers.append(handle)
                        servers_by_gpu[gpu] = handles
                except Exception as exc:
                    # Serve failure is this model/suite only. Do not abort the
                    # remaining sweep (next models still run).
                    print(
                        f"[sweep] ERROR: model {spec.name} suite {spec.suite or '-'} "
                        f"server start failed: {exc}",
                        file=sys.stderr,
                    )
                    for job in jobs:
                        append_failure(
                            self.cfg.sweep_dir,
                            job,
                            exit_code=1,
                            log_path=self.cfg.sweep_dir,
                            error=f"serve start: {exc}",
                        )
                    return failures + len(jobs)

            pending = list(jobs)
            n_slots = len(slot_gpus)
            if n_slots < 1:
                return failures + len(jobs)
            # One process per GPU slot. When a job exits, refill that slot
            # immediately so a finished card does not wait on the slowest
            # job in a synchronized batch.
            slots: list[tuple[JobSpec, subprocess.Popen[Any], Path] | None] = [
                None
            ] * n_slots

            def _spawn(slot_i: int, job: JobSpec) -> tuple[JobSpec, subprocess.Popen[Any], Path]:
                gpu = slot_gpus[slot_i]
                log_path = (
                    self.cfg.sweep_dir
                    / "jobs"
                    / f"{job.backend}_{job.model}_{job.suite}_t{job.task_id}_l{job.level}_s{job.seed}.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                pin_eval_threads(env)
                # Writer package lives at <workspace>/A_robosafe_capture_datasets
                # (not under src/). Eval children inherit this env.
                if os.environ.get("DYNI_CAPTURE", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    workspace = str(find_workspace_root())
                    existing = env.get("PYTHONPATH", "")
                    env["PYTHONPATH"] = (
                        workspace
                        if not existing
                        else f"{workspace}{os.pathsep}{existing}"
                    )
                # Re-resolve so per_backend.eval_env (and its libs) apply.
                be_spec = load_model_serve_spec(
                    job.model,
                    configs_root=self.cfg.configs_root,
                    suite=job.suite,
                    backend=job.backend,
                )
                eval_env_name = (
                    be_spec.eval_env
                    if be_spec.eval_env is not None
                    else spec.eval_env
                )
                _prefer_conda_env_libs(env, eval_env_name)
                server_port: int | None = None
                if gpu in servers_by_gpu:
                    handles = servers_by_gpu[gpu]
                    gpu_slot_k = sum(1 for g in slot_gpus[:slot_i] if g == gpu)
                    handle = handles[gpu_slot_k % len(handles)]
                    env["HOST"] = _LOOPBACK_HOST
                    env["PORT"] = str(handle.port)
                    # Also inject via --set so ModelConfig.extra.server_port
                    # wins over YAML server.default_port (remote_host_port).
                    # Env alone is not enough: adapters must not read PORT
                    # (H10), and clients otherwise hang on the wrong port.
                    server_port = int(handle.port)
                cmd = self._eval_command(job, spec, server_port=server_port)
                with log_path.open("w", encoding="utf-8") as log_fp:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_fp,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                return job, proc, log_path

            while pending or any(entry is not None for entry in slots):
                for slot_i in range(n_slots):
                    if slots[slot_i] is None and pending:
                        slots[slot_i] = _spawn(slot_i, pending.pop(0))
                if all(entry is None for entry in slots):
                    break
                progressed = False
                for slot_i, entry in enumerate(slots):
                    if entry is None:
                        continue
                    job, proc, log_path = entry
                    rc = proc.poll()
                    if rc is None:
                        continue
                    progressed = True
                    slots[slot_i] = None
                    if rc != 0:
                        failures += 1
                        append_failure(
                            self.cfg.sweep_dir,
                            job,
                            exit_code=int(rc),
                            log_path=log_path,
                            error=f"eval exited {rc}",
                        )
                        print(
                            f"[fail] {job.job_key} rc={rc} log={log_path}",
                            file=sys.stderr,
                        )
                    else:
                        print(f"[done] {job.job_key}")
                if not progressed:
                    time.sleep(0.2)
        finally:
            for handle in started_servers:
                stop_server(handle)
                if handle in self._servers:
                    self._servers.remove(handle)
        return failures

    def _eval_command(
        self,
        job: JobSpec,
        spec: ModelServeSpec,
        *,
        server_port: int | None = None,
    ) -> list[str]:
        # Re-resolve with job.backend so per_backend.eval_env applies
        # (e.g. starvla: openvla-oft on libero, internvla_a1 on robotwin).
        be_spec = load_model_serve_spec(
            job.model,
            configs_root=self.cfg.configs_root,
            suite=job.suite,
            backend=job.backend,
        )
        eval_env = be_spec.eval_env if be_spec.eval_env is not None else spec.eval_env
        # When wrapping with ``conda run -n <env>``, use bare ``python`` so the
        # env interpreter is selected. Passing ``sys.executable`` (often base)
        # makes conda run that absolute binary and bypasses the target env's
        # site-packages — which broke smoke live (missing json_numpy, etc.).
        python_exe = "python" if eval_env else sys.executable
        argv = [
            python_exe,
            "-m",
            "dynisafety.cli",
            "eval",
            "--backend",
            job.backend,
            "--model",
            job.model,
            "--suite",
            job.suite,
            "--level",
            str(job.level),
            "--task",
            str(job.task_id),
            "--seed",
            str(job.seed),
            "--episodes",
            str(job.episodes),
            "--output-root",
            str(self.cfg.results_root),
        ]
        if server_port is not None:
            if isinstance(server_port, bool) or not isinstance(server_port, int):
                raise ValueError(f"server_port must be int, got {server_port!r}")
            if server_port < 1 or server_port > 65535:
                raise ValueError(f"server_port out of range: {server_port}")
            argv.extend(
                [
                    "--set",
                    f"extra.server_host={_LOOPBACK_HOST}",
                    "--set",
                    f"extra.server_port={server_port}",
                ]
            )
        if eval_env:
            argv = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                eval_env,
                *argv,
            ]
        return argv

    def _cleanup_servers(self) -> None:
        for handle in list(self._servers):
            stop_server(handle)
        self._servers.clear()


def default_results_root(workspace: Path | None = None) -> Path:
    root = workspace if workspace is not None else find_workspace_root()
    return root / "results"


def resolve_suite_path(suite: str, *, configs_root: Path) -> Path:
    """``--suite NAME`` → configs/suites/NAME.yaml (or absolute/relative path)."""
    candidate = Path(suite)
    if candidate.is_file():
        return candidate.resolve()
    path = Path(configs_root) / "suites" / f"{suite}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"sweep suite not found: {path}")
    return path.resolve()
