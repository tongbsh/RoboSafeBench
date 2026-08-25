"""Spawn 1 Infer + N Env processes and collect ``EpisodeRecord``s.

Infer ``accept()`` is a synchronous barrier: every Env must connect or the
run hangs until ``channel_timeout_s``. Listen lines are scanned by prefix
because ``conda run`` / transformers dump ``libgomp`` noise first.

``finish`` already ``close_env``s exactly once — this module never calls
``handle.close()``. Named ``drive_parallel_eval`` (not ``run_episode(s)``).
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any, TextIO

from dynisafety.parallel_eval.config import REQUIRED_KEYS, ParallelEvalConfig
from dynisafety.parallel_eval.protocol import LISTEN_LINE_PREFIX
from dynisafety.parallel_eval.seed_warm import jobs_from_warm_document, load_warm_document

# OS reap grace, not a benchmark hyperparameter (those live in YAML).
_CHILD_REAP_GRACE_S = 2.0

INFER_SUBPROCESS_SOURCE = """
import json
import sys
from pathlib import Path

from dynisafety.parallel_eval.config import ParallelEvalConfig
from dynisafety.parallel_eval.infer_worker import serve_inference

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config = ParallelEvalConfig.from_extra({"parallel_eval": payload["parallel_eval"]})
serve_inference(
    config=config,
    model_path=payload["model_path"],
    repo_root=payload["repo_root"],
    processor_path=payload.get("processor_path"),
    runtime=None,
    listener=None,
    on_bound=None,
)
"""

ENV_SUBPROCESS_SOURCE = """
import sys

from dynisafety.parallel_eval.driver import env_slot_subprocess_main

env_slot_subprocess_main(sys.argv[1])
"""

PopenFactory = Callable[..., Any]


class ParallelEvalError(RuntimeError):
    """Driver-level failure with log tails already in the message."""


class ChildProcessGroup:
    """Terminate/kill every child on any exit path, including KeyboardInterrupt."""

    def __init__(self) -> None:
        self.processes: list[Any] = []

    def add(self, proc: Any) -> Any:
        self.processes.append(proc)
        return proc

    def terminate_all(self) -> None:
        terminate_child_processes(self.processes)

    def __enter__(self) -> ChildProcessGroup:
        return self

    def __exit__(self, *exc: object) -> None:
        self.terminate_all()


def parse_infer_listen_endpoint(
    lines: Iterable[str],
    *,
    prefix: str,
) -> tuple[str, int]:
    """First line whose stripped form starts with ``prefix``. Not assumed first."""
    if not isinstance(prefix, str) or not prefix:
        raise ValueError(f"prefix must be a non-empty str, got {prefix!r}")
    for raw in lines:
        line = raw.strip()
        if not line.startswith(prefix):
            continue
        rest = line[len(prefix) :].strip()
        if not rest or ":" not in rest:
            raise ValueError(f"listen line missing host:port after {prefix!r}: {line!r}")
        host, port_s = rest.rsplit(":", 1)
        if not host.strip():
            raise ValueError(f"listen line missing host: {line!r}")
        try:
            port = int(port_s)
        except ValueError as exc:
            raise ValueError(f"listen line port is not an int: {line!r}") from exc
        if port < 1 or port > 65535:
            raise ValueError(f"listen line port out of range: {port}")
        return host.strip(), port
    raise ValueError(f"no line starting with {prefix!r}")


def conda_run_python_argv(*, conda_env: str, source: str, payload_path: Path) -> list[str]:
    """Spawn via ``conda run --no-capture-output`` (R6: no shell activate)."""
    if not isinstance(conda_env, str) or not conda_env.strip():
        raise ValueError(f"conda_env must be a non-empty str, got {conda_env!r}")
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env.strip(),
        "python",
        "-c",
        source,
        str(payload_path),
    ]


def terminate_child_processes(procs: Sequence[Any]) -> None:
    """SIGTERM the process group, then SIGKILL leftovers. Safe on fake popen objects."""
    for proc in procs:
        if not _proc_running(proc):
            continue
        _signal_proc(proc, kill=False)
    deadline = time.monotonic() + _CHILD_REAP_GRACE_S
    for proc in procs:
        if not _proc_running(proc):
            continue
        remaining = deadline - time.monotonic()
        waiter = getattr(proc, "wait", None)
        if callable(waiter) and remaining > 0:
            try:
                waiter(timeout=remaining)
            except Exception:
                pass
        if _proc_running(proc):
            _signal_proc(proc, kill=True)


def resolve_env_slot_count(*, num_envs: int, n_jobs: int) -> int:
    """How many Env processes Infer must ``accept()``. Never larger than job count.

    An empty slot connects then closes the channel; Infer's accept barrier
    and the main loop then have an undefined peer. Clamp here, and keep
    Infer ``num_envs`` in lockstep with the number of processes we spawn.
    """
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError(f"num_envs must be int >= 1, got {num_envs!r}")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(
            f"cannot start Env workers: num_envs={num_envs} but job count is {n_jobs}. "
            "seed-warm produced no episodes; add jobs or do not run parallel-eval."
        )
    return min(num_envs, n_jobs)


def assign_jobs_to_slots(jobs: Sequence[Any], num_envs: int) -> list[list[Any]]:
    """Round-robin jobs onto ``num_envs`` slots. Empty slots are rejected."""
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError(f"num_envs must be int >= 1, got {num_envs!r}")
    n_jobs = len(jobs)
    if n_jobs < 1:
        raise ValueError(
            f"cannot assign 0 jobs to {num_envs} env slots; "
            "seed-warm produced no episodes"
        )
    if n_jobs < num_envs:
        raise ValueError(
            f"num_envs={num_envs} exceeds job count {n_jobs}; "
            f"that would start {num_envs - n_jobs} empty Env slot(s) whose "
            "closed channels leave Infer accept() undefined. "
            f"Lower extra.parallel_eval.num_envs to <= {n_jobs}, or add jobs."
        )
    slots: list[list[Any]] = [[] for _ in range(num_envs)]
    for index, job in enumerate(jobs):
        slots[index % num_envs].append(job)
    return slots


def parallel_eval_block(config: ParallelEvalConfig) -> dict[str, Any]:
    raw = asdict(config)
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"ParallelEvalConfig missing {missing[0]!r}")
    return {key: raw[key] for key in REQUIRED_KEYS}


def env_slot_subprocess_main(payload_path: str) -> None:
    """Env-conda entry. Reconstructs one handle and drives assigned jobs."""
    from dynisafety.backends.robotwin.backend import RoboTwinBackend
    from dynisafety.config.loader import LoadRequest, load_config
    from dynisafety.core.rollout import BoundRun
    from dynisafety.parallel_eval.env_worker import SlotEpisodeJob, serve_env_slot
    from dynisafety.policies.base import model_config_from_resolved

    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    req_raw = payload["load_request"]
    extras = tuple(Path(item) for item in req_raw.get("config_files", ()))
    request = LoadRequest(
        backend=str(req_raw["backend"]),
        level=int(req_raw["level"]),
        suite=req_raw.get("suite"),
        task=req_raw.get("task"),
        model=req_raw.get("model"),
        config_files=extras,
        overrides=tuple(req_raw.get("overrides", ())),
    )
    resolved = load_config(request)
    data = dict(resolved.data)
    if "name" not in data:
        data["name"] = str(data.get("model", request.model))
    model_cfg = model_config_from_resolved(data, backend=str(request.backend))
    backend = RoboTwinBackend()
    workspace = Path(payload["workspace"])
    backend.bind_run(
        config_hash=resolved.config_hash,
        resolved_data=resolved.data,
        action_space=model_cfg.action_space,
        chunk_size=int(model_cfg.chunk_size),
        observation_images=model_cfg.observation_images,
        observation_needs_state=model_cfg.observation_needs_state,
        model_extra=model_cfg.extra,
        workspace=workspace,
    )
    task_id = int(payload["task_id"])
    matched = [spec for spec in backend.list_tasks(str(request.suite)) if spec.task_id == task_id]
    if not matched:
        raise RuntimeError(f"task_id={task_id} not in suite {request.suite!r}")
    handle = backend.make_env(resolved.run, matched[0])
    jobs = [
        SlotEpisodeJob(
            episode_uid=str(item["episode_uid"]),
            episode_index=int(item["episode_index"]),
            start_seed=int(item["start_seed"]),
        )
        for item in payload["jobs"]
    ]
    try:
        records = serve_env_slot(
            slot_id=int(payload["slot_id"]),
            host=str(payload["host"]),
            port=int(payload["port"]),
            timeout_s=float(payload["timeout_s"]),
            handle=handle,
            cfg=resolved.run,
            bound=BoundRun(
                config_hash=resolved.config_hash,
                action_space=model_cfg.action_space,
                chunk_size=int(model_cfg.chunk_size),
            ),
            resolved_data=resolved.data,
            workspace=workspace,
            observation_needs_state=bool(payload["observation_needs_state"]),
            warmup_steps=payload.get("warmup_steps"),
            post_warmup_space=payload.get("post_warmup_space"),
            gripper_threshold=float(payload["gripper_threshold"]),
            domain_id=int(payload["domain_id"]),
            denoising_steps=int(payload["denoising_steps"]),
            rng_seed=_require_payload_int(payload, "rng_seed"),
            jobs=jobs,
        )
        _write_records_jsonl(Path(payload["records_path"]), records)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # finish() already closed the env exactly once; do not handle.close().
        del handle


def drive_parallel_eval(
    *,
    pe_cfg: ParallelEvalConfig,
    jobs: Sequence[Any],
    workspace: Path,
    log_dir: Path,
    infer_payload: Mapping[str, Any],
    env_payload_base: Mapping[str, Any],
    popen: PopenFactory | None = None,
) -> list[Any]:
    """Start Infer, then N Env workers, then read per-slot record files."""
    if not isinstance(pe_cfg, ParallelEvalConfig):
        raise TypeError(f"pe_cfg must be ParallelEvalConfig, got {type(pe_cfg)!r}")
    if not pe_cfg.enabled:
        raise ParallelEvalError(
            "extra.parallel_eval.enabled is false; this path is opt-in. "
            "Pass --set extra.parallel_eval.enabled=true."
        )
    if pe_cfg.pipeline_stage_num > 1:
        raise NotImplementedError(
            "extra.parallel_eval.pipeline_stage_num > 1 is not implemented (P2)"
        )
    if pe_cfg.num_envs < 1:
        raise ValueError(f"extra.parallel_eval.num_envs must be >= 1, got {pe_cfg.num_envs}")
    if pe_cfg.torch_dtype != "float32":
        raise ValueError(
            f"extra.parallel_eval.torch_dtype must be 'float32', got {pe_cfg.torch_dtype!r}"
        )

    from dynisafety.core.episode import EpisodeRecord

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(workspace)
    src_root = workspace / "src"
    spawn = popen if popen is not None else subprocess.Popen
    n_slots = resolve_env_slot_count(num_envs=pe_cfg.num_envs, n_jobs=len(jobs))
    slot_jobs = assign_jobs_to_slots(jobs, n_slots)

    infer_payload_path = log_dir / "infer_payload.json"
    infer_out = _infer_payload_with_slot_count(infer_payload, n_slots=n_slots)
    _write_json_atomic(infer_payload_path, infer_out)
    infer_log_path = log_dir / "infer.log"
    infer_argv = conda_run_python_argv(
        conda_env=pe_cfg.infer_conda_env,
        source=INFER_SUBPROCESS_SOURCE,
        payload_path=infer_payload_path,
    )

    group = ChildProcessGroup()
    infer_log_fp = infer_log_path.open("w", encoding="utf-8")
    drain_thread: threading.Thread | None = None
    try:
        infer_env = _child_env(src_root=src_root)
        infer_proc = group.add(
            spawn(
                infer_argv,
                cwd=str(workspace),
                env=infer_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        )
        if infer_proc.stdout is None:
            raise ParallelEvalError("Infer Popen has no stdout pipe")
        host, port = wait_for_infer_listen(
            proc=infer_proc,
            stream=infer_proc.stdout,
            log_fp=infer_log_fp,
            prefix=LISTEN_LINE_PREFIX,
            timeout_s=float(pe_cfg.channel_timeout_s),
            log_path=infer_log_path,
        )
        drain_thread = threading.Thread(
            target=_drain_stream,
            args=(infer_proc.stdout, infer_log_fp),
            name="parallel-eval-infer-drain",
            daemon=True,
        )
        drain_thread.start()

        env_record_paths: list[Path] = []
        env_log_paths: list[Path] = []
        env_procs: list[Any] = []
        for slot_id, assigned in enumerate(slot_jobs):
            records_path = log_dir / f"env_slot{slot_id}_records.jsonl"
            payload_path = log_dir / f"env_slot{slot_id}_payload.json"
            log_path = log_dir / f"env_slot{slot_id}.log"
            env_record_paths.append(records_path)
            env_log_paths.append(log_path)
            slot_payload = dict(env_payload_base)
            slot_payload.update(
                {
                    "slot_id": slot_id,
                    "host": host,
                    "port": port,
                    "timeout_s": float(pe_cfg.channel_timeout_s),
                    "rng_seed": int(pe_cfg.rng_seed),
                    "jobs": [_job_to_dict(job) for job in assigned],
                    "records_path": str(records_path),
                    "workspace": str(workspace),
                }
            )
            _write_json_atomic(payload_path, slot_payload)
            env_argv = conda_run_python_argv(
                conda_env=pe_cfg.env_conda_env,
                source=ENV_SUBPROCESS_SOURCE,
                payload_path=payload_path,
            )
            env_env = _child_env(
                src_root=src_root,
                extra={"CUDA_VISIBLE_DEVICES": pe_cfg.env_cuda_visible_devices},
            )
            log_fp = log_path.open("w", encoding="utf-8")
            try:
                proc = group.add(
                    spawn(
                        env_argv,
                        cwd=str(workspace),
                        env=env_env,
                        stdout=log_fp,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                )
            finally:
                log_fp.close()
            env_procs.append(proc)
            if _proc_exited_quickly(proc):
                raise ParallelEvalError(
                    _env_startup_message(
                        slot_id=slot_id,
                        num_envs=n_slots,
                        timeout_s=pe_cfg.channel_timeout_s,
                        code=_proc_returncode(proc),
                        log_path=log_path,
                    )
                )

        failures: list[str] = []
        for slot_id, proc in enumerate(env_procs):
            code = _wait_proc(proc)
            if code not in (0, None):
                failures.append(
                    _env_startup_message(
                        slot_id=slot_id,
                        num_envs=n_slots,
                        timeout_s=pe_cfg.channel_timeout_s,
                        code=code,
                        log_path=env_log_paths[slot_id],
                    )
                )
        if failures:
            raise ParallelEvalError("\n".join(failures))

        records: list[Any] = []
        for path in env_record_paths:
            records.extend(_read_records_jsonl(path))
        records.sort(key=lambda rec: int(rec.episode_index))
        if records and not isinstance(records[0], EpisodeRecord):
            raise TypeError(f"expected EpisodeRecord, got {type(records[0])!r}")
        return records
    except KeyboardInterrupt:
        raise
    finally:
        group.terminate_all()
        infer_log_fp.close()
        if drain_thread is not None:
            drain_thread.join(timeout=_CHILD_REAP_GRACE_S)


def wait_for_infer_listen(
    *,
    proc: Any,
    stream: IO[str],
    log_fp: TextIO,
    prefix: str,
    timeout_s: float,
    log_path: Path,
) -> tuple[str, int]:
    """Read Infer stdout until the listen prefix or ``timeout_s`` / unexpected exit."""
    deadline = time.monotonic() + float(timeout_s)
    buffered: list[str] = []
    while time.monotonic() < deadline:
        if _proc_exited(proc):
            leftover = _safe_read(stream)
            if leftover:
                log_fp.write(leftover)
                log_fp.flush()
                buffered.append(leftover)
            raise ParallelEvalError(
                "Infer exited "
                f"{_proc_returncode(proc)} before printing {prefix!r}. "
                f"Log {log_path}:\n{_tail_text(''.join(buffered), log_path)}"
            )
        remaining = deadline - time.monotonic()
        ready: list[Any]
        try:
            ready, _, _ = select.select([stream], [], [], max(remaining, 0.0))
        except (ValueError, OSError):
            ready = []
        if not ready:
            continue
        line = stream.readline()
        if not line:
            continue
        log_fp.write(line)
        log_fp.flush()
        buffered.append(line)
        try:
            return parse_infer_listen_endpoint([line], prefix=prefix)
        except ValueError:
            continue
    raise ParallelEvalError(
        f"Timed out after {timeout_s}s waiting for {prefix!r} on Infer stdout. "
        "conda run / transformers often print libgomp or FutureWarning first — "
        "the listen line is not assumed to be line 1. "
        f"Infer accept() is a barrier: Env workers are not started until this line. "
        f"Log {log_path}:\n{_tail_text(''.join(buffered), log_path)}"
    )


def drive_from_warm_file(
    *,
    pe_cfg: ParallelEvalConfig,
    warm_path: Path,
    workspace: Path,
    log_dir: Path,
    infer_payload: Mapping[str, Any],
    env_payload_base: Mapping[str, Any],
    popen: PopenFactory | None = None,
) -> list[Any]:
    document = load_warm_document(warm_path)
    jobs = jobs_from_warm_document(document)
    return drive_parallel_eval(
        pe_cfg=pe_cfg,
        jobs=jobs,
        workspace=workspace,
        log_dir=log_dir,
        infer_payload=infer_payload,
        env_payload_base=env_payload_base,
        popen=popen,
    )


def _job_to_dict(job: Any) -> dict[str, Any]:
    return {
        "episode_uid": str(job.episode_uid),
        "episode_index": int(job.episode_index),
        "start_seed": int(job.start_seed),
    }


def _infer_payload_with_slot_count(
    infer_payload: Mapping[str, Any], *, n_slots: int
) -> dict[str, Any]:
    """Copy Infer payload and set ``parallel_eval.num_envs`` to the Env count.

    Infer ``accept()`` loops ``config.num_envs`` times. That number must
    equal the processes we actually spawn, not the YAML request.
    """
    if "parallel_eval" not in infer_payload:
        raise ValueError("infer_payload missing required key 'parallel_eval'")
    raw_pe = infer_payload["parallel_eval"]
    if not isinstance(raw_pe, Mapping):
        raise TypeError(
            f"infer_payload['parallel_eval'] must be a mapping, got {type(raw_pe)!r}"
        )
    pe_block = dict(raw_pe)
    pe_block["num_envs"] = n_slots
    out = dict(infer_payload)
    out["parallel_eval"] = pe_block
    return out


def _require_payload_int(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload:
        raise ValueError(f"env payload missing required key {key!r}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"env payload {key!r} must be int, got {value!r}")
    return value


def _child_env(*, src_root: Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    src = str(src_root)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    env["PYTHONUNBUFFERED"] = "1"
    _prefer_conda_env_libs(env)
    if extra:
        env.update(dict(extra))
    return env


def _prefer_conda_env_libs(env: dict[str, str]) -> None:
    """Prepend the target env's lib when ``DYNI_CONDA_ROOT`` / ``CONDA_PREFIX`` exist."""
    roots: list[Path] = []
    for key in ("DYNI_CONDA_ROOT", "CONDA_ROOT", "CONDA_PREFIX"):
        raw = env.get(key) or os.environ.get(key)
        if not raw:
            continue
        path = Path(raw)
        if (path / "envs").is_dir():
            roots.append(path)
        elif path.parent.name == "envs":
            roots.append(path.parent.parent)
    roots.extend([Path.home() / "miniconda3", Path.home() / "anaconda3"])
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        lib = resolved / "envs" / env.get("CONDA_DEFAULT_ENV", "") / "lib"
        if lib.is_dir():
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = str(lib) if not prev else f"{lib}{os.pathsep}{prev}"
            return


def _proc_running(proc: Any) -> bool:
    poll = getattr(proc, "poll", None)
    if callable(poll):
        try:
            return poll() is None
        except Exception:
            return False
    return False


def _proc_exited(proc: Any) -> bool:
    return not _proc_running(proc) and getattr(proc, "poll", None) is not None


def _proc_exited_quickly(proc: Any) -> bool:
    poll = getattr(proc, "poll", None)
    if not callable(poll):
        return False
    try:
        return poll() not in (None, 0)
    except Exception:
        return False


def _proc_returncode(proc: Any) -> object:
    return getattr(proc, "returncode", None)


def _wait_proc(proc: Any) -> object:
    waiter = getattr(proc, "wait", None)
    if callable(waiter):
        try:
            return waiter()
        except Exception as exc:
            return exc
    return _proc_returncode(proc)


def _signal_proc(proc: Any, *, kill: bool) -> None:
    pid = getattr(proc, "pid", None)
    if isinstance(pid, int) and pid > 0:
        sig = signal.SIGKILL if kill else signal.SIGTERM
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        except Exception:
            pass
    method_name = "kill" if kill else "terminate"
    method = getattr(proc, method_name, None)
    if callable(method):
        try:
            method()
        except Exception:
            pass


def _drain_stream(stream: IO[str], log_fp: TextIO) -> None:
    try:
        while True:
            line = stream.readline()
            if not line:
                break
            log_fp.write(line)
            log_fp.flush()
    except Exception:
        pass


def _safe_read(stream: IO[str]) -> str:
    try:
        return stream.read() or ""
    except Exception:
        return ""


def _env_startup_message(
    *,
    slot_id: int,
    num_envs: int,
    timeout_s: float,
    code: object,
    log_path: Path,
) -> str:
    tail = _tail_text("", log_path)
    return (
        f"Env slot {slot_id} exited {code}. "
        f"Infer accept() waits for all {num_envs} Env connections "
        f"(channel_timeout_s={timeout_s}); a dead Env looks like a hang. "
        f"Log {log_path}:\n{tail}"
    )


def _tail_text(buffered: str, log_path: Path, *, n_lines: int = 40) -> str:
    text = buffered
    if log_path.is_file():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace") or text
        except OSError:
            pass
    lines = text.splitlines()
    if not lines:
        return "(log empty)"
    return "\n".join(lines[-n_lines:])


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _write_records_jsonl(path: Path, records: Sequence[Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record.to_json_dict(), ensure_ascii=False, separators=(",", ":"))
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _read_records_jsonl(path: Path) -> list[Any]:
    from dynisafety.core.episode import EpisodeRecord

    dest = Path(path)
    if not dest.is_file():
        return []
    records = []
    for line in dest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(EpisodeRecord.from_json_dict(json.loads(line)))
    return records

