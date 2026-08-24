"""``dynisafety eval`` — single-run evaluation entry (frozen flags)."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from dynisafety.config.loader import (
    ConfigValidationError,
    LoadRequest,
    ResolvedConfig,
    load_config,
    task_yaml_stem,
)
from dynisafety.config.schema import KNOWN_BACKENDS, MISSING
from dynisafety.core.protocols import Backend, TaskSpec
from dynisafety.core.registry import Registry, RegistryLoadError
from dynisafety.policies.base import Policy
from dynisafety.runner.loop import make_run_dir, make_run_id, run_episodes

# Lazy-filled by plugin import (M5–M7 register into these).
BACKEND_REGISTRY: Registry[Any] = Registry("backend")
POLICY_REGISTRY: Registry[Any] = Registry("policy")

_PLUGINS_LOADED = False


def register_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register frozen ``eval`` flags (no business ``default=``)."""
    p = subparsers.add_parser("eval", help="Run a single evaluation job")
    p.add_argument(
        "--backend",
        required=True,
        choices=KNOWN_BACKENDS,
        help="Backend id",
    )
    p.add_argument("--model", required=True, help="Model registry / YAML stem")
    p.add_argument("--suite", required=True, help="Backend suite id (full name)")
    p.add_argument("--level", type=int, required=True, help="Disturbance level 0..3")
    p.add_argument(
        "--task",
        type=int,
        default=None,
        help="Task id (int); omit to run all tasks from Backend.list_tasks",
    )
    p.add_argument("--seed", type=int, default=None, help="Override runtime.seed")
    p.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override runtime.episodes",
    )
    p.add_argument(
        "--config",
        type=Path,
        action="append",
        default=None,
        help="Extra experiment YAML (repeatable; later files override)",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override dotted key (repeatable)",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Force output.save_video=false",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output.results_dir",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only; do not start simulator",
    )
    p.set_defaults(eval_handler=dispatch_eval)


def build_eval_overrides(args: argparse.Namespace) -> tuple[str, ...]:
    """Map CLI flags → LoadRequest.overrides (no hardcoded fallback values).

    When ``HOST`` / ``PORT`` are present in the environment (set by
    ``dynisafety sweep`` / ``scripts/run/20_serve.sh``), inject them as
    ``extra.server_host`` / ``extra.server_port`` unless the caller already
    passed those keys via ``--set``. This is the supported way for remote
    transports to learn the live listen port — adapters must not read
    ``PORT`` themselves (H10).

    ``DYNI_SAVE_VIDEO`` (``1``/``true``/``yes``/``on``) enables
    ``output.save_video`` for debug sweeps that cannot pass ``--set`` through
    the scheduler; ignored when ``--no-video`` or an explicit
    ``output.save_video=...`` override is already present.
    """
    items: list[str] = []
    if args.overrides:
        items.extend(args.overrides)
    if args.seed is not None:
        items.append(f"runtime.seed={args.seed}")
    if args.episodes is not None:
        items.append(f"runtime.episodes={args.episodes}")
    if args.no_video:
        items.append("output.save_video=false")
    if args.output_root is not None:
        items.append(f"output.results_dir={args.output_root}")

    override_keys = {
        raw.split("=", 1)[0].strip()
        for raw in items
        if isinstance(raw, str) and "=" in raw
    }
    env_host = os.environ.get("HOST")
    env_port = os.environ.get("PORT")
    if (
        env_host
        and env_host.strip()
        and "extra.server_host" not in override_keys
    ):
        items.append(f"extra.server_host={env_host.strip()}")
    if env_port and env_port.strip() and "extra.server_port" not in override_keys:
        try:
            port_i = int(env_port.strip())
        except ValueError as exc:
            raise ValueError(
                f"env PORT must be an int port, got {env_port!r}"
            ) from exc
        if port_i < 1 or port_i > 65535:
            raise ValueError(f"env PORT out of range: {port_i}")
        items.append(f"extra.server_port={port_i}")
    env_save_video = os.environ.get("DYNI_SAVE_VIDEO", "").strip().lower()
    if (
        env_save_video in {"1", "true", "yes", "on"}
        and "output.save_video" not in override_keys
        and not args.no_video
    ):
        items.append("output.save_video=true")
    return tuple(items)


def ensure_registries_loaded() -> None:
    """Import plugins and populate backend/policy registries (no warn-skip)."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    try:
        from dynisafety.backends.registry import populate_backends

        populate_backends(BACKEND_REGISTRY)
    except ImportError:
        pass
    try:
        from dynisafety.policies.registry import populate_policies

        populate_policies(POLICY_REGISTRY)
    except ImportError:
        pass
    # Keep optional import probes for future plugin packages.
    for mod_name in (
        "dynisafety.backends",
        "dynisafety.policies.registry",
    ):
        try:
            importlib.import_module(mod_name)
        except ImportError:
            continue
    _PLUGINS_LOADED = True


def _resolve_backend(name: str) -> Backend:
    ensure_registries_loaded()
    try:
        cls = BACKEND_REGISTRY.get(name)
    except KeyError as exc:
        from dynisafety.backends.registry import BACKEND_IMPORT_ERRORS

        detail = ""
        err = BACKEND_IMPORT_ERRORS.get(name)
        if err is not None:
            detail = f" Import failed: {type(err).__name__}: {err}."
        elif BACKEND_IMPORT_ERRORS:
            skipped = ", ".join(
                f"{k} ({type(v).__name__}: {v})"
                for k, v in sorted(BACKEND_IMPORT_ERRORS.items())
            )
            detail = f" Other backends skipped due to ImportError: {skipped}."
        raise RuntimeError(
            f"backend {name!r} is not registered. "
            f"Available: {BACKEND_REGISTRY.available() or '(none)'}.{detail} "
            "Implement and register backends in M5–M7."
        ) from exc
    return cls()  # type: ignore[call-arg]


def _resolve_policy(model: str, resolved: ResolvedConfig, *, backend: str) -> Policy:
    from dynisafety.policies.base import model_config_from_resolved
    from dynisafety.policies.registry import create_policy

    ensure_registries_loaded()
    try:
        POLICY_REGISTRY.get(model)
    except KeyError as exc:
        raise RuntimeError(
            f"policy/model {model!r} is not registered. "
            f"Available: {POLICY_REGISTRY.available() or '(none)'}. "
            "Implement and register policies in M4–M7."
        ) from exc
    # Loader injects identity under RunConfig keys; model YAML uses ``name``.
    data = dict(resolved.data)
    if "name" not in data:
        data["name"] = str(data.get("model", model))
    cfg = model_config_from_resolved(data, backend=backend)
    return create_policy(cfg, POLICY_REGISTRY)


def _select_tasks(
    backend: Backend,
    suite: str,
    task_id: int | None,
) -> list[TaskSpec]:
    tasks = backend.list_tasks(suite)
    if not tasks:
        raise RuntimeError(f"backend {backend.name!r} returned no tasks for suite {suite!r}")
    if task_id is None:
        return list(tasks)
    matched = [t for t in tasks if t.task_id == task_id]
    if not matched:
        ids = ", ".join(str(t.task_id) for t in tasks)
        raise RuntimeError(
            f"task_id={task_id} not in suite {suite!r}; available ids: {ids}"
        )
    return matched


def _yaml_safe(value: Any) -> Any:
    """Convert config MISSING sentinels so dry-run YAML dump cannot crash."""
    if value is MISSING:
        return None
    return value


def _print_dry_run(
    *,
    resolved: ResolvedConfig,
    task_id: int | None,
    overrides: tuple[str, ...],
    tasks: list[dict[str, Any]] | None = None,
    list_tasks_error: str | None = None,
) -> None:
    run = resolved.run
    episodes = (
        run.runtime.episodes
        if run.runtime is not MISSING and run.runtime is not None
        else MISSING
    )
    seed = run.seed if run.seed is not MISSING else (
        run.runtime.seed
        if run.runtime is not MISSING and run.runtime is not None
        else MISSING
    )
    plan: dict[str, Any] = {
        "backend": _yaml_safe(run.backend),
        "model": _yaml_safe(run.model),
        "suite": _yaml_safe(run.suite),
        "level": _yaml_safe(run.level),
        "seed": _yaml_safe(seed),
        "episodes": _yaml_safe(episodes),
        "max_steps": _yaml_safe(run.max_steps),
        "checkpoint": _yaml_safe(run.checkpoint),
        "results_dir": _yaml_safe(
            run.output.results_dir if run.output is not MISSING else MISSING
        ),
        "save_video": _yaml_safe(
            run.output.save_video if run.output is not MISSING else MISSING
        ),
        "config_hash": resolved.config_hash,
        "task_id": task_id if task_id is not None else "(all from Backend.list_tasks)",
        "run_id_pattern": "{backend}-{model}-{suite}-t{task}-l{level}-s{seed}-{YYYYmmddTHHMMSS}",
        "run_dir_pattern": "results/l{level}/{run_id}",
        "overrides": list(overrides),
        "warnings": list(resolved.warnings),
    }
    if tasks is not None:
        plan["tasks"] = tasks
        plan["n_tasks"] = len(tasks)
    if list_tasks_error:
        plan["list_tasks_error"] = list_tasks_error
    print(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), end="")
    print(
        "# dry-run: config + list_tasks only; make_env/rollout not started",
        file=sys.stderr,
    )


def run_eval(
    *,
    backend: str,
    model: str,
    suite: str,
    level: int,
    task: int | None,
    config_file: Path | None = None,
    config_files: tuple[Path, ...] = (),
    overrides: tuple[str, ...],
    dry_run: bool,
) -> int:
    """load_config → dry-run plan | registries → per-task run_episodes."""
    extras: list[Path] = []
    if config_file is not None:
        extras.append(config_file)
    extras.extend(config_files)
    req = LoadRequest(
        backend=backend,
        level=level,
        suite=suite,
        task=task_yaml_stem(suite, task) if task is not None else None,
        model=model,
        config_files=tuple(extras),
        overrides=overrides,
    )
    try:
        resolved = load_config(req)
    except ConfigValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if dry_run:
        # Config + list_tasks only. Never make_env / rollout (Window B: robocasa
        # make_env raises NotImplementedError until Window C).
        task_rows: list[dict[str, Any]] | None = None
        list_tasks_error: str | None = None
        try:
            ensure_registries_loaded()
            if backend in BACKEND_REGISTRY.available():
                backend_obj = _resolve_backend(backend)
                specs = _select_tasks(backend_obj, suite, task)
                task_rows = [
                    {
                        "task_id": int(spec.task_id),
                        "task_name": spec.task_name,
                        "instruction": spec.instruction,
                        "max_steps": int(spec.max_steps),
                    }
                    for spec in specs
                ]
        except (
            RuntimeError,
            RegistryLoadError,
            ValueError,
            TypeError,
            ImportError,
            FileNotFoundError,
            OSError,
        ) as exc:
            list_tasks_error = str(exc)
        _print_dry_run(
            resolved=resolved,
            task_id=task,
            overrides=overrides,
            tasks=task_rows,
            list_tasks_error=list_tasks_error,
        )
        return 0

    try:
        backend_obj = _resolve_backend(backend)
        from dynisafety.policies.base import model_config_from_resolved

        data = dict(resolved.data)
        if "name" not in data:
            data["name"] = str(data.get("model", model))
        model_cfg = model_config_from_resolved(data, backend=backend)
        policy = _resolve_policy(model, resolved, backend=backend)
        bind = getattr(backend_obj, "bind_run", None)
        if callable(bind):
            bind(
                config_hash=resolved.config_hash,
                resolved_data=resolved.data,
                action_space=model_cfg.action_space,
                chunk_size=model_cfg.chunk_size,
                observation_images=model_cfg.observation_images,
                observation_needs_state=model_cfg.observation_needs_state,
                model_extra=model_cfg.extra,
            )
        tasks = _select_tasks(backend_obj, suite, task)
    except (RuntimeError, RegistryLoadError, ValueError, TypeError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    cfg = resolved.run
    if cfg.seed is MISSING or cfg.runtime is MISSING or cfg.runtime.episodes is MISSING:
        print(
            "runtime.seed and runtime.episodes must be set in YAML or via "
            "--seed / --episodes / --set (iron law 1: no CLI defaults).",
            file=sys.stderr,
        )
        return 2
    if cfg.output is MISSING or cfg.output.results_dir is MISSING:
        print("output.results_dir missing from resolved config.", file=sys.stderr)
        return 2

    n_episodes = int(cfg.runtime.episodes)
    seed = int(cfg.seed)
    results_root = Path(str(cfg.output.results_dir))
    save_video = bool(cfg.output.save_video)

    exit_code = 0
    for task_spec in tasks:
        stem = task_yaml_stem(suite, int(task_spec.task_id))
        if req.task == stem:
            task_resolved = resolved
        else:
            try:
                task_resolved = load_config(replace(req, task=stem))
            except ConfigValidationError as exc:
                print(str(exc), file=sys.stderr)
                exit_code = 1
                continue
        task_cfg = task_resolved.run
        if task_cfg.seed is MISSING or task_cfg.runtime is MISSING:
            task_cfg = cfg
        run_id = make_run_id(
            backend=str(task_cfg.backend),
            model=str(task_cfg.model),
            suite=str(task_cfg.suite),
            level=int(task_cfg.level),
            seed=seed,
            task_id=int(task_spec.task_id),
        )
        run_cfg = replace(task_cfg, run_id=run_id)
        run_dir = make_run_dir(results_root, level=int(task_cfg.level), run_id=run_id)
        try:
            result = run_episodes(
                backend=backend_obj,
                policy=policy,
                cfg=run_cfg,
                resolved=task_resolved,
                task=task_spec,
                run_dir=run_dir,
                n_episodes=n_episodes,
                save_video=save_video,
                close_policy=False,
            )
        except Exception as exc:  # noqa: BLE001 — isolate per-task failure
            print(
                f"eval failed for task_id={task_spec.task_id} ({task_spec.task_name}): {exc}",
                file=sys.stderr,
            )
            exit_code = 1
            continue
        print(
            f"eval done task_id={task_spec.task_id} "
            f"episodes={result.episodes_written} "
            f"successes={result.successes} "
            f"dir={result.run_dir}"
        )
    try:
        policy.close()
    except Exception:  # noqa: BLE001 — eval must still return the task exit code
        pass
    return exit_code


def dispatch_eval(args: argparse.Namespace) -> int:
    return run_eval(
        backend=args.backend,
        model=args.model,
        suite=args.suite,
        level=args.level,
        task=args.task,
        config_files=tuple(args.config or ()),
        overrides=build_eval_overrides(args),
        dry_run=bool(args.dry_run),
    )
