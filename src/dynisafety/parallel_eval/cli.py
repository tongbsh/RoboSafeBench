"""``dynisafety parallel-eval`` — opt-in RoboTwin + X-VLA process split.

Flags match ``register_eval_parser``. Config still goes through
``load_config`` so ``extra.parallel_eval.*`` participates in ``config_hash``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from dynisafety.config.loader import (
    ConfigValidationError,
    LoadRequest,
    load_config,
    task_yaml_stem,
)
from dynisafety.config.schema import KNOWN_BACKENDS, MISSING
from dynisafety.parallel_eval.config import load_parallel_eval_config

_SUPPORTED_BACKENDS = frozenset(KNOWN_BACKENDS)


def register_parallel_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    """Same frozen flags as ``eval``, plus seed-warm controls."""
    p = subparsers.add_parser(
        "parallel-eval",
        help="Opt-in parallel RoboTwin + X-VLA eval (Infer/Env split)",
        description=(
            "Opt-in Infer/Env split. --seed overrides runtime.seed (expert-check "
            "cursor / EpisodeRecord.seed). extra.parallel_eval.rng_seed (override "
            "with --rng-seed) is the per-slot x1 noise key and is independent of "
            "--seed. Omitting --rng-seed leaves the YAML value; there is no "
            "Python default (R4)."
        ),
    )
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
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override runtime.seed (expert-check cursor / EpisodeRecord.seed). "
            "Does not change extra.parallel_eval.rng_seed (per-slot x1); "
            "use --rng-seed for that."
        ),
    )
    p.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        dest="rng_seed",
        help=(
            "Override extra.parallel_eval.rng_seed (per-slot x1 noise). "
            "Independent of --seed. When omitted the YAML value is used "
            "(no Python default)."
        ),
    )
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
    p.add_argument(
        "--seed-warm",
        type=Path,
        default=None,
        help="Accepted-seed JSON from seed_warm (derived under results/ if omitted)",
    )
    p.add_argument(
        "--warm-only",
        action="store_true",
        help="Write the seed-warm JSON and exit (run in the Env conda)",
    )
    p.set_defaults(parallel_eval_handler=dispatch_parallel_eval)


def build_parallel_eval_overrides(args: argparse.Namespace) -> tuple[str, ...]:
    """Map CLI flags → LoadRequest.overrides. No embedded numeric defaults."""
    items: list[str] = []
    if args.overrides:
        items.extend(args.overrides)
    if args.seed is not None:
        items.append(f"runtime.seed={args.seed}")
    if getattr(args, "rng_seed", None) is not None:
        items.append(f"extra.parallel_eval.rng_seed={args.rng_seed}")
    if args.episodes is not None:
        items.append(f"runtime.episodes={args.episodes}")
    if args.no_video:
        items.append("output.save_video=false")
    if args.output_root is not None:
        items.append(f"output.results_dir={args.output_root}")
    return tuple(items)


def dispatch_parallel_eval(args: argparse.Namespace) -> int:
    return run_parallel_eval(
        backend=str(args.backend),
        model=str(args.model),
        suite=str(args.suite),
        level=int(args.level),
        task=args.task,
        config_files=tuple(args.config or ()),
        overrides=build_parallel_eval_overrides(args),
        dry_run=bool(args.dry_run),
        seed_warm=args.seed_warm,
        warm_only=bool(args.warm_only),
    )


def run_parallel_eval(
    *,
    backend: str,
    model: str,
    suite: str,
    level: int,
    task: int | None,
    config_files: tuple[Path, ...],
    overrides: tuple[str, ...],
    dry_run: bool,
    seed_warm: Path | None,
    warm_only: bool,
) -> int:
    if backend not in _SUPPORTED_BACKENDS:
        print(f"unknown backend {backend!r}", file=sys.stderr)
        return 2
    req = LoadRequest(
        backend=backend,
        level=level,
        suite=suite,
        task=task_yaml_stem(suite, task) if task is not None else None,
        model=model,
        config_files=config_files,
        overrides=overrides,
    )
    try:
        resolved = load_config(req)
    except ConfigValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        pe_cfg = load_parallel_eval_config(resolved)
    except (ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not pe_cfg.enabled and not warm_only:
        print(
            "extra.parallel_eval.enabled is false. parallel-eval is opt-in.\n"
            "Pass --set extra.parallel_eval.enabled=true "
            "(model stem xvla_official_parallel).",
            file=sys.stderr,
        )
        return 2
    if pe_cfg.pipeline_stage_num > 1 and not warm_only:
        print(
            "extra.parallel_eval.pipeline_stage_num > 1 is not implemented (P2)",
            file=sys.stderr,
        )
        return 2

    if dry_run:
        print(
            f"backend={backend} model={model} suite={suite} level={level} "
            f"task={task if task is not None else '(all)'} "
            f"enabled={pe_cfg.enabled} num_envs={pe_cfg.num_envs} "
            f"pipeline_stage_num={pe_cfg.pipeline_stage_num} "
            f"rng_seed={pe_cfg.rng_seed}"
        )
        return 0

    cfg = resolved.run
    if cfg.seed is MISSING or cfg.runtime is MISSING or cfg.runtime.episodes is MISSING:
        print(
            "runtime.seed and runtime.episodes must be set in YAML or via "
            "--seed / --episodes / --set.",
            file=sys.stderr,
        )
        return 2
    if cfg.output is MISSING or cfg.output.results_dir is MISSING:
        print("output.results_dir missing from resolved config.", file=sys.stderr)
        return 2

    from dynisafety.cli.eval import (
        _resolve_backend,
        _select_tasks,
        ensure_registries_loaded,
    )
    from dynisafety.config.resolve import find_workspace_root
    from dynisafety.core.episode import SCHEMA_VERSION
    from dynisafety.core.writer import RunWriter, build_manifest
    from dynisafety.parallel_eval.driver import (
        drive_from_warm_file,
        parallel_eval_block,
    )
    from dynisafety.parallel_eval.seed_warm import (
        build_warm_document,
        default_warm_path,
        warm_with_handle,
        write_warm_document,
    )
    from dynisafety.policies.base import model_config_from_resolved
    from dynisafety.runner.loop import (
        build_run_summary,
        make_run_dir,
        make_run_id,
    )

    try:
        ensure_registries_loaded()
        backend_obj = _resolve_backend(backend)
        tasks = _select_tasks(backend_obj, suite, task)
    except (RuntimeError, ValueError, TypeError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    workspace = find_workspace_root()
    n_episodes = int(cfg.runtime.episodes)
    seed = int(cfg.seed)
    print(
        f"parallel-eval runtime.seed={seed} "
        f"(--seed / expert-check cursor) "
        f"extra.parallel_eval.rng_seed={pe_cfg.rng_seed} "
        f"(per-slot x1; --seed does not change this)"
    )
    results_root = Path(str(cfg.output.results_dir))
    data = dict(resolved.data)
    if "name" not in data:
        data["name"] = str(data.get("model", model))
    model_cfg = model_config_from_resolved(data, backend=backend)

    extra_robotwin = _require_mapping(resolved.data.get("extra"), "extra").get("robotwin")
    extra_robotwin = _require_mapping(extra_robotwin, "extra.robotwin")
    seed_stride = int(cfg.runtime.seed_stride)

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
        if seed_warm is not None and len(tasks) == 1:
            warm_path = Path(seed_warm)
        else:
            warm_path = default_warm_path(
                results_dir=results_root,
                task=task_spec.task_name,
                suite=suite,
                level=int(task_cfg.level),
                run_seed=seed,
                n_episodes=n_episodes,
            )

        if warm_only or not warm_path.is_file():
            try:
                _write_warm_for_task(
                    backend_obj=backend_obj,
                    task_resolved=task_resolved,
                    model_cfg=model_cfg,
                    task_spec=task_spec,
                    workspace=workspace,
                    n_episodes=n_episodes,
                    run_seed=seed,
                    seed_stride=seed_stride,
                    extra_robotwin=extra_robotwin,
                    warm_path=warm_path,
                    warm_with_handle=warm_with_handle,
                    build_warm_document=build_warm_document,
                    write_warm_document=write_warm_document,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"seed-warm failed for task_id={task_spec.task_id} "
                    f"({task_spec.task_name}): {exc}",
                    file=sys.stderr,
                )
                exit_code = 1
                continue
            print(f"seed-warm wrote {warm_path}")
            if warm_only:
                continue

        if warm_only:
            continue

        run_id = make_run_id(
            backend=str(task_cfg.backend),
            model=str(task_cfg.model),
            suite=str(task_cfg.suite),
            level=int(task_cfg.level),
            seed=seed,
            task_id=int(task_spec.task_id),
        )
        run_cfg = replace(task_cfg, run_id=run_id)
        run_dir = make_run_dir(
            results_root,
            backend=str(task_cfg.backend),
            suite=str(task_cfg.suite),
            level=int(task_cfg.level),
            run_id=run_id,
        )
        log_dir = run_dir / "parallel_eval_logs"
        infer_payload = {
            "parallel_eval": parallel_eval_block(pe_cfg),
            "model_path": str(model_cfg.checkpoint),
            "repo_root": _require_server_root(model_cfg),
            "processor_path": None,
        }
        env_payload_base = {
            "load_request": {
                "backend": backend,
                "level": int(level),
                "suite": suite,
                "task": stem,
                "model": model,
                "config_files": [str(path) for path in config_files],
                "overrides": list(overrides),
            },
            "task_id": int(task_spec.task_id),
            "observation_needs_state": bool(model_cfg.observation_needs_state),
            "warmup_steps": _optional_int(model_cfg.extra.get("warmup_steps")),
            "post_warmup_space": _optional_str(
                model_cfg.extra.get("post_warmup_action_space")
            ),
            "gripper_threshold": _require_number(model_cfg.extra, "gripper_threshold"),
            "domain_id": _require_int(model_cfg.extra, "domain_id"),
            "denoising_steps": _require_int(model_cfg.extra, "denoising_steps"),
        }
        try:
            records = drive_from_warm_file(
                pe_cfg=pe_cfg,
                warm_path=warm_path,
                workspace=workspace,
                log_dir=log_dir,
                infer_payload=infer_payload,
                env_payload_base=env_payload_base,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"parallel-eval failed for task_id={task_spec.task_id} "
                f"({task_spec.task_name}): {exc}",
                file=sys.stderr,
            )
            exit_code = 1
            continue

        successes = sum(1 for rec in records if rec.success)
        with RunWriter(run_dir) as writer:
            writer.write_resolved_config(task_resolved.data)
            writer.write_manifest(
                build_manifest(
                    config_hash=task_resolved.config_hash,
                    schema_version=SCHEMA_VERSION,
                )
            )
            for record in records:
                writer.append_episode(record)
            writer.write_summary(
                build_run_summary(
                    expected_episodes=n_episodes,
                    records_meta={
                        "n_episodes_written": len(records),
                        "n_successes": successes,
                        "config_hash": task_resolved.config_hash,
                        "backend": run_cfg.backend,
                        "model": run_cfg.model,
                        "suite": run_cfg.suite,
                        "level": run_cfg.level,
                        "seed": seed,
                        "task_id": task_spec.task_id,
                        "task_name": task_spec.task_name,
                    },
                )
            )
        print(
            f"parallel-eval done task_id={task_spec.task_id} "
            f"episodes={len(records)} successes={successes} dir={run_dir}"
        )
    return exit_code


def _write_warm_for_task(
    *,
    backend_obj: Any,
    task_resolved: Any,
    model_cfg: Any,
    task_spec: Any,
    workspace: Path,
    n_episodes: int,
    run_seed: int,
    seed_stride: int,
    extra_robotwin: dict[str, Any],
    warm_path: Path,
    warm_with_handle: Any,
    build_warm_document: Any,
    write_warm_document: Any,
) -> None:
    from dynisafety.backends.robotwin.expert_cache import resolve_expert_check_cache_dir

    bind = getattr(backend_obj, "bind_run", None)
    if callable(bind):
        bind(
            config_hash=task_resolved.config_hash,
            resolved_data=task_resolved.data,
            action_space=model_cfg.action_space,
            chunk_size=model_cfg.chunk_size,
            observation_images=model_cfg.observation_images,
            observation_needs_state=model_cfg.observation_needs_state,
            model_extra=model_cfg.extra,
            workspace=workspace,
        )
    handle = backend_obj.make_env(task_resolved.run, task_spec)
    try:
        cache_dir = None
        if bool(_require_key(extra_robotwin, "expert_check")) and bool(
            _require_key(extra_robotwin, "expert_check_cache")
        ):
            cache_dir = resolve_expert_check_cache_dir(workspace)
        episodes = warm_with_handle(
            handle,
            n_episodes=n_episodes,
            run_seed=run_seed,
            seed_stride=seed_stride,
            enabled=bool(_require_key(extra_robotwin, "expert_check")),
            max_attempts=int(_require_key(extra_robotwin, "expert_check_max_attempts")),
            timeout_s=float(_require_key(extra_robotwin, "expert_check_timeout_s")),
            log_every=int(_require_key(extra_robotwin, "expert_check_log_every")),
            cache_dir=cache_dir,
        )
    finally:
        closer = getattr(handle, "close", None)
        if callable(closer):
            closer()
    document = build_warm_document(
        task=task_spec.task_name,
        suite=str(task_resolved.run.suite),
        level=int(task_resolved.run.level),
        run_seed=run_seed,
        seed_stride=seed_stride,
        episodes=episodes,
    )
    write_warm_document(document, warm_path)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_key(block: dict[str, Any], key: str) -> Any:
    if key not in block:
        raise ValueError(f"extra.robotwin.{key} is required")
    return block[key]


def _require_int(block: dict[str, Any], key: str) -> int:
    if key not in block:
        raise ValueError(f"model extra missing required key {key!r}")
    value = block[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"model extra {key!r} must be int, got {value!r}")
    return value


def _require_number(block: dict[str, Any], key: str) -> float:
    if key not in block:
        raise ValueError(f"model extra missing required key {key!r}")
    value = block[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"model extra {key!r} must be a number, got {value!r}")
    return float(value)


def _require_server_root(model_cfg: Any) -> str:
    root = getattr(model_cfg, "server_root", None)
    if isinstance(root, str) and root.strip():
        return root
    extra = getattr(model_cfg, "extra", {})
    if isinstance(extra, dict):
        fallback = extra.get("server_root")
        if isinstance(fallback, str) and fallback.strip():
            return fallback
    raise ValueError(
        "model server.root is required for Infer (X-VLA repo on sys.path)"
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int or None, got {value!r}")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"expected str or None, got {value!r}")
    return value
