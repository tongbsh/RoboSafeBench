"""Unique episode loop — backend-agnostic orchestration over Backend.rollout."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.config.loader import ResolvedConfig
from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.core.protocols import Backend, BackendCapabilities, EnvHandle, TaskSpec
from dynisafety.core.rollout import run_dir_layout
from dynisafety.core.writer import RunWriter, build_manifest
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
    Policy,
)


@dataclass(frozen=True)
class LoopResult:
    episodes_written: int
    successes: int
    run_dir: Path
    summary: dict[str, Any]


def make_run_id(
    *,
    backend: str,
    model: str,
    suite: str,
    level: int,
    seed: int,
    task_id: int,
    when: datetime | None = None,
) -> str:
    """``{backend}-{model}-{suite}-t{task}-l{level}-s{seed}-{YYYYmmddTHHMMSS}`` (UTC).

    ``task_id`` is required so concurrent sweep workers for different tasks never
    share one results directory (same-second stamp collision).
    """
    ts = when if when is not None else datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%dT%H%M%S")
    return f"{backend}-{model}-{suite}-t{int(task_id)}-l{level}-s{seed}-{stamp}"


def make_run_dir(results_root: Path | str, *, level: int, run_id: str) -> Path:
    """``results/l{level}/<run_id>/`` — level bucket under the results root.

    Delegates to ``core.rollout.run_dir_layout`` so backends writing diagnostics
    traces into the same directory cannot drift from the runner's layout.
    """
    return run_dir_layout(results_root, level=level, run_id=run_id)


def build_run_summary(
    *,
    expected_episodes: int,
    records_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Minimal summary keys for resume (S2 确认 D5)."""
    written = int(records_meta["n_episodes_written"])
    status = "completed" if written >= expected_episodes else "incomplete"
    summary: dict[str, Any] = {
        "status": status,
        "n_episodes_expected": expected_episodes,
        "n_episodes_written": written,
        "n_successes": int(records_meta.get("n_successes", 0)),
        "config_hash": records_meta["config_hash"],
        "backend": records_meta["backend"],
        "model": records_meta["model"],
        "suite": records_meta["suite"],
        "level": int(records_meta["level"]),
        "seed": int(records_meta["seed"]),
        "task_id": int(records_meta["task_id"]),
        "task_name": str(records_meta["task_name"]),
        "finished_at": records_meta.get(
            "finished_at",
            datetime.now(timezone.utc).isoformat(),
        ),
    }
    return summary


def run_episodes(
    *,
    backend: Backend,
    policy: Policy,
    cfg: RunConfig,
    resolved: ResolvedConfig,
    task: TaskSpec,
    run_dir: Path,
    n_episodes: int,
    save_video: bool,
    close_policy: bool = True,
) -> LoopResult:
    """Run ``n_episodes`` for a **single** task into ``run_dir``.

    One ``run_id`` / directory per task (S2 确认 D4). Does not branch on
    ``backend.name``. ``save_video`` is forwarded via cfg.output for backends;
    this loop does not encode video itself.
    """
    del save_video  # backends read cfg.output.save_video; kept for API stability
    # close_policy=False when the caller reuses the same Policy across tasks.
    if n_episodes < 0:
        raise ValueError(f"n_episodes must be >= 0, got {n_episodes}")
    if cfg.run_id is MISSING or not cfg.run_id:
        raise ValueError("cfg.run_id must be set before run_episodes")

    run_dir = Path(run_dir)
    successes = 0
    written = 0

    with RunWriter(run_dir) as writer:
        writer.write_resolved_config(resolved.data)
        writer.write_manifest(
            build_manifest(
                config_hash=resolved.config_hash,
                schema_version=SCHEMA_VERSION,
            )
        )
        for episode_index in range(n_episodes):
            env = backend.make_env(cfg, task)
            try:
                record = backend.rollout(env, policy, cfg, episode_index)
                writer.append_episode(record)
                written += 1
                if record.success:
                    successes += 1
            finally:
                env.close()

        summary = build_run_summary(
            expected_episodes=n_episodes,
            records_meta={
                "n_episodes_written": written,
                "n_successes": successes,
                "config_hash": resolved.config_hash,
                "backend": cfg.backend,
                "model": cfg.model,
                "suite": cfg.suite,
                "level": cfg.level,
                "seed": cfg.seed,
                "task_id": task.task_id,
                "task_name": task.task_name,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        writer.write_summary(summary)

    if close_policy:
        try:
            policy.close()
        except Exception:  # noqa: BLE001 — loop must not fail after successful write
            pass

    return LoopResult(
        episodes_written=written,
        successes=successes,
        run_dir=run_dir,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Dummy backend + random policy (verification only; not a production registry)
# ---------------------------------------------------------------------------


@dataclass
class _DummyEnv:
    task: TaskSpec

    def close(self) -> None:
        return None


class DummyBackend:
    """Minimal Backend for unit / smoke verification of ``run_episodes``."""

    name = "dummy"
    capabilities = BackendCapabilities(
        levels=(0, 1, 2, 3),
        has_force_measurement=False,
        has_topple_detection=False,
        force_source="mujoco_mj_contactForce",
    )

    def __init__(
        self,
        tasks: Sequence[TaskSpec] | None = None,
        *,
        config_hash: str = "dummyhash0001",
    ) -> None:
        self._config_hash = config_hash
        self._tasks = list(tasks) if tasks is not None else [
            TaskSpec(
                task_id=0,
                task_name="dummy_task",
                suite="dummy_suite",
                instruction="do nothing",
                max_steps=4,
            )
        ]

    def list_tasks(self, suite: str) -> list[TaskSpec]:
        return [t for t in self._tasks if t.suite == suite] or list(self._tasks)

    def make_env(self, cfg: RunConfig, task: TaskSpec) -> EnvHandle:
        del cfg
        return _DummyEnv(task=task)

    def rollout(
        self,
        env: EnvHandle,
        policy: Policy,
        cfg: RunConfig,
        episode_index: int,
    ) -> EpisodeRecord:
        task = env.task  # type: ignore[attr-defined]
        t0 = time.perf_counter()
        ctx = EpisodeContext(
            instruction=task.instruction,
            task_id=task.task_id,
            task_name=task.task_name,
            episode_index=episode_index,
            seed=int(cfg.seed),
            level=int(cfg.level),
            backend=str(cfg.backend),
            suite=str(cfg.suite),
            extra={},
        )
        policy.reset(ctx)
        obs = Observation(
            images={"primary": np.zeros((8, 8, 3), dtype=np.uint8)},
            state=np.zeros(7, dtype=np.float32),
            instruction=ctx.instruction,
            step=0,
            extra={},
        )
        chunk = policy.act(obs)
        _ = chunk.actions
        success = (episode_index % 2) == 0
        wall = time.perf_counter() - t0
        return EpisodeRecord(
            schema_version=SCHEMA_VERSION,
            run_id=str(cfg.run_id),
            config_hash=self._config_hash,
            backend=str(cfg.backend),
            suite=str(cfg.suite),
            task_id=task.task_id,
            task_name=task.task_name,
            instruction=task.instruction,
            model=str(cfg.model),
            checkpoint=str(cfg.checkpoint),
            level=int(cfg.level),
            seed=int(cfg.seed),
            episode_index=episode_index,
            steps=1,
            max_steps=int(cfg.max_steps),
            termination="success" if success else "max_steps",
            wall_time_s=float(wall),
            success=success,
            success_step=0 if success else None,
            contact_steps=0,
            contact_events=0,
            first_contact_step=None,
            contact_partners={},
            peak_force_n=0.0,
            peak_impulse_ns=0.0,
            cumulative_impulse_ns=0.0,
            force_equiv_dt_s=0.05,
            max_tilt_deg=None,
            toppled=None,
            video_path=None,
            extra={},
        )


class RandomPolicy:
    """Policy that emits random ``delta_ee_7d`` chunks (verification only)."""

    def __init__(self, *, seed: int, action_dim: int = 7, chunk_size: int = 1) -> None:
        self._rng = np.random.default_rng(seed)
        self._action_dim = action_dim
        self._chunk_size = chunk_size

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> RandomPolicy:
        extra = cfg.extra or {}
        seed = int(extra["seed"]) if "seed" in extra else 0
        return cls(seed=seed)

    def reset(self, ctx: EpisodeContext) -> None:
        del ctx

    def act(self, obs: Observation) -> ActionChunk:
        del obs
        actions = self._rng.standard_normal((self._chunk_size, self._action_dim)).astype(
            np.float32
        )
        return ActionChunk(actions=actions, space="delta_ee_7d")

    def close(self) -> None:
        return None


def _make_minimal_run_config(
    *,
    run_id: str,
    seed: int,
    n_episodes: int,
    results_dir: str,
) -> tuple[RunConfig, ResolvedConfig]:
    """Build a tiny RunConfig/ResolvedConfig for dummy verification (not YAML)."""
    from dynisafety.config.schema import (
        ContactConfig,
        DisturbanceConfig,
        DisturbanceL1Config,
        DisturbanceL2Config,
        DisturbanceL3Config,
        HandConfig,
        MetricsConfig,
        ObstacleConfig,
        OutputConfig,
        ReactivityConfig,
        RuntimeConfig,
        SceneConfig,
    )

    # Values are explicit for the dummy path only; production uses YAML via loader.
    contact = ContactConfig(
        topple_threshold_deg=30.0,
        force_equiv_dt="impact_window",
        force_report_window_s=0.05,
        impulse_reduce="sum",
    )
    reactivity = ReactivityConfig(
        window_size=10,
        sensitivity=2.0,
        freeze_threshold=0.01,
        freeze_min_steps=20,
    )
    metrics = MetricsConfig(
        injury_thresholds_n={"safe": 50.0, "minor": 100.0, "moderate": 150.0},
        sasr_penalty={"force_based": [], "contact_based": {}},
        reactivity=reactivity,
    )
    output = OutputConfig(results_dir=results_dir, save_video=False)
    runtime = RuntimeConfig(
        seed=seed,
        seed_stride=2000,
        episodes=n_episodes,
        max_steps=4,
        stop_on_success=False,
        control_freq=5,
        sim_freq=100,
        obj_episode_range=[0, n_episodes],
    )
    obstacle = ObstacleConfig(
        pool=["x"],
        rotation_pool=["identity"],
        models=["x"],
        scale=1.0,
        half_size=[0.1, 0.1, 0.1],
        mass=0.1,
        color=[1.0, 0.0, 0.0],
        selection_mode="fixed",
        target_mode="source_object",
        rescale={},
        upright_quats={},
        semantic_families={},
        default_upright_quat=[1.0, 0.0, 0.0, 0.0],
    )
    l1 = DisturbanceL1Config(
        trigger_dist=0.1,
        drop_height=0.1,
        alpha=0.9,
        settle_steps=1,
        drop_z_bias=0.0,
        trigger_dist_bias=0.0,
        spawn_link_clearance_xy=0.0,
        spawn_z_clearance=0.0,
    )
    l2 = DisturbanceL2Config(
        alpha=0.9,
        move_start_step=1,
        move_end_step=2,
        settle_steps=1,
        arc_height=0.1,
        start_offset=[0.0, 0.0, 0.0],
        align_start_y_to_eef=False,
        outside_table_dist=0.1,
        release_after_settle=False,
    )
    l3 = DisturbanceL3Config(
        alpha=0.9,
        dwell_steps=1,
        entry_steps=1,
        withdraw_speed_mult=1.0,
        arm_speed="medium",
        start_mode="corner",
        enter_height=0.1,
        move_start_step=1,
        flip_side=False,
        corner_inward_shift=0.0,
        corner_entry_backoff=0.0,
        start_offset=[0.0, 0.0, 0.0],
    )
    disturbance = DisturbanceConfig(level=0, l1=l1, l2=l2, l3=l3, obstacle=obstacle)
    hand = HandConfig(
        geometry="box",
        mesh_path="",
        scale=1.0,
        peak_velocity={"x": 0.1, "y": 0.1, "z": 0.1},
        mass_kg=0.1,
        contact_solref=[0.02, 1.0],
        contact_solimp=[0.9, 0.95, 0.001],
        contact_friction=[1.0, 0.005, 0.0001],
        near_eef_m=0.1,
        near_eef_max_step_m=0.1,
        in_contact_max_step_m=0.1,
        big_jump_m=0.5,
        half_size=[0.05, 0.05, 0.05],
        collision_offset=[0.0, 0.0, 0.0],
        enter_height=0.1,
        flinch_dist=0.1,
        pre_physics_obb_half_scale=1.0,
    )
    scene = SceneConfig(
        table_surface_z=0.0,
        table_height=0.0,
        support_clearance_z=0.0,
        obstacle_avoid_margin=0.0,
        l1_spawn_link_clearance_xy=0.0,
        l1_spawn_z_clearance=0.0,
        l1_spawn_eef_clearance_xy=0.0,
        l1_spawn_eef_work_height=0.0,
        l1_spawn_eef_radius=0.0,
        l1_home_approach_eps=0.0,
        l1_spawn_nudge_step=0.0,
        l1_spawn_nudge_max_tries=0,
        held_grasp_dist=0.0,
        held_gripper_closed=0.0,
        l12_collision_exclude_substrings=[],
        l12_contact_require_impulse=False,
        l2_keep_last_free_pose=False,
        l2_release_on_penetration=True,
        obstacle_avoid_shape="disk",
        l2_landing_push_m=0.0,
    )
    cfg = RunConfig(
        backend="dummy",
        suite="dummy_suite",
        level=0,
        seed=seed,
        model="random",
        checkpoint="none",
        max_steps=4,
        run_id=run_id,
        contact=contact,
        metrics=metrics,
        output=output,
        runtime=runtime,
        disturbance=disturbance,
        hand=hand,
        scene=scene,
    )
    data = {
        "backend": "dummy",
        "suite": "dummy_suite",
        "level": 0,
        "model": "random",
        "checkpoint": "none",
        "runtime": {"seed": seed, "episodes": n_episodes, "max_steps": 4},
        "output": {"results_dir": results_dir, "save_video": False},
    }
    resolved = ResolvedConfig(
        data=data,
        provenance={},
        config_hash="dummyhash0001",
        run=cfg,
        warnings=(),
    )
    return cfg, resolved


def run_dummy_smoke(run_root: Path, *, n_episodes: int = 2, seed: int = 0) -> LoopResult:
    """End-to-end loop check with DummyBackend + RandomPolicy (no simulator)."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    task = TaskSpec(
        task_id=0,
        task_name="dummy_task",
        suite="dummy_suite",
        instruction="do nothing",
        max_steps=4,
    )
    run_id = make_run_id(
        backend="dummy",
        model="random",
        suite="dummy_suite",
        level=0,
        seed=seed,
        task_id=task.task_id,
    )
    cfg, resolved = _make_minimal_run_config(
        run_id=run_id,
        seed=seed,
        n_episodes=n_episodes,
        results_dir=str(run_root),
    )
    # Align EpisodeRecord.config_hash with resolved.
    cfg = replace(cfg, run_id=run_id)
    backend = DummyBackend(tasks=[task], config_hash=resolved.config_hash)
    policy = RandomPolicy(seed=seed)
    return run_episodes(
        backend=backend,
        policy=policy,
        cfg=cfg,
        resolved=resolved,
        task=task,
        run_dir=make_run_dir(run_root, level=0, run_id=run_id),
        n_episodes=n_episodes,
        save_video=False,
    )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory(prefix="dynisafety_loop_") as tmp:
        result = run_dummy_smoke(Path(tmp), n_episodes=2, seed=7)
        summary_path = result.run_dir / "summary.json"
        episodes_path = result.run_dir / "episodes.jsonl"
        assert summary_path.is_file(), summary_path
        assert episodes_path.is_file(), episodes_path
        assert result.episodes_written == 2
        assert result.summary["status"] == "completed"
        print(f"dummy smoke ok: {result.run_dir}")
        print(f"summary: {result.summary}")
