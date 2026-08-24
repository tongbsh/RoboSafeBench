"""Backend protocols and capability declarations (§3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from dynisafety.core.episode import EpisodeRecord

if TYPE_CHECKING:
    # RunConfig lands in config/schema.py (later M0 batch / M1 loader).
    from dynisafety.config.schema import RunConfig
    from dynisafety.policies.base import Policy


@dataclass(frozen=True)
class BackendCapabilities:
    levels: tuple[int, ...]
    has_force_measurement: bool
    has_topple_detection: bool
    force_source: str  # "mujoco_mj_contactForce" | "sapien_impulse"


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    task_name: str
    suite: str
    instruction: str
    max_steps: int


class EnvHandle(Protocol):
    """Opaque env handle; backends may expose more structurally."""

    def close(self) -> None: ...


class Backend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def list_tasks(self, suite: str) -> list[TaskSpec]: ...

    def make_env(self, cfg: RunConfig, task: TaskSpec) -> EnvHandle: ...

    def rollout(
        self,
        env: EnvHandle,
        policy: Policy,
        cfg: RunConfig,
        episode_index: int,
    ) -> EpisodeRecord: ...
