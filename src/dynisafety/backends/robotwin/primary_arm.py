"""Primary-arm resolver registry (D3=B — logic migrated from vendored task envs)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

PrimaryArmResolver = Callable[[Any], Any]

_REGISTRY: dict[str, PrimaryArmResolver] = {}


def register_primary_arm(task_name: str, resolver: PrimaryArmResolver) -> None:
    _REGISTRY[task_name] = resolver


def resolve_primary_arm_tag(task_env: Any, *, task_name: str | None = None) -> Any:
    """Look up by ``task_name`` (prefer explicit, else ``type(task_env).__name__``).

    Missing registration → ``KeyError``.
    """
    name = task_name or type(task_env).__name__
    if name not in _REGISTRY:
        raise KeyError(f"No primary_arm resolver registered for task {name!r}")
    return _REGISTRY[name](task_env)


def ensure_open_laptop_arm_tag(task_env: Any) -> None:
    """Set ``open_laptop.arm_tag`` when policy eval skipped ``play_once``.

    Runtime ``check_success`` reads ``self.arm_tag`` on the first sim step.
    Expert-check cache hits never call ``play_once``, so the attribute is
    missing. Only this task is touched; existing values are left alone.
    """
    if type(task_env).__name__ != "open_laptop":
        return
    if getattr(task_env, "arm_tag", None) is not None:
        return
    task_env.arm_tag = resolve_primary_arm_tag(task_env)


def _arm_tag(value: str) -> Any:
    from envs.utils.action import ArmTag

    return ArmTag(value)


def _pose_x(actor: Any) -> float:
    return float(actor.get_pose().p[0])


def _resolve_lift_pot(_task: Any) -> Any:
    return _arm_tag("left")


def _resolve_pick_dual_bottles(_task: Any) -> Any:
    return _arm_tag("left")


def _resolve_adjust_bottle(task: Any) -> Any:
    return _arm_tag("right" if task.qpose_tag == 1 else "left")


def _resolve_beat_block_hammer(task: Any) -> Any:
    block_pose = task.block.get_functional_point(0, "pose").p
    return _arm_tag("left" if block_pose[0] < 0 else "right")


def _resolve_blocks_ranking_rgb(task: Any) -> Any:
    return _arm_tag("left" if _pose_x(task.block1) < 0 else "right")


def _resolve_stack_blocks_two(task: Any) -> Any:
    return _arm_tag("left" if _pose_x(task.block1) < 0 else "right")


def _resolve_stack_bowls_three(task: Any) -> Any:
    return _arm_tag("left" if _pose_x(task.bowl1) < 0 else "right")


def _resolve_click_bell(task: Any) -> Any:
    return _arm_tag("right" if _pose_x(task.bell) > 0 else "left")


def _resolve_handover_block(task: Any) -> Any:
    return _arm_tag("left" if _pose_x(task.box) < 0 else "right")


def _resolve_move_can_pot(task: Any) -> Any:
    return task.arm_tag


def _resolve_move_pillbottle_pad(task: Any) -> Any:
    return _arm_tag("right" if _pose_x(task.pillbottle) > 0 else "left")


def _resolve_move_playingcard_away(task: Any) -> Any:
    return _arm_tag("right" if _pose_x(task.playingcards) > 0 else "left")


def _resolve_move_stapler_pad(task: Any) -> Any:
    return _arm_tag("right" if _pose_x(task.stapler) > 0 else "left")


def _resolve_open_laptop(task: Any) -> Any:
    from envs.utils.transforms import get_face_prod

    face_prod = get_face_prod(task.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
    return _arm_tag("left" if face_prod > 0 else "right")


def _resolve_place_a2b_left(task: Any) -> Any:
    return _arm_tag("right" if _pose_x(task.object) > 0 else "left")


def _resolve_place_a2b_right(task: Any) -> Any:
    return _arm_tag("right" if _pose_x(task.object) > 0 else "left")


def _resolve_place_empty_cup(task: Any) -> Any:
    return _arm_tag("right" if _pose_x(task.cup) > 0 else "left")


def _resolve_place_mouse_pad(task: Any) -> Any:
    return _arm_tag("right" if _pose_x(task.mouse) > 0 else "left")


def _resolve_shake_bottle(task: Any) -> Any:
    return _arm_tag("right" if _pose_x(task.bottle) > 0 else "left")


def _resolve_stamp_seal(task: Any) -> Any:
    return _arm_tag("right" if _pose_x(task.seal) > 0 else "left")


register_primary_arm("lift_pot", _resolve_lift_pot)
register_primary_arm("pick_dual_bottles", _resolve_pick_dual_bottles)
register_primary_arm("adjust_bottle", _resolve_adjust_bottle)
register_primary_arm("beat_block_hammer", _resolve_beat_block_hammer)
register_primary_arm("blocks_ranking_rgb", _resolve_blocks_ranking_rgb)
register_primary_arm("stack_blocks_two", _resolve_stack_blocks_two)
register_primary_arm("stack_bowls_three", _resolve_stack_bowls_three)
register_primary_arm("click_bell", _resolve_click_bell)
register_primary_arm("handover_block", _resolve_handover_block)
register_primary_arm("move_can_pot", _resolve_move_can_pot)
register_primary_arm("move_pillbottle_pad", _resolve_move_pillbottle_pad)
register_primary_arm("move_playingcard_away", _resolve_move_playingcard_away)
register_primary_arm("move_stapler_pad", _resolve_move_stapler_pad)
register_primary_arm("open_laptop", _resolve_open_laptop)
register_primary_arm("place_a2b_left", _resolve_place_a2b_left)
register_primary_arm("place_a2b_right", _resolve_place_a2b_right)
register_primary_arm("place_empty_cup", _resolve_place_empty_cup)
register_primary_arm("place_mouse_pad", _resolve_place_mouse_pad)
register_primary_arm("shake_bottle", _resolve_shake_bottle)
register_primary_arm("stamp_seal", _resolve_stamp_seal)

__all__ = [
    "PrimaryArmResolver",
    "ensure_open_laptop_arm_tag",
    "register_primary_arm",
    "resolve_primary_arm_tag",
]
