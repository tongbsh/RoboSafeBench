"""Add similar-sized GSO extras on unused SafeLIBERO slots (reversible).

Official types keep stock visual + collision. Pack does **not** replace them.
When enabled, unused BDDL slots that can host a similar-sized GSO (or are
already similar in stock form) are added and some official episodes are
stolen so the new types appear across trajectories. Tasks that already use
all 6 slots are unchanged.

Does not add free joints or rewrite ``pruned_init``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_PACK_ROOT = Path(__file__).resolve().parent / "assets" / "obstacle_pack_gso"
_OFFICIAL_JSON = DEFAULT_PACK_ROOT / "official_identities.json"

_REQUIRED_KEYS = (
    "enabled",
    "assets",
    "height_ratio_min",
    "height_ratio_max",
    "footprint_ratio_min",
    "footprint_ratio_max",
)

# Standing AABB (min_xy_cm, max_xy_cm, height_cm) from compiled stock / GSO XML.
STOCK_STANDING_CM: dict[str, tuple[float, float, float]] = {
    "wine_bottle_obstacle": (6.9, 6.9, 25.4),
    "wine_bottle_small_obstacle": (4.3, 4.3, 15.9),
    "red_coffee_mug_obstacle": (9.1, 12.6, 13.5),
    "yellow_book_obstacle": (3.0, 11.3, 13.8),
    "moka_pot_obstacle": (12.7, 23.2, 24.7),
    "moka_pot_small_obstacle": (7.9, 14.5, 15.4),
    "milk_obstacle": (10.8, 11.0, 28.6),
    "milk_small_obstacle": (5.4, 5.5, 14.3),
    "white_storage_box_obstacle": (11.9, 13.9, 14.2),
}
GSO_STANDING_CM: dict[str, tuple[float, float, float]] = {
    "gso_weston_tonic_22": (6.3, 6.4, 19.9),
    "gso_weston_tonic_33": (6.2, 6.5, 19.9),
    "gso_ace_mug": (9.1, 12.6, 13.5),
    "gso_threshold_mug": (9.7, 13.1, 11.5),
}

# Carrier local frame ← GSO (Z-up, zmin≈0).
_CARRIER_ALIGN: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
    "moka_pot_obstacle": ((0.0, 0.0, -0.1074), (1.0, 0.0, 0.0, 0.0)),
    "moka_pot_small_obstacle": ((0.0, 0.0, -0.0671), (1.0, 0.0, 0.0, 0.0)),
    "milk_obstacle": ((0.0, -0.1428, 0.0), (0.70711, -0.70711, 0.0, 0.0)),
    "milk_small_obstacle": ((0.0, -0.0714, 0.0), (0.70711, -0.70711, 0.0, 0.0)),
    "white_storage_box_obstacle": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    "yellow_book_obstacle": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    "wine_bottle_obstacle": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    "wine_bottle_small_obstacle": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    "red_coffee_mug_obstacle": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
}


@dataclass(frozen=True)
class ObstaclePack:
    enabled: bool
    assets: tuple[str, ...]
    height_ratio_min: float
    height_ratio_max: float
    footprint_ratio_min: float
    footprint_ratio_max: float
    root: Path
    remesh: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskPackPlan:
    remesh: dict[str, str]
    extras: tuple[str, ...]
    planned: tuple[str, ...]


def default_pack_root() -> Path:
    return DEFAULT_PACK_ROOT


def instance_to_type(name: str) -> str:
    return name[:-2] if name.endswith("_1") else name


def similar_standing(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    *,
    height_lo: float,
    height_hi: float,
    footprint_lo: float,
    footprint_hi: float,
) -> bool:
    """True if height and footprint area (min_xy * max_xy) both sit in ratio bands."""
    ha, hb = a[2], b[2]
    if ha <= 0 or hb <= 0:
        return False
    hr = ha / hb
    if hr < height_lo or hr > height_hi:
        return False
    area_a = a[0] * a[1]
    area_b = b[0] * b[1]
    if area_a <= 0 or area_b <= 0:
        return False
    ar = area_a / area_b
    return footprint_lo <= ar <= footprint_hi


def parse_obstacle_pack(
    extra_libero: Mapping[str, Any] | None,
    *,
    root: Path | None = None,
) -> ObstaclePack | None:
    """Parse ``extra.libero.obstacle_pack``. Missing block ⇒ disabled (``None``)."""
    if extra_libero is None or "obstacle_pack" not in extra_libero:
        return None
    raw = extra_libero["obstacle_pack"]
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"extra.libero.obstacle_pack must be a mapping, got {type(raw)!r}"
        )
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(
            f"extra.libero.obstacle_pack missing keys {missing} from resolved config"
        )
    enabled = raw["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(
            f"extra.libero.obstacle_pack.enabled must be bool, got {type(enabled)!r}"
        )
    assets_raw = raw["assets"]
    if not isinstance(assets_raw, (list, tuple)):
        raise TypeError(
            f"extra.libero.obstacle_pack.assets must be a list, got {type(assets_raw)!r}"
        )
    assets = tuple(str(item).strip() for item in assets_raw if str(item).strip())
    if enabled and not assets:
        raise ValueError("extra.libero.obstacle_pack.enabled is true but assets is empty")
    pack = ObstaclePack(
        enabled=enabled,
        assets=assets,
        height_ratio_min=float(raw["height_ratio_min"]),
        height_ratio_max=float(raw["height_ratio_max"]),
        footprint_ratio_min=float(raw["footprint_ratio_min"]),
        footprint_ratio_max=float(raw["footprint_ratio_max"]),
        root=(root or default_pack_root()).resolve(),
    )
    if pack.height_ratio_min <= 0 or pack.height_ratio_max < pack.height_ratio_min:
        raise ValueError("extra.libero.obstacle_pack height_ratio_* is invalid")
    if pack.footprint_ratio_min <= 0 or pack.footprint_ratio_max < pack.footprint_ratio_min:
        raise ValueError("extra.libero.obstacle_pack footprint_ratio_* is invalid")
    if enabled:
        _require_mesh_assets(pack)
    return pack


def with_remesh(pack: ObstaclePack, remesh: Mapping[str, str]) -> ObstaclePack:
    return replace(pack, remesh=dict(remesh))


def resolve_obstacle_xml(obj_name: str, pack: ObstaclePack | None) -> Path | None:
    """Return pack XML for a remeshed **type**, or ``None`` to keep stock."""
    if pack is None or not pack.enabled:
        return None
    asset_id = pack.remesh.get(obj_name)
    if not asset_id:
        return None
    return _ensure_carrier_xml(pack.root, obj_name, asset_id)


def rewrite_obstacle_xml_path(fname: str, pack: ObstaclePack | None) -> str:
    if pack is None or not pack.enabled or not pack.remesh:
        return fname
    path = Path(fname)
    if path.parent.name != path.stem:
        return fname
    if path.parent.parent.name != "obstacle_objects":
        return fname
    xml = resolve_obstacle_xml(path.stem, pack)
    return str(xml) if xml is not None else fname


@lru_cache(maxsize=1)
def load_official_identities(path: Path | None = None) -> dict[str, Any]:
    src = path or _OFFICIAL_JSON
    return json.loads(src.read_text())


def official_task_record(
    suite: str,
    task_name: str,
    *,
    table: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    root = table if table is not None else load_official_identities()
    suite_map = root.get(suite)
    if not isinstance(suite_map, Mapping):
        return None
    rec = suite_map.get(task_name)
    return rec if isinstance(rec, Mapping) else None


def assign_task_extras(
    pack: ObstaclePack,
    *,
    slots: Sequence[str],
    official_episodes: Sequence[str],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Pick unused slots that are already similar, then remesh the rest to GSO."""
    used = {name for name in official_episodes if name}
    unused = [name for name in slots if name not in used]
    official_aabbs = [
        STOCK_STANDING_CM[instance_to_type(name)]
        for name in used
        if instance_to_type(name) in STOCK_STANDING_CM
    ]
    if not official_aabbs:
        return {}, ()

    def _ok(aabb: tuple[float, float, float]) -> bool:
        return any(
            similar_standing(
                aabb,
                other,
                height_lo=pack.height_ratio_min,
                height_hi=pack.height_ratio_max,
                footprint_lo=pack.footprint_ratio_min,
                footprint_hi=pack.footprint_ratio_max,
            )
            for other in official_aabbs
        )

    extras: list[str] = []
    remesh: dict[str, str] = {}
    leftover: list[str] = []
    for inst in unused:
        typ = instance_to_type(inst)
        stock = STOCK_STANDING_CM.get(typ)
        if stock is not None and _ok(stock):
            extras.append(inst)
        else:
            leftover.append(inst)

    similar_gso = [
        asset
        for asset in pack.assets
        if asset in GSO_STANDING_CM and _ok(GSO_STANDING_CM[asset])
    ]
    taken = set()
    for inst in leftover:
        typ = instance_to_type(inst)
        if typ not in _CARRIER_ALIGN:
            continue
        for asset in similar_gso:
            if asset in taken:
                continue
            remesh[typ] = asset
            extras.append(inst)
            taken.add(asset)
            break
    return remesh, tuple(extras)


def steal_episode_names(
    official: Sequence[str],
    extras: Sequence[str],
) -> tuple[str, ...]:
    """Keep every official type; steal surplus episodes so extras also appear."""
    planned = list(official)
    extra_list = [name for name in extras if name]
    if not extra_list or not planned:
        return tuple(planned)
    official_order: list[str] = []
    seen: set[str] = set()
    for name in planned:
        if name and name not in seen:
            official_order.append(name)
            seen.add(name)
    all_types = official_order + [name for name in extra_list if name not in seen]
    n = len(planned)
    n_all = len(all_types)
    if n_all > n:
        all_types = all_types[:n]
        extra_list = [name for name in extra_list if name in all_types]
        n_all = len(all_types)
    base, rem = divmod(n, n_all)
    target = {name: base for name in all_types}
    for name in official_order:
        if rem == 0:
            break
        target[name] += 1
        rem -= 1
    official_counts = Counter(planned)
    surplus = {name: official_counts[name] - target.get(name, 0) for name in official_order}
    need = {name: target[name] for name in extra_list}
    for i in range(n - 1, -1, -1):
        cur = planned[i]
        if surplus.get(cur, 0) <= 0:
            continue
        for extra in extra_list:
            if need[extra] <= 0:
                continue
            planned[i] = extra
            surplus[cur] -= 1
            need[extra] -= 1
            break
    return tuple(planned)


def plan_for_task(
    pack: ObstaclePack | None,
    suite: str,
    task_name: str,
    *,
    table: Mapping[str, Any] | None = None,
) -> TaskPackPlan | None:
    if pack is None or not pack.enabled:
        return None
    rec = official_task_record(suite, task_name, table=table)
    if rec is None:
        return None
    slots = [str(x) for x in rec.get("slots") or ()]
    episodes = [str(x) for x in rec.get("episodes") or () if x]
    remesh, extras = assign_task_extras(pack, slots=slots, official_episodes=episodes)
    planned = steal_episode_names(episodes, extras)
    return TaskPackPlan(remesh=remesh, extras=extras, planned=planned)


def planned_instance(
    plan: TaskPackPlan | None,
    episode_index: int,
    official_name: str,
) -> str:
    """Return the extra to swap in, or keep the detected official identity."""
    if plan is None or not plan.planned:
        return official_name
    desired = plan.planned[int(episode_index) % len(plan.planned)]
    if desired == official_name or desired in plan.extras:
        return desired
    return official_name


def _require_mesh_assets(pack: ObstaclePack) -> None:
    missing = [
        asset
        for asset in pack.assets
        if not (pack.root / "meshes" / asset / "model.obj").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "obstacle pack enabled but GSO meshes missing for "
            f"{missing}; run scripts/libero/materialize_obstacle_pack_gso.py"
        )


def _ensure_carrier_xml(root: Path, carrier: str, asset_id: str) -> Path:
    xml = root / carrier / f"{asset_id}.xml"
    if xml.is_file():
        return xml
    align = _CARRIER_ALIGN.get(carrier)
    mesh_dir = root / "meshes" / asset_id
    if align is None or not (mesh_dir / "model.obj").is_file():
        raise FileNotFoundError(f"cannot build pack XML for {carrier} -> {asset_id}")
    collisions = sorted(
        mesh_dir.glob("model_collision_*.obj"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    pos, quat = align
    rel = f"../meshes/{asset_id}"
    col_lines = "\n".join(
        f'        <geom type="mesh" mesh="{p.stem}" group="0" '
        f'rgba="0.8 0.8 0.8 0.3" density="100" friction="0.95 0.3 0.1" '
        f'solimp="0.998 0.998 0.001" solref="0.001 1"/>'
        for p in collisions
    )
    mesh_lines = "\n".join(
        f'    <mesh name="{p.stem}" file="{rel}/{p.name}"/>' for p in collisions
    )
    xml.parent.mkdir(parents=True, exist_ok=True)
    xml.write_text(
        f"""<mujoco model="{carrier}_{asset_id}">
  <asset>
    <texture type="2d" name="tex-gso" file="{rel}/texture.png"/>
    <material name="mat-gso" texture="tex-gso" reflectance="0.5" specular="0.5"/>
    <mesh name="visual" file="{rel}/model.obj"/>
{mesh_lines}
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <body name="gso_align" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" quat="{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}">
        <geom type="mesh" mesh="visual" material="mat-gso" group="1"
              conaffinity="0" contype="0" density="100" friction="0.95 0.3 0.1"
              solimp="0.998 0.998 0.001" solref="0.001 1"/>
{col_lines}
        </body>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 -0.06" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0.04" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0.025 0.025 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
    )
    return xml


@contextmanager
def overlay_obstacle_pack(pack: ObstaclePack | None) -> Iterator[None]:
    """Remap unused carrier XML paths onto GSO while constructing the env."""
    if pack is None or not pack.enabled or not pack.remesh:
        yield
        return

    from robosuite.models.objects import MujocoXMLObject

    original = MujocoXMLObject.__init__

    def _patched(self: Any, fname: str, *args: Any, **kwargs: Any) -> None:
        original(self, rewrite_obstacle_xml_path(str(fname), pack), *args, **kwargs)

    MujocoXMLObject.__init__ = _patched  # type: ignore[method-assign]
    try:
        mapped = ", ".join(f"{src}->{dst}" for src, dst in sorted(pack.remesh.items()))
        print(f"[obstacle_pack] remesh unused {mapped}", flush=True)
        yield
    finally:
        MujocoXMLObject.__init__ = original  # type: ignore[method-assign]
