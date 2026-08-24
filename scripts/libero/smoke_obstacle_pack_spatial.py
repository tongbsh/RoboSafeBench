#!/usr/bin/env python3
"""Spatial L1 smoke: pack off vs on, visual + metric probes (no VLA).

Writes under artifacts/debug/pi05_libero_spatial_smoke/obstacle_pack_check/.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/debug/pi05_libero_spatial_smoke/obstacle_pack_check"
SUITE = "safelibero_spatial"


def _setup(task_id: int, *, pack: bool) -> tuple[Any, Any, Any, Any]:
    from dynisafety.backends.libero.backend import LiberoBackend
    from dynisafety.config.loader import LoadRequest, load_config
    from dynisafety.policies.base import model_config_from_resolved

    overrides = [
        "output.save_video=false",
        "runtime.episodes=1",
        f"extra.libero.obstacle_pack.enabled={'true' if pack else 'false'}",
    ]
    resolved = load_config(
        LoadRequest(
            backend="libero",
            level=1,
            suite=SUITE,
            task=str(task_id),
            model="random",
            overrides=tuple(overrides),
        )
    )
    data = dict(resolved.data)
    if "name" not in data:
        data["name"] = "random"
    model_cfg = model_config_from_resolved(data, backend="libero")
    backend = LiberoBackend()
    backend.bind_run(
        config_hash=resolved.config_hash,
        resolved_data=resolved.data,
        action_space=model_cfg.action_space,
        chunk_size=model_cfg.chunk_size,
        observation_images=model_cfg.observation_images,
        observation_needs_state=model_cfg.observation_needs_state,
        model_extra=model_cfg.extra,
    )
    tasks = backend.list_tasks(SUITE)
    task = next(item for item in tasks if int(item.task_id) == int(task_id))
    handle = backend.make_env(resolved.run, task)
    return backend, handle, resolved, backend._extra_libero  # noqa: SLF001


def _reset_l1(backend: Any, handle: Any, cfg: Any, extra_libero: Mapping[str, Any], episode: int):
    from dynisafety.backends.libero.backend import _apply_obstacle_rescale
    from dynisafety.backends.libero.disturbance.assemble import build_controller
    from dynisafety.backends.libero.env import pad_init_state_for_hand

    handle.env.reset()
    _apply_obstacle_rescale(handle, cfg)
    init_state = backend._load_init_state(handle, episode)  # noqa: SLF001
    padded = pad_init_state_for_hand(init_state, handle)
    handle.env.set_init_state(padded)
    return build_controller(
        1,
        handle,
        cfg,
        episode_index=episode,
        extra_libero=extra_libero,
    )


def _place_at_landing(ctrl: Any) -> None:
    ctrl._set_pos(ctrl.target_pos, ctrl.target_quat)  # noqa: SLF001


def _geom_probe(handle: Any, obstacle_name: str, geom_ids: set[int]) -> dict[str, Any]:
    import mujoco

    from dynisafety.backends.libero.safety.aegis_bridge import _obstacle_axes_from_body

    sim = handle.env.env.sim
    model = sim.model
    obj = handle.env.env.objects_dict[obstacle_name]
    body_id = int(model.body_name2id(obj.root_body))
    types: list[int] = []
    sizes: list[list[float]] = []
    rbounds: list[float] = []
    groups: list[int] = []
    for gid in sorted(geom_ids):
        types.append(int(model.geom_type[gid]))
        sizes.append([float(x) for x in np.asarray(model.geom_size[gid]).reshape(-1)[:3]])
        rbounds.append(float(model.geom_rbound[gid]))
        groups.append(int(model.geom_group[gid]))
    collision = [gid for gid in geom_ids if int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid])]
    aabb_min = np.array([math.inf, math.inf, math.inf])
    aabb_max = np.array([-math.inf, -math.inf, -math.inf])
    for gid in collision:
        pos = np.asarray(sim.data.geom_xpos[gid], dtype=np.float64)
        rb = max(float(model.geom_rbound[gid]), 1e-4)
        aabb_min = np.minimum(aabb_min, pos - rb)
        aabb_max = np.maximum(aabb_max, pos + rb)
    extent = aabb_max - aabb_min
    axes_geom = _obstacle_axes_from_body(
        sim,
        body_id,
        fallback=np.array([0.06, 0.06, 0.08]),
        use_geom_size=True,
    )
    axes_fallback = _obstacle_axes_from_body(
        sim,
        body_id,
        fallback=np.array([0.06, 0.06, 0.08]),
        use_geom_size=False,
    )
    footprint = 0.0
    for gid in collision:
        gtype = int(model.geom_type[gid])
        size = model.geom_size[gid]
        local_xy = np.asarray(model.geom_pos[gid][:2], dtype=np.float64)
        if gtype in (2, 3, 5):
            ext = float(size[0])
        elif gtype == 4:
            ext = float(max(size[0], size[1]))
        elif gtype == 6:
            ext = float(np.hypot(size[0], size[1]))
        else:
            ext = float(model.geom_rbound[gid])
        footprint = max(footprint, float(np.linalg.norm(local_xy) + ext))
    return {
        "n_geoms_controller": len(geom_ids),
        "n_collision_geoms": len(collision),
        "geom_types": sorted(set(types)),
        "max_geom_size": max((max(s) if s else 0.0) for s in sizes) if sizes else 0.0,
        "max_rbound": max(rbounds) if rbounds else 0.0,
        "aabb_extent_xyz": [float(x) for x in extent],
        "aegis_axes_geom_size": [float(x) for x in axes_geom],
        "aegis_axes_fallback": [float(x) for x in axes_fallback],
        "l2_footprint_radius": float(footprint),
        "qpos_has_nan": bool(np.any(np.isnan(sim.data.qpos))),
        "body_xy": [float(x) for x in sim.data.body_xpos[body_id][:2]],
        "body_z": float(sim.data.body_xpos[body_id][2]),
        "mj_version": mujoco.__version__,
    }


def _contact_probe(handle: Any, ctrl: Any) -> dict[str, Any]:
    from dynisafety.backends.libero.contact import (
        collect_robot_geom_ids,
        collect_task_object_geom_ids,
        measure_contact_step,
        measure_tilt_deg,
    )

    env = handle.env
    arm = collect_robot_geom_ids(env)
    task = collect_task_object_geom_ids(env)
    obstacle = set(ctrl.obstacle_geom_ids)
    scored = measure_contact_step(
        env,
        robot_geom_ids=arm | task,
        obstacle_geom_ids=obstacle,
        force_equiv_dt_s=0.05,
        impulse_reduce="sum",
    )
    vs_arm = measure_contact_step(
        env,
        robot_geom_ids=arm,
        obstacle_geom_ids=obstacle,
        force_equiv_dt_s=0.05,
        impulse_reduce="sum",
    )
    vs_task = measure_contact_step(
        env,
        robot_geom_ids=task,
        obstacle_geom_ids=obstacle,
        force_equiv_dt_s=0.05,
        impulse_reduce="sum",
    )
    body_name = env.env.sim.model.body_id2name(int(ctrl._body_id))  # noqa: SLF001
    tilt = measure_tilt_deg(env, body_name, ref_quat=np.asarray(ctrl.target_quat))
    raw_ncon = 0
    table_like = 0
    sim = env.env.sim
    for i in range(int(sim.data.ncon)):
        con = sim.data.contact[i]
        pair = {int(con.geom1), int(con.geom2)}
        if pair & obstacle:
            raw_ncon += 1
            names = []
            for gid in pair - obstacle:
                try:
                    names.append(str(sim.model.geom_id2name(gid) or ""))
                except Exception:
                    names.append("")
            blob = " ".join(names).lower()
            if any(key in blob for key in ("table", "floor", "worksurface")):
                table_like += 1
    return {
        "scored_in_contact": bool(scored.in_contact),
        "scored_force_n": float(scored.force_n),
        "scored_partners": dict(scored.partners or {}),
        "vs_arm_in_contact": bool(vs_arm.in_contact),
        "vs_task_in_contact": bool(vs_task.in_contact),
        "vs_task_force_n": float(vs_task.force_n),
        "raw_obstacle_ncon": raw_ncon,
        "table_like_ncon": table_like,
        "tilt_deg": float(tilt),
    }


def _save_image(handle: Any, path: Path) -> float:
    from dynisafety.backends.libero.env import get_primary_image
    from PIL import Image

    handle.env._update_observables(force=True)
    obs = handle.env.env._get_observations()
    img = np.asarray(get_primary_image(obs))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return float(img.mean())


def _image_mse(a: Path, b: Path) -> float:
    from PIL import Image

    ia = np.asarray(Image.open(a), dtype=np.float32)
    ib = np.asarray(Image.open(b), dtype=np.float32)
    if ia.shape != ib.shape:
        return float("inf")
    return float(np.mean((ia - ib) ** 2))


def _probe_one(
    *,
    tag: str,
    task_id: int,
    pack: bool,
    episode: int,
    backend: Any,
    handle: Any,
    resolved: Any,
    extra_libero: Mapping[str, Any],
) -> dict[str, Any]:
    from dynisafety.backends.libero.obstacle_pack import parse_obstacle_pack, plan_for_task

    ctrl = _reset_l1(backend, handle, resolved.run, extra_libero, episode)
    if ctrl is None:
        return {"tag": tag, "ok": False, "error": "controller_unbound"}
    _place_at_landing(ctrl)
    img_path = OUT / "frames" / f"{tag}.png"
    mean_rgb = _save_image(handle, img_path)
    geom = _geom_probe(handle, str(ctrl.obstacle_name), set(ctrl.obstacle_geom_ids))
    contact = _contact_probe(handle, ctrl)
    pack_obj = parse_obstacle_pack(extra_libero)
    plan = plan_for_task(pack_obj, handle.suite, handle.task_name)
    row = {
        "tag": tag,
        "ok": True,
        "task_id": task_id,
        "task_name": handle.task_name,
        "pack": pack,
        "episode": episode,
        "obstacle_name": str(ctrl.obstacle_name),
        "planned": None if plan is None else plan.planned[episode % len(plan.planned)],
        "extras": [] if plan is None else list(plan.extras),
        "remesh": {} if plan is None else dict(plan.remesh),
        "image": str(img_path.relative_to(OUT)),
        "image_mean_rgb": mean_rgb,
        **geom,
        **contact,
    }
    return row


def _pick_t3_episodes(extra_libero: Mapping[str, Any], task_name: str) -> tuple[int, int]:
    from dynisafety.backends.libero.obstacle_pack import parse_obstacle_pack, plan_for_task

    pack = parse_obstacle_pack(extra_libero)
    pack_on = pack
    if pack is None or not pack.enabled:
        raise RuntimeError("need pack enabled to pick stolen episodes")
    plan = plan_for_task(pack_on, SUITE, task_name)
    if plan is None or not plan.extras:
        raise RuntimeError(f"task {task_name} has no extras")
    official = next(i for i, name in enumerate(plan.planned) if name not in plan.extras)
    extra = next(i for i, name in enumerate(plan.planned) if name in plan.extras)
    return official, extra


def _close(handle: Any) -> None:
    closer = getattr(handle.env, "close", None)
    if callable(closer):
        closer()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    print("== t0 pack-off / pack-on (should be identical)", flush=True)
    for pack in (False, True):
        backend, handle, resolved, extra = _setup(0, pack=pack)
        try:
            rows.append(
                _probe_one(
                    tag=f"t0_ep0_pack{int(pack)}",
                    task_id=0,
                    pack=pack,
                    episode=0,
                    backend=backend,
                    handle=handle,
                    resolved=resolved,
                    extra_libero=extra,
                )
            )
        finally:
            _close(handle)

    print("== t3 pick stolen vs kept episodes", flush=True)
    backend, handle, resolved, extra = _setup(3, pack=True)
    try:
        kept, stolen = _pick_t3_episodes(extra, handle.task_name)
        print(f"   kept_ep={kept} stolen_ep={stolen}", flush=True)
        for episode, label in ((kept, "kept"), (stolen, "stolen")):
            rows.append(
                _probe_one(
                    tag=f"t3_ep{episode}_{label}_pack1",
                    task_id=3,
                    pack=True,
                    episode=episode,
                    backend=backend,
                    handle=handle,
                    resolved=resolved,
                    extra_libero=extra,
                )
            )
    finally:
        _close(handle)

    backend, handle, resolved, extra = _setup(3, pack=False)
    try:
        for episode, label in ((kept, "kept"), (stolen, "stolen")):
            rows.append(
                _probe_one(
                    tag=f"t3_ep{episode}_{label}_pack0",
                    task_id=3,
                    pack=False,
                    episode=episode,
                    backend=backend,
                    handle=handle,
                    resolved=resolved,
                    extra_libero=extra,
                )
            )
    finally:
        _close(handle)

    comparisons = {
        "t0_pack_off_vs_on_mse": _image_mse(
            OUT / "frames/t0_ep0_pack0.png",
            OUT / "frames/t0_ep0_pack1.png",
        ),
        "t3_kept_pack_off_vs_on_mse": _image_mse(
            OUT / f"frames/t3_ep{kept}_kept_pack0.png",
            OUT / f"frames/t3_ep{kept}_kept_pack1.png",
        ),
        "t3_stolen_pack_off_vs_on_mse": _image_mse(
            OUT / f"frames/t3_ep{stolen}_stolen_pack0.png",
            OUT / f"frames/t3_ep{stolen}_stolen_pack1.png",
        ),
    }
    payload = {"rows": rows, "comparisons": comparisons}
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    _write_report(payload)
    print(json.dumps(comparisons, indent=2), flush=True)
    return 0


def _write_report(payload: Mapping[str, Any]) -> None:
    lines = [
        "# spatial obstacle-pack smoke",
        "",
        "对照 `artifacts/debug/pi05_libero_spatial_smoke`：不开 VLA，只做 L1 落点视觉 + 指标探针。",
        "",
        "## 画面 MSE（pack 关 vs 开）",
        "",
    ]
    for key, value in payload["comparisons"].items():
        lines.append(f"- `{key}`: {value:.3f}")
    lines += ["", "## 每条探针", ""]
    keys = (
        "tag",
        "obstacle_name",
        "n_collision_geoms",
        "max_geom_size",
        "max_rbound",
        "aegis_axes_geom_size",
        "l2_footprint_radius",
        "scored_in_contact",
        "vs_task_in_contact",
        "tilt_deg",
        "qpos_has_nan",
    )
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join("---" for _ in keys) + " |")
    for row in payload["rows"]:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
