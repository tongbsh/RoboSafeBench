"""Probe: upward-facing support surfaces of the CALVIN table, the table-top height and the workspace bounds
(01 §3.4, D13 k).

1. Vertical rays (``rayTestBatch``) on a 5 mm grid from z = 1.5 m down to z = 0 over the table; hits on the table
   whose normal is within 1 degree of +z are grouped by (link, height rounded to 1 mm); for each group with at
   least 400 hits (0.01 m^2) the bounding box of the hit points is printed.
2. The largest rectangle of the table-top group that is completely covered by hits (no holes), shrunk by one grid
   cell: the box written to ``support_surfaces`` in configs/hazard/calvin.yaml.
3. Workspace: TCP position range over every frame of the dataset splits (robot_obs[0:3]), and the scene bodies'
   AABB over all bodies except the floor plane.

    python -m hazardhosts.calvin.probes.surfaces --egl-device 2
"""
import os
import sys

import numpy as np

from hazardhosts.calvin.probes.common import body_names, build_env, link_names, probe_args

GRID_M = 0.005
MIN_HITS = 400


def largest_full_rectangle(mask):
    """(i0, i1, j0, j1) inclusive of the largest all-True axis-aligned rectangle (histogram method)."""
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=int)
    best = (0, None)
    for i in range(rows):
        heights = np.where(mask[i], heights + 1, 0)
        stack = []
        for j in range(cols + 1):
            h = heights[j] if j < cols else 0
            start = j
            while stack and stack[-1][1] >= h:
                start, height = stack.pop()
                area = height * (j - start)
                if area > best[0]:
                    best = (area, (i - height + 1, i, start, j - 1))
            stack.append((start, h))
    return best[1]


def main() -> int:
    args = probe_args(__doc__).parse_args()
    env = build_env(args)
    try:
        env.reset()
        p, cid = env.p, env.cid
        names = body_names(env)
        table = [o.uid for o in env.scene.fixed_objects if o.name == "table"][0]
        links = link_names(env, table)
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for uid, name in sorted(names.items()):
            if name == "plane":
                continue
            for link in links_of(p, cid, uid):
                a, b = p.getAABB(uid, link, physicsClientId=cid)
                lo, hi = np.minimum(lo, a), np.maximum(hi, b)
        # the robot and the blocks would occlude the table: move every other body (except the floor) away
        for k, (uid, name) in enumerate(sorted(names.items())):
            if uid not in (table,) and name != "plane":
                p.resetBasePositionAndOrientation(uid, [10.0 + k, 10.0, -5.0], [0, 0, 0, 1], physicsClientId=cid)
        xs = np.arange(-0.8, 0.8 + 1e-9, GRID_M)
        ys = np.arange(-0.4, 0.4 + 1e-9, GRID_M)
        starts, ends, index = [], [], []
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                starts.append([x, y, 1.5])
                ends.append([x, y, 0.0])
                index.append((i, j))
        hits = []
        for k in range(0, len(starts), 4000):
            hits.extend(p.rayTestBatch(starts[k:k + 4000], ends[k:k + 4000], physicsClientId=cid))
        groups = {}
        for (i, j), hit in zip(index, hits):
            uid, link, _, pos, normal = hit
            if uid != table or normal[2] < np.cos(np.radians(1.0)):
                continue
            groups.setdefault((link, round(pos[2], 3)), []).append((i, j, pos))
        print("== 1. upward table surfaces (>= %d hits on a %.3f m grid)" % (MIN_HITS, GRID_M))
        top_key = None
        for key in sorted(groups, key=lambda k: -len(groups[k])):
            pts = np.array([g[2] for g in groups[key]])
            if len(pts) < MIN_HITS:
                continue
            print("  link %-12s z %.3f  hits %5d  x [%.3f, %.3f]  y [%.3f, %.3f]"
                  % (links[key[0]], key[1], len(pts), pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(),
                     pts[:, 1].max()))
            if top_key is None and key[0] == -1:
                top_key = key
        print("== 2. largest hole-free rectangle of the largest base_link group (shrunk by one cell)")
        cells = groups[top_key]
        mask = np.zeros((len(ys), len(xs)), dtype=bool)
        zs = []
        for i, j, pos in cells:
            mask[i, j] = True
            zs.append(pos[2])
        i0, i1, j0, j1 = largest_full_rectangle(mask)
        i0, i1, j0, j1 = i0 + 1, i1 - 1, j0 + 1, j1 - 1
        print("  table top z = %.5f (hit heights %.5f..%.5f)  x [%.3f, %.3f]  y [%.3f, %.3f]"
              % (float(np.median(zs)), min(zs), max(zs), xs[j0], xs[j1], ys[i0], ys[i1]))
        print("== 3. workspace")
        for split in ("validation", "training"):
            root = os.path.join(args.dataset_dir, split)
            ids = np.load(os.path.join(root, "ep_start_end_ids.npy"))
            tcp = []
            for a, b in ids:
                for f in range(int(a), int(b) + 1):
                    with np.load(os.path.join(root, "episode_%07d.npz" % f)) as d:
                        tcp.append(d["robot_obs"][:3])
            tcp = np.array(tcp)
            print("  %s: %d frames, TCP min %s max %s" % (split, len(tcp), np.round(tcp.min(0), 3).tolist(),
                                                         np.round(tcp.max(0), 3).tolist()))
        print("  scene bodies except the plane (reset pose, before moving them): AABB min %s max %s"
              % (np.round(lo, 3).tolist(), np.round(hi, 3).tolist()))
    finally:
        env.close()
        env.cid = -1
    return 0


def links_of(p, cid, uid):
    return [-1] + list(range(p.getNumJoints(uid, physicsClientId=cid)))


if __name__ == "__main__":
    sys.exit(main())
