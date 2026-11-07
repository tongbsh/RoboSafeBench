"""Probe: which CALVIN tasks the debug dataset's recorded play data completes, and where (task windows for demo
replay, H1 "≥10 tasks spanning push / rotate / drawer / lift").

The debug dataset annotates only a few windows (``lang_annotations/auto_lang_ann.npy``). Here the CALVIN oracle
itself (``Tasks.get_task_info``, the same object ``evaluate_policy`` uses) labels the play data: the env is reset to
every recorded frame and ``env.get_info()`` is stored; for every end frame ``e`` the tasks completed between
``e - max_len`` and ``e`` are listed. A task's *first completion* is an ``e`` where it is complete and was not
complete one frame earlier; its window starts at the latest ``s`` for which the oracle still reports the task
between ``s`` and ``e``, minus ``lead`` frames (so the replay starts before the decisive motion).

Output: the annotated windows next to the oracle's verdict on them, then per task the number of first completions
and the first few replay windows ``<split>:<s>-<e>`` (reset to ``s``, replay the recorded actions up to ``e``);
``--out`` writes all windows as JSON ``{task: [[split, s, e], ...]}``.

    python -m hazardhosts.calvin.probes.windows --egl-device 2 --out windows.json
"""
import json
import os
import sys

import numpy as np

from hazardhosts.calvin.dataset import CalvinDataset
from hazardhosts.calvin.probes.common import build_env, probe_args


def main() -> int:
    parser = probe_args(__doc__)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lead", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    import hydra
    from omegaconf import OmegaConf
    from hazardhosts.calvin.env import calvin_env_root
    oracle = hydra.utils.instantiate(OmegaConf.load(str(
        calvin_env_root().parent / "calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")))
    env = build_env(args)
    splits = {"val": "validation", "train": "training"}
    data = CalvinDataset(args.dataset_dir, splits)
    windows = {}
    try:
        for short, split in sorted(splits.items()):
            ann = np.load(os.path.join(args.dataset_dir, split, "lang_annotations", "auto_lang_ann.npy"),
                          allow_pickle=True).item()
            infos = {}
            for a, b in data.split(short).ranges:
                for f in range(a, b + 1):
                    robot_obs, scene_obs = data.state(short, f)
                    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                    infos[f] = env.get_info()
            print("== %s: %d frames reset" % (split, len(infos)))
            for task, (a, b) in zip(ann["language"]["task"], ann["info"]["indx"]):
                done = oracle.get_task_info_for_set(infos[int(a)], infos[int(b)], {task})
                print("  annotated %-24s %s/%d..%d  oracle on (start, end): %s" % (task, short, a, b, bool(done)))
            for a, b in data.split(short).ranges:
                prev = set()
                for e in range(a + 1, b + 1):
                    s0 = max(a, e - args.max_len)
                    done = oracle.get_task_info(infos[s0], infos[e])
                    for task in sorted(done - prev):
                        s = s0
                        while s + 1 < e and oracle.get_task_info_for_set(infos[s + 1], infos[e], {task}):
                            s += 1
                        start = max(a, s - args.lead)
                        windows.setdefault(task, []).append([short, start, e])
                    prev = done
    finally:
        env.close()
        env.cid = -1
    print("== first completions per task (replay window <split>:<start>-<end>)")
    for task in sorted(windows):
        shown = " ".join("%s:%d-%d" % tuple(w) for w in windows[task][:4])
        print("  %-28s %3d  %s" % (task, len(windows[task]), shown))
    print("tasks: %d" % len(windows))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(windows, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
