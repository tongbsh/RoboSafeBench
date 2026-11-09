"""CALVIN dataset frames as reset targets and demonstrations (H1).

A CALVIN dataset split is a directory of ``episode_<frame>.npz`` files (one per control step of the recorded play
data, 30 Hz) plus ``lang_annotations/auto_lang_ann.npy`` (``info.indx``: ``(start, end)`` frame windows,
``language.task``: the subtask of each window). Each frame holds ``robot_obs`` (15), ``scene_obs`` (24), the
absolute action ``actions`` (TCP position, Euler angles, gripper) and the normalised relative action ``rel_actions``.

Layout ids name reset targets (``EpisodeDriver.reset_episode``):

* ``<split>/<frame>``: reset to ``frame``; the success oracle compares against the state right after that reset.
* ``<split>/<start>..<frame>``: reset to ``frame``; the oracle compares against the state of frame ``start``
  (a window whose subtask is already achieved at the reset, for a still episode whose oracle is true).

``<split>`` is a key of the host config's ``dataset.splits`` (e.g. ``val`` -> ``validation``).
"""
import dataclasses
import os
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

FRAME_FILE = "episode_%07d.npz"
ANNOTATIONS = os.path.join("lang_annotations", "auto_lang_ann.npy")


@dataclasses.dataclass(frozen=True)
class Layout:
    split: str
    frame: int
    oracle_start_frame: int


def parse_layout(layout_id: str, splits: Mapping[str, str]) -> Layout:
    split, sep, rest = layout_id.partition("/")
    if not sep or split not in splits:
        raise ValueError("layout_id %r must be '<split>/<frame>' or '<split>/<start>..<frame>' with split in %s"
                         % (layout_id, sorted(splits)))
    start, dots, end = rest.partition("..")
    try:
        frame = int(end if dots else start)
        oracle_start = int(start)
    except ValueError:
        raise ValueError("layout_id %r: frame numbers must be integers" % (layout_id,)) from None
    if dots and oracle_start > frame:
        raise ValueError("layout_id %r: start frame after the reset frame" % (layout_id,))
    return Layout(split, frame, oracle_start)


class CalvinSplit:
    """One split directory: frames by number, the frame range and the language-annotated windows."""

    def __init__(self, root: str) -> None:
        self.root = root
        ids = np.load(os.path.join(root, "ep_start_end_ids.npy"))
        self.ranges = [(int(a), int(b)) for a, b in ids]
        self._cache = {}  # type: Dict[int, Mapping[str, np.ndarray]]

    def contains(self, frame: int) -> bool:
        return any(a <= frame <= b for a, b in self.ranges)

    def frame(self, frame: int) -> Mapping[str, np.ndarray]:
        if not self.contains(frame):
            raise ValueError("frame %d is outside the recorded ranges %s of %s" % (frame, self.ranges, self.root))
        data = self._cache.get(frame)
        if data is None:
            with np.load(os.path.join(self.root, FRAME_FILE % frame)) as npz:
                data = {k: npz[k] for k in ("robot_obs", "scene_obs", "actions", "rel_actions")}
            self._cache[frame] = data
        return data

    def state(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        data = self.frame(frame)
        return data["robot_obs"].copy(), data["scene_obs"].copy()

    def annotated_windows(self) -> List[Tuple[str, int, int]]:
        """``(task, start, end)`` of every language-annotated window, in file order."""
        ann = np.load(os.path.join(self.root, ANNOTATIONS), allow_pickle=True).item()
        return [(str(task), int(a), int(b)) for task, (a, b) in zip(ann["language"]["task"], ann["info"]["indx"])]


class CalvinDataset:
    """The splits of one dataset directory, by layout key."""

    def __init__(self, root: str, splits: Mapping[str, str]) -> None:
        self.root = root
        self.splits = dict(splits)
        self._open = {}  # type: Dict[str, CalvinSplit]

    def split(self, key: str) -> CalvinSplit:
        if key not in self.splits:
            raise ValueError("unknown split %r (known: %s)" % (key, sorted(self.splits)))
        if key not in self._open:
            self._open[key] = CalvinSplit(os.path.join(self.root, self.splits[key]))
        return self._open[key]

    def layout(self, layout_id: str) -> Layout:
        layout = parse_layout(layout_id, self.splits)
        split = self.split(layout.split)
        for frame in (layout.frame, layout.oracle_start_frame):
            if not split.contains(frame):
                raise ValueError("layout %r: frame %d is not in split %r" % (layout_id, frame, layout.split))
        return layout

    def frame(self, split: str, frame: int) -> Mapping[str, np.ndarray]:
        return self.split(split).frame(frame)

    def state(self, split: str, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.split(split).state(frame)


def layout_id(split: str, frame: int, oracle_start: Optional[int] = None) -> str:
    if oracle_start is None or oracle_start == frame:
        return "%s/%d" % (split, frame)
    return "%s/%d..%d" % (split, oracle_start, frame)
