"""Regression checks for original scene preservation and the policy/video boundary."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from hazardhosts.libero.demos import map_plain_state
from hazardhosts.libero.scene import add_hazard_pool
from hazardhosts.libero.video import Mp4FrameSink
from hazardhosts.libero.seeded import PROTOCOL, RNG_KEY, SeededTransport
from hazardlib.config import load_config


def test_pool_append_preserves_original_task_xml():
    xml = '<mujoco><worldbody><body name="robot"><joint name="hinge"/>' \
          '<geom name="original" type="sphere" size="0.1"/></body></worldbody></mujoco>'
    original = ET.fromstring(xml)
    config = load_config("libero", strict=True)
    augmented = ET.fromstring(add_hazard_pool(xml, config["pool"]))
    assert ET.tostring(augmented.find("worldbody")[0]) == ET.tostring(original.find("worldbody")[0])
    assert len(augmented.find("worldbody")) == 1 + len(config["pool"]["members"])
    assert config["suites"] == ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    with pytest.raises(ValueError, match="already exists"):
        add_hazard_pool(ET.tostring(augmented, encoding="unicode"), config["pool"])


def test_original_state_mapping_does_not_shift_task_dofs():
    source = np.array([1., .2, .3, .4, .5])
    layout = [("a", "hinge", 1, 1), ("b", "slide", 1, 1)]
    target = {"a": (0, 1, 0, 1), "b": (1, 1, 1, 1), "hazard": (2, 7, 2, 6)}
    state, unfilled = map_plain_state(source, layout, target, 1 + 9 + 8)
    np.testing.assert_array_equal(state[:3], source[:3])
    np.testing.assert_array_equal(state[10:12], source[3:])
    assert unfilled == ["hazard"]


def test_mp4_sink_preserves_frame_count_and_simulation_timestamps(tmp_path):
    sink = Mp4FrameSink(tmp_path, fps=20, crf=18, preset="fast")
    episode = "original/" + "long_task_name_" * 30
    sink.open(episode, {"pilot": True})
    for i in range(3):
        sink.add("primary", np.full((32, 48, 3), i * 60, np.uint8), i * .05, "kinematic")
    paths = sink.close()
    result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                             "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", paths["primary"]],
                            check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    assert (stream["width"], stream["height"], stream["nb_read_frames"]) == (48, 32, "3")
    index = json.loads(Path(paths["index"]).read_text())
    assert index["episode_id"] == episode
    assert [f["t_s"] for f in index["cameras"]["primary"]["frames"]] == [0., .05, .1]


def test_video_rejects_invalid_camera_arrays(tmp_path):
    sink = Mp4FrameSink(tmp_path, fps=20, crf=18, preset="fast")
    sink.open("episode", {})
    with pytest.raises(ValueError, match="uint8"):
        sink.add("primary", np.zeros((32, 32, 3)), 0., None)
    sink.close()


def test_paired_openpi_requests_restart_the_same_random_sequence():
    class Transport:
        _codec = SimpleNamespace(server_metadata={"hazardlib_rng_protocol": PROTOCOL})
        def infer(self, payload):
            return payload
        def reset(self, payload):
            pass
    seeded = SeededTransport(Transport())
    with pytest.raises(RuntimeError, match="not reset"):
        seeded.infer({})
    seeded.set_seed(7)
    first = seeded.infer({"prompt": "task"})
    second = seeded.infer({"prompt": "task"})
    assert first[RNG_KEY] == [7, 0] and second[RNG_KEY] == [7, 1]
    seeded.reset(None)
    assert seeded.infer({"prompt": "task"}) == first
    seeded.set_seed(8)
    assert seeded.infer({})[RNG_KEY] == [8, 0]


def test_unseeded_server_is_rejected_before_paired_runs():
    closed = []
    transport = SimpleNamespace(_codec=SimpleNamespace(server_metadata={}), close=lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="seeded server"):
        SeededTransport(transport)
    assert closed == [True]
