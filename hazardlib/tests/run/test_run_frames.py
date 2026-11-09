"""What the policy saw (R1 item 4): ``FrameRecorder``, the core frame sinks, ``TimedPolicy``."""
import json

import numpy as np
import pytest

import run_world as RW
from run_world import run_world_chain  # noqa: F401 (fixture)
from hazardlib.run.frames import FrameRecorder, MemoryFrameSink, NpyFrameSink, TimedPolicy, make_npy_sink


class Recording(RW.Nominal):
    """Keeps a copy of every observation image it acted on."""

    def reset(self, task_id, layout_id, seed):
        super().reset(task_id, layout_id, seed)
        self.seen = []

    def act(self, observation):
        self.seen.append(np.array(observation["rgb"], copy=True))
        return super().act(observation)


def _episode(world, sink, instance, control_onset_s=None):
    host = RW.make_host()
    session = world.session(host)
    policy = Recording(host)
    recorder = FrameRecorder(TimedPolicy(policy), sink, ["rgb"], session, 0)
    recorder.begin("ep", {"note": "test"})
    result = session.run_episode(0, instance, recorder, RW.TASK, RW.LAYOUT, 0, 1.2,
                                 control_id=None if instance is not None else "control:x",
                                 control_onset_s=control_onset_s)
    return recorder, policy, result, recorder.end()


def test_frames_are_what_the_policy_saw_one_per_control_step(run_world_chain):
    instance = [i for i in run_world_chain.instances if i.spec_id == "fall_wait"][0]
    sink = MemoryFrameSink()
    recorder, policy, result, paths = _episode(run_world_chain, sink, instance)
    assert paths == {}
    episode = sink.episodes["ep"]
    frames, times, phases = episode["frames"]["rgb"], episode["t_s"]["rgb"], episode["phase"]["rgb"]
    n_steps = len(result.times_s) - 1
    assert recorder.n_frames == len(frames) == len(times) == len(phases) == n_steps == 60
    assert all(np.array_equal(f, s) for f, s in zip(frames, policy.seen))
    assert times[0] == 0.0 and np.all(np.diff(times) > 0)
    assert np.allclose(np.diff(times), RW.W.CONTROL_DT)
    assert phases[0] == "parked" and phases[-1] == "released" and set(phases) == {"parked", "released"}
    first = phases.index("released")   # the first control sample after the onset (0.45 s, released at a hook)
    assert times[first - 1] < instance.onset_time_s < times[first] == pytest.approx(0.46, abs=1e-9)
    assert episode["meta"] == {"note": "test"}


def test_control_frames_have_no_hazard_phase(run_world_chain):
    sink = MemoryFrameSink()
    recorder, _, _, _ = _episode(run_world_chain, sink, None)
    assert set(sink.episodes["ep"]["phase"]["rgb"]) == {None} and recorder.n_frames == 60


def test_npy_sink_writes_stacked_frames_and_an_index(run_world_chain, tmp_path):
    instance = run_world_chain.instances[0]
    sink = make_npy_sink(tmp_path / "frames")
    assert isinstance(sink, NpyFrameSink)
    recorder, policy, _, paths = _episode(run_world_chain, sink, instance)
    assert sorted(paths) == ["index", "rgb"]
    stack = np.load(paths["rgb"])
    assert stack.shape == (60, 32, 32, 3) and stack.dtype == np.uint8
    assert np.array_equal(stack, np.stack(policy.seen))
    index = json.loads((tmp_path / "frames" / "ep" / "frames.json").read_text())
    assert index["keys"]["rgb"]["n_frames"] == 60 and index["keys"]["rgb"]["file"] == "rgb.npy"
    assert index["keys"]["rgb"]["t_s"][1] == pytest.approx(RW.W.CONTROL_DT)


def test_timed_policy_and_recorder_misuse(run_world_chain):
    host = RW.make_host()
    session = run_world_chain.session(host)
    timed = TimedPolicy(RW.Nominal(host))
    assert (timed.kind, timed.name, timed.under_test) == ("external_policy", "nominal_bot", True)
    timed.reset(RW.TASK, RW.LAYOUT, 0)
    assert timed.summary()["n_steps"] == 0 and timed.summary()["mean_s"] is None
    with pytest.raises(TypeError, match="not a FrameSink"):
        FrameRecorder(timed, object(), ["rgb"], session, 0)
    with pytest.raises(ValueError, match="distinct"):
        FrameRecorder(timed, MemoryFrameSink(), ["rgb", "rgb"], session, 0)
    recorder = FrameRecorder(timed, MemoryFrameSink(), ["depth"], session, 0)
    with pytest.raises(RuntimeError, match="outside begin"):
        recorder.act({"rgb": None})
    recorder.begin("e", {})
    with pytest.raises(RuntimeError, match="while episode"):
        recorder.sink.open("f", {})
    session.runtime.arm(0, None, control_id="control:x")
    host.reset_episode(0, RW.TASK, RW.LAYOUT, 0)
    with pytest.raises(KeyError, match="'depth' is not in the policy observation"):
        recorder.act(host.policy_observation(0))
