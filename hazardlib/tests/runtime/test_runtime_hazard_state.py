"""The privileged accessor ``HazardRuntime.hazard_state(env)`` (D13 d) and one live runtime per host (01 §3.3
item 16, D13 c)."""
import numpy as np
import pytest

import runtime_world as W
from hazardlib.conformance import FakeAction, FakeHost
from hazardlib.runtime import KINEMATIC, PARKED, RELEASED, HazardRuntime

STATE_FIELDS = ("pos", "quat", "lin_vel_mps", "ang_vel_radps")


class RecordingHost(FakeHost):
    """Remembers every allocated handle, so the test reads the host's truth without the runtime."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.allocated = {}

    def alloc_hazard(self, env, agent):
        handle = super().alloc_hazard(env, agent)
        self.allocated.setdefault(env, []).append(handle)
        return handle


def make():
    host = RecordingHost(W.world_config())
    return host, HazardRuntime(host, W.runtime_config())


def same_state(a, b):
    return all(np.array_equal(getattr(a, name), getattr(b, name)) for name in STATE_FIELDS)


def truth(host, env):
    return host.hazard_state(host.allocated[env][-1])


def test_no_hazard_before_the_episode_begins():
    host, runtime = make()
    assert runtime.hazard_state(0) is None and runtime.hazard_handle(0) is None
    runtime.arm(0, W.slide_instance(host))
    assert runtime.hazard_state(0) is None  # armed, not begun: nothing allocated yet
    host.reset_episode(0, "static", "empty", 0)
    assert runtime.hazard_handle(0) == host.allocated[0][-1]
    runtime.end_episode(0)


def test_parked_hazard_reads_the_host_truth_at_the_park_pose():
    host, runtime = make()
    inst = W.slide_instance(host)  # placed at the onset, 0.2 s
    runtime.arm(0, inst)
    host.reset_episode(0, "static", "empty", 0)
    assert runtime.episode_info(0)["phase"] == PARKED
    state = runtime.hazard_state(0)
    assert same_state(state, truth(host, 0))
    assert np.array_equal(state.pos, np.array(W.PARK)) and not np.any(state.lin_vel_mps)
    runtime.end_episode(0)


@pytest.mark.parametrize("make_instance, phases_seen", [
    (W.slide_instance, {PARKED, KINEMATIC, RELEASED}),
    (W.fall_instance, {PARKED, RELEASED}),
], ids=["slide", "fall"])
def test_every_step_reads_the_host_truth_and_none_after_the_episode(make_instance, phases_seen):
    host, runtime = make()
    runtime.arm(0, make_instance(host))
    host.reset_episode(0, "static", "empty", 0)
    phases = set()
    for _ in range(50):
        host.step(0, FakeAction.follow())
        phases.add(runtime.episode_info(0)["phase"])
        assert same_state(runtime.hazard_state(0), truth(host, 0))
    assert phases == phases_seen
    handle = host.allocated[0][-1]
    runtime.end_episode(0)
    assert runtime.hazard_state(0) is None and runtime.hazard_handle(0) is None
    with pytest.raises(ValueError, match="not allocated"):
        host.hazard_state(handle)  # freed by end_episode: there is nothing left to read


def test_control_episode_has_no_hazard_state():
    host, runtime = make()
    runtime.arm(0, None, control_id="ctl")
    host.reset_episode(0, "static", "empty", 0)
    for _ in range(10):
        host.step(0, FakeAction.follow())
        assert runtime.hazard_state(0) is None
    runtime.end_episode(0)
    assert runtime.hazard_state(0) is None and 0 not in host.allocated


def test_environments_have_their_own_hazard_state():
    host, runtime = make()
    runtime.arm(0, W.slide_instance(host))
    runtime.arm(1, W.fall_onto(W.table_height_centroid(0.1)))
    host.reset_episode(0, "static", "empty", 0)
    host.reset_episode(1, "static", "empty", 1)
    assert runtime.hazard_handle(0) != runtime.hazard_handle(1)
    differed = False
    for _ in range(40):
        for env in (0, 1):
            host.step(env, FakeAction.follow())
        for env in (0, 1):
            assert same_state(runtime.hazard_state(env), truth(host, env))
        differed = differed or not np.array_equal(runtime.hazard_state(0).pos, runtime.hazard_state(1).pos)
    assert differed
    runtime.end_episode(1)
    assert runtime.hazard_state(1) is None
    assert same_state(runtime.hazard_state(0), truth(host, 0))  # env 0 still running
    runtime.end_episode(0)
    runtime.arm(1, None, control_id="ctl")
    host.reset_episode(1, "static", "empty", 2)
    runtime.arm(0, W.slide_instance(host))
    host.reset_episode(0, "static", "empty", 3)
    assert runtime.hazard_state(1) is None and same_state(runtime.hazard_state(0), truth(host, 0))
    runtime.end_episode(0)
    runtime.end_episode(1)


def test_a_later_runtime_takes_the_hooks_over():
    """No guard against a second runtime (the conformance suite relies on taking the hooks over); the earlier
    runtime is void: the host's reset hook now reaches the later one, which was not armed."""
    host, first = make()
    second = HazardRuntime(host, W.runtime_config())
    first.arm(0, W.slide_instance(host))
    with pytest.raises(RuntimeError, match="without arm"):
        host.reset_episode(0, "static", "empty", 0)
    assert first.hazard_state(0) is None
    second.arm(0, W.slide_instance(host))
    host.reset_episode(0, "static", "empty", 0)
    assert same_state(second.hazard_state(0), truth(host, 0))
    second.end_episode(0)
