import numpy as np
import pytest

from hazardlib.kinematics import (arc_path, free_fall_path, hook_times, needs_substeps, reach_path,
                                  step_displacements, straight_path, subdivide)

PHYSICS_DT = 1.0 / 240.0


def max_step(path, times):
    return float(np.max(step_displacements(path, times)))


def test_hook_times_grid():
    grid = hook_times(1.0, 2.0, 0.25)
    np.testing.assert_allclose(grid, [1.0, 1.25, 1.5, 1.75, 2.0])
    grid = hook_times(0.0, 1.0, 0.3)           # last interval is shorter
    np.testing.assert_allclose(grid, [0.0, 0.3, 0.6, 0.9, 1.0])
    assert grid[0] == 0.0 and grid[-1] == 1.0
    assert hook_times(0.0, 0.1, 5.0).tolist() == [0.0, 0.1]
    for bad in ((1.0, 1.0, 0.1), (2.0, 1.0, 0.1), (0.0, 1.0, 0.0), (0.0, 1.0, -1.0),
                (0.0, 1.0, float("nan"))):
        with pytest.raises(ValueError):
            hook_times(*bad)


def test_fast_straight_path_is_subdivided_to_the_step_limit():
    path = straight_path([0, 0, 0], [5, 0, 0], 0.0, 1.0)             # 5 m/s
    hook_dt, limit = 1.0 / 30.0, 0.01
    assert needs_substeps(path, limit, hook_dt)
    grid = hook_times(path.t_start_s, path.t_end_s, hook_dt)
    assert max_step(path, grid) > limit                              # the plain grid is too coarse
    times = subdivide(path, limit, hook_dt)
    assert np.all(np.diff(times) > 0)
    assert times[0] == path.t_start_s and times[-1] == path.t_end_s
    assert max_step(path, times) <= limit                            # the acceptance criterion
    assert set(grid.tolist()) <= set(times.tolist())                 # every hook instant is kept


def test_uniform_path_gets_the_minimal_number_of_steps():
    path = straight_path([0, 0, 0], [0.095, 0, 0], 0.0, 1.0)           # one hook interval, 9.5 cm
    times = subdivide(path, 0.01, 1.0)
    assert len(times) == 11                                           # ceil(9.5) = 10 steps
    np.testing.assert_allclose(np.diff(times), 0.1, rtol=1e-12)


def test_slow_path_is_left_alone():
    path = straight_path([0, 0, 0], [0.01, 0, 0], 0.0, 1.0)
    assert not needs_substeps(path, 0.05, PHYSICS_DT)
    np.testing.assert_array_equal(subdivide(path, 0.05, PHYSICS_DT),
                                  hook_times(0.0, 1.0, PHYSICS_DT))


def test_stationary_path_needs_no_substeps():
    path = straight_path([0.3, 0.3, 0.3], [0.3, 0.3, 0.3], 0.0, 1.0)
    assert not needs_substeps(path, 1e-9, 0.1)
    assert len(subdivide(path, 1e-9, 0.1)) == len(hook_times(0.0, 1.0, 0.1))


@pytest.mark.parametrize("make", [
    lambda: arc_path([0, 0, 0], [3, 0, 0], 0.0, 0.5, 0.4),
    lambda: reach_path([0, 0, 0], [2, 1, 0], 0.0, 0.6, "min_jerk"),
    lambda: free_fall_path([0, 0, 3.0], 0.0, 0.7, 9.81),
    lambda: straight_path([0, 0, 0], [-4, 2, 1], 2.0, 2.5),
])
def test_curved_and_accelerating_paths_meet_the_limit(make):
    path = make()
    for hook_dt, limit in ((1.0 / 30.0, 0.02), (PHYSICS_DT, 0.005), (0.1, 0.05)):
        times = subdivide(path, limit, hook_dt)
        assert np.all(np.diff(times) > 0)
        assert times[0] == path.t_start_s and times[-1] == path.t_end_s
        assert max_step(path, times) <= limit
        grid = hook_times(path.t_start_s, path.t_end_s, hook_dt)
        assert set(grid.tolist()) <= set(times.tolist())


def test_needs_substeps_is_consistent_with_subdivide():
    rng = np.random.RandomState(5)
    for _ in range(30):
        path = straight_path(rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3), 0.0, rng.uniform(0.2, 2.0))
        limit, hook_dt = rng.uniform(0.005, 0.2), rng.uniform(0.005, 0.2)
        grid = hook_times(path.t_start_s, path.t_end_s, hook_dt)
        assert needs_substeps(path, limit, hook_dt) == (len(subdivide(path, limit, hook_dt)) > len(grid))


def test_step_displacements_helper():
    path = straight_path([0, 0, 0], [1, 0, 0], 0.0, 1.0)
    np.testing.assert_allclose(step_displacements(path, [0.0, 0.25, 1.0]), [0.25, 0.75])
    assert step_displacements(path, [0.5]).shape == (0,)


@pytest.mark.parametrize("args", [(0.0, 0.1), (-1.0, 0.1), (float("nan"), 0.1), (0.01, 0.0),
                                  (0.01, -0.1)])
def test_invalid_limits_are_rejected(args):
    path = straight_path([0, 0, 0], [1, 0, 0], 0.0, 1.0)
    with pytest.raises(ValueError):
        subdivide(path, *args)
    with pytest.raises(ValueError):
        needs_substeps(path, *args)
