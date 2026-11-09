"""The CALVIN environment with hooks around every physics step (H1).

``HookedPlayTableEnv`` subclasses ``calvin_env.envs.play_table_env.PlayTableSimEnv`` and replaces ``reset`` and
``step`` with copies of the upstream lines (``play_table_env.py:170-174`` and ``:222-243`` at ``calvin_env@797142c``)
that count the physics steps of the simulation clock and call a listener before and after each of the
``action_repeat`` (8) ``stepSimulation`` calls of one control step. Nothing else changes: the robot, the scene, the
cameras and the observation / info dictionaries are the upstream ones. Because lines are copied, the upstream
commit is pinned and checked when this module is imported (R13).

``make_env`` is ``play_table_env.get_env`` with the hooked class: it loads the dataset's
``.hydra/merged_config.yaml`` and keeps only the requested cameras.
"""
import os
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

import calvin_env
import hydra
from calvin_env.envs.play_table_env import PlayTableSimEnv
from omegaconf import OmegaConf

CALVIN_ENV_COMMIT = "797142c588c21e76717268b7b430958dbd13bf48"
COPIED_FROM = ("calvin_env/envs/play_table_env.py",)
EGL_PLUGIN_ENV = "EGL_VISIBLE_DEVICES"  # PyBullet eglRendererPlugin
EGL_PYRENDER_ENV = "EGL_DEVICE_ID"  # pyrender (tacto)


class UpstreamMismatchError(RuntimeError):
    """The installed calvin_env is not the commit the copied lines come from."""


def calvin_env_root() -> Path:
    return Path(calvin_env.__file__).resolve().parents[1]


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(root)] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         check=True)
    return out.stdout.decode().strip()


def check_pinned_commit() -> str:
    """The calvin_env checkout must be ``CALVIN_ENV_COMMIT`` with the copied file unmodified."""
    root = calvin_env_root()
    try:
        head = _git(root, "rev-parse", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--", *COPIED_FROM)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise UpstreamMismatchError("cannot read the git state of %s: %s" % (root, exc)) from None
    if head != CALVIN_ENV_COMMIT:
        raise UpstreamMismatchError("calvin_env is at %s, the hooked step/reset copy %s" % (head, CALVIN_ENV_COMMIT))
    if dirty:
        raise UpstreamMismatchError("calvin_env has local changes in the copied file(s): %s" % dirty)
    return head


check_pinned_commit()


def pin_egl_device(egl_index: int) -> None:
    """Render on EGL device ``egl_index`` (PyBullet plugin and pyrender); must run before the env is built.
    The index is the position in ``eglQueryDevicesEXT`` (see ``probes/egl_devices.py``), not a CUDA index."""
    value = str(int(egl_index))
    for name in (EGL_PLUGIN_ENV, EGL_PYRENDER_ENV):
        current = os.environ.get(name)
        if current is not None and current != value:
            raise RuntimeError("%s is already %r, refusing to switch it to %r" % (name, current, value))
        os.environ[name] = value


class HookedPlayTableEnv(PlayTableSimEnv):
    """``PlayTableSimEnv`` with a physics-step clock and a step listener (see the module docstring).

    ``listener`` (or ``None``) provides ``before_physics_step(env)`` and ``after_physics_step(env)``;
    ``physics_steps`` counts every ``stepSimulation`` of ``reset`` and ``step``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.physics_steps = 0
        self.listener = None  # type: Any
        super().__init__(*args, **kwargs)

    def reset(self, robot_obs: Any = None, scene_obs: Any = None) -> Any:
        self.scene.reset(scene_obs)
        self.robot.reset(robot_obs)
        self.p.stepSimulation(physicsClientId=self.cid)
        self.physics_steps += 1
        return self.get_obs()

    def step(self, action: Any) -> Any:
        if self.use_vr:
            raise RuntimeError("the hooked CALVIN env does not support VR mode (real-time simulation)")
        self.robot.apply_action(action)
        for _ in range(self.action_repeat):
            if self.listener is not None:
                self.listener.before_physics_step(self)
            self.p.stepSimulation(physicsClientId=self.cid)
            self.physics_steps += 1
            if self.listener is not None:
                self.listener.after_physics_step(self)
        self.scene.step()
        obs = self.get_obs()
        info = self.get_info()
        return obs, 0, False, info


def make_env(dataset_split_dir: str, cameras: Optional[Sequence[str]] = None) -> HookedPlayTableEnv:
    """``play_table_env.get_env(dataset_split_dir, show_gui=False)`` with :class:`HookedPlayTableEnv`;
    ``cameras`` keeps only those camera names (``None``: all cameras of the dataset config)."""
    render_conf = OmegaConf.load(Path(dataset_split_dir) / ".hydra" / "merged_config.yaml")
    if cameras is not None:
        unknown = sorted(set(cameras) - set(render_conf.cameras.keys()))
        if unknown:
            raise ValueError("unknown camera(s) %s (dataset has %s)" % (unknown, sorted(render_conf.cameras.keys())))
        for name in set(render_conf.cameras.keys()) - set(cameras):
            del render_conf.cameras[name]
    render_conf.env["_target_"] = "%s.%s" % (__name__, HookedPlayTableEnv.__name__)
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(render_conf.env, show_gui=False, use_vr=False, use_scene_info=True)
