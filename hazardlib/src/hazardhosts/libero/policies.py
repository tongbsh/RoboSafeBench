"""LIBERO demonstration reference and bridge to the existing DyniSafety adapters."""
from collections import deque
import importlib

import numpy as np
import yaml

from .demos import parse_layout


class DemoReplay:
    kind = "demo_replay"
    under_test = False

    def __init__(self, host, name="libero_official_demos"):
        self.host, self.name = host, name
        self.actions, self.index = None, 0

    def reset(self, task_id, layout_id, seed):
        layout = parse_layout(layout_id)
        if layout.kind != "demo":
            raise ValueError("DemoReplay requires a demonstration layout")
        self.actions = self.host._demo(task_id, layout.demo_set).actions(layout.episode)[layout.start:]
        self.index = 0

    def act(self, observation):
        if self.index >= len(self.actions):
            return self.host.hold_action(0)
        action = self.actions[self.index].copy()
        self.index += 1
        return action


# Adapters that load the model in-process (which the simulator interpreter cannot do) and the hazardlib client that
# talks to the same model served in its own environment (``serve_oft``): ``module:Class`` -> ``module:Class``.
REMOTE_ADAPTERS = {"dynisafety.policies.adapters.openvla_oft:OpenVLAOFTAdapter":
                   "hazardhosts.libero.oft_client:OftRemoteAdapter"}


class DyniPolicyBridge:
    """ReferencePolicy interface over an existing Policy adapter; one queued action per control step.

    The model's suite-specific checkpoint must be passed explicitly. No conversion
    from original suite names to SafeLIBERO suite names takes place.
    """
    kind = "external_policy"
    under_test = True

    def __init__(self, host, model_yaml, checkpoint, server_host, server_port, openpi_client_src):
        from dynisafety.policies.base import model_config_from_resolved
        with open(model_yaml) as handle:
            raw = yaml.safe_load(handle)
        raw["checkpoint"] = checkpoint
        raw["extra"] = dict(raw.get("extra") or {}, server_host=server_host,
                            server_port=server_port, openpi_client_src=openpi_client_src)
        self.config = model_config_from_resolved(raw, backend="libero")
        if self.config.action_space != "delta_ee_7d":
            raise ValueError("this bridge requires delta_ee_7d; got %s" % self.config.action_space)
        module, cls = REMOTE_ADAPTERS.get(self.config.adapter, self.config.adapter).split(":")
        self.policy = getattr(importlib.import_module(module), cls).from_config(self.config)
        self.seeded_transport = None
        if raw.get("extra", {}).get("ws_codec") == "openpi":
            from .seeded import SeededTransport
            self.seeded_transport = SeededTransport(self.policy._transport)
            self.policy._transport = self.seeded_transport
        self.host, self.name = host, self.config.name
        self.queue, self.index = deque(), 0

    def reset(self, task_id, layout_id, seed):
        from dynisafety.policies.base import EpisodeContext
        suite, name = task_id.split("/", 1)
        self.instruction = self.host.language_instruction(0)
        if self.seeded_transport is not None:
            self.seeded_transport.set_seed(seed)
        self.policy.reset(EpisodeContext(instruction=self.instruction, task_id=self.host._suite_index,
                                        task_name=name, episode_index=parse_layout(layout_id).index,
                                        seed=seed, level=0, backend="libero", suite=suite,
                                        extra={"layout_id": layout_id, "pilot": True}))
        self.queue.clear()
        self.index = 0

    def act(self, observation):
        from dynisafety.policies.base import Observation
        if not self.queue:
            obs = Observation(images={k: observation[k] for k in self.config.observation_images},
                              state=observation["state"], instruction=self.instruction,
                              step=self.index, extra={"ee_proprio_20d": observation["ee_proprio_20d"]})
            chunk = self.policy.act(obs)
            actions = np.asarray(chunk.actions)
            if chunk.space != self.config.action_space or actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError("invalid action chunk: space=%r shape=%s" % (chunk.space, actions.shape))
            if not len(actions) or not np.all(np.isfinite(actions)):
                raise ValueError("empty or nonfinite policy action chunk")
            self.queue.extend(actions[:self.config.chunk_size].copy())
        self.index += 1
        return self.queue.popleft()

    def close(self):
        self.policy.close()
