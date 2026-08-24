"""OpenVLA-OFT local adapter (LIBERO canonical ``openvla_oft``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    ensure_repo_on_path,
    require_extra,
    require_image,
    require_state,
)
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)


class OpenVLAOFTAdapter:
    """In-process OpenVLA-OFT.

    Required ``extra``: ``repo_root``, ``unnorm_key``, ``center_crop``,
    ``use_proprio``, ``load_in_8bit``, ``load_in_4bit``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        ensure_repo_on_path(str(require_extra(cfg, "repo_root")))
        from experiments.robot.openvla_utils import (  # noqa: PLC0415
            get_action_head,
            get_processor,
            get_proprio_projector,
            resize_image_for_policy,
        )
        from experiments.robot.robot_utils import (  # noqa: PLC0415
            get_action,
            get_image_resize_size,
            get_model,
            invert_gripper_action,
            normalize_gripper_action,
        )

        self._resize_image_for_policy = resize_image_for_policy
        self._get_action = get_action
        self._invert = invert_gripper_action
        self._normalize = normalize_gripper_action

        center_crop = bool(require_extra(cfg, "center_crop"))
        use_proprio = bool(require_extra(cfg, "use_proprio"))
        if "image_aug" in cfg.checkpoint and not center_crop:
            raise ValueError("center_crop must be true for image_aug checkpoints")

        unnorm_key = str(require_extra(cfg, "unnorm_key"))
        by_ckpt = cfg.extra.get("unnorm_key_by_checkpoint")
        if isinstance(by_ckpt, dict):
            ckpt_l = str(cfg.checkpoint).lower()
            for needle, key in by_ckpt.items():
                if str(needle).lower() in ckpt_l:
                    unnorm_key = str(key)
                    break

        ns = _Namespace(
            pretrained_checkpoint=cfg.checkpoint,
            # Upstream robot_utils/openvla_utils branch on cfg.model_family and
            # reject anything outside {"openvla", "openvla_oft"} (see
            # experiments/robot/robot_utils.py MODEL_IMAGE_SIZES). Declared in
            # configs/models/*.yaml so the value stays out of Python.
            model_family=str(require_extra(cfg, "model_family")),
            center_crop=center_crop,
            use_proprio=use_proprio,
            unnorm_key=unnorm_key,
            load_in_8bit=bool(require_extra(cfg, "load_in_8bit")),
            load_in_4bit=bool(require_extra(cfg, "load_in_4bit")),
        )
        # Required, not optional: upstream reads these unconditionally, so a
        # missing key surfaces as an obscure AttributeError deep in get_vla().
        for key in (
            "use_l1_regression",
            "use_diffusion",
            "use_film",
            "num_images_in_input",
        ):
            setattr(ns, key, require_extra(cfg, key))

        self._ns = ns
        self._model = get_model(ns)
        # Upstream signatures: get_processor(cfg),
        # get_action_head(cfg, llm_dim), get_proprio_projector(cfg, llm_dim, proprio_dim).
        # They take the LLM width, not the model object.
        self._processor = get_processor(ns)
        self._action_head = (
            get_action_head(ns, self._model.llm_dim)
            if (ns.use_l1_regression or ns.use_diffusion)
            else None
        )
        self._proprio_projector = (
            get_proprio_projector(
                ns, self._model.llm_dim, int(require_extra(cfg, "proprio_dim"))
            )
            if use_proprio
            else None
        )
        self._resize = get_image_resize_size(ns)
        self._resolve_unnorm_key()

    def _resolve_unnorm_key(self) -> None:
        """Reconcile the configured ``unnorm_key`` with the checkpoint's stats.

        Ported from the pre-refactor ``OpenVLAPolicy._resolve_unnorm_key``:
        fine-tuned checkpoints publish suite keys with a ``_no_noops`` suffix
        (e.g. ``libero_spatial_no_noops``), so an exact-name lookup alone fails.
        """
        stats = getattr(self._model, "norm_stats", None) or {}
        want = str(self._ns.unnorm_key)
        for candidate in (want, f"{want}_no_noops"):
            if candidate in stats:
                self._ns.unnorm_key = candidate
                return
        # Single-key checkpoints (the common fine-tuned case) are unambiguous.
        if len(stats) == 1:
            only = next(iter(stats))
            self._ns.unnorm_key = only
            return
        raise ValueError(
            f"unnorm_key {want!r} (and {want + '_no_noops'!r}) not found in "
            f"checkpoint norm_stats. Available: {sorted(stats)}"
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> OpenVLAOFTAdapter:
        if cfg.transport != "local":
            raise ValueError("OpenVLAOFTAdapter requires transport='local'")
        return cls(cfg)

    def reset(self, ctx: EpisodeContext) -> None:
        del ctx

    def act(self, obs: Observation) -> ActionChunk:
        primary = self._resize_image_for_policy(
            require_image(obs, "primary"), self._resize
        )
        observation: dict[str, Any] = {
            "full_image": primary,
            "task_description": obs.instruction,
        }
        if "wrist" in obs.images:
            observation["wrist_image"] = self._resize_image_for_policy(
                require_image(obs, "wrist"), self._resize
            )
        if self._ns.use_proprio:
            observation["state"] = require_state(obs)

        # Upstream: get_action(cfg, model, obs, task_label, processor=None,
        # action_head=None, proprio_projector=None, noisy_action_projector=None,
        # use_film=False). The 4th positional is the task label, so processor and
        # friends must be passed by keyword.
        actions = self._get_action(
            self._ns,
            self._model,
            observation,
            obs.instruction,
            processor=self._processor,
            action_head=self._action_head,
            proprio_projector=self._proprio_projector,
            use_film=bool(self._ns.use_film),
        )
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = []
        for row in arr:
            a = self._normalize(row, binarize=True)
            a = self._invert(a)
            out.append(a.astype(np.float32))
        return actions_to_chunk(np.stack(out, axis=0), space=self._cfg.action_space)

    def close(self) -> None:
        self._model = None


class _Namespace:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
