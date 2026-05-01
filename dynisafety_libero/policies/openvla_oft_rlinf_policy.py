"""Standalone RLinf OpenVLA-OFT policy for DyniSafety-LIBERO.

This adapter is intentionally separate from the existing ``openvla`` /
``openvla_oft`` policy path. It talks directly to RLinf's OpenVLA-OFT
inference stack and keeps all RLinf-specific loading and output conversion in
one place.

Supported checkpoint inputs:

1. A Hugging Face / converted checkpoint directory.
2. A raw ``full_weights.pt`` file saved by RLinf training.
3. A RLinf checkpoint directory that contains
   ``model_state_dict/full_weights.pt`` (or the older ``full_weigths.pt``).
4. A LoRA adapter directory (``adapter_config.json`` or ``lora_adapter/``)
   together with ``base_model_path``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, List

import numpy as np

from .base_policy import VLAPolicy

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RLINF_ROOT = _WORKSPACE_ROOT / "RLinf_dynamicEval"
_DEFAULT_OPENVLA_ROOT = _WORKSPACE_ROOT / "vlaModelHome" / "openvla-oft"

_SAFELIBERO_TO_LIBERO_KEY = {
    "safelibero_spatial": "libero_spatial",
    "safelibero_object": "libero_object",
    "safelibero_goal": "libero_goal",
    "safelibero_long": "libero_10",
}

_HF_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def _normalize_gripper_action(action: np.ndarray, *, binarize: bool = True) -> np.ndarray:
    """Match DyniSafety OpenVLA gripper post-processing."""
    normalized = action.copy()
    normalized[..., -1] = 2.0 * normalized[..., -1] - 1.0
    if binarize:
        normalized[..., -1] = np.sign(normalized[..., -1])
    return normalized


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Match LIBERO env convention (-1=open, +1=close)."""
    inverted = action.copy()
    inverted[..., -1] *= -1.0
    return inverted


class OpenVLAOFTRLinfPolicy(VLAPolicy):
    """RLinf-specific OpenVLA-OFT policy adapter."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._warned_terminate_channel = False

        if getattr(cfg, "load_in_8bit", False) or getattr(cfg, "load_in_4bit", False):
            raise ValueError(
                "openvla_oft_rlinf does not support load_in_8bit/load_in_4bit. "
                "Keep both flags false."
            )

        self.rlinf_root = self._resolve_repo_root(
            cfg_value=getattr(cfg, "rlinf_root", ""),
            env_key="RLINF_ROOT",
            default_path=_DEFAULT_RLINF_ROOT,
            repo_name="RLinf_dynamicEval",
        )
        self.openvla_root = self._resolve_repo_root(
            cfg_value=getattr(cfg, "openvla_root", ""),
            env_key="OPENVLA_ROOT",
            default_path=_DEFAULT_OPENVLA_ROOT,
            repo_name="openvla-oft",
        )
        self._ensure_import_paths()

        checkpoint = os.fspath(getattr(cfg, "pretrained_checkpoint", ""))
        if not checkpoint:
            raise ValueError("pretrained_checkpoint must not be empty.")

        (
            self.resolved_model_path,
            self.resolved_state_dict_path,
            self.resolved_adapter_path,
            self._checkpoint_root,
        ) = self._resolve_model_source(checkpoint)

        self.model = self._build_model()
        self.model.eval()

        self._action_dim = int(
            getattr(cfg, "action_dim", getattr(self.model, "action_dim", 7))
        )
        self._num_action_chunks = int(
            getattr(cfg, "num_action_chunks", getattr(self.model, "num_action_chunks", 1))
        )
        self._resolve_unnorm_key()

    @property
    def action_chunk_size(self) -> int:
        return self._num_action_chunks

    def reset(self) -> None:
        """Benchmark hook for episode boundaries."""
        self._warned_terminate_channel = False

    def predict_actions(self, obs: dict, task_description: str) -> List[np.ndarray]:
        env_obs = self._build_env_obs(obs, task_description)

        predict_kwargs = {
            "env_obs": env_obs,
            "do_sample": bool(getattr(self.cfg, "do_sample", False)),
            "temperature": float(getattr(self.cfg, "temperature", 1.0)),
            "top_k": int(getattr(self.cfg, "top_k", 50)),
        }
        if self._implement_version() == "official":
            predict_kwargs["calulate_logprobs"] = False
            predict_kwargs["calulate_values"] = False
        else:
            predict_kwargs["calculate_logprobs"] = False
            predict_kwargs["calculate_values"] = False

        chunk_actions, _ = self.model.predict_action_batch(**predict_kwargs)
        chunk_array = self._to_numpy(chunk_actions)
        if chunk_array.ndim == 3:
            chunk_array = chunk_array[0]
        if chunk_array.ndim != 2:
            raise ValueError(
                f"Expected chunk actions with shape [T, D] or [1, T, D]; got {chunk_array.shape}"
            )
        return self._chunk_to_env_actions(chunk_array)

    def _resolve_repo_root(
        self,
        *,
        cfg_value: str,
        env_key: str,
        default_path: Path,
        repo_name: str,
    ) -> Path:
        raw_value = os.environ.get(env_key) or str(cfg_value or default_path)
        path = Path(raw_value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"{repo_name} root not found at '{path}'. "
                f"Set `{env_key}` or pass `--{env_key.lower()}`."
            )
        return path

    def _ensure_import_paths(self) -> None:
        for path in (self.rlinf_root, self.openvla_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    def _resolve_model_source(
        self,
        checkpoint: str,
    ) -> tuple[str, str | None, str | None, Path | None]:
        local_path = Path(checkpoint).expanduser()
        if not local_path.exists():
            # Allow HF repo IDs or other remote strings to pass through unchanged.
            return checkpoint, None, None, None

        if local_path.is_file():
            return (
                self._require_base_model_path(),
                str(local_path.resolve()),
                None,
                local_path.parent.resolve(),
            )

        adapter_path = self._find_adapter_dir(local_path)
        if adapter_path is not None and not self._has_hf_weights(local_path):
            return (
                self._require_base_model_path(),
                None,
                str(adapter_path),
                local_path.resolve(),
            )

        state_dict_path = self._find_state_dict_path(local_path)
        if state_dict_path is not None:
            return (
                self._require_base_model_path(),
                str(state_dict_path),
                None,
                local_path.resolve(),
            )

        if self._has_local_shard_checkpoint(local_path):
            raise ValueError(
                f"Checkpoint '{local_path}' only contains FSDP shard files. "
                "Export or convert it to `full_weights.pt` (or HF format) before evaluation."
            )

        return str(local_path.resolve()), None, None, local_path.resolve()

    def _require_base_model_path(self) -> str:
        base_model_path = os.fspath(getattr(self.cfg, "base_model_path", "") or "")
        if not base_model_path:
            raise ValueError(
                "Raw RLinf checkpoints require `base_model_path`. Pass the base OpenVLA-OFT "
                "HF directory (or repo id) via `--base_model_path`."
            )

        local_path = Path(base_model_path).expanduser()
        return str(local_path.resolve()) if local_path.exists() else base_model_path

    def _find_adapter_dir(self, root: Path) -> Path | None:
        direct = root / "adapter_config.json"
        if direct.exists():
            return root.resolve()

        nested = root / "lora_adapter" / "adapter_config.json"
        if nested.exists():
            return (root / "lora_adapter").resolve()
        return None

    def _find_state_dict_path(self, root: Path) -> Path | None:
        candidates = (
            root / "full_weights.pt",
            root / "full_weigths.pt",
            root / "model_state_dict" / "full_weights.pt",
            root / "model_state_dict" / "full_weigths.pt",
            root / "actor" / "model_state_dict" / "full_weights.pt",
            root / "actor" / "model_state_dict" / "full_weigths.pt",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _has_hf_weights(self, root: Path) -> bool:
        return any((root / name).exists() for name in _HF_WEIGHT_FILES)

    def _has_local_shard_checkpoint(self, root: Path) -> bool:
        return (root / "local_shard_checkpoint").exists() or (
            root / "actor" / "local_shard_checkpoint"
        ).exists()

    def _build_model(self):
        from omegaconf import OmegaConf
        from rlinf.models import get_model as get_rlinf_model

        config_dict: dict[str, Any] = {
            "model_type": "openvla_oft",
            "implement_version": self._implement_version(),
            "model_path": self.resolved_model_path,
            "precision": getattr(self.cfg, "precision", "bf16"),
            "value_type": getattr(self.cfg, "value_type", "action_level"),
            "action_dim": int(getattr(self.cfg, "action_dim", 7)),
            "num_action_chunks": int(getattr(self.cfg, "num_action_chunks", 8)),
            "use_proprio": bool(getattr(self.cfg, "use_proprio", False)),
            "use_film": bool(getattr(self.cfg, "use_film", False)),
            "unnorm_key": str(getattr(self.cfg, "unnorm_key", "")),
            "center_crop": bool(getattr(self.cfg, "center_crop", True)),
            "max_prompt_length": int(getattr(self.cfg, "max_prompt_length", 128)),
            "hidden_size": int(getattr(self.cfg, "hidden_size", 4096)),
            "add_value_head": bool(getattr(self.cfg, "add_value_head", False)),
            "is_lora": bool(getattr(self.cfg, "is_lora", False)),
            "lora_rank": int(getattr(self.cfg, "lora_rank", 32)),
            "lora_path": self.resolved_adapter_path or (getattr(self.cfg, "lora_path", None) or None),
            "num_images_in_input": int(getattr(self.cfg, "num_images_in_input", 1)),
            "attn_implementation": getattr(self.cfg, "attn_implementation", "flash_attention_2"),
            "low_cpu_mem_usage": bool(getattr(self.cfg, "low_cpu_mem_usage", True)),
            "trust_remote_code": bool(getattr(self.cfg, "trust_remote_code", True)),
            "proprio_dim": int(getattr(self.cfg, "proprio_dim", 8)),
        }

        norm_stats = self._load_norm_stats_override()
        if norm_stats is not None:
            config_dict["norm_stats"] = norm_stats

        if self.resolved_adapter_path and not config_dict["is_lora"]:
            config_dict["is_lora"] = True

        model_cfg = OmegaConf.create(config_dict)
        model = get_rlinf_model(model_cfg)

        if self.resolved_state_dict_path is not None:
            self._load_extra_state_dict(model, self.resolved_state_dict_path, expect_lora=config_dict["is_lora"])

        return model

    def _load_norm_stats_override(self) -> dict[str, Any] | None:
        if self._checkpoint_root is None:
            return None

        candidate_dirs = []
        for path in (
            self._checkpoint_root,
            self._checkpoint_root / "actor",
            self._checkpoint_root / "model_state_dict",
        ):
            candidate_dirs.append(path)

        if self.resolved_state_dict_path is not None:
            state_path = Path(self.resolved_state_dict_path)
            candidate_dirs.extend(
                [
                    state_path.parent,
                    state_path.parent.parent,
                    state_path.parent.parent.parent,
                ]
            )

        seen = set()
        for directory in candidate_dirs:
            directory = directory.resolve()
            if directory in seen or not directory.exists():
                continue
            seen.add(directory)

            stats_path = directory / "dataset_statistics.json"
            if stats_path.exists():
                with open(stats_path, "r", encoding="utf-8") as f:
                    logger.info("Using dataset statistics from %s", stats_path)
                    return json.load(f)
        return None

    def _load_extra_state_dict(
        self,
        model,
        state_dict_path: str,
        *,
        expect_lora: bool,
    ) -> None:
        import torch

        try:
            state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(state_dict_path, map_location="cpu")

        if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        normalized_state_dict = {}
        for key, value in state_dict.items():
            if not hasattr(value, "shape"):
                continue
            normalized_key = key[7:] if key.startswith("module.") else key
            normalized_state_dict[normalized_key] = value

        has_lora_weights = any("lora_" in key for key in normalized_state_dict)
        if has_lora_weights and not expect_lora:
            raise ValueError(
                "The RLinf checkpoint contains LoRA weights, but `is_lora=False`. "
                "Pass `--is_lora True` or export merged HF weights first."
            )

        model_state = model.state_dict()
        filtered_state_dict = {}
        skipped_shape = []
        skipped_special = 0
        skipped_unknown = 0
        for key, value in normalized_state_dict.items():
            if "value_head" in key or "q_head" in key:
                skipped_special += 1
                continue
            if key not in model_state:
                skipped_unknown += 1
                continue
            if tuple(model_state[key].shape) != tuple(value.shape):
                skipped_shape.append(key)
                continue
            filtered_state_dict[key] = value

        if not filtered_state_dict:
            raise ValueError(
                f"No compatible inference weights were found in '{state_dict_path}'. "
                "Check `base_model_path`, `implement_version`, and `is_lora`."
            )

        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        logger.info(
            "Loaded RLinf weights from %s (%d tensors, skipped_special=%d, skipped_unknown=%d, "
            "skipped_shape=%d, missing=%d, unexpected=%d)",
            state_dict_path,
            len(filtered_state_dict),
            skipped_special,
            skipped_unknown,
            len(skipped_shape),
            len(missing),
            len(unexpected),
        )

    def _resolve_unnorm_key(self) -> None:
        norm_stats = getattr(self.model, "norm_stats", None)
        if not isinstance(norm_stats, dict) or not norm_stats:
            raise ValueError("Model does not expose valid norm_stats; cannot resolve unnorm_key.")

        explicit_key = str(getattr(self.cfg, "unnorm_key", "") or "").strip()
        if explicit_key:
            candidates = [explicit_key]
        else:
            task_key = str(getattr(self.cfg, "task_suite_name", "") or "").strip()
            mapped = _SAFELIBERO_TO_LIBERO_KEY.get(task_key, task_key)
            candidates = [task_key, f"{task_key}_no_noops", mapped, f"{mapped}_no_noops"]

        resolved = None
        for key in candidates:
            if key and key in norm_stats:
                resolved = key
                break

        if resolved is None and len(norm_stats) == 1:
            resolved = next(iter(norm_stats))
            logger.warning(
                "Could not resolve unnorm_key from task suite; using the only available key '%s'.",
                resolved,
            )

        if resolved is None:
            raise ValueError(
                f"Could not resolve unnorm_key for task_suite_name='{getattr(self.cfg, 'task_suite_name', '')}'. "
                f"Available keys: {list(norm_stats.keys())}. "
                "Pass `--unnorm_key` explicitly to match the RLinf checkpoint."
            )

        self.cfg.unnorm_key = resolved
        if hasattr(self.model, "unnorm_key"):
            self.model.unnorm_key = resolved

    def _build_env_obs(self, obs: dict, task_description: str) -> dict[str, Any]:
        state = self._format_state(obs["state"])
        main_image = np.asarray(obs["full_image"], dtype=np.uint8)
        wrist_image = np.asarray(obs["wrist_image"], dtype=np.uint8)
        num_images = int(
            getattr(
                self.model.vision_backbone,
                "get_num_images_in_input",
                lambda: int(getattr(self.cfg, "num_images_in_input", 1)),
            )()
        )
        num_aux_images = max(0, num_images - 1)

        if self._implement_version() == "official":
            env_obs: dict[str, Any] = {
                "task_descriptions": [task_description],
                "main_images": [main_image],
                "states": [state],
            }
            if num_aux_images > 0:
                wrist_stack = np.stack([wrist_image] * num_aux_images, axis=0)
                env_obs["wrist_images"] = [wrist_stack]
            return env_obs

        import torch

        env_obs = {
            "task_descriptions": [task_description],
            "main_images": torch.from_numpy(main_image).unsqueeze(0),
            "states": torch.from_numpy(state).unsqueeze(0),
        }
        if num_aux_images > 0:
            # DyniSafety-LIBERO exposes a single wrist image. Replicate it when the
            # RLinf checkpoint expects more than one auxiliary image.
            wrist_stack = np.stack([wrist_image] * num_aux_images, axis=0)
            env_obs["wrist_images"] = torch.from_numpy(wrist_stack).unsqueeze(0)
        return env_obs

    def _format_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        target_dim = int(getattr(self.cfg, "proprio_dim", state_array.shape[0]))
        if state_array.shape[0] == target_dim:
            return state_array
        if state_array.shape[0] > target_dim:
            logger.warning(
                "State dim %d is larger than proprio_dim=%d; truncating for RLinf policy.",
                state_array.shape[0],
                target_dim,
            )
            return state_array[:target_dim]

        padded = np.zeros((target_dim,), dtype=np.float32)
        padded[: state_array.shape[0]] = state_array
        logger.warning(
            "State dim %d is smaller than proprio_dim=%d; zero-padding for RLinf policy.",
            state_array.shape[0],
            target_dim,
        )
        return padded

    def _chunk_to_env_actions(self, chunk_actions: np.ndarray) -> List[np.ndarray]:
        env_actions: List[np.ndarray] = []
        for step in chunk_actions:
            action_dict = self._split_chunk_step(step)
            should_stop_after_step = bool(action_dict["terminate_episode"][0] > 0)
            if (
                should_stop_after_step
                and not self._warned_terminate_channel
            ):
                logger.info(
                    "RLinf checkpoint emitted terminate_episode>0. "
                    "The adapter will stop consuming the remaining chunk tail, while "
                    "DyniSafety-LIBERO still executes only the first 7 dims."
                )
                self._warned_terminate_channel = True

            env_action = np.concatenate(
                [
                    action_dict["world_vector"],
                    action_dict["rot_axangle"],
                    action_dict["gripper"],
                ],
                axis=0,
            ).astype(np.float32)

            if getattr(self.cfg, "postprocess_gripper", True):
                env_action = _normalize_gripper_action(env_action, binarize=True)
                env_action = _invert_gripper_action(env_action)

            env_actions.append(env_action)
            if should_stop_after_step:
                break
        return env_actions

    def _split_chunk_step(self, step: np.ndarray) -> dict[str, np.ndarray]:
        flat_step = np.asarray(step, dtype=np.float32).reshape(-1)
        if flat_step.size not in (7, 8):
            raise ValueError(
                f"Expected RLinf OpenVLA-OFT action_dim to be 7 or 8 for DyniSafety-LIBERO; "
                f"got {flat_step.size}."
            )

        terminate = (
            flat_step[7:8].astype(np.float32)
            if flat_step.size == 8
            else np.array([0.0], dtype=np.float32)
        )
        return {
            "world_vector": flat_step[0:3].astype(np.float32),
            "rot_axangle": flat_step[3:6].astype(np.float32),
            "gripper": flat_step[6:7].astype(np.float32),
            "terminate_episode": terminate,
        }

    def _to_numpy(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value

        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _implement_version(self) -> str:
        return str(getattr(self.cfg, "implement_version", "rlinf"))
