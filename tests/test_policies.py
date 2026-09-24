"""Policy layer tests (M4) — no simulator."""

from __future__ import annotations

import ast
import json
import sys
import textwrap
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from dynisafety.core.registry import Registry, RegistryLoadError
from dynisafety.policies.action_space import (
    ACTION_SPACES,
    convert_action,
    effective_action_space,
    validate_chunk,
)
from dynisafety.policies.adapters.random import RandomAdapter
from dynisafety.policies.adapters.xvla import XVLAAdapter
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
    apply_per_backend,
    model_config_from_resolved,
)
from dynisafety.policies.registry import (
    available,
    create_policy,
    load_model_decls,
    populate_policies,
    refresh_decls,
)
from dynisafety.policies.transports.http import HttpTransport
from dynisafety.policies.transports.websocket import WebSocketTransport

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "configs" / "models"
ADAPTERS_DIR = REPO_ROOT / "src" / "dynisafety" / "policies" / "adapters"


def _minimal_local_cfg(**overrides: Any) -> ModelConfig:
    data: dict[str, Any] = {
        "name": "random",
        "transport": "local",
        "adapter": "dynisafety.policies.adapters.random:RandomAdapter",
        "action_space": "delta_ee_7d",
        "chunk_size": 2,
        "observation": {"images": ["primary"], "needs_state": False},
        "checkpoint": {"default": "/tmp/dynisafety-fake-ckpt"},
        "resources": {"max_jobs_per_gpu": 1},
        "backends": ["libero"],
        "extra": {"seed": 0},
    }
    data.update(overrides)
    return model_config_from_resolved(data)


# --- checkpoint.per_suite (P0-4) ---------------------------------------------
#
# Several models ship one fine-tuned checkpoint per task suite. Picking the
# wrong one produces plausible-but-wrong numbers instead of an error, so the
# resolution rules are pinned here.

_PER_SUITE = {
    "default": "ckpt/spatial",
    "per_suite": {
        "safelibero_spatial": "ckpt/spatial",
        "safelibero_object": "ckpt/object",
    },
}


def test_checkpoint_per_suite_selects_matching_suite() -> None:
    for suite, want in [
        ("safelibero_spatial", "ckpt/spatial"),
        ("safelibero_object", "ckpt/object"),
    ]:
        cfg = _minimal_local_cfg(checkpoint=dict(_PER_SUITE), suite=suite)
        assert cfg.checkpoint == want, suite


def test_checkpoint_per_suite_without_suite_uses_default() -> None:
    """Suite-less contexts (e.g. `dynisafety models`) must still resolve."""
    cfg = _minimal_local_cfg(checkpoint=dict(_PER_SUITE))
    assert cfg.checkpoint == "ckpt/spatial"


def test_checkpoint_per_suite_rejects_uncovered_suite() -> None:
    """Never silently fall back — that would run the wrong fine-tuned weights."""
    with pytest.raises(ValueError, match="per_suite has no entry for suite"):
        _minimal_local_cfg(checkpoint=dict(_PER_SUITE), suite="safelibero_goal")


def test_checkpoint_plain_string_ignores_suite() -> None:
    cfg = _minimal_local_cfg(checkpoint="ckpt/only", suite="safelibero_goal")
    assert cfg.checkpoint == "ckpt/only"


@pytest.mark.parametrize(
    "bad",
    [
        {"default": "d", "per_suite": []},
        {"default": "d", "per_suite": {}},
        {"default": "d", "per_suite": {"s": ""}},
        {"default": "d", "per_suite": {"": "c"}},
    ],
)
def test_checkpoint_per_suite_validation(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="per_suite"):
        _minimal_local_cfg(checkpoint=bad)


def test_apply_per_backend_noop_without_block() -> None:
    data = {"name": "x", "action_space": "delta_ee_7d", "extra": {"a": 1}}
    out = apply_per_backend(data, "robotwin")
    assert out["action_space"] == "delta_ee_7d"
    assert out["extra"] == {"a": 1}
    assert out is not data


def test_apply_per_backend_merges_robotwin_overlay() -> None:
    data = {
        "name": "starvla",
        "action_space": "delta_ee_7d",
        "observation": {"images": ["primary", "wrist"], "needs_state": True},
        "extra": {"obs_layout": "libero", "eval_env": "openvla-oft", "keep": True},
        "per_backend": {
            "robotwin": {
                "action_space": "qpos_14d",
                "observation": {"images": ["primary", "left", "right"]},
                "extra": {"obs_layout": "robotwin", "eval_env": "internvla_a1"},
            }
        },
    }
    out = apply_per_backend(data, "robotwin")
    assert out["action_space"] == "qpos_14d"
    assert out["observation"]["images"] == ["primary", "left", "right"]
    assert out["observation"]["needs_state"] is True
    assert out["extra"]["obs_layout"] == "robotwin"
    assert out["extra"]["eval_env"] == "internvla_a1"
    assert out["extra"]["keep"] is True
    # Defaults unchanged when backend has no overlay.
    libero = apply_per_backend(data, "libero")
    assert libero["action_space"] == "delta_ee_7d"
    assert libero["extra"]["obs_layout"] == "libero"


def test_apply_per_backend_rejects_unknown_overlay_keys() -> None:
    data = {
        "name": "x",
        "action_space": "delta_ee_7d",
        "per_backend": {"robotwin": {"transport": "http"}},
    }
    with pytest.raises(ValueError, match="unsupported keys"):
        apply_per_backend(data, "robotwin")


def test_apply_per_backend_chunk_size_overlay() -> None:
    data = {
        "name": "x",
        "action_space": "delta_ee_7d",
        "chunk_size": 8,
        "per_backend": {"robotwin": {"chunk_size": 50}},
    }
    out = apply_per_backend(data, "robotwin")
    assert out["chunk_size"] == 50
    assert apply_per_backend(data, "libero")["chunk_size"] == 8


def test_starvla_per_backend_robotwin_and_simplerenv() -> None:
    raw = yaml.safe_load((MODELS_DIR / "starvla.yaml").read_text(encoding="utf-8"))
    libero = model_config_from_resolved(raw, backend="libero")
    assert libero.action_space == "delta_ee_7d"
    assert libero.chunk_size == 8
    assert libero.observation_images == ("primary", "wrist")
    assert libero.extra["obs_layout"] == "libero"
    assert libero.extra["eval_env"] == "openvla-oft"
    assert libero.extra.get("use_ddim") is True
    assert libero.extra.get("num_ddim_steps") == 10

    rt = model_config_from_resolved(raw, backend="robotwin")
    assert rt.action_space == "qpos_14d"
    assert rt.chunk_size == 50
    assert rt.observation_images == ("primary", "left", "right")
    assert rt.extra["obs_layout"] == "robotwin"
    assert rt.extra["eval_env"] == "internvla_a1"

    se = model_config_from_resolved(raw, backend="simplerenv")
    assert se.action_space == "delta_ee_7d"
    assert se.chunk_size == 1
    assert se.extra["obs_layout"] == "simpler"
    assert se.extra["eval_env"] == "simpler_env"
    assert se.extra.get("unnorm_key") == "oxe_bridge"
    assert se.extra.get("action_ensemble") is True


def test_starvla_per_suite_checkpoints_and_bf16_args() -> None:
    from dynisafety.runner.scheduler import load_model_serve_spec

    root = REPO_ROOT / "configs"
    libero = load_model_serve_spec(
        "starvla", configs_root=root, suite="safelibero_spatial"
    )
    assert "steps_50000_pytorch_model.pt" in libero.checkpoint
    assert "--use_bf16" not in libero.serve_command

    robotwin = load_model_serve_spec("starvla", configs_root=root, suite="main20")
    assert "steps_140000_pytorch_model.pt" in robotwin.checkpoint
    assert "--use_bf16" in robotwin.serve_command

    bridge = load_model_serve_spec("starvla", configs_root=root, suite="bridge")
    assert "steps_45000_pytorch_model.pt" in bridge.checkpoint
    assert "--use_bf16" in bridge.serve_command


def test_xvla_official_per_backend_domain_and_ckpt() -> None:
    from dynisafety.runner.scheduler import load_model_serve_spec

    raw = yaml.safe_load((MODELS_DIR / "xvla_official.yaml").read_text(encoding="utf-8"))
    rt = model_config_from_resolved(raw, backend="robotwin")
    assert rt.extra["domain_id"] == 6
    assert rt.extra["gripper_threshold"] == 0.7
    assert rt.extra["eval_env"] == "internvla_a1"
    assert rt.action_space == "abs_ee_16d"

    se = model_config_from_resolved(raw, backend="simplerenv")
    assert se.extra["domain_id"] == 0
    assert se.extra["gripper_threshold"] == 0.95
    assert se.extra["eval_env"] == "simpler_env"
    assert se.action_space == "delta_ee_7d"
    assert se.observation_images == ("primary",)

    root = REPO_ROOT / "configs"
    main20 = load_model_serve_spec(
        "xvla_official", configs_root=root, suite="main20", backend="robotwin"
    )
    bridge = load_model_serve_spec(
        "xvla_official", configs_root=root, suite="bridge", backend="simplerenv"
    )
    assert "X-VLA-RoboTwin2" in main20.checkpoint
    assert "xvla-simpler-widowx" in bridge.checkpoint
    assert bridge.eval_env == "simpler_env"


def test_shipped_openvla_oft_covers_all_libero_suites() -> None:
    """The real config must cover every suite paper_main runs it on."""
    import yaml

    repo = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load(
        (repo / "configs" / "models" / "openvla_oft.yaml").read_text(encoding="utf-8")
    )
    per_suite = raw["checkpoint"]["per_suite"]
    suite_ids = {
        p.stem
        for p in (repo / "configs" / "backends" / "libero" / "suites").glob("*.yaml")
    }
    assert suite_ids, "no libero suite configs found"
    assert suite_ids <= set(per_suite), sorted(suite_ids - set(per_suite))
    # Distinct weights per suite — a copy-paste slip would defeat the purpose.
    assert len(set(per_suite.values())) == len(per_suite), per_suite


# ---------------------------------------------------------------------------
# action_space
# ---------------------------------------------------------------------------


def test_action_space_roundtrip_delta_abs() -> None:
    eef = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.0], dtype=np.float64)
    delta = np.array([[0.1, -0.2, 0.05, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    chunk = ActionChunk(actions=delta, space="delta_ee_7d")
    abs_actions = convert_action(
        chunk, target_space="abs_ee_7d", context={"eef_state": eef}
    )
    expected_abs = delta.copy()
    expected_abs[0, :6] = eef[:6] + delta[0, :6]
    np.testing.assert_allclose(abs_actions, expected_abs)

    back = convert_action(
        ActionChunk(actions=abs_actions, space="abs_ee_7d"),
        target_space="delta_ee_7d",
        context={"eef_state": eef},
    )
    np.testing.assert_allclose(back, delta)


def test_action_space_qpos_14d_identity_roundtrip() -> None:
    qpos = np.linspace(-0.5, 0.5, 14, dtype=np.float64).reshape(1, 14)
    chunk = ActionChunk(actions=qpos, space="qpos_14d")
    validate_chunk(chunk)
    out = convert_action(chunk, target_space="qpos_14d", context={})
    np.testing.assert_allclose(out, qpos)
    with pytest.raises(ValueError, match="unsupported"):
        convert_action(chunk, target_space="delta_ee_7d", context={})


def test_action_space_rejects_unknown() -> None:
    chunk = ActionChunk(
        actions=np.zeros((1, 7), dtype=np.float32), space="not_a_space"
    )
    with pytest.raises(ValueError, match="unknown"):
        validate_chunk(chunk)
    with pytest.raises(ValueError, match="unknown"):
        convert_action(
            ActionChunk(actions=np.zeros((1, 7)), space="delta_ee_7d"),
            target_space="bogus",
            context={"eef_state": np.zeros(7)},
        )


def test_validate_chunk_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        validate_chunk(
            ActionChunk(actions=np.zeros(7, dtype=np.float32), space="delta_ee_7d")
        )
    with pytest.raises(ValueError, match="dim mismatch"):
        validate_chunk(
            ActionChunk(actions=np.zeros((1, 3), dtype=np.float32), space="delta_ee_7d")
        )
    with pytest.raises(ValueError, match="chunk size"):
        validate_chunk(
            ActionChunk(actions=np.zeros((0, 7), dtype=np.float32), space="delta_ee_7d")
        )


def test_effective_action_space_warmup() -> None:
    assert (
        effective_action_space(
            configured_space="delta_ee_7d",
            step=0,
            warmup_steps=10,
            post_warmup_space="abs_ee_7d",
        )
        == "delta_ee_7d"
    )
    assert (
        effective_action_space(
            configured_space="delta_ee_7d",
            step=10,
            warmup_steps=10,
            post_warmup_space="abs_ee_7d",
        )
        == "abs_ee_7d"
    )
    assert (
        effective_action_space(
            configured_space="delta_ee_7d",
            step=99,
            warmup_steps=None,
            post_warmup_space="abs_ee_7d",
        )
        == "delta_ee_7d"
    )


# ---------------------------------------------------------------------------
# HTTP transport (mock)
# ---------------------------------------------------------------------------


def test_http_transport_infer_with_mock_server() -> None:
    transport = HttpTransport(
        base_url="http://127.0.0.1:9",
        health_path="/health",
        startup_timeout_s=1,
        act_timeout_s=5,
        reset_path="/reset",
        act_path="/act",
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"action": [[0.0] * 7]}

    with patch.object(transport._session, "post", return_value=mock_resp) as post:
        out = transport.infer({"image": [0], "instruction": "x"})
    assert out["action"] == [[0.0] * 7]
    assert post.call_args.args[0].endswith("/act")
    transport.close()


def test_http_transport_reset_404_is_noop() -> None:
    """X-VLA (and similar) expose /act only — missing /reset must not abort episodes."""
    transport = HttpTransport(
        base_url="http://127.0.0.1:9",
        health_path="/openapi.json",
        startup_timeout_s=1,
        act_timeout_s=5,
        reset_path="/reset",
        act_path="/act",
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status = MagicMock(
        side_effect=AssertionError("404 must not raise_for_status")
    )
    with patch.object(transport._session, "post", return_value=mock_resp):
        transport.reset({})  # must not raise
    transport.close()


def test_http_transport_reset_404_is_cached() -> None:
    """X-VLA has no /reset — probe once, then skip the dead POST each episode."""
    transport = HttpTransport(
        base_url="http://127.0.0.1:9",
        health_path="/openapi.json",
        startup_timeout_s=1,
        act_timeout_s=5,
        reset_path="/reset",
        act_path="/act",
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status = MagicMock(
        side_effect=AssertionError("404 must not raise_for_status")
    )
    with patch.object(transport._session, "post", return_value=mock_resp) as post:
        transport.reset({})
        transport.reset({})
    assert post.call_count == 1
    transport.close()


def test_websocket_transport_protocol_with_mock_codec() -> None:
    """WebSocketTransport must delegate to codec; host/port injected (no code defaults)."""

    class _FakeCodec:
        def __init__(self) -> None:
            self.connected: tuple[str, int] | None = None
            self.last_infer: dict[str, Any] | None = None
            self.closed = False

        def connect(self, host: str, port: int) -> None:
            self.connected = (host, port)

        def reset(self, payload: Any = None) -> None:
            return None

        def infer(self, payload: Any) -> dict[str, Any]:
            self.last_infer = dict(payload)
            return {"action": [[0.1] * 7]}

        def close(self) -> None:
            self.closed = True

    codec = _FakeCodec()
    transport = WebSocketTransport(host="127.0.0.1", port=8765, codec=codec)
    assert codec.connected == ("127.0.0.1", 8765)
    out = transport.infer({"instruction": "pick"})
    assert out["action"] == [[0.1] * 7]
    assert codec.last_infer == {"instruction": "pick"}
    transport.close()
    assert codec.closed is True


# ---------------------------------------------------------------------------
# Random adapter protocol
# ---------------------------------------------------------------------------


def test_adapter_protocol_compliance_random() -> None:
    cfg = _minimal_local_cfg()
    policy = RandomAdapter.from_config(cfg)
    ctx = EpisodeContext(
        instruction="pick",
        task_id=0,
        task_name="t0",
        episode_index=0,
        seed=0,
        level=0,
        backend="libero",
        suite="smoke",
        extra={},
    )
    policy.reset(ctx)
    obs = Observation(
        images={"primary": np.zeros((8, 8, 3), dtype=np.uint8)},
        state=None,
        instruction="pick",
        step=0,
        extra={},
    )
    chunk = policy.act(obs)
    assert isinstance(chunk, ActionChunk)
    assert chunk.space == "delta_ee_7d"
    assert chunk.actions.shape == (2, 7)
    validate_chunk(chunk)
    policy.close()


# ---------------------------------------------------------------------------
# Adapters: no raw transport primitives
# ---------------------------------------------------------------------------


_FORBIDDEN_ATTRS = frozenset({"post", "connect", "Context"})
_FORBIDDEN_MODULES = frozenset({"requests", "websockets", "zmq"})


def _adapter_uses_raw_transport(path: Path) -> list[str]:
    """AST scan for requests.post / websockets.connect / zmq.Context in adapters."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in _FORBIDDEN_MODULES and node.attr in _FORBIDDEN_ATTRS:
                hits.append(f"{node.value.id}.{node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if (
                isinstance(func.value, ast.Name)
                and func.value.id in _FORBIDDEN_MODULES
                and func.attr in _FORBIDDEN_ATTRS
            ):
                hits.append(f"{func.value.id}.{func.attr}()")
    return hits


def test_adapters_have_no_raw_transport_imports() -> None:
    offenders: list[str] = []
    for path in sorted(ADAPTERS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        hits = _adapter_uses_raw_transport(path)
        if hits:
            offenders.append(f"{path.name}: {hits}")
    assert not offenders, "adapters must not call raw transport APIs:\n" + "\n".join(
        offenders
    )


# ---------------------------------------------------------------------------
# Registry / available / YAML catalog
# ---------------------------------------------------------------------------


def test_available_reports_missing_checkpoint(tmp_path: Path) -> None:
    models = tmp_path / "configs" / "models"
    models.mkdir(parents=True)
    (models / "toy.yaml").write_text(
        textwrap.dedent(
            """\
            name: toy
            transport: local
            adapter: dynisafety.policies.adapters.random:RandomAdapter
            action_space: delta_ee_7d
            chunk_size: 1
            observation:
              images: [primary]
              needs_state: false
            checkpoint:
              default: /definitely/missing/dynisafety_ckpt_xyz
            resources:
              max_jobs_per_gpu: 1
            backends: [libero]
            extra:
              seed: 0
            """
        ),
        encoding="utf-8",
    )
    statuses = available(configs_root=tmp_path / "configs")
    by_name = {s.name: s for s in statuses}
    assert "toy" in by_name
    assert by_name["toy"].state == "missing_checkpoint"
    assert "memoryvla" not in by_name


def test_openvla_oft_rlinf_is_registered() -> None:
    path = MODELS_DIR / "openvla_oft_rlinf.yaml"
    assert path.is_file()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = model_config_from_resolved(raw)
    assert cfg.name == "openvla_oft_rlinf"
    assert "openvla_oft_rlinf" in cfg.adapter
    # Paper口径: OFT loader + single image, no proprio, no rlinf repo.
    assert cfg.observation_images == ("primary",)
    assert cfg.extra.get("use_proprio") is False
    assert cfg.extra.get("num_images_in_input") == 1
    assert cfg.extra.get("use_l1_regression") is False
    assert cfg.extra.get("do_sample") is True
    assert cfg.extra.get("temperature") == 1.6
    assert cfg.extra.get("top_p") == 1.0
    assert "rlinf_root" not in cfg.extra

    reg: Registry[Any] = Registry("policy-test-rlinf")
    populate_policies(reg, models_dir=MODELS_DIR)
    # Adapter import may fail without model deps; declaration + load_entry path
    # must not warn-skip the id from the catalog.
    decls = load_model_decls(MODELS_DIR)
    assert "openvla_oft_rlinf" in decls
    assert decls["openvla_oft_rlinf"].adapter.endswith("OpenVLAOFTRLinfAdapter")


def test_rlinf_sampling_extras_match_official_eval() -> None:
    from dynisafety.policies.adapters.openvla_oft_rlinf import (
        episode_sampling_seed,
        parse_rlinf_sampling_extras,
    )

    cfg = _minimal_local_cfg(
        extra={"do_sample": True, "temperature": 1.6, "top_p": 1.0}
    )
    assert parse_rlinf_sampling_extras(cfg) == (True, 1.6, 1.0)
    with pytest.raises(ValueError, match="nucleus"):
        parse_rlinf_sampling_extras(
            _minimal_local_cfg(
                extra={"do_sample": True, "temperature": 1.6, "top_p": 0.9}
            )
        )
    ctx = EpisodeContext(
        instruction="pick up the bowl",
        task_id=0,
        task_name="t0",
        episode_index=3,
        seed=1,
        level=0,
        backend="libero",
        suite="safelibero_spatial",
        extra={},
    )
    other = EpisodeContext(
        instruction=ctx.instruction,
        task_id=ctx.task_id,
        task_name=ctx.task_name,
        episode_index=ctx.episode_index,
        seed=7,
        level=ctx.level,
        backend=ctx.backend,
        suite=ctx.suite,
        extra={},
    )
    assert episode_sampling_seed(ctx) != episode_sampling_seed(other)


def test_openvla_oft_rlinf_sample_is_isolated_stem() -> None:
    path = MODELS_DIR / "openvla_oft_rlinf_sample.yaml"
    assert path.is_file()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = model_config_from_resolved(raw)
    assert cfg.name == "openvla_oft_rlinf_sample"
    assert cfg.adapter.endswith("OpenVLAOFTRLinfAdapter")
    assert cfg.extra.get("do_sample") is True
    assert cfg.extra.get("temperature") == 1.6
    assert cfg.extra.get("top_p") == 1.0
    decls = load_model_decls(MODELS_DIR)
    assert "openvla_oft_rlinf_sample" in decls


def test_rlinf_verl_output_to_chunk() -> None:
    from dynisafety.policies.adapters.openvla_oft_rlinf import verl_output_to_chunk

    chunk = np.arange(16, dtype=np.float32).reshape(2, 8)
    got = verl_output_to_chunk(chunk)
    assert got.shape == (2, 8)
    batched = chunk[np.newaxis, ...]
    got_b = verl_output_to_chunk(batched)
    np.testing.assert_array_equal(got_b, chunk)
    with pytest.raises(ValueError, match="batch dim"):
        verl_output_to_chunk(np.zeros((2, 8, 7), dtype=np.float32))


def test_lingbot_va_serve_uses_wrapper_script() -> None:
    from dynisafety.runner.scheduler import load_model_serve_spec

    spec = load_model_serve_spec(
        "lingbot_va", configs_root=REPO_ROOT / "configs", suite="main20"
    )
    assert not spec.missing_fields
    assert spec.needs_server
    assert "serve.sh" in " ".join(spec.serve_command)
    assert "{port}" in spec.serve_command
    assert "{checkpoint}" in spec.serve_command
    assert spec.health_url_template and "/healthz" in spec.health_url_template
    assert "lingbot-va-posttrain-robotwin" in spec.checkpoint
    assert "model_repos/lingbot-va" in str(
        yaml.safe_load((MODELS_DIR / "lingbot_va.yaml").read_text(encoding="utf-8"))[
            "server"
        ]["root"]
    )


def test_lingbot_serve_sh_walks_up_to_workspace_not_scripts() -> None:
    """serve.sh used to stop at scripts/ (three levels) and miss lingbot-va."""
    script = REPO_ROOT / "scripts" / "extra" / "models" / "lingbot_va" / "serve.sh"
    text = script.read_text(encoding="utf-8")
    assert "configs/paths.yaml" in text
    assert "src/dynisafety" in text
    three_up = (script.parent / "../../..").resolve()
    assert three_up.name == "scripts"
    wan = REPO_ROOT / "external" / "model_repos" / "lingbot-va" / "wan_va" / "wan_va_server.py"
    assert wan.is_file()
    assert f"{three_up}/external/model_repos/lingbot-va" not in str(wan)


def test_xvla_and_xvla_official_share_adapter_class() -> None:
    xvla = yaml.safe_load((MODELS_DIR / "xvla.yaml").read_text(encoding="utf-8"))
    official = yaml.safe_load(
        (MODELS_DIR / "xvla_official.yaml").read_text(encoding="utf-8")
    )
    assert xvla["adapter"] == official["adapter"]
    assert xvla["adapter"].endswith("XVLAAdapter")
    assert xvla["extra"]["variant"] == "default"
    assert official["extra"]["variant"] == "official"
    assert XVLAAdapter is XVLAAdapter  # shared class importable
    # LIBERO X-VLA predicts absolute EEF; settle is num_steps_wait, not warmup.
    assert xvla["action_space"] == "abs_ee_7d"
    assert "warmup_steps" not in xvla.get("extra", {})
    assert "post_warmup_action_space" not in xvla.get("extra", {})


def test_xvla_json_numpy_payload_is_string_not_list() -> None:
    """X-VLA /act requires json_numpy string fields (old libero client contract)."""
    json_numpy = pytest.importorskip("json_numpy")
    from dynisafety.policies.adapters.xvla import json_numpy_dumps

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    encoded = json_numpy_dumps(img)
    assert isinstance(encoded, str)
    # Must round-trip via json_numpy.loads (server side).
    decoded = json_numpy.loads(encoded)
    assert decoded.shape == img.shape
    assert decoded.dtype == img.dtype

    # Nested list (as_jsonable) is what previously broke the server.
    from dynisafety.policies.adapters._common import as_jsonable

    listed = as_jsonable(img)
    assert isinstance(listed, list)
    try:
        json_numpy.loads(listed)  # type: ignore[arg-type]
        raise AssertionError("list payload must not be accepted by json_numpy.loads")
    except TypeError:
        pass


def test_xvla_axangle_to_rot6d_uses_column_major() -> None:
    """Proprio 6D must be col0||col1, not C-order reshape of mat[:, :2]."""
    from scipy.spatial.transform import Rotation

    from dynisafety.policies.adapters.xvla import _axangle_to_rot6d

    aa = np.array([0.2, -0.4, 0.7], dtype=np.float64)
    mat = Rotation.from_rotvec(aa).as_matrix()
    got = _axangle_to_rot6d(aa)
    expect = np.concatenate([mat[:, 0], mat[:, 1]]).astype(np.float32)
    wrong = mat[:, :2].reshape(-1).astype(np.float32)
    np.testing.assert_allclose(got, expect, atol=1e-6)
    assert not np.allclose(got, wrong, atol=1e-5)


def _xvla_official_bridge_cfg():
    return model_config_from_resolved(
        {
            "name": "xvla_official",
            "transport": "http",
            "adapter": "dynisafety.policies.adapters.xvla:XVLAAdapter",
            "action_space": "delta_ee_7d",
            "chunk_size": 30,
            "observation": {"images": ["primary"], "needs_state": True},
            "checkpoint": {"default": "/tmp/fake-xvla"},
            "resources": {"max_jobs_per_gpu": 1},
            "backends": ["simplerenv"],
            "server": {
                "env": "XVLA",
                "root": "/tmp",
                "launch": "deploy.py",
                "default_port": 8010,
                "health_path": "/openapi.json",
                "startup_timeout_s": 60,
                "args": ["--model_path", "{checkpoint}", "--port", "{port}"],
            },
            "extra": {
                "variant": "official",
                "domain_id": 0,
                "denoising_steps": 10,
                "gripper_threshold": 0.95,
                "act_timeout_s": 60,
                "reset_path": "/reset",
                "act_path": "/act",
                "server_host": "127.0.0.1",
            },
        }
    )


def test_xvla_official_bridge_accepts_primary_only() -> None:
    """SimplerEnv Bridge has only primary; must not require left/right/wrist."""
    from dynisafety.policies.adapters.xvla import XVLAAdapter

    class _FakeTransport:
        def __init__(self) -> None:
            self.last = None
            self.n_infer = 0

        def reset(self, payload=None) -> None:
            return None

        def infer(self, payload):
            self.last = payload
            self.n_infer += 1
            # One 10-D absolute EE row (pos3 + rot6 interleaved cols + grip).
            # rot6 = identity columns interleaved: [c0x,c1x,c0y,c1y,c0z,c1z]
            return {"action": [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2]]}

        def close(self) -> None:
            return None

    transport = _FakeTransport()
    adapter = XVLAAdapter(_xvla_official_bridge_cfg(), transport)  # type: ignore[arg-type]
    adapter.reset(
        EpisodeContext(
            instruction="put carrot on plate",
            task_id=0,
            task_name="bridge_carrot",
            episode_index=0,
            seed=7,
            level=0,
            backend="simplerenv",
            suite="bridge",
            extra={},
        )
    )
    # ManiSkill WidowX eef_pos: xyz + wxyz + gripper (8D).
    eef8 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    obs = Observation(
        images={"primary": np.zeros((48, 48, 3), dtype=np.uint8)},
        state=eef8[:7].astype(np.float32),
        instruction="put carrot on plate",
        step=0,
        extra={"eef_pos": eef8},
    )
    chunk = adapter.act(obs)
    assert transport.last is not None
    assert transport.n_infer == 1
    assert "image0" in transport.last
    assert "image1" not in transport.last
    assert chunk.actions.shape == (1, 7)
    assert chunk.space == "delta_ee_7d"
    np.testing.assert_allclose(chunk.actions[0, :3], [0.1, 0.2, 0.3], atol=1e-5)
    # grip 0.2 < 0.95 → open/+1
    assert chunk.actions[0, 6] == pytest.approx(1.0)
    adapter.close()


def test_xvla_official_bridge_drains_plan_with_current_eef() -> None:
    """Abs→delta must use *current* 8D eef each step; one HTTP fill for the plan."""
    from dynisafety.policies.adapters.xvla import XVLAAdapter

    class _FakeTransport:
        def __init__(self) -> None:
            self.n_infer = 0

        def reset(self, payload=None) -> None:
            return None

        def infer(self, payload):
            self.n_infer += 1
            # Two absolute targets along +x.
            return {
                "action": [
                    [0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2],
                    [0.20, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2],
                ]
            }

        def close(self) -> None:
            return None

    transport = _FakeTransport()
    adapter = XVLAAdapter(_xvla_official_bridge_cfg(), transport)  # type: ignore[arg-type]
    adapter.reset(
        EpisodeContext(
            instruction="put carrot on plate",
            task_id=0,
            task_name="bridge_carrot",
            episode_index=0,
            seed=7,
            level=0,
            backend="simplerenv",
            suite="bridge",
            extra={},
        )
    )
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    eef_a = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    eef_b = np.array([0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    chunk0 = adapter.act(
        Observation(
            images={"primary": img},
            state=eef_a[:7].astype(np.float32),
            instruction="put carrot on plate",
            step=0,
            extra={"eef_pos": eef_a},
        )
    )
    chunk1 = adapter.act(
        Observation(
            images={"primary": img},
            state=eef_b[:7].astype(np.float32),
            instruction="put carrot on plate",
            step=1,
            extra={"eef_pos": eef_b},
        )
    )
    assert transport.n_infer == 1
    assert chunk0.actions.shape == (1, 7)
    assert chunk1.actions.shape == (1, 7)
    np.testing.assert_allclose(chunk0.actions[0, :3], [0.10, 0.0, 0.0], atol=1e-5)
    # Second step must subtract *current* eef (0.05), not the query-time eef (0.0).
    np.testing.assert_allclose(chunk1.actions[0, :3], [0.15, 0.0, 0.0], atol=1e-5)
    adapter.close()


def test_xvla_official_bridge_drain_can_omit_images() -> None:
    """HTTP fill needs primary; subsequent abs→delta steps only need current eef."""
    from dynisafety.policies.adapters.xvla import XVLAAdapter

    class _FakeTransport:
        def __init__(self) -> None:
            self.n_infer = 0

        def reset(self, payload=None) -> None:
            return None

        def infer(self, payload):
            self.n_infer += 1
            return {
                "action": [
                    [0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2],
                    [0.20, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2],
                ]
            }

        def close(self) -> None:
            return None

    transport = _FakeTransport()
    adapter = XVLAAdapter(_xvla_official_bridge_cfg(), transport)  # type: ignore[arg-type]
    adapter.reset(
        EpisodeContext(
            instruction="put carrot on plate",
            task_id=0,
            task_name="bridge_carrot",
            episode_index=0,
            seed=7,
            level=0,
            backend="simplerenv",
            suite="bridge",
            extra={},
        )
    )
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    eef_a = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    eef_b = np.array([0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    assert adapter.needs_visual_observation() is True
    chunk0 = adapter.act(
        Observation(
            images={"primary": img},
            state=eef_a[:7].astype(np.float32),
            instruction="put carrot on plate",
            step=0,
            extra={"eef_pos": eef_a},
        )
    )
    assert adapter.needs_visual_observation() is False
    chunk1 = adapter.act(
        Observation(
            images={},
            state=eef_b[:7].astype(np.float32),
            instruction="put carrot on plate",
            step=1,
            extra={"eef_pos": eef_b},
        )
    )
    assert transport.n_infer == 1
    np.testing.assert_allclose(chunk0.actions[0, :3], [0.10, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(chunk1.actions[0, :3], [0.15, 0.0, 0.0], atol=1e-5)
    adapter.close()


def test_registry_import_failure_is_loud() -> None:
    reg: Registry[Any] = Registry("policy-test-loud")
    with pytest.raises(RegistryLoadError):
        reg.load_entry("broken", "dynisafety.policies.adapters.no_such_mod:Nope")

    # available() must surface status rows for every stem (never omit / warn-skip)
    statuses = available(configs_root=REPO_ROOT / "configs")
    names = {s.name for s in statuses}
    assert "openvla_oft_rlinf" in names
    assert "memoryvla" not in names
    for s in statuses:
        assert s.state in {
            "ready",
            "missing_dependency",
            "missing_checkpoint",
            "load_error",
        }


def test_create_policy_random() -> None:
    reg: Registry[Any] = Registry("policy-test-create")
    reg.register("random")(RandomAdapter)
    cfg = _minimal_local_cfg(chunk_size=1)
    policy = create_policy(cfg, reg)
    assert isinstance(policy, RandomAdapter)


def test_frozen_model_yamls_parse() -> None:
    """Every configs/models/*.yaml builds a ModelConfig (unresolved ${} ok)."""
    assert not (MODELS_DIR / "openpi.yaml").exists()
    assert not (MODELS_DIR / "memoryvla.yaml").exists()
    stems = sorted(p.stem for p in MODELS_DIR.glob("*.yaml"))
    assert "random" in stems
    assert "lingbot_va" in stems
    assert "openvla_oft_rlinf" in stems
    assert "xiaomi" in stems
    for stem in stems:
        raw = yaml.safe_load((MODELS_DIR / f"{stem}.yaml").read_text(encoding="utf-8"))
        cfg = model_config_from_resolved(raw)
        assert cfg.name == stem
        assert cfg.action_space in ACTION_SPACES
        assert cfg.transport in {"local", "http", "websocket", "zmq"}
        assert cfg.chunk_size >= 1
        assert cfg.max_jobs_per_gpu >= 1


def test_lingbot_va_dump_artifacts_default_false() -> None:
    raw = yaml.safe_load((MODELS_DIR / "lingbot_va.yaml").read_text(encoding="utf-8"))
    cfg = model_config_from_resolved(raw)
    assert cfg.extra.get("dump_artifacts") is False
    assert cfg.action_space == "abs_ee_16d"
    assert cfg.chunk_size == 1


def test_refresh_decls_lists_all_yamls() -> None:
    decls = refresh_decls(configs_root=REPO_ROOT / "configs")
    assert "memoryvla" not in decls
    assert set(decls) == {p.stem for p in MODELS_DIR.glob("*.yaml")}


# --- StarVLA wire contract (regression: all-fail on missing predict_action) ---


def test_starvla_codec_prefers_predict_action(tmp_path: Path) -> None:
    """Upstream StarVLA client exposes predict_action, not infer/__call__."""
    from dynisafety.policies.transports.codecs.starvla import StarVLACodec

    class _FakeStarClient:
        def __init__(self, host: str, port: int) -> None:
            self.host = host
            self.port = port
            self.last = None

        def predict_action(self, query_info: dict) -> dict:
            self.last = query_info
            return {"data": {"normalized_actions": [[[0.0] * 7]]}}

    tools_mod = types.ModuleType("deployment.model_server.tools.websocket_policy_client")
    tools_mod.WebsocketClientPolicy = _FakeStarClient  # type: ignore[attr-defined]
    sys.modules["deployment"] = types.ModuleType("deployment")
    sys.modules["deployment.model_server"] = types.ModuleType("deployment.model_server")
    sys.modules["deployment.model_server.tools"] = types.ModuleType(
        "deployment.model_server.tools"
    )
    sys.modules["deployment.model_server.tools.websocket_policy_client"] = tools_mod
    codec = StarVLACodec(repo_root=str(tmp_path))
    try:
        codec.connect("127.0.0.1", 10093)
        out = codec.infer({"examples": [{"image": [], "lang": "pick"}]})
    finally:
        for key in (
            "deployment.model_server.tools.websocket_policy_client",
            "deployment.model_server.tools",
            "deployment.model_server",
            "deployment",
        ):
            sys.modules.pop(key, None)
    assert "data" in out
    assert codec._client.last["examples"][0]["lang"] == "pick"  # noqa: SLF001


def test_starvla_adapter_libero_payload_and_unnorm(tmp_path: Path) -> None:
    """Adapter must wrap examples + de-normalize like archived LIBERO client."""
    from dynisafety.policies.adapters.starvla import StarVLAAdapter

    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "step.pt"
    ckpt.write_bytes(b"fake")
    (run_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "framework": {"action_model": {"future_action_window_size": 1}},
                "datasets": {"vla_data": {"default_image_resolution": [224, 224]}},
            }
        ),
        encoding="utf-8",
    )
    # min/max identity-ish range so unnorm of ±1 maps cleanly; gripper channel [0,1].
    (run_dir / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "franka": {
                    "action": {
                        "min": [-1, -1, -1, -1, -1, -1, 0],
                        "max": [1, 1, 1, 1, 1, 1, 1],
                        "mask": [True] * 7,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    class _FakeTransport:
        def __init__(self) -> None:
            self.last_payload = None

        def reset(self, payload=None) -> None:
            return None

        def infer(self, payload):
            self.last_payload = payload
            # chunk=2: first grip open-ish (>0.5), second closed-ish
            return {
                "data": {
                    "normalized_actions": [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
                        ]
                    ]
                }
            }

        def close(self) -> None:
            return None

    cfg = model_config_from_resolved(
        {
            "name": "starvla",
            "transport": "websocket",
            "adapter": "dynisafety.policies.adapters.starvla:StarVLAAdapter",
            "action_space": "delta_ee_7d",
            "chunk_size": 2,
            "observation": {"images": ["primary", "wrist"], "needs_state": True},
            "checkpoint": {"default": str(ckpt)},
            "resources": {"max_jobs_per_gpu": 1},
            "backends": ["libero"],
            "server": {
                "env": "starVLA",
                "root": str(tmp_path),
                "launch": "deployment/model_server/server_policy.py",
                "default_port": 10093,
                "health_path": "/health",
                "startup_timeout_s": 60,
                "args": ["--ckpt_path", "{checkpoint}", "--port", "{port}"],
            },
            "extra": {
                "ws_codec": "starvla",
                "obs_layout": "libero",
                "use_ddim": True,
                "num_ddim_steps": 10,
                "server_host": "127.0.0.1",
            },
        }
    )
    transport = _FakeTransport()
    adapter = StarVLAAdapter(cfg, transport)  # type: ignore[arg-type]
    obs = Observation(
        images={
            "primary": np.zeros((64, 64, 3), dtype=np.uint8),
            "wrist": np.zeros((64, 64, 3), dtype=np.uint8),
        },
        state=np.zeros(8, dtype=np.float32),
        instruction="put the bowl on the plate",
        step=0,
        extra={},
    )
    chunk = adapter.act(obs)
    assert transport.last_payload is not None
    assert "examples" in transport.last_payload
    assert transport.last_payload["do_sample"] is False
    assert transport.last_payload["use_ddim"] is True
    assert len(transport.last_payload["examples"][0]["image"]) == 2
    assert chunk.actions.shape == (2, 7)
    # LIBERO gripper binarise: open (>0.5) -> -1, closed -> +1
    assert chunk.actions[0, 6] == pytest.approx(-1.0)
    assert chunk.actions[1, 6] == pytest.approx(1.0)
    adapter.close()


def test_starvla_codec_falls_back_to_infer_for_internvla(tmp_path: Path) -> None:
    """InternVLA-M1 client only has infer(); codec must still work."""
    from dynisafety.policies.transports.codecs.starvla import StarVLACodec

    class _FakeM1Client:
        def __init__(self, host: str, port: int) -> None:
            self.host = host
            self.port = port

        def infer(self, obs: dict) -> dict:
            return {"ok": True, "data": {"normalized_actions": [[[0.0] * 7]]}}

    tools_mod = types.ModuleType("deployment.model_server.tools.websocket_policy_client")
    tools_mod.WebsocketClientPolicy = _FakeM1Client  # type: ignore[attr-defined]
    sys.modules["deployment"] = types.ModuleType("deployment")
    sys.modules["deployment.model_server"] = types.ModuleType("deployment.model_server")
    sys.modules["deployment.model_server.tools"] = types.ModuleType(
        "deployment.model_server.tools"
    )
    sys.modules["deployment.model_server.tools.websocket_policy_client"] = tools_mod
    codec = StarVLACodec(repo_root=str(tmp_path))
    try:
        codec.connect("127.0.0.1", 10094)
        out = codec.infer({"batch_images": [], "instructions": ["x"]})
    finally:
        for key in (
            "deployment.model_server.tools.websocket_policy_client",
            "deployment.model_server.tools",
            "deployment.model_server",
            "deployment",
        ):
            sys.modules.pop(key, None)
    assert out["ok"] is True
