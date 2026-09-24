"""VLA-Adapter-Pro wiring tests (no GPU, no network, no other-model expectation changes)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from dynisafety.policies.adapters.vla_adapter import VLAAdapter
from dynisafety.policies.base import EpisodeContext, Observation, model_config_from_resolved
from dynisafety.runner.scheduler import load_model_serve_spec

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "configs" / "models"
SCRIPT = REPO / "scripts" / "setup" / "download_assets.py"
SERVE = REPO / "scripts" / "vla_adapter" / "deploy_vla_adapter_dynisafety_server.py"
ADAPTER = REPO / "src" / "dynisafety" / "policies" / "adapters" / "vla_adapter.py"

_UNNORM = {
    "safelibero_spatial": ("libero_spatial", "libero_spatial_no_noops"),
    "safelibero_object": ("libero_object", "libero_object_no_noops"),
    "safelibero_goal": ("libero_goal", "libero_goal_no_noops"),
    "safelibero_long": ("libero_10", "libero_10_no_noops"),
}


def _load_da():
    spec = importlib.util.spec_from_file_location("download_assets_vla_adapter", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_serve():
    spec = importlib.util.spec_from_file_location("vla_adapter_serve", SERVE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _raw_yaml() -> dict:
    return yaml.safe_load((MODELS / "vla_adapter.yaml").read_text(encoding="utf-8"))


def test_vla_adapter_yaml_parses() -> None:
    raw = _raw_yaml()
    cfg = model_config_from_resolved(raw)
    assert cfg.name == "vla_adapter"
    assert cfg.transport == "http"
    assert cfg.action_space == "delta_ee_7d"
    assert cfg.chunk_size == 8
    assert cfg.observation_images == ("primary", "wrist")
    assert cfg.observation_needs_state is True
    assert cfg.backends == ("libero",)
    assert cfg.max_jobs_per_gpu == 6
    assert cfg.extra["eval_env"] == "openvla-oft"
    assert cfg.extra["serve_env"] == "vla-adapter"
    assert cfg.extra["needs_server"] is True
    assert cfg.extra["act_path"] == "/act"
    assert cfg.extra["reset_path"] == "/reset"
    assert cfg.server_default_port == 8050
    assert cfg.server_launch == "scripts/vla_adapter/deploy_vla_adapter_dynisafety_server.py"


def test_vla_adapter_serve_spec_locks_official_protocol() -> None:
    root = REPO / "configs"
    spatial = load_model_serve_spec(
        "vla_adapter", configs_root=root, suite="safelibero_spatial", backend="libero"
    )
    long = load_model_serve_spec(
        "vla_adapter", configs_root=root, suite="safelibero_long", backend="libero"
    )
    assert spatial.needs_server is True
    assert spatial.missing_fields == ()
    assert spatial.checkpoint_error is None
    assert spatial.max_jobs_per_gpu == 6
    assert spatial.eval_env == "openvla-oft"
    assert spatial.serve_env == "vla-adapter"
    assert spatial.checkpoint.endswith("vla_adapter/LIBERO-Spatial-Pro")
    assert long.checkpoint.endswith("vla_adapter/LIBERO-Long-Pro")
    argv = " ".join(spatial.serve_command)
    assert "deploy_vla_adapter_dynisafety_server.py" in argv
    assert spatial.serve_vars["task_suite_name"] == "libero_spatial"
    assert spatial.serve_vars["unnorm_key"] == "libero_spatial_no_noops"
    assert long.serve_vars["task_suite_name"] == "libero_10"
    assert long.serve_vars["unnorm_key"] == "libero_10_no_noops"
    assert "libero_long" not in long.serve_vars["task_suite_name"]
    assert "libero_long" not in long.serve_vars["unnorm_key"]
    for token in (
        "--model_family",
        "openvla",
        "--use_pro_version",
        "true",
        "--use_minivlm",
        "true",
        "--center_crop",
        "true",
        "--num_images_in_input",
        "2",
        "--num_open_loop_steps",
        "8",
    ):
        assert token in spatial.serve_command
    assert "openvla_oft" not in spatial.serve_command
    launch = Path(spatial.serve_cwd or "") / "scripts/vla_adapter/deploy_vla_adapter_dynisafety_server.py"
    assert launch.is_file()
    assert spatial.health_url_template and "/health" in spatial.health_url_template


@pytest.mark.parametrize("suite,expected", list(_UNNORM.items()))
def test_vla_adapter_per_suite_unnorm_keys(suite: str, expected: tuple[str, str]) -> None:
    spec = load_model_serve_spec(
        "vla_adapter", configs_root=REPO / "configs", suite=suite, backend="libero"
    )
    task_suite, unnorm = expected
    assert spec.serve_vars["task_suite_name"] == task_suite
    assert spec.serve_vars["unnorm_key"] == unnorm


def test_paper_select_excludes_vla_adapter() -> None:
    da = _load_da()
    selected = da.select_specs(workspace=REPO, paper=True)
    names = {spec.name for spec in selected}
    assert "vla-adapter-libero-spatial-pro" not in names
    assert "vla-adapter-libero-object-pro" not in names
    assert "vla-adapter-libero-goal-pro" not in names
    assert "vla-adapter-libero-long-pro" not in names


def test_model_vla_adapter_selects_four_pro_assets() -> None:
    da = _load_da()
    selected = da.select_specs(workspace=REPO, models=["vla_adapter"])
    assert {spec.name for spec in selected} == {
        "vla-adapter-libero-spatial-pro",
        "vla-adapter-libero-object-pro",
        "vla-adapter-libero-goal-pro",
        "vla-adapter-libero-long-pro",
    }
    dests = {spec.relative_dest for spec in selected}
    assert dests == {
        "vla_adapter/LIBERO-Spatial-Pro",
        "vla_adapter/LIBERO-Object-Pro",
        "vla_adapter/LIBERO-Goal-Pro",
        "vla_adapter/LIBERO-Long-Pro",
    }


def test_paper_main_yaml_does_not_list_vla_adapter() -> None:
    text = (REPO / "configs" / "suites" / "paper_main.yaml").read_text(encoding="utf-8")
    assert "vla_adapter" not in text


class _FakeTransport:
    def __init__(self) -> None:
        self.reset_payloads: list[dict] = []
        self.infer_payloads: list[dict] = []

    def reset(self, payload):
        self.reset_payloads.append(dict(payload))

    def infer(self, payload):
        self.infer_payloads.append(payload)
        return {"ok": True, "actions": [[0.0] * 7] * 8}

    def close(self) -> None:
        return None


def test_vla_adapter_adapter_uses_examples_http_contract() -> None:
    raw = _raw_yaml()
    cfg = model_config_from_resolved(raw)
    transport = _FakeTransport()
    adapter = VLAAdapter(cfg, transport)
    adapter.reset(
        EpisodeContext(
            instruction="pick up the bowl",
            task_id=0,
            task_name="t0",
            episode_index=0,
            seed=7,
            level=0,
            backend="libero",
            suite="safelibero_spatial",
            extra={},
        )
    )
    obs = Observation(
        images={
            "primary": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist": np.zeros((256, 256, 3), dtype=np.uint8),
        },
        state=np.zeros(8, dtype=np.float32),
        instruction="pick up the bowl",
        step=10,
        extra={},
    )
    chunk = adapter.act(obs)
    assert chunk.space == "delta_ee_7d"
    assert chunk.actions.shape == (8, 7)
    assert transport.reset_payloads
    sent = transport.infer_payloads[0]
    assert set(sent["images"]) == {"primary", "wrist"}
    assert sent["horizon"] == 8
    assert sent["space"] == "delta_ee_7d"
    assert sent["instruction"] == "pick up the bowl"
    adapter.close()


def test_client_does_not_invert_or_rotate() -> None:
    text = ADAPTER.read_text(encoding="utf-8")
    assert "invert_gripper" not in text
    assert "normalize_gripper" not in text
    assert "[::-1, ::-1]" not in text
    assert "get_libero_image" not in text


def test_serve_protocol_helpers() -> None:
    serve = _load_serve()
    assert serve._NUM_OPEN_LOOP_STEPS == 8
    assert serve._ACTION_DIM == 7
    assert serve._PROPRIO_DIM == 8
    assert serve._REQUIRED_SPACE == "delta_ee_7d"
    stats = {
        "libero_spatial_no_noops": {},
        "libero_10_no_noops": {},
    }
    assert (
        serve.resolve_unnorm_key("libero_spatial", "libero_spatial", stats)
        == "libero_spatial_no_noops"
    )
    assert (
        serve.resolve_unnorm_key("libero_spatial_no_noops", "libero_spatial", stats)
        == "libero_spatial_no_noops"
    )
    assert serve.resolve_unnorm_key("libero_10", "libero_10", stats) == "libero_10_no_noops"
    with pytest.raises(ValueError, match="libero_10"):
        serve.resolve_unnorm_key("libero_long", "libero_long", stats)
    raw = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
    out = serve.process_openvla_gripper(raw, model_family="openvla")
    # 0.9 → normalize to +1, invert → -1 (LIBERO open).
    assert out[-1] == -1.0
    closed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    assert serve.process_openvla_gripper(closed, model_family="openvla")[-1] == 1.0
    mapped = serve.observation_for_vla(
        np.zeros((256, 256, 3), dtype=np.uint8),
        np.ones((256, 256, 3), dtype=np.uint8),
        np.zeros(8, dtype=np.float32),
    )
    assert set(mapped) == {"full_image", "wrist_image", "state"}
    src = SERVE.read_text(encoding="utf-8")
    assert "get_libero_image" not in src
    assert "[::-1, ::-1]" not in src
    assert "process_openvla_gripper" in src


def test_parse_args_rejects_wrong_family_and_chunk() -> None:
    serve = _load_serve()
    base = [
        "--port",
        "8050",
        "--pretrained_checkpoint",
        "/tmp",
        "--repo_root",
        "/tmp",
        "--task_suite_name",
        "libero_spatial",
        "--unnorm_key",
        "libero_spatial_no_noops",
        "--model_family",
        "openvla",
        "--use_pro_version",
        "true",
        "--use_minivlm",
        "true",
        "--use_l1_regression",
        "true",
        "--use_film",
        "false",
        "--use_proprio",
        "true",
        "--center_crop",
        "true",
        "--num_images_in_input",
        "2",
        "--num_open_loop_steps",
        "8",
    ]
    parsed = serve.parse_args(base)
    assert parsed.model_family == "openvla"
    assert parsed.num_open_loop_steps == 8
    wrong_family = list(base)
    family_at = wrong_family.index("--model_family")
    wrong_family[family_at + 1] = "openvla_oft"
    with pytest.raises(SystemExit, match="openvla"):
        serve.parse_args(wrong_family)
    wrong_chunk = list(base)
    wrong_chunk[-1] = "12"
    with pytest.raises(SystemExit, match="num_open_loop_steps"):
        serve.parse_args(wrong_chunk)
    wrong_long = list(base)
    suite_at = wrong_long.index("--task_suite_name")
    wrong_long[suite_at + 1] = "libero_long"
    with pytest.raises(SystemExit, match="libero_10"):
        serve.parse_args(wrong_long)


def test_vla_adapter_libero_suite_covers_four_suites() -> None:
    path = REPO / "configs" / "suites" / "vla_adapter_libero.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["name"] == "vla_adapter_libero"
    cell = raw["matrix"][0]
    assert cell["models"] == ["vla_adapter"]
    assert cell["levels"] == [0, 1, 2, 3]
    assert cell["tasks"] == [0, 1, 2, 3]
    assert cell["suites"] == [
        "safelibero_spatial",
        "safelibero_object",
        "safelibero_goal",
        "safelibero_long",
    ]
    assert "vla_adapter" not in (
        (REPO / "configs" / "suites" / "paper_main.yaml").read_text(encoding="utf-8")
    )
