"""TurboVLA wiring tests (no GPU, no network, no other-model expectation changes)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

from dynisafety.policies.adapters.turbovla import TurboVLAAdapter
from dynisafety.policies.base import EpisodeContext, Observation, model_config_from_resolved
from dynisafety.runner.scheduler import load_model_serve_spec

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "configs" / "models"
SCRIPT = REPO / "scripts" / "setup" / "download_assets.py"


def _load_da():
    spec = importlib.util.spec_from_file_location("download_assets_turbovla", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _raw_yaml() -> dict:
    return yaml.safe_load((MODELS / "turbovla.yaml").read_text(encoding="utf-8"))


def test_turbovla_yaml_parses() -> None:
    raw = _raw_yaml()
    cfg = model_config_from_resolved(raw)
    assert cfg.name == "turbovla"
    assert cfg.transport == "http"
    assert cfg.action_space == "delta_ee_7d"
    assert cfg.chunk_size == 12
    assert cfg.observation_images == ("primary", "wrist")
    assert cfg.observation_needs_state is True
    assert cfg.backends == ("libero",)
    assert cfg.max_jobs_per_gpu == 6
    assert cfg.extra["eval_env"] == "openvla-oft"
    assert cfg.extra["serve_env"] == "turbovla"
    assert cfg.extra["needs_server"] is True
    assert cfg.extra["act_path"] == "/act"
    assert cfg.extra["reset_path"] == "/reset"
    assert cfg.server_default_port == 8040
    assert cfg.server_launch == "scripts/turbovla/deploy_turbovla_dynisafety_server.py"


def test_turbovla_serve_spec_uses_unified_libero_ckpt() -> None:
    root = REPO / "configs"
    spatial = load_model_serve_spec(
        "turbovla", configs_root=root, suite="safelibero_spatial", backend="libero"
    )
    obj = load_model_serve_spec(
        "turbovla", configs_root=root, suite="safelibero_object", backend="libero"
    )
    assert spatial.needs_server is True
    assert spatial.missing_fields == ()
    assert spatial.checkpoint_error is None
    assert spatial.max_jobs_per_gpu == 6
    assert spatial.eval_env == "openvla-oft"
    assert spatial.serve_env == "turbovla"
    unified = "checkpoints/libero/turbovla_libero.pth"
    assert spatial.checkpoint.endswith(unified)
    assert obj.checkpoint.endswith(unified)
    argv = " ".join(spatial.serve_command)
    assert "deploy_turbovla_dynisafety_server.py" in argv
    assert "--dinov3_path" in spatial.serve_command
    assert spatial.serve_vars["dinov3_path"].endswith("backbones/dinov3-vitb16")
    assert spatial.serve_vars["bert_path"].endswith("backbones/bert-base-uncased")
    assert spatial.serve_vars["stats_path"].endswith("libero_all4_stats.json")
    assert spatial.serve_vars["stats_key"] == "libero_all4_no_noops"
    assert spatial.serve_vars["precision"] == "bf16"
    assert spatial.health_url_template and "/health" in spatial.health_url_template
    launch = Path(spatial.serve_cwd or "") / "scripts/turbovla/deploy_turbovla_dynisafety_server.py"
    assert launch.is_file()


def test_paper_main_yaml_does_not_list_turbovla() -> None:
    text = (REPO / "configs" / "suites" / "paper_main.yaml").read_text(encoding="utf-8")
    assert "turbovla" not in text


def test_turbovla_libero_suite_covers_four_suites() -> None:
    path = REPO / "configs" / "suites" / "turbovla_libero.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["name"] == "turbovla_libero"
    cell = raw["matrix"][0]
    assert cell["backend"] == "libero"
    assert cell["models"] == ["turbovla"]
    assert cell["levels"] == [0, 1, 2, 3]
    assert cell["tasks"] == [0, 1, 2, 3]
    assert cell["suites"] == [
        "safelibero_spatial",
        "safelibero_object",
        "safelibero_goal",
        "safelibero_long",
    ]
    eval_sh = (REPO / "scripts" / "eval" / "01_eval.sh").read_text(encoding="utf-8")
    assert "turbovla_libero" in eval_sh
    assert 'csv_to_pipe "$MODELS")" == "turbovla"' in eval_sh
    serve_sh = (REPO / "scripts" / "eval" / "serve.sh").read_text(encoding="utf-8")
    assert '"$MODEL" == "turbovla"' in serve_sh


def test_paper_select_excludes_turbovla() -> None:
    da = _load_da()
    selected = da.select_specs(workspace=REPO, paper=True)
    names = {spec.name for spec in selected}
    assert "turbovla-libero" not in names
    assert "turbovla-dinov3-vitb" not in names
    assert "turbovla-bert-base" not in names


def test_model_turbovla_selects_three_assets() -> None:
    da = _load_da()
    selected = da.select_specs(workspace=REPO, models=["turbovla"])
    assert {spec.name for spec in selected} == {
        "turbovla-libero",
        "turbovla-dinov3-vitb",
        "turbovla-bert-base",
    }
    dests = {spec.relative_dest for spec in selected}
    assert "turbovla" in dests
    assert "turbovla/backbones/dinov3-vitb16" in dests
    assert "turbovla/backbones/bert-base-uncased" in dests


class _FakeTransport:
    def __init__(self) -> None:
        self.reset_payloads: list[dict] = []
        self.infer_payloads: list[dict] = []

    def reset(self, payload):
        self.reset_payloads.append(dict(payload))

    def infer(self, payload):
        self.infer_payloads.append(payload)
        return {"ok": True, "actions": [[0.0] * 7] * 12}

    def close(self) -> None:
        return None


def test_turbovla_adapter_uses_examples_http_contract() -> None:
    raw = _raw_yaml()
    cfg = model_config_from_resolved(raw)
    transport = _FakeTransport()
    adapter = TurboVLAAdapter(cfg, transport)
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
    assert chunk.actions.shape == (12, 7)
    assert transport.reset_payloads
    sent = transport.infer_payloads[0]
    assert set(sent["images"]) == {"primary", "wrist"}
    assert sent["horizon"] == 12
    assert sent["space"] == "delta_ee_7d"
    assert sent["instruction"] == "pick up the bowl"
    adapter.close()
