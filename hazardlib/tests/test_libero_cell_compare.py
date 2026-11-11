"""L6: policy comparison statistics / tables and the OFT client (no simulator, no model)."""
import json
import socket
import sys
import threading
import types

import numpy as np
import pytest

from hazardhosts.libero import cell_compare, oft_client, policies, serve_oft
from test_libero_cell_eval import SETTINGS, _records, _selection


def test_exact_mcnemar_holm_and_paired_difference():
    binom, holm, paired = cell_compare.binom_two_sided_p, cell_compare.holm, cell_compare.paired_difference
    assert binom(0, 0) == 1.0 and binom(5, 10) == 1.0
    assert binom(0, 10) == pytest.approx(2.0 / 1024) and binom(10, 10) == pytest.approx(2.0 / 1024)
    assert binom(2, 12) == pytest.approx(2 * (1 + 12 + 66) / 4096.0)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    a = [True] * 6 + [False] * 2 + [True, False]
    b = [True] * 2 + [False] * 4 + [True, False] + [True, False]
    r = paired(a, b)
    assert (r["n"], r["both"], r["only_a"], r["only_b"], r["neither"]) == (10, 3, 4, 1, 2)
    assert r["rate_a"] == 0.7 and r["rate_b"] == 0.4 and r["diff"] == pytest.approx(0.3)
    assert r["p"] == pytest.approx(binom(4, 5)) and r["ci"][0] < 0.3 < r["ci"][1]
    assert paired([], [])["diff"] is None
    with pytest.raises(ValueError):
        paired([True], [])


def test_comparison_pairs_by_instance_and_keeps_main_only(tmp_path):
    selection = _selection(tmp_path)
    first = _records()
    second = []
    for r in first:                      # the second policy: no contact anywhere, a.demo_4 fails
        r = dict(r, policy_name="toy2", log=dict(r["log"]))
        if r["kind"] == "hazard":
            r["log"]["first_official_contact_t_s"] = None
        if r["episode_id"] == "a.demo_4":
            r["success_t_s"] = None
        second.append(r)
    comparison = cell_compare.build_comparison({"p1": (selection, first, SETTINGS), "p2": (selection, second, SETTINGS)})
    assert comparison["labels"] == ["p1", "p2"] and comparison["n_instances"] == 5
    total = [r for r in comparison["tables"]["overall"] if r["cell"] == "ALL"][0]
    assert total["g1_passed"] == 3 and total["policies"]["p1"]["contact"]["k"] == 1
    assert total["policies"]["p2"]["contact"]["k"] == 0 and total["policies"]["p2"]["success"]["k"] == 1
    # main under both: a.demo_3, b.demo_3, a.demo_4 (a.demo_2 is low exposure, b.demo_4 invalid)
    assert comparison["n_main_all_policies"] == 3
    pair = comparison["paired"]["contact"]["p1 vs p2"]["overall"]["ALL"]
    assert (pair["n"], pair["only_a"], pair["only_b"], pair["p"]) == (3, 1, 0, 1.0)
    free = comparison["paired"]["contact_free_success"]["p1 vs p2"]["overall"]["ALL"]
    assert (free["only_a"], free["only_b"]) == (1, 1)
    assert set(comparison["paired"]["contact"]["p1 vs p2"]) == {"overall", "suite", "cell"}
    text = cell_compare.render_markdown(comparison)
    assert "G1 oracle pass" in text and "McNemar" in text and "p1 vs p2" in text
    json.dumps(comparison)
    (tmp_path / "other").mkdir()
    other = _selection(tmp_path / "other")
    other["tasks"]["libero_toy_task0"]["layouts"].pop()
    with pytest.raises(ValueError, match="other instances"):
        cell_compare.build_comparison({"p1": (selection, first, SETTINGS), "p2": (other, second, SETTINGS)})
    assert cell_compare.parse_run("a=/x/y:pol") == ("a", "/x/y", "pol") and cell_compare.parse_run("a=/x") == ("a", "/x", "a")
    with pytest.raises(ValueError):
        cell_compare.parse_run("nolabel")


def test_oft_client_framing_handshake_and_suite_routing(monkeypatch):
    assert oft_client.PROTOCOL == serve_oft.PROTOCOL
    seen = []
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        stream = conn.makefile("rwb")
        serve_oft.send_message(stream, {"meta": {"protocol": serve_oft.PROTOCOL}})
        while True:
            request = serve_oft.recv_message(stream)
            if request is None or request["cmd"] == "close":
                return
            seen.append(request)
            serve_oft.send_message(stream, {"ok": True} if request["cmd"] == "reset" else
                                   {"actions": np.ones((8, 7), np.float32)})
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    fake = types.ModuleType("dynisafety.policies.adapters._common")
    fake.actions_to_chunk = lambda actions, space: (np.asarray(actions), space)
    for name in ("dynisafety", "dynisafety.policies", "dynisafety.policies.adapters"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "dynisafety.policies.adapters._common", fake)
    cfg = types.SimpleNamespace(action_space="delta_ee_7d", extra={"server_host": "127.0.0.1",
                                                                  "server_port": server.getsockname()[1]})
    adapter = oft_client.OftRemoteAdapter.from_config(cfg)
    with pytest.raises(RuntimeError, match="before reset"):
        adapter.act(None)
    adapter.reset(types.SimpleNamespace(suite="libero_goal"))
    obs = types.SimpleNamespace(images={"primary": np.zeros((4, 4, 3), np.uint8), "wrist": np.ones((4, 4, 3), np.uint8)},
                                state=np.arange(8.0), instruction="open the drawer")
    actions, space = adapter.act(obs)
    adapter.close()
    thread.join(timeout=5)
    assert actions.shape == (8, 7) and space == "delta_ee_7d"
    assert seen[0]["cmd"] == "reset" and seen[1]["suite"] == "libero_goal" and seen[1]["prompt"] == "open the drawer"
    assert seen[1]["state"].shape == (8,) and seen[1]["wrist"].max() == 1
    server.close()


def test_the_bridge_swaps_the_in_process_oft_adapter_for_the_client():
    assert policies.REMOTE_ADAPTERS["dynisafety.policies.adapters.openvla_oft:OpenVLAOFTAdapter"] == \
        "hazardhosts.libero.oft_client:OftRemoteAdapter"
