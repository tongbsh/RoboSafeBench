# Example embodied-agent integration

[简体中文](README.zh-CN.md)

Walkthrough for plugging a VLA / WAM into DyniSafety. This folder is a
runnable example, not a second contract.

Canonical procedure, frozen contracts, and install rules live elsewhere — read
those, do not re-learn them here:

- [docs/adding_a_model.md](../../docs/adding_a_model.md) — 6-step procedure
- [docs/contracts.md](../../docs/contracts.md) — frozen contracts
- [docs/installation.md](../../docs/installation.md) — one backend per environment; extras are mutually exclusive
- [docs/benchmark_design.md](../../docs/benchmark_design.md), [docs/metrics.md](../../docs/metrics.md)

## What this is (and is not)

`MockBackbone` is a deterministic toy projector. It is **not** a real policy.
Numbers it produces must **never** be reported as DyniSafety benchmark results.
Use this tree to check that your adapter, YAML, and transport wiring match the
harness — then swap the toy weights for your model.

## 5-minute run

From the repository root, with DyniSafety already installed in the current env:

```bash
pip install -e examples/example_agent
# or: make example-agent-install
python examples/example_agent/verify.py
```

Expected table (CLI rows SKIP if `dynisafety` is not on `PATH`):

```
check                                result
-----------------------------------  ------
offline libero                       PASS
offline simplerenv                   PASS
offline robotwin                     PASS
chunk modes                          PASS
http round-trip                      PASS
eval libero --dry-run                PASS
eval simplerenv --dry-run            PASS
eval robotwin --dry-run              PASS
sweep smoke_example_agent --dry-run  PASS
```

## The two-file rule

| Piece | Where it may live | Why |
|-------|-------------------|-----|
| Adapter class | Anywhere importable | The YAML `adapter:` field is an import path. This package is **not** under `src/`. |
| Model YAML | `configs/models/<stem>.yaml` only | The registry scans that directory. The registry key **is** the filename stem. `--model` does not accept a path. |

`name:` inside the YAML **must equal the stem**. `create_policy` looks up
`cfg.name`; a mismatch is a `KeyError`.

This example: `configs/models/example_agent.yaml` →
`adapter: dyni_example_agent.local_adapter:ExampleAgentAdapter`.

## Contract surface

`Observation` is a dataclass, not a raw env dict.

| Field | Type | Notes |
|-------|------|-------|
| `images` | `dict[str, np.ndarray]` | Keys may **only** be `primary` / `wrist` / `left` / `right` |
| `state` | `np.ndarray \| None` | Required when `observation.needs_state: true` |
| `instruction` | `str` | Task language |
| `step` | `int` | Episode step |
| `extra` | `dict` | Backend-specific extras |

`act()` returns `ActionChunk(actions, space)`:

- `actions` is `(C, D)` `float32`
- `space` must equal `cfg.action_space`
- `D` must equal `dim_for_space(space)`

| `action_space` | `D` |
|----------------|-----|
| `delta_ee_7d` | 7 |
| `abs_ee_7d` | 7 |
| `abs_ee_16d` | 16 |
| `qpos_14d` | 14 |
| `hybrid_mobile_12d` | 12 |

Actions must already be in **environment units**. The harness `convert_action`
only does delta↔abs EE conversion. It never applies dataset q01/q99
unnormalization.

## Action-chunk semantics

`ActionChunkQueue.refill` keeps only `min(YAML chunk_size, returned_rows)` and
**discards the rest**. Returning a 16-row plan when `chunk_size: 8` executes 8
rows; the other 8 never run.

This example implements both modes with one `deque` (local and HTTP):

| YAML | What `act()` returns | Harness behaviour | Mirrors |
|------|----------------------|-------------------|---------|
| `replan_steps == chunk_size` | `(chunk_size, D)` | Open-loop block | `openvla_oft`, `starvla` |
| `replan_steps == 1` | `(1, D)` every call | Re-plans every `chunk_size` calls | `internvla_a1`, `fastwam` |

In the second mode, YAML `chunk_size` is **only** the internal planning
horizon. It does **not** make the harness run open-loop.

The HTTP adapter requests `horizon=cfg.chunk_size` once, then emits at most
`replan_steps` rows per `act()` from a local buffer — the same composition as
a remote WAM client.

## Per-backend observations

| Backend | `action_space` | `D` | Images actually delivered | Typical `state` |
|---------|----------------|-----|---------------------------|-----------------|
| `libero` | `delta_ee_7d` | 7 | `primary`, `wrist` | 8-D |
| `simplerenv` | `delta_ee_7d` | 7 | **`primary` only** | 7-D |
| `robotwin` | `qpos_14d` | 14 | `primary`, `left`, `right` | 14-D |

The SimplerEnv backend's `bind_run` discards `observation_images`. Declare
`per_backend.simplerenv.observation.images: [primary]` or `require_image(obs,
"wrist")` will raise.

RoboCasa forces `hybrid_mobile_12d` and is out of scope here. See
`configs/models/random_mobile.yaml` and `configs/models/xiaomi.yaml`.

## `per_backend` overlays

Allowed keys: `action_space`, `chunk_size`, `observation`, `extra`. Anything
else raises.

- `observation` and `extra` are **shallow-merged** (overlay keys win).
- `action_space` and `chunk_size` are **replaced wholesale**.

Top-level defaults in this example are LIBERO-oriented. SimplerEnv and RoboTwin
overrides live under `per_backend`.

## Remote / HTTP path

This example uses HTTP, not WebSocket: `build_transport`'s websocket path
requires `extra.ws_codec` ∈ `{openpi, starvla, lingbot, xiaomi}` — model-specific
codecs in `src/dynisafety/policies/transports/codecs/`. A generic toy websocket
server is not possible.

`configs/models/example_agent_http.yaml` requires a full `server:` block
(`env`, `root`, `launch`, `default_port`, `health_path`, `startup_timeout_s`)
plus `extra.act_timeout_s`, `extra.reset_path`, `extra.act_path`,
`extra.server_host`. `connect_transport` → `_build_http` **blocks** in
`wait_until_ready()`, polling `/health` until healthy or
`server.startup_timeout_s` elapses. The process must already be listening
before `from_config`.

Harness path (scheduler / `dynisafety serve` injects host and port; adapters
must never read `os.getenv("PORT")`):

```bash
dynisafety serve --model example_agent_http
# add --gpu N if discovery finds no GPU; --port to pin the listen port
```

Manual path:

```bash
python examples/example_agent/server/toy_vla_server.py \
  --port 8090 \
  --checkpoint examples/example_agent/assets/mock_vla_weights.npz
```

Then point the client at that socket (a live eval, not `--dry-run`):

```bash
dynisafety eval \
  --backend libero \
  --model example_agent_http \
  --suite safelibero_spatial \
  --level 0 \
  --task 0 \
  --set extra.server_host=127.0.0.1 \
  --set extra.server_port=8090
```

A sweep injects `HOST` / `PORT` as `extra.server_host` / `extra.server_port`
for you.

## Verification ladder

From the repository root:

```bash
ruff check examples tests/test_example_agent.py
python -m pytest tests/test_example_agent.py -q
python examples/example_agent/verify.py

dynisafety eval --backend libero --model example_agent --suite safelibero_spatial --level 0 --task 0 --dry-run
dynisafety eval --backend simplerenv --model example_agent --suite bridge --level 0 --task 0 --dry-run
dynisafety eval --backend robotwin --model example_agent --suite main20 --level 0 --task 0 --dry-run
dynisafety sweep --suite smoke_example_agent --dry-run
```

`dynisafety models` is a **stub** (`not implemented yet`). Inspect registration
with `available()`:

```bash
python -c "from pathlib import Path; from dynisafety.policies.registry import available; [print(s.name, s.state, s.detail) for s in available(configs_root=Path('configs')) if s.name.startswith('example_agent')]"
```

Two-stage signal (identical to `random` and every other in-tree model):

| When | `state` | Why |
|------|---------|-----|
| Before `pip install -e examples/example_agent` (adapter not importable) | `missing_dependency` | YAML is declared; the class cannot be imported |
| After install | `missing_checkpoint` | Catalog listing does not expand `${workspace}` |

`missing_checkpoint` after install is **expected**, not an error.

## Common mistakes

- Hardcoding ports, timeouts, or chunk sizes in Python. They live in YAML once;
  adapters read them with `require_extra` and **raise** when missing.
- Calling `requests.post` / `websockets.connect` / `zmq.Context`, or reading
  `os.getenv("PORT")`. Remote I/O goes through `connect_transport(cfg)`.
- Returning a 1-D array. Return `(C, D)`.
- Returning a `space` that disagrees with `cfg.action_space`.
- Expecting the harness to unnormalize q01/q99. It does not. Return environment
  units.
- `name:` not matching the YAML filename stem.
- Adding a fifth image key (only `primary` / `wrist` / `left` / `right`).
- Assuming `dynisafety models` lists the catalog. Use `available()` as above.

## File map

| Path | Role |
|------|------|
| `src/dyni_example_agent/local_adapter.py` | `transport: local` Policy |
| `src/dyni_example_agent/http_adapter.py` | `transport: http` Policy |
| `src/dyni_example_agent/mock_backbone.py` | Toy projector (not a real VLA) |
| `server/toy_vla_server.py` | Stdlib HTTP `/health` `/reset` `/act` |
| `assets/make_mock_weights.py` | Writes the committed `.npz` (`.pt` is gitignored) |
| `assets/mock_vla_weights.npz` | Toy weights |
| `verify.py` | One-shot PASS/FAIL table |
| `../../configs/models/example_agent.yaml` | Local model (stem = `name`) |
| `../../configs/models/example_agent_http.yaml` | HTTP model |
| `../../configs/suites/smoke_example_agent.yaml` | Smoke sweep cell |
| `../../tests/test_example_agent.py` | CI tests (no sim, no GPU) |
