# Adding a VLA model to DyniSafety

Goal: add a new model by writing **one adapter file** and **one YAML**, then
verify with `dynisafety models` and a smoke eval (no new shell entrypoints).

A complete out-of-tree example (adapter + YAML + tests, no `src/` edits) is
[examples/example_agent/README.md](../../examples/example_agent/README.md).

## Prerequisites

- Editable install for one backend: `pip install -e ".[<backend>]"` — see
  [installation.md](installation.md) (extras are mutually exclusive).
- Read `docs/user/contracts.md` §4 (Policy protocol) — do not invent a second signature.
- Iron law 1: every hyperparameter default lives in YAML once. Adapters and CLI
  must not hardcode ports, chunk sizes, or timeouts.
- Confirm the eval loop / backend with `dynisafety doctor --backend <b>` before
  live runs.

## Steps

### 1. Copy the template

```bash
cp templates/model_adapter_template.py \
   src/dynisafety/policies/adapters/my_model.py
```

Rename `TemplateAdapter` → `MyModelAdapter`. Implement:

| Method | Responsibility |
|--------|----------------|
| `from_config(cfg)` | Validate `cfg.transport`; connect remote via `connect_transport(cfg)` |
| `reset(ctx)` | Clear episode state; optional `transport.reset` |
| `act(obs)` | Read `Observation.images` / `state`; return `ActionChunk` |
| `close()` | Release sockets / GPU |

**Forbidden in adapters**

- `requests.post` / `websockets.connect` / `zmq.Context`
- `task_env.take_action(...)`
- `conda activate` / hardcoded ports

Remote models must call `dynisafety.policies.adapters._common.connect_transport`.

### 2. Write `configs/models/my_model.yaml`

Use the frozen layout (see `configs/models/starvla.yaml` or `random.yaml`):

```yaml
name: my_model
transport: local          # or http | websocket | zmq
adapter: dynisafety.policies.adapters.my_model:MyModelAdapter
action_space: delta_ee_7d # or abs_ee_7d | abs_ee_16d | qpos_14d | hybrid_mobile_12d
chunk_size: 8
observation:
  images: [primary, wrist]
  needs_state: true
# server:                 # required when transport != local
#   env: myenv
#   root: ${env:MY_ROOT|${workspace}/third_party/my_model}
#   launch: serve.py
#   default_port: 8000
#   health_path: /health
#   startup_timeout_s: 600
#   args: ["--ckpt", "{checkpoint}", "--port", "{port}"]
checkpoint:
  default: "${env:MY_CKPT|${workspace}/checkpoints/my_model}"
resources:
  max_jobs_per_gpu: 2
backends: [libero]        # subset of libero | simplerenv | robotwin | robocasa
extra:
  # model-specific required keys — no silent defaults in Python
  seed: 0                 # example for RandomAdapter-style local models
```

HTTP transports also need under `extra`: `act_timeout_s`, `reset_path`,
`act_path`, `server_host`. WebSocket needs `ws_codec` (`openpi`|`starvla`|`lingbot`)
and codec-specific keys (see existing YAMLs).

`server.args` is the argv tail handed to `python <root>/<launch>`; the process runs
with `cwd` set to `root`. Always available: `{port}` `{gpu}` `{checkpoint}` `{model}`
`{suite}`. Any other placeholder must be supplied per suite:

```yaml
server:
  args: ["--ckpt", "{checkpoint}", "--unnorm_key", "{unnorm_key}", "--port", "{port}"]
  per_suite:
    safelibero_object: {unnorm_key: libero_object_no_noops}
    bridge:            {unnorm_key: bridge_oxe, args: [...]}   # `args` overrides the tail
```

Use this whenever the server loads suite-specific weights: the checkpoint is read at
startup, so `dynisafety serve` requires `--suite` and a sweep starts one server
generation per suite. Unresolved placeholders are reported at config-load time, not
after the model has loaded.

### 3. List models

```bash
dynisafety models
```

Your stem must appear. States:

| state | Meaning |
|-------|---------|
| `ready` | Adapter imports and checkpoint path looks OK |
| `missing_dependency` | Import failed (package not installed) |
| `missing_checkpoint` | Local checkpoint path missing |
| `load_error` | YAML/config/adapter error (detail string) |

Listing always prints every declared YAML (including failures). There is no
`memoryvla` entry by design.

### 4. Smoke with the random reference model

Before wiring heavy weights, confirm the loop accepts a Policy:

```bash
dynisafety eval \
  --backend libero \
  --model random \
  --suite safelibero_spatial \
  --level 0 \
  --dry-run
```

(`--dry-run` prints the plan without starting the simulator.)

### 5. End-to-end example (recommended)

**A — Dry-run (no simulator):** after adding `configs/models/my_model.yaml` with
`backends` including `libero` (or your target):

```bash
dynisafety eval \
  --backend libero \
  --model my_model \
  --suite safelibero_spatial \
  --level 0 \
  --dry-run

dynisafety sweep --suite smoke --only model=my_model --dry-run
```

**B — Live smoke (needs install + checkpoint):** backends are registered for
`libero` / `simplerenv` / `robotwin` / `robocasa`. Prefer a local
`transport: local` fake or `random` / `random_mobile` first, then your weights:

```bash
dynisafety doctor --backend libero
dynisafety sweep --suite smoke
# RoboCasa kitchen-nav (numpy 2 env; not paper_main):
# dynisafety doctor --backend robocasa
# dynisafety eval --backend robocasa --model random_mobile --suite robocasa_nav --level 0 --dry-run
# dynisafety sweep --suite smoke_robocasa --dry-run
# or only your model, once it is listed in a suite matrix / --only filter:
# dynisafety sweep --suite smoke --only model=my_model
```

If live smoke fails on scheduler / env naming issues, fix the adapter or model
YAML — do not add shell wrappers as a workaround. (Historical cross-module notes
live under `old_cache/docs_internal/` on developer machines.)

### 6. Checklist before PR

- [ ] `configs/models/<id>.yaml` uses the frozen keys only
- [ ] `adapter:` import path resolves
- [ ] `grep -rn "requests.post\\|websockets.connect\\|zmq.Context" src/dynisafety/policies/adapters/` is empty for your file
- [ ] No new shell user-entry script
- [ ] `pytest tests/test_policies.py -q` passes
- [ ] LingBot-like servers: `extra.dump_artifacts: false` (never leave debug dumps on)

## Reference adapters

| Need | Example |
|------|---------|
| Local OFT | `adapters/openvla_oft.py` + `configs/models/openvla_oft.yaml` |
| HTTP | `adapters/univla.py` + `univla.yaml` |
| WebSocket + codec | `adapters/pi0.py` / `starvla.py` / `lingbot_va.py` |
| ZMQ (GR00T) | `adapters/gr00t.py` |
| Shared class + variant | `adapters/xvla.py` (`extra.variant`) |
| Smoke / tutorial | `adapters/random.py` (`random`, `random_mobile`) |
| Out-of-tree example | `examples/example_agent/` ([README](../../examples/example_agent/README.md)) |
