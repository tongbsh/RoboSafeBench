# DyniSafety-RoboTwin

The RoboTwin variant of the **DyniSafety** benchmark — same dynamic
disturbance levels as
[`dynamic_SimplerEnv/`](../dynamic_SimplerEnv/) and
[`dynisafety_libero/`](../dynisafety_libero/), but built on top of the
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) bimanual SAPIEN
simulator.

This package is **standalone**: the bundled RoboTwin checkout lives under
[`RoboTwin/`](RoboTwin/) and the dynisafety injection code is the
canonical implementation under
[`RoboTwin/envs/dynisafety/`](RoboTwin/envs/dynisafety/). The
[`dynisafety/`](dynisafety/) folder at this level is just a re-export
shell so external code can write `from dynisafety import …` without
caring about the underlying layout.

## Dynamic Disturbance Levels

| Level | Name              | Description |
|-------|-------------------|-------------|
| 0     | Baseline          | No dynamic disturbance — identical to upstream RoboTwin |
| 1     | Sudden Drop       | An obstacle appears on the EEF→target line when the end-effector approaches |
| 2     | Sliding Obstacle  | An obstacle slides along an arc trajectory into the workspace |
| 3     | Shared Reaching   | A simulated human hand reaches toward the workspace |

## Repository Layout

```
dynamic_RoboTwin/
├── README.md
├── main_inference.py          # Standalone entry point (policy registry + episode loop)
├── RoboTwin/                  # Bundled RoboTwin checkout (envs, policy, task_config, …)
│   └── envs/dynisafety/       # Canonical dynisafety implementation
├── dynisafety/                # Top-level re-export shell
├── policies/                  # VLA model adapters
│   ├── base_policy.py
│   ├── internvla_a1_policy.py # In-process InternVLA-A1 (lerobot stack)
│   ├── lingbot_va_policy.py   # Websocket client to wan_va_server
│   └── lingbot_client/        # Vendored msgpack/websocket helpers
└── scripts/
    ├── _common.sh             # Conda + path bootstrap shared by all launchers
    ├── run_dynisafety.sh      # Top-level (model, level, task) dispatcher
    ├── internvla_a1/
    │   ├── eval_place_a2b_right_common.sh
    │   └── eval_place_a2b_right_level{0,1,2,3}.sh
    └── lingbot_va/
        ├── launch_server.sh   # Run the wan_va_server in `lingbot_va` env
        ├── eval_place_a2b_right_common.sh
        └── eval_place_a2b_right_level{0,1,2,3}.sh
```

## Installed conda envs (this workspace)

| Conda env       | Purpose                                                      |
|-----------------|--------------------------------------------------------------|
| `internvla_a1`  | RoboTwin runtime (sapien, curobo, lerobot). Hosts the env loop, the InternVLA-A1 in-process policy AND the LingBot-VA websocket *client*. |
| `lingbot_va`    | LingBot-VA wan_va inference *server* only (no RoboTwin).      |

`nvidia_curobo` is installed in editable mode pointing to
`dynamic_RoboTwin/RoboTwin/envs/curobo`. If you ever move this package,
re-run::

    pip install -e dynamic_RoboTwin/RoboTwin/envs/curobo --no-deps --no-build-isolation

inside the `internvla_a1` env to refresh the editable path.

## Quick start

### InternVLA-A1 (in-process)

```bash
bash dynamic_RoboTwin/scripts/run_dynisafety.sh \
    --model internvla_a1 --level 0 --task place_a2b_right
```

The launcher activates `internvla_a1`, resolves the checkpoint
(falls back to the HF repo `InternRobotics/InternVLA-A1-3B-RoboTwin` when
no local copy exists), and runs `main_inference.py` with
`--policy_name internvla_a1`.

### LingBot-VA (client / server, two terminals)

```bash
# Terminal 1 — start the wan_va_server in the lingbot_va env
bash dynamic_RoboTwin/scripts/lingbot_va/launch_server.sh

# Terminal 2 — once the server logs "Starting websocket server on …",
# launch the client in the internvla_a1 env
bash dynamic_RoboTwin/scripts/run_dynisafety.sh \
    --model lingbot_va --level 0 --task place_a2b_right
```

Use `--level all` to iterate over levels 1, 2, 3 (excludes level 0
baseline, exactly like the SimplerEnv variant).

## Adding a new VLA model

1. Create `policies/my_model_policy.py` implementing
   `RoboTwinPolicy.load`, `reset`, `step`.
2. Decorate the class with `@register_policy("my_model")` (or call
   `register_policy("my_model", MyModelPolicy)` at module load time).
3. Append `"policies.my_model_policy"` to the `_BUILTIN` list in
   [`policies/__init__.py`](policies/__init__.py).
4. Drop a `scripts/my_model/` folder mirroring `internvla_a1/` so the
   `run_dynisafety.sh` dispatcher picks it up automatically.
