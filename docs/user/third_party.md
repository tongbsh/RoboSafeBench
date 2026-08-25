# Third-party components (DyniSafety)

This document lists simulators and restricted assets that DyniSafety depends on:
upstream location, pinned commit, license, our patches, install entrypoints, and
intended runtime path keys.

Path placeholders follow M1 config resolution: `${workspace}`, `${third_party}`
(`<workspace>/third_party`). Do not hard-code machine-absolute paths in YAML.

**Install (user guide):** see **[installation.md](installation.md)** — canonical
steps for conda, `make install-*`, extras mutex, MANO, Docker, and
troubleshooting.

Short pin-check reminder (after submodule init):

```bash
scripts/setup/apply_patches.sh --check all   # exit 2 => MISSING or DRIFTED
scripts/setup/install_backend.sh <backend> # or: make install-libero|simplerenv|robotwin|robocasa
dynisafety doctor --backend <backend>
```

Conda envs: `environment/dynisafety-{libero,simplerenv,robotwin,robocasa}.yml`
(historical aliases and model **server** envs: `environment/README.md`).
Docker (libero only): root `Dockerfile` + `docker/README.md`.
This document remains the **pin / license / patch** authority; it is not a
second install manual.

License texts / notices: see [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)
(coverage = **core simulators + MANO + RoboCasa/robosuite**).

---

## Summary

| Component | Layout | Pin | License | Install |
|-----------|--------|-----|---------|---------|
| SimplerEnv-OpenVLA | submodule `third_party/SimplerEnv-OpenVLA` | `ccfe3809766839a2fcfb7a3d3c9abff585189188` | MIT | `scripts/setup/install_simplerenv.sh` |
| ManiSkill2_real2sim | nested under SimplerEnv | `91d154bfd864577f8d2e80f3fc2f8b4d9df9ae5c` | Apache-2.0 | via SimplerEnv + `apply_patches.sh` |
| RoboTwin | submodule `third_party/RoboTwin` (private mirror, D1=C) | `afd5f1bbc5759ac1512dc42b3cdaf8a218fb7c00` (`dynisafety-pin-20260806`) | MIT | `scripts/setup/install_robotwin.sh` |
| dyniSafety_libero (SafeLIBERO) | **not** a submodule (D2=B) | install tree; upstream ref `8f1084e3132a39270c3a13ebe37270a43ece2a01` | MIT | `scripts/setup/install_libero.sh` |
| robosuite | **not** a submodule; `third_party/robosuite` | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` (git **master**, ≥1.5; not PyPI 1.4) | MIT | `scripts/setup/install_robocasa.sh` |
| RoboCasa | **not** a submodule; `third_party/robocasa` | `921c9a5736a8d0ea5589657898aadcfa55a6a195` | MIT (code); CC BY 4.0 (assets) | `scripts/setup/install_robocasa.sh` |
| MANO / SMPL+H | **not distributed** | n/a | MPI/MPG non-commercial | `scripts/setup/download_assets.py --mano-notice` |

Deprecated: `simulator_env/safelibero/` (D15=A). Use `dyniSafety_libero` only.

---

## 1. SimplerEnv-OpenVLA

| Field | Value |
|-------|-------|
| Upstream | `git@github.com:DelinQu/SimplerEnv-OpenVLA.git` (fork of simpler-env / OpenVLA SimplerEnv) |
| Path | `${third_party}/SimplerEnv-OpenVLA` |
| Pin | `ccfe3809766839a2fcfb7a3d3c9abff585189188` (`gr00t support action queue`) |
| License | MIT — `third_party/SimplerEnv-OpenVLA/LICENSE` |
| Runtime key | intended YAML: `extra.simplerenv.sim_root: ${third_party}/SimplerEnv-OpenVLA` (wiring in a later batch) |
| Install | `scripts/setup/install_simplerenv.sh` then `scripts/setup/apply_patches.sh simplerenv` |

### Our changes (`patches/simplerenv/`)

| Patch | Why |
|-------|-----|
| `requirements_json_numpy.patch` | HTTP VLA clients need `json_numpy` |
| `maniskill2_evaluator_optional_visualize.patch` | Guard missing `visualize_epoch` on some policies |
| `openpi_websocket_host_and_obs_keys.patch` | Connect to `127.0.0.1`; LiberoInputs observation key names |
| `maniskill2_submodule_sha.patch` | **Not** a unified diff — pins nested ManiSkill2 via `git checkout` (D5=A) |

Not patched (D4=A): local `main_inference.py` / `scripts/run_*.sh` edits; untracked
`simpler_env/policies/{dreamvla,…}` (policy logic lives under
`src/dynisafety/policies/adapters/`).

---

## 2. ManiSkill2_real2sim

| Field | Value |
|-------|-------|
| Upstream | `https://github.com/allenzren/ManiSkill2_real2sim.git` (as recorded in SimplerEnv `.gitmodules`) |
| Path | `${third_party}/SimplerEnv-OpenVLA/ManiSkill2_real2sim` |
| Pin | `91d154bfd864577f8d2e80f3fc2f8b4d9df9ae5c` (D5=A; upstream SimplerEnv@`ccfe380` records `cd45dd2`) |
| License | Apache-2.0 — `…/ManiSkill2_real2sim/LICENSE` |
| Apply | `scripts/setup/apply_patches.sh simplerenv` (checkout helper) |

---

## 3. RoboTwin (private mirror)

| Field | Value |
|-------|-------|
| Upstream lineage | Based on [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin); **this pin is a DyniSafety private mirror** (D1=C), not a pure upstream SHA |
| Path | `${third_party}/RoboTwin` |
| Pin | `afd5f1bbc5759ac1512dc42b3cdaf8a218fb7c00` / tag `dynisafety-pin-20260806` |
| Mirror | Local bare repo `mirrors/RoboTwin.git` (gitignored). `.gitmodules` may use a **machine-local absolute** `url` until a private host is configured — see `scripts/setup/install_robotwin.sh` |
| License | MIT — `third_party/RoboTwin/LICENSE` |
| Runtime key | intended YAML: `extra.robotwin.sim_root: ${third_party}/RoboTwin` |
| Install | `scripts/setup/install_robotwin.sh` then `scripts/setup/apply_patches.sh robotwin` |
| Assets | Large `assets/` (~16GB) are **not** in the submodule; obtain separately (do not commit) |

### Our changes (`patches/robotwin/`)

| Patch | Why |
|-------|-----|
| `dynisafety_hooks.patch` | Minimal `Base_Task` hooks (`enable`/`disable`/`before_step`/`after_step`/`update_info`/close); business logic in `dynisafety.backends.robotwin.hooks` |

Not in patch (D3=A): per-task `get_dynisafety_*` helpers — M7 runtime wrapping.

Pin tree excludes `envs/dynisafety/` (migrated into `src/dynisafety/backends/robotwin/`).
Vendored `dynamic_RoboTwin/RoboTwin/` remains tracked until M9 (D10=B).

---

## 4. LIBERO / dyniSafety_libero (SafeLIBERO)

| Field | Value |
|-------|-------|
| Upstream | `https://github.com/Lifelong-Robot-Learning/LIBERO.git` |
| Path | `${third_party}/dyniSafety_libero` (created by install; **not** listed in `.gitmodules`, D2=B) |
| Pin / source | Prefer local `simulator_env/dyniSafety_libero` or `DYNISAFETY_LIBERO_SRC`; else clone upstream @ `8f1084e3132a39270c3a13ebe37270a43ece2a01` (may lack DyniSafety BDDL/assets) |
| License | MIT — upstream / fork `LICENSE` |
| Runtime key | intended YAML: `extra.libero.sim_root: ${third_party}/dyniSafety_libero` (replaces legacy `DYNISAFETY_SIMENV`, D7=A) |
| Install | `scripts/setup/install_libero.sh` |
| Patches | none (`patches/libero/` empty by design) |

`simulator_env/safelibero/` is **deprecated** (D15=A).

---

## 5. robosuite (RoboCasa backend)

| Field | Value |
|-------|-------|
| Upstream | `https://github.com/ARISE-Initiative/robosuite.git` |
| Path | `${third_party}/robosuite` (**not** a git submodule this window) |
| Pin | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` (git **master**; package version 1.5.2) |
| License | MIT — `third_party/robosuite/LICENSE` |
| Install | `scripts/setup/install_robocasa.sh` then `pip install -e` this tree |
| Forbidden | PyPI `robosuite==1.4.*` — RoboCasa asserts robosuite ≥ 1.5.2 |

Clone may use `https://ghfast.top/https://github.com/ARISE-Initiative/robosuite.git` when GitHub TLS fails. Override with `DYNISAFETY_ROBOSUITE_URL`.

---

## 6. RoboCasa

| Field | Value |
|-------|-------|
| Upstream | `https://github.com/robocasa/robocasa.git` |
| Path | `${third_party}/robocasa` (**not** a git submodule this window) |
| Pin | `921c9a5736a8d0ea5589657898aadcfa55a6a195` (package version 1.0.1) |
| License (code) | MIT — `third_party/robocasa/LICENSE` |
| License (assets / datasets) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Runtime key | YAML: `extra.robocasa.sim_root: ${third_party}/robocasa` |
| Install | `scripts/setup/install_robocasa.sh` |
| Conda env | `dynisafety-robocasa` (Python 3.11 only) |
| Official pins | `numpy==2.2.5`, `mujoco==3.3.1` (`import robocasa` asserts both) |
| NumPy mutex | Do **not** install into `dynisafety-libero` / `simplerenv` / `robotwin`. Do **not** remove `numpy>=1.26,<2.0` from `pyproject.toml`. This env uses `pip install -e . --no-deps` then `requirements/robocasa.txt`. |
| Assets (default) | lightweight: `tex`, `tex_generative`, `fixtures_lw`, `objs_lw`. `dynisafety doctor --backend robocasa` **FAIL**s if they are missing (FIX: `install_robocasa.sh`). |
| Assets (full ~10GB) | only if `DYNISAFETY_ROBOCASA_FULL_ASSETS=1` |
| Asset host | Official UTexas Box `shared/static/*.zip` links in `box_links_assets.json` currently **404**. `install_robocasa.sh` falls back to Hugging Face mirrors (`Livfour/robocasa-assets`, `IIFAN/robocasa365-lightwheel-*`). If `huggingface.co` times out, it retries `https://hf-mirror.com`. |
| Headless | `export MUJOCO_GL=egl` |
| Patches | none (`patches/robocasa/` not used) |
| Eval | kitchen **wheeled** navigation (`NavigateKitchen`, suite `robocasa_nav`) + floor disturbances L0–L3. Not countertop PickPlace, not bipedal. **Not** in `paper_main`. Camera `robot0_agentview_center` looks at the stove — spawn is `body z = floor`. |

Clone may use `https://ghfast.top/https://github.com/robocasa/robocasa.git`. Override with `DYNISAFETY_ROBOCASA_URL`.

---

## 7. MANO / SMPL+H (not redistributed)

| Field | Value |
|-------|-------|
| Owner | Max Planck Institute for Intelligent Systems / Max-Planck-Gesellschaft |
| License | Proprietary non-commercial research terms — see `THIRD_PARTY_NOTICES.md` |
| Distribution | **Must not** ship in this repository or releases |
| Obtain | Accept official MANO terms yourself; place under `<workspace>/assets/mano_v1_2/` (gitignored) |
| Helper | `python scripts/setup/download_assets.py --mano-notice` |

Note: RoboTwin L3 default hand mesh in DyniSafety is the small `hand_pusher` asset
(see backend YAML / `hand_objects`), not MANO. RoboCasa L3 default pedestrian
visuals are original in-tree OBJ parts under
`src/dynisafety/backends/robocasa/assets/pedestrian/`, not SMPL.

---

## 8. Checkpoints & other downloads

Use `scripts/setup/download_assets.py` (canonical). Root `download_hf.py` is a
**deprecated wrapper** that forwards here (no hard-coded machine paths):

- `--list` — catalog
- `--asset NAME` — download under `--dest` or `DYNISAFETY_CHECKPOINTS` or `<workspace>/download_checkpoints`
- `--mano-notice` — MANO instructions only (no binary fetch)

Individual HF model/dataset licenses are those of their respective repos; they are
outside D6=A notices scope.

---

## 9. Verify pins & patches

```bash
git submodule status
# expect:
#   afd5f1b… third_party/RoboTwin (dynisafety-pin-20260806)
#   ccfe380… third_party/SimplerEnv-OpenVLA

# RoboCasa trees are not submodules:
git -C third_party/robosuite rev-parse HEAD
# expect 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
git -C third_party/robocasa rev-parse HEAD
# expect 921c9a5736a8d0ea5589657898aadcfa55a6a195

scripts/setup/apply_patches.sh --check all
# exit 0 = all APPLIED; exit 2 = MISSING or DRIFTED
```

`dynisafety doctor --backend <backend>` is the full environment gate (M10):
submodule pins, `apply_patches.sh --check`, simulator imports, assets,
checkpoints, conda env names, GPU. Optional gaps are WARN unless `--strict`.
