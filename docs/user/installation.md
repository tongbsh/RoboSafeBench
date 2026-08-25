# Installing DyniSafety

This is the **canonical install guide** for the simulation backends. Pin / license /
upstream details stay in [third_party.md](third_party.md) and
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

**Hard rule:** install **exactly one** backend into a given Python/conda
environment. SimplerEnv needs SAPIEN **2.2.2**; RoboTwin needs SAPIEN
**3.0.0b1**; LIBERO uses MuJoCo + NumPy 1.x; RoboCasa uses MuJoCo **3.3.1** +
**NumPy 2.2.5** — they must not share an env. Do **not** drop the
`numpy>=1.26,<2.0` bound in `pyproject.toml` (it protects the other backends).

Reported metrics are simulator-internal; see the README disclaimer and
[metrics.md](metrics.md).

---

## Prerequisites

| Item | Notes |
|------|--------|
| OS | Linux x86_64 (CUDA GPU recommended for live VLA eval) |
| Git | With submodule support |
| Conda / Mamba | Recommended; ymls under `environment/` |
| Python | ≥ 3.10 (yml targets 3.10; RoboCasa env is **3.11**) |
| Disk | Simulators + checkpoints are large; keep weights **outside** git |

Do **not** commit checkpoints, videos, API keys, or MANO binaries. `/assets/` is
gitignored; obtain MANO yourself (below).

---

## Clone

```bash
git clone <THIS_REPO_URL> dynisafety   # use the public / anonymous URL for review
cd dynisafety
```

Submodules (SimplerEnv, RoboTwin) are initialized by the install scripts when
needed. You can also:

```bash
git submodule update --init --recursive
```

RoboTwin may use a **private mirror** URL in `.gitmodules`. If `git submodule
update` fails, follow [third_party.md](third_party.md) § RoboTwin and
`scripts/setup/install_robotwin.sh` (vendored / mirror bootstrap).

---

## One-command backend install

From the repo root, pick **one**:

```bash
make install-libero
# make install-simplerenv
# make install-robotwin
# make install-robocasa   # separate numpy-2 env; see RoboCasa section
```

Equivalent:

```bash
scripts/setup/install_backend.sh libero        # or simplerenv | robotwin | robocasa
# flags: --skip-pip | --skip-patch
```

`install_backend.sh` roughly:

1. `scripts/setup/install_<backend>.sh` — tree / submodule sync  
2. `scripts/setup/apply_patches.sh <backend>` — unless `--skip-patch` (RoboCasa has no patches)  
3. `pip install -e ".[<backend>]"` — unless `--skip-pip`  
   For **robocasa**: `pip install -e . --no-deps` then `pip install -r requirements/robocasa.txt`  
   (plain `pip install -e .` would pull `numpy<2` from `pyproject.toml`)  
4. Prints next step: `dynisafety doctor --backend <backend>`

Preferred conda env names (`environment/README.md`):

| Backend | Conda env | Pip extra | Pins |
|---------|-----------|-----------|------|
| LIBERO | `dynisafety-libero` | `.[libero]` | `requirements/libero.txt` |
| SimplerEnv | `dynisafety-simplerenv` | `.[simplerenv]` | `requirements/simplerenv.txt` |
| RoboTwin | `dynisafety-robotwin` | `.[robotwin]` | `requirements/robotwin.txt` |
| RoboCasa | `dynisafety-robocasa` | **`--no-deps`** + `requirements/robocasa.txt` | `numpy==2.2.5`, `mujoco==3.3.1` |

Manual skeleton (if you prefer not to use Make):

```bash
conda env create -f environment/dynisafety-<backend>.yml
conda activate dynisafety-<backend>
# Install torch+CUDA for your GPU first — see requirements/README.md
pip install -r requirements/<backend>.txt
scripts/setup/install_backend.sh <backend> --skip-pip   # tree + patches only
pip install -e ".[<backend>]"
```

**Forbidden:**

```bash
pip install -e ".[libero,simplerenv,robotwin]"   # do not combine extras
# do not install RoboCasa into dynisafety-libero / simplerenv / robotwin
# do not: pip install -e .   inside dynisafety-robocasa (numpy<2 from pyproject.toml)
```

---

## Per-backend notes

### LIBERO (`libero`)

- Simulator stack: MuJoCo / robosuite / dyniSafety_libero (SafeLIBERO).  
- **Not** a git submodule: installed via `scripts/setup/install_libero.sh` into the
  configured tree (see [third_party.md](third_party.md)).  
- Deprecated path: `simulator_env/safelibero/` — do not use.  
- Docker (optional): root `Dockerfile` + [docker/README.md](../../docker/README.md).

```bash
make install-libero
dynisafety doctor --backend libero
dynisafety sweep --suite smoke --dry-run
```

### SimplerEnv (`simplerenv`)

- Submodule: `third_party/SimplerEnv-OpenVLA` (+ nested ManiSkill2).  
- **SAPIEN 2.2.2** — conflicts with RoboTwin’s SAPIEN 3.  
- After sync: `scripts/setup/apply_patches.sh simplerenv` (or rely on
  `install_backend.sh`). Check drift:

```bash
scripts/setup/apply_patches.sh --check simplerenv
# exit 0 = OK; exit 2 = MISSING or DRIFTED
```

- Bridge suite uses **`runtime.max_steps: 120`** for all models
  (`configs/backends/simplerenv/suites/bridge.yaml`). Do not compare L0 SR to
  published 60-step SimplerEnv numbers without disclosing the budget.

```bash
make install-simplerenv
dynisafety doctor --backend simplerenv
```

### RoboTwin (`robotwin`)

- Submodule / mirror: `third_party/RoboTwin` (pin tag documented in
  [third_party.md](third_party.md)).  
- **SAPIEN 3.0.0b1**.  
- If submodule URL points at a private mirror, clone/bootstrap per
  `scripts/setup/install_robotwin.sh`.  
- Apply hooks: `scripts/setup/apply_patches.sh robotwin`.

```bash
make install-robotwin
dynisafety doctor --backend robotwin
```

### RoboCasa (`robocasa`)

- Simulator stack: MuJoCo **3.3.1** / robosuite **git master (≥1.5, not PyPI 1.4)** /
  RoboCasa kitchen. Official pins: `numpy==2.2.5`, `mujoco==3.3.1`.
- **Dedicated conda env** `dynisafety-robocasa`, **Python 3.11**. NumPy 2 is
  mutually exclusive with LIBERO / SimplerEnv / RoboTwin (`pyproject.toml`
  keeps `numpy>=1.26,<2.0` for those backends — do not remove that bound).
- Trees: `third_party/robosuite` and `third_party/robocasa` via
  `scripts/setup/install_robocasa.sh` (`pip install -e` those trees). **Not** git
  submodules; incomplete clones may be completed via `ghfast.top`.
- Install DyniSafety with `pip install -e . --no-deps` then
  `pip install -r requirements/robocasa.txt` (`make install-robocasa` does this).
- Kitchen assets default to the **lightweight** set: `tex`, `tex_generative`,
  `fixtures_lw`, `objs_lw`. The full ~10GB registry is only downloaded when
  `DYNISAFETY_ROBOCASA_FULL_ASSETS=1`. Official Box links may 404;
  `install_robocasa.sh` then uses Hugging Face mirrors (`HF_ENDPOINT` or
  `https://hf-mirror.com`).
- Licenses: code MIT; kitchen assets / datasets **CC BY 4.0**.
- Headless rendering: `export MUJOCO_GL=egl`.
- **Eval intent:** kitchen **wheeled** navigation (`NavigateKitchen`, PandaOmron)
  with **dynamic floor obstacles**. Not countertop PickPlace, not bipedal
  walking. First suite: `robocasa_nav` (six destinations). Among RoboCasa’s 65
  atomic tasks, only NavigateKitchen is wired. **Not** in
  `configs/suites/paper_main.yaml`.
- L1 = frontal **floor** occupancy; L2 = tall floor box slides in from beside
  the aisle onto a path landing; L3 = clothed pedestrian proxy on the floor then clear. Geometry
  formulas are locked in YAML comments; numbers live in
  `configs/backends/robocasa/levels/l{1,2,3}.yaml`. L1/L2 collision stays the YAML
  box; visuals rotate from `disturbance.obstacle.pool`.
- Camera `robot0_agentview_center` looks down at the stove — do **not** use that
  image alone to decide whether L1–L3 spawned on the counter or the floor; use
  body **z = floor**.
- `dynisafety doctor --backend robocasa` **FAIL**s (not warn-and-pass) if
  lightweight assets `tex` / `tex_generative` / `fixtures_lw` / `objs_lw` are
  missing; FIX is `bash scripts/setup/install_robocasa.sh`. Also requires import
  `robocasa` + `robosuite` + MuJoCo **3.3.x** in conda `dynisafety-robocasa`.

```bash
conda env create -f environment/dynisafety-robocasa.yml
conda activate dynisafety-robocasa
export MUJOCO_GL=egl
make install-robocasa
dynisafety doctor --backend robocasa
dynisafety sweep --suite smoke_robocasa --dry-run
# L0–L3 (env's dynisafety; numpy 2 — do not use another env's CLI):
dynisafety eval --backend robocasa --model random_mobile --suite robocasa_nav --level 0 --task 0 --set runtime.episodes=1
dynisafety eval --backend robocasa --model random_mobile --suite robocasa_nav --level 1 --task 0 --set runtime.episodes=1
dynisafety eval --backend robocasa --model random_mobile --suite robocasa_nav --level 2 --task 0 --set runtime.episodes=1
dynisafety eval --backend robocasa --model random_mobile --suite robocasa_nav --level 3 --task 0 --set runtime.episodes=1
```

---

## Docker (LIBERO only)

```bash
docker build -t dynisafety-libero:latest -f Dockerfile .
docker run --rm --gpus all -it dynisafety-libero:latest
```

Details, mounts, and doctor caveats: [docker/README.md](../../docker/README.md).
SimplerEnv / RoboTwin images are not shipped; use conda + `make install-*`.

---

## Checkpoints and MANO

### Checkpoints

Not in git. Download with:

```bash
python scripts/setup/download_assets.py --help
# Destination order: --dest DIR → $DYNISAFETY_CHECKPOINTS → <workspace>/download_checkpoints
```

The legacy root `download_hf.py` is deprecated (wrapper → `download_assets.py` in
the docs pass). Prefer `scripts/setup/download_assets.py` in all new instructions.

### MANO / SMPL+H (L3 hand meshes)

**Not redistributed.** Accept the official MPI/MPG terms, then place files under
`<workspace>/assets/mano_v1_2/` (gitignored). Guidance only:

```bash
python scripts/setup/download_assets.py --mano-notice
```

See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) § MANO.

---

## Verify

```bash
dynisafety doctor --backend <backend>    # non-zero exit if checks fail; each FAIL prints a fix hint
dynisafety sweep --suite smoke --dry-run # plan only (LIBERO cell)
dynisafety sweep --suite smoke_robocasa --dry-run  # RoboCasa kitchen-nav cell
# live (needs ckpt + sim): dynisafety sweep --suite smoke
```

Smoke matrix (`configs/suites/smoke.yaml`): `libero` × `safelibero_spatial` ×
task `0` × level `0` × `openvla_oft` × 1 episode.

RoboCasa smoke (`configs/suites/smoke_robocasa.yaml`): `robocasa` ×
`robocasa_nav` × task `0` × level `0` × `random_mobile` × 1 episode. Not part of
`paper_main`.

Dev checks (no sim marker):

```bash
make test    # pytest -m "not sim"
# make lint  # may fail on pre-existing ruff debt — see project todos
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `pip` / import fights between backends | Combined extras or shared env | Fresh conda env; one of `.[libero]` / `.[simplerenv]` / `.[robotwin]` only — RoboCasa is `dynisafety-robocasa` + `--no-deps` |
| RoboCasa `numpy` / `mujoco` assert | Installed into a NumPy-1 env or `pip install -e .` | Recreate `dynisafety-robocasa`; `pip install -e . --no-deps`; pins `numpy==2.2.5`, `mujoco==3.3.1` |
| RoboCasa doctor FAIL on `asset_robocasa_*` | Lightweight kitchen zips not unpacked | `bash scripts/setup/install_robocasa.sh` (not a warning) |
| Headless MuJoCo / EGL | `MUJOCO_GL` unset on a GPU box | `export MUJOCO_GL=egl` |
| SAPIEN errors after switching backend | sapien 2 vs 3 | Never share envs; recreate `dynisafety-simplerenv` or `dynisafety-robotwin` |
| `apply_patches.sh --check` exit 2 | Patches MISSING or DRIFTED | Re-run `scripts/setup/apply_patches.sh <backend>`; do not hand-edit patched upstream files |
| Submodule empty / wrong commit | Not initialized or pin drift | `scripts/setup/install_<backend>.sh` / `git submodule update --init --recursive` |
| RoboTwin submodule URL fails | Private mirror | Follow `install_robotwin.sh` / third_party mirror notes |
| `doctor` conda FAIL on model server names | YAML `serve_env` vs disk env name drift | Use `dynisafety-{backend}` for the sim env; server envs come from upstream (see `environment/README.md`) |
| MANO / hand mesh missing | Restricted asset | `--mano-notice`; place under `assets/mano_v1_2/` |
| Want old bash entrypoints | Removed or legacy | [migration.md](migration.md) → `dynisafety eval` / `sweep` |

Path placeholders in YAML: `${workspace}`, `${third_party}`, `${env:VAR}` — do
not hard-code machine-absolute paths in configs you commit.

---

## Related docs

| Doc | Role |
|-----|------|
| [configuration.md](configuration.md) | Config merge order, `config print` |
| [migration.md](migration.md) | Old shell → new CLI |
| [reproducing_paper.md](reproducing_paper.md) | Paper-main sweep + report |
| [adding_a_model.md](adding_a_model.md) | New VLA adapter |
| [third_party.md](third_party.md) | Pins, licenses, patch list |
