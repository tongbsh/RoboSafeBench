# DyniSafety backend requirements (short pins)

These files are **short, hand-maintained pins** for the simulation backends.
They are **mutually exclusive**: never install more than one backend pin set into
the same Python/conda environment.

| Backend | Conda env (`name:`) | Pip pins | Simulator |
|---------|---------------------|----------|-----------|
| LIBERO | `dynisafety-libero` | `libero.txt` | MuJoCo (+ robosuite/LIBERO tree), NumPy 1.x |
| SimplerEnv | `dynisafety-simplerenv` | `simplerenv.txt` | SAPIEN **2.2.2** + ManiSkill2 |
| RoboTwin | `dynisafety-robotwin` | `robotwin.txt` | SAPIEN **3.0.0b1** |
| RoboCasa | `dynisafety-robocasa` | `robocasa.txt` | MuJoCo **3.3.1** + robosuite git master, **NumPy 2.2.5** |

**Hard conflict:** `sapien==2.2.2` (SimplerEnv) vs `sapien==3.0.0b1` (RoboTwin)
cannot coexist in one env.

**Hard conflict (NumPy):** RoboCasa asserts `numpy==2.2.5`. The other backends
follow `pyproject.toml` `numpy>=1.26,<2.0`. Do not remove that upper bound.
Install DyniSafety into `dynisafety-robocasa` with `pip install -e . --no-deps`
then `pip install -r requirements/robocasa.txt`.

## Recommended install

```bash
conda env create -f environment/dynisafety-<backend>.yml
conda activate dynisafety-<backend>
# Install torch+CUDA for your GPU first (matrix below), then:
pip install -r requirements/<backend>.txt
pip install -e ".[<backend>]"
# Then follow docs/user/third_party.md / scripts/setup/install_<backend>.sh for trees + patches.
```

Do **not**:

```bash
pip install -e ".[libero,simplerenv,robotwin]"   # forbidden
```

## Torch / CUDA matrix (reference)

Pins below match **working Autodl-style envs** used for DyniSafety development.
Choose one row, install torch **before** the backend requirement file.

| Backend env | Python | Torch (reference) | CUDA wheel tag | Notes |
|-------------|--------|-------------------|----------------|-------|
| `dynisafety-libero` | 3.10 | 2.4.x | **cu124** | Historical alias: `openvla-oft` |
| `dynisafety-simplerenv` | 3.10 | 2.3.x | **cu118** | Historical alias: `simpler_env` |
| `dynisafety-robotwin` | 3.10 | 2.4.x | **cu121** | Historical alias: `RoboTwin` |
| `dynisafety-robocasa` | **3.11** | (not required for sim import) | — | `numpy==2.2.5`; `MUJOCO_GL=egl` |

Footnote: InternVLA-A1 in-process stacks have been observed on **cu128** / torch 2.7.x
in a separate env (`internvla_a1`); that is a **model** env, not one of the four
backend ymls. Prefer the robotwin row unless you are reproducing that stack.

Example (libero / cu124):

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

Exact torch pins are **not** duplicated inside `*.txt` so CUDA choice stays explicit.

## What is not pinned here

- Full TensorFlow / Jax stacks for some SimplerEnv policies — see
  `third_party/SimplerEnv-OpenVLA/requirements_full_install.txt` when needed.
- Editable LIBERO / ManiSkill2 / SimplerEnv / RoboTwin / RoboCasa trees — `scripts/setup/install_*.sh`.
- Model **server** conda envs (`openpi`, `univla`, `starVLA`, …) — see
  `environment/README.md` (upstream self-built; Q2=B).
