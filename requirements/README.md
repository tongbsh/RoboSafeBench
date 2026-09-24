# DyniSafety backend requirements (short pins)

These files are **short, hand-maintained pins** for the simulation backends.
They are **mutually exclusive**: never install more than one backend pin set into
the same Python/conda environment.

Conda **names** are not defined here. The runner uses
`configs/models/*.yaml`; see **[docs/user/install.md](../docs/user/install.md)**.
Do not rename envs that already match those YAML strings.

| Backend | Eval env (YAML / paper) | Pip pins | Simulator |
|---------|-------------------------|----------|-----------|
| LIBERO | `openvla-oft` | `libero.txt` | MuJoCo (+ robosuite/LIBERO tree), NumPy 1.x |
| SimplerEnv | `simpler_env` | `simplerenv.txt` | SAPIEN **2.2.2** + ManiSkill2 |
| RoboTwin | `internvla_a1` | `robotwin.txt` | SAPIEN **3.0.0b1** |
| RoboCasa | `dynisafety-robocasa` | `robocasa.txt` | MuJoCo **3.3.1** + robosuite git master, **NumPy 2.2.5** |

`environment/dynisafety-*.yml` are recipes only. Their `name:` fields
(`dynisafety-libero`, …) are not what `paper_main` looks up. A leftover env
named `RoboTwin` is not the paper RoboTwin eval env.

**Hard conflict:** `sapien==2.2.2` (SimplerEnv) vs `sapien==3.0.0b1` (RoboTwin)
cannot coexist in one env.

**Hard conflict (NumPy):** RoboCasa asserts `numpy==2.2.5`. The other backends
follow `pyproject.toml` `numpy>=1.26,<2.0`. Do not remove that upper bound.
Install DyniSafety into `dynisafety-robocasa` with `pip install -e . --no-deps`
then `pip install -r requirements/robocasa.txt`.

## Recommended install

```bash
# -n = YAML extra.eval_env (docs/user/install.md). Skip if that env already exists.
conda env create -n openvla-oft -f environment/dynisafety-libero.yml
conda activate openvla-oft
# Install torch+CUDA for your GPU first (matrix below), then:
pip install -r requirements/libero.txt
pip install -e ".[libero]"
# Then follow docs/user/install.md / scripts/setup/install_<backend>.sh for trees + patches.
```

Do **not**:

```bash
pip install -e ".[libero,simplerenv,robotwin]"   # forbidden
```

## Torch / CUDA matrix (reference)

Pins below match **working paper-box eval envs**. Choose one row, install torch
**before** the backend requirement file. Serve envs (`XVLA`, `openpi`, …) keep
their own upstream torch — do not unify them.

| Eval env | Python | Torch (reference) | CUDA wheel tag | Notes |
|----------|--------|-------------------|----------------|-------|
| `openvla-oft` | 3.10 | 2.4.x | **cu124** | LIBERO; yml recipe `dynisafety-libero` |
| `simpler_env` | 3.10 | 2.3.x | **cu118** | yml recipe `dynisafety-simplerenv` |
| `internvla_a1` | 3.10 | 2.7.x | **cu128** | RoboTwin paper eval; not `RoboTwin` / `dynisafety-robotwin` |
| `dynisafety-robocasa` | **3.11** | (not required for sim import) | — | `numpy==2.2.5`; `MUJOCO_GL=egl` |

Example (libero / cu124):

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

Exact torch pins are **not** duplicated inside `*.txt` so CUDA choice stays explicit.

## What is not pinned here

- Full TensorFlow / Jax stacks for some SimplerEnv policies — see
  `third_party/SimplerEnv-OpenVLA/requirements_full_install.txt` when needed.
- Editable LIBERO / ManiSkill2 / SimplerEnv / RoboTwin / RoboCasa trees — `scripts/setup/install_*.sh`.
- Model **server** conda envs (`openpi`, `univla`, `starVLA`, …) — upstream
  self-built; names in [docs/user/install.md](../docs/user/install.md).
- A bit-exact clone of a paper box. These `*.txt` files are **short recipes**,
  not `pip freeze` / `conda env export` dumps. Working envs can have metadata
  conflicts that still import; replaying a dump often makes the solver abort
  or “repair” a stack that already ran. See
  [docs/user/install.md](../docs/user/install.md) §6.

Do **not** check in or reinstall from:

```bash
conda env export -n <eval_env>
pip freeze
```

Use those only as a private log of what is installed. Install from the recipe
layers (yml → torch → this `*.txt` → `pip install -e ".[<backend>]"` →
`install_*.sh` → upstream model repo).
