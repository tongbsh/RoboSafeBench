# DyniSafety conda environment recipes

These ymls are **optional clean-slate recipes** for a new machine
(`python` + `requirements/<backend>.txt`). They are **not** the names
`paper_main` looks up.

Runtime names live in `configs/models/*.yaml`. The matrix, aliases, and
“do not rename working envs” rule: **[docs/user/install.md](../docs/user/install.md)**.

| yml `name:` (recipe only) | Create as (`conda env create -n …`) | Backend |
|---------------------------|-------------------------------------|---------|
| `dynisafety-libero` | `openvla-oft` | LIBERO / MuJoCo |
| `dynisafety-simplerenv` | `simpler_env` | SimplerEnv / SAPIEN 2.2.2 |
| `dynisafety-robotwin` | `internvla_a1` (paper eval; not this yml name) | RoboTwin / SAPIEN 3 |
| `dynisafety-robocasa` | `dynisafety-robocasa` | RoboCasa / NumPy 2 — mutex with the three above |

```bash
conda env create -n openvla-oft -f environment/dynisafety-libero.yml
conda activate openvla-oft
# torch (requirements/README.md) + pip -e ".[libero]" + scripts/setup/install_backend.sh libero
```

RoboCasa (numpy 2; do not mix with the three eval envs above):

```bash
conda env create -n dynisafety-robocasa -f environment/dynisafety-robocasa.yml
conda activate dynisafety-robocasa
export MUJOCO_GL=egl
scripts/setup/install_backend.sh robocasa
```

A leftover env named `RoboTwin` is **not** the paper RoboTwin eval env.
Paper robotwin jobs use `internvla_a1`.

Model **server** envs are not shipped as yml. Keep the `server.env` spelling
(`XVLA`, `starVLA`, `openpi`, `univla`, `internvla-m1`, `lingbot_va`, …) —
see [docs/user/install.md](../docs/user/install.md).
