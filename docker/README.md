# DyniSafety Docker (LIBERO backend)

This directory documents the **LIBERO / MuJoCo** image defined by the root
[`Dockerfile`](../Dockerfile). One backend per image — do **not** combine
`libero` / `simplerenv` / `robotwin` extras (SAPIEN 2 vs 3 / MuJoCo conflict).
RoboCasa is a **fourth** env (`dynisafety-robocasa`, NumPy 2) and is **not**
in this image.

| Item | Value |
|------|-------|
| Backend | `libero` |
| Base image | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime` (matches `requirements/README.md` cu124 row) |
| Conda env twin | Recipe `environment/dynisafety-libero.yml`; host paper eval name is `openvla-oft` ([install.md](../docs/user/install.md)) |
| Host install twin | `conda activate openvla-oft` then `make install-libero` |

SimplerEnv / RoboTwin / RoboCasa images are **not** shipped in this PR; use the matching
conda yml + `make install-*` on bare metal, or add sibling Dockerfiles later (M11).

## Build

From the repository root:

```bash
docker build -t dynisafety-libero:latest -f Dockerfile .
```

`.dockerignore` keeps checkpoints, `simulator_env/`, SimplerEnv/RoboTwin trees,
and MANO out of the build context.

## Run (GPU)

```bash
docker run --rm --gpus all -it dynisafety-libero:latest
# → prints dynisafety --help
```

Doctor (expects some FAILs until the LIBERO tree is installed/mounted):

```bash
docker run --rm --gpus all dynisafety-libero:latest \
  dynisafety doctor --backend libero
```

Unit tests / lint (no simulator):

```bash
docker run --rm dynisafety-libero:latest make lint
docker run --rm dynisafety-libero:latest make test
```

## Volume mounts

| Host path | Container path | Purpose |
|-----------|----------------|---------|
| `./third_party/dyniSafety_libero` or `./simulator_env/dyniSafety_libero` | same under `/opt/dynisafety/…` | LIBERO/SafeLIBERO tree after `install_libero.sh` |
| `./download_checkpoints` | `/opt/dynisafety/download_checkpoints` | model weights |
| `./results` | `/opt/dynisafety/results` | eval outputs (`contracts` §6) |
| `./assets/mano_v1_2` | `/opt/dynisafety/assets/mano_v1_2` | optional MANO (not redistributed) |

Example:

```bash
docker run --rm --gpus all -it \
  -v "$PWD/simulator_env/dyniSafety_libero:/opt/dynisafety/third_party/dyniSafety_libero:ro" \
  -v "$PWD/download_checkpoints:/opt/dynisafety/download_checkpoints" \
  -v "$PWD/results:/opt/dynisafety/results" \
  dynisafety-libero:latest bash
```

Inside the container, complete install if the tree is not mounted yet:

```bash
bash scripts/setup/install_backend.sh libero --skip-pip   # tree + patches; pip already done in image
dynisafety doctor --backend libero
```

## Smoke eval

Live smoke needs the LIBERO tree, task assets, and a policy checkpoint — not
baked into the image. After mounts:

```bash
dynisafety eval --backend libero --model random --suite smoke --dry-run
# then drop --dry-run when assets/weights are ready
```

## Relation to Make targets

On the **host** (paper LIBERO eval env `openvla-oft`; do not rename it):

```bash
make install-libero
make lint
make test
make report
```

Inside the image, `make lint` / `make test` / `make report` work the same;
`make install-libero` still needs network or a mounted fork.
