# DyniSafety conda environments
#
# Authority (M10 Q1=A): backend env *names* defined here:
#
#   dynisafety-libero
#   dynisafety-simplerenv
#   dynisafety-robotwin
#   dynisafety-robocasa   # Python 3.11, numpy==2.2.5; MUTEX with the three above
#
# Historical aliases (same role, old machines):
#
#   openvla-oft   → dynisafety-libero
#   simpler_env   → dynisafety-simplerenv
#   RoboTwin      → dynisafety-robotwin
#
# Create:
#
#   conda env create -f environment/dynisafety-libero.yml
#   conda activate dynisafety-libero
#   # then torch (requirements/README.md) + pip -r requirements/libero.txt + pip -e ".[libero]"
#
# RoboCasa (numpy 2; do not mix with the three envs above):
#
#   conda env create -f environment/dynisafety-robocasa.yml
#   conda activate dynisafety-robocasa
#   export MUJOCO_GL=egl
#   scripts/install/install_backend.sh robocasa   # pip install -e . --no-deps + requirements/robocasa.txt
#
# Model **server** envs are NOT shipped as yml (M10 Q2=B). Build them from upstream
# model repos. Names referenced by configs/models/*.yaml include:
#
# | YAML eval_env / serve_env / env | Typical upstream / notes |
# |---------------------------------|--------------------------|
# | openvla-oft                     | → use dynisafety-libero (alias) |
# | simpler_env                     | → use dynisafety-simplerenv |
# | RoboTwin                        | → use dynisafety-robotwin |
# | internvla_a1                    | InternVLA-A1 in-process; upstream + sapien3 |
# | openpi                          | openpi serve_policy (π₀ / π₀.₅) |
# | univla                          | UniVLA HTTP server |
# | starVLA                         | StarVLA websocket server |
# | XVLA                            | X-VLA HTTP server |
# | lingbot / lingbot_va            | LingBot-VA server (name drift → M4) |
# | spatialvla                      | SpatialVLA server |
# | GR00T / isaac-gr00t             | GR00T server (name drift → M4) |
# | internvla_m1 / internvla-m1     | InternVLA-M1 server (name drift → M4) |
# | dreamvla, octo, base            | upstream or any backend env (random) |
#
# Acceptance: the four ymls above + this table declaring the
# rest as upstream-provided satisfies M10 docs clause (Q2=B).
