# DyniSafety — LIBERO (MuJoCo) backend image
#
# Mutual exclusion: this image targets the libero extra only.
# Do NOT layer simplerenv (sapien 2.x) or robotwin (sapien 3.x) into the same image.
#
# Build:
#   docker build -t dynisafety-libero:latest -f Dockerfile .
# See docker/README.md for run / volume mounts / completing install_libero.

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DYNISAFETY_DOCKER_BACKEND=libero

# git/curl: install_libero.sh / submodule helpers; build-essential for occasional wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dynisafety

# Copy workspace (filtered by .dockerignore). ARG is image metadata only — not an eval hyperparameter.
ARG BACKEND=libero
RUN test "$BACKEND" = "libero" \
    || (echo "This Dockerfile only supports BACKEND=libero" >&2 && exit 1)

COPY pyproject.toml README.md LICENSE Makefile ./
COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY requirements ./requirements
COPY environment ./environment
COPY scripts ./scripts
COPY patches ./patches
COPY docs ./docs
# Required, not optional: tests/test_example_agent.py imports dyni_example_agent
# and `make lint` covers examples/, so omitting this breaks make test/lint here.
COPY examples ./examples
# First-party hand assets ship inside the package (src/dynisafety/backends/*/assets).

# Core + libero extra + dev tooling (ruff/pytest for make lint/test inside the image)
RUN pip install --no-cache-dir -r requirements/libero.txt \
    && pip install --no-cache-dir -e ".[libero,dev]"

# LIBERO/SafeLIBERO tree is NOT baked in (needs network / local fork at runtime):
#   bash scripts/install/install_backend.sh libero
# or mount third_party/dyniSafety_libero / external/simulators/dyniSafety_libero.

# Default command: help. Prefer explicit `docker run ... dynisafety doctor --backend libero`.
CMD ["dynisafety", "--help"]
