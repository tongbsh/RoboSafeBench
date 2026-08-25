#!/usr/bin/env bash
# Shared setup sourced by scripts/{setup,eval,debug,extra}/*. Not executed directly.
#
# Override anything by exporting it before calling a script, e.g.
#   DYNI_CONDA_SH=/opt/conda/etc/profile.d/conda.sh bash scripts/eval/00_smoke.sh
#   DYNI_GPUS=0,1 bash scripts/eval/01_eval.sh

set -euo pipefail

# --- paths -----------------------------------------------------------------
DYNI_ROOT="${DYNI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DYNI_ROOT
cd "$DYNI_ROOT"

# conda installation (the hook script), NOT the env directory
DYNI_CONDA_SH="${DYNI_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

# env used for repo-level tooling that needs no simulator (doctor/report/config)
DYNI_TOOL_ENV="${DYNI_TOOL_ENV:-base}"

# Simulator envs, per backend. A run needs an env that satisfies BOTH the
# simulator (backend) and the policy (model). Model YAMLs encode the common
# combinations in extra.eval_env; these are the fallbacks for generic models
# such as `random` whose eval_env is `base`.
DYNI_ENV_LIBERO="${DYNI_ENV_LIBERO:-openvla-oft}"
DYNI_ENV_SIMPLERENV="${DYNI_ENV_SIMPLERENV:-simpler_env}"
DYNI_ENV_ROBOTWIN="${DYNI_ENV_ROBOTWIN:-internvla_a1}"
DYNI_ENV_ROBOCASA="${DYNI_ENV_ROBOCASA:-dynisafety-robocasa}"

dyni_backend_env() {
  case "$1" in
    libero)     printf '%s' "$DYNI_ENV_LIBERO" ;;
    simplerenv) printf '%s' "$DYNI_ENV_SIMPLERENV" ;;
    robotwin)   printf '%s' "$DYNI_ENV_ROBOTWIN" ;;
    robocasa)   printf '%s' "$DYNI_ENV_ROBOCASA" ;;
    *) die "unknown backend '$1' (expected libero|simplerenv|robotwin|robocasa)" ;;
  esac
}

# archived golden logs used by 50_verify_golden.sh
DYNI_GOLDEN_TAR="${DYNI_GOLDEN_TAR:-${HOME}/autodl-tmp/dynisafety_archive/golden_logs_20260805.tar.gz}"

# --- pretty logging --------------------------------------------------------
if [[ -t 1 ]]; then
  _C_R=$'\033[31m'; _C_G=$'\033[32m'; _C_Y=$'\033[33m'; _C_B=$'\033[36m'; _C_0=$'\033[0m'
else
  _C_R=""; _C_G=""; _C_Y=""; _C_B=""; _C_0=""
fi
info()  { printf '%s==>%s %s\n' "$_C_B" "$_C_0" "$*"; }
ok()    { printf '%s[ok]%s %s\n' "$_C_G" "$_C_0" "$*"; }
warn()  { printf '%s[warn]%s %s\n' "$_C_Y" "$_C_0" "$*" >&2; }
die()   { printf '%s[fail]%s %s\n' "$_C_R" "$_C_0" "$*" >&2; exit 1; }

# --- conda -----------------------------------------------------------------
dyni_conda_init() {
  if ! declare -F conda >/dev/null 2>&1; then
    [[ -f "$DYNI_CONDA_SH" ]] || die "conda hook not found: $DYNI_CONDA_SH (set DYNI_CONDA_SH)"
    # shellcheck disable=SC1090
    source "$DYNI_CONDA_SH"
  fi
}

# dyni_activate <env-name>
dyni_activate() {
  local env="$1"
  dyni_conda_init
  if [[ "$env" == "base" ]]; then
    conda activate base
  else
    conda activate "$env" 2>/dev/null || die "conda env '$env' not found. Run: bash scripts/setup/audit_envs.sh"
  fi
  # Prefer the env's own shared libraries. Without this, models that pull in
  # tokenizers/ICU hit `libstdc++.so.6: version CXXABI_… not found` because the
  # older system libstdc++ wins over the env's.
  if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  ok "conda env: ${CONDA_DEFAULT_ENV:-?}  ($(command -v python))"
}

# dyni_model_env <model> <eval|serve> [backend]
# For role=eval, optional backend applies configs/models per_backend overlays
# (starvla×robotwin → internvla_a1).
dyni_model_env() {
  local model="$1" role="$2" backend="${3:-}"
  if [[ -n "$backend" && "$role" == "eval" ]]; then
    "${DYNI_PY:-python}" "$DYNI_ROOT/tools/model_env.py" "$model" "$role" --backend "$backend"
  else
    "${DYNI_PY:-python}" "$DYNI_ROOT/tools/model_env.py" "$model" "$role"
  fi
}

# dyni_activate_for <model> <eval|serve> [backend]
# For role=eval the env must also satisfy the simulator, so a generic model
# (eval_env: base) falls back to the backend's simulator env.
dyni_activate_for() {
  local model="$1" role="$2" backend="${3:-}" env
  if [[ "$role" == "eval" && -n "$backend" ]]; then
    env="$(dyni_model_env "$model" "$role" "$backend")" || die "cannot resolve $role env for model '$model'"
  else
    env="$(dyni_model_env "$model" "$role")" || die "cannot resolve $role env for model '$model'"
  fi
  if [[ "$role" == "eval" && -n "$backend" && ( "$env" == "base" || -z "$env" ) ]]; then
    env="$(dyni_backend_env "$backend")"
    info "model '$model' is backend-agnostic; using $backend simulator env: $env"
  else
    info "model '$model' needs $role env: $env"
  fi
  dyni_activate "$env"
  # the target env may not have dynisafety installed; make src/ importable regardless
  export PYTHONPATH="$DYNI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
}

# dyni_needs_server <model> -> exit 0 if the model requires a server process
dyni_needs_server() {
  "${DYNI_PY:-python}" "$DYNI_ROOT/tools/model_env.py" --needs-server "$1"
}

# Always invoke the CLI through the *currently active* interpreter.
#
# Do not use the `dynisafety` console script here: base's bin stays on PATH
# after `conda activate <env>`, so `command -v dynisafety` can resolve to
# base's script and silently run the wrong interpreter (missing gymnasium,
# sapien, ...). `python -m` binds to the active env by construction.
dyni_cli() {
  PYTHONPATH="$DYNI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m dynisafety.cli "$@"
}
