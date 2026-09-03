#!/usr/bin/env bash
# Optional: pick a backend + VLA, then create a *missing* eval conda env from
# the short recipe and install that backend into it.
#
#   bash scripts/setup/install_conda_env.sh
#   bash scripts/setup/install_conda_env.sh --backend libero --model openvla_oft
#   bash scripts/setup/install_conda_env.sh --backend robotwin --model starvla --yes
#   bash scripts/setup/install_conda_env.sh --list
#   bash scripts/setup/install_conda_env.sh --backend libero --model starvla --dry-run
#
# Safe on a paper machine that already works:
#   - never renames or removes an env
#   - never conda env export / pip freeze
#   - if the YAML eval env already exists, create is skipped
#   - install_backend.sh is NOT run on an existing env unless --install-backend
#   - serve / in-process VLA stacks are NOT installed (upstream only)
#
# Exit: 0 ok / skip; 1 create/install failed; 2 usage.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

BACKEND=""
MODEL=""
ASSUME_YES=0
DRY_RUN=0
SKIP_TORCH=0
INSTALL_BACKEND=""   # empty = auto (yes if we created the env, no if it existed)
LIST_ONLY=0
PAPER_ONLY=0

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  echo
  echo "Flags: --backend libero|simplerenv|robotwin|robocasa"
  echo "       --model <stem>     configs/models/<stem>.yaml"
  echo "       --yes              skip the confirm prompt"
  echo "       --dry-run          print the plan only"
  echo "       --skip-torch       do not pip-install the reference torch wheel"
  echo "       --install-backend  run install_backend.sh even if the eval env already exists"
  echo "       --paper            interactive model list = paper_main only"
  echo "       --list             print backend × paper models and exit"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --backend) BACKEND="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-torch) SKIP_TORCH=1; shift ;;
    --install-backend) INSTALL_BACKEND=1; shift ;;
    --paper) PAPER_ONLY=1; shift ;;
    --list) LIST_ONLY=1; shift ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

BACKENDS=(libero simplerenv robotwin robocasa)

yml_for_backend() {
  case "$1" in
    libero)     printf '%s' "$DYNI_ROOT/environment/dynisafety-libero.yml" ;;
    simplerenv) printf '%s' "$DYNI_ROOT/environment/dynisafety-simplerenv.yml" ;;
    robotwin)   printf '%s' "$DYNI_ROOT/environment/dynisafety-robotwin.yml" ;;
    robocasa)   printf '%s' "$DYNI_ROOT/environment/dynisafety-robocasa.yml" ;;
    *) die "unknown backend '$1'" ;;
  esac
}

# Reference torch for a *new* eval env only (requirements/README.md).
torch_spec_for_backend() {
  case "$1" in
    libero)     printf '%s' "torch==2.4.1|--index-url https://download.pytorch.org/whl/cu124" ;;
    simplerenv) printf '%s' "torch==2.3.1|--index-url https://download.pytorch.org/whl/cu118" ;;
    robotwin)   printf '%s' "torch==2.7.1|--index-url https://download.pytorch.org/whl/cu128" ;;
    robocasa)   printf '%s' "" ;;
    *) printf '%s' "" ;;
  esac
}

conda_env_exists() {
  local name="$1"
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$name"
}

# Prints "stem[TAB]paper|extra" lines for models that declare this backend.
list_models() {
  local backend="$1" paper_only="$2"
  "${DYNI_PY:-python}" - "$DYNI_ROOT" "$backend" "$paper_only" <<'PY'
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1])
backend = sys.argv[2]
paper_only = sys.argv[3] == "1"
paper = {
    "libero": [
        "openvla_oft", "xvla", "starvla", "pi0", "pi05", "univla", "openvla_oft_rlinf",
    ],
    "robotwin": ["internvla_a1", "fastwam", "lingbot_va", "starvla", "xvla_official"],
    "simplerenv": ["starvla", "internvla_m1", "xvla_official"],
    "robocasa": ["random_mobile", "xiaomi"],
}.get(backend, [])
seen = set()
ordered = []
for stem in paper:
    path = root / "configs" / "models" / f"{stem}.yaml"
    if path.exists():
        ordered.append((stem, "paper"))
        seen.add(stem)
if not paper_only:
    for path in sorted((root / "configs" / "models").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        backends = data.get("backends") or []
        if backend in backends and path.stem not in seen:
            ordered.append((path.stem, "extra"))
for stem, kind in ordered:
    print(f"{stem}\t{kind}")
PY
}

print_catalog() {
  local b
  for b in "${BACKENDS[@]}"; do
    echo
    echo "== $b  (eval name from YAML; * = paper_main) =="
    list_models "$b" 0 | while IFS=$'\t' read -r stem kind; do
      if [[ "$kind" == paper ]]; then
        printf '  * %s\n' "$stem"
      else
        printf '    %s\n' "$stem"
      fi
    done
  done
  echo
  echo "This script only creates a *missing* eval env from environment/dynisafety-*.yml"
  echo "and installs the simulator. Serve VLAs (XVLA, starVLA, openpi, …) stay upstream."
  echo "See docs/user/install.md"
}

prompt_choice() {
  local prompt="$1"
  local max="$2"
  local reply
  while true; do
    read -r -p "$prompt" reply || die "no input"
    if [[ "$reply" =~ ^[0-9]+$ ]] && (( reply >= 1 && reply <= max )); then
      printf '%s' "$reply"
      return 0
    fi
    echo "enter a number 1..$max"
  done
}

if [[ "$LIST_ONLY" == 1 ]]; then
  print_catalog
  exit 0
fi

dyni_conda_init

if [[ -z "$BACKEND" ]]; then
  if [[ ! -t 0 ]]; then
    usage >&2
    die "non-interactive: pass --backend and --model (or --list)"
  fi
  echo "Backend:"
  local_i=1
  for b in "${BACKENDS[@]}"; do
    printf '  %d) %s\n' "$local_i" "$b"
    local_i=$((local_i + 1))
  done
  idx="$(prompt_choice 'select backend [1-4]: ' 4)"
  BACKEND="${BACKENDS[$((idx - 1))]}"
fi

case "$BACKEND" in
  libero|simplerenv|robotwin|robocasa) ;;
  *) die "unknown backend '$BACKEND' (libero|simplerenv|robotwin|robocasa)" ;;
esac

if [[ -z "$MODEL" ]]; then
  if [[ ! -t 0 ]]; then
    usage >&2
    die "non-interactive: pass --model"
  fi
  mapfile -t MODEL_ROWS < <(list_models "$BACKEND" "$PAPER_ONLY")
  [[ ${#MODEL_ROWS[@]} -gt 0 ]] || die "no models declare backend '$BACKEND'"
  echo
  echo "Models on $BACKEND (* = paper_main):"
  i=1
  for row in "${MODEL_ROWS[@]}"; do
    stem="${row%%$'\t'*}"
    kind="${row#*$'\t'}"
    mark=""
    [[ "$kind" == paper ]] && mark="*"
    printf '  %2d) %-24s %s\n' "$i" "$stem" "$mark"
    i=$((i + 1))
  done
  idx="$(prompt_choice "select model [1-${#MODEL_ROWS[@]}]: " "${#MODEL_ROWS[@]}")"
  MODEL="${MODEL_ROWS[$((idx - 1))]%%$'\t'*}"
fi

[[ -f "$DYNI_ROOT/configs/models/${MODEL}.yaml" ]] || \
  die "unknown model '$MODEL' (no configs/models/${MODEL}.yaml)"

EVAL_ENV="$(dyni_model_env "$MODEL" eval "$BACKEND" 2>/dev/null || true)"
if [[ -z "$EVAL_ENV" || "$EVAL_ENV" == "base" ]]; then
  EVAL_ENV="$(dyni_backend_env "$BACKEND")"
  info "model '$MODEL' has no dedicated eval_env; using $BACKEND fallback: $EVAL_ENV"
fi

SERVE_ENV=""
SERVE_ROOT=""
SERVE_LAUNCH=""
if dyni_needs_server "$MODEL"; then
  SERVE_ENV="$(dyni_model_env "$MODEL" serve)"
  SERVE_ROOT="$("${DYNI_PY:-python}" - "$DYNI_ROOT/configs/models/${MODEL}.yaml" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print((data.get("server") or {}).get("root") or "-")
PY
)"
  SERVE_LAUNCH="$("${DYNI_PY:-python}" - "$DYNI_ROOT/configs/models/${MODEL}.yaml" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print((data.get("server") or {}).get("launch") or "-")
PY
)"
fi

YML="$(yml_for_backend "$BACKEND")"
[[ -f "$YML" ]] || die "missing recipe $YML"

EXISTS=0
if conda_env_exists "$EVAL_ENV"; then
  EXISTS=1
fi

if [[ -z "$INSTALL_BACKEND" ]]; then
  if [[ "$EXISTS" == 1 ]]; then
    INSTALL_BACKEND=0
  else
    INSTALL_BACKEND=1
  fi
fi

TORCH_SPEC="$(torch_spec_for_backend "$BACKEND")"
DO_TORCH=0
if [[ "$SKIP_TORCH" == 0 && -n "$TORCH_SPEC" && "$EXISTS" == 0 ]]; then
  DO_TORCH=1
fi

echo
info "plan (optional installer — existing envs are never overwritten)"
echo "    backend     = $BACKEND"
echo "    model       = $MODEL"
echo "    eval_env    = $EVAL_ENV  ($( [[ "$EXISTS" == 1 ]] && echo 'EXISTS — skip create' || echo "MISSING — create from ${YML##*/}" ))"
if [[ -n "$SERVE_ENV" ]]; then
  if conda_env_exists "$SERVE_ENV"; then
    echo "    serve_env   = $SERVE_ENV  (EXISTS — leave it alone)"
  else
    echo "    serve_env   = $SERVE_ENV  (MISSING — not created here; install from upstream)"
  fi
  echo "    serve root  = ${SERVE_ROOT:-?}"
  echo "    serve launch= ${SERVE_LAUNCH:-?}"
else
  echo "    serve_env   = —  (in-process / local; extra VLA deps come from upstream into eval_env)"
fi
echo "    torch       = $( [[ "$DO_TORCH" == 1 ]] && echo "install ${TORCH_SPEC%%|*} on the new env" || echo 'skip' )"
echo "    backend pip = $( [[ "$INSTALL_BACKEND" == 1 ]] && echo "install_backend.sh $BACKEND" || echo 'skip (pass --install-backend to run on an existing env)' )"
if [[ "$BACKEND" == robotwin && "$EVAL_ENV" == internvla_a1 && ( "$MODEL" == internvla_a1 || "$MODEL" == fastwam ) ]]; then
  warn "internvla_a1 from the robotwin yml is the *sim* layer only (sapien 3)."
  warn "in-process $MODEL weights still need the upstream model repo in extra.repo_root."
fi

if [[ "$DRY_RUN" == 1 ]]; then
  ok "dry-run; nothing changed"
  exit 0
fi

if [[ "$ASSUME_YES" != 1 ]]; then
  if [[ ! -t 0 ]]; then
    die "non-interactive: pass --yes to apply the plan"
  fi
  read -r -p "proceed? [y/N] " ans || die "no input"
  [[ "$ans" == y || "$ans" == Y || "$ans" == yes ]] || die "aborted"
fi

CREATED=0
if [[ "$EXISTS" == 1 ]]; then
  ok "eval env '$EVAL_ENV' already exists — skip create (will not rename or overwrite)"
else
  info "conda env create -n $EVAL_ENV -f $YML"
  conda env create -n "$EVAL_ENV" -f "$YML" || die "conda env create failed"
  CREATED=1
  ok "created $EVAL_ENV"
fi

if [[ "$DO_TORCH" == 1 || "$INSTALL_BACKEND" == 1 ]]; then
  dyni_activate "$EVAL_ENV"
fi

if [[ "$DO_TORCH" == 1 ]]; then
  pkg="${TORCH_SPEC%%|*}"
  extra="${TORCH_SPEC#*|}"
  info "pip install $pkg $extra"
  # shellcheck disable=SC2086
  pip install "$pkg" $extra || die "torch install failed (retry with --skip-torch and install a wheel yourself)"
fi

if [[ "$INSTALL_BACKEND" == 1 ]]; then
  if [[ "$BACKEND" == robocasa ]]; then
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
  fi
  info "install_backend.sh $BACKEND"
  bash "$DYNI_ROOT/scripts/setup/install_backend.sh" "$BACKEND" \
    || die "install_backend.sh $BACKEND failed"
else
  info "skipped install_backend.sh (env already present)"
fi

echo
if [[ -n "$SERVE_ENV" ]]; then
  if conda_env_exists "$SERVE_ENV"; then
    ok "serve env '$SERVE_ENV' is already on this machine — not modified"
  else
    warn "serve env '$SERVE_ENV' is missing. Build it from the upstream repo"
    warn "  ($SERVE_ROOT) and keep this exact conda name."
    warn "This script will not create an empty '$SERVE_ENV' or pip-freeze a dump."
  fi
fi

if [[ "$CREATED" == 1 && ( "$MODEL" == internvla_a1 || "$MODEL" == fastwam ) ]]; then
  warn "next: install $MODEL into '$EVAL_ENV' from extra.repo_root (sim layer only is not enough)"
fi

ok "done. names: bash scripts/setup/audit_envs.sh"
ok "docs: docs/user/install.md"
