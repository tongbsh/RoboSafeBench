#!/usr/bin/env bash
# Apply DyniSafety patches onto pinned third_party trees (idempotent).
#
# Usage:
#   scripts/setup/apply_patches.sh [--check] [--dry-run] [robotwin|simplerenv|libero|robocasa|all]
#
# Exit codes:
#   0  applied / already applied / --check all APPLIED
#   1  apply failed or conflict
#   2  --check found MISSING or DRIFTED
#
# Detection (per unified-diff patch):
#   APPLIED  — git apply --reverse --check succeeds
#   MISSING  — git apply --check succeeds (clean apply possible)
#   DRIFTED  — neither reverse nor forward check succeeds
#   ERROR    — tree missing / not a git work tree
#
# Special: patches that contain "# apply: NOT git-apply" (e.g. ManiSkill2 SHA)
# are handled via `git checkout <sha>` instead of git-apply.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECK_ONLY=0
DRY_RUN=0
TARGET="all"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    robotwin|simplerenv|libero|robocasa|all) TARGET="$1"; shift ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

MANISKILL2_SHA="91d154bfd864577f8d2e80f3fc2f8b4d9df9ae5c"

is_checkout_pin_patch() {
  local patch="$1"
  grep -qE '^# apply: NOT git-apply' "$patch"
}

tree_for_group() {
  case "$1" in
    simplerenv) echo "${ROOT}/third_party/SimplerEnv-OpenVLA" ;;
    robotwin) echo "${ROOT}/third_party/RoboTwin" ;;
    libero) echo "${ROOT}/third_party/dyniSafety_libero" ;;
    robocasa) echo "${ROOT}/third_party/robocasa" ;;
    *) return 1 ;;
  esac
}

list_patches() {
  local group="$1"
  local dir="${ROOT}/patches/${group}"
  if [[ ! -d "$dir" ]]; then
    return 0
  fi
  find "$dir" -maxdepth 1 -type f -name '*.patch' | sort
}

classify_unified_patch() {
  local tree="$1"
  local patch="$2"
  if [[ ! -e "$tree" ]]; then
    echo ERROR
    return 0
  fi
  if (cd "$tree" && git apply --reverse --check "$patch") >/dev/null 2>&1; then
    echo APPLIED
    return 0
  fi
  if (cd "$tree" && git apply --check "$patch") >/dev/null 2>&1; then
    echo MISSING
    return 0
  fi
  echo DRIFTED
}

maniskill2_state() {
  local tree="$1"
  local ms="${tree}/ManiSkill2_real2sim"
  if [[ ! -e "$ms" ]]; then
    echo ERROR
    return 0
  fi
  local head
  head="$(git -C "$ms" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$head" == "$MANISKILL2_SHA" ]]; then
    echo APPLIED
  elif [[ -z "$head" ]]; then
    echo ERROR
  else
    echo MISSING
  fi
}

apply_unified_patch() {
  local tree="$1"
  local patch="$2"
  local state
  state="$(classify_unified_patch "$tree" "$patch")"
  case "$state" in
    APPLIED)
      echo "[skip] already applied: $(basename "$patch")"
      return 0
      ;;
    MISSING)
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] would apply: $(basename "$patch")"
        return 0
      fi
      echo "[apply] $(basename "$patch")"
      (cd "$tree" && git apply "$patch")
      return $?
      ;;
    DRIFTED)
      echo "[error] DRIFTED (neither apply nor reverse-check): $(basename "$patch")" >&2
      return 1
      ;;
    ERROR)
      echo "[error] tree missing or not usable: $tree" >&2
      return 1
      ;;
  esac
}

apply_maniskill2_pin() {
  local tree="$1"
  local patch="$2"
  local state
  state="$(maniskill2_state "$tree")"
  case "$state" in
    APPLIED)
      echo "[skip] ManiSkill2 already at ${MANISKILL2_SHA:0:7}"
      return 0
      ;;
    MISSING)
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] would checkout ManiSkill2 @ ${MANISKILL2_SHA:0:7}"
        return 0
      fi
      echo "[checkout] ManiSkill2_real2sim @ ${MANISKILL2_SHA:0:7} ($(basename "$patch"))"
      git -C "${tree}/ManiSkill2_real2sim" fetch --all --quiet 2>/dev/null || true
      git -C "${tree}/ManiSkill2_real2sim" checkout "$MANISKILL2_SHA"
      return $?
      ;;
    *)
      echo "[error] cannot pin ManiSkill2 under $tree" >&2
      return 1
      ;;
  esac
}

process_group() {
  local group="$1"
  local tree
  tree="$(tree_for_group "$group")"
  local patches
  mapfile -t patches < <(list_patches "$group")

  if [[ ${#patches[@]} -eq 0 ]]; then
    if [[ "$group" == "libero" || "$group" == "robocasa" ]]; then
      echo "[info] no patches/${group} (install-only); nothing to apply"
      return 0
    fi
    echo "[warn] no patches for group=$group under patches/${group}/" >&2
    return 0
  fi

  if [[ ! -e "$tree" ]]; then
    echo "[error] missing tree for ${group}: $tree" >&2
    echo "        run scripts/setup/install_${group}.sh (or git submodule update --init) first" >&2
    return 1
  fi

  local rc=0
  local patch state
  for patch in "${patches[@]}"; do
    if is_checkout_pin_patch "$patch"; then
      if [[ "$CHECK_ONLY" -eq 1 ]]; then
        state="$(maniskill2_state "$tree")"
        echo "[check] $(basename "$patch"): $state"
        if [[ "$state" != "APPLIED" ]]; then
          rc=2
        fi
      else
        apply_maniskill2_pin "$tree" "$patch" || return 1
      fi
      continue
    fi

    if [[ "$CHECK_ONLY" -eq 1 ]]; then
      state="$(classify_unified_patch "$tree" "$patch")"
      echo "[check] $(basename "$patch"): $state"
      if [[ "$state" != "APPLIED" ]]; then
        rc=2
      fi
    else
      apply_unified_patch "$tree" "$patch" || return 1
    fi
  done
  return "$rc"
}

PATCH_GROUPS=()
case "$TARGET" in
  all) PATCH_GROUPS=(simplerenv robotwin libero) ;;
  *) PATCH_GROUPS=("$TARGET") ;;
esac

OVERALL=0
for g in "${PATCH_GROUPS[@]}"; do
  echo "=== ${g} ==="
  set +e
  process_group "$g"
  prc=$?
  set -e
  if [[ "$prc" -eq 1 ]]; then
    exit 1
  fi
  if [[ "$prc" -eq 2 ]]; then
    OVERALL=2
  fi
done

if [[ "$CHECK_ONLY" -eq 1 && "$OVERALL" -eq 2 ]]; then
  echo "patch check failed: MISSING or DRIFTED present" >&2
  exit 2
fi

echo "done (target=${TARGET}, check=${CHECK_ONLY}, dry_run=${DRY_RUN})"
exit 0
