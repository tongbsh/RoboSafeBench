#!/usr/bin/env bash
# DyniSafety debug eval — 快速切换 VLA × backend × level × 轨迹数。
#
# 面向日常调参 / 冒烟，不是论文正式矩阵（正式请用 scripts/run/30_sweep.sh 或
# configs/suites/paper_main.yaml）。内部走 scripts/run/10_eval.sh（自动选 conda）。
#
# 用法:
#   bash scripts/debug/eval_debug.sh [选项]
#
# 示例见 --help / scripts/README.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/run/00_env.sh"

# --- defaults (debug-friendly) ---------------------------------------------
BACKEND="libero"
MODEL="pi05"
SUITE=""
LEVELS="0"
TASKS="0"
EPISODES="1"
SEED="7"
DRY_RUN=0
NO_VIDEO=0
ORACLE=0
AEGIS=0
EXTRA_ARGS=()

DEFAULT_SUITE_FOR_BACKEND() {
  case "$1" in
    libero)     printf '%s' "safelibero_spatial" ;;
    simplerenv) printf '%s' "bridge" ;;
    robotwin)   printf '%s' "main20" ;;
    robocasa)   printf '%s' "robocasa_nav" ;;
    *)          printf '%s' "" ;;
  esac
}

usage() {
  cat <<'EOF'
DyniSafety debug eval — 单机调试不同 VLA / benchmark / level

必选语义（CLI 无业务默认；本脚本仅提供 debug 默认值）:
  --backend   libero | simplerenv | robotwin | robocasa  (默认: libero)
  --model     configs/models/<stem>.yaml 的 stem  (默认: pi05)
  --suite     backend 侧 suite id                 (默认随 backend)
  --level     单个 level，或逗号列表 / all        (默认: 0)
  --task      单个 task id，或逗号列表 / all      (默认: 0；all=省略 --task)
  --episodes  每个 (task,level) 的轨迹数          (默认: 1)
  --seed      基种子                              (默认: 7)

可选:
  --dry-run           只打印计划，不启仿真
  --no-video          关闭视频
  --oracle            叠加 Oracle stop-wait-resume（仅 libero 有意义）
  --aegis             叠加 AEGIS enable overlay（建议同时 --oracle）
  --set KEY=VALUE     透传 dynisafety --set（可重复）
  --config PATH       透传额外 YAML（可重复）
  -h | --help         本帮助

常用组合:
  # π₀.₅ × LIBERO × L0–L3，每格 3 条轨迹
  bash scripts/debug/eval_debug.sh \
    --backend libero --model pi05 --suite safelibero_spatial \
    --level all --task 0 --episodes 3

  # openvla_oft × spatial × L1，dry-run
  bash scripts/debug/eval_debug.sh \
    --model openvla_oft --level 1 --episodes 1 --dry-run

  # SimplerEnv bridge × starvla × L2
  bash scripts/debug/eval_debug.sh \
    --backend simplerenv --model starvla --suite bridge \
    --level 2 --task 0 --episodes 1

  # RoboTwin × internvla_a1 × L0
  bash scripts/debug/eval_debug.sh \
    --backend robotwin --model internvla_a1 --suite main20 \
    --level 0 --task 21 --episodes 1

  # RoboCasa kitchen-nav × random_mobile × L0（numpy 2 env；不是 paper_main）
  bash scripts/debug/eval_debug.sh \
    --backend robocasa --model random_mobile --suite robocasa_nav \
    --level 0 --task 0 --episodes 1 --dry-run

  # LIBERO + Oracle + AEGIS（调试护栏）
  bash scripts/debug/eval_debug.sh \
    --model pi05 --level 1 --episodes 1 --oracle --aegis

结果目录: results/l{level}/<run_id>/
正式大规模跑: bash scripts/run/30_sweep.sh --dry-run
EOF
}

die_usage() { echo "error: $*" >&2; usage >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --backend)  BACKEND="${2:?}"; shift 2 ;;
    --model)    MODEL="${2:?}"; shift 2 ;;
    --suite)    SUITE="${2:?}"; shift 2 ;;
    --level)    LEVELS="${2:?}"; shift 2 ;;
    --task)     TASKS="${2:?}"; shift 2 ;;
    --episodes) EPISODES="${2:?}"; shift 2 ;;
    --seed)     SEED="${2:?}"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --no-video) NO_VIDEO=1; shift ;;
    --oracle)   ORACLE=1; shift ;;
    --aegis)    AEGIS=1; shift ;;
    --set)
      EXTRA_ARGS+=(--set "${2:?}")
      shift 2
      ;;
    --config)
      EXTRA_ARGS+=(--config "${2:?}")
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      die_usage "unknown argument: $1"
      ;;
  esac
done

case "$BACKEND" in
  libero|simplerenv|robotwin|robocasa) ;;
  *) die_usage "invalid --backend '$BACKEND'" ;;
esac

if [[ -z "$SUITE" ]]; then
  SUITE="$(DEFAULT_SUITE_FOR_BACKEND "$BACKEND")"
fi
[[ -n "$SUITE" ]] || die_usage "suite empty; pass --suite"

# expand levels
if [[ "$LEVELS" == "all" ]]; then
  LEVEL_LIST=(0 1 2 3)
else
  IFS=',' read -r -a LEVEL_LIST <<< "$LEVELS"
fi
for lv in "${LEVEL_LIST[@]}"; do
  [[ "$lv" =~ ^[0-3]$ ]] || die_usage "invalid level '$lv' (expect 0..3 or all)"
done

# expand tasks: "all" → omit --task (backend list_tasks)
TASK_MODE="list"
TASK_LIST=()
if [[ "$TASKS" == "all" ]]; then
  TASK_MODE="all"
else
  IFS=',' read -r -a TASK_LIST <<< "$TASKS"
  for t in "${TASK_LIST[@]}"; do
    [[ "$t" =~ ^[0-9]+$ ]] || die_usage "invalid task id '$t'"
  done
fi

[[ "$EPISODES" =~ ^[0-9]+$ ]] || die_usage "--episodes must be int >= 0"
[[ "$SEED" =~ ^[0-9]+$ ]] || die_usage "--seed must be int >= 0"

COMMON=(--seed "$SEED" --episodes "$EPISODES")
[[ "$DRY_RUN" -eq 1 ]] && COMMON+=(--dry-run)
[[ "$NO_VIDEO" -eq 1 ]] && COMMON+=(--no-video)

if [[ "$ORACLE" -eq 1 ]]; then
  COMMON+=(
    --config configs/backends/libero/oracle/stop_wait_resume.yaml
    --set extra.libero.oracle.enabled=true
  )
fi
if [[ "$AEGIS" -eq 1 ]]; then
  COMMON+=(
    --config configs/backends/libero/aegis/enable.yaml
    --set extra.libero.aegis.enabled=true
  )
fi
COMMON+=("${EXTRA_ARGS[@]}")

info "debug eval plan"
echo "  backend=$BACKEND  model=$MODEL  suite=$SUITE"
echo "  levels=${LEVEL_LIST[*]}  tasks=${TASK_MODE}:${TASKS}  episodes=$EPISODES  seed=$SEED"
echo "  oracle=$ORACLE  aegis=$AEGIS  dry_run=$DRY_RUN"
echo

fail=0
n_jobs=0
if [[ "$TASK_MODE" == "all" ]]; then
  for lv in "${LEVEL_LIST[@]}"; do
    n_jobs=$((n_jobs + 1))
    info "[$n_jobs] eval level=$lv task=ALL"
    if ! bash "$ROOT/scripts/run/10_eval.sh" "$BACKEND" "$MODEL" "$SUITE" "$lv" "${COMMON[@]}"; then
      warn "failed: level=$lv task=ALL"
      fail=1
    fi
  done
else
  for lv in "${LEVEL_LIST[@]}"; do
    for tid in "${TASK_LIST[@]}"; do
      n_jobs=$((n_jobs + 1))
      info "[$n_jobs] eval level=$lv task=$tid"
      if ! bash "$ROOT/scripts/run/10_eval.sh" "$BACKEND" "$MODEL" "$SUITE" "$lv" \
          --task "$tid" "${COMMON[@]}"; then
        warn "failed: level=$lv task=$tid"
        fail=1
      fi
    done
  done
fi

if [[ "$fail" -ne 0 ]]; then
  die "debug eval finished with failures ($n_jobs jobs). See results/ and stderr above."
fi
ok "debug eval done ($n_jobs jobs). results under $ROOT/results/"
