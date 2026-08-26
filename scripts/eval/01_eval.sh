#!/usr/bin/env bash
# 傻瓜评测 01：启动评测。不指定就跑 paper_main 全量；可按 backend / level / model 裁剪。
#
#   bash scripts/eval/01_eval.sh --dry-run
#   bash scripts/eval/01_eval.sh
#   bash scripts/eval/01_eval.sh --backend libero
#   bash scripts/eval/01_eval.sh --backend libero --level 2 --model pi05
#   bash scripts/eval/01_eval.sh --level 1,2,3 --model pi05,xvla
#   bash scripts/eval/01_eval.sh --resume
#   DYNI_GPUS=0,1 bash scripts/eval/01_eval.sh --backend simplerenv
#
# RoboCasa 不在 paper_main：
#   bash scripts/eval/01_eval.sh --backend robocasa
#   bash scripts/eval/01_eval.sh --backend robocasa --level 1 --suite smoke_robocasa
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

_csv_to_pipe() {
  local raw="$1"
  raw="${raw// /}"
  raw="${raw//,/|}"
  printf '%s' "$raw"
}

usage() {
  sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

标志（均可逗号分隔多个值）：
  --backend   libero|simplerenv|robotwin|robocasa
  --level     0|1|2|3
  --model     configs/models/*.yaml 的 stem
  --suite     默认 paper_main（robocasa 单独跑时默认 smoke_robocasa）
  --task      任务 id
  --dry-run / --resume / --retry-failed
  其余参数原样传给 dynisafety sweep

不指定 backend/level/model 时跑 configs/suites/paper_main.yaml 全量
（约 884 job，务必先 --dry-run）。
EOF
}

SUITE="${DYNI_SUITE:-paper_main}"
BACKENDS=""
LEVELS=""
MODELS=""
TASKS=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --backend)
      BACKENDS="$2"
      shift 2
      ;;
    --backend=*)
      BACKENDS="${1#*=}"
      shift
      ;;
    --level)
      LEVELS="$2"
      shift 2
      ;;
    --level=*)
      LEVELS="${1#*=}"
      shift
      ;;
    --model)
      MODELS="$2"
      shift 2
      ;;
    --model=*)
      MODELS="${1#*=}"
      shift
      ;;
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --suite=*)
      SUITE="${1#*=}"
      shift
      ;;
    --task)
      TASKS="$2"
      shift 2
      ;;
    --task=*)
      TASKS="${1#*=}"
      shift
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

ONLY_PARTS=()
[[ -n "$BACKENDS" ]] && ONLY_PARTS+=("backend=$(_csv_to_pipe "$BACKENDS")")
[[ -n "$LEVELS" ]] && ONLY_PARTS+=("level=$(_csv_to_pipe "$LEVELS")")
[[ -n "$MODELS" ]] && ONLY_PARTS+=("model=$(_csv_to_pipe "$MODELS")")
[[ -n "$TASKS" ]] && ONLY_PARTS+=("task=$(_csv_to_pipe "$TASKS")")

if [[ -n "$BACKENDS" && "$BACKENDS" == *robocasa* ]]; then
  if [[ "$SUITE" == "paper_main" ]]; then
    if [[ "$(_csv_to_pipe "$BACKENDS")" == "robocasa" ]]; then
      SUITE="${DYNI_ROBOCASA_SUITE:-smoke_robocasa}"
      warn "robocasa 不在 paper_main，改用 suite=$SUITE"
    else
      die "robocasa 不在 paper_main，请单独跑: bash scripts/eval/01_eval.sh --backend robocasa"
    fi
  fi
fi

ONLY=""
if [[ ${#ONLY_PARTS[@]} -gt 0 ]]; then
  ONLY="$(IFS=','; echo "${ONLY_PARTS[*]}")"
fi

export DYNI_SUITE="$SUITE"
LOG_DIR="$(dyni_artifacts_dir eval "01_${SUITE}")"
dyni_tee_logs "$LOG_DIR/eval.screen.log"
info "process log: $LOG_DIR/eval.screen.log"
info "suite=$SUITE${ONLY:+  only=$ONLY}"
if [[ ${#ONLY_PARTS[@]} -eq 0 && "$SUITE" == "paper_main" ]]; then
  warn "未指定 --backend/--level/--model：将跑 paper_main 全量。先加 --dry-run 看计划。"
fi

SWEEP_ARGS=()
[[ -n "$ONLY" ]] && SWEEP_ARGS+=(--only "$ONLY")
SWEEP_ARGS+=("${EXTRA[@]}")

bash "$DYNI_ROOT/scripts/eval/sweep.sh" "${SWEEP_ARGS[@]}"
ok "next: bash scripts/eval/02_report.sh   # 或先 --allow-incomplete 看部分结果"
