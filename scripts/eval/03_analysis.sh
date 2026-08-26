#!/usr/bin/env bash
# 傻瓜评测 03：一键离线分析（src/dynisafety/analysis）。
# 只读 out/eval/，写到 out/analysis/（可用 DYNI_ANALYSIS 改）。
#
#   bash scripts/eval/03_analysis.sh
#   bash scripts/eval/03_analysis.sh termination
#   bash scripts/eval/03_analysis.sh rank --metrics ssr,sr
#   bash scripts/eval/03_analysis.sh --results results --out reports/analysis
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

CMD="all"
RESULTS="${DYNI_RESULTS:-$(dyni_paths eval-root)}"
OUT="${DYNI_ANALYSIS:-$(dyni_paths analysis-root)}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    --results)
      RESULTS="$2"
      shift 2
      ;;
    --results=*)
      RESULTS="${1#*=}"
      shift
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    --out=*)
      OUT="${1#*=}"
      shift
      ;;
    all|termination|exemplars|diversity|kinematics|rank)
      CMD="$1"
      shift
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

[[ -d "$RESULTS" ]] || die "results dir not found: $RESULTS (run scripts/eval/01_eval.sh first)"

dyni_activate "$DYNI_TOOL_ENV"
info "analysis $CMD  results=$RESULTS  out=$OUT"
dyni_cli analysis "$CMD" --results "$RESULTS" --out "$OUT" "${EXTRA[@]}"
ok "analysis written under $OUT"
info "next: 看 $OUT/{termination,exemplars,diversity,kinematics,rank}/"
