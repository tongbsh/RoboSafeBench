#!/usr/bin/env bash
# 傻瓜评测 03：一键离线分析（src/dynisafety/analysis）。
# 只读 out/eval/（eval-root），写到 out/analysis/（可用 DYNI_ANALYSIS 改）。
#
#   bash scripts/eval/03_analysis.sh
#   bash scripts/eval/03_analysis.sh termination
#   bash scripts/eval/03_analysis.sh counterfactual   # 需要配对的 L0 run
#   bash scripts/eval/03_analysis.sh paired-axes      # 同上；四轴换成 L0 孪生参考系
#   bash scripts/eval/03_analysis.sh key-findings     # 跨 backend 结论清单（每 backend 再合并）
#   bash scripts/eval/03_analysis.sh rank --metrics ssr,sr
#   bash scripts/eval/03_analysis.sh --results results          # 只读旧树
#   bash scripts/eval/03_analysis.sh --results results --out out/analysis
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

CMD="all"
RESULTS="$(dyni_eval_read_root)"
OUT="${DYNI_ANALYSIS:-$(dyni_paths analysis-root)}"
# The counterfactual and paired-axes stages read diagnostics.* thresholds; they
# are skipped rather than run on implicit defaults when this file is absent.
BASE_YAML="${DYNI_METRICS_YAML:-$DYNI_ROOT/configs/base.yaml}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
    all|termination|exemplars|diversity|kinematics|rank|counterfactual|paired-axes|key-findings)
      CMD="$1"
      shift
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

[[ -d "$RESULTS" ]] || die "eval-root not found: $RESULTS (run scripts/eval/01_eval.sh first)"

dyni_activate "$DYNI_TOOL_ENV"
if [[ "$CMD" == "all" || "$CMD" == "counterfactual" || "$CMD" == "paired-axes" || "$CMD" == "key-findings" ]]; then
  if [[ -f "$BASE_YAML" ]]; then
    EXTRA+=(--base-yaml "$BASE_YAML")
  elif [[ "$CMD" != "all" ]]; then
    die "$CMD needs a diagnostics YAML: $BASE_YAML not found"
  else
    warn "no $BASE_YAML — skipping the counterfactual, paired-axes and key-findings stages"
  fi
fi
info "analysis $CMD  results=$RESULTS  out=$OUT"
if [[ "$CMD" == "key-findings" ]]; then
  backends=()
  for b in libero robotwin simplerenv; do
    if [[ -d "$RESULTS/$b" ]]; then
      backends+=("$b")
    fi
  done
  if [[ ${#backends[@]} -gt 0 ]]; then
    for b in "${backends[@]}"; do
      info "analysis key-findings backend=$b"
      dyni_cli analysis key-findings --results "$RESULTS/$b" --out "$OUT/$b" --analysis-root "$OUT/$b" "${EXTRA[@]}"
    done
    info "analysis key-findings merge"
    dyni_cli analysis key-findings --results "$RESULTS" --out "$OUT" --analysis-root "$OUT" "${EXTRA[@]}"
  else
    dyni_cli analysis "$CMD" --results "$RESULTS" --out "$OUT" --analysis-root "$OUT" "${EXTRA[@]}"
  fi
else
  dyni_cli analysis "$CMD" --results "$RESULTS" --out "$OUT" "${EXTRA[@]}"
fi
ok "analysis written under $OUT"
info "next: bash scripts/eval/04_paper_figures.sh   # ICLR 插图 → out/paper_figures/"
