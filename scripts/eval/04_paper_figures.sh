#!/usr/bin/env bash
# 傻瓜评测 04：ICLR 论文插图（src/dynisafety/paper_figures）。
# 只读 out/eval/ 与 out/analysis/，写到 out/paper_figures/。不改 report generate。
#
#   bash scripts/eval/04_paper_figures.sh
#   bash scripts/eval/04_paper_figures.sh --overview-only
#   bash scripts/eval/04_paper_figures.sh --figures fig03,fig04,fig07
#   bash scripts/eval/04_paper_figures.sh --results results --out out/paper_figures
#   bash scripts/eval/04_paper_figures.sh --analysis-root out/analysis
#
# fig09 / figA5 用配对四轴（docs/user/eval.md §2.5），读的是
# <analysis-root>/{backend}/paired_axes/paired_axes_episodes.csv，
# 所以这一步依赖先跑 `bash scripts/eval/03_analysis.sh paired-axes`。
# CSV 缺失时那两张图渲染 n/a 占位，不会静默出 0。
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

RESULTS="$(dyni_eval_read_root)"
ANALYSIS="${DYNI_ANALYSIS:-$(dyni_paths analysis-root)}"
OUT="${DYNI_PAPER_FIGURES:-$(dyni_paths out)/paper_figures}"
SUITE_YAML="${DYNI_SUITE_YAML:-$DYNI_ROOT/configs/suites/paper_main.yaml}"
SETTINGS="${DYNI_PAPER_SETTINGS:-$DYNI_ROOT/configs/paper_figures/settings.yaml}"
PEERS="${DYNI_PAPER_PEERS:-$DYNI_ROOT/configs/paper_figures/peers.yaml}"
DISPLAY="${DYNI_PAPER_DISPLAY:-$DYNI_ROOT/configs/paper_figures/display.yaml}"
EXTRA=()
OVERVIEW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    --analysis-root)
      ANALYSIS="$2"
      shift 2
      ;;
    --analysis-root=*)
      ANALYSIS="${1#*=}"
      shift
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
    --overview-only)
      OVERVIEW=1
      EXTRA+=(--overview-only)
      shift
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

[[ -f "$SUITE_YAML" ]] || die "missing suite yaml: $SUITE_YAML"
[[ -f "$SETTINGS" ]] || die "missing settings yaml: $SETTINGS"
if [[ "$OVERVIEW" -eq 0 ]]; then
  [[ -d "$RESULTS" ]] || die "eval-root not found: $RESULTS (run scripts/eval/01_eval.sh first)"
  [[ -d "$ANALYSIS" ]] || warn "analysis-root not found: $ANALYSIS (fig09/figA5 → n/a; run scripts/eval/03_analysis.sh paired-axes)"
fi

dyni_activate "$DYNI_TOOL_ENV"
info "paper-figures  results=$RESULTS  analysis=$ANALYSIS  out=$OUT"
ARGS=(
  paper-figures
  --out "$OUT"
  --suite-yaml "$SUITE_YAML"
  --settings-yaml "$SETTINGS"
  --peers-yaml "$PEERS"
  --display-yaml "$DISPLAY"
  --analysis-root "$ANALYSIS"
)
if [[ "$OVERVIEW" -eq 0 ]]; then
  ARGS+=(--results "$RESULTS")
fi
dyni_cli "${ARGS[@]}" "${EXTRA[@]}"
ok "paper figures written under $OUT"
info "next: 看 $OUT/catalog.md 与 $OUT/fig*.pdf"
