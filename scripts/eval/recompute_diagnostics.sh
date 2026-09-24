#!/usr/bin/env bash
# Offline rewrite of EpisodeRecord.diagnostics from traces/.
# Loops every completed run under the eval root. Does not step a simulator
# and does not touch traces/.
#
#   bash scripts/eval/recompute_diagnostics.sh
#   bash scripts/eval/recompute_diagnostics.sh --dry-run
#   bash scripts/eval/recompute_diagnostics.sh --results out/eval/libero
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

RESULTS="$(dyni_eval_read_root)"
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
    --base-yaml)
      BASE_YAML="$2"
      shift 2
      ;;
    --base-yaml=*)
      BASE_YAML="${1#*=}"
      shift
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

[[ -d "$RESULTS" ]] || die "eval-root not found: $RESULTS (run scripts/eval/01_eval.sh first)"
[[ -f "$BASE_YAML" ]] || die "missing diagnostics YAML: $BASE_YAML"

dyni_activate "$DYNI_TOOL_ENV"
info "diagnostics recompute  results=$RESULTS  base-yaml=$BASE_YAML"
dyni_cli diagnostics recompute --results "$RESULTS" --base-yaml "$BASE_YAML" "${EXTRA[@]}"
ok "next: bash scripts/eval/02_report.sh && bash scripts/eval/03_analysis.sh"
