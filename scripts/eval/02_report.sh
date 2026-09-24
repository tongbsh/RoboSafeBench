#!/usr/bin/env bash
# Aggregate results into the regular report tables.
# Reads out/eval/ (eval-root); writes out/reports/{grid}/.
#
#   bash scripts/eval/02_report.sh                # validate coverage, then generate
#   bash scripts/eval/02_report.sh --validate     # coverage check only
#   bash scripts/eval/02_report.sh --allow-incomplete
#   bash scripts/eval/02_report.sh --results results   # read a legacy tree
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

RESULTS="$(dyni_eval_read_root)"
OUT="${DYNI_REPORTS:-$(dyni_paths reports-root)}"
EXTRA=()
VALIDATE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    --validate)
      VALIDATE_ONLY=1
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
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

dyni_activate "$DYNI_TOOL_ENV"

SETTINGS="${DYNI_REPORT_SETTINGS:-$DYNI_ROOT/configs/report/report_settings.yaml}"
METRICS="${DYNI_METRICS_YAML:-$DYNI_ROOT/configs/base.yaml}"
GRID_DIR="${DYNI_GRID_DIR:-$DYNI_ROOT/configs/report/grids}"

# One grid per (backend, suite) — backends differ in tasks/models/episodes.
if [[ -n "${DYNI_GRID:-}" ]]; then
  GRIDS=("$DYNI_GRID")
else
  mapfile -t GRIDS < <(find "$GRID_DIR" -name '*.yaml' | sort)
fi
if [[ ${#GRIDS[@]} -eq 0 ]]; then
  die "no coverage grid found under $GRID_DIR — run: python tools/make_report_grids.py"
fi
[[ -f "$SETTINGS" ]] || die "missing report settings: $SETTINGS"
[[ -d "$RESULTS" ]] || die "eval-root not found: $RESULTS (run scripts/eval/01_eval.sh first)"

info "report  results=$RESULTS  out=$OUT/<grid>/"

for grid in "${GRIDS[@]}"; do
  name="$(basename "$grid" .yaml)"
  info "coverage validation: $name"
  dyni_cli report validate --results "$RESULTS" --out "$OUT/$name" \
      --grid "$grid" --report-yaml "$SETTINGS" \
    || warn "coverage gaps for $name (see $OUT/$name/coverage.csv)"

  [[ "$VALIDATE_ONLY" == 1 ]] && continue

  info "generating tables: $name"
  dyni_cli report generate --results "$RESULTS" --out "$OUT/$name" \
      --grid "$grid" --report-yaml "$SETTINGS" --metrics-yaml "$METRICS" "${EXTRA[@]}" \
    || warn "report generate failed for $name (coverage incomplete? add --allow-incomplete)"
done

if [[ "$VALIDATE_ONLY" == 1 ]]; then
  ok "validation only; coverage written under $OUT/<grid>/"
  exit 0
fi
ok "reports written under $OUT/<grid>/"
