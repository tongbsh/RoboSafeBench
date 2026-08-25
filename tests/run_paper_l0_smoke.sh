#!/usr/bin/env bash
# Paper L0 VLA success-rate smoke.
#
# Runs every paper_main (model × backend × suite × task) cell at **level 0**
# with **5 episodes per task**. Server lifecycle is handled by `dynisafety sweep`.
#
# Usage:
#   bash tests/run_paper_l0_smoke.sh --dry-run          # plan only (always first)
#   bash tests/run_paper_l0_smoke.sh                    # full L0 smoke
#   bash tests/run_paper_l0_smoke.sh --resume           # skip finished jobs
#   bash tests/run_paper_l0_smoke.sh --only backend=libero
#   bash tests/run_paper_l0_smoke.sh --only model=starvla
#   bash tests/run_paper_l0_smoke.sh --summarize        # print SR table from results
#   DYNI_GPUS=0 bash tests/run_paper_l0_smoke.sh
#
# Suite definition: tests/paper_l0_smoke.yaml
#   ~221 jobs / 1105 episodes (libero 112 + robotwin 100 + simplerenv 9)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/lib/env.sh
source "${SCRIPT_DIR}/../scripts/lib/env.sh"

SUITE_YAML="${SCRIPT_DIR}/paper_l0_smoke.yaml"
[[ -f "$SUITE_YAML" ]] || die "missing suite: $SUITE_YAML"

SUMMARIZE_ONLY=0
SWEEP_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --summarize|--summarize-only)
      SUMMARIZE_ONLY=1
      shift
      ;;
    --dry-run|--resume|--retry-failed)
      SWEEP_ARGS+=("$1")
      shift
      ;;
    --only)
      [[ $# -ge 2 ]] || die "--only needs a value (e.g. backend=libero)"
      SWEEP_ARGS+=(--only "$2")
      shift 2
      ;;
    --gpus)
      [[ $# -ge 2 ]] || die "--gpus needs a value"
      DYNI_GPUS="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      die "unknown arg: $1 (see --help)"
      ;;
  esac
done

summarize_l0() {
  dyni_activate "$DYNI_TOOL_ENV"
  RESULTS="${DYNI_RESULTS:-$DYNI_ROOT/results}"
  python - "$RESULTS" "$SUITE_YAML" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

results_root = Path(sys.argv[1])
suite_yaml = Path(sys.argv[2])
raw = yaml.safe_load(suite_yaml.read_text(encoding="utf-8")) or {}
paper_models = set()
paper_backends = set()
for block in raw.get("matrix") or []:
    paper_backends.add(str(block.get("backend")))
    for m in block.get("models") or []:
        paper_models.add(str(m))

# (backend, model, suite) -> succ, ep, runs, err_eps
stats: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
skipped = 0

if not results_root.is_dir():
    print(f"no results dir: {results_root}", file=sys.stderr)
    sys.exit(1)

for summary_path in sorted(results_root.glob("*/summary.json")):
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        skipped += 1
        continue
    if int(summary.get("level", -1)) != 0:
        continue
    backend = str(summary.get("backend") or "")
    model = str(summary.get("model") or "")
    suite = str(summary.get("suite") or "")
    if model not in paper_models or backend not in paper_backends:
        continue

    n_succ = int(summary.get("n_successes") or 0)
    n_ep = int(summary.get("n_episodes_written") or 0)
    n_err = 0
    ep_path = summary_path.parent / "episodes.jsonl"
    if ep_path.is_file():
        n_succ = 0
        n_ep = 0
        for line in ep_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            n_ep += 1
            if rec.get("success"):
                n_succ += 1
            steps = int(rec.get("steps") or 0)
            if rec.get("termination") == "error" or (steps == 0 and not rec.get("success")):
                n_err += 1
    if n_ep <= 0:
        skipped += 1
        continue

    key = (backend, model, suite)
    stats[key][0] += n_succ
    stats[key][1] += n_ep
    stats[key][2] += 1
    stats[key][3] += n_err

print(
    f"{'BACKEND':<12}{'MODEL':<22}{'SUITE':<22}"
    f"{'SR':>8}{'SUCC':>6}{'EP':>6}{'RUNS':>6}{'ERR':>6}"
)
print("-" * 88)
if not stats:
    print("(no paper-model L0 results found under results/)")
else:
    for key in sorted(stats):
        backend, model, suite = key
        succ, ep, runs, err = stats[key]
        sr = f"{100.0 * succ / ep:.1f}%" if ep else "n/a"
        print(f"{backend:<12}{model:<22}{suite:<22}{sr:>8}{succ:>6}{ep:>6}{runs:>6}{err:>6}")
print("-" * 88)
print(f"cells={len(stats)}  skipped={skipped}")
print("ERR = termination=error or steps=0 (and not success). Investigate ERR first.")
PY
}

if [[ "$SUMMARIZE_ONLY" == 1 ]]; then
  summarize_l0
  exit 0
fi

dyni_activate "$DYNI_TOOL_ENV"

ARGS=(--suite "$SUITE_YAML")
[[ -n "${DYNI_GPUS:-}" ]] && ARGS+=(--gpus "$DYNI_GPUS")
[[ -n "${DYNI_JOBS_PER_GPU:-}" ]] && ARGS+=(--jobs-per-gpu "$DYNI_JOBS_PER_GPU")
ARGS+=("${SWEEP_ARGS[@]}")

warn "L0 smoke = paper matrix × level0 × 5 ep/task (~205 jobs / 1025 episodes)."
warn "Sweep starts/stops model servers itself. Prefer: nohup … > l0_smoke.log 2>&1 &"
info "dynisafety sweep ${ARGS[*]}"
dyni_cli sweep "${ARGS[@]}"

ok "sweep finished. Failures: results/_sweep/paper_l0_smoke/failures.jsonl"
info "Success-rate table:"
summarize_l0
ok "Done. Re-print table anytime: bash tests/run_paper_l0_smoke.sh --summarize"
