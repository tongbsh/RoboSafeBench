#!/usr/bin/env bash
# P0 alignment gate: seed-warm → HTTP baseline → parallel-eval → compare.
#
#   bash scripts/debug/robotwin/launch_xvla_official_robotwin_serve.sh
#   bash scripts/debug/parallel_eval/p0_align.sh --task 21 --level 0 --episodes 1
#
# prompt_mode=task_name is required: official/random call np.random.choice and
# the two paths do not share a global RNG stream (plan 0.7).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/env.sh"
cd "$ROOT"

BACKEND="robotwin"
SUITE="main20"
LEVEL="0"
TASK="21"
EPISODES="1"
SEED="7"
RNG_SEED=""
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/debug/parallel_eval}"
HTTP_MODEL="xvla_official"
PAR_MODEL="xvla_official_parallel"
WAIT_SERVE=1
SKIP_HTTP=0

usage() {
  sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Flags:
  --task N          RoboTwin catalog id (default 21 = place_a2b_right)
  --level N         0..3 (default 0)
  --episodes N      default 1
  --seed N          runtime.seed / expert-check cursor (default 7). Not x1.
  --rng-seed N      extra.parallel_eval.rng_seed (per-slot x1). Omit to use YAML.
  --suite NAME      default main20
  --host / --port   HTTP serve (default 127.0.0.1:8010)
  --out DIR         artifacts (default artifacts/debug/parallel_eval)
  --no-wait         do not wait for HTTP serve
  --skip-http       only warm + parallel (no baseline)
  -h | --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --task) TASK="$2"; shift 2 ;;
    --task=*) TASK="${1#*=}"; shift ;;
    --level) LEVEL="$2"; shift 2 ;;
    --level=*) LEVEL="${1#*=}"; shift ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --episodes=*) EPISODES="${1#*=}"; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --seed=*) SEED="${1#*=}"; shift ;;
    --rng-seed) RNG_SEED="$2"; shift 2 ;;
    --rng-seed=*) RNG_SEED="${1#*=}"; shift ;;
    --suite) SUITE="$2"; shift 2 ;;
    --suite=*) SUITE="${1#*=}"; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --no-wait) WAIT_SERVE=0; shift ;;
    --skip-http) SKIP_HTTP=1; shift ;;
    *) die "unknown arg: $1" ;;
  esac
done

mkdir -p "$OUT_DIR"
WARM_JSON="$OUT_DIR/accepted_seeds_t${TASK}_l${LEVEL}_s${SEED}_n${EPISODES}.json"
HTTP_DIR_FILE="$OUT_DIR/http_run_dir.txt"
PAR_DIR_FILE="$OUT_DIR/parallel_run_dir.txt"
PROMPT_SET="extra.robotwin.prompt_mode=task_name"

ENV_CONDA="${DYNI_ENV_ROBOTWIN:-internvla_a1}"

_wait_http() {
  local url="http://${HOST}:${PORT}/openapi.json"
  local i
  info "waiting for HTTP serve at $url"
  for i in $(seq 1 180); do
    if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      ok "serve ready"
      return 0
    fi
    sleep 2
  done
  die "HTTP serve not ready at $url (start scripts/debug/robotwin/launch_xvla_official_robotwin_serve.sh)"
}

RNG_ARGS=()
if [[ -n "$RNG_SEED" ]]; then
  RNG_ARGS+=(--rng-seed "$RNG_SEED")
fi

info "P0 align  task=$TASK level=$LEVEL episodes=$EPISODES seed=$SEED rng_seed=${RNG_SEED:-<yaml>} prompt_mode=task_name"

info "1/4 seed-warm (serial cursor → accepted_seeds.json + expert cache)"
conda run --no-capture-output -n "$ENV_CONDA" python -m dynisafety.cli parallel-eval \
  --backend "$BACKEND" --model "$PAR_MODEL" --suite "$SUITE" --level "$LEVEL" \
  --task "$TASK" --episodes "$EPISODES" --seed "$SEED" \
  "${RNG_ARGS[@]}" \
  --seed-warm "$WARM_JSON" --warm-only \
  --set "$PROMPT_SET" \
  --output-root "$ROOT/results"
ok "warm file $WARM_JSON"

if [[ "$SKIP_HTTP" -eq 0 ]]; then
  if [[ "$WAIT_SERVE" -eq 1 ]]; then
    _wait_http
  fi
  info "2/4 HTTP baseline (xvla_official + /act)"
  HOST="$HOST" PORT="$PORT" conda run --no-capture-output -n "$ENV_CONDA" \
    python -m dynisafety.cli eval \
    --backend "$BACKEND" --model "$HTTP_MODEL" --suite "$SUITE" --level "$LEVEL" \
    --task "$TASK" --episodes "$EPISODES" --seed "$SEED" \
    --set "$PROMPT_SET" \
    --output-root "$ROOT/results" | tee "$OUT_DIR/http_eval.log"
  python - "$OUT_DIR/http_eval.log" "$HTTP_DIR_FILE" <<'PY'
import pathlib
import re
import sys
text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
matches = re.findall(r"dir=(\S+)", text)
if not matches:
    raise SystemExit("HTTP eval log has no dir=")
pathlib.Path(sys.argv[2]).write_text(matches[-1] + "\n", encoding="utf-8")
PY
else
  info "2/4 skip HTTP baseline"
  : > "$HTTP_DIR_FILE"
fi

info "3/4 parallel-eval (xvla_official_parallel, Infer/Env split)"
conda run --no-capture-output -n "$ENV_CONDA" python -m dynisafety.cli parallel-eval \
  --backend "$BACKEND" --model "$PAR_MODEL" --suite "$SUITE" --level "$LEVEL" \
  --task "$TASK" --episodes "$EPISODES" --seed "$SEED" \
  "${RNG_ARGS[@]}" \
  --seed-warm "$WARM_JSON" \
  --set "$PROMPT_SET" \
  --set extra.parallel_eval.enabled=true \
  --set extra.parallel_eval.num_envs=1 \
  --set extra.parallel_eval.pipeline_stage_num=1 \
  --set extra.parallel_eval.deterministic_rng=true \
  --output-root "$ROOT/results" | tee "$OUT_DIR/parallel_eval.log"
python - "$OUT_DIR/parallel_eval.log" "$PAR_DIR_FILE" <<'PY'
import pathlib
import re
import sys
text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
matches = re.findall(r"dir=(\S+)", text)
if not matches:
    raise SystemExit("parallel-eval log has no dir=")
pathlib.Path(sys.argv[2]).write_text(matches[-1] + "\n", encoding="utf-8")
PY

if [[ "$SKIP_HTTP" -eq 1 ]]; then
  ok "parallel-eval finished; HTTP skipped so no compare"
  exit 0
fi

info "4/4 compare accepted_seed / success / termination / contact clocks / trace actions"
HTTP_DIR="$(tr -d '[:space:]' < "$HTTP_DIR_FILE")"
PAR_DIR="$(tr -d '[:space:]' < "$PAR_DIR_FILE")"
python - "$HTTP_DIR" "$PAR_DIR" "$OUT_DIR/p0_compare.txt" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from dynisafety.diagnostics.reader import iter_traces

http_dir = Path(sys.argv[1])
par_dir = Path(sys.argv[2])
out_path = Path(sys.argv[3])


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def _trace_actions(run_dir: Path) -> dict[int, list[list[float]]]:
    out: dict[int, list[list[float]]] = {}
    for trace in iter_traces(run_dir, strict=False):
        actions = []
        for step in trace.steps:
            if step.action is None:
                continue
            actions.append([float(v) for v in step.action])
        out[int(trace.header.episode_index)] = actions
    return out


http_rows = _load_jsonl(http_dir / "episodes.jsonl")
par_rows = _load_jsonl(par_dir / "episodes.jsonl")
if len(http_rows) != len(par_rows):
    raise SystemExit(f"episode count {len(http_rows)} != {len(par_rows)}")

lines = [
    f"http={http_dir}",
    f"parallel={par_dir}",
    f"n={len(http_rows)}",
]
ok = True
http_actions = _trace_actions(http_dir)
par_actions = _trace_actions(par_dir)
for http, par in zip(http_rows, par_rows, strict=True):
    idx = int(http["episode_index"])
    checks = {
        "accepted_seed": (http.get("extra", {}).get("accepted_seed"), par.get("extra", {}).get("accepted_seed")),
        "success": (http.get("success"), par.get("success")),
        "termination": (http.get("termination"), par.get("termination")),
        "metrics.first_contact_step": (http.get("first_contact_step"), par.get("first_contact_step")),
        "extra.first_contact_control_step": (
            http.get("extra", {}).get("first_contact_control_step"),
            par.get("extra", {}).get("first_contact_control_step"),
        ),
        "extra.contact_control_steps": (
            http.get("extra", {}).get("contact_control_steps"),
            par.get("extra", {}).get("contact_control_steps"),
        ),
    }
    for name, (left, right) in checks.items():
        match = left == right
        ok = ok and match
        lines.append(f"ep{idx} {name}: http={left} parallel={right} {'OK' if match else 'DIFF'}")
    ha = http_actions.get(idx)
    pa = par_actions.get(idx)
    if ha is None or pa is None:
        lines.append(f"ep{idx} action_chunk: missing trace (http={ha is not None} parallel={pa is not None})")
        ok = False
    elif ha == pa:
        lines.append(f"ep{idx} action_chunk: OK steps={len(ha)}")
    else:
        ok = False
        lines.append(f"ep{idx} action_chunk: DIFF http_steps={len(ha)} parallel_steps={len(pa)}")

text = "\n".join(lines) + "\n"
out_path.write_text(text, encoding="utf-8")
print(text, end="")
if not ok:
    raise SystemExit("P0 alignment failed; see " + str(out_path))
print("P0 alignment passed")
PY

ok "compare written to $OUT_DIR/p0_compare.txt"
