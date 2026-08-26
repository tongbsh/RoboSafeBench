#!/usr/bin/env bash
# Full LIBERO L1/L2 × 4 suites × 4 tasks × 50 ep: π₀.₅ + Oracle + AEGIS vision.
# Assumes serve is already up on HOST:PORT (default 127.0.0.1:8011).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/env.sh"

OUT_DIR="${OUT_DIR:-$(dyni_artifacts_dir feasibility pi05_aegis_vision_full)}"
RESULTS_ROOT="${RESULTS_ROOT:-$(dyni_results_root)}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8011}"
EPISODES="${EPISODES:-50}"
# Space-separated levels, e.g. "1 2" (default) or "1" / "2" for parallel jobs.
LEVELS="${LEVELS:-1 2}"
# Inclusive resume point, e.g. START_SUITE=safelibero_spatial START_TASK=2
START_SUITE="${START_SUITE:-}"
START_TASK="${START_TASK:-}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DINO_CFG="${DINO_CFG:-$ROOT/artifacts/feasibility/aegis_vision_gpu_smoke/GroundingDINO_SwinT_OGC.py}"
PROGRESS="${PROGRESS:-$OUT_DIR/progress.log}"

mkdir -p "$OUT_DIR" "$RESULTS_ROOT"
dyni_activate_for pi05 eval libero
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HOST PORT
export DYNI_SAVE_VIDEO=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

[[ -f "$DINO_CFG" ]] || die "missing DINO sidecar: $DINO_CFG"

echo "$(date -Is) start episodes=$EPISODES host=$HOST port=$PORT levels='$LEVELS' align_obstacle_with_safelibero=${ALIGN_OBSTACLE_WITH_SAFELIBERO:-true} start_suite='${START_SUITE:-}' start_task='${START_TASK:-}' continue_on_error=$CONTINUE_ON_ERROR" | tee -a "$PROGRESS"

started=1
if [[ -n "$START_SUITE" ]]; then
  started=0
fi

for suite in safelibero_spatial safelibero_object safelibero_goal safelibero_long; do
  for task in 0 1 2 3; do
    if [[ "$started" == "0" ]]; then
      if [[ "$suite" == "$START_SUITE" && "$task" == "$START_TASK" ]]; then
        started=1
      else
        echo "$(date -Is) SKIP  suite=$suite task=$task (before resume point $START_SUITE/$START_TASK)" | tee -a "$PROGRESS"
        continue
      fi
    fi
    for level in $LEVELS; do
      echo "$(date -Is) BEGIN suite=$suite task=$task level=$level" | tee -a "$PROGRESS"
      set +e
      python -m dynisafety.cli eval \
        --backend libero --model pi05 --suite "$suite" \
        --task "$task" --level "$level" --seed 7 --episodes "$EPISODES" \
        --output-root "$RESULTS_ROOT" \
        --config configs/backends/libero/oracle/stop_wait_resume.yaml \
        --config configs/backends/libero/aegis/enable.yaml \
        --config configs/backends/libero/aegis/vision.yaml \
        --set extra.libero.oracle.enabled=true \
        --set extra.libero.aegis.enabled=true \
        --set extra.libero.aegis.vision.grounding_dino_config="$DINO_CFG" \
        --set extra.server_host="$HOST" \
        --set extra.server_port="$PORT" \
        --set extra.feasibility_condition=oracle_swr_aegis \
        --set extra.libero.align_obstacle_with_safelibero="${ALIGN_OBSTACLE_WITH_SAFELIBERO:-true}"
      rc=$?
      set -e
      if [[ "$rc" -eq 0 ]]; then
        echo "$(date -Is) DONE  suite=$suite task=$task level=$level" | tee -a "$PROGRESS"
      else
        echo "$(date -Is) FAIL  suite=$suite task=$task level=$level rc=$rc" | tee -a "$PROGRESS"
        if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
          exit "$rc"
        fi
      fi
    done
  done
done

echo "$(date -Is) ALL_JOBS_FINISHED" | tee -a "$PROGRESS"
ok "full pi05 AEGIS vision eval finished → $RESULTS_ROOT"
dyni_print_output_roots feasibility pi05_aegis_vision_full
