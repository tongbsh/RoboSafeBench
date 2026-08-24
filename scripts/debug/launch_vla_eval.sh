#!/usr/bin/env bash
# Debug VLA eval 矩阵 — 等 serve ready 后按 suite × task × level 循环 dynisafety eval。
#
# 改编自 artifacts/feasibility/pi05_base_full_alignv2/launch_eval.sh
# 与 scripts/feasibility/run_pi05_base_full_eval.sh。
#
# 已接通: libero / robotwin / simplerenv。
# 论文 X-VLA：RoboTwin / SimplerEnv 用 --model xvla_official（不要用 xvla）。
#
#   bash scripts/debug/launch_vla_eval.sh --model pi05 --backend libero --level 1
#   bash scripts/debug/launch_vla_eval.sh --model xvla_official --backend robotwin --level 0
#   bash scripts/debug/launch_vla_eval.sh --model xvla_official --backend simplerenv --level 1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/run/00_env.sh"
cd "$ROOT"

BACKEND="libero"
MODEL=""
SUITE_SPEC="all"
LEVEL_SPEC="1,2,3"
TASK_SPEC="all"
EPISODES="1"
SEED="7"
HOST="127.0.0.1"
PORT=""
OUT_DIR=""
WAIT_SERVE=1
SAVE_VIDEO=1
ORACLE=0
AEGIS=0
ALIGN=1
DIVERSITY=0
OBSTACLE_PACK=0
EXTRA_SETS=()
START_SUITE=""
START_TASK=""
CONTINUE_ON_ERROR=0

# RoboTwin main20 catalog ids（不是 0..19）。权威：configs/backends/robotwin/suites/main20.yaml
ROBOTWIN_MAIN20_TASKS=(0 1 2 5 8 11 12 13 14 15 16 19 20 21 29 31 42 45 46 48)
# SimplerEnv paper Bridge：carrot / stack / spoon（不含 eggplant=3）
SIMPLERENV_PAPER_TASKS=(0 1 2)

usage() {
  cat <<'EOF'
Debug VLA eval 矩阵（写 eval.screen.log / progress.log，适合 screen）

  bash scripts/debug/launch_vla_eval.sh --model pi05 --backend libero --level 1 --episodes 1
  bash scripts/debug/launch_vla_eval.sh --model xvla_official --backend robotwin --level 0 --task 21
  bash scripts/debug/launch_vla_eval.sh --model xvla_official --backend simplerenv --level 1 --task 0

选项:
  --backend NAME     libero | robotwin | simplerenv（默认: libero）
  --model STEM       默认随 backend：libero→pi05，其它→xvla_official
  --suite LIST|all   libero: spatial,object,goal,long
                     robotwin: main20
                     simplerenv: bridge
  --level LIST|all   0..3 逗号列表（默认: 1,2,3）
  --task  LIST|all   task id（robotwin 是 catalog id，如 21=place_a2b_right）
  --episodes N       每格轨迹数（默认: 1）
  --seed N           基种子（默认: 7）
  --host HOST        默认: 127.0.0.1
  --port N           默认: xvla* → 8010，其它 → 8011
  --out DIR          默认: artifacts/debug/<model>_<backend>
  --no-wait          不等 [serve] ready
  --no-video         不存视频
  --oracle / --aegis 仅 libero
  --no-align         仅 libero：关闭 align_obstacle_with_safelibero
  --diversity        仅 libero：打开 L2 spawn_diversity 与 L3 reach_diversity
  --obstacle-pack    仅 libero：在空槽上增加体量相近的 GSO（不换官方种类）
  --set KEY=VALUE    透传 dynisafety --set（可重复）
  --start-suite ID / --start-task N   续跑
  --continue-on-error
  -h | --help
EOF
}

die_usage() { echo "error: $*" >&2; usage >&2; exit 2; }

backend_matrix() {
  case "$1" in
    libero)
      DEFAULT_SUITES=(safelibero_spatial safelibero_object safelibero_goal safelibero_long)
      DEFAULT_TASKS=(0 1 2 3)
      DEFAULT_MODEL="pi05"
      ;;
    robotwin)
      DEFAULT_SUITES=(main20)
      DEFAULT_TASKS=("${ROBOTWIN_MAIN20_TASKS[@]}")
      DEFAULT_MODEL="xvla_official"
      ;;
    simplerenv)
      DEFAULT_SUITES=(bridge)
      DEFAULT_TASKS=("${SIMPLERENV_PAPER_TASKS[@]}")
      DEFAULT_MODEL="xvla_official"
      ;;
    robocasa)
      die_usage "backend 'robocasa' 请用 scripts/debug/eval_debug.sh"
      ;;
    *)
      die_usage "invalid --backend '$1'"
      ;;
  esac
}

normalize_suite() {
  local backend="$1" raw="$2"
  case "$backend" in
    libero)
      case "$raw" in
        spatial|safelibero_spatial) printf '%s' "safelibero_spatial" ;;
        object|safelibero_object)   printf '%s' "safelibero_object" ;;
        goal|safelibero_goal)       printf '%s' "safelibero_goal" ;;
        long|safelibero_long)       printf '%s' "safelibero_long" ;;
        *)                          printf '%s' "$raw" ;;
      esac
      ;;
    robotwin)
      case "$raw" in
        main20|robotwin|rt) printf '%s' "main20" ;;
        *)                  printf '%s' "$raw" ;;
      esac
      ;;
    simplerenv)
      case "$raw" in
        bridge|simpler) printf '%s' "bridge" ;;
        *)              printf '%s' "$raw" ;;
      esac
      ;;
    *)
      printf '%s' "$raw"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --backend) BACKEND="${2:?}"; shift 2 ;;
    --model)   MODEL="${2:?}"; shift 2 ;;
    --suite)   SUITE_SPEC="${2:?}"; shift 2 ;;
    --level)   LEVEL_SPEC="${2:?}"; shift 2 ;;
    --task)    TASK_SPEC="${2:?}"; shift 2 ;;
    --episodes) EPISODES="${2:?}"; shift 2 ;;
    --seed)    SEED="${2:?}"; shift 2 ;;
    --host)    HOST="${2:?}"; shift 2 ;;
    --port)    PORT="${2:?}"; shift 2 ;;
    --out)     OUT_DIR="${2:?}"; shift 2 ;;
    --no-wait) WAIT_SERVE=0; shift ;;
    --no-video) SAVE_VIDEO=0; shift ;;
    --oracle)  ORACLE=1; shift ;;
    --aegis)   AEGIS=1; shift ;;
    --no-align) ALIGN=0; shift ;;
    --diversity) DIVERSITY=1; shift ;;
    --obstacle-pack) OBSTACLE_PACK=1; shift ;;
    --set)
      EXTRA_SETS+=("${2:?}")
      shift 2
      ;;
    --start-suite) START_SUITE="${2:?}"; shift 2 ;;
    --start-task)  START_TASK="${2:?}"; shift 2 ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    *) die_usage "unknown argument: $1" ;;
  esac
done

backend_matrix "$BACKEND"
MODEL="${MODEL:-$DEFAULT_MODEL}"

if [[ "$ORACLE" -eq 1 || "$AEGIS" -eq 1 || "$OBSTACLE_PACK" -eq 1 ]] && [[ "$BACKEND" != "libero" ]]; then
  die_usage "--oracle / --aegis / --obstacle-pack 仅 libero 有意义"
fi

OUT_DIR="${OUT_DIR:-$ROOT/artifacts/debug/${MODEL}_${BACKEND}}"
RESULTS_ROOT="${RESULTS_ROOT:-$OUT_DIR/results}"
HOST="${HOST:-127.0.0.1}"
if [[ -z "$PORT" ]]; then
  case "$MODEL" in
    xvla|xvla_official) PORT="8010" ;;
    *) PORT="8011" ;;
  esac
fi
EPISODES="${EPISODES:-1}"
if [[ -n "${LEVELS:-}" && "$LEVEL_SPEC" == "1,2,3" ]]; then
  LEVEL_SPEC="${LEVELS// /,}"
fi

if [[ "$LEVEL_SPEC" == "all" ]]; then
  LEVEL_LIST=(0 1 2 3)
else
  IFS=',' read -r -a LEVEL_LIST <<< "$LEVEL_SPEC"
fi
for lv in "${LEVEL_LIST[@]}"; do
  [[ "$lv" =~ ^[0-3]$ ]] || die_usage "invalid level '$lv'"
done

if [[ "$SUITE_SPEC" == "all" ]]; then
  SUITE_LIST=("${DEFAULT_SUITES[@]}")
else
  SUITE_LIST=()
  IFS=',' read -r -a _raw_suites <<< "$SUITE_SPEC"
  for s in "${_raw_suites[@]}"; do
    SUITE_LIST+=("$(normalize_suite "$BACKEND" "$s")")
  done
fi

if [[ "$TASK_SPEC" == "all" ]]; then
  TASK_LIST=("${DEFAULT_TASKS[@]}")
else
  IFS=',' read -r -a TASK_LIST <<< "$TASK_SPEC"
  for t in "${TASK_LIST[@]}"; do
    [[ "$t" =~ ^[0-9]+$ ]] || die_usage "invalid task id '$t'"
  done
fi

[[ "$EPISODES" =~ ^[0-9]+$ ]] || die_usage "--episodes must be int"
[[ "$SEED" =~ ^[0-9]+$ ]] || die_usage "--seed must be int"

if [[ "$MODEL" == "xvla" && "$BACKEND" != "libero" ]]; then
  die_usage "stem 'xvla' 只挂 LIBERO；RoboTwin / SimplerEnv 请用 --model xvla_official"
fi

mkdir -p "$OUT_DIR" "$RESULTS_ROOT"
exec > >(tee -a "$OUT_DIR/eval.screen.log") 2>&1

PROGRESS="${PROGRESS:-$OUT_DIR/progress.log}"

if [[ "$WAIT_SERVE" -eq 1 ]] && dyni_needs_server "$MODEL"; then
  echo "$(date -Is) waiting for serve at ${HOST}:${PORT} (log: $OUT_DIR/serve.screen.log)"
  ready=0
  # X-VLA 等大权重启动常超过 6 min；与 xvla_official.yaml startup_timeout_s=600 对齐
  wait_iters="${DYNI_SERVE_WAIT_ITERS:-360}"
  for _ in $(seq 1 "$wait_iters"); do
    if grep -q '\[serve\] ready' "$OUT_DIR/serve.screen.log" 2>/dev/null; then
      ready=1
      break
    fi
    if python -c "import socket; socket.create_connection(('${HOST}', int('${PORT}')), 1).close()" \
        >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "$(date -Is) FAIL serve ready timeout at ${HOST}:${PORT}" >&2
    echo "start serve: bash scripts/debug/launch_vla_serve.sh --model $MODEL --backend $BACKEND --port $PORT --out $OUT_DIR" >&2
    exit 1
  fi
  echo "$(date -Is) serve ready"
fi

dyni_activate_for "$MODEL" eval "$BACKEND"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HOST PORT
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [[ "$SAVE_VIDEO" -eq 1 ]]; then
  export DYNI_SAVE_VIDEO=1
else
  export DYNI_SAVE_VIDEO=0
fi

EVAL_EXTRA=()
ORACLE_ENABLED=false
AEGIS_ENABLED=false
ALIGN_VAL=false
DIVERSITY_VAL=false
OBSTACLE_PACK_VAL=false
if [[ "$BACKEND" == "libero" ]]; then
  if [[ "$ORACLE" -eq 1 ]]; then
    ORACLE_ENABLED=true
    EVAL_EXTRA+=(--config configs/backends/libero/oracle/stop_wait_resume.yaml)
  fi
  if [[ "$AEGIS" -eq 1 ]]; then
    AEGIS_ENABLED=true
    EVAL_EXTRA+=(--config configs/backends/libero/aegis/enable.yaml)
  fi
  [[ "$ALIGN" -eq 1 ]] && ALIGN_VAL=true
  [[ "$DIVERSITY" -eq 1 ]] && DIVERSITY_VAL=true
  [[ "$OBSTACLE_PACK" -eq 1 ]] && OBSTACLE_PACK_VAL=true
  EVAL_EXTRA+=(
    --set extra.libero.oracle.enabled="$ORACLE_ENABLED"
    --set extra.libero.aegis.enabled="$AEGIS_ENABLED"
    --set extra.libero.align_obstacle_with_safelibero="$ALIGN_VAL"
    --set extra.libero.l2.spawn_diversity.enabled="$DIVERSITY_VAL"
    --set extra.libero.l3.reach_diversity.enabled="$DIVERSITY_VAL"
    --set extra.libero.obstacle_pack.enabled="$OBSTACLE_PACK_VAL"
  )
fi
EVAL_EXTRA+=(
  --set extra.server_host="$HOST"
  --set extra.server_port="$PORT"
  --set extra.feasibility_condition=base
)
for kv in "${EXTRA_SETS[@]}"; do
  EVAL_EXTRA+=(--set "$kv")
done

echo "$(date -Is) start backend=$BACKEND model=$MODEL episodes=$EPISODES host=$HOST port=$PORT levels='${LEVEL_LIST[*]}' suites='${SUITE_LIST[*]}' tasks='${TASK_LIST[*]}' oracle=$ORACLE_ENABLED aegis=$AEGIS_ENABLED align=$ALIGN_VAL diversity=${DIVERSITY_VAL:-false} start_suite='${START_SUITE:-}' start_task='${START_TASK:-}'" \
  | tee -a "$PROGRESS"

started=1
if [[ -n "$START_SUITE" ]]; then
  started=0
  START_SUITE="$(normalize_suite "$BACKEND" "$START_SUITE")"
  START_TASK="${START_TASK:-${TASK_LIST[0]}}"
fi

for suite in "${SUITE_LIST[@]}"; do
  for task in "${TASK_LIST[@]}"; do
    if [[ "$started" == "0" ]]; then
      if [[ "$suite" == "$START_SUITE" && "$task" == "${START_TASK}" ]]; then
        started=1
      else
        echo "$(date -Is) SKIP  suite=$suite task=$task (before resume $START_SUITE/${START_TASK})" \
          | tee -a "$PROGRESS"
        continue
      fi
    fi
    for level in "${LEVEL_LIST[@]}"; do
      echo "$(date -Is) BEGIN suite=$suite task=$task level=$level" | tee -a "$PROGRESS"
      set +e
      python -m dynisafety.cli eval \
        --backend "$BACKEND" --model "$MODEL" --suite "$suite" \
        --task "$task" --level "$level" --seed "$SEED" --episodes "$EPISODES" \
        --output-root "$RESULTS_ROOT" \
        "${EVAL_EXTRA[@]}"
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
ok "debug VLA eval finished → $RESULTS_ROOT"
