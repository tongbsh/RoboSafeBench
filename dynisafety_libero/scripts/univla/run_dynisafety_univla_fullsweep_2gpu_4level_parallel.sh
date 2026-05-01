#!/usr/bin/env bash
# ============================================================
# 2-GPU / 4-level parallel full sweep: UniVLA
#
# For each selected suite, this script will:
#   1. start one UniVLA server per selected level
#   2. map levels onto two GPUs (2 model instances / GPU by default)
#   3. evaluate all selected levels in parallel
#   4. stop all servers, then move to the next suite
#
# Default worker layout:
#   gpu0 -> levels 0,1
#   gpu1 -> levels 2,3
#
# This keeps total concurrency at 4 level-workers per suite, which matches
# the "2 GPUs x 2 UniVLA instances" workflow.
#
# Usage:
#   bash dynisafety_libero/scripts/univla/run_dynisafety_univla_fullsweep_2gpu_4level_parallel.sh [OPTIONS]
#
# Optional:
#   --suites           <CSV>    Comma-separated suite filter:
#                                spatial,object,goal,long or full names
#                                Default: all 4 suites
#   --gpu0-id          <ID>     CUDA_VISIBLE_DEVICES for worker group 0. Default: 0
#   --gpu1-id          <ID>     CUDA_VISIBLE_DEVICES for worker group 1. Default: 1
#   --levels-gpu0      <CSV>    Levels assigned to gpu0. Default: 0,1
#   --levels-gpu1      <CSV>    Levels assigned to gpu1. Default: 2,3
#   --server-host      <HOST>   UniVLA server host. Default: localhost
#   --port-level0      <PORT>   Port for level 0 worker. Default: 8020
#   --port-level1      <PORT>   Port for level 1 worker. Default: 8021
#   --port-level2      <PORT>   Port for level 2 worker. Default: 8022
#   --port-level3      <PORT>   Port for level 3 worker. Default: 8023
#   --weights-base     <DIR>    Base dir for per-suite UniVLA checkpoints
#                                Default: download_checkpoints/univla/sft-libero
#   --ckpt-spatial     <DIR>    Override spatial checkpoint dir
#   --ckpt-object      <DIR>    Override object checkpoint dir
#   --ckpt-goal        <DIR>    Override goal checkpoint dir
#   --ckpt-long        <DIR>    Override long checkpoint dir
#   --decoder-spatial  <PATH>   Override spatial action_decoder.pt
#   --decoder-object   <PATH>   Override object action_decoder.pt
#   --decoder-goal     <PATH>   Override goal action_decoder.pt
#   --decoder-long     <PATH>   Override long action_decoder.pt
#   --unnorm-spatial   <KEY>    dataset_statistics.json key for spatial
#   --unnorm-object    <KEY>    dataset_statistics.json key for object
#   --unnorm-goal      <KEY>    dataset_statistics.json key for goal
#   --unnorm-long      <KEY>    dataset_statistics.json key for long
#   --window-size      <N>      UniVLA ActionDecoder window. Default: 12
#   --center-crop               Pass --center-crop when starting servers
#   --num-trials       <N>      Episodes per task. Default: 50
#   --arm-speed        <SPEED>  slow | medium | fast. Default: medium
#   --task-ids         <CSV>    Comma-separated task ids to evaluate. Default: 0,1,2,3
#   --save-video       <BOOL>   True | False. Default: True
#   --server-env-name  <NAME>   Conda env for UniVLA server. Default: univla
#   --client-env-name  <NAME>   Conda env for benchmark client. Default: openvla-oft
#   --startup-timeout  <SEC>    Wait timeout per server. Default: 600
#   --dry-run                   Print commands without executing them
# ============================================================

set -euo pipefail

if (( BASH_VERSINFO[0] < 4 )); then
    echo "ERROR: bash 4+ required (current: ${BASH_VERSION})" >&2
    exit 1
fi

normalize_suite_name() {
    case "$1" in
        spatial|safelibero_spatial) echo "safelibero_spatial" ;;
        object|safelibero_object)   echo "safelibero_object" ;;
        goal|safelibero_goal)       echo "safelibero_goal" ;;
        long|safelibero_long)       echo "safelibero_long" ;;
        *)
            echo "ERROR: unknown suite '$1'. Use: spatial, object, goal, long," >&2
            echo "       or full names safelibero_spatial/object/goal/long." >&2
            return 1
            ;;
    esac
}

suite_short_name() {
    case "$1" in
        safelibero_spatial) echo "spatial" ;;
        safelibero_object)  echo "object" ;;
        safelibero_goal)    echo "goal" ;;
        safelibero_long)    echo "long" ;;
        *) return 1 ;;
    esac
}

wait_for_univla_health() {
    local host="$1"
    local port="$2"
    local timeout_s="$3"
    local deadline=$(( SECONDS + timeout_s ))

    while (( SECONDS < deadline )); do
        if python - "$host" "$port" <<'PY'
import sys
import urllib.request

host = sys.argv[1]
port = sys.argv[2]
url = f"http://{host}:{port}/health"
try:
    with urllib.request.urlopen(url, timeout=3) as resp:
        sys.exit(0 if 200 <= resp.status < 300 else 1)
except Exception:
    sys.exit(1)
PY
        then
            return 0
        fi
        sleep 2
    done
    return 1
}

stop_port_processes() {
    local port="$1"
    local pids
    pids="$(lsof -ti tcp:"${port}" || true)"
    if [[ -z "${pids}" ]]; then
        return 0
    fi

    echo "Stopping existing process(es) on port ${port}: ${pids}"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    sleep 2

    pids="$(lsof -ti tcp:"${port}" || true)"
    if [[ -n "${pids}" ]]; then
        echo "Force-killing remaining process(es) on port ${port}: ${pids}"
        # shellcheck disable=SC2086
        kill -9 ${pids} 2>/dev/null || true
    fi
}

parse_level_csv() {
    local csv="$1"
    local gpu_id="$2"
    local -n out_levels_ref="$3"
    local token
    local -a raw_levels=()

    csv="${csv// /}"
    if [[ -z "${csv}" ]]; then
        echo "ERROR: level CSV for gpu=${gpu_id} is empty." >&2
        exit 1
    fi

    IFS=',' read -r -a raw_levels <<< "${csv}"
    for token in "${raw_levels[@]}"; do
        [[ -z "${token}" ]] && continue
        case "${token}" in
            0|1|2|3) ;;
            *)
                echo "ERROR: invalid level '${token}'. Expected 0,1,2,3." >&2
                exit 1
                ;;
        esac
        if [[ -n "${LEVEL_TO_GPU[${token}]:-}" ]]; then
            echo "ERROR: level ${token} assigned multiple times." >&2
            exit 1
        fi
        LEVEL_TO_GPU["${token}"]="${gpu_id}"
        out_levels_ref+=("${token}")
        ALL_LEVELS+=("${token}")
    done
}

parse_task_id_csv() {
    local csv="$1"
    local -n out_task_ids_ref="$2"
    local token
    local -a raw_task_ids=()
    local -A seen_task_ids=()

    csv="${csv// /}"
    if [[ -z "${csv}" ]]; then
        echo "ERROR: task id CSV is empty." >&2
        exit 1
    fi

    IFS=',' read -r -a raw_task_ids <<< "${csv}"
    for token in "${raw_task_ids[@]}"; do
        [[ -z "${token}" ]] && continue
        case "${token}" in
            0|1|2|3) ;;
            *)
                echo "ERROR: invalid task id '${token}'. Expected 0,1,2,3." >&2
                exit 1
                ;;
        esac
        if [[ -n "${seen_task_ids[${token}]:-}" ]]; then
            continue
        fi
        seen_task_ids["${token}"]=1
        out_task_ids_ref+=("${token}")
    done
}

stop_active_evals() {
    local level
    local pid

    for level in "${ACTIVE_EVAL_LEVELS[@]}"; do
        pid="${ACTIVE_EVAL_PID_BY_LEVEL[${level}]:-}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done

    sleep 2

    for level in "${ACTIVE_EVAL_LEVELS[@]}"; do
        pid="${ACTIVE_EVAL_PID_BY_LEVEL[${level}]:-}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done

    ACTIVE_EVAL_LEVELS=()
    ACTIVE_EVAL_PID_BY_LEVEL=()
}

stop_active_servers() {
    local level
    local pid

    for level in "${ACTIVE_SERVER_LEVELS[@]}"; do
        pid="${ACTIVE_SERVER_PID_BY_LEVEL[${level}]:-}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done

    sleep 2

    for level in "${ACTIVE_SERVER_LEVELS[@]}"; do
        pid="${ACTIVE_SERVER_PID_BY_LEVEL[${level}]:-}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done

    ACTIVE_SERVER_LEVELS=()
    ACTIVE_SERVER_PID_BY_LEVEL=()
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    stop_active_evals
    stop_active_servers
    exit "${exit_code}"
}

# ---- defaults -------------------------------------------------------
GPU0_ID="0"
GPU1_ID="1"
LEVELS_GPU0_CSV="0,1"
LEVELS_GPU1_CSV="2,3"
SERVER_HOST="localhost"
PORT_LEVEL0="8020"
PORT_LEVEL1="8021"
PORT_LEVEL2="8022"
PORT_LEVEL3="8023"
WEIGHTS_BASE="download_checkpoints/univla/sft-libero"
CKPT_SPATIAL=""
CKPT_OBJECT=""
CKPT_GOAL=""
CKPT_LONG=""
DECODER_SPATIAL=""
DECODER_OBJECT=""
DECODER_GOAL=""
DECODER_LONG=""
UNNORM_SPATIAL="libero_spatial_no_noops"
UNNORM_OBJECT="libero_object_no_noops"
UNNORM_GOAL="libero_goal_no_noops"
UNNORM_LONG="libero_10_no_noops"
WINDOW_SIZE="12"
CENTER_CROP="0"
NUM_TRIALS="50"
ARM_SPEED="medium"
SAVE_VIDEO="True"
SERVER_ENV_NAME="univla"
CLIENT_ENV_NAME="openvla-oft"
STARTUP_TIMEOUT="600"
DRY_RUN="false"
SUITES_CSV=""
TASK_IDS_CSV="0,1,2,3"

DEFAULT_SUITES=(
    safelibero_spatial
    safelibero_object
    safelibero_goal
    safelibero_long
)
ALL_SUITES=("${DEFAULT_SUITES[@]}")
ALL_TASK_IDS=()

ACTIVE_SERVER_LEVELS=()
ACTIVE_EVAL_LEVELS=()
declare -A ACTIVE_SERVER_PID_BY_LEVEL=()
declare -A ACTIVE_EVAL_PID_BY_LEVEL=()

# ---- arg parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suites)           SUITES_CSV="$2";         shift 2 ;;
        --gpu0-id)          GPU0_ID="$2";            shift 2 ;;
        --gpu1-id)          GPU1_ID="$2";            shift 2 ;;
        --levels-gpu0)      LEVELS_GPU0_CSV="$2";    shift 2 ;;
        --levels-gpu1)      LEVELS_GPU1_CSV="$2";    shift 2 ;;
        --server-host)      SERVER_HOST="$2";        shift 2 ;;
        --port-level0)      PORT_LEVEL0="$2";        shift 2 ;;
        --port-level1)      PORT_LEVEL1="$2";        shift 2 ;;
        --port-level2)      PORT_LEVEL2="$2";        shift 2 ;;
        --port-level3)      PORT_LEVEL3="$2";        shift 2 ;;
        --weights-base)     WEIGHTS_BASE="$2";       shift 2 ;;
        --ckpt-spatial)     CKPT_SPATIAL="$2";       shift 2 ;;
        --ckpt-object)      CKPT_OBJECT="$2";        shift 2 ;;
        --ckpt-goal)        CKPT_GOAL="$2";          shift 2 ;;
        --ckpt-long)        CKPT_LONG="$2";          shift 2 ;;
        --decoder-spatial)  DECODER_SPATIAL="$2";    shift 2 ;;
        --decoder-object)   DECODER_OBJECT="$2";     shift 2 ;;
        --decoder-goal)     DECODER_GOAL="$2";       shift 2 ;;
        --decoder-long)     DECODER_LONG="$2";       shift 2 ;;
        --unnorm-spatial)   UNNORM_SPATIAL="$2";     shift 2 ;;
        --unnorm-object)    UNNORM_OBJECT="$2";      shift 2 ;;
        --unnorm-goal)      UNNORM_GOAL="$2";        shift 2 ;;
        --unnorm-long)      UNNORM_LONG="$2";        shift 2 ;;
        --window-size)      WINDOW_SIZE="$2";        shift 2 ;;
        --center-crop)      CENTER_CROP="1";         shift 1 ;;
        --num-trials)       NUM_TRIALS="$2";         shift 2 ;;
        --arm-speed)        ARM_SPEED="$2";          shift 2 ;;
        --task-ids)         TASK_IDS_CSV="$2";       shift 2 ;;
        --save-video)       SAVE_VIDEO="$2";         shift 2 ;;
        --server-env-name)  SERVER_ENV_NAME="$2";    shift 2 ;;
        --client-env-name)  CLIENT_ENV_NAME="$2";    shift 2 ;;
        --startup-timeout)  STARTUP_TIMEOUT="$2";    shift 2 ;;
        --dry-run)          DRY_RUN="true";          shift 1 ;;
        -h|--help)
            sed -n '3,58p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

if [[ -n "${SUITES_CSV}" ]]; then
    SUITES_CSV="${SUITES_CSV// /}"
    IFS=',' read -r -a SUITE_TOKENS <<< "${SUITES_CSV}"
    ALL_SUITES=()
    declare -A SEEN_SUITES=()
    for TOKEN in "${SUITE_TOKENS[@]}"; do
        [[ -z "${TOKEN}" ]] && continue
        SUITE="$(normalize_suite_name "${TOKEN}")" || exit 1
        if [[ -z "${SEEN_SUITES[${SUITE}]:-}" ]]; then
            ALL_SUITES+=("${SUITE}")
            SEEN_SUITES["${SUITE}"]=1
        fi
    done
    if [[ ${#ALL_SUITES[@]} -eq 0 ]]; then
        echo "ERROR: --suites resolved to an empty suite set." >&2
        exit 1
    fi
fi

parse_task_id_csv "${TASK_IDS_CSV}" ALL_TASK_IDS

if [[ ${#ALL_TASK_IDS[@]} -eq 0 ]]; then
    echo "ERROR: no task ids selected." >&2
    exit 1
fi

# ---- level / gpu layout ---------------------------------------------
declare -A LEVEL_TO_GPU=()
LEVELS_GPU0=()
LEVELS_GPU1=()
ALL_LEVELS=()

parse_level_csv "${LEVELS_GPU0_CSV}" "${GPU0_ID}" LEVELS_GPU0
parse_level_csv "${LEVELS_GPU1_CSV}" "${GPU1_ID}" LEVELS_GPU1

if [[ ${#ALL_LEVELS[@]} -eq 0 ]]; then
    echo "ERROR: no levels selected." >&2
    exit 1
fi

declare -A LEVEL_PORT=(
    [0]="${PORT_LEVEL0}"
    [1]="${PORT_LEVEL1}"
    [2]="${PORT_LEVEL2}"
    [3]="${PORT_LEVEL3}"
)

declare -A USED_PORTS=()
for LEVEL in "${ALL_LEVELS[@]}"; do
    PORT="${LEVEL_PORT[${LEVEL}]}"
    if [[ -z "${PORT}" ]]; then
        echo "ERROR: missing port for level ${LEVEL}." >&2
        exit 1
    fi
    if [[ -n "${USED_PORTS[${PORT}]:-}" ]]; then
        echo "ERROR: duplicate port ${PORT} for levels ${USED_PORTS[${PORT}]} and ${LEVEL}." >&2
        exit 1
    fi
    USED_PORTS["${PORT}"]="${LEVEL}"
done

# ---- paths ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
START_SERVER_SH="${SCRIPT_DIR}/start_univla_server.sh"
EVAL_PY="${SCRIPT_DIR}/run_dynisafety_univla_eval.py"

if [[ "${WEIGHTS_BASE}" != /* ]]; then
    WEIGHTS_BASE="${PROJ_ROOT}/${WEIGHTS_BASE}"
fi

if [[ -z "${CKPT_SPATIAL}" ]]; then
    CKPT_SPATIAL="${WEIGHTS_BASE}/univla-libero-spatial"
fi
if [[ -z "${CKPT_OBJECT}" ]]; then
    CKPT_OBJECT="${WEIGHTS_BASE}/univla-libero-object"
fi
if [[ -z "${CKPT_GOAL}" ]]; then
    CKPT_GOAL="${WEIGHTS_BASE}/univla-libero-goal"
fi
if [[ -z "${CKPT_LONG}" ]]; then
    CKPT_LONG="${WEIGHTS_BASE}/univla-libero-10"
fi

if [[ -z "${DECODER_SPATIAL}" ]]; then
    DECODER_SPATIAL="${CKPT_SPATIAL}/action_decoder.pt"
fi
if [[ -z "${DECODER_OBJECT}" ]]; then
    DECODER_OBJECT="${CKPT_OBJECT}/action_decoder.pt"
fi
if [[ -z "${DECODER_GOAL}" ]]; then
    DECODER_GOAL="${CKPT_GOAL}/action_decoder.pt"
fi
if [[ -z "${DECODER_LONG}" ]]; then
    DECODER_LONG="${CKPT_LONG}/action_decoder.pt"
fi

declare -A SUITE_CHECKPOINT=(
    [safelibero_spatial]="${CKPT_SPATIAL}"
    [safelibero_object]="${CKPT_OBJECT}"
    [safelibero_goal]="${CKPT_GOAL}"
    [safelibero_long]="${CKPT_LONG}"
)

declare -A SUITE_ACTION_DECODER=(
    [safelibero_spatial]="${DECODER_SPATIAL}"
    [safelibero_object]="${DECODER_OBJECT}"
    [safelibero_goal]="${DECODER_GOAL}"
    [safelibero_long]="${DECODER_LONG}"
)

declare -A SUITE_UNNORM_KEY=(
    [safelibero_spatial]="${UNNORM_SPATIAL}"
    [safelibero_object]="${UNNORM_OBJECT}"
    [safelibero_goal]="${UNNORM_GOAL}"
    [safelibero_long]="${UNNORM_LONG}"
)

RUN_STAMP="$(date +%Y_%m_%d-%H_%M_%S)"
RUN_LOG_DIR="${PROJ_ROOT}/dynisafety_libero/outputs/univla/logs/fullsweep_2gpu_4level_parallel/${RUN_STAMP}"
RUN_VIDEO_DIR="${PROJ_ROOT}/dynisafety_libero/outputs/univla/videos/fullsweep_2gpu_4level_parallel/${RUN_STAMP}"
mkdir -p "${RUN_LOG_DIR}" "${RUN_VIDEO_DIR}"

trap cleanup EXIT INT TERM

if [[ "${DRY_RUN}" != "true" ]]; then
    set +u
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
        || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
    conda activate "${CLIENT_ENV_NAME}"
    set -u
fi

TOTAL_EVAL_RUNS=$(( ${#ALL_SUITES[@]} * ${#ALL_LEVELS[@]} * ${#ALL_TASK_IDS[@]} ))

echo "============================================================"
echo "UniVLA Full Sweep 2-GPU / 4-Level Parallel Orchestrator"
echo "  Suites           : ${ALL_SUITES[*]}"
echo "  Levels           : ${ALL_LEVELS[*]}"
echo "  Task IDs         : ${ALL_TASK_IDS[*]}"
echo "  gpu0             : id=${GPU0_ID} levels=${LEVELS_GPU0[*]}"
echo "  gpu1             : id=${GPU1_ID} levels=${LEVELS_GPU1[*]}"
echo "  Ports            :"
for LEVEL in "${ALL_LEVELS[@]}"; do
    echo "    level ${LEVEL} -> ${LEVEL_PORT[${LEVEL}]} (gpu ${LEVEL_TO_GPU[${LEVEL}]})"
done
echo "  Total eval runs  : ${TOTAL_EVAL_RUNS} (${#ALL_LEVELS[@]} levels x ${#ALL_TASK_IDS[@]} tasks per suite)"
echo "  Weights base     : ${WEIGHTS_BASE}"
echo "  window_size      : ${WINDOW_SIZE}"
echo "  center_crop      : ${CENTER_CROP}"
echo "  Trials/task      : ${NUM_TRIALS}"
echo "  Arm speed        : ${ARM_SPEED}"
echo "  Save video       : ${SAVE_VIDEO}"
echo "  Server env       : ${SERVER_ENV_NAME}"
echo "  Client env       : ${CLIENT_ENV_NAME}"
echo "  Startup timeout  : ${STARTUP_TIMEOUT}s"
echo "  Logs dir         : ${RUN_LOG_DIR}"
echo "  Videos dir       : ${RUN_VIDEO_DIR}"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  *** DRY RUN - commands will be printed, not executed ***"
fi
echo "============================================================"

FAILED=()

for SUITE in "${ALL_SUITES[@]}"; do
    SHORT_SUITE="$(suite_short_name "${SUITE}")"
    CKPT="${SUITE_CHECKPOINT[${SUITE}]}"
    DECODER="${SUITE_ACTION_DECODER[${SUITE}]}"
    UNNORM_KEY="${SUITE_UNNORM_KEY[${SUITE}]}"

    SUITE_LOG_DIR="${RUN_LOG_DIR}/${SHORT_SUITE}"
    SUITE_SERVER_LOG_DIR="${SUITE_LOG_DIR}/servers"
    SUITE_EVAL_LOG_DIR="${SUITE_LOG_DIR}/workers"
    SUITE_PY_LOG_DIR="${SUITE_LOG_DIR}/python_logs"
    SUITE_VIDEO_DIR="${RUN_VIDEO_DIR}/${SHORT_SUITE}"
    mkdir -p "${SUITE_SERVER_LOG_DIR}" "${SUITE_EVAL_LOG_DIR}" "${SUITE_PY_LOG_DIR}" "${SUITE_VIDEO_DIR}"

    ACTIVE_SERVER_LEVELS=()
    ACTIVE_EVAL_LEVELS=()
    ACTIVE_SERVER_PID_BY_LEVEL=()
    ACTIVE_EVAL_PID_BY_LEVEL=()

    echo ""
    echo "============================================================"
    echo "Suite: ${SUITE}"
    echo "  Checkpoint     : ${CKPT}"
    echo "  Action decoder : ${DECODER}"
    echo "  unnorm_key     : ${UNNORM_KEY}"
    echo "============================================================"

    for LEVEL in "${ALL_LEVELS[@]}"; do
        GPU_ID="${LEVEL_TO_GPU[${LEVEL}]}"
        PORT="${LEVEL_PORT[${LEVEL}]}"
        SERVER_LOG="${SUITE_SERVER_LOG_DIR}/level${LEVEL}_gpu${GPU_ID}.log"

        SERVER_CMD=(
            bash "${START_SERVER_SH}"
                --task-suite "${SUITE}"
                --pretrained-checkpoint "${CKPT}"
                --action-decoder-path "${DECODER}"
                --unnorm-key "${UNNORM_KEY}"
                --window-size "${WINDOW_SIZE}"
                --port "${PORT}"
                --env-name "${SERVER_ENV_NAME}"
        )
        if [[ "${CENTER_CROP}" == "1" ]]; then
            SERVER_CMD+=(--center-crop)
        fi

        echo "Server worker: suite=${SUITE} level=${LEVEL} gpu=${GPU_ID} port=${PORT}"
        echo "  log: ${SERVER_LOG}"
        echo "  cmd: CUDA_VISIBLE_DEVICES=${GPU_ID} UNIVLA_LIBERO_WEIGHTS_BASE=${WEIGHTS_BASE} ${SERVER_CMD[*]}"

        if [[ "${DRY_RUN}" == "true" ]]; then
            continue
        fi

        stop_port_processes "${PORT}"
        (
            export CUDA_VISIBLE_DEVICES="${GPU_ID}"
            export UNIVLA_LIBERO_WEIGHTS_BASE="${WEIGHTS_BASE}"
            "${SERVER_CMD[@]}"
        ) > "${SERVER_LOG}" 2>&1 &

        ACTIVE_SERVER_LEVELS+=("${LEVEL}")
        ACTIVE_SERVER_PID_BY_LEVEL["${LEVEL}"]=$!
    done

    if [[ "${DRY_RUN}" == "true" ]]; then
        for TASK_ID in "${ALL_TASK_IDS[@]}"; do
            echo "Task batch    : suite=${SUITE} task=${TASK_ID}"
            for LEVEL in "${ALL_LEVELS[@]}"; do
                GPU_ID="${LEVEL_TO_GPU[${LEVEL}]}"
                PORT="${LEVEL_PORT[${LEVEL}]}"
                EVAL_LOG="${SUITE_EVAL_LOG_DIR}/task${TASK_ID}_level${LEVEL}_gpu${GPU_ID}.log"
                RUN_NOTE="univla-${SHORT_SUITE}-task${TASK_ID}-level${LEVEL}-gpu${GPU_ID}"
                EVAL_CMD=(
                    python "${EVAL_PY}"
                        --server_host "${SERVER_HOST}"
                        --server_port "${PORT}"
                        --task_suite_name "${SUITE}"
                        --dynamic_level "${LEVEL}"
                        --task_id_override "${TASK_ID}"
                        --num_trials_per_task "${NUM_TRIALS}"
                        --arm_speed "${ARM_SPEED}"
                        --save_video "${SAVE_VIDEO}"
                        --run_id_note "${RUN_NOTE}"
                        --local_log_dir "${SUITE_PY_LOG_DIR}"
                        --video_dir "${SUITE_VIDEO_DIR}/level${LEVEL}"
                        --pretrained_checkpoint "${CKPT}"
                        --action_decoder_path "${DECODER}"
                )
                echo "Eval worker  : suite=${SUITE} task=${TASK_ID} level=${LEVEL} gpu=${GPU_ID} port=${PORT}"
                echo "  log: ${EVAL_LOG}"
                echo "  cmd: CUDA_VISIBLE_DEVICES=${GPU_ID} ${EVAL_CMD[*]}"
            done
        done
        continue
    fi

    SUITE_SERVER_FAILED="false"
    for LEVEL in "${ALL_LEVELS[@]}"; do
        PORT="${LEVEL_PORT[${LEVEL}]}"
        GPU_ID="${LEVEL_TO_GPU[${LEVEL}]}"
        SERVER_LOG="${SUITE_SERVER_LOG_DIR}/level${LEVEL}_gpu${GPU_ID}.log"

        if wait_for_univla_health "${SERVER_HOST}" "${PORT}" "${STARTUP_TIMEOUT}"; then
            echo "Server ready: suite=${SUITE} level=${LEVEL} gpu=${GPU_ID} host=${SERVER_HOST} port=${PORT}"
        else
            echo "ERROR: server failed to become ready for suite=${SUITE} level=${LEVEL} within ${STARTUP_TIMEOUT}s" >&2
            echo "Check server log: ${SERVER_LOG}" >&2
            FAILED+=("suite=${SUITE} level=${LEVEL} stage=server_start")
            SUITE_SERVER_FAILED="true"
        fi
    done

    if [[ "${SUITE_SERVER_FAILED}" == "true" ]]; then
        stop_active_servers
        continue
    fi

    for TASK_ID in "${ALL_TASK_IDS[@]}"; do
        ACTIVE_EVAL_LEVELS=()
        ACTIVE_EVAL_PID_BY_LEVEL=()

        echo ""
        echo "Task batch    : suite=${SUITE} task=${TASK_ID}"

        for LEVEL in "${ALL_LEVELS[@]}"; do
            GPU_ID="${LEVEL_TO_GPU[${LEVEL}]}"
            PORT="${LEVEL_PORT[${LEVEL}]}"
            EVAL_LOG="${SUITE_EVAL_LOG_DIR}/task${TASK_ID}_level${LEVEL}_gpu${GPU_ID}.log"
            RUN_NOTE="univla-${SHORT_SUITE}-task${TASK_ID}-level${LEVEL}-gpu${GPU_ID}"
            EVAL_CMD=(
                python "${EVAL_PY}"
                    --server_host "${SERVER_HOST}"
                    --server_port "${PORT}"
                    --task_suite_name "${SUITE}"
                    --dynamic_level "${LEVEL}"
                    --task_id_override "${TASK_ID}"
                    --num_trials_per_task "${NUM_TRIALS}"
                    --arm_speed "${ARM_SPEED}"
                    --save_video "${SAVE_VIDEO}"
                    --run_id_note "${RUN_NOTE}"
                    --local_log_dir "${SUITE_PY_LOG_DIR}"
                    --video_dir "${SUITE_VIDEO_DIR}/level${LEVEL}"
                    --pretrained_checkpoint "${CKPT}"
                    --action_decoder_path "${DECODER}"
            )

            echo "Eval worker  : suite=${SUITE} task=${TASK_ID} level=${LEVEL} gpu=${GPU_ID} port=${PORT}"
            echo "  log: ${EVAL_LOG}"
            echo "  cmd: CUDA_VISIBLE_DEVICES=${GPU_ID} ${EVAL_CMD[*]}"

            (
                export CUDA_VISIBLE_DEVICES="${GPU_ID}"
                cd "${PROJ_ROOT}"
                "${EVAL_CMD[@]}"
            ) > "${EVAL_LOG}" 2>&1 &

            ACTIVE_EVAL_LEVELS+=("${LEVEL}")
            ACTIVE_EVAL_PID_BY_LEVEL["${LEVEL}"]=$!
        done

        for LEVEL in "${ALL_LEVELS[@]}"; do
            PID="${ACTIVE_EVAL_PID_BY_LEVEL[${LEVEL}]}"
            if wait "${PID}"; then
                echo "Eval worker finished: suite=${SUITE} task=${TASK_ID} level=${LEVEL}"
            else
                EXIT_CODE=$?
                GPU_ID="${LEVEL_TO_GPU[${LEVEL}]}"
                EVAL_LOG="${SUITE_EVAL_LOG_DIR}/task${TASK_ID}_level${LEVEL}_gpu${GPU_ID}.log"
                echo "Eval worker FAILED: suite=${SUITE} task=${TASK_ID} level=${LEVEL} (exit ${EXIT_CODE})" >&2
                echo "Check eval log: ${EVAL_LOG}" >&2
                FAILED+=("suite=${SUITE} task=${TASK_ID} level=${LEVEL} stage=eval exit=${EXIT_CODE}")
            fi
        done

        stop_active_evals
    done

    stop_active_servers
done

echo ""
echo "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All selected UniVLA suite sweeps succeeded."
else
    echo "Failed suite/task/level workers (${#FAILED[@]}):"
    for ITEM in "${FAILED[@]}"; do
        echo "  - ${ITEM}"
    done
fi
echo "Logs dir: ${RUN_LOG_DIR}"
echo "Videos dir: ${RUN_VIDEO_DIR}"
echo "============================================================"

if [[ ${#FAILED[@]} -ne 0 ]]; then
    exit 1
fi
