#!/usr/bin/env bash
# ============================================================
# Auto orchestration: UniVLA suite servers + parallel full sweeps
#
# 1. 按 suite 依次重启 / 拉起 UniVLA server
# 2. 等待每个 server 的 /health 就绪
# 3. 并行执行 run_dynisafety_univla_fullsweep.sh（每个 suite 一个子进程）
#
# 默认会并行跑 4 个 suite，每个 suite 各自连接自己的 UniVLA server / 权重。
#
# Usage:
#   bash dynisafety_libero/scripts/univla/run_dynisafety_univla_fullsweep_autoservers.sh [OPTIONS]
#
# Optional:
#   --suites           <CSV>    Comma-separated suite filter:
#                                spatial,object,goal,long or full names
#                                Default: all 4 suites
#   --server-host      <HOST>   UniVLA server host. Default: localhost
#   --server-port      <PORT>   Use one shared port for all selected suites
#                                Only safe when a single suite is selected
#   --port-spatial     <PORT>   Port for safelibero_spatial. Default: 8020
#   --port-object      <PORT>   Port for safelibero_object.  Default: 8021
#   --port-goal        <PORT>   Port for safelibero_goal.    Default: 8022
#   --port-long        <PORT>   Port for safelibero_long.    Default: 8023
#   --weights-base     <DIR>    Base dir for per-suite UniVLA checkpoints
#                                Default: download_checkpoints/univla/sft-libero
#   --ckpt-spatial     <DIR>    Checkpoint dir for spatial server/log hint
#   --ckpt-object      <DIR>    Checkpoint dir for object server/log hint
#   --ckpt-goal        <DIR>    Checkpoint dir for goal server/log hint
#   --ckpt-long        <DIR>    Checkpoint dir for long server/log hint
#   --decoder-spatial  <PATH>   action_decoder.pt for spatial
#   --decoder-object   <PATH>   action_decoder.pt for object
#   --decoder-goal     <PATH>   action_decoder.pt for goal
#   --decoder-long     <PATH>   action_decoder.pt for long
#   --unnorm-spatial   <KEY>    dataset_statistics.json key for spatial
#   --unnorm-object    <KEY>    dataset_statistics.json key for object
#   --unnorm-goal      <KEY>    dataset_statistics.json key for goal
#   --unnorm-long      <KEY>    dataset_statistics.json key for long
#   --window-size      <N>      UniVLA ActionDecoder window. Default: 12
#   --center-crop               Pass --center-crop when starting servers
#   --num-trials       <N>      Episodes per task. Default: 50
#   --arm-speed        <SPEED>  slow | medium | fast. Default: medium
#   --server-env-name  <NAME>   Conda env for UniVLA server. Default: univla
#   --client-env-name  <NAME>   Conda env for benchmark client. Default: openvla-oft
#   --startup-timeout  <SEC>    Wait timeout per server. Default: 600
#   --keep-servers              Keep started servers alive after eval
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

append_suite_eval_args() {
    local suite="$1"
    local -n out_ref="$2"
    case "${suite}" in
        safelibero_spatial)
            out_ref+=(--ckpt-spatial "${CKPT_SPATIAL}" --decoder-spatial "${DECODER_SPATIAL}")
            ;;
        safelibero_object)
            out_ref+=(--ckpt-object "${CKPT_OBJECT}" --decoder-object "${DECODER_OBJECT}")
            ;;
        safelibero_goal)
            out_ref+=(--ckpt-goal "${CKPT_GOAL}" --decoder-goal "${DECODER_GOAL}")
            ;;
        safelibero_long)
            out_ref+=(--ckpt-long "${CKPT_LONG}" --decoder-long "${DECODER_LONG}")
            ;;
    esac
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    for suite in "${RUNNING_EVAL_SUITES[@]}"; do
        pid="${EVAL_PID_BY_SUITE[${suite}]:-}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done

    if [[ "${KEEP_SERVERS}" != "true" ]]; then
        for suite in "${STARTED_SUITES[@]}"; do
            pid="${SERVER_PID_BY_SUITE[${suite}]:-}"
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                kill "${pid}" 2>/dev/null || true
            fi
        done
        sleep 2
        for suite in "${STARTED_SUITES[@]}"; do
            pid="${SERVER_PID_BY_SUITE[${suite}]:-}"
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                kill -9 "${pid}" 2>/dev/null || true
            fi
        done
    fi

    exit "${exit_code}"
}

# ---- defaults -------------------------------------------------------
SERVER_HOST="localhost"
PORT_SPATIAL="8020"
PORT_OBJECT="8021"
PORT_GOAL="8022"
PORT_LONG="8023"
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
SERVER_ENV_NAME="univla"
CLIENT_ENV_NAME="openvla-oft"
STARTUP_TIMEOUT="600"
KEEP_SERVERS="false"
DRY_RUN="false"
SUITES_CSV=""

DEFAULT_SUITES=(
    safelibero_spatial
    safelibero_object
    safelibero_goal
    safelibero_long
)
ALL_SUITES=("${DEFAULT_SUITES[@]}")

# ---- arg parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suites)           SUITES_CSV="$2";         shift 2 ;;
        --server-host)      SERVER_HOST="$2";        shift 2 ;;
        --server-port)
            PORT_SPATIAL="$2"
            PORT_OBJECT="$2"
            PORT_GOAL="$2"
            PORT_LONG="$2"
            shift 2
            ;;
        --port-spatial)     PORT_SPATIAL="$2";       shift 2 ;;
        --port-object)      PORT_OBJECT="$2";        shift 2 ;;
        --port-goal)        PORT_GOAL="$2";          shift 2 ;;
        --port-long)        PORT_LONG="$2";          shift 2 ;;
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
        --server-env-name)  SERVER_ENV_NAME="$2";    shift 2 ;;
        --client-env-name)  CLIENT_ENV_NAME="$2";    shift 2 ;;
        --startup-timeout)  STARTUP_TIMEOUT="$2";    shift 2 ;;
        --keep-servers)     KEEP_SERVERS="true";     shift 1 ;;
        --dry-run)          DRY_RUN="true";          shift 1 ;;
        -h|--help)
            sed -n '3,53p' "$0"
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

# ---- paths ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
START_SERVER_SH="${SCRIPT_DIR}/start_univla_server.sh"
FULLSWEEP_SH="${SCRIPT_DIR}/run_dynisafety_univla_fullsweep.sh"

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

declare -A SUITE_PORT=(
    [safelibero_spatial]="${PORT_SPATIAL}"
    [safelibero_object]="${PORT_OBJECT}"
    [safelibero_goal]="${PORT_GOAL}"
    [safelibero_long]="${PORT_LONG}"
)

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

declare -A USED_PORTS=()
for SUITE in "${ALL_SUITES[@]}"; do
    PORT="${SUITE_PORT[${SUITE}]}"
    if [[ -n "${USED_PORTS[${PORT}]:-}" ]]; then
        echo "ERROR: duplicate port ${PORT} for selected suites:" >&2
        echo "  ${USED_PORTS[${PORT}]} and ${SUITE}" >&2
        echo "Use distinct --port-* values when running multiple suites in parallel." >&2
        exit 1
    fi
    USED_PORTS["${PORT}"]="${SUITE}"
done

RUN_STAMP="$(date +%Y_%m_%d-%H_%M_%S)"
RUN_LOG_DIR="${PROJ_ROOT}/dynisafety_libero/outputs/univla/logs/fullsweep_autoservers/${RUN_STAMP}"
mkdir -p "${RUN_LOG_DIR}"

STARTED_SUITES=()
RUNNING_EVAL_SUITES=()
declare -A SERVER_PID_BY_SUITE=()
declare -A EVAL_PID_BY_SUITE=()

trap cleanup EXIT INT TERM

echo "============================================================"
echo "UniVLA Full Sweep Auto-Servers"
echo "  Suites           : ${ALL_SUITES[*]}"
echo "  Server host      : ${SERVER_HOST}"
echo "  Ports            :"
echo "    spatial -> ${SUITE_PORT[safelibero_spatial]}"
echo "    object  -> ${SUITE_PORT[safelibero_object]}"
echo "    goal    -> ${SUITE_PORT[safelibero_goal]}"
echo "    long    -> ${SUITE_PORT[safelibero_long]}"
echo "  Weights base     : ${WEIGHTS_BASE}"
echo "  window_size      : ${WINDOW_SIZE}"
echo "  center_crop      : ${CENTER_CROP}"
echo "  Trials/task      : ${NUM_TRIALS}"
echo "  Arm speed        : ${ARM_SPEED}"
echo "  Server env       : ${SERVER_ENV_NAME}"
echo "  Client env       : ${CLIENT_ENV_NAME}"
echo "  Startup timeout  : ${STARTUP_TIMEOUT}s"
echo "  Logs dir         : ${RUN_LOG_DIR}"
if [[ "${KEEP_SERVERS}" == "true" ]]; then
    echo "  Keep servers     : true"
fi
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  *** DRY RUN - commands will be printed, not executed ***"
fi
echo "============================================================"

for SUITE in "${ALL_SUITES[@]}"; do
    PORT="${SUITE_PORT[${SUITE}]}"
    CKPT="${SUITE_CHECKPOINT[${SUITE}]}"
    DECODER="${SUITE_ACTION_DECODER[${SUITE}]}"
    UNNORM_KEY="${SUITE_UNNORM_KEY[${SUITE}]}"
    SHORT_SUITE="$(suite_short_name "${SUITE}")"
    SERVER_LOG="${RUN_LOG_DIR}/server_${SHORT_SUITE}.log"

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

    echo ""
    echo "============================================================"
    echo "Restart UniVLA server for ${SUITE} on port ${PORT}"
    echo "Server log: ${SERVER_LOG}"
    echo "CMD: UNIVLA_LIBERO_WEIGHTS_BASE=${WEIGHTS_BASE} ${SERVER_CMD[*]}"
    echo "============================================================"

    if [[ "${DRY_RUN}" == "true" ]]; then
        continue
    fi

    stop_port_processes "${PORT}"
    (
        export UNIVLA_LIBERO_WEIGHTS_BASE="${WEIGHTS_BASE}"
        "${SERVER_CMD[@]}"
    ) > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    STARTED_SUITES+=("${SUITE}")
    SERVER_PID_BY_SUITE["${SUITE}"]="${SERVER_PID}"

    if wait_for_univla_health "${SERVER_HOST}" "${PORT}" "${STARTUP_TIMEOUT}"; then
        echo "Server ready: suite=${SUITE} host=${SERVER_HOST} port=${PORT}"
    else
        echo "ERROR: server failed to become ready for suite=${SUITE} within ${STARTUP_TIMEOUT}s" >&2
        echo "Check server log: ${SERVER_LOG}" >&2
        exit 1
    fi
done

declare -A EVAL_LOG_BY_SUITE=()
for SUITE in "${ALL_SUITES[@]}"; do
    PORT="${SUITE_PORT[${SUITE}]}"
    SHORT_SUITE="$(suite_short_name "${SUITE}")"
    EVAL_LOG="${RUN_LOG_DIR}/eval_${SHORT_SUITE}.log"
    EVAL_LOG_BY_SUITE["${SUITE}"]="${EVAL_LOG}"

    EVAL_CMD=(
        bash "${FULLSWEEP_SH}"
            --suites "${SUITE}"
            --server-host "${SERVER_HOST}"
            --server-port "${PORT}"
            --weights-base "${WEIGHTS_BASE}"
            --num-trials "${NUM_TRIALS}"
            --arm-speed "${ARM_SPEED}"
            --env-name "${CLIENT_ENV_NAME}"
    )
    append_suite_eval_args "${SUITE}" EVAL_CMD

    echo ""
    echo "============================================================"
    echo "Launch full sweep for ${SUITE} on port ${PORT}"
    echo "Eval log: ${EVAL_LOG}"
    echo "CMD: ${EVAL_CMD[*]}"
    echo "============================================================"

    if [[ "${DRY_RUN}" == "true" ]]; then
        continue
    fi

    "${EVAL_CMD[@]}" > "${EVAL_LOG}" 2>&1 &
    EVAL_PID=$!
    RUNNING_EVAL_SUITES+=("${SUITE}")
    EVAL_PID_BY_SUITE["${SUITE}"]="${EVAL_PID}"
done

if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi

FAILED=()
for SUITE in "${ALL_SUITES[@]}"; do
    PID="${EVAL_PID_BY_SUITE[${SUITE}]}"
    if wait "${PID}"; then
        echo "Full sweep finished: ${SUITE}"
    else
        EXIT_CODE=$?
        echo "Full sweep FAILED: ${SUITE} (exit ${EXIT_CODE})" >&2
        echo "Check eval log: ${EVAL_LOG_BY_SUITE[${SUITE}]}" >&2
        FAILED+=("${SUITE}")
    fi
done

echo ""
echo "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All selected UniVLA suite sweeps succeeded."
else
    echo "Failed suite sweeps (${#FAILED[@]}):"
    for SUITE in "${FAILED[@]}"; do
        echo "  - ${SUITE}"
    done
fi
echo "Logs dir: ${RUN_LOG_DIR}"
echo "============================================================"

if [[ ${#FAILED[@]} -ne 0 ]]; then
    exit 1
fi
