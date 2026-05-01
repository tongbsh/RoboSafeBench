#!/usr/bin/env bash
# ============================================================
# Auto orchestration: X-VLA server + parallel suite full sweeps
#
# 1. 重启 / 拉起单个 X-VLA server
# 2. 等待 server ready（探测 /act）
# 3. 并行执行每个 suite 的 full sweep 子任务
#
# 默认会并行跑 4 个 suite，使用同一个 X-VLA LIBERO full server。
#
# Usage:
#   bash dynisafety_libero/scripts/x-vla/run_dynisafety_xvla_fullsweep_parallel.sh [OPTIONS]
#
# Optional:
#   --suites           <CSV>    Comma-separated suite filter:
#                                spatial,object,goal,long or full names
#                                Default: all 4 suites
#   --server-host      <HOST>   X-VLA server host. Default: localhost
#   --server-port      <PORT>   X-VLA server port. Default: 8010
#   --model-path       <PATH>   Model path / HF id for X-VLA server
#                                Default: download_checkpoints/xVLA/xvla-libero-full
#   --lora-path        <PATH>   Optional LoRA path for X-VLA server
#   --domain-id        <ID>     XVLA_DOMAIN_ID. Default: 3
#   --diff-steps       <N>      XVLA_STEPS. Default: 10
#   --num-trials       <N>      Episodes per task. Default: 50
#   --arm-speed        <SPEED>  slow | medium | fast. Default: medium
#   --server-env-name  <NAME>   Conda env for X-VLA server. Default: XVLA
#   --client-env-name  <NAME>   Conda env for benchmark client. Default: openvla-oft
#   --startup-timeout  <SEC>    Wait timeout for server. Default: 600
#   --keep-server               Keep started server alive after eval
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

wait_for_xvla_ready() {
    local host="$1"
    local port="$2"
    local timeout_s="$3"
    local deadline=$(( SECONDS + timeout_s ))

    while (( SECONDS < deadline )); do
        if python - "$host" "$port" <<'PY'
import sys
import urllib.error
import urllib.request

host = sys.argv[1]
port = sys.argv[2]
url = f"http://{host}:{port}/act"
req = urllib.request.Request(
    url,
    data=b"{}",
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=5):
        sys.exit(0)
except urllib.error.HTTPError:
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
        then
            return 0
        fi
        sleep 5
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

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    for suite in "${RUNNING_EVAL_SUITES[@]}"; do
        pid="${EVAL_PID_BY_SUITE[${suite}]:-}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done

    if [[ "${KEEP_SERVER}" != "true" ]]; then
        if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill "${SERVER_PID}" 2>/dev/null || true
            sleep 2
            if kill -0 "${SERVER_PID}" 2>/dev/null; then
                kill -9 "${SERVER_PID}" 2>/dev/null || true
            fi
        fi
    fi

    exit "${exit_code}"
}

# ---- defaults -------------------------------------------------------
SERVER_HOST="localhost"
SERVER_PORT="8010"
MODEL_PATH="download_checkpoints/xVLA/xvla-libero-full"
LORA_PATH=""
DOMAIN_ID="3"
DIFF_STEPS="10"
NUM_TRIALS="50"
ARM_SPEED="medium"
SERVER_ENV_NAME="XVLA"
CLIENT_ENV_NAME="openvla-oft"
STARTUP_TIMEOUT="600"
KEEP_SERVER="false"
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
        --server-port)      SERVER_PORT="$2";        shift 2 ;;
        --model-path)       MODEL_PATH="$2";         shift 2 ;;
        --lora-path)        LORA_PATH="$2";          shift 2 ;;
        --domain-id)        DOMAIN_ID="$2";          shift 2 ;;
        --diff-steps)       DIFF_STEPS="$2";         shift 2 ;;
        --num-trials)       NUM_TRIALS="$2";         shift 2 ;;
        --arm-speed)        ARM_SPEED="$2";          shift 2 ;;
        --server-env-name)  SERVER_ENV_NAME="$2";    shift 2 ;;
        --client-env-name)  CLIENT_ENV_NAME="$2";    shift 2 ;;
        --startup-timeout)  STARTUP_TIMEOUT="$2";    shift 2 ;;
        --keep-server)      KEEP_SERVER="true";      shift 1 ;;
        --dry-run)          DRY_RUN="true";          shift 1 ;;
        -h|--help)
            sed -n '3,33p' "$0"
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
START_SERVER_SH="${SCRIPT_DIR}/start_xvla_server.sh"
FULLSWEEP_SH="${SCRIPT_DIR}/run_dynisafety_xvla_fullsweep.sh"

RUN_STAMP="$(date +%Y_%m_%d-%H_%M_%S)"
RUN_LOG_DIR="${PROJ_ROOT}/dynisafety_libero/outputs/xvla/logs/fullsweep_parallel/${RUN_STAMP}"
mkdir -p "${RUN_LOG_DIR}"

SERVER_PID=""
RUNNING_EVAL_SUITES=()
declare -A EVAL_PID_BY_SUITE=()
declare -A EVAL_LOG_BY_SUITE=()

trap cleanup EXIT INT TERM

echo "============================================================"
echo "X-VLA Full Sweep Parallel"
echo "  Suites           : ${ALL_SUITES[*]}"
echo "  Server           : ${SERVER_HOST}:${SERVER_PORT}"
echo "  Model path       : ${MODEL_PATH}"
echo "  LoRA path        : ${LORA_PATH:-none}"
echo "  XVLA_DOMAIN_ID   : ${DOMAIN_ID}"
echo "  XVLA_STEPS       : ${DIFF_STEPS}"
echo "  Trials/task      : ${NUM_TRIALS}"
echo "  Arm speed        : ${ARM_SPEED}"
echo "  Server env       : ${SERVER_ENV_NAME}"
echo "  Client env       : ${CLIENT_ENV_NAME}"
echo "  Startup timeout  : ${STARTUP_TIMEOUT}s"
echo "  Logs dir         : ${RUN_LOG_DIR}"
if [[ "${KEEP_SERVER}" == "true" ]]; then
    echo "  Keep server      : true"
fi
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  *** DRY RUN - commands will be printed, not executed ***"
fi
echo "============================================================"

SERVER_LOG="${RUN_LOG_DIR}/server.log"
SERVER_CMD=(
    bash "${START_SERVER_SH}"
        --model-path "${MODEL_PATH}"
        --port "${SERVER_PORT}"
        --env-name "${SERVER_ENV_NAME}"
)
if [[ -n "${LORA_PATH}" ]]; then
    SERVER_CMD+=(--lora-path "${LORA_PATH}")
fi

echo ""
echo "============================================================"
echo "Restart X-VLA server on port ${SERVER_PORT}"
echo "Server log: ${SERVER_LOG}"
echo "CMD: ${SERVER_CMD[*]}"
echo "============================================================"

if [[ "${DRY_RUN}" != "true" ]]; then
    stop_port_processes "${SERVER_PORT}"
    "${SERVER_CMD[@]}" > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    if wait_for_xvla_ready "${SERVER_HOST}" "${SERVER_PORT}" "${STARTUP_TIMEOUT}"; then
        echo "Server ready: host=${SERVER_HOST} port=${SERVER_PORT}"
    else
        echo "ERROR: X-VLA server failed to become ready within ${STARTUP_TIMEOUT}s" >&2
        echo "Check server log: ${SERVER_LOG}" >&2
        exit 1
    fi
fi

for SUITE in "${ALL_SUITES[@]}"; do
    SHORT_SUITE="$(suite_short_name "${SUITE}")"
    EVAL_LOG="${RUN_LOG_DIR}/eval_${SHORT_SUITE}.log"
    EVAL_LOG_BY_SUITE["${SUITE}"]="${EVAL_LOG}"

    EVAL_CMD=(
        bash "${FULLSWEEP_SH}"
            --suites "${SUITE}"
            --server-host "${SERVER_HOST}"
            --server-port "${SERVER_PORT}"
            --domain-id "${DOMAIN_ID}"
            --diff-steps "${DIFF_STEPS}"
            --num-trials "${NUM_TRIALS}"
            --arm-speed "${ARM_SPEED}"
            --env-name "${CLIENT_ENV_NAME}"
    )

    echo ""
    echo "============================================================"
    echo "Launch full sweep for ${SUITE}"
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
    echo "All selected X-VLA suite sweeps succeeded."
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
