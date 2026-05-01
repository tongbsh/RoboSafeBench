#!/usr/bin/env bash
# ============================================================
# Serial benchmark orchestrator:
# DyniSafety-SimplerEnv × WidowX Bridge × {x-vla, starvla, internvla_m1}
#
# What this script does by default:
#   1. Start the model server when required.
#   2. Run the existing serial Bridge fullsweep for the model:
#        3 subtasks (carrot, stack, spoon) × levels (0,1,2,3)
#   3. Stop the server.
#   4. Continue to the next model.
#
# Default model order:
#   x-vla -> starvla -> internvla_m1
#
# Usage:
#   bash dynamic_SimplerEnv/scripts/benchmark_widowx_three_models.sh
#
# Common options:
#   --models <CSV>      Model subset / order (default: x-vla,starvla,internvla_m1)
#   --subtasks <CSV>    carrot,stack,spoon            (default: carrot,stack,spoon)
#   --levels <CSV>      0,1,2,3                       (default: 0,1,2,3)
#   --gpu <ID>          Sim/render GPU                (default: 0)
#   --server-gpu <ID>   Inference server GPU          (default: same as --gpu)
#   --output-root DIR   Forwarded to model fullsweeps (default: ./outputs)
#   --output-layout N   bench | legacy                (default: bench)
#   --log-root DIR      Forwarded to model fullsweeps (default: results)
#   --run-id-note STR   Optional suffix for sub-runs  (default: auto timestamp)
#   --dry-run           Print commands only
#
# Server / model overrides:
#   --xvla-model-path PATH
#   --xvla-lora-path PATH
#   --xvla-port PORT
#   --xvla-env NAME
#   --xvla-domain-id N
#   --xvla-official
#   --xvla-official-simplerenv-root DIR
#
#   --starvla-ckpt PATH
#   --starvla-port PORT
#   --starvla-env NAME
#   --starvla-unnorm-key KEY
#
#   --internvla-m1-ckpt PATH
#   --internvla-m1-port PORT
#   --internvla-m1-env NAME
#   --internvla-m1-unnorm-key KEY
#
#   --server-timeout SEC   Wait time for server port readiness (default: 600)
#   -h | --help
# ============================================================


# bash dynamic_SimplerEnv/scripts/benchmark_widowx_three_models.sh  --levels 3 --models internvla-m1 --subtasks carrot


# bash dynamic_SimplerEnv/scripts/benchmark_widowx_three_models.sh  --levels 3 

set -euo pipefail

if (( BASH_VERSINFO[0] < 4 )); then
    echo "ERROR: bash 4+ required (current: ${BASH_VERSION})" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SIMPLER_ROOT}/.." && pwd)"

MODELS_CSV="x-vla,starvla,internvla_m1"
SUBTASKS_CSV="carrot,stack,spoon"
LEVELS_CSV="0,1,2,3"
SIM_GPU="0"
SERVER_GPU=""
OUTPUT_ROOT="./outputs"
OUTPUT_LAYOUT="bench"
LOG_ROOT="results"
RUN_ID_NOTE=""
DRY_RUN="false"
SERVER_TIMEOUT="600"

XVLA_MODEL_PATH="download_checkpoints/xVLA/xvla-simpler-widowx"
XVLA_LORA_PATH=""
XVLA_PORT="8010"
XVLA_ENV_NAME="XVLA"
XVLA_DOMAIN_ID="0"
XVLA_OFFICIAL="0"
XVLA_OFFICIAL_SIMPLERENV_ROOT=""

STARVLA_CKPT="download_checkpoints/starVLA/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt"
STARVLA_PORT="10093"
STARVLA_ENV_NAME="starVLA"
STARVLA_UNNORM_KEY="oxe_bridge"

INTERNVLA_M1_CKPT="download_checkpoints/InternVLA/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt"
INTERNVLA_M1_PORT="10094"
INTERNVLA_M1_ENV_NAME="internvla-m1"
INTERNVLA_M1_UNNORM_KEY="bridge_dataset"

ACTIVE_SERVER_PID=""
ACTIVE_SERVER_MODEL=""
ACTIVE_SERVER_LOG=""
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_LOG_DIR="${SIMPLER_ROOT}/outputs/benchmark_widowx_three_models/${RUN_STAMP}"
SERVER_LOG_DIR="${BENCH_LOG_DIR}/server_logs"
mkdir -p "${SERVER_LOG_DIR}"

usage() {
    sed -n '3,53p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s\n' "${value}"
}

normalize_model_name() {
    case "$1" in
        x-vla|xvla) echo "x-vla" ;;
        starvla|star-vla) echo "starvla" ;;
        internvla_m1|internvla-m1|m1) echo "internvla_m1" ;;
        *)
            echo "ERROR: unknown model '$1'. Use x-vla, starvla, internvla_m1." >&2
            return 1
            ;;
    esac
}

port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -tuln 2>/dev/null | grep -q ":${port}[[:space:]]"
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tuln 2>/dev/null | grep -q ":${port}[[:space:]]"
    else
        python - <<PY >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(("127.0.0.1", ${port}))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
    fi
}

wait_for_port() {
    local port="$1"
    local timeout_s="$2"
    local elapsed=0
    while (( elapsed < timeout_s )); do
        if port_in_use "${port}"; then
            return 0
        fi
        if [[ -n "${ACTIVE_SERVER_PID}" ]] && ! kill -0 "${ACTIVE_SERVER_PID}" 2>/dev/null; then
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

cleanup_server() {
    if [[ -n "${ACTIVE_SERVER_PID}" ]] && kill -0 "${ACTIVE_SERVER_PID}" 2>/dev/null; then
        echo "[cleanup] stopping ${ACTIVE_SERVER_MODEL} server (pid=${ACTIVE_SERVER_PID})"
        kill "${ACTIVE_SERVER_PID}" 2>/dev/null || true
        sleep 2
        if kill -0 "${ACTIVE_SERVER_PID}" 2>/dev/null; then
            kill -9 "${ACTIVE_SERVER_PID}" 2>/dev/null || true
        fi
        wait "${ACTIVE_SERVER_PID}" 2>/dev/null || true
    fi
    ACTIVE_SERVER_PID=""
    ACTIVE_SERVER_MODEL=""
    ACTIVE_SERVER_LOG=""
}

trap cleanup_server EXIT INT TERM

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)        MODELS_CSV="$2";                 shift 2 ;;
        --subtasks)      SUBTASKS_CSV="$2";               shift 2 ;;
        --levels)        LEVELS_CSV="$2";                 shift 2 ;;
        --gpu)           SIM_GPU="$2";                    shift 2 ;;
        --server-gpu)    SERVER_GPU="$2";                 shift 2 ;;
        --output-root)   OUTPUT_ROOT="$2";                shift 2 ;;
        --output-layout) OUTPUT_LAYOUT="$2";              shift 2 ;;
        --log-root)      LOG_ROOT="$2";                   shift 2 ;;
        --run-id-note)   RUN_ID_NOTE="$2";                shift 2 ;;
        --dry-run)       DRY_RUN="true";                  shift ;;
        --server-timeout) SERVER_TIMEOUT="$2";            shift 2 ;;

        --xvla-model-path) XVLA_MODEL_PATH="$2";          shift 2 ;;
        --xvla-lora-path)  XVLA_LORA_PATH="$2";           shift 2 ;;
        --xvla-port)       XVLA_PORT="$2";                shift 2 ;;
        --xvla-env)        XVLA_ENV_NAME="$2";            shift 2 ;;
        --xvla-domain-id)  XVLA_DOMAIN_ID="$2";           shift 2 ;;
        --xvla-official)   XVLA_OFFICIAL="1";             shift ;;
        --xvla-official-simplerenv-root) XVLA_OFFICIAL_SIMPLERENV_ROOT="$2"; shift 2 ;;

        --starvla-ckpt)       STARVLA_CKPT="$2";          shift 2 ;;
        --starvla-port)       STARVLA_PORT="$2";          shift 2 ;;
        --starvla-env)        STARVLA_ENV_NAME="$2";      shift 2 ;;
        --starvla-unnorm-key) STARVLA_UNNORM_KEY="$2";    shift 2 ;;

        --internvla-m1-ckpt)       INTERNVLA_M1_CKPT="$2";       shift 2 ;;
        --internvla-m1-port)       INTERNVLA_M1_PORT="$2";       shift 2 ;;
        --internvla-m1-env)        INTERNVLA_M1_ENV_NAME="$2";   shift 2 ;;
        --internvla-m1-unnorm-key) INTERNVLA_M1_UNNORM_KEY="$2"; shift 2 ;;

        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${SERVER_GPU}" ]]; then
    SERVER_GPU="${SIM_GPU}"
fi

MODELS=()
declare -A SEEN_MODELS=()
IFS=',' read -r -a MODEL_TOKENS <<< "${MODELS_CSV// /}"
for token in "${MODEL_TOKENS[@]}"; do
    [[ -z "${token}" ]] && continue
    model="$(normalize_model_name "${token}")" || exit 1
    if [[ -z "${SEEN_MODELS[${model}]:-}" ]]; then
        MODELS+=("${model}")
        SEEN_MODELS["${model}"]=1
    fi
done
if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: --models resolved to an empty set." >&2
    exit 1
fi

SUBTASKS_CSV="${SUBTASKS_CSV// /}"
LEVELS_CSV="${LEVELS_CSV// /}"
if [[ -z "${SUBTASKS_CSV}" || -z "${LEVELS_CSV}" ]]; then
    echo "ERROR: --subtasks / --levels cannot be empty." >&2
    exit 1
fi

START_TS="$(date +%s)"
declare -a PASSED_MODELS=()
declare -a FAILED_MODELS=()

start_server_for_model() {
    local model="$1"
    local log_path="$2"
    local cmd=()
    local port=""

    case "${model}" in
        x-vla)
            port="${XVLA_PORT}"
            cmd=(
                bash "${SIMPLER_ROOT}/scripts/x-vla/start_xvla_server.sh"
                    --model-path "${XVLA_MODEL_PATH}"
                    --port "${XVLA_PORT}"
                    --env-name "${XVLA_ENV_NAME}"
            )
            if [[ -n "${XVLA_LORA_PATH}" ]]; then
                cmd+=(--lora-path "${XVLA_LORA_PATH}")
            fi
            ;;
        starvla)
            port="${STARVLA_PORT}"
            cmd=(
                bash "${SIMPLER_ROOT}/scripts/starvla/start_starvla_server.sh"
                    --ckpt-path "${STARVLA_CKPT}"
                    --port "${STARVLA_PORT}"
                    --env-name "${STARVLA_ENV_NAME}"
                    --use-bf16
            )
            ;;
        internvla_m1)
            port="${INTERNVLA_M1_PORT}"
            cmd=(
                bash "${SIMPLER_ROOT}/scripts/internvla_m1/start_internvla_m1_server.sh"
                    --ckpt-path "${INTERNVLA_M1_CKPT}"
                    --port "${INTERNVLA_M1_PORT}"
                    --env-name "${INTERNVLA_M1_ENV_NAME}"
                    --use-bf16
            )
            ;;
        *)
            echo "ERROR: unsupported model in start_server_for_model: ${model}" >&2
            return 1
            ;;
    esac

    if port_in_use "${port}"; then
        echo "ERROR: port ${port} is already in use before starting ${model}." >&2
        echo "  Stop the existing service or pass a different --*-port value." >&2
        return 1
    fi

    echo "[server] start ${model} on port ${port}"
    echo "[server] log  : ${log_path}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[dry-run] CUDA_VISIBLE_DEVICES=${SERVER_GPU} ${cmd[*]}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES="${SERVER_GPU}" "${cmd[@]}" > "${log_path}" 2>&1 &
    ACTIVE_SERVER_PID=$!
    ACTIVE_SERVER_MODEL="${model}"
    ACTIVE_SERVER_LOG="${log_path}"

    if ! wait_for_port "${port}" "${SERVER_TIMEOUT}"; then
        echo "ERROR: ${model} server did not become ready on port ${port} within ${SERVER_TIMEOUT}s." >&2
        echo "  See log: ${log_path}" >&2
        return 1
    fi
    echo "[server] ready ${model} pid=${ACTIVE_SERVER_PID} port=${port}"
}

run_fullsweep_for_model() {
    local model="$1"
    local note="$2"
    local cmd=()

    case "${model}" in
        x-vla)
            cmd=(
                bash "${SIMPLER_ROOT}/scripts/x-vla/run_bridge_xvla_fullsweep.sh"
                    --subtasks "${SUBTASKS_CSV}"
                    --levels "${LEVELS_CSV}"
                    --gpu "${SIM_GPU}"
                    --ckpt "${XVLA_MODEL_PATH}"
                    --server-host "127.0.0.1"
                    --server-port "${XVLA_PORT}"
                    --domain-id "${XVLA_DOMAIN_ID}"
                    --output-root "${OUTPUT_ROOT}"
                    --output-layout "${OUTPUT_LAYOUT}"
                    --log-root "${LOG_ROOT}"
                    --run-id-note "${note}"
            )
            if [[ "${XVLA_OFFICIAL}" -eq 1 ]]; then
                cmd+=(--official)
                if [[ -n "${XVLA_OFFICIAL_SIMPLERENV_ROOT}" ]]; then
                    cmd+=(--official-simplerenv-root "${XVLA_OFFICIAL_SIMPLERENV_ROOT}")
                fi
            fi
            ;;
        starvla)
            cmd=(
                bash "${SIMPLER_ROOT}/scripts/starvla/run_bridge_starvla_fullsweep.sh"
                    --subtasks "${SUBTASKS_CSV}"
                    --levels "${LEVELS_CSV}"
                    --gpu "${SIM_GPU}"
                    --ckpt "${STARVLA_CKPT}"
                    --server-host "127.0.0.1"
                    --server-port "${STARVLA_PORT}"
                    --unnorm-key "${STARVLA_UNNORM_KEY}"
                    --output-root "${OUTPUT_ROOT}"
                    --output-layout "${OUTPUT_LAYOUT}"
                    --log-root "${LOG_ROOT}"
                    --run-id-note "${note}"
            )
            ;;
        internvla_m1)
            cmd=(
                bash "${SIMPLER_ROOT}/scripts/internvla_m1/run_bridge_internvla_m1_fullsweep.sh"
                    --subtasks "${SUBTASKS_CSV}"
                    --levels "${LEVELS_CSV}"
                    --gpu "${SIM_GPU}"
                    --ckpt "${INTERNVLA_M1_CKPT}"
                    --server-host "127.0.0.1"
                    --server-port "${INTERNVLA_M1_PORT}"
                    --unnorm-key "${INTERNVLA_M1_UNNORM_KEY}"
                    --output-root "${OUTPUT_ROOT}"
                    --output-layout "${OUTPUT_LAYOUT}"
                    --log-root "${LOG_ROOT}"
                    --run-id-note "${note}"
            )
            ;;
        *)
            echo "ERROR: unsupported model in run_fullsweep_for_model: ${model}" >&2
            return 1
            ;;
    esac

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[dry-run] ${cmd[*]}"
        return 0
    fi

    "${cmd[@]}"
}

echo "============================================================"
echo "Benchmark: DyniSafety-SimplerEnv × WidowX Bridge × 3 models"
echo "  Models       : ${MODELS[*]}"
echo "  Subtasks     : ${SUBTASKS_CSV}"
echo "  Levels       : ${LEVELS_CSV}"
echo "  Sim GPU      : ${SIM_GPU}"
echo "  Server GPU   : ${SERVER_GPU}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  Output layout: ${OUTPUT_LAYOUT}"
echo "  Log root     : ${LOG_ROOT}"
echo "  Run stamp    : ${RUN_STAMP}"
echo "  Server logs  : ${SERVER_LOG_DIR}"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  Mode         : DRY RUN"
fi
echo "============================================================"

for model in "${MODELS[@]}"; do
    model_note="${RUN_ID_NOTE:-${RUN_STAMP}}-${model}"
    server_log="${SERVER_LOG_DIR}/${model}.log"

    echo ""
    echo "############################################################"
    echo "[model] ${model}"
    echo "############################################################"

    if ! start_server_for_model "${model}" "${server_log}"; then
        FAILED_MODELS+=("${model} (server start failed)")
        cleanup_server
        continue
    fi

    if run_fullsweep_for_model "${model}" "${model_note}"; then
        PASSED_MODELS+=("${model}")
    else
        rc=$?
        FAILED_MODELS+=("${model} (eval failed exit=${rc})")
    fi

    cleanup_server
done

END_TS="$(date +%s)"

echo ""
echo "============================================================"
echo "Benchmark summary (elapsed $((END_TS - START_TS))s)"
echo "  Passed models : ${#PASSED_MODELS[@]}"
for item in "${PASSED_MODELS[@]}"; do
    echo "    [OK]   ${item}"
done
echo "  Failed models : ${#FAILED_MODELS[@]}"
for item in "${FAILED_MODELS[@]}"; do
    echo "    [FAIL] ${item}"
done
echo "  Server logs   : ${SERVER_LOG_DIR}"
echo "============================================================"

if (( ${#FAILED_MODELS[@]} > 0 )); then
    exit 1
fi
exit 0
