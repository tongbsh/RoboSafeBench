#!/usr/bin/env bash
#
# Benchmark the registered VLA models on DyniSafety-RoboTwin.
# Within each model:
#   * the full (task, level) job list is scheduled globally
#   * workers are assigned over task-level jobs, not just levels
#   * each GPU can serve multiple concurrent jobs (model-dependent)
#   * when only one level is selected, multiple tasks can still run in parallel
#
# Supported models:
#   internvla_a1, fastwam, lingbot_va, starvla, xvla_official
#
# Examples:
#   bash run_all_vla_benchmark.sh
#   bash run_all_vla_benchmark.sh --test-num 5 --gpu-ids 0,1
#   bash dynamic_RoboTwin/scripts/run_all_vla_benchmark.sh  --test-num 1 --models fastwam --tasks adjust_bottle --levels 0,1,2,3
#   bash dynamic_RoboTwin/scripts/run_all_vla_benchmark.sh --models internvla_a1,starvla --tasks adjust_bottle --levels 0,1
#   bash run_all_vla_benchmark.sh --dry-run


#   bash dynamic_RoboTwin/scripts/run_all_vla_benchmark.sh --test-num 1 --gpu-ids 0,1 --models xvla_official --levels 0,1,2,3

# bash dynamic_RoboTwin/scripts/run_all_vla_benchmark.sh --gpu-ids 0 --test-num 5  --levels 3 --models internvla_a1
# bash dynamic_RoboTwin/scripts/run_all_vla_benchmark.sh --gpu-ids 0,1 --test-num 3 --levels 3  --models xvla_official
# bash dynamic_RoboTwin/scripts/run_all_vla_benchmark.sh --gpu-ids 0,1  --models fastwam

# bash dynamic_RoboTwin/scripts/run_all_vla_benchmark.sh --gpu-ids 0,1 --levels 1,2 --models internvla_a1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=task_registry.sh
source "${SCRIPT_DIR}/task_registry.sh"

DEFAULT_MODELS=(  starvla xvla_official)
# DEFAULT_MODELS=( lingbot_va starvla xvla_official)
DEFAULT_LEVELS=(0 1 2 3)
DEFAULT_TEST_NUM=20
DEFAULT_EXPERT_CHECK_TIMEOUT=36000
DEFAULT_JOBS_PER_GPU=2
DEFAULT_SERVER_WAIT_TIMEOUT=600
XVLA_OFFICIAL_PORT_BASE="${XVLA_OFFICIAL_PORT_BASE:-38010}"
LINGBOT_START_PORT_BASE="${LINGBOT_START_PORT_BASE:-39056}"
LINGBOT_MASTER_PORT_BASE="${LINGBOT_MASTER_PORT_BASE:-39061}"
STARVLA_PORT_BASE="${STARVLA_PORT_BASE:-40093}"

MODEL_LIST=("${DEFAULT_MODELS[@]}")
TASK_LIST=("${DYNISAFETY_DEFAULT_TASKS[@]}")
LEVEL_LIST=("${DEFAULT_LEVELS[@]}")
GPU_IDS=()
TEST_NUM="${TEST_NUM:-${DEFAULT_TEST_NUM}}"
EXPERT_CHECK_TIMEOUT_S="${EXPERT_CHECK_TIMEOUT_S:-${DEFAULT_EXPERT_CHECK_TIMEOUT}}"
JOBS_PER_GPU="${JOBS_PER_GPU:-${DEFAULT_JOBS_PER_GPU}}"
DRY_RUN=0
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-${DEFAULT_SERVER_WAIT_TIMEOUT}}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_ROOT_DEFAULT="${PROJ_ROOT}/outputs/dynisafety/_benchmark_runs/${TIMESTAMP}"
LOG_ROOT="${LOG_ROOT:-${LOG_ROOT_DEFAULT}}"

declare -a ACTIVE_SERVER_PIDS=()
declare -A ACTIVE_SERVER_PORT_BY_GPU=()
declare -A ACTIVE_SERVER_LOG_BY_GPU=()

usage() {
    cat <<'EOF'
Usage: bash run_all_vla_benchmark.sh [options]

Options:
  --models <a,b,c>      Comma-separated model list.
  --tasks <a,b,c>       Comma-separated task list.
  --levels <0,1,2,3>    Comma-separated level list.
  --gpu-ids <0,1>       Comma-separated GPU ids to use. Defaults to all visible GPUs.
  --test-num <N>        Trajectories per (task, level). Default 20.
  --expert-check-timeout <S>
                        Max total expert-check wall-clock seconds per job. Default 3600.
  --jobs-per-gpu <N>    Concurrent level jobs per GPU. Default 2.
  --log-root <PATH>     Root directory for benchmark logs.
  --server-timeout <S>  Seconds to wait for each server to become healthy. Default 600.
  --dry-run             Print the execution plan without launching anything.
  -h, --help            Show this help.

Environment overrides:
  TEST_NUM, EXPERT_CHECK_TIMEOUT_S, JOBS_PER_GPU, LOG_ROOT, SERVER_WAIT_TIMEOUT
  FASTWAM_ROOT, FASTWAM_CKPT, FASTWAM_DATASET_STATS
  XVLA_OFFICIAL_PORT_BASE, LINGBOT_START_PORT_BASE, LINGBOT_MASTER_PORT_BASE, STARVLA_PORT_BASE
EOF
}

effective_jobs_per_gpu_for_model() {
    local model="$1"
    if is_single_job_per_gpu_model "${model}"; then
        printf '1\n'
    else
        printf '%s\n' "${JOBS_PER_GPU}"
    fi
}

is_single_job_per_gpu_model() {
    local model="$1"
    case "${model}" in
        lingbot_va|fastwam) return 0 ;;
        *) return 1 ;;
    esac
}

split_list_arg() {
    local raw="$1"
    raw="${raw//,/ }"
    read -r -a _split_result <<< "${raw}"
    printf '%s\n' "${_split_result[@]}"
}

load_list_from_arg() {
    local target_name="$1"
    local raw="$2"
    local -a values=()
    mapfile -t values < <(split_list_arg "${raw}")
    if [[ ${#values[@]} -eq 0 ]]; then
        echo "[ERROR] Empty value for ${target_name}" >&2
        exit 2
    fi
    case "${target_name}" in
        models) MODEL_LIST=("${values[@]}") ;;
        tasks) TASK_LIST=("${values[@]}") ;;
        levels) LEVEL_LIST=("${values[@]}") ;;
        gpus) GPU_IDS=("${values[@]}") ;;
        *)
            echo "[ERROR] Unsupported target list: ${target_name}" >&2
            exit 2
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            load_list_from_arg models "$2"
            shift 2
            ;;
        --tasks)
            load_list_from_arg tasks "$2"
            shift 2
            ;;
        --levels)
            load_list_from_arg levels "$2"
            shift 2
            ;;
        --gpu-ids)
            load_list_from_arg gpus "$2"
            shift 2
            ;;
        --test-num)
            TEST_NUM="$2"
            shift 2
            ;;
        --expert-check-timeout)
            EXPERT_CHECK_TIMEOUT_S="$2"
            shift 2
            ;;
        --jobs-per-gpu)
            JOBS_PER_GPU="$2"
            shift 2
            ;;
        --log-root)
            LOG_ROOT="$2"
            shift 2
            ;;
        --server-timeout)
            SERVER_WAIT_TIMEOUT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if ! [[ "${TEST_NUM}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --test-num must be a non-negative integer, got: ${TEST_NUM}" >&2
    exit 2
fi
if ! [[ "${EXPERT_CHECK_TIMEOUT_S}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[ERROR] --expert-check-timeout must be a non-negative number, got: ${EXPERT_CHECK_TIMEOUT_S}" >&2
    exit 2
fi
if ! [[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --jobs-per-gpu must be a positive integer, got: ${JOBS_PER_GPU}" >&2
    exit 2
fi
if ! [[ "${SERVER_WAIT_TIMEOUT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --server-timeout must be a non-negative integer, got: ${SERVER_WAIT_TIMEOUT}" >&2
    exit 2
fi

is_valid_model() {
    local model="$1"
    case "${model}" in
        internvla_a1|fastwam|lingbot_va|starvla|xvla_official) return 0 ;;
        *) return 1 ;;
    esac
}

for model in "${MODEL_LIST[@]}"; do
    if ! is_valid_model "${model}"; then
        echo "[ERROR] Unsupported model: ${model}" >&2
        exit 2
    fi
    if [[ ! -f "${SCRIPT_DIR}/${model}/eval_task.sh" ]]; then
        echo "[ERROR] Missing eval_task.sh for ${model}" >&2
        exit 2
    fi
done

for task in "${TASK_LIST[@]}"; do
    if ! dynisafety_has_task "${task}"; then
        echo "[ERROR] Unknown DyniSafety task: ${task}" >&2
        exit 2
    fi
done

for level in "${LEVEL_LIST[@]}"; do
    if ! [[ "${level}" =~ ^[0-3]$ ]]; then
        echo "[ERROR] Level must be one of 0,1,2,3; got: ${level}" >&2
        exit 2
    fi
done

detect_gpu_ids() {
    if [[ ${#GPU_IDS[@]} -gt 0 ]]; then
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[ERROR] nvidia-smi not found. Please pass --gpu-ids explicitly." >&2
        exit 2
    fi
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
    if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
        echo "[ERROR] No GPUs detected." >&2
        exit 2
    fi
}

is_server_model() {
    local model="$1"
    case "${model}" in
        lingbot_va|starvla|xvla_official) return 0 ;;
        *) return 1 ;;
    esac
}

port_is_open() {
    local host="$1"
    local port="$2"
    python - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local timeout_s="$3"
    local pid="$4"
    local start_ts
    start_ts=$(date +%s)

    while true; do
        if port_is_open "${host}" "${port}"; then
            return 0
        fi
        if ! kill -0 "${pid}" >/dev/null 2>&1; then
            return 1
        fi
        if (( $(date +%s) - start_ts >= timeout_s )); then
            return 1
        fi
        sleep 2
    done
}

cleanup_active_servers() {
    if [[ ${#ACTIVE_SERVER_PIDS[@]} -eq 0 ]]; then
        return
    fi
    for pid in "${ACTIVE_SERVER_PIDS[@]}"; do
        if kill -0 "${pid}" >/dev/null 2>&1; then
            kill "${pid}" >/dev/null 2>&1 || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
    ACTIVE_SERVER_PIDS=()
    ACTIVE_SERVER_PORT_BY_GPU=()
    ACTIVE_SERVER_LOG_BY_GPU=()
}

trap cleanup_active_servers EXIT INT TERM

build_slot_gpus() {
    local job_count="$1"
    local jobs_per_gpu="$2"
    local max_slots=$(( ${#GPU_IDS[@]} * jobs_per_gpu ))
    local worker_count=$(( job_count < max_slots ? job_count : max_slots ))
    if (( worker_count > 4 )); then
        worker_count=4
    fi
    if (( worker_count < 1 )); then
        echo "[ERROR] No worker slots available." >&2
        exit 2
    fi

    SLOT_GPUS=()
    local server_gpu_count=$(( (worker_count + jobs_per_gpu - 1) / jobs_per_gpu ))
    local idx=0
    while (( idx < server_gpu_count )); do
        local gpu="${GPU_IDS[$idx]}"
        local replica=0
        while (( replica < jobs_per_gpu && ${#SLOT_GPUS[@]} < worker_count )); do
            SLOT_GPUS+=("${gpu}")
            replica=$((replica + 1))
        done
        idx=$((idx + 1))
    done
    SLOT_GPUS=("${SLOT_GPUS[@]:0:${worker_count}}")
}

start_servers_for_model() {
    local model="$1"
    local -a unique_gpus=()
    local last_gpu=""
    local slot_gpu
    for slot_gpu in "${SLOT_GPUS[@]}"; do
        if [[ "${slot_gpu}" != "${last_gpu}" ]]; then
            unique_gpus+=("${slot_gpu}")
            last_gpu="${slot_gpu}"
        fi
    done

    local server_idx=0
    local launch_script="${SCRIPT_DIR}/${model}/launch_server.sh"
    if [[ ! -f "${launch_script}" ]]; then
        echo "[ERROR] Missing launch_server.sh for ${model}: ${launch_script}" >&2
        exit 1
    fi

    for gpu in "${unique_gpus[@]}"; do
        local port
        local log_file="${LOG_ROOT}/servers/${model}/gpu${gpu}.log"
        mkdir -p "$(dirname "${log_file}")"
        case "${model}" in
            xvla_official)
                port=$(( XVLA_OFFICIAL_PORT_BASE + server_idx ))
                local output_dir="${LOG_ROOT}/servers/${model}/gpu${gpu}_artifacts"
                CUDA_VISIBLE_DEVICES="${gpu}" PORT="${port}" OUTPUT_DIR="${output_dir}" \
                    bash "${launch_script}" >"${log_file}" 2>&1 &
                ;;
            starvla)
                port=$(( STARVLA_PORT_BASE + server_idx ))
                CUDA_VISIBLE_DEVICES="${gpu}" PORT="${port}" \
                    bash "${launch_script}" >"${log_file}" 2>&1 &
                ;;
            lingbot_va)
                port=$(( LINGBOT_START_PORT_BASE + server_idx ))
                local master_port=$(( LINGBOT_MASTER_PORT_BASE + server_idx ))
                local save_root="${LOG_ROOT}/servers/${model}/gpu${gpu}_artifacts"
                CUDA_VISIBLE_DEVICES="${gpu}" START_PORT="${port}" MASTER_PORT="${master_port}" SAVE_ROOT="${save_root}" \
                    bash "${launch_script}" >"${log_file}" 2>&1 &
                ;;
            *)
                echo "[ERROR] start_servers_for_model called for non-server model ${model}" >&2
                exit 1
                ;;
        esac
        local pid=$!
        ACTIVE_SERVER_PIDS+=("${pid}")
        ACTIVE_SERVER_PORT_BY_GPU["${gpu}"]="${port}"
        ACTIVE_SERVER_LOG_BY_GPU["${gpu}"]="${log_file}"
        echo "[server] ${model} gpu=${gpu} port=${port} log=${log_file}"
        if (( DRY_RUN == 0 )); then
            if ! wait_for_port "127.0.0.1" "${port}" "${SERVER_WAIT_TIMEOUT}" "${pid}"; then
                echo "[ERROR] ${model} server failed to become ready on 127.0.0.1:${port}" >&2
                echo "[ERROR] Server log: ${log_file}" >&2
                tail -n 40 "${log_file}" >&2 || true
                exit 1
            fi
        fi
        server_idx=$((server_idx + 1))
    done
}

launch_eval_job() {
    local model="$1"
    local task="$2"
    local level="$3"
    local gpu="$4"
    local log_file="${LOG_ROOT}/${model}/${task}/level${level}.log"
    mkdir -p "$(dirname "${log_file}")"

    (
        export TEST_NUM="${TEST_NUM}"
        export EXPERT_CHECK_TIMEOUT_S="${EXPERT_CHECK_TIMEOUT_S}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        if is_server_model "${model}"; then
            export HOST="127.0.0.1"
            export PORT="${ACTIVE_SERVER_PORT_BY_GPU[${gpu}]}"
        fi
        bash "${SCRIPT_DIR}/${model}/eval_task.sh" --task "${task}" --level "${level}"
    ) >"${log_file}" 2>&1 &
    LAST_LAUNCH_PID="$!"
    LAST_LAUNCH_LOG="${log_file}"
}

print_plan() {
    echo "================================================================="
    echo "[benchmark] MODELS    = ${MODEL_LIST[*]}"
    echo "[benchmark] TASKS     = ${TASK_LIST[*]}"
    echo "[benchmark] LEVELS    = ${LEVEL_LIST[*]}"
    echo "[benchmark] TEST_NUM  = ${TEST_NUM}"
    echo "[benchmark] EC_TIMEOUT= ${EXPERT_CHECK_TIMEOUT_S}"
    echo "[benchmark] JOBS/GPU  = ${JOBS_PER_GPU}"
    echo "[benchmark] GPU_IDS   = ${GPU_IDS[*]}"
    echo "[benchmark] LOG_ROOT  = ${LOG_ROOT}"
    echo "================================================================="
}

run_model() {
    local model="$1"
    local level_count="${#LEVEL_LIST[@]}"
    local task_count="${#TASK_LIST[@]}"
    local job_count=$(( task_count * level_count ))
    local model_jobs_per_gpu
    model_jobs_per_gpu="$(effective_jobs_per_gpu_for_model "${model}")"
    build_slot_gpus "${job_count}" "${model_jobs_per_gpu}"
    local worker_count="${#SLOT_GPUS[@]}"
    local -a queued_tasks=()
    local -a queued_levels=()
    local task
    local level

    for task in "${TASK_LIST[@]}"; do
        for level in "${LEVEL_LIST[@]}"; do
            queued_tasks+=("${task}")
            queued_levels+=("${level}")
        done
    done

    echo ""
    echo "#################################################################"
    echo "[benchmark] MODEL      = ${model}"
    echo "[benchmark] SLOT_GPUS  = ${SLOT_GPUS[*]}"
    echo "[benchmark] JOBS       = ${job_count} (${task_count} task x ${level_count} level)"
    echo "[benchmark] WORKERS    = ${worker_count} (${model_jobs_per_gpu} jobs / GPU, capped by pending jobs)"
    if is_single_job_per_gpu_model "${model}"; then
        echo "[benchmark] NOTE       = forcing ${model} to 1 job / GPU because video VLA inference uses substantial GPU memory; concurrent level jobs on the same GPU can OOM."
        if [[ "${JOBS_PER_GPU}" != "${model_jobs_per_gpu}" ]]; then
            echo "[benchmark] NOTE       = configured JOBS_PER_GPU=${JOBS_PER_GPU}, but ${model} overrides it to ${model_jobs_per_gpu}."
        fi
    fi
    echo "#################################################################"

    cleanup_active_servers
    if is_server_model "${model}"; then
        if (( DRY_RUN == 1 )); then
            echo "[dry-run] would start ${model} servers on GPUs: ${SLOT_GPUS[*]}"
        else
            start_servers_for_model "${model}"
        fi
    fi

    local model_passed=0
    local model_failed=0
    local job_offset=0
    while (( job_offset < job_count )); do
        local batch_size=$(( job_count - job_offset ))
        if (( batch_size > worker_count )); then
            batch_size=${worker_count}
        fi

        local -a pids=()
        local -a logs=()
        local -a tags=()
        local -a gpus=()
        local i
        for (( i=0; i<batch_size; i++ )); do
            task="${queued_tasks[$((job_offset + i))]}"
            level="${queued_levels[$((job_offset + i))]}"
            local gpu="${SLOT_GPUS[$i]}"
            tags+=("${task} level${level}")
            gpus+=("${gpu}")
            if (( DRY_RUN == 1 )); then
                echo "[dry-run] ${model} | ${task} | level${level} | gpu=${gpu}"
            else
                launch_eval_job "${model}" "${task}" "${level}" "${gpu}"
                pids+=("${LAST_LAUNCH_PID}")
                logs+=("${LAST_LAUNCH_LOG}")
                echo "[launch] ${model} | ${task} | level${level} | gpu=${gpu} | log=${LAST_LAUNCH_LOG}"
            fi
        done

        for (( i=0; i<batch_size; i++ )); do
            if (( DRY_RUN == 1 )); then
                model_passed=$((model_passed + 1))
                continue
            fi
            if wait "${pids[$i]}"; then
                echo "[done]  ${model} | ${tags[$i]} | gpu=${gpus[$i]}"
                model_passed=$((model_passed + 1))
            else
                echo "[fail]  ${model} | ${tags[$i]} | gpu=${gpus[$i]} | log=${logs[$i]}"
                tail -n 20 "${logs[$i]}" || true
                model_failed=$((model_failed + 1))
            fi
        done

        job_offset=$((job_offset + batch_size))
    done

    cleanup_active_servers

    echo ""
    echo "================================================================="
    echo "[benchmark] MODEL SUMMARY : ${model}"
    echo "[benchmark] Passed        : ${model_passed}"
    echo "[benchmark] Failed        : ${model_failed}"
    echo "================================================================="

    MODEL_PASSED=$((MODEL_PASSED + model_passed))
    MODEL_FAILED=$((MODEL_FAILED + model_failed))
}

detect_gpu_ids
mkdir -p "${LOG_ROOT}"
print_plan

MODEL_PASSED=0
MODEL_FAILED=0

if (( DRY_RUN == 1 )); then
    echo "[dry-run] no processes will be launched."
fi

for model in "${MODEL_LIST[@]}"; do
    run_model "${model}"
done

echo ""
echo "================================================================="
echo "[benchmark] OVERALL Passed : ${MODEL_PASSED}"
echo "[benchmark] OVERALL Failed : ${MODEL_FAILED}"
echo "[benchmark] LOG_ROOT       : ${LOG_ROOT}"
echo "================================================================="

if (( MODEL_FAILED > 0 )); then
    exit 1
fi
exit 0
