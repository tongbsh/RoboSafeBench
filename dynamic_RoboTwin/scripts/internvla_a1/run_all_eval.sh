#!/usr/bin/env bash
# 串行测试 InternVLA-A1 下默认 20 个 DyniSafety task 的 level0-3。
#
# 用法:
#   bash run_all_eval.sh                                   # 默认测试全部 20 个 task 的 level0-3
#   bash run_all_eval.sh handover_block                    # 只测 handover_block 的 level0-3
#   bash run_all_eval.sh place_a2b_left place_a2b_right    # 测多个指定 task
#   bash run_all_eval.sh --test-num 5                      # 每个 task 只跑 5 条轨迹
#   bash run_all_eval.sh --test-num 5 handover_block       # 组合使用
#
#   TASKS="open_laptop click_bell" bash run_all_eval.sh
#   LEVELS="0" bash dynamic_RoboTwin/scripts/internvla_a1/run_all_eval.sh place_a2b_right      # 只跑指定 level
#   TEST_NUM=5  bash run_all_eval.sh open_laptop           # 等价于 --test-num 5
#
# 环境变量:
#   TASKS     覆盖任务列表（空格分隔）。优先级低于命令行位置参数。
#   LEVELS    覆盖 level 列表（空格分隔），默认 "0 1 2 3"。
#   TEST_NUM  每个 (task, level) 跑多少条轨迹，默认 20（由 *_common.sh 决定）。
#             命令行 --test-num 优先于环境变量 TEST_NUM。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="${SCRIPT_DIR}/per_task_eval"
# shellcheck source=../task_registry.sh
source "${SCRIPT_DIR}/../task_registry.sh"

ALL_TASKS=("${DYNISAFETY_DEFAULT_TASKS[@]}")

DEFAULT_TEST_NUM=20

usage() {
    sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

POSITIONAL=()
CLI_TEST_NUM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-num|-n)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --test-num requires a value"; exit 2
            fi
            CLI_TEST_NUM="$2"
            shift 2
            ;;
        --test-num=*)
            CLI_TEST_NUM="${1#*=}"
            shift
            ;;
        -h|--help)
            usage; exit 0
            ;;
        --)
            shift
            POSITIONAL+=("$@")
            break
            ;;
        -*)
            echo "[ERROR] Unknown option: $1"; usage; exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ -n "${CLI_TEST_NUM}" ]]; then
    if ! [[ "${CLI_TEST_NUM}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] --test-num must be a non-negative integer, got: ${CLI_TEST_NUM}"
        exit 2
    fi
    export TEST_NUM="${CLI_TEST_NUM}"
elif [[ -n "${TEST_NUM:-}" ]]; then
    export TEST_NUM
fi
EFFECTIVE_TEST_NUM="${TEST_NUM:-${DEFAULT_TEST_NUM}}"

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
    TASK_LIST=("${POSITIONAL[@]}")
elif [[ -n "${TASKS:-}" ]]; then
    read -r -a TASK_LIST <<< "${TASKS}"
else
    TASK_LIST=("${ALL_TASKS[@]}")
fi

read -r -a LEVEL_LIST <<< "${LEVELS:-0 1 2 3}"

is_valid_task() {
    local t="$1"
    dynisafety_has_task "${t}"
}

for t in "${TASK_LIST[@]}"; do
    if ! is_valid_task "${t}"; then
        echo "[ERROR] Unknown task: ${t}"
        echo "[INFO]  Valid tasks: ${ALL_TASKS[*]}"
        exit 1
    fi
done

TOTAL_RUNS=$(( ${#TASK_LIST[@]} * ${#LEVEL_LIST[@]} ))
TOTAL_TRAJ=$(( TOTAL_RUNS * EFFECTIVE_TEST_NUM ))

echo "================================================================="
echo "[run_all_eval] TASKS     = ${TASK_LIST[*]}"
echo "[run_all_eval] LEVELS    = ${LEVEL_LIST[*]}"
echo "[run_all_eval] TEST_NUM  = ${EFFECTIVE_TEST_NUM} (per task-level)"
echo "[run_all_eval] PLAN      = ${#TASK_LIST[@]} task * ${#LEVEL_LIST[@]} level = ${TOTAL_RUNS} run(s), ${TOTAL_TRAJ} traj total"
echo "[run_all_eval] EVAL_DIR  = ${EVAL_DIR}"
echo "================================================================="

declare -a PASSED=()
declare -a FAILED=()
TOTAL=0

START_TS=$(date +%s)

for task in "${TASK_LIST[@]}"; do
    for level in "${LEVEL_LIST[@]}"; do
        script="${SCRIPT_DIR}/eval_task.sh --task ${task} --level ${level}"
        tag="${task} level${level}"
        TOTAL=$((TOTAL + 1))

        echo ""
        echo "-----------------------------------------------------------------"
        echo "[$(date '+%F %T')] >>> START : ${tag}  (TEST_NUM=${EFFECTIVE_TEST_NUM})"
        echo "[run_all_eval] script = ${script}"
        echo "-----------------------------------------------------------------"

        if [[ ! -f "${SCRIPT_DIR}/eval_task.sh" ]]; then
            echo "[ERROR] Script not found: ${SCRIPT_DIR}/eval_task.sh"
            FAILED+=("${tag} (missing eval_task.sh)")
            continue
        fi

        step_start=$(date +%s)
        if bash "${SCRIPT_DIR}/eval_task.sh" --task "${task}" --level "${level}"; then
            step_end=$(date +%s)
            echo "[$(date '+%F %T')] <<< DONE  : ${tag} ($((step_end - step_start))s)"
            PASSED+=("${tag}")
        else
            rc=$?
            step_end=$(date +%s)
            echo "[$(date '+%F %T')] <<< FAIL  : ${tag} (exit=${rc}, $((step_end - step_start))s)"
            FAILED+=("${tag} (exit=${rc})")
        fi
    done
done

END_TS=$(date +%s)

echo ""
echo "================================================================="
echo "[run_all_eval] Summary  (elapsed $((END_TS - START_TS))s, TEST_NUM=${EFFECTIVE_TEST_NUM})"
echo "[run_all_eval] Total    : ${TOTAL}"
echo "[run_all_eval] Passed   : ${#PASSED[@]}"
for t in "${PASSED[@]}"; do echo "    [OK]   ${t}"; done
echo "[run_all_eval] Failed   : ${#FAILED[@]}"
for t in "${FAILED[@]}"; do echo "    [FAIL] ${t}"; done
echo "================================================================="

if (( ${#FAILED[@]} > 0 )); then
    exit 1
fi
exit 0
