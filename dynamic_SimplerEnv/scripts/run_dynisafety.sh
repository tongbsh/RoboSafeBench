#!/bin/bash
# ============================================================
# Generic DyniSafety evaluation script — model-agnostic.
#
# Run from the dynamic_SimplerEnv/ root directory.
#
# ── Supported tasks (--task) ────────────────────────────────
#   pick_coke_can  – GraspSingleOpenedCokeCanInScene-v0  (default)
#   move_near      – MoveNearGoogleInScene-v0
#   drawer         – Open/CloseTop/Middle/BottomDrawerCustomInScene-v0
#   bridge         – WidowX Bridge 4-task suite (bridge.sh parity); policy-setup widowx_bridge
#   bridge_carrot | bridge_stack | bridge_spoon | bridge_eggplant – single Bridge subtask
#
# ── Supported levels (--level) ──────────────────────────────
#   0 | 1 | 2 | 3 | all  (all = 1,2,3 only; use 0 explicitly for baseline)
#
# ── Registered models (--model) ─────────────────────────────
#   openvla / openvla_oft / cogact / spatialvla / spatialvla_server
#   openpifast / openpi / pi0 / pi05 / lerobotpifast
#   memoryvla / xvla / xvla_official / starvla / univla / univla_local
#   internvla_m1 / dreamvla
#   gr00t / rt1 / octo-base / octo-small / octo-server
#   <custom>  — register via register_policy() in
#               SimplerEnv-OpenVLA/simpler_env/policies/__init__.py
#
# ── Usage examples ───────────────────────────────────────────
#   # Standard OpenVLA (in-process)
#   bash scripts/run_dynisafety.sh --model openvla \
#       --ckpt openvla/openvla-7b --level 1
#
#   # OpenVLA-OFT (in-process, openvla-oft conda env)
#   bash scripts/run_dynisafety.sh --model openvla_oft \
#       --ckpt moojink/openvla-7b-oft-finetuned-libero-spatial --level 1
#
#   # π₀ via openpi server (start server first with start_pi0_server.sh)
#   OPENPI_SERVER_HOST=127.0.0.1 OPENPI_SERVER_PORT=8000 \
#   bash scripts/run_dynisafety.sh --model pi0 \
#       --ckpt ignored --level 1
#
#   # π₀.₅ — same client as pi0; point OPENPI_SERVER_* at a π₀.₅ openpi server
#   OPENPI_SERVER_HOST=127.0.0.1 OPENPI_SERVER_PORT=8001 \
#   bash scripts/run_dynisafety.sh --model pi05 --ckpt ignored --level 1
#
#   # MemoryVLA (start server first):
#   #   bash scripts/memoryvla/start_memoryvla_server.sh \
#   #       --model-path shihao1895/memvla-bridge --unnorm-key bridge_orig
#   MEMORYVLA_SERVER_HOST=127.0.0.1 MEMORYVLA_SERVER_PORT=8111 \
#   bash scripts/run_dynisafety.sh --model memoryvla --ckpt ignored --level 1
#
#   # X-VLA (start server first):
#   #   bash scripts/x-vla/start_xvla_server.sh \
#   #       --model-path 2toINF/X-VLA-WidowX
#   XVLA_SERVER_HOST=127.0.0.1 XVLA_SERVER_PORT=8010 XVLA_DOMAIN_ID=0 \
#   bash scripts/run_dynisafety.sh --model xvla --ckpt ignored --level 1
#
#   # StarVLA (start server first):
#   #   bash scripts/starvla/start_starvla_server.sh \
#   #       --ckpt-path /path/to/starvla/checkpoints/steps_50000_pytorch_model.pt
#   STARVLA_SERVER_HOST=127.0.0.1 STARVLA_SERVER_PORT=10093 \
#   bash scripts/run_dynisafety.sh --model starvla \
#       --ckpt /path/to/starvla/checkpoints/steps_50000_pytorch_model.pt \
#       --level 1
#
#   # UniVLA (start server first, univla conda):
#   #   bash scripts/univla/start_univla_server.sh \
#   #       bash scripts/univla/start_univla_server.sh --port 8020
#   UNIVLA_SERVER_HOST=127.0.0.1 UNIVLA_SERVER_PORT=8020 \
#   bash scripts/run_dynisafety.sh --model univla --ckpt ignored --level 1
#
#   # UniVLA in-process (univla / openvla-oft style conda; UNIVLA_ROOT, action_decoder.pt)
#   bash scripts/run_dynisafety.sh --model univla_local \
#       --ckpt /path/to/univla-libero-spatial --level 1
#
#   # InternVLA-M1 (start server first, internvla_m1 conda):
#   #   bash scripts/internvla_m1/start_internvla_m1_server.sh \
#   #       --ckpt-path /path/to/RUN/checkpoints/steps_50000_pytorch_model.pt
#   INTERNVLA_M1_SERVER_HOST=127.0.0.1 INTERNVLA_M1_SERVER_PORT=10094 \
#   bash scripts/run_dynisafety.sh --model internvla_m1 \
#       --ckpt /path/to/RUN/checkpoints/steps_50000_pytorch_model.pt \
#       --level 1
#
#   # Dream-VLA (start server first, dreamvla conda; best-effort zero-shot Bridge):
#   #   bash scripts/dreamvla/start_dreamvla_server.sh \
#   #       --pretrained-checkpoint download_checkpoints/dream-vla
#   DREAMVLA_SERVER_HOST=127.0.0.1 DREAMVLA_SERVER_PORT=8030 \
#   bash scripts/run_dynisafety.sh --model dreamvla --ckpt ignored --level 1
#
#   # SpatialVLA HTTP server (start spatialVLA/server.py first)
#   SPATIALVLA_SERVER_HOST=127.0.0.1 SPATIALVLA_SERVER_PORT=8112 \
#   bash scripts/run_dynisafety.sh --model spatialvla_server --ckpt ignored --level 1
#
#   bash scripts/run_dynisafety.sh --model cogact \
#       --ckpt CogACT/CogACT-Large --task move_near --level all
#
# ── Environment variables ────────────────────────────────────
#   SIMPLERENV_ROOT              Override SimplerEnv-OpenVLA root path.
#                                Default: dynamic_SimplerEnv/SimplerEnv-OpenVLA (bundled)
#   OPENPI_SERVER_HOST           openpi server host (default: 127.0.0.1)
#   OPENPI_SERVER_PORT           openpi server port (default: 8000)
#   MEMORYVLA_SERVER_HOST        MemoryVLA server host (default: 127.0.0.1)
#   MEMORYVLA_SERVER_PORT        MemoryVLA server port (default: 8111)
#   MEMORYVLA_UNNORM_KEY         unnorm_key override for MemoryVLA
#   XVLA_SERVER_HOST             X-VLA server host (default: 127.0.0.1)
#   XVLA_SERVER_PORT             X-VLA server port (default: 8010)
#   XVLA_DOMAIN_ID               X-VLA domain id (0=WidowX, 3=LIBERO; default: 0)
#   OPENVLA_OFT_ROOT             openvla-oft repo root (default: auto)
#   OPENVLA_OFT_USE_L1_REGRESSION  true/false (default: true)
#   OPENVLA_OFT_USE_PROPRIO      true/false (default: false)
#   OPENVLA_OFT_UNNORM_KEY       Override action unnorm key (default: auto)
#   STARVLA_SERVER_HOST          StarVLA server host (default: 127.0.0.1)
#   STARVLA_SERVER_PORT          StarVLA server port (default: 10093)
#   STARVLA_ROOT                 StarVLA repo root (default: vlaModelHome/starVLA)
#   STARVLA_USE_DDIM             true/false (default: true)
#   STARVLA_NUM_DDIM_STEPS       int (default: 10)
#   STARVLA_UNNORM_KEY           e.g. oxe_bridge | oxe_rt1 when dataset_statistics.json has multiple keys
#   STARVLA_DISABLE_ACTION_ENSEMBLE  true = first chunk step only (default: off; matches StarVLA ModelClient)
#   STARVLA_ADAPTIVE_ENSEMBLE_ALPHA   float (default 0.1); STARVLA_ACTION_ENSEMBLE_HORIZON optional override
#   UNIVLA_SERVER_HOST           UniVLA server host (default: 127.0.0.1)
#   UNIVLA_SERVER_PORT           UniVLA server port (default: 8020)
#   INTERNVLA_M1_SERVER_HOST     InternVLA-M1 server host (default: 127.0.0.1)
#   INTERNVLA_M1_SERVER_PORT     InternVLA-M1 server port (default: 10094 - avoid clash with starvla 10093)
#   INTERNVLA_M1_ROOT            InternVLA-M1 repo root (default: vlaModelHome/InternVLA-M1)
#   INTERNVLA_M1_USE_DDIM        true/false (default: true)
#   INTERNVLA_M1_NUM_DDIM_STEPS  int (default: 5; matches official M1Inference)
#   INTERNVLA_M1_CFG_SCALE       float (default: 1.5; matches official M1Inference)
#   INTERNVLA_M1_UNNORM_KEY      e.g. bridge_dataset (default for widowx_bridge)
#   INTERNVLA_M1_DISABLE_ACTION_ENSEMBLE  true = first chunk step only (default: off)
#   INTERNVLA_M1_ADAPTIVE_ENSEMBLE_ALPHA  float (default 0.1)
#   INTERNVLA_M1_ACTION_ENSEMBLE_HORIZON  optional override (default: 7 for widowx_bridge)
#   DREAMVLA_SERVER_HOST         Dream-VLA HTTP server (default: 127.0.0.1)
#   DREAMVLA_SERVER_PORT         (default: 8030)
#   DREAMVLA_VLA_HOME            Dream-VLX/vla repo root (default: vlaModelHome/Dream-VLX/vla)
#   DREAMVLA_CHECKPOINT          Override default ckpt dir (default: download_checkpoints/dream-vla)
#   DREAMVLA_ATTN_IMPLEMENTATION Attn backend (default: flash_attention_2; sdpa fallback via --no-flash-attn)
#   SPATIALVLA_SERVER_HOST       SpatialVLA HTTP server (default: 127.0.0.1)
#   SPATIALVLA_SERVER_PORT       (default: 8112)
#   SPATIALVLA_UNNORM_KEY        optional per-request unnorm key
#   UNIVLA_ROOT / UNIVLA_LOCAL_*  see policies/univla_local/univla_local_model.py
#   BRIDGE_SUBTASK               carrot | stack | spoon | eggplant | all (only for bridge_widowx_dynisafety.sh;
#                                run_dynisafety sets this automatically for subtasks)
#
# ── Outputs ──────────────────────────────────────────────────────
#   Default --output-layout bench  →  dynamic_SimplerEnv/outputs/{model}/logs/
#   and outputs/{model}/videos/{task}_level{N}/...
#   Use --output-layout legacy + --log-root results  for the old nested paths under
#   results/dynisafety_* required by SimplerEnv-OpenVLA/tools/calc_metrics_evaluation_videos.py.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."   # → dynamic_SimplerEnv/

# ──────────────────────────── defaults ──────────────────────
POLICY_MODEL=""
CKPT_PATH=""
TASK="pick_coke_can"
DYNAMIC_LEVEL_ARG="1"
ACTION_ENSEMBLE_TEMP=-0.8
GPU_ID=0
LOGGING_ROOT="results"
OUTPUT_ROOT="./outputs"
OUTPUT_LAYOUT="bench"
RUN_ID_NOTE=""
BRIDGE_L2_OUTSIDE_TABLE_DIST_ARG=""
BRIDGE_L3_ENTER_HEIGHT_ARG=""
BRIDGE_L3_HAND_SCALE_ARG=""
BRIDGE_L3_START_MODE_ARG=""
BRIDGE_L3_FLIP_SIDE_ARG=0

# ──────────────────────────── argument parsing ──────────────
usage() {
    echo "Usage: $0 --model <name> --ckpt <path> [options]"
    echo ""
    echo "Required:"
    echo "  --model NAME     Registered policy model name"
    echo "  --ckpt  PATH     Checkpoint path or HuggingFace hub ID"
    echo ""
    echo "Optional:"
    echo "  --task   NAME    Task: pick_coke_can | move_near | drawer | bridge |"
    echo "                    bridge_carrot | bridge_stack | bridge_spoon | bridge_eggplant"
    echo "                    (default: pick_coke_can)"
    echo "  --level  N       Dynamic level: 0 | 1 | 2 | 3 | all     (default: 1; all=1,2,3)"
    echo "  --temp   F       Action ensemble temperature               (default: -0.8)"
    echo "  --gpu    N       CUDA device ID                            (default: 0)"
    echo "  --log-root DIR   Legacy only: root for nested results/     (default: results)"
    echo "  --output-root D  Bench layout root (default: ./outputs → dynamic_SimplerEnv/outputs)"
    echo "  --output-layout  bench | legacy  (default: bench; legacy for calc_metrics tool)"
    echo "  --run-id-note S  Optional log filename suffix (LIBERO run_id_note)"
    echo "  --bridge-l2-outside-table-dist F  Bridge Level 2 start distance outside tabletop edge"
    echo "  --bridge-l3-enter-height F        Bridge Level 3 visible hand height above target z"
    echo "  --bridge-l3-hand-scale F          Bridge Level 3 hand scale"
    echo "  --bridge-l3-start-mode MODE       Bridge Level 3 hand start mode: legacy | robot_base_corner"
    echo "  --bridge-l3-flip-side             Bridge Level 3 switch to the opposite lateral start side"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    shift; POLICY_MODEL="$1"; shift ;;
        --ckpt)     shift; CKPT_PATH="$1";    shift ;;
        --task)     shift; TASK="$1";         shift ;;
        --level)    shift; DYNAMIC_LEVEL_ARG="$1"; shift ;;
        --temp)     shift; ACTION_ENSEMBLE_TEMP="$1"; shift ;;
        --gpu)      shift; GPU_ID="$1";       shift ;;
        --log-root) shift; LOGGING_ROOT="$1"; shift ;;
        --output-root) shift; OUTPUT_ROOT="$1"; shift ;;
        --output-layout) shift; OUTPUT_LAYOUT="$1"; shift ;;
        --run-id-note) shift; RUN_ID_NOTE="$1"; shift ;;
        --bridge-l2-outside-table-dist) shift; BRIDGE_L2_OUTSIDE_TABLE_DIST_ARG="$1"; shift ;;
        --bridge-l3-enter-height) shift; BRIDGE_L3_ENTER_HEIGHT_ARG="$1"; shift ;;
        --bridge-l3-hand-scale) shift; BRIDGE_L3_HAND_SCALE_ARG="$1"; shift ;;
        --bridge-l3-start-mode) shift; BRIDGE_L3_START_MODE_ARG="$1"; shift ;;
        --bridge-l3-flip-side) BRIDGE_L3_FLIP_SIDE_ARG=1; shift ;;
        -h|--help)  usage ;;
        *)
            echo "Unknown option: $1"
            usage ;;
    esac
done

if [[ -z "${POLICY_MODEL}" || -z "${CKPT_PATH}" ]]; then
    echo "ERROR: --model and --ckpt are required."
    echo ""
    usage
fi

if [[ "${OUTPUT_LAYOUT}" != "bench" && "${OUTPUT_LAYOUT}" != "legacy" ]]; then
    echo "ERROR: --output-layout must be bench or legacy (got: ${OUTPUT_LAYOUT})"
    exit 1
fi

if [[ "${DYNAMIC_LEVEL_ARG}" == "all" ]]; then
    LEVELS="1 2 3"
else
    LEVELS="${DYNAMIC_LEVEL_ARG}"
fi

MODEL_BASENAME=$(basename "${CKPT_PATH}")

export DYNISAFETY_OUTPUT_ROOT="${OUTPUT_ROOT}"
export DYNISAFETY_OUTPUT_LAYOUT="${OUTPUT_LAYOUT}"
export DYNISAFETY_BENCH_TASK="${TASK}"
if [[ -n "${BRIDGE_L2_OUTSIDE_TABLE_DIST_ARG}" ]]; then
    export BRIDGE_L2_OUTSIDE_TABLE_DIST="${BRIDGE_L2_OUTSIDE_TABLE_DIST_ARG}"
fi
if [[ -n "${BRIDGE_L3_ENTER_HEIGHT_ARG}" ]]; then
    export BRIDGE_L3_ENTER_HEIGHT="${BRIDGE_L3_ENTER_HEIGHT_ARG}"
fi
if [[ -n "${BRIDGE_L3_HAND_SCALE_ARG}" ]]; then
    export BRIDGE_L3_HAND_SCALE="${BRIDGE_L3_HAND_SCALE_ARG}"
fi
if [[ -n "${BRIDGE_L3_START_MODE_ARG}" ]]; then
    export BRIDGE_L3_START_MODE="${BRIDGE_L3_START_MODE_ARG}"
fi
if [[ "${BRIDGE_L3_FLIP_SIDE_ARG}" == "1" ]]; then
    export BRIDGE_L3_FLIP_SIDE=1
fi

for level in ${LEVELS}; do
    level_names=("baseline" "sudden_drop" "sliding_obstacle" "shared_reaching")
    level_name="${level_names[$level]}"

    if [[ -n "${RUN_ID_NOTE}" ]]; then
        export DYNISAFETY_RUN_ID_NOTE="${RUN_ID_NOTE}-L${level}"
    else
        export DYNISAFETY_RUN_ID_NOTE=""
    fi

    LOGGING_DIR="${LOGGING_ROOT}/dynisafety_${POLICY_MODEL}_${MODEL_BASENAME}_${TASK}_level${level}"
    mkdir -p "${LOGGING_DIR}"

    echo "============================================================"
    echo "DyniSafety Level ${level} — ${level_name}"
    echo "  Model      : ${POLICY_MODEL}"
    echo "  Checkpoint : ${CKPT_PATH}"
    echo "  Task       : ${TASK}"
    echo "  Output layout: ${OUTPUT_LAYOUT}"
    if [[ "${OUTPUT_LAYOUT}" == "bench" ]]; then
        echo "  Bench root : ${OUTPUT_ROOT}  (videos/logs under outputs/${POLICY_MODEL}/)"
    fi
    echo "  Legacy log dir: ${LOGGING_DIR}  (--logging-dir; used fully in legacy layout)"
    echo "============================================================"

    case "${TASK}" in
        pick_coke_can)
            bash scripts/pick_coke_can_dynisafety.sh \
                "${CKPT_PATH}" "${POLICY_MODEL}" "${ACTION_ENSEMBLE_TEMP}" \
                "${LOGGING_DIR}" "${GPU_ID}" "${level}"
            ;;
        move_near)
            bash scripts/move_near_dynisafety.sh \
                "${CKPT_PATH}" "${POLICY_MODEL}" "${ACTION_ENSEMBLE_TEMP}" \
                "${LOGGING_DIR}" "${GPU_ID}" "${level}"
            ;;
        drawer)
            bash scripts/drawer_dynisafety.sh \
                "${CKPT_PATH}" "${POLICY_MODEL}" "${ACTION_ENSEMBLE_TEMP}" \
                "${LOGGING_DIR}" "${GPU_ID}" "${level}"
            ;;
        bridge)
            BRIDGE_SUBTASK=all bash scripts/bridge_widowx_dynisafety.sh \
                "${CKPT_PATH}" "${POLICY_MODEL}" "${ACTION_ENSEMBLE_TEMP}" \
                "${LOGGING_DIR}" "${GPU_ID}" "${level}"
            ;;
        bridge_carrot)
            BRIDGE_SUBTASK=carrot bash scripts/bridge_widowx_dynisafety.sh \
                "${CKPT_PATH}" "${POLICY_MODEL}" "${ACTION_ENSEMBLE_TEMP}" \
                "${LOGGING_DIR}" "${GPU_ID}" "${level}"
            ;;
        bridge_stack)
            BRIDGE_SUBTASK=stack bash scripts/bridge_widowx_dynisafety.sh \
                "${CKPT_PATH}" "${POLICY_MODEL}" "${ACTION_ENSEMBLE_TEMP}" \
                "${LOGGING_DIR}" "${GPU_ID}" "${level}"
            ;;
        bridge_spoon)
            BRIDGE_SUBTASK=spoon bash scripts/bridge_widowx_dynisafety.sh \
                "${CKPT_PATH}" "${POLICY_MODEL}" "${ACTION_ENSEMBLE_TEMP}" \
                "${LOGGING_DIR}" "${GPU_ID}" "${level}"
            ;;
        bridge_eggplant)
            BRIDGE_SUBTASK=eggplant bash scripts/bridge_widowx_dynisafety.sh \
                "${CKPT_PATH}" "${POLICY_MODEL}" "${ACTION_ENSEMBLE_TEMP}" \
                "${LOGGING_DIR}" "${GPU_ID}" "${level}"
            ;;
        *)
            echo "ERROR: Unknown task '${TASK}'. Choose: pick_coke_can | move_near | drawer | bridge | bridge_carrot | bridge_stack | bridge_spoon | bridge_eggplant"
            exit 1
            ;;
    esac

    if [[ "${OUTPUT_LAYOUT}" == "legacy" ]]; then
        echo "  Calculating metrics (legacy layout)..."
        SIMPLERENV_PATH="${SIMPLERENV_ROOT:-${SCRIPT_DIR}/../SimplerEnv-OpenVLA}"
        python "${SIMPLERENV_PATH}/tools/calc_metrics_evaluation_videos.py" \
            --log-dir-root "${LOGGING_DIR}" \
            >>"${LOGGING_DIR}/total.metrics"
    else
        echo "  Skipping calc_metrics_evaluation_videos.py (bench layout; use --output-layout legacy to enable)."
    fi

    echo "Level ${level} (${level_name}) — done."
    echo ""
done

echo "============================================================"
echo "ALL REQUESTED LEVELS COMPLETE."
echo "============================================================"
