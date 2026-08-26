#!/usr/bin/env bash
# X-VLA 流程走查：dynamic LIBERO + RoboTwin + SimplerEnv × 全部 paper 任务 × L1–L3。
# 缩小版 scripts/eval/01_eval.sh：只跑 X-VLA，默认每格 20 条轨迹、录像、
# L2/L3 diversity，以及 LIBERO obstacle pack。
#
#   bash scripts/debug/xvla_walkthrough/00_run.sh --dry-run
#   bash scripts/debug/xvla_walkthrough/00_run.sh
#   bash scripts/debug/xvla_walkthrough/00_run.sh --backend libero
#   bash scripts/debug/xvla_walkthrough/00_run.sh --backend robotwin,simplerenv
#   bash scripts/debug/xvla_walkthrough/00_run.sh --episodes 3
#   bash scripts/debug/xvla_walkthrough/00_run.sh --full          # 论文条数：L 50 / RT 20 / SE 50
#   bash scripts/debug/xvla_walkthrough/00_run.sh --resume
#   DYNI_GPUS=0 bash scripts/debug/xvla_walkthrough/00_run.sh
#   DYNI_SAVE_VIDEO=0 bash scripts/debug/xvla_walkthrough/00_run.sh --no-video
#   bash scripts/debug/xvla_walkthrough/00_run.sh --no-diversity --no-obstacle-pack
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/env.sh"

SUITE_SRC="$ROOT/configs/suites/debug_xvla_dynamic.yaml"
[[ -f "$SUITE_SRC" ]] || die "missing sweep suite: $SUITE_SRC"

export DYNI_SAVE_VIDEO="${DYNI_SAVE_VIDEO:-1}"
export DYNI_DIVERSITY="${DYNI_DIVERSITY:-1}"
export DYNI_OBSTACLE_PACK="${DYNI_OBSTACLE_PACK:-1}"
# One loaded X-VLA shared by both eval slots (set 2 to pin a copy per eval).
export DYNI_SERVERS_PER_GPU="${DYNI_SERVERS_PER_GPU:-1}"

BACKENDS=""
LEVELS=""
TASKS=""
EPISODES=""
FULL=0
NO_VIDEO=0
SKIP_CHECK=0
SKIP_FREE=0
EXTRA=()

_csv_to_pipe() {
  local raw="$1"
  raw="${raw// /}"
  raw="${raw//,/|}"
  printf '%s' "$raw"
}

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

标志:
  --backend     libero|robotwin|simplerenv（可逗号）
  --level       1|2|3（可逗号；默认 suite 里的 1,2,3）
  --task        任务 id（可逗号）
  --episodes N  每格轨迹数（三端相同；覆盖 --full）
  --full        论文条数：LIBERO/SimplerEnv 50，RoboTwin 20
  --no-video          不录像（默认录像）
  --no-diversity      关闭 L2 spawn / L3 reach diversity（默认开）
  --no-obstacle-pack  关闭 LIBERO GSO pack（默认开；仅 libero 生效）
  --skip-check        跑完不自动 01_check
  --skip-free         开跑前不清理 8010/8011 与遗留 xvla_* screen
  --dry-run / --resume / --retry-failed
  其余参数原样传给 dynisafety sweep

默认 20 条轨迹/格（117 job × 20 ep），单卡一份权重 + 两个 eval。LIBERO/SimplerEnv 要论文 50 条用 --full。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --backend)
      BACKENDS="$2"
      shift 2
      ;;
    --backend=*)
      BACKENDS="${1#*=}"
      shift
      ;;
    --level)
      LEVELS="$2"
      shift 2
      ;;
    --level=*)
      LEVELS="${1#*=}"
      shift
      ;;
    --task)
      TASKS="$2"
      shift 2
      ;;
    --task=*)
      TASKS="${1#*=}"
      shift
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --episodes=*)
      EPISODES="${1#*=}"
      shift
      ;;
    --full)
      FULL=1
      shift
      ;;
    --no-video)
      NO_VIDEO=1
      shift
      ;;
    --no-diversity)
      export DYNI_DIVERSITY=0
      shift
      ;;
    --no-obstacle-pack)
      export DYNI_OBSTACLE_PACK=0
      shift
      ;;
    --skip-check)
      SKIP_CHECK=1
      shift
      ;;
    --skip-free)
      SKIP_FREE=1
      shift
      ;;
    --dry-run|--resume|--retry-failed)
      EXTRA+=("$1")
      shift
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

if [[ "$NO_VIDEO" -eq 1 ]]; then
  export DYNI_SAVE_VIDEO=0
fi

_free_gpu_and_ports() {
  local port pid
  info "清理遗留 X-VLA serve/eval，释放 8010/8011 与显存"
  for name in xvla_rt_ep5_serve xvla_rt_ep5_eval xvla_se_ep5_serve xvla_se_ep5_eval \
              xvla_official_robotwin xvla_official_simplerenv; do
    screen -S "$name" -X quit >/dev/null 2>&1 || true
  done
  for port in 8010 8011; do
    if command -v fuser >/dev/null 2>&1; then
      fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    fi
    if command -v ss >/dev/null 2>&1; then
      while read -r pid; do
        [[ -n "$pid" ]] || continue
        kill "$pid" >/dev/null 2>&1 || true
      done < <(ss -lptn "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p')
    fi
  done
  sleep 2
}

if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
elif [[ -f /etc/vulkan/icd.d/my_nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/my_nvidia_icd.json}"
fi
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -n "$BACKENDS" ]]; then
  case ",${BACKENDS}," in
    *,robocasa,*) die "本走查不含 RoboCasa；backend 只允许 libero|robotwin|simplerenv" ;;
  esac
fi

dyni_activate "$DYNI_TOOL_ENV"

SUITE_NAME="debug_xvla_dynamic"
EP_DEFAULT=20
EP_ROBOTWIN=20
if [[ -n "$EPISODES" ]]; then
  [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "--episodes 必须是正整数"
  EP_DEFAULT="$EPISODES"
  EP_ROBOTWIN="$EPISODES"
  if [[ "$FULL" -eq 1 ]]; then
    warn "--episodes=$EPISODES 覆盖 --full 的论文条数"
  fi
elif [[ "$FULL" -eq 1 ]]; then
  SUITE_NAME="debug_xvla_dynamic_full"
  EP_DEFAULT=50
  EP_ROBOTWIN=20
fi

SUITE_ARG="$SUITE_NAME"
USED_YAML="$(dyni_paths sweep-root "$SUITE_NAME")/suite.used.yaml"
mkdir -p "$(dirname "$USED_YAML")"
if [[ "$SUITE_NAME" != "debug_xvla_dynamic" || "$EP_DEFAULT" != "20" ]]; then
  SUITE_ARG="$USED_YAML"
  python - "$SUITE_SRC" "$USED_YAML" "$SUITE_NAME" "$EP_DEFAULT" "$EP_ROBOTWIN" <<'PY'
import pathlib
import sys

import yaml

src, dest, name, ep, ep_rt = sys.argv[1:6]
data = yaml.safe_load(pathlib.Path(src).read_text(encoding="utf-8"))
data["name"] = name
data["episodes_per_task"] = int(ep)
for block in data["matrix"]:
    if block.get("backend") == "robotwin":
        if int(ep_rt) != int(ep):
            block["episodes_per_task"] = int(ep_rt)
        else:
            block.pop("episodes_per_task", None)
path = pathlib.Path(dest)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY
else
  cp -f "$SUITE_SRC" "$USED_YAML"
fi

ONLY_PARTS=()
[[ -n "$BACKENDS" ]] && ONLY_PARTS+=("backend=$(_csv_to_pipe "$BACKENDS")")
[[ -n "$LEVELS" ]] && ONLY_PARTS+=("level=$(_csv_to_pipe "$LEVELS")")
[[ -n "$TASKS" ]] && ONLY_PARTS+=("task=$(_csv_to_pipe "$TASKS")")
ONLY=""
if [[ ${#ONLY_PARTS[@]} -gt 0 ]]; then
  ONLY="$(IFS=','; echo "${ONLY_PARTS[*]}")"
fi

export DYNI_SUITE="$SUITE_ARG"
LOG_DIR="$(dyni_artifacts_dir debug xvla_walkthrough)"
dyni_tee_logs "$LOG_DIR/sweep.screen.log"
info "process log: $LOG_DIR/sweep.screen.log"
info "X-VLA walkthrough  suite=$SUITE_NAME  episodes=L/SE:${EP_DEFAULT} RT:${EP_ROBOTWIN}"
info "save_video=${DYNI_SAVE_VIDEO}  diversity=${DYNI_DIVERSITY}  obstacle_pack=${DYNI_OBSTACLE_PACK}  servers_per_gpu=${DYNI_SERVERS_PER_GPU}${ONLY:+  only=$ONLY}"
warn "X-VLA 是服务型：sweep 会按 backend 切 checkpoint（LIBERO=xvla，RT/SE=xvla_official）并自己起 serve。"

SWEEP_ARGS=()
[[ -n "$ONLY" ]] && SWEEP_ARGS+=(--only "$ONLY")
SWEEP_ARGS+=("${EXTRA[@]}")

DRY=0
for arg in "${EXTRA[@]}"; do
  [[ "$arg" == "--dry-run" ]] && DRY=1
done

if [[ "$DRY" -eq 0 && "$SKIP_FREE" -eq 0 ]]; then
  _free_gpu_and_ports
fi

set +e
bash "$ROOT/scripts/eval/sweep.sh" "${SWEEP_ARGS[@]}"
sweep_rc=$?
set -e

if [[ "$DRY" -eq 0 && "$SKIP_CHECK" -eq 0 ]]; then
  info "sweep exit=$sweep_rc；接着扫指标 / 动态痕迹 / 视频"
  bash "$SCRIPT_DIR/01_check.sh" --suite "$USED_YAML" || warn "01_check 发现问题，看 $(dyni_artifacts_dir debug xvla_walkthrough)/"
fi

dyni_print_output_roots debug xvla_walkthrough
ok "失败: $(dyni_results_root)/_sweep/${SUITE_NAME}/failures.jsonl"
ok "检查: bash scripts/debug/xvla_walkthrough/01_check.sh --suite $USED_YAML"
exit "$sweep_rc"
