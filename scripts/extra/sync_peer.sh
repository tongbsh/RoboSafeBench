#!/usr/bin/env bash
# 两台 AutoDL 克隆机之间同步：代码单向推/拉，日志双向合并（不删除）。
#
# 前置：对端控制台复制 SSH（host + 端口），并把本机公钥写入对端
#   ~/.ssh/authorized_keys。本机公钥：
#     cat ~/.ssh/id_ed25519.pub
#
#   PEER=root@connect.westb.seetacloud.com PEER_PORT=12345 \
#     bash scripts/extra/sync_peer.sh check
#   PEER=... PEER_PORT=... bash scripts/extra/sync_peer.sh push-code
#   PEER=... PEER_PORT=... bash scripts/extra/sync_peer.sh merge-logs
#   PEER=... PEER_PORT=... bash scripts/extra/sync_peer.sh both
#
# 两台不要对同一条未完成 run 同时 --resume；同路径文件按较新 mtime 整份覆盖，
# 不会做 jsonl 行级合并。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PEER="${PEER:-}"
PEER_PORT="${PEER_PORT:-22}"
PEER_ROOT="${PEER_ROOT:-$ROOT}"
IDENTITY="${IDENTITY:-$HOME/.ssh/id_ed25519}"
DRY_RUN=0
WITH_VIDEOS=0
WITH_CAPTURE=0
CODE_DELETE=0
MODE=""

CODE_DIRS=(
  src
  configs
  scripts
  tests
  tools
  docs
  website
  templates
  requirements
  environment
  docker
  examples
  patches
)
CODE_FILES=(
  Makefile
  pyproject.toml
  README.md
  .gitignore
  .gitmodules
  CITATION.cff
  THIRD_PARTY_NOTICES.md
  LICENSE
)
LOG_DIRS=(
  out/eval
  out/logs
  out/_sweep
  out/reports
  out/analysis
  out/setup
)

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

子命令：
  check         测 SSH，并列出将同步的本地路径
  push-code     本机代码 → 对端（按内容覆盖；默认不删对端多出来的文件）
  pull-code     对端代码 → 本机
  push-logs     本机日志 → 对端（不删除，较新不覆盖）
  pull-logs     对端日志 → 本机
  merge-logs    两边各推一次（先本机→对端，再对端→本机）
  both          push-code + merge-logs

常用环境变量 / 标志：
  PEER / --peer         对端，如 root@connect.xxx.seetacloud.com
  PEER_PORT / --port    SSH 端口（AutoDL 控制台「自定义服务」里的那个）
  PEER_ROOT             对端仓库绝对路径，默认与本机相同
  IDENTITY / -i         SSH 私钥，默认 ~/.ssh/id_ed25519
  --dry-run             只打印 rsync 将做什么
  --with-videos         日志同步带上 mp4 / videos/
  --with-capture        同时同步 out/capture（体积大）
  --code-delete         推代码时删除对端同目录里本机没有的文件
EOF
}

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --peer) PEER="$2"; shift 2 ;;
    --peer=*) PEER="${1#*=}"; shift ;;
    --port) PEER_PORT="$2"; shift 2 ;;
    --port=*) PEER_PORT="${1#*=}"; shift ;;
    --peer-root) PEER_ROOT="$2"; shift 2 ;;
    -i|--identity) IDENTITY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --with-videos) WITH_VIDEOS=1; shift ;;
    --with-capture) WITH_CAPTURE=1; shift ;;
    --code-delete) CODE_DELETE=1; shift ;;
    check|push-code|pull-code|push-logs|pull-logs|merge-logs|both)
      MODE="$1"; shift ;;
    *) die "unknown arg: $1（见 --help）" ;;
  esac
done

[[ -n "$MODE" ]] || { usage; exit 1; }
[[ -n "$PEER" ]] || die "未设置 PEER。例：PEER=root@connect.xxx.seetacloud.com PEER_PORT=xxxxx $0 $MODE"

ssh_base=(ssh -p "$PEER_PORT" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
if [[ -f "$IDENTITY" ]]; then
  ssh_base+=(-i "$IDENTITY")
fi

rsync_ssh() {
  local quoted=""
  local a
  for a in "${ssh_base[@]}"; do
    quoted+="$(printf '%q ' "$a")"
  done
  printf '%s' "${quoted% }"
}

rsync_common=(-aH --partial --human-readable --info=stats1,flist0)
if [[ "$DRY_RUN" -eq 1 ]]; then
  rsync_common+=(-n)
fi

code_excludes=(
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.mypy_cache/'
  --exclude '.pytest_cache/'
  --exclude '.ruff_cache/'
  --exclude '.deploy/'
  --exclude '*.mp4'
  --exclude '*.webm'
)

log_excludes=(
  --exclude '*.pid'
  --exclude '*.tmp'
  --exclude '.lock'
  --exclude '__pycache__/'
)
if [[ "$WITH_VIDEOS" -eq 0 ]]; then
  log_excludes+=(--exclude 'videos/' --exclude '*.mp4' --exclude '*.webm' --exclude '*.gif')
fi

log_dirs=("${LOG_DIRS[@]}")
if [[ "$WITH_CAPTURE" -eq 1 ]]; then
  log_dirs+=(out/capture)
fi

run_rsync() {
  local src="$1" dest="$2"
  shift 2
  echo "==> rsync $src  ->  $dest"
  rsync "${rsync_common[@]}" -e "$(rsync_ssh)" "$@" "$src" "$dest"
}

push_code() {
  local dest="${PEER}:${PEER_ROOT}/"
  local dir
  for dir in "${CODE_DIRS[@]}"; do
    if [[ -d "$ROOT/$dir" ]]; then
      local extra=(--checksum)
      if [[ "$CODE_DELETE" -eq 1 ]]; then
        extra+=(--delete)
      fi
      run_rsync "$ROOT/$dir/" "${dest}${dir}/" "${code_excludes[@]}" "${extra[@]}"
    fi
  done
  local file
  for file in "${CODE_FILES[@]}"; do
    if [[ -e "$ROOT/$file" ]]; then
      run_rsync "$ROOT/$file" "${dest}${file}" --checksum
    fi
  done
}

pull_code() {
  local src="${PEER}:${PEER_ROOT}"
  local dir
  for dir in "${CODE_DIRS[@]}"; do
    mkdir -p "$ROOT/$dir"
    local extra=()
    if [[ "$CODE_DELETE" -eq 1 ]]; then
      extra+=(--delete)
    fi
    run_rsync "${src}/${dir}/" "$ROOT/$dir/" "${code_excludes[@]}" "${extra[@]}"
  done
  local file
  for file in "${CODE_FILES[@]}"; do
    run_rsync "${src}/${file}" "$ROOT/$file" || true
  done
}

# --update：对端较新的文件不覆盖（正在写的 run 不会被旧副本冲掉）
push_logs() {
  local dest="${PEER}:${PEER_ROOT}/"
  local dir
  for dir in "${log_dirs[@]}"; do
    if [[ -d "$ROOT/$dir" ]]; then
      "${ssh_base[@]}" "$PEER" "mkdir -p $(printf %q "${PEER_ROOT}/${dir}")"
      run_rsync "$ROOT/$dir/" "${dest}${dir}/" --update "${log_excludes[@]}"
    fi
  done
}

pull_logs() {
  local src="${PEER}:${PEER_ROOT}"
  local dir
  for dir in "${log_dirs[@]}"; do
    mkdir -p "$ROOT/$dir"
    run_rsync "${src}/${dir}/" "$ROOT/$dir/" --update "${log_excludes[@]}" || true
  done
}

do_check() {
  echo "local  $ROOT"
  echo "peer   $PEER  port=$PEER_PORT"
  echo "remote $PEER_ROOT"
  echo "key    $IDENTITY"
  echo "==> ssh"
  "${ssh_base[@]}" "$PEER" "hostname; pwd; test -d $(printf %q "$PEER_ROOT") && echo \"repo ok: $PEER_ROOT\" || echo \"MISSING repo: $PEER_ROOT\""
  echo "==> local code dirs"
  local dir
  for dir in "${CODE_DIRS[@]}"; do
    [[ -d "$ROOT/$dir" ]] && du -sh "$ROOT/$dir"
  done
  echo "==> local log dirs"
  for dir in "${log_dirs[@]}"; do
    [[ -d "$ROOT/$dir" ]] && du -sh "$ROOT/$dir"
  done
}

case "$MODE" in
  check) do_check ;;
  push-code) push_code ;;
  pull-code) pull_code ;;
  push-logs) push_logs ;;
  pull-logs) pull_logs ;;
  merge-logs) push_logs; pull_logs ;;
  both) push_code; push_logs; pull_logs ;;
esac

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "done ($MODE)  [dry-run]"
else
  echo "done ($MODE)"
fi
