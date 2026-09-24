#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8765}"
echo "RoboSafeBench site → http://127.0.0.1:${PORT}"
exec python -m http.server "$PORT"
