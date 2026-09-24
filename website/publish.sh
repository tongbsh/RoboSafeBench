#!/usr/bin/env bash
# Publish local website/ to the anonymous GitHub Pages repo.
# Usage (from repo root or this directory):
#   bash website/publish.sh
#   ./publish.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REMOTE="${ROBOSAFE_PAGES_REMOTE:-git@github.com:bshtong-demo/RoboSafeBench.git}"
KEY="${ROBOSAFE_PAGES_KEY:-$ROOT/.deploy/id_ed25519}"
CLONE="${ROBOSAFE_PAGES_CLONE:-$ROOT/.deploy/pages-clone}"
SITE_URL="${ROBOSAFE_PAGES_URL:-https://bshtong-demo.github.io/RoboSafeBench/}"

if [[ ! -f "$KEY" ]]; then
  echo "missing deploy key: $KEY" >&2
  echo "copy the pages deploy private key there, or set ROBOSAFE_PAGES_KEY" >&2
  exit 1
fi
chmod 600 "$KEY"

export GIT_SSH_COMMAND="ssh -i $(printf %q "$KEY") -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

if [[ ! -d "$CLONE/.git" ]]; then
  mkdir -p "$(dirname "$CLONE")"
  git clone "$REMOTE" "$CLONE"
else
  git -C "$CLONE" remote set-url origin "$REMOTE"
  git -C "$CLONE" fetch origin
  git -C "$CLONE" checkout --force main
  git -C "$CLONE" reset --hard origin/main
fi

rsync -a --delete \
  --exclude '.git/' \
  --exclude '.github/' \
  --exclude '.deploy/' \
  --exclude 'publish.sh' \
  --exclude 'serve.sh' \
  --exclude 'README.md' \
  "$ROOT/" "$CLONE/"

if [[ ! -f "$CLONE/.nojekyll" ]]; then
  : > "$CLONE/.nojekyll"
fi

if git -C "$CLONE" diff --quiet && git -C "$CLONE" diff --cached --quiet && [[ -z "$(git -C "$CLONE" ls-files --others --exclude-standard)" ]]; then
  echo "no website changes to publish"
  echo "$SITE_URL"
  exit 0
fi

git -C "$CLONE" add -A
git -C "$CLONE" \
  -c user.name="RoboSafeBench" \
  -c user.email="robosafebench@users.noreply.github.com" \
  commit -m "Update project page."
git -C "$CLONE" push origin HEAD:main
echo "published $SITE_URL"
echo "GitHub Pages usually refreshes in under a minute."
