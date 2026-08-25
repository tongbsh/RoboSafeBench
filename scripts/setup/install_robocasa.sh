#!/usr/bin/env bash
# Install RoboCasa + robosuite git trees into the dynisafety-robocasa conda env.
#
# Not a git submodule (this window). Prefer existing third_party clones; if a
# clone is incomplete, re-fetch via ghfast.top then upstream GitHub.
#
# Pins (full SHA; keep in sync with docs/user/third_party.md):
#   ROBOSUITE_SHA — git master, robosuite >= 1.5 (do NOT use PyPI 1.4)
#   ROBOCASA_SHA  — robocasa/robocasa main
#
# Kitchen assets default to the lightweight set:
#   tex tex_generative fixtures_lw objs_lw
# Full ~10GB registry only when DYNISAFETY_ROBOCASA_FULL_ASSETS=1.
#
# Does not wire backends / CLI / disturbances / action spaces.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Pin SHAs (authoritative for this window; also recorded in docs/user/third_party.md).
ROBOSUITE_SHA="5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
ROBOCASA_SHA="921c9a5736a8d0ea5589657898aadcfa55a6a195"

ROBOSUITE_TREE="${ROOT}/third_party/robosuite"
ROBOCASA_TREE="${ROOT}/third_party/robocasa"

ROBOSUITE_URL_GHFAST="${DYNISAFETY_ROBOSUITE_URL:-https://ghfast.top/https://github.com/ARISE-Initiative/robosuite.git}"
ROBOSUITE_URL_UPSTREAM="https://github.com/ARISE-Initiative/robosuite.git"
ROBOCASA_URL_GHFAST="${DYNISAFETY_ROBOCASA_URL:-https://ghfast.top/https://github.com/robocasa/robocasa.git}"
ROBOCASA_URL_UPSTREAM="https://github.com/robocasa/robocasa.git"

cd "$ROOT"
mkdir -p "${ROOT}/third_party"

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  case "${CONDA_DEFAULT_ENV}" in
    dynisafety-libero|dynisafety-simplerenv|dynisafety-robotwin)
      echo "error: refuse to install RoboCasa into ${CONDA_DEFAULT_ENV}" >&2
      echo "       use conda env dynisafety-robocasa (python 3.11, numpy 2.2.5)" >&2
      exit 1
      ;;
  esac
fi

pyver="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$pyver" != "3.11" ]]; then
  echo "error: dynisafety-robocasa requires Python 3.11 (got ${pyver})." >&2
  echo "       conda activate dynisafety-robocasa first." >&2
  exit 1
fi

ensure_git_tree() {
  local dest="$1"
  local sha="$2"
  local marker="$3"
  local name="$4"
  shift 4
  local urls=("$@")

  if [[ -d "${dest}/.git" ]]; then
    echo "==> ${name}: existing git tree at ${dest}"
    if [[ ! -f "${dest}/${marker}" ]]; then
      echo "==> ${name}: incomplete worktree; fetch + hard reset to pin"
      git -C "$dest" fetch --all --tags || true
      git -C "$dest" checkout -f "$sha"
      git -C "$dest" reset --hard "$sha"
    fi
  else
    if [[ -e "$dest" ]]; then
      echo "==> ${name}: removing incomplete non-git tree at ${dest}"
      rm -rf "$dest"
    fi
    local ok=0 url
    for url in "${urls[@]}"; do
      echo "==> ${name}: cloning ${url}"
      if git clone "$url" "$dest"; then
        ok=1
        break
      fi
      echo "warn: ${name} clone failed from ${url}" >&2
      rm -rf "$dest"
    done
    if [[ "$ok" -ne 1 ]]; then
      echo "error: failed to clone ${name} into ${dest}" >&2
      exit 1
    fi
  fi

  git -C "$dest" fetch --all --tags || true
  local head
  head="$(git -C "$dest" rev-parse HEAD)"
  if [[ "$head" != "$sha" ]]; then
    echo "==> ${name}: checkout ${sha} (was ${head})"
    git -C "$dest" checkout "$sha"
  else
    echo "==> ${name}: already at pin ${sha}"
  fi
  head="$(git -C "$dest" rev-parse HEAD)"
  if [[ "$head" != "$sha" ]]; then
    echo "error: ${name} HEAD ${head} != pin ${sha}" >&2
    exit 1
  fi
  if [[ ! -f "${dest}/${marker}" ]]; then
    echo "error: ${name} missing ${marker} after pin checkout (incomplete clone)" >&2
    exit 1
  fi
  echo "==> ${name} pin ${head}"
}

ensure_git_tree \
  "$ROBOSUITE_TREE" "$ROBOSUITE_SHA" "setup.py" "robosuite" \
  "$ROBOSUITE_URL_GHFAST" "$ROBOSUITE_URL_UPSTREAM"

ensure_git_tree \
  "$ROBOCASA_TREE" "$ROBOCASA_SHA" "setup.py" "robocasa" \
  "$ROBOCASA_URL_GHFAST" "$ROBOCASA_URL_UPSTREAM"

echo "==> pip install -e ${ROBOSUITE_TREE}  (git master, not PyPI 1.4)"
pip install -e "$ROBOSUITE_TREE"

echo "==> pip install -e ${ROBOCASA_TREE}"
pip install -e "$ROBOCASA_TREE"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

MACROS_PRIVATE="${ROBOCASA_TREE}/robocasa/macros_private.py"
if [[ -f "$MACROS_PRIVATE" ]]; then
  echo "==> macros_private.py already present; skip setup_macros"
else
  echo "==> python -m robocasa.scripts.setup_macros"
  python -m robocasa.scripts.setup_macros
fi

echo "==> download kitchen assets (lightweight unless DYNISAFETY_ROBOCASA_FULL_ASSETS=1)"
# Official Box shared/static URLs currently 404. Try Box briefly, then Hugging Face
# mirrors. On AutoDL, `source /etc/network_turbo` before this script so wget can
# reach huggingface.co; do NOT leave the proxy on for pip (it slows PyPI).
if [[ -f /etc/network_turbo && -z "${https_proxy:-}" ]]; then
  echo "note: AutoDL HF accel available: source /etc/network_turbo  (then re-run asset step)"
fi
python - <<'PY'
import os
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from robocasa.scripts.download_kitchen_assets import DOWNLOAD_ASSET_REGISTRY

# Hugging Face mirrors of the official Box packs (CC BY 4.0). Used when Box 404s.
HF_MIRRORS = {
    "tex": ("Livfour/robocasa-assets", "textures.zip", "models/assets"),
    "tex_generative": ("Livfour/robocasa-assets", "generative_textures.zip", "models/assets"),
    "fixtures_lw": (
        "IIFAN/robocasa365-lightwheel-fixtures",
        "fixtures_lightwheel.zip",
        "models/assets",
    ),
    "objs_lw": (
        "IIFAN/robocasa365-lightwheel-objects",
        "objects_lightwheel.zip",
        "models/assets/objects",
    ),
    "objs_objaverse": ("Livfour/robocasa-assets", "objaverse.zip", "models/assets/objects"),
}

# Direct resolve URLs (wget/curl). huggingface_hub can stall on hf-mirror 429s.
HF_RESOLVE = {
    "tex": "https://hf-mirror.com/datasets/Livfour/robocasa-assets/resolve/main/textures.zip",
    "tex_generative": "https://hf-mirror.com/datasets/Livfour/robocasa-assets/resolve/main/generative_textures.zip",
    "fixtures_lw": "https://hf-mirror.com/datasets/IIFAN/robocasa365-lightwheel-fixtures/resolve/main/fixtures_lightwheel.zip",
    "objs_lw": "https://hf-mirror.com/datasets/IIFAN/robocasa365-lightwheel-objects/resolve/main/objects_lightwheel.zip",
    "objs_objaverse": "https://hf-mirror.com/datasets/Livfour/robocasa-assets/resolve/main/objaverse.zip",
}


def _asset_ready(name, folder):
    if not os.path.isdir(folder):
        return False
    if name == "fixtures_lw":
        return os.path.isdir(os.path.join(folder, "fridges"))
    if name == "objs_lw":
        return any(Path(folder).rglob("*.obj")) or any(Path(folder).rglob("*.stl"))
    if name in {"tex", "tex_generative"}:
        return any(
            p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            for p in Path(folder).rglob("*")
            if p.is_file()
        )
    return bool(os.listdir(folder))


def _download_box(url, dest, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "dynisafety-robocasa/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    if os.path.getsize(dest) < 1024:
        raise RuntimeError("Box download too small: %s" % dest)


def _wget_download(url, dest):
    import subprocess

    print("  wget -c %s -> %s" % (url, dest), flush=True)
    subprocess.check_call(
        [
            "wget",
            "-c",
            "--tries=0",
            "--retry-connrefused",
            "--progress=dot:giga",
            "-O",
            dest,
            url,
        ]
    )
    if not os.path.isfile(dest) or os.path.getsize(dest) < 1024:
        raise RuntimeError("wget produced empty file: %s" % dest)
    return dest


def _hf_download(repo, filename):
    from huggingface_hub import hf_hub_download

    endpoints = []
    current = os.environ.get("HF_ENDPOINT")
    if current:
        endpoints.append(current)
    endpoints.extend(["https://huggingface.co", "https://hf-mirror.com"])
    seen = set()
    last_err = None
    for ep in endpoints:
        if ep in seen:
            continue
        seen.add(ep)
        os.environ["HF_ENDPOINT"] = ep
        print("  HF_ENDPOINT=%s repo=%s file=%s" % (ep, repo, filename), flush=True)
        try:
            return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
        except Exception as exc:
            last_err = exc
            print("  HF failed (%s): %s" % (ep, exc), flush=True)
    raise RuntimeError("HF download failed for %s/%s: %s" % (repo, filename, last_err))


def _extract(zip_path, extract_parent):
    Path(extract_parent).mkdir(parents=True, exist_ok=True)
    print("  extracting %s -> %s" % (zip_path, extract_parent), flush=True)
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(path=extract_parent)


full = os.environ.get("DYNISAFETY_ROBOCASA_FULL_ASSETS", "0") == "1"
if full:
    names = list(DOWNLOAD_ASSET_REGISTRY)
    print("DYNISAFETY_ROBOCASA_FULL_ASSETS=1: full kitchen asset registry", flush=True)
else:
    names = ["tex", "tex_generative", "fixtures_lw", "objs_lw"]
    print("lightweight kitchen assets:", ", ".join(names), flush=True)

import robocasa

root = Path(robocasa.__path__[0])
failed = []
sources = []

for name in names:
    if name not in DOWNLOAD_ASSET_REGISTRY:
        failed.append(name)
        continue
    cfg = DOWNLOAD_ASSET_REGISTRY[name]
    folder = cfg["folder"]
    if _asset_ready(name, folder):
        print("skip existing %s: %s" % (name, folder), flush=True)
        sources.append("%s=already-present" % name)
        continue

    print("==> %s" % name, flush=True)
    got = False
    tmp_zip = folder.rstrip("/") + ".zip"
    Path(os.path.dirname(tmp_zip)).mkdir(parents=True, exist_ok=True)

    try:
        print("  trying Box: %s" % cfg["url"], flush=True)
        _download_box(cfg["url"], tmp_zip)
        _extract(tmp_zip, os.path.dirname(folder))
        os.remove(tmp_zip)
        if _asset_ready(name, folder):
            got = True
            sources.append("%s=box" % name)
    except Exception as exc:
        print("  Box failed: %s" % exc, flush=True)
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)

    if not got and name in HF_RESOLVE:
        try:
            zip_path = _wget_download(HF_RESOLVE[name], tmp_zip)
            extract_rel = HF_MIRRORS[name][2]
            _extract(zip_path, str(root / extract_rel))
            os.remove(zip_path)
            if _asset_ready(name, folder):
                got = True
                sources.append("%s=wget:%s" % (name, HF_RESOLVE[name]))
        except Exception as exc:
            print("  wget HF-mirror failed: %s" % exc, flush=True)
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)

    if not got and name in HF_MIRRORS:
        repo, filename, extract_rel = HF_MIRRORS[name]
        try:
            zip_path = _hf_download(repo, filename)
            _extract(zip_path, str(root / extract_rel))
            if _asset_ready(name, folder):
                got = True
                sources.append("%s=hf:%s/%s" % (name, repo, filename))
        except Exception as exc:
            print("  HF hub failed: %s" % exc, flush=True)

    if not got:
        failed.append(name)

print("asset sources:", "; ".join(sources), flush=True)
if failed:
    print("error: kitchen asset download failed for: " + ", ".join(failed), file=sys.stderr)
    sys.exit(1)
print("kitchen assets ok:", ", ".join(names), flush=True)
PY

echo "==> import smoke (robocasa / robosuite / mujoco / numpy)"
python -c "import robocasa, robosuite, mujoco, numpy; print(robocasa.__version__, mujoco.__version__, numpy.__version__)"

echo
echo "robosuite SHA: $(git -C "$ROBOSUITE_TREE" rev-parse HEAD)"
echo "robocasa  SHA: $(git -C "$ROBOCASA_TREE" rev-parse HEAD)"
echo
echo "Next steps:"
echo "  conda activate dynisafety-robocasa"
echo "  export MUJOCO_GL=egl"
echo "  pip install -e ${ROOT} --no-deps"
echo "  pip install -r ${ROOT}/requirements/robocasa.txt"
echo "  dynisafety doctor --backend robocasa"
echo
echo "Reminder: do not install RoboCasa into dynisafety-libero / simplerenv / robotwin."
echo "sim_root (later YAML): \${third_party}/robocasa → ${ROBOCASA_TREE}"
