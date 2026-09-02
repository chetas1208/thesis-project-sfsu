#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

load_local_env
ensure_cache_dirs
export_benchmark_env

if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util
required = ["kaggle", "kagglehub"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(0 if not missing else 1)
PY
then
  python3 -m pip install --user --cache-dir "${PIP_CACHE_DIR}" kaggle kagglehub
fi

python3 "${PROJECT_ROOT}/src/download_kaggle_sources.py" "$@"
