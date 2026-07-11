#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PROJECT_PYTHON="/Users/huchuanen/miniforge3/envs/campus311/bin/python"
if [[ -x "$PROJECT_PYTHON" ]]; then
  exec "$PROJECT_PYTHON" -m streamlit run app.py
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  exec python -m streamlit run app.py
fi

exec ./.venv/bin/python -m streamlit run app.py
