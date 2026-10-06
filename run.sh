#!/usr/bin/env bash
# Starts the FastAPI backend and Streamlit frontend together for local dev.
# Requires backend/.venv and frontend/.venv to already exist:
#   cd backend  && python -m venv .venv && .venv/Scripts/python -m pip install -r requirements-dev.txt
#   cd frontend && python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"

venv_python() {
    if [ -f "$1/Scripts/python.exe" ]; then
        echo "$1/Scripts/python.exe"
    else
        echo "$1/bin/python"
    fi
}

BACKEND_PY="$(venv_python backend/.venv)"
FRONTEND_PY="$(venv_python frontend/.venv)"

for py in "$BACKEND_PY" "$FRONTEND_PY"; do
    if [ ! -f "$py" ]; then
        echo "Missing virtualenv interpreter: $py" >&2
        echo "Create it first (see the comment at the top of this script)." >&2
        exit 1
    fi
done

cleanup() {
    echo ""
    echo "Stopping..."
    kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting backend on http://localhost:$BACKEND_PORT ..."
(cd backend && exec "../$BACKEND_PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$BACKEND_PORT") &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:$FRONTEND_PORT ..."
(
    cd frontend
    export BACKEND_BASE_URL="http://localhost:$BACKEND_PORT"
    exec "../$FRONTEND_PY" -m streamlit run streamlit_app.py --server.port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

wait
