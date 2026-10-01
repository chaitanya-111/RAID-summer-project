#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r frontend/backend/requirements.txt
python -m uvicorn frontend.backend.main:app --host 0.0.0.0 --port 8000
