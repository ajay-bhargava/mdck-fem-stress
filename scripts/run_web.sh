#!/usr/bin/env bash
set -euo pipefail

export PYVISTA_OFF_SCREEN=true

uv run uvicorn src.web.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}"
