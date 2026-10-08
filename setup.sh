#!/bin/bash
# Run from anywhere; all paths are relative to this checkout.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash setup.sh [--register-only]'
  echo 'Installs Python dependencies and registers this checkout. Model downloads are separate.'
  exit 0
fi
if [[ "${1:-}" == --register-only ]]; then
  python3 scripts/register_install.py
  exit 0
fi
if [[ $# -ne 0 ]]; then echo 'Unknown argument. Use --help.' >&2; exit 2; fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'Mac Diffusion requires Apple Silicon macOS, running natively (not Rosetta).' >&2
  exit 1
fi
if [[ "$(sw_vers -productVersion | cut -d. -f1)" -lt 14 ]]; then
  echo 'macOS 14 or newer is required.' >&2; exit 1
fi
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.11 .venv; fi
  .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,11), "Use a Python 3.11 virtual environment"'
  uv pip install --python .venv/bin/python -r backend/requirements.txt
else
  if ! command -v python3.11 >/dev/null 2>&1; then
    echo 'Install Python 3.11 or uv, then rerun: bash setup.sh' >&2; exit 1
  fi
  if [[ ! -x .venv/bin/python ]]; then python3.11 -m venv .venv; fi
  .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,11), "Use a Python 3.11 virtual environment"'
  .venv/bin/python -m pip install -r backend/requirements.txt
fi
.venv/bin/python scripts/register_install.py
printf '\nSetup complete. Download and convert the default model next:\n'
echo '.venv/bin/python -m backend.convert_models --model sdxs'
echo 'Then drag dist/Mac Diffusion Core.tox into your TouchDesigner network.'
