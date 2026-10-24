#!/usr/bin/env bash
# Local CI gate for Link2Kaggle&Colab (offline / mock-first).
# Prefer this over GitHub Actions when cloud minutes are unavailable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
  RUFF="$ROOT/.venv/bin/ruff"
  PYTEST="$ROOT/.venv/bin/pytest"
else
  PY="${PYTHON:-python3}"
  RUFF="${RUFF:-ruff}"
  PYTEST="${PYTEST:-pytest}"
fi

SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) SKIP_BUILD=1 ;;
    -h|--help)
      echo "Usage: scripts/ci.sh [--fast]"
      echo "  --fast  skip python -m build"
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

echo "==> compileall"
"$PY" -m compileall -q src

echo "==> ruff"
"$RUFF" check .

echo "==> pytest (offline: not live)"
"$PYTEST" -m "not live" -q --cov=link2kaggle_colab --cov-report=term-missing:skip-covered

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "==> build"
  "$PY" -m build
else
  echo "==> build skipped (--fast)"
fi

echo "OK: local CI passed"
