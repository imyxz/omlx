#!/usr/bin/env bash
set -euo pipefail

# Package release artifacts for oMLX.
#
# Usage:
#   scripts/package_release.sh                # python sdist/wheel (+ DMG on macOS)
#   scripts/package_release.sh --python-only  # python sdist/wheel only

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ONLY=0

if [[ "${1:-}" == "--python-only" ]]; then
  PYTHON_ONLY=1
fi

cd "$ROOT_DIR"

echo "[release] Cleaning python dist artifacts..."
rm -rf dist build *.egg-info

echo "[release] Building Python sdist + wheel..."
python -m build

echo "[release] Python artifacts:"
ls -lh dist

echo "[release] SHA256 checksums (Python artifacts):"
(
  cd dist
  shasum -a 256 ./*
)

if [[ "$PYTHON_ONLY" -eq 1 ]]; then
  echo "[release] --python-only set; skipping macOS app packaging."
  exit 0
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[release] Non-macOS environment detected; skipping DMG build."
  exit 0
fi

echo "[release] Building macOS app + DMG..."
(
  cd packaging
  python build.py
)

echo "[release] DMG artifacts:"
ls -lh packaging/dist

echo "[release] SHA256 checksums (DMG artifacts):"
(
  cd packaging/dist
  shasum -a 256 ./*.dmg
)
