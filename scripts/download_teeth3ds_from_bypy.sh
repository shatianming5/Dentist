#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REMOTE_ROOT="${1:-dentist_full_20260130_1124Z}"
REMOTE_PATH="${REMOTE_ROOT}/data/teeth3ds"
LOCAL_PATH="${2:-data/teeth3ds}"
PROCESSES="${BYPY_PROCESSES:-4}"

mkdir -p "$LOCAL_PATH"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] bypy --downloader aria2 --processes ${PROCESSES} downdir $REMOTE_PATH $LOCAL_PATH"
bypy --downloader aria2 --processes "${PROCESSES}" downdir "$REMOTE_PATH" "$LOCAL_PATH"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE"
