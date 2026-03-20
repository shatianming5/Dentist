#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REMOTE_ROOT="${REMOTE_ROOT:-dentist_full_20260130_1124Z/data/teeth3ds}"
LOCAL_ROOT="${LOCAL_ROOT:-data/teeth3ds}"
JAWS="${JAWS:-lower}"
WORKERS="${WORKERS:-4}"
BYPY_RETRY="${BYPY_RETRY:-5}"
LOG_DIR="${LOG_DIR:-runs/teeth3ds_download}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_teeth3ds_cases_parallel.sh [--jaw lower,upper] [--workers 4] [--dry-run]

Environment overrides:
  REMOTE_ROOT   Remote Teeth3DS root under bypy app dir.
  LOCAL_ROOT    Local Teeth3DS root.
  JAWS          Comma-separated jaw list, default: lower
  WORKERS       Number of case-level parallel workers, default: 4
  BYPY_RETRY    Retry count passed to bypy, default: 5

Notes:
  - This script downloads each case directory independently.
  - Complete cases (both OBJ and JSON present) are skipped.
  - It is intended to avoid one bad file blocking the whole jaw download.
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jaw|--jaws)
      JAWS="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

have_complete_case() {
  local jaw="$1"
  local case_id="$2"
  local local_case="$LOCAL_ROOT/$jaw/$case_id"
  [[ -s "$local_case/${case_id}_${jaw}.obj" && -s "$local_case/${case_id}_${jaw}.json" ]]
}

download_case() {
  local jaw="$1"
  local case_id="$2"
  local remote_case="$REMOTE_ROOT/$jaw/$case_id"
  local local_case="$LOCAL_ROOT/$jaw/$case_id"
  mkdir -p "$local_case"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN ${jaw}/${case_id}"
  bypy -r "$BYPY_RETRY" --downloader aria2 --processes 1 downdir "$remote_case" "$local_case"
}

run_case() {
  local jaw="$1"
  local case_id="$2"
  local fail_log="$LOG_DIR/fail_${jaw}.tsv"
  local done_log="$LOG_DIR/done_${jaw}.tsv"
  mkdir -p "$LOG_DIR"
  if download_case "$jaw" "$case_id"; then
    printf '%s\t%s\tok\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$jaw" "$case_id" >>"$done_log"
  else
    printf '%s\t%s\tfail\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$jaw" "$case_id" >>"$fail_log"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] FAIL ${jaw}/${case_id}" >&2
  fi
}

export ROOT_DIR REMOTE_ROOT LOCAL_ROOT BYPY_RETRY LOG_DIR
export -f have_complete_case
export -f download_case
export -f run_case

for jaw in ${JAWS//,/ }; do
  local_jaw="$LOCAL_ROOT/$jaw"
  if [[ ! -d "$local_jaw" ]]; then
    echo "Missing local jaw dir: $local_jaw" >&2
    echo "Create it first with a directory-level bypy downdir or mkdir skeleton." >&2
    exit 2
  fi

  mapfile -t case_ids < <(find "$local_jaw" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  if [[ ${#case_ids[@]} -eq 0 ]]; then
    echo "No local case dirs found under $local_jaw" >&2
    exit 2
  fi

  tmp_tasks="$(mktemp)"
  trap 'rm -f "$tmp_tasks"' EXIT
  for case_id in "${case_ids[@]}"; do
    if have_complete_case "$jaw" "$case_id"; then
      continue
    fi
    printf '%s\t%s\n' "$jaw" "$case_id" >>"$tmp_tasks"
  done

  total_tasks="$(wc -l <"$tmp_tasks" | tr -d ' ')"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] jaw=$jaw pending_cases=$total_tasks workers=$WORKERS"
  if [[ "$DRY_RUN" == "1" ]]; then
    sed -n '1,20p' "$tmp_tasks"
    rm -f "$tmp_tasks"
    trap - EXIT
    continue
  fi

  if [[ "$total_tasks" == "0" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] jaw=$jaw already complete"
    rm -f "$tmp_tasks"
    trap - EXIT
    continue
  fi

  xargs -P "$WORKERS" -n 2 bash -lc 'run_case "$1" "$2"' _ < <(tr '\t' '\n' <"$tmp_tasks")
  rm -f "$tmp_tasks"
  trap - EXIT
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE case-parallel teeth3ds download"
