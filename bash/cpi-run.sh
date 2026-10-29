#!/usr/bin/env bash
# Run from /mnt/CPI-raw: ./cpi-run.sh ACTIVE process 2.46 [cores]
#                     ./cpi-run.sh ACTIVE png [cores]
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: $0 DATASET process PIXEL_UM [CORES]  |  $0 DATASET png [CORES]" >&2
  exit 2
fi
DATASET="${1%/}"
MODE="$2"
ROOT="${CPI_DATA_ROOT:-/mnt/CPI-raw}"
REPO="${CPI_REPO:-$HOME/CPI_dev/cpi3v}"
if [[ "$DATASET" == /* ]]; then DATA_DIR="$DATASET"; DATASET="$(basename "$DATASET")"; else DATA_DIR="$ROOT/$DATASET"; fi
[[ -d "$DATA_DIR" ]] || { echo "Directory not found: $DATA_DIR" >&2; exit 1; }
[[ -f "$REPO/python/proc/batchProcess.py" ]] || { echo "CPI repo not found: $REPO" >&2; exit 1; }
PYTHON="${CPI_PYTHON:-$REPO/.venv311/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
case "$MODE" in
  process|first)
    [[ $# -ge 3 ]] || { echo 'Processing requires PIXEL_UM (e.g. 2.46)' >&2; exit 2; }
    PIXEL_UM="$3"; CORES="${4:-10}"
    [[ -f "$DATA_DIR/background_selection.json" ]] && echo "Note: existing background selection found; --backgrounds choose will open GUI to review it."
    exec "$PYTHON" "$REPO/python/proc/batchProcess.py" "$DATA_DIR" \
      --pixel-um "$PIXEL_UM" --cores "$CORES" --no-images --backgrounds choose
    ;;
  png|export|second)
    CORES="${3:-10}"
    [[ -f "$DATA_DIR/flights.csv" ]] || { echo "Missing flight list: $DATA_DIR/flights.csv" >&2; exit 1; }
    exec "$PYTHON" "$REPO/python/proc/batchProcess.py" "$DATA_DIR" \
      --export-only --export-png --cores "$CORES" --flight-csv "$DATA_DIR/flights.csv" \
      --png-dir "$ROOT" --min-size-um 40 --min-focus 5 --overwrite-png \
      --greyscale --background-level 180
    ;;
  *) echo "Unknown mode: $MODE (use process or png)" >&2; exit 2;;
esac
