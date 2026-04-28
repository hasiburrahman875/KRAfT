#!/usr/bin/env bash
set -euo pipefail

# Stages the W2C motion-model checkpoint into the path expected by configs/w2c_fold*_test.yaml.

SRC_DEFAULT="/cluster/pixstor/madrias-lab/Hasibur/AT/kraft/experiments/uavswarm_ddm_1000_deeper/_epoch2100.pt"
SRC="${1:-${SRC_DEFAULT}}"
DEST_DIR="${2:-experiments/uavswarm_ddm_1000_deeper}"
DEST_FILE="${DEST_DIR}/_epoch2100.pt"

if [[ ! -f "${SRC}" ]]; then
  echo "Source checkpoint not found: ${SRC}"
  echo "Usage: bash scripts/stage_w2c_weights.sh /path/to/_epoch2100.pt [dest_dir]"
  exit 1
fi

mkdir -p "${DEST_DIR}"
cp -f "${SRC}" "${DEST_FILE}"

if command -v sha256sum >/dev/null 2>&1; then
  echo "Staged checkpoint: ${DEST_FILE}"
  sha256sum "${DEST_FILE}"
else
  echo "Staged checkpoint: ${DEST_FILE}"
fi
