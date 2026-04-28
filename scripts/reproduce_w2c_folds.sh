#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KRAFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AT_ROOT="$(cd "${KRAFT_DIR}/.." && pwd)"

TRACK_EVAL_ROOT="${AT_ROOT}/YOLOv12-BoT-SORT-ReID-w2c/BoT-SORT/TrackEval"
TRACKERS_ROOT="${TRACK_EVAL_ROOT}/data/trackers/mot_challenge"
GT_ROOT="${TRACK_EVAL_ROOT}/data/gt/mot_challenge/UAVSwarm"
DEFAULT_PYTHON="/home/mrpk9/.conda/envs/yolov12_botsort/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

if [[ $# -gt 0 ]]; then
  FOLDS=("$@")
else
  FOLDS=(1 2 3)
fi

echo "KRAfT W2C fold reproduction"
echo "KRAFT_DIR=${KRAFT_DIR}"
echo "TRACK_EVAL_ROOT=${TRACK_EVAL_ROOT}"
echo "FOLDS=${FOLDS[*]}"
echo "PYTHON_BIN=${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not executable: ${PYTHON_BIN}"
  echo "Set an explicit interpreter with: PYTHON_BIN=/path/to/python bash ./scripts/reproduce_w2c_folds.sh"
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["numpy", "torch", "yaml", "easydict"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}: {e}")
if missing:
    print("Environment check failed:")
    for row in missing:
        print(" -", row)
    raise SystemExit(1)
print("Environment check passed.")
PY

for fold in "${FOLDS[@]}"; do
  if [[ ! "${fold}" =~ ^[123]$ ]]; then
    echo "Unsupported fold '${fold}'. Use 1, 2, or 3."
    exit 1
  fi

  cfg="${KRAFT_DIR}/configs/w2c_fold${fold}_test.yaml"
  tracker_name="fold${fold}-test-kraft"
  benchmark="fold${fold}"
  split="test"

  if [[ ! -f "${cfg}" ]]; then
    echo "Missing config: ${cfg}"
    exit 1
  fi

  echo
  echo "============================"
  echo "Running fold ${fold}"
  echo "Config: ${cfg}"
  echo "Tracker name: ${tracker_name}"
  echo "============================"

  "${PYTHON_BIN}" "${KRAFT_DIR}/main.py" --config "${cfg}"

  "${PYTHON_BIN}" "${TRACK_EVAL_ROOT}/scripts/run_mot_challenge.py" \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1 \
    --METRICS HOTA CLEAR Identity \
    --TRACKERS_TO_EVAL "${tracker_name}" \
    --GT_FOLDER "${GT_ROOT}/w2c-fold-${fold}" \
    --TRACKERS_FOLDER "${TRACKERS_ROOT}" \
    --BENCHMARK "${benchmark}" \
    --SPLIT_TO_EVAL "${split}"

  summary_path="${TRACKERS_ROOT}/fold${fold}-test/${tracker_name}/UAV_summary.txt"
  if [[ -f "${summary_path}" ]]; then
    echo
    echo "Summary (${summary_path}):"
    cat "${summary_path}"
  else
    echo "Expected summary not found: ${summary_path}"
    exit 1
  fi
done

echo
echo "Compact fold summary:"
"${PYTHON_BIN}" - "${TRACKERS_ROOT}" "${FOLDS[@]}" <<'PY'
import pathlib
import sys

trackers_root = pathlib.Path(sys.argv[1])
folds = sys.argv[2:]
keys = ["HOTA", "DetA", "AssA", "MOTA", "IDF1"]

print("Fold\t" + "\t".join(keys))
for fold in folds:
    summary = trackers_root / f"fold{fold}-test" / f"fold{fold}-test-kraft" / "UAV_summary.txt"
    if not summary.exists():
        print(f"{fold}\tmissing")
        continue
    lines = [ln.strip() for ln in summary.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        print(f"{fold}\tinvalid summary")
        continue
    header = lines[0].split()
    values = lines[1].split()
    idx = {k: i for i, k in enumerate(header)}
    row = [values[idx[k]] if k in idx and idx[k] < len(values) else "NA" for k in keys]
    print(f"{fold}\t" + "\t".join(row))
PY
