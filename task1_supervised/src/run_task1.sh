#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

"${PYTHON_BIN}" src/build_split_manifest.py \
	--seed "${SEED}" \
	--bucket-size 20 \
	--min-val-support 30 \
	--min-test-support 30
"${PYTHON_BIN}" src/train_hog_svm.py --seed "${SEED}"
"${PYTHON_BIN}" src/train_resnet18.py --seed "${SEED}" --class-weighting sqrt_balanced
"${PYTHON_BIN}" src/compare_results.py --seed "${SEED}"
"${PYTHON_BIN}" src/sync_report_assets.py

echo "Task1 pipeline completed for seed ${SEED}."
