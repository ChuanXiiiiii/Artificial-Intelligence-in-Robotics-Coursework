#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
FIXED_K="${2:-5}"
ARTIFACT_PREFIX="${3:-preprocessed_main}"
PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

"${PYTHON_BIN}" src/preprocess_data.py \
  --artifact-prefix "${ARTIFACT_PREFIX}"
"${PYTHON_BIN}" src/train_kmeans.py \
  --seed "${SEED}" \
  --artifact-prefix "${ARTIFACT_PREFIX}"
"${PYTHON_BIN}" src/train_gmm.py \
  --seed "${SEED}" \
  --artifact-prefix "${ARTIFACT_PREFIX}"
"${PYTHON_BIN}" src/compare_methods.py \
  --seed "${SEED}" \
  --fixed-k "${FIXED_K}" \
  --artifact-prefix "${ARTIFACT_PREFIX}"
"${PYTHON_BIN}" src/visualize_embeddings.py \
  --seed "${SEED}" \
  --view method_best \
  --artifact-prefix "${ARTIFACT_PREFIX}"

echo "Task2 pipeline completed for seed ${SEED} (fixed_k=${FIXED_K})."
