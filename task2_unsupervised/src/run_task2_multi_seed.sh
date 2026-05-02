#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"
SEEDS=(42 123 2026)
FIXED_K="${1:-5}"
ARTIFACT_PREFIX="${2:-preprocessed_main}"

echo "Running shared preprocessing once..."
"${PYTHON_BIN}" src/preprocess_data.py --artifact-prefix "${ARTIFACT_PREFIX}"

for seed in "${SEEDS[@]}"; do
  echo "Running seed ${seed}..."
  "${PYTHON_BIN}" src/train_kmeans.py --seed "${seed}" --artifact-prefix "${ARTIFACT_PREFIX}"
  "${PYTHON_BIN}" src/train_gmm.py --seed "${seed}" --artifact-prefix "${ARTIFACT_PREFIX}"
  "${PYTHON_BIN}" src/compare_methods.py --seed "${seed}" --fixed-k "${FIXED_K}" --artifact-prefix "${ARTIFACT_PREFIX}"
  "${PYTHON_BIN}" src/visualize_embeddings.py --seed "${seed}" --view method_best --artifact-prefix "${ARTIFACT_PREFIX}"
done

"${PYTHON_BIN}" src/aggregate_multi_seed.py --seeds "42,123,2026"

echo "Task2 multi-seed pipeline completed (fixed_k=${FIXED_K})."
