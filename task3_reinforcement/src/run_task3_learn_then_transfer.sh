#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

PRETRAIN_TIMESTEPS="${1:-120000}"
TRANSFER_TIMESTEPS="${2:-120000}"
N_EVAL="${3:-100}"
SEED="${4:-42}"

PRE_SUFFIX="learn_k1_stage"
TRANSFER_SUFFIX="transfer_stage"

PRETRAIN_TAG="ppo_seed${SEED}_rewardA_curriculum_${PRE_SUFFIX}"
TRANSFER_TAG="ppo_seed${SEED}_rewardC_curriculum_${TRANSFER_SUFFIX}"
PRETRAIN_CKPT="outputs/checkpoints/${PRETRAIN_TAG}/best_model.zip"

echo "[LearnThenTransfer] Stage1: fixed k=1 pretrain (reward A)"
"${PYTHON_BIN}" src/train_agents.py \
  --method ppo \
  --seed "${SEED}" \
  --reward-scheme A \
  --total-timesteps "${PRETRAIN_TIMESTEPS}" \
  --n-eval-episodes "${N_EVAL}" \
  --use-curriculum true \
  --curriculum-initial-level 1 \
  --curriculum-train-fixed-level 1 \
  --curriculum-radius-step 5 \
  --curriculum-success-window 50 \
  --curriculum-levelup-threshold 0.7 \
  --curriculum-ent-coef 0.03 \
  --curriculum-ent-coef-end 0.001 \
  --curriculum-ent-coef-phases 3 \
  --curriculum-export-level-gifs false \
  --eval-deterministic false \
  --run-tag-suffix "${PRE_SUFFIX}"

if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "[LearnThenTransfer] missing pretrain checkpoint: ${PRETRAIN_CKPT}"
  exit 1
fi

echo "[LearnThenTransfer] Stage2: transfer with curriculum progression (reward C)"
"${PYTHON_BIN}" src/train_agents.py \
  --method ppo \
  --seed "${SEED}" \
  --reward-scheme C \
  --total-timesteps "${TRANSFER_TIMESTEPS}" \
  --n-eval-episodes "${N_EVAL}" \
  --use-curriculum true \
  --curriculum-initial-level 1 \
  --curriculum-train-fixed-level 0 \
  --curriculum-radius-step 5 \
  --curriculum-success-window 50 \
  --curriculum-levelup-threshold 0.7 \
  --curriculum-smooth-bridge-stages 4 \
  --curriculum-ent-coef 0.02 \
  --curriculum-ent-coef-end 0.001 \
  --curriculum-ent-coef-phases 3 \
  --curriculum-export-level-gifs true \
  --curriculum-gif-levels "1,2,3" \
  --curriculum-gif-fps 6 \
  --eval-deterministic false \
  --init-model-path "${PRETRAIN_CKPT}" \
  --run-tag-suffix "${TRANSFER_SUFFIX}"

echo "[LearnThenTransfer] done"
echo "Stage1 metrics: outputs/tables/metrics_${PRETRAIN_TAG}.csv"
echo "Stage1 layered metrics: outputs/tables/metrics_layers_${PRETRAIN_TAG}.csv"
echo "Stage2 metrics: outputs/tables/metrics_${TRANSFER_TAG}.csv"
echo "Stage2 layered metrics: outputs/tables/metrics_layers_${TRANSFER_TAG}.csv"