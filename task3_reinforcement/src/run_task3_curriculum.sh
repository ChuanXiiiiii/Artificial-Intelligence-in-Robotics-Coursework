#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

TOTAL_TIMESTEPS="${1:-300000}"
COOLDOWN_SEC="${2:-30}"
SEED="${3:-42}"

run_tag="ppo_seed${SEED}_rewardC_curriculum"

echo "[Curriculum] Launching ${run_tag} with total_timesteps=${TOTAL_TIMESTEPS}"
"${PYTHON_BIN}" src/train_agents.py \
  --method ppo \
  --seed "${SEED}" \
  --reward-scheme C \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --n-eval-episodes 100 \
  --success-threshold 0.8 \
  --use-curriculum true \
  --curriculum-initial-level 1 \
  --curriculum-radius-step 5 \
  --curriculum-success-window 50 \
  --curriculum-levelup-threshold 0.7 \
  --curriculum-smooth-bridge-stages 4 \
  --curriculum-ent-coef 0.05 \
  --curriculum-export-level-gifs true \
  --curriculum-gif-levels "1,2,3" \
  --curriculum-gif-fps 6

metrics_path="outputs/tables/metrics_${run_tag}.csv"
loss_path="outputs/tables/loss_${run_tag}.csv"
model_path="outputs/checkpoints/${run_tag}/best_model.zip"

if [[ ! -f "${metrics_path}" || ! -f "${loss_path}" || ! -f "${model_path}" ]]; then
  echo "[Curriculum] artifact check failed"
  echo "Expected: ${metrics_path}, ${loss_path}, ${model_path}"
  exit 1
fi

echo "[Curriculum] Completed with required artifacts"
echo "[Curriculum] Cooling down for ${COOLDOWN_SEC}s to stabilize thermal state"
sleep "${COOLDOWN_SEC}"
