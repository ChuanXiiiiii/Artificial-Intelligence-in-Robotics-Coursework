#!/usr/bin/env bash
set -euo pipefail

# Stage 7: Recurrent PPO (CnnLstm) on Reward J (== reuse Reward H env config).
# LSTM grants the agent "object permanence" memory across the 7x7 radar window.

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

SEED="${1:-42}"
TOTAL_TIMESTEPS="${2:-6000000}"
N_EVAL_EPISODES="${3:-100}"
RUN_SUFFIX="${4:-stage7J_lstm}"

"${PYTHON_BIN}" src/train_agents.py \
        --method ppo \
        --seed "${SEED}" \
        --reward-scheme H \
        --env-config env_stage6_rewardH.json \
        --total-timesteps "${TOTAL_TIMESTEPS}" \
        --n-eval-episodes "${N_EVAL_EPISODES}" \
        --success-threshold 0.8 \
        --use-recurrent true \
        --lstm-hidden-size 256 \
        --n-lstm-layers 1 \
        --use-curriculum true \
        --curriculum-initial-level 1 \
        --curriculum-radius-step 3 \
        --curriculum-success-window 250 \
        --curriculum-levelup-threshold 0.50 \
        --curriculum-smooth-bridge-stages 4 \
        --curriculum-mixed-sampling true \
        --curriculum-standard-start-ratio 0.25 \
        --curriculum-standard-potential-multiplier 1.5 \
        --curriculum-ent-coef 0.08 \
        --curriculum-ent-coef-end 0.015 \
        --curriculum-ent-coef-phases 3 \
        --curriculum-export-level-gifs true \
        --curriculum-gif-levels "1,2,3" \
        --curriculum-gif-fps 6 \
        --run-tag-suffix "${RUN_SUFFIX}"

echo "Task3 Stage7 LSTM run completed: seed=${SEED}, timesteps=${TOTAL_TIMESTEPS}, reward=H+LSTM"
