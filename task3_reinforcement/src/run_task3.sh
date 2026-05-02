#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
METHOD="${2:-dqn}"
REWARD_SCHEME="${3:-G}"
TOTAL_TIMESTEPS="${4:-1000000}"

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

ENV_CONFIG_ARGS=()
if [[ "${REWARD_SCHEME}" == "F2" ]]; then
  ENV_CONFIG_ARGS+=(--env-config env_stage5_rewardF2.json)
elif [[ "${REWARD_SCHEME}" == "G" ]]; then
  ENV_CONFIG_ARGS+=(--env-config env_final_rewardG.json)
elif [[ "${REWARD_SCHEME}" == "H" ]]; then
  ENV_CONFIG_ARGS+=(--env-config env_stage6_rewardH.json)
fi

"${PYTHON_BIN}" src/train_agents.py \
  --method "${METHOD}" \
  --seed "${SEED}" \
  --reward-scheme "${REWARD_SCHEME}" \
  "${ENV_CONFIG_ARGS[@]}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --n-eval-episodes 100 \
  --success-threshold 0.8

echo "Task3 single run completed: method=${METHOD}, seed=${SEED}, reward=${REWARD_SCHEME}" 
