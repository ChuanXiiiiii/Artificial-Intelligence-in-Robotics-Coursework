#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

SEED="${1:-42}"
TOTAL_TIMESTEPS="${2:-6000000}"
N_EVAL_EPISODES="${3:-100}"
RUN_SUFFIX="${4:-stage7J_dqn_baseline}"

"${PYTHON_BIN}" src/train_agents.py \
  --method dqn \
  --seed "${SEED}" \
  --reward-scheme H \
  --env-config env_stage6_rewardH.json \
  --dqn-config dqn_stage7_baseline.json \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --n-eval-episodes "${N_EVAL_EPISODES}" \
  --success-threshold 0.8 \
  --use-curriculum true \
  --curriculum-initial-level 1 \
  --curriculum-radius-step 3 \
  --curriculum-success-window 250 \
  --curriculum-levelup-threshold 0.50 \
  --curriculum-smooth-bridge-stages 4 \
  --curriculum-mixed-sampling true \
  --curriculum-standard-start-ratio 0.25 \
  --curriculum-standard-potential-multiplier 1.5 \
  --run-tag-suffix "${RUN_SUFFIX}"

run_tag="dqn_seed${SEED}_rewardH_curriculum_${RUN_SUFFIX}"
metrics_path="outputs/tables/metrics_${run_tag}.csv"
loss_path="outputs/tables/loss_${run_tag}.csv"
model_path="outputs/checkpoints/${run_tag}/best_model.zip"
tb_dir_glob="outputs/logs/sb3_${run_tag}_phase*"

if [[ ! -f "${metrics_path}" || ! -f "${loss_path}" || ! -f "${model_path}" ]]; then
  echo "[DQN Curriculum] artifact check failed"
  echo "Expected: ${metrics_path}, ${loss_path}, ${model_path}"
  exit 1
fi

if ! find ${tb_dir_glob} -name 'events.out.tfevents.*' -print -quit 2>/dev/null | grep -q .; then
  echo "[DQN Curriculum] tensorboard artifact check failed"
  echo "Expected event file under: ${tb_dir_glob}"
  exit 1
fi

echo "Task3 DQN curriculum run completed: seed=${SEED}, timesteps=${TOTAL_TIMESTEPS}, reward=H"