#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

SEED="42"
CHUNK_STEPS="500000"
RUN_PREFIX="ppo_seed42_rewardE2_master_astar"
REPORT_PATH="outputs/logs/report_${RUN_PREFIX}.txt"

{
  echo "Stage3 RewardE2 MASTER Curriculum Report (Million Steps)"
  echo "reward_e=true_path_shaping + step_penalty=-0.02 + penalties"
  echo "curriculum=gentle step (radius+2), high threshold (80%), large window (100)"
} > "${REPORT_PATH}"

CUR_MODEL=""

for IDX in 1 2 3 4; do
  RUN_TAG="ppo_seed42_rewardE_master_chunk${IDX}"
  echo "========== [MASTER RUN] CHUNK ${IDX}/4 START =========="

  CMD=(
    "${PYTHON_BIN}" src/train_agents.py
    --method ppo
    --seed "${SEED}"
    --reward-scheme E
    --total-timesteps "${CHUNK_STEPS}"
    --n-eval-episodes 100
    --success-threshold 0.8
    --configs-root configs
    --env-config env_stage2_rewardE.json
    --ppo-config ppo_stage2_transfer_lr1e4.json
    --use-curriculum true
    --curriculum-initial-level 1
    --curriculum-radius-step 2
    --curriculum-success-window 100
    --curriculum-levelup-threshold 0.8
    --curriculum-ent-coef 0.05
    --eval-deterministic false
    --run-tag-suffix "master_chunk${IDX}"
  )

  if [[ -n "${CUR_MODEL}" ]]; then
    CMD+=(--init-model-path "${CUR_MODEL}")
  fi

  "${CMD[@]}"

  CUR_MODEL="outputs/checkpoints/${RUN_TAG}/best_model.zip"
  
  if [[ ! -f "${CUR_MODEL}" ]]; then
    echo "Model checkpoint not found, exiting..."
    exit 1
  fi
  sleep 10
done

echo "MASTER RUN FINISHED. Check ${REPORT_PATH}"
