#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

SEED="${1:-42}"
N_EVAL="${2:-100}"
COOLDOWN_SEC="${3:-10}"
CHUNK_STEPS="${4:-100000}"

RUN_PREFIX="ppo_seed${SEED}_rewardE2_curriculum_astar"
REPORT_PATH="outputs/logs/report_${RUN_PREFIX}.txt"

CUR_MODEL=""

{
  echo "Stage2 RewardE2 Curriculum Report"
  echo "seed=${SEED}"
  echo "strategy=curriculum learning built-in wrapper"
  echo "reward_e=true_path_shaping + step_penalty=-0.02 + penalties"
  echo "schedule=3x100k with ${COOLDOWN_SEC}s cooldown"
} > "${REPORT_PATH}"

for IDX in 1 2 3; do
  SUFFIX="rewardE2_curr_chunk${IDX}"
  RUN_TAG="ppo_seed${SEED}_rewardE_curriculum_${SUFFIX}"

  echo "[RewardE2 Curr] chunk ${IDX}/3 start"

  CMD=(
    "${PYTHON_BIN}" src/train_agents.py
    --method ppo
    --seed "${SEED}"
    --reward-scheme E
    --total-timesteps "${CHUNK_STEPS}"
    --n-eval-episodes "${N_EVAL}"
    --success-threshold 0.8
    --configs-root configs
    --env-config env_stage2_rewardE.json
    --ppo-config ppo_stage2_transfer_lr1e4.json
    --use-curriculum true
    --curriculum-initial-level 1
    --curriculum-radius-step 5
    --curriculum-success-window 50
    --curriculum-levelup-threshold 0.7
    --curriculum-ent-coef 0.05
    --eval-deterministic false
    --run-tag-suffix "${SUFFIX}"
  )

  if [[ -n "${CUR_MODEL}" ]]; then
    CMD+=(--init-model-path "${CUR_MODEL}")
  fi

  "${CMD[@]}"

  METRICS="outputs/tables/metrics_${RUN_TAG}.csv"
  RUN_LOG="outputs/logs/run_${RUN_TAG}.json"
  CUR_MODEL="outputs/checkpoints/${RUN_TAG}/best_model.zip"

  if [[ ! -f "${CUR_MODEL}" || ! -f "${METRICS}" || ! -f "${RUN_LOG}" ]]; then
    echo "[RewardE2 Curr] missing artifacts after chunk ${IDX}" | tee -a "${REPORT_PATH}"
    exit 1
  fi

  CHUNK_REPORT=$("${PYTHON_BIN}" - <<PY
import json
import pandas as pd

metrics = pd.read_csv("${METRICS}")
row = metrics.iloc[0]
with open("${RUN_LOG}", "r", encoding="utf-8") as f:
    run_log = json.load(f)

print(f"success_rate={float(row['success_rate']):.4f}")
print(f"episode_return_mean={float(row['episode_return_mean']):.4f}")
print(f"dead_loop_rate={float(row['dead_loop_rate']):.4f}")
print(f"wall_collision_case_rate={float(row['wall_collision_case_rate']):.4f}")
print(f"model_artifact_path={run_log.get('model_artifact_path', '')}")
PY
)

  echo "chunk=${IDX}" >> "${REPORT_PATH}"
  echo "${CHUNK_REPORT}" >> "${REPORT_PATH}"

  if [[ "${IDX}" -lt 3 ]]; then
    echo "[RewardE2 Curr] cooldown ${COOLDOWN_SEC}s"
    sleep "${COOLDOWN_SEC}"
  fi
done

echo "[RewardE2 Curr] Final report generation..."

FINAL_REPORT=$("${PYTHON_BIN}" - <<PY
import pandas as pd

m1 = pd.read_csv("outputs/tables/metrics_ppo_seed${SEED}_rewardE_curriculum_rewardE2_curr_chunk1.csv")
m2 = pd.read_csv("outputs/tables/metrics_ppo_seed${SEED}_rewardE_curriculum_rewardE2_curr_chunk2.csv")
m3 = pd.read_csv("outputs/tables/metrics_ppo_seed${SEED}_rewardE_curriculum_rewardE2_curr_chunk3.csv")

r1 = float(m1.iloc[0]['episode_return_mean'])
r2 = float(m2.iloc[0]['episode_return_mean'])
r3 = float(m3.iloc[0]['episode_return_mean'])

slope1_2 = r2 - r1
slope2_3 = r3 - r2
slope_tot = (r3 - r1) / 2.0

print(f"episode_return_slope_overall={slope_tot:.4f} per_chunk")
print(f"episode_return_slope_chunk1_to_chunk2={slope1_2:.4f}")
print(f"episode_return_slope_chunk2_to_chunk3={slope2_3:.4f}")
PY
)

echo "${FINAL_REPORT}" >> "${REPORT_PATH}"
echo "[RewardE2 Curr] appended return slope diagnostics"
echo "[RewardE2 Curr] report: ${REPORT_PATH}"
