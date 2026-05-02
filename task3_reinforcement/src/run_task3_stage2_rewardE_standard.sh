#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

SEED="${1:-42}"
N_EVAL="${2:-100}"
COOLDOWN_SEC="${3:-30}"
CHUNK_STEPS="${4:-100000}"

if [[ "${CHUNK_STEPS}" -ne 100000 ]]; then
  echo "[RewardE] this script is designed for 100k chunks, got ${CHUNK_STEPS}"
fi

RUN_PREFIX="ppo_seed${SEED}_rewardE2_standard_astar"
REPORT_PATH="outputs/logs/report_${RUN_PREFIX}.txt"

CUR_MODEL=""

{
  echo "Stage2 RewardE2 Standard Report"
  echo "seed=${SEED}"
  echo "strategy=no curriculum (equivalent standard_start_ratio=1.0)"
  echo "reward_e=true_path_shaping + step_penalty=-0.02 + penalties"
  echo "schedule=3x100k with ${COOLDOWN_SEC}s cooldown"
} > "${REPORT_PATH}"

for IDX in 1 2 3; do
  SUFFIX="rewardE2_std_chunk${IDX}"
  RUN_TAG="ppo_seed${SEED}_rewardE_${SUFFIX}"

  echo "[RewardE] chunk ${IDX}/3 start"

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
    --use-curriculum false
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
    echo "[RewardE] missing artifacts after chunk ${IDX}" | tee -a "${REPORT_PATH}"
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
    echo "[RewardE] cooldown ${COOLDOWN_SEC}s"
    sleep "${COOLDOWN_SEC}"
  fi
done

"${PYTHON_BIN}" - <<PY
from pathlib import Path

report_path = Path("${REPORT_PATH}")
lines = report_path.read_text(encoding="utf-8").splitlines()

chunk_ret = {}
for i, line in enumerate(lines):
    if line.startswith("chunk="):
        try:
            idx = int(line.split("=", 1)[1])
        except Exception:
            continue
        ret = None
        for j in range(i + 1, min(i + 10, len(lines))):
            if lines[j].startswith("episode_return_mean="):
                try:
                    ret = float(lines[j].split("=", 1)[1])
                except Exception:
                    ret = None
                break
        if ret is not None:
            chunk_ret[idx] = ret

if len(chunk_ret) >= 2:
    idxs = sorted(chunk_ret.keys())
    first, last = idxs[0], idxs[-1]
    overall_slope = (chunk_ret[last] - chunk_ret[first]) / float(last - first)
    lines.append(f"episode_return_slope_overall={overall_slope:.4f} per_chunk")
    for a, b in zip(idxs, idxs[1:]):
        slope = chunk_ret[b] - chunk_ret[a]
        lines.append(f"episode_return_slope_chunk{a}_to_chunk{b}={slope:.4f}")
else:
    lines.append("episode_return_slope_overall=NA")

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[RewardE] appended return slope diagnostics")
PY

echo "[RewardE] report: ${REPORT_PATH}"
