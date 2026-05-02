#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

SEEDS=(42 123 2026 9 20)
METHODS=(random dqn ppo)
TOTAL_TIMESTEPS="${1:-1000000}"
MAX_PARALLEL="${2:-2}"
COOLDOWN_SEC="${3:-30}"
STRICT_EXPORT="${4:-true}"
REWARD_CSV="${5:-G}"

IFS=',' read -r -a REWARDS <<< "${REWARD_CSV}"

if [[ "${MAX_PARALLEL}" -lt 1 || "${MAX_PARALLEL}" -gt 3 ]]; then
  echo "MAX_PARALLEL must be in [1,3], got ${MAX_PARALLEL}"
  exit 1
fi

if [[ "${STRICT_EXPORT}" != "true" && "${STRICT_EXPORT}" != "false" ]]; then
  echo "STRICT_EXPORT must be true or false, got ${STRICT_EXPORT}"
  exit 1
fi

mkdir -p outputs/logs outputs/tables outputs/checkpoints outputs/figures

JOBS=()
for reward in "${REWARDS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOBS+=("${method},${reward},${seed}")
    done
  done
done

TOTAL_JOBS="${#JOBS[@]}"
JOB_INDEX=0
BATCH_INDEX=0

echo "Starting batched queue: total_jobs=${TOTAL_JOBS}, max_parallel=${MAX_PARALLEL}, cooldown=${COOLDOWN_SEC}s, strict_export=${STRICT_EXPORT}"
echo "Rewards=${REWARD_CSV}"

while [[ "${JOB_INDEX}" -lt "${TOTAL_JOBS}" ]]; do
  BATCH_INDEX=$((BATCH_INDEX + 1))
  PIDS=()
  META=()

  for ((slot = 0; slot < MAX_PARALLEL && JOB_INDEX < TOTAL_JOBS; slot++)); do
    IFS=',' read -r method reward seed <<< "${JOBS[$JOB_INDEX]}"
    run_tag="${method}_seed${seed}_reward${reward}"
    log_file="outputs/logs/batch_${BATCH_INDEX}_${run_tag}.log"

    echo "[Batch ${BATCH_INDEX}] Launching ${run_tag} (timesteps=${TOTAL_TIMESTEPS})"
    ENV_CONFIG_ARGS=()
    if [[ "${reward}" == "F2" ]]; then
      ENV_CONFIG_ARGS+=(--env-config env_stage5_rewardF2.json)
    elif [[ "${reward}" == "G" ]]; then
      ENV_CONFIG_ARGS+=(--env-config env_final_rewardG.json)
    elif [[ "${reward}" == "H" ]]; then
      ENV_CONFIG_ARGS+=(--env-config env_stage6_rewardH.json)
    fi

    "${PYTHON_BIN}" src/train_agents.py \
      --method "${method}" \
      --seed "${seed}" \
      --reward-scheme "${reward}" \
      "${ENV_CONFIG_ARGS[@]}" \
      --total-timesteps "${TOTAL_TIMESTEPS}" \
      --n-eval-episodes 100 \
      --success-threshold 0.8 > "${log_file}" 2>&1 &

    PIDS+=("$!")
    META+=("${run_tag}|${log_file}")
    JOB_INDEX=$((JOB_INDEX + 1))
  done

  BATCH_FAILED=0
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    meta="${META[$i]}"
    IFS='|' read -r run_tag log_file <<< "${meta}"

    if wait "${pid}"; then
      metrics_path="outputs/tables/metrics_${run_tag}.csv"
      loss_path="outputs/tables/loss_${run_tag}.csv"
      model_path="outputs/checkpoints/${run_tag}/best_model.zip"

      if [[ ! -f "${metrics_path}" || ! -f "${loss_path}" || ! -f "${model_path}" ]]; then
        echo "Artifact check failed for ${run_tag}"
        echo "Expected: ${metrics_path}, ${loss_path}, ${model_path}"
        echo "See log: ${log_file}"
        BATCH_FAILED=1
      else
        echo "[Batch ${BATCH_INDEX}] Completed ${run_tag} with required artifacts"
      fi
    else
      echo "Run failed: ${run_tag}. See ${log_file}"
      BATCH_FAILED=1
    fi
  done

  if [[ "${BATCH_FAILED}" -ne 0 ]]; then
    echo "Batch ${BATCH_INDEX} failed; stopping queue."
    exit 1
  fi

  if [[ "${JOB_INDEX}" -lt "${TOTAL_JOBS}" ]]; then
    echo "Batch ${BATCH_INDEX} done. Cooling down for ${COOLDOWN_SEC}s to stabilize thermal state."
    sleep "${COOLDOWN_SEC}"
  fi
done

"${PYTHON_BIN}" src/aggregate_multi_seed.py \
  --seeds "42,123,2026,9,20" \
  --methods "random,dqn,ppo" \
  --reward-schemes "${REWARD_CSV}"

"${PYTHON_BIN}" src/export_best_trajectories.py \
  --methods "random,dqn,ppo" \
  --reward-schemes "${REWARD_CSV}" \
  --strict "${STRICT_EXPORT}"

"${PYTHON_BIN}" src/sync_report_assets.py

echo "Task3 full matrix run completed (batched queue mode)."
