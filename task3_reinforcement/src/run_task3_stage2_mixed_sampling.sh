#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

SEED="${1:-42}"
N_EVAL="${2:-100}"
COOLDOWN_SEC="${3:-30}"
CHUNK_STEPS="${4:-100000}"

if [[ "${CHUNK_STEPS}" -ne 100000 ]]; then
  echo "[MixedStage2] this script is designed for 100k chunks, got ${CHUNK_STEPS}"
fi

TARGET_STD=0.05
RUN_PREFIX="ppo_seed${SEED}_rewardC_curriculum_transfer_mixed"
REPORT_PATH="outputs/logs/report_${RUN_PREFIX}.txt"

STAGE1_CKPT="outputs/checkpoints/ppo_seed${SEED}_rewardA_curriculum_learn_k1_stage/best_model.zip"
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[MixedStage2] missing Stage1 checkpoint: ${STAGE1_CKPT}"
  exit 1
fi

CUR_MODEL="${STAGE1_CKPT}"
CUR_LEVEL=2
PASS_REACHED=0
PASS_TAG=""

RATIO_CHUNK1=0.20
RATIO_CHUNK2=0.35
RATIO_CHUNK3=0.50

echo "Stage2 Mixed Sampling Report" > "${REPORT_PATH}"
echo "seed=${SEED} target_standard_success=${TARGET_STD}" >> "${REPORT_PATH}"
echo "strategy=linear mixed sampling (20% -> 35% -> 50% standard_start)" >> "${REPORT_PATH}"
echo "stagnation_penalty=-0.05 over short 5-step window" >> "${REPORT_PATH}"
echo "standard_potential_multiplier=1.5 during standard_start sampling" >> "${REPORT_PATH}"

for IDX in 1 2 3; do
  SUFFIX="transfer_mixed_chunk${IDX}"
  RUN_TAG="${RUN_PREFIX}_chunk${IDX}"

  CHUNK_RATIO="${RATIO_CHUNK1}"
  if [[ "${IDX}" -eq 2 ]]; then
    CHUNK_RATIO="${RATIO_CHUNK2}"
  elif [[ "${IDX}" -eq 3 ]]; then
    CHUNK_RATIO="${RATIO_CHUNK3}"
  fi

  echo "[MixedStage2] chunk ${IDX}/3 start (initial_level=${CUR_LEVEL}, standard_ratio=${CHUNK_RATIO})"
  "${PYTHON_BIN}" src/train_agents.py \
    --method ppo \
    --seed "${SEED}" \
    --reward-scheme C \
    --total-timesteps "${CHUNK_STEPS}" \
    --n-eval-episodes "${N_EVAL}" \
    --success-threshold 0.8 \
    --configs-root configs \
    --env-config env_stage2_mixed.json \
    --ppo-config ppo_stage2_transfer_lr1e4.json \
    --use-curriculum true \
    --curriculum-initial-level "${CUR_LEVEL}" \
    --curriculum-train-fixed-level 0 \
    --curriculum-mixed-sampling true \
    --curriculum-standard-start-ratio "${CHUNK_RATIO}" \
    --curriculum-standard-potential-multiplier 1.5 \
    --curriculum-radius-step 5 \
    --curriculum-success-window 50 \
    --curriculum-levelup-threshold 0.7 \
    --curriculum-smooth-bridge-stages 4 \
    --curriculum-ent-coef 0.02 \
    --curriculum-ent-coef-end 0.001 \
    --curriculum-ent-coef-phases 3 \
    --curriculum-export-level-gifs false \
    --eval-deterministic false \
    --init-model-path "${CUR_MODEL}" \
    --run-tag-suffix "${SUFFIX}"

  METRICS_LAYERS="outputs/tables/metrics_layers_${RUN_TAG}.csv"
  RUN_LOG="outputs/logs/run_${RUN_TAG}.json"
  CUR_MODEL="outputs/checkpoints/${RUN_TAG}/best_model.zip"
  if [[ ! -f "${CUR_MODEL}" || ! -f "${METRICS_LAYERS}" || ! -f "${RUN_LOG}" ]]; then
    echo "[MixedStage2] missing artifacts after chunk ${IDX}" | tee -a "${REPORT_PATH}"
    exit 1
  fi

  CHUNK_REPORT=$("${PYTHON_BIN}" - <<PY
import json
import pandas as pd

layers = pd.read_csv("${METRICS_LAYERS}")
with open("${RUN_LOG}", "r", encoding="utf-8") as f:
    run_log = json.load(f)

std_sr = float(layers.loc[layers["layer"] == "standard_start", "success_rate"].iloc[0])
std_ret = float(layers.loc[layers["layer"] == "standard_start", "episode_return_mean"].iloc[0])
k2_ret = float(layers.loc[layers["layer"] == "k2", "episode_return_mean"].iloc[0])
ret_gap = k2_ret - std_ret

cur = run_log.get("curriculum", {}) if isinstance(run_log.get("curriculum"), dict) else {}
lvl = int(cur.get("level", 1))
events = cur.get("level_up_events", []) if isinstance(cur.get("level_up_events"), list) else []
ratio_now = float(cur.get("current_standard_start_ratio", cur.get("standard_start_ratio", ${CHUNK_RATIO})))
boost_events = cur.get("standard_ratio_boost_events", []) if isinstance(cur.get("standard_ratio_boost_events"), list) else []

print(f"standard_success_rate={std_sr:.4f}")
print(f"standard_return_mean={std_ret:.4f}")
print(f"k2_return_mean={k2_ret:.4f}")
print(f"k2_minus_standard_return={ret_gap:.4f}")
print(f"curriculum_level={lvl}")
print(f"level_up_events_count={len(events)}")
print(f"configured_standard_ratio=${CHUNK_RATIO}")
print(f"effective_standard_ratio={ratio_now:.4f}")
print(f"standard_ratio_boost_events_count={len(boost_events)}")
print(f"pass={std_sr >= ${TARGET_STD}}")
PY
)

  echo "chunk=${IDX}" >> "${REPORT_PATH}"
  echo "${CHUNK_REPORT}" >> "${REPORT_PATH}"

  CHUNK_PASS=$(echo "${CHUNK_REPORT}" | awk -F= '/^pass=/{print $2}')
  CUR_LEVEL=$(echo "${CHUNK_REPORT}" | awk -F= '/^curriculum_level=/{print $2}')

  if [[ "${CHUNK_PASS}" == "True" ]]; then
    PASS_REACHED=1
    PASS_TAG="${RUN_TAG}"
    echo "[MixedStage2] threshold reached at chunk ${IDX}, export long GIFs and stop early"
    break
  fi

  if [[ "${IDX}" -lt 3 ]]; then
    echo "[MixedStage2] cooldown ${COOLDOWN_SEC}s"
    sleep "${COOLDOWN_SEC}"
  fi
done

"${PYTHON_BIN}" - <<PY
from pathlib import Path

report_path = Path("${REPORT_PATH}")
lines = report_path.read_text(encoding="utf-8").splitlines()

chunk_std = {}
for i, line in enumerate(lines):
  if line.startswith("chunk="):
    try:
      idx = int(line.split("=", 1)[1])
    except Exception:
      continue
    std = None
    for j in range(i + 1, min(i + 12, len(lines))):
      if lines[j].startswith("standard_return_mean="):
        try:
          std = float(lines[j].split("=", 1)[1])
        except Exception:
          std = None
        break
    if std is not None:
      chunk_std[idx] = std

if len(chunk_std) >= 2:
  idxs = sorted(chunk_std.keys())
  first, last = idxs[0], idxs[-1]
  overall_slope = (chunk_std[last] - chunk_std[first]) / float(last - first)
  lines.append(f"standard_return_slope_overall={overall_slope:.4f} per_chunk")
  for a, b in zip(idxs, idxs[1:]):
    slope = chunk_std[b] - chunk_std[a]
    lines.append(f"standard_return_slope_chunk{a}_to_chunk{b}={slope:.4f}")
else:
  lines.append("standard_return_slope_overall=NA")

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[MixedStage2] appended standard_return_mean slope diagnostics")
PY

if [[ "${PASS_REACHED}" -eq 1 ]]; then
  PASS_LOG="outputs/logs/run_${PASS_TAG}.json"
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from hero_task3_env import HeroTask3Env

with open("${PASS_LOG}", "r", encoding="utf-8") as f:
    run_log = json.load(f)

ckpt = Path(run_log.get("model_artifact_path", ""))
if not ckpt.exists():
    raise SystemExit(f"checkpoint missing: {ckpt}")

model = PPO.load(str(ckpt), device="cpu")
fig_dir = Path("outputs/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

for i in range(3):
    s = int(${SEED}) + 99000 + i
    env = HeroTask3Env(seed=s, reward_scheme="C", render_mode="rgb_array", potential_scale_c=0.05, wall_hit_penalty_c=0.0, dead_loop_penalty_c=0.0, stagnation_penalty_c=-0.05)
    obs, _ = env.reset(seed=s)
    frames = []
    f0 = env.render()
    if f0 is not None:
        frames.append(np.asarray(f0, dtype=np.uint8))

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, _r, t, tr, _info = env.step(int(action))
        fr = env.render()
        if fr is not None:
            frames.append(np.asarray(fr, dtype=np.uint8))
        done = bool(t or tr)
    env.close()

    out = fig_dir / f"long_navigation_${RUN_PREFIX}_{i+1}.gif"
    if frames:
        imageio.mimsave(out, frames, fps=6)
        print(f"exported={out}")
PY
else
  echo "[MixedStage2] threshold not reached after 3 chunks" >> "${REPORT_PATH}"
fi

echo "[MixedStage2] report: ${REPORT_PATH}"
