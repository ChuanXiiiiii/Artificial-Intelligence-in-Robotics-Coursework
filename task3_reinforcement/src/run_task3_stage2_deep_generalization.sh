#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

TOTAL_TIMESTEPS="${1:-300000}"
SEED="${2:-42}"
N_EVAL="${3:-100}"
COOLDOWN_SEC="${4:-30}"

if [[ "${TOTAL_TIMESTEPS}" -lt 300000 ]]; then
  echo "[Stage2Deep] TOTAL_TIMESTEPS must be >= 300000"
  exit 1
fi

STAGE1_TAG="ppo_seed${SEED}_rewardA_curriculum_learn_k1_stage"
STAGE1_CKPT="outputs/checkpoints/${STAGE1_TAG}/best_model.zip"
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[Stage2Deep] missing Stage1 checkpoint: ${STAGE1_CKPT}"
  exit 1
fi

RUN_SUFFIX="transfer_deep300k"
RUN_TAG="ppo_seed${SEED}_rewardC_curriculum_${RUN_SUFFIX}"

CHUNK1=100000
CHUNK2=100000
CHUNK3=$((TOTAL_TIMESTEPS - CHUNK1 - CHUNK2))
if [[ "${CHUNK3}" -le 0 ]]; then
  echo "[Stage2Deep] invalid chunk split"
  exit 1
fi

echo "[Stage2Deep] Stage2 deep generalization: total=${TOTAL_TIMESTEPS}, chunks=${CHUNK1}+${CHUNK2}+${CHUNK3}"
echo "[Stage2Deep] using init checkpoint: ${STAGE1_CKPT}"

CUR_INIT_MODEL="${STAGE1_CKPT}"
CUR_TOTAL=0
CUR_INIT_LEVEL=1

for CHUNK in "${CHUNK1}" "${CHUNK2}" "${CHUNK3}"; do
  IDX=$((CUR_TOTAL / 100000 + 1))
  CHUNK_SUFFIX="${RUN_SUFFIX}_chunk${IDX}"

  echo "[Stage2Deep] chunk ${IDX}: timesteps=${CHUNK}"
  "${PYTHON_BIN}" src/train_agents.py \
    --method ppo \
    --seed "${SEED}" \
    --reward-scheme C \
    --total-timesteps "${CHUNK}" \
    --n-eval-episodes "${N_EVAL}" \
    --success-threshold 0.8 \
    --use-curriculum true \
    --curriculum-initial-level "${CUR_INIT_LEVEL}" \
    --curriculum-train-fixed-level 0 \
    --curriculum-radius-step 5 \
    --curriculum-success-window 50 \
    --curriculum-levelup-threshold 0.7 \
    --curriculum-smooth-bridge-stages 4 \
    --curriculum-ent-coef 0.02 \
    --curriculum-ent-coef-end 0.001 \
    --curriculum-ent-coef-phases 3 \
    --curriculum-export-level-gifs false \
    --eval-deterministic false \
    --init-model-path "${CUR_INIT_MODEL}" \
    --ppo-config "ppo_stage2_transfer_lr1e4.json" \
    --run-tag-suffix "${CHUNK_SUFFIX}"

  CUR_TOTAL=$((CUR_TOTAL + CHUNK))
  LAST_TAG="ppo_seed${SEED}_rewardC_curriculum_${CHUNK_SUFFIX}"
  CUR_INIT_MODEL="outputs/checkpoints/${LAST_TAG}/best_model.zip"

  if [[ ! -f "${CUR_INIT_MODEL}" ]]; then
    echo "[Stage2Deep] missing chunk checkpoint: ${CUR_INIT_MODEL}"
    exit 1
  fi

  CHUNK_LOG="outputs/logs/run_${LAST_TAG}.json"
  if [[ -f "${CHUNK_LOG}" ]]; then
    NEXT_LEVEL=$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
p = Path("${CHUNK_LOG}")
lvl = 1
if p.exists():
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    c = d.get("curriculum", {}) if isinstance(d.get("curriculum"), dict) else {}
    try:
        lvl = int(c.get("level", 1))
    except Exception:
        lvl = 1
print(max(1, lvl))
PY
)
    CUR_INIT_LEVEL="${NEXT_LEVEL}"
  fi

  echo "[Stage2Deep] next chunk initial curriculum level=${CUR_INIT_LEVEL}"
  echo "[Stage2Deep] cooldown ${COOLDOWN_SEC}s"
  sleep "${COOLDOWN_SEC}"
done

echo "[Stage2Deep] final evaluation and artifact export"
"${PYTHON_BIN}" src/train_agents.py \
  --method ppo \
  --seed "${SEED}" \
  --reward-scheme C \
  --total-timesteps 1 \
  --n-eval-episodes "${N_EVAL}" \
  --success-threshold 0.8 \
  --use-curriculum true \
  --curriculum-initial-level "${CUR_INIT_LEVEL}" \
  --curriculum-train-fixed-level 0 \
  --curriculum-radius-step 5 \
  --curriculum-success-window 50 \
  --curriculum-levelup-threshold 0.7 \
  --curriculum-smooth-bridge-stages 4 \
  --curriculum-ent-coef 0.02 \
  --curriculum-ent-coef-end 0.001 \
  --curriculum-ent-coef-phases 1 \
  --curriculum-export-level-gifs true \
  --curriculum-gif-levels "1,2,3" \
  --curriculum-gif-fps 6 \
  --eval-deterministic false \
  --init-model-path "${CUR_INIT_MODEL}" \
  --ppo-config "ppo_stage2_transfer_lr1e4.json" \
  --run-tag-suffix "${RUN_SUFFIX}"

METRICS_PATH="outputs/tables/metrics_${RUN_TAG}.csv"
LAYERS_PATH="outputs/tables/metrics_layers_${RUN_TAG}.csv"
LOG_PATH="outputs/logs/run_${RUN_TAG}.json"
REPORT_PATH="outputs/logs/report_${RUN_TAG}.txt"

CHUNK1_LOG="outputs/logs/run_ppo_seed${SEED}_rewardC_curriculum_${RUN_SUFFIX}_chunk1.json"
CHUNK2_LOG="outputs/logs/run_ppo_seed${SEED}_rewardC_curriculum_${RUN_SUFFIX}_chunk2.json"
CHUNK3_LOG="outputs/logs/run_ppo_seed${SEED}_rewardC_curriculum_${RUN_SUFFIX}_chunk3.json"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import pandas as pd

metrics_path = Path("${METRICS_PATH}")
layers_path = Path("${LAYERS_PATH}")
log_path = Path("${LOG_PATH}")
report_path = Path("${REPORT_PATH}")
chunk_logs = [
  Path("${CHUNK1_LOG}"),
  Path("${CHUNK2_LOG}"),
  Path("${CHUNK3_LOG}"),
]

if not metrics_path.exists() or not layers_path.exists() or not log_path.exists():
    raise SystemExit("Missing required outputs for report generation")

m = pd.read_csv(metrics_path)
lm = pd.read_csv(layers_path)
with open(log_path, "r", encoding="utf-8") as f:
    run_log = json.load(f)

std = float(lm.loc[lm["layer"] == "standard_start", "success_rate"].iloc[0])
target = 0.05
passed = std >= target

cur = run_log.get("curriculum", {}) if isinstance(run_log.get("curriculum"), dict) else {}
events = cur.get("level_up_events", []) if isinstance(cur.get("level_up_events"), list) else []
final_level = cur.get("level")

# Aggregate difficulty-transition events across all 3 chunks.
all_events = []
for i, p in enumerate(chunk_logs, start=1):
  if not p.exists():
    continue
  with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
  c = d.get("curriculum", {}) if isinstance(d.get("curriculum"), dict) else {}
  evs = c.get("level_up_events", []) if isinstance(c.get("level_up_events"), list) else []
  for ev in evs:
    if isinstance(ev, dict):
      rec = dict(ev)
      rec["chunk"] = i
      all_events.append(rec)

transition_note = "no level-up events across chunks"
if all_events:
  levels = [int(e.get("new_level", -1)) for e in all_events]
  if 3 in levels and 4 in levels:
    transition_note = "crossed k3->k4"
  elif 3 in levels and 4 not in levels:
    transition_note = "appears stuck around k3->k4"
  elif 2 in levels and 3 not in levels:
    transition_note = "appears stuck around k2->k3"
  elif 2 in levels and 3 in levels:
    transition_note = "progressed across k2->k3"

lines = []
lines.append("Stage2 Deep Generalization Report")
lines.append("run_tag=${RUN_TAG}")
lines.append(f"standard_start_success_rate={std:.4f}")
lines.append(f"target=0.0500")
lines.append(f"pass={passed}")
lines.append(f"overall_success_rate={float(m['success_rate'].iloc[0]):.4f}")
lines.append(f"final_curriculum_level={final_level}")
lines.append(f"final_run_level_up_events_count={len(events)}")
lines.append(f"all_chunks_level_up_events_count={len(all_events)}")
lines.append(f"difficulty_transition_assessment={transition_note}")
lines.append("all_chunk_level_up_events=")
for ev in all_events:
    lines.append(f"  {ev}")

if passed:
    # Export 3 additional long-distance GIFs from standard start.
    from stable_baselines3 import PPO
    from hero_task3_env import HeroTask3Env
    import imageio.v2 as imageio
    import numpy as np

    ckpt = Path(run_log.get("model_artifact_path", ""))
    if ckpt.exists():
        model = PPO.load(str(ckpt), device="cpu")
        fig_dir = Path("outputs/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            seed = int(${SEED}) + 70000 + i
            env = HeroTask3Env(seed=seed, reward_scheme="C", render_mode="rgb_array")
            obs, _ = env.reset(seed=seed)
            frames = []
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _r, t, tr, _info = env.step(int(action))
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))
                done = bool(t or tr)
            env.close()
            out = fig_dir / f"long_navigation_${RUN_TAG}_{i+1}.gif"
            if frames:
                imageio.mimsave(out, frames, fps=6)
                lines.append(f"exported_long_gif={out}")
    else:
        lines.append("pass met but checkpoint missing for long GIF export")

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "[Stage2Deep] report: ${REPORT_PATH}"
