#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ziyanlei/Desktop/AIR/.venv311/bin/python"

cd "$(dirname "$0")/.."

SEED="${1:-42}"
N_EVAL="${2:-100}"
COOLDOWN_SEC="${3:-30}"
CHUNK_STEPS="${4:-50000}"

INIT_CKPT="outputs/checkpoints/ppo_seed${SEED}_rewardC_curriculum_transfer_deep300k_chunk3/best_model.zip"
if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "[Std6Adapt] missing init checkpoint: ${INIT_CKPT}"
  exit 1
fi

echo "[Std6Adapt] chunk1 start"
"${PYTHON_BIN}" src/train_agents.py \
  --method ppo \
  --seed "${SEED}" \
  --reward-scheme C \
  --total-timesteps "${CHUNK_STEPS}" \
  --n-eval-episodes "${N_EVAL}" \
  --success-threshold 0.8 \
  --use-curriculum true \
  --curriculum-initial-level 6 \
  --curriculum-train-fixed-level 0 \
  --curriculum-radius-step 5 \
  --curriculum-success-window 50 \
  --curriculum-levelup-threshold 0.7 \
  --curriculum-smooth-bridge-stages 0 \
  --curriculum-ent-coef 0.01 \
  --curriculum-ent-coef-end 0.001 \
  --curriculum-ent-coef-phases 3 \
  --curriculum-export-level-gifs false \
  --eval-deterministic false \
  --init-model-path "${INIT_CKPT}" \
  --ppo-config "ppo_stage2_transfer_lr1e4.json" \
  --run-tag-suffix "transfer_std6_chunk1"

echo "[Std6Adapt] cooldown ${COOLDOWN_SEC}s"
sleep "${COOLDOWN_SEC}"

CKPT1="outputs/checkpoints/ppo_seed${SEED}_rewardC_curriculum_transfer_std6_chunk1/best_model.zip"
if [[ ! -f "${CKPT1}" ]]; then
  echo "[Std6Adapt] missing chunk1 checkpoint: ${CKPT1}"
  exit 1
fi

echo "[Std6Adapt] chunk2 start"
"${PYTHON_BIN}" src/train_agents.py \
  --method ppo \
  --seed "${SEED}" \
  --reward-scheme C \
  --total-timesteps "${CHUNK_STEPS}" \
  --n-eval-episodes "${N_EVAL}" \
  --success-threshold 0.8 \
  --use-curriculum true \
  --curriculum-initial-level 6 \
  --curriculum-train-fixed-level 0 \
  --curriculum-radius-step 5 \
  --curriculum-success-window 50 \
  --curriculum-levelup-threshold 0.7 \
  --curriculum-smooth-bridge-stages 0 \
  --curriculum-ent-coef 0.01 \
  --curriculum-ent-coef-end 0.001 \
  --curriculum-ent-coef-phases 3 \
  --curriculum-export-level-gifs false \
  --eval-deterministic false \
  --init-model-path "${CKPT1}" \
  --ppo-config "ppo_stage2_transfer_lr1e4.json" \
  --run-tag-suffix "transfer_std6_chunk2"

echo "[Std6Adapt] final evaluation"
CKPT2="outputs/checkpoints/ppo_seed${SEED}_rewardC_curriculum_transfer_std6_chunk2/best_model.zip"
if [[ ! -f "${CKPT2}" ]]; then
  echo "[Std6Adapt] missing chunk2 checkpoint: ${CKPT2}"
  exit 1
fi

"${PYTHON_BIN}" src/train_agents.py \
  --method ppo \
  --seed "${SEED}" \
  --reward-scheme C \
  --total-timesteps 1 \
  --n-eval-episodes "${N_EVAL}" \
  --success-threshold 0.8 \
  --use-curriculum true \
  --curriculum-initial-level 6 \
  --curriculum-train-fixed-level 0 \
  --curriculum-radius-step 5 \
  --curriculum-success-window 50 \
  --curriculum-levelup-threshold 0.7 \
  --curriculum-smooth-bridge-stages 0 \
  --curriculum-ent-coef 0.01 \
  --curriculum-ent-coef-end 0.001 \
  --curriculum-ent-coef-phases 1 \
  --curriculum-export-level-gifs false \
  --eval-deterministic false \
  --init-model-path "${CKPT2}" \
  --ppo-config "ppo_stage2_transfer_lr1e4.json" \
  --run-tag-suffix "transfer_std6"

RUN_TAG="ppo_seed${SEED}_rewardC_curriculum_transfer_std6"
METRICS_PATH="outputs/tables/metrics_${RUN_TAG}.csv"
LAYERS_PATH="outputs/tables/metrics_layers_${RUN_TAG}.csv"
LOG_PATH="outputs/logs/run_${RUN_TAG}.json"
REPORT_PATH="outputs/logs/report_${RUN_TAG}.txt"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import pandas as pd

metrics_path = Path("${METRICS_PATH}")
layers_path = Path("${LAYERS_PATH}")
log_path = Path("${LOG_PATH}")
report_path = Path("${REPORT_PATH}")

m = pd.read_csv(metrics_path)
lm = pd.read_csv(layers_path)
with open(log_path, "r", encoding="utf-8") as f:
    run_log = json.load(f)

std = float(lm.loc[lm["layer"] == "standard_start", "success_rate"].iloc[0])
overall = float(m["success_rate"].iloc[0])
passed = std >= 0.05

lines = []
lines.append("Stage2 Std6 Adapt Report")
lines.append("run_tag=${RUN_TAG}")
lines.append(f"standard_start_success_rate={std:.4f}")
lines.append(f"overall_success_rate={overall:.4f}")
lines.append(f"target=0.0500")
lines.append(f"pass={passed}")
lines.append(f"model_artifact_path={run_log.get('model_artifact_path', '')}")

if passed:
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
            seed = int(${SEED}) + 88000 + i
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

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "[Std6Adapt] report: ${REPORT_PATH}"
