# Task 3 - Reinforcement Learning

## Goal
Build and compare Random, DQN, PPO, and recurrent PPO for HeroBot navigation in DungeonMazeWorld, while integrating Task1 entity semantics, Task2 cluster features, and controlled reward/observation redesigns.

## Current status snapshot (Update: 2026-04-30)
- The original 30-run matrix over Reward A/B is complete and remains the baseline diagnostic set. All method x reward groups stayed at 0.0 success and are used as failure-analysis evidence rather than final navigation evidence.
- Reward C, E/E2, F, and F2 established the main diagnosis chain: simple potential shaping was too weak, over-penalized C2 collapsed return scale, A* true-path shaping broke zero only with curriculum, and observation hints accelerated curriculum progression without solving long-horizon generalization.
- Reward G froze a fair CNN-based contract: ego-centric 4x7x7 radar, Task2 cluster IDs in the entity channel, braid-maze topology, and explicit wall/stagnation/dead-loop penalties.
- **Stage 6 / Reward H (Stealth)** extended Reward G with dynamic patrol enemies, kill-zone terminal penalty, weapon-state observation channel, and frame stacking. The resulting effective model input is `15x7x7` (`5 channels x frame_stack=3`).
- **Stage 6 PPO v4** (`ppo_seed42_rewardH_curriculum_stage6H_curriculum`) finished 6M steps and reached `success_rate=0.32`, `episode_return_mean=-49.31`, `dead_loop_rate=0.04`, `wall_collision_case_rate=0.02`.
- **Stage 7 "Reward J"** is a naming layer only. The environment physics still use Reward H and `configs/env_stage6_rewardH.json`; the change is architectural and experimental rather than a new reward formula.
- **Stage 7 PPO v5 + LSTM** switched PPO to `sb3-contrib RecurrentPPO` with `CnnLstmPolicy + TinyCNN` (`lstm_hidden_size=256`, `n_lstm_layers=1`) under the same Reward H curriculum. Final seed42 result improved to `success_rate=0.41`, `episode_return_mean=-44.34`, `steps_to_goal_mean=54.90`, `dead_loop_rate=0.15`, `wall_collision_case_rate=0.05`.
- Stage 7 layered evaluation confirms the memory gain is strongest on near-to-mid radii and still non-trivial on standard starts: `k1=0.91`, `k2=0.81`, `k3=0.73`, `standard_start=0.33`.
- Recurrent PPO on Apple MPS exposed an LSTM backward bug, so the recurrent path is intentionally forced onto CPU. Non-recurrent PPO/DQN still follow the normal `mps -> cpu` selection logic.
- **Stage 7 DQN fairness baseline** was launched under the same Reward H physics and the same curriculum schedule so that the final comparison is algorithmic rather than environmental. It is still running at the time of this update, so the final three-way comparison table is intentionally marked pending.
- The current best Task3 policy is therefore `ppo_seed42_rewardH_curriculum_stage7J_lstm/best_model.zip`, with paired marker/sprite GIFs exported from the same successful rollout.

## Input environment
- ../SEMTM0016_DungeonMazeWorld-main/

## Implemented methods
- Baseline: Random Agent
- Value-based: DQN (Stable-Baselines3, `CnnPolicy` + `TinyCNN`)
- Policy-based: PPO (Stable-Baselines3, `CnnPolicy` + `TinyCNN`)
- Recurrent policy-based: Recurrent PPO (sb3-contrib, `CnnLstmPolicy` + `TinyCNN`)

## Reward schemes
- Reward A (sparse): base step penalty from env + goal reward.
- Reward B (dense): Reward A + Manhattan-distance shaping + hostile collision penalty + bribable approach shaping.
- Reward C: Reward A + potential-based goal shaping + explicit wall-hit/dead-loop penalties.
- Reward E2 (A* Path Shaping): Reward A + true shortest-path delta shaping + explicit wall, dead-loop, and stagnation penalties. Requires curriculum learning to bootstrap.
- Reward F (World A* Hint): Reward E2 plus the A* next-step direction `(dx, dy) ∈ {-1, 0, 1}` appended to the observation vector.
- Reward F2 (World + Ego A* Hint): Reward F plus ego-frame one-hot `[is_forward, is_left, is_right]` to remove the world-to-ego mapping burden.
- Reward G (Frozen CNN Contract): ego-centric `4x7x7` radar + Task2 cluster entity channel + braid topology + explicit hostile/wall/loop penalties + dynamic low bribe costs.
- Reward H (Stage 6 Stealth): Reward G plus dynamic patrol enemies, kill-zone terminal penalty, weapon-state channel, and frame stacking. Active config values in `configs/env_stage6_rewardH.json`: `n_virtual_entities=8`, `path_scale_e=0.05`, `step_penalty_e=-0.03`, `kill_zone_penalty_h=-50.0`, `frame_stack=3`, `include_astar_hint=true`, `include_astar_ego_hint=true`.
- Reward J (Stage 7 label): not a separate reward file. It means "use Reward H physics for a controlled Stage 7 comparison", currently covering PPO+LSTM and the matching DQN baseline.

## Seed protocol
- Fixed five-seed set for robustness: 42, 123, 2026, 9, 20

## Logic integration
- Task1 integration: front-view entity prediction adapter (`src/entity_inference.py`) with hostile/bribable mapping.
- Task2 integration: sensor feature adapter (`src/sensor_adapter.py`) and cluster-id augmentation from Task2 outputs.
- Environment extension policy: wrapper-based extension only (`src/hero_task3_env.py`); the baseline source remains untouched.
- Topology extension: braid-maze post-processing in wrapper (`_carve_braid_loops`) guarantees five genuine loop additions after reset.
- Observation extension:
	- Reward G uses ego-centric `4x7x7` radar.
	- Reward H uses `5x7x7` radar, where channel 4 stores weapon state.
	- Frame stacking is handled at the vectorized-env level, so the effective network input for Reward H defaults to `15x7x7`.
	- Channel 1 always encodes Task2 cluster id + 1 for entity cells.

## Required metrics
- Success rate
- Episode return
- Steps to goal
- Sample efficiency (episodes to threshold)

Additional diagnostics:
- Dead-loop rate
- Wall-collision case rate
- Reflection case tables (`reflection_cases_*.csv`)
- Layered curriculum eval tables (`metrics_layers_*.csv`, `eval_rows_layers_*.csv`)

## Folder usage
- rl_env_baseline/: notes about how baseline environment constraints are preserved
- notebooks/: training and evaluation notebooks
- src/: agents, trainers, evaluation helpers, and rendering tools
- configs/: algorithm and environment configs
- outputs/figures/: learning curves, comparison plots, trajectory GIFs
- outputs/tables/: aggregate scores across seeds/settings
- outputs/checkpoints/: saved policies
- outputs/logs/: training and evaluation logs

## Main scripts
- `src/train_agents.py`: single-run training/evaluation entry for random, DQN, PPO, and recurrent PPO
- `src/aggregate_multi_seed.py`: multi-seed aggregation and plots (mean with 95% CI)
- `src/export_best_trajectories.py`: export best trajectories as GIF/MP4 for completed runs
- `src/render_stage6_gif.py`: render Reward H / Stage 7 checkpoints as marker-style and/or sprite-style GIFs; supports recurrent inference and paired rollout export
- `src/sync_report_assets.py`: sync Task3 figures/tables into `../docs/report/`
- `src/run_task3.sh`: one method/seed/reward run
- `src/run_task3_multi_seed.sh`: thermal-safe batch queue for earlier A/B/F2/G experiments
- `src/run_task3_final_curriculum.sh`: frozen Reward G PPO curriculum launcher
- `src/run_task3_stage6_curriculum.sh`: Stage 6 PPO launcher for Reward H
- `src/run_task3_stage7_lstm.sh`: Stage 7 recurrent PPO launcher (`Reward H` physics + LSTM memory)
- `src/run_task3_dqn_curriculum.sh`: Stage 7 DQN fairness-baseline launcher under the same Reward H curriculum

## Configuration files
- `configs/experiment.json`: global method/seed/reward/timestep contract
- `configs/env.json`: legacy wrapper environment and reward parameters for early experiments
- `configs/env_final_rewardG.json`: frozen Reward G environment contract
- `configs/env_stage6_rewardH.json`: Stage 6 stealth environment + Reward H parameters
- `configs/dqn.json`: generic DQN hyperparameters with long-run decay
- `configs/dqn_stage7_baseline.json`: Stage 7 DQN fairness-baseline config (`buffer 100k->50k`, `exploration_fraction=0.15`, `exploration_final_eps=0.03`, `target_update_interval=10000`)
- `configs/ppo.json`: PPO hyperparameters with LR decay and entropy-phase controls

## Runtime and device
- Python: 3.11 (`/Users/ziyanlei/Desktop/AIR/.venv311/bin/python`)
- Core RL packages: `stable-baselines3 2.8.0`, `sb3-contrib 2.8.0`, `tensorboard`
- Default device policy: prefer `mps`, fallback to `cpu` (selected in `src/common.py`)
- Recurrent PPO exception: Stage 7 LSTM runs force `cpu` because the MPS backend hits a known LSTM backward assertion in `GPURNNOps.mm`

## Quick start
Install dependencies:

```bash
cd task3_reinforcement
/Users/ziyanlei/Desktop/AIR/.venv311/bin/pip install -r requirements_task3.txt
```

Legacy single-run example:

```bash
bash src/run_task3.sh 42 dqn B 1000000
```

Frozen Reward G main-line run:

```bash
bash src/run_task3_final_curriculum.sh 42 2000000 100 finalG_curriculum
```

Stage 6 Reward H PPO run:

```bash
bash src/run_task3_stage6_curriculum.sh 42
```

Stage 7 recurrent PPO run:

```bash
bash src/run_task3_stage7_lstm.sh 42 6000000 100 stage7J_lstm
```

Stage 7 DQN fairness baseline:

```bash
bash src/run_task3_dqn_curriculum.sh 42 6000000 100 stage7J_dqn_baseline
```

Render paired marker/sprite GIFs for the v5 LSTM checkpoint:

```bash
/Users/ziyanlei/Desktop/AIR/.venv311/bin/python src/render_stage6_gif.py \
	--checkpoint outputs/checkpoints/ppo_seed42_rewardH_curriculum_stage7J_lstm/best_model.zip \
	--env-config env_stage6_rewardH.json \
	--method ppo \
	--use-recurrent true \
	--render-style markers \
	--paired-render-style sprites \
	--seed 42 \
	--max-seed-attempts 25 \
	--output-path outputs/figures/stage7_lstm_standard_markers.gif \
	--paired-output-path outputs/figures/stage7_lstm_standard_sprites.gif
```

Checkpoint eval-only (non-deterministic) example from the older F2+Cnn diagnostic line:

```bash
/Users/ziyanlei/Desktop/AIR/.venv311/bin/python src/train_agents.py --method ppo --seed 42 --reward-scheme F2 --env-config env_stage5_rewardF2.json --eval-only true --resume-from outputs/checkpoints/ppo_seed42_rewardF2_curriculum_f2_cnn_curriculum_2m/best_model.zip --eval-deterministic false --n-eval-episodes 100 --use-curriculum true --curriculum-initial-level 1 --curriculum-radius-step 3 --curriculum-success-window 50 --curriculum-levelup-threshold 0.6 --curriculum-smooth-bridge-stages 4 --run-tag-suffix eval_only_nondet
```

## Output contract
- Per-run metrics: `outputs/tables/metrics_<method>_seed<seed>_reward<scheme>.csv`
- Per-run curves: `outputs/tables/curve_<method>_seed<seed>_reward<scheme>.csv`
- Per-run optimization logs: `outputs/tables/loss_<method>_seed<seed>_reward<scheme>.csv`
- Per-run full progress logs: `outputs/tables/progress_<method>_seed<seed>_reward<scheme>.csv`
- Per-run model artifact: `outputs/checkpoints/<method>_seed<seed>_reward<scheme>/best_model.zip`
- Reflection cases: `outputs/tables/reflection_cases_<method>_seed<seed>_reward<scheme>.csv`
- Layered curriculum metrics: `outputs/tables/metrics_layers_<method>_seed<seed>_reward<scheme>*.csv`
- Layered curriculum eval rows: `outputs/tables/eval_rows_layers_<method>_seed<seed>_reward<scheme>*.csv`
- Summary figures: `outputs/figures/learning_curve_*.png`, `outputs/figures/metrics_multi_seed_summary.png`
- Trajectory visuals: `outputs/figures/best_trajectory_<method>_reward<scheme>.gif`, `outputs/figures/stage7_lstm_standard_markers.gif`, `outputs/figures/stage7_lstm_standard_sprites.gif`

## Latest artifacts worth citing
- Stage 6 PPO v4 best model: `outputs/checkpoints/ppo_seed42_rewardH_curriculum_stage6H_curriculum/best_model.zip`
- Stage 7 PPO v5 LSTM best model: `outputs/checkpoints/ppo_seed42_rewardH_curriculum_stage7J_lstm/best_model.zip`
- Stage 7 paired GIFs: `outputs/figures/stage7_lstm_standard_markers.gif`, `outputs/figures/stage7_lstm_standard_sprites.gif`
- Stage 7 DQN baseline log: `outputs/logs/stage7_dqn_seed42_6M_baseline.log`

## Notebook
- Experiment journal template: `notebooks/task3_experiment_journal.ipynb`
