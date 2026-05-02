# Task3 Execution Protocol (2026-04-18)

## Scope
This note records the implemented Task3 RL execution chain, official run standard, and completion outcome for the full 30-run matrix.

## Implemented chain
- Environment wrapper and reward integration are implemented in `task3_reinforcement/src/hero_task3_env.py`.
- Training/evaluation entry is implemented in `task3_reinforcement/src/train_agents.py`.
- Aggregation and CI plotting are implemented in `task3_reinforcement/src/aggregate_multi_seed.py`.
- Best-trajectory export is implemented in `task3_reinforcement/src/export_best_trajectories.py`.
- Batched thermal-safe orchestrator is implemented in `task3_reinforcement/src/run_task3_multi_seed.sh`.

## Hardware and runtime policy
- Target hardware: MacBook Air M4 (fanless).
- Runtime policy: enable torch MPS when available, fallback to CPU.
- Thermal policy: run queue in small parallel batches (2 by default), with 30-second cooldown between batches.

## Strict reproducibility policy
- Official full-matrix runs must enable strict trajectory export.
- If any required metrics or model artifact is missing, the pipeline must fail instead of skipping silently.

## Required per-run artifacts
For each run tag `<method>_seed<seed>_reward<scheme>`:
- `outputs/tables/metrics_<run_tag>.csv`
- `outputs/tables/curve_<run_tag>.csv`
- `outputs/tables/progress_<run_tag>.csv`
- `outputs/tables/loss_<run_tag>.csv`
- `outputs/tables/eval_rows_<run_tag>.csv`
- `outputs/tables/reflection_cases_<run_tag>.csv`
- `outputs/checkpoints/<run_tag>/best_model.zip`
- `outputs/logs/run_<run_tag>.json`

## Official full-matrix command
```bash
cd task3_reinforcement
bash src/run_task3_multi_seed.sh 1000000 2 30 true
```

## Matrix definition
- Methods: random, dqn, ppo
- Rewards: A, B
- Seeds: 42, 123, 2026, 9, 20
- Total runs: 30

## Execution outcome (2026-04-18)
- Full matrix command executed successfully with strict mode enabled.
- 30/30 runs completed and passed required artifact checks.
- Aggregation, strict best-trajectory export, and report sync completed in one pipeline run.
- Aggregate summary indicates `success_rate_mean=0.0` across all methods under both reward schemes in this configuration.

## Notes for report writing
- 30/30 completion condition is satisfied for this run set.
- Use aggregated mean and CI results from `metrics_multi_seed_summary.csv`.
- Use strict-export trajectory outputs as visual evidence only after strict checks pass (satisfied for this run).

## Addendum (2026-04-19 quick retuning protocol)
- Quick matrix scope: methods={dqn, ppo}, rewards={A, B, C}, seeds={42}, total timesteps=200k.
- C1 result: 0/6 break-zero.
- C2 result (stronger Reward C penalties): 0/6 break-zero and severe Reward C return degradation (about -101 mean return for both dqn/ppo).
- Implementation note: SB3 PPO requires numeric `ent_coef`; callable schedule caused runtime `TypeError` and was replaced by phase-wise float updates.
- Operational conclusion: keep this as failed tuning evidence and move to rebalanced C3 design before any new 1M multi-seed expansion.

## Addendum (2026-04-27 Stage 6 Reward H stealth protocol)
- Reward H is the first controlled post-Reward-G extension: dynamic patrol enemies, kill-zone terminal penalty, weapon-state observation channel, and `frame_stack=3`.
- Active environment config: `task3_reinforcement/configs/env_stage6_rewardH.json`.
- Official launcher:

```bash
cd task3_reinforcement
bash src/run_task3_stage6_curriculum.sh 42
```

- Result snapshot for seed42 after 6M steps: `success_rate=0.32`, `episode_return_mean=-49.31`, `dead_loop_rate=0.04`, `wall_collision_case_rate=0.02`.
- Interpretation: the stealth redesign finally produced a materially non-zero standard-start policy under the CNN pipeline, but partial observability still limited robustness.

## Addendum (2026-04-30 Stage 7 recurrent PPO and fair DQN baseline)
- Stage 7 keeps Reward H physics fixed and changes only the algorithmic surface.
- PPO branch switched to `sb3-contrib RecurrentPPO` with `CnnLstmPolicy + TinyCNN` (`lstm_hidden_size=256`, `n_lstm_layers=1`).
- Because Apple MPS hits an LSTM backward assertion for this path, recurrent training is forced to CPU while non-recurrent paths keep the normal `mps -> cpu` policy.
- Stage 7 recurrent launcher:

```bash
cd task3_reinforcement
bash src/run_task3_stage7_lstm.sh 42 6000000 100 stage7J_lstm
```

- Seed42 recurrent PPO result after 6M steps: `success_rate=0.41`, `episode_return_mean=-44.34`, `steps_to_goal_mean=54.90`, `dead_loop_rate=0.15`, `wall_collision_case_rate=0.05`.
- Layered recurrent eval: `k1=0.91`, `k2=0.81`, `k3=0.73`, `standard_start=0.33`.
- Matching DQN fairness baseline uses the same Reward H environment and the same curriculum schedule, with `configs/dqn_stage7_baseline.json` and the launcher below:

```bash
cd task3_reinforcement
bash src/run_task3_dqn_curriculum.sh 42 6000000 100 stage7J_dqn_baseline
```

- DQN baseline is still running at the time of this note, so the final v4 PPO vs v5 PPO+LSTM vs DQN comparison is intentionally deferred until convergence.

## Addendum (2026-04-30 visualization protocol)
- `task3_reinforcement/src/render_stage6_gif.py` now supports recurrent inference and paired exports from the same rollout.
- Latest paired artifacts:
	- `task3_reinforcement/outputs/figures/stage7_lstm_standard_markers.gif`
	- `task3_reinforcement/outputs/figures/stage7_lstm_standard_sprites.gif`
