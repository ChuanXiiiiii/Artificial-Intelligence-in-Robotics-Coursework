# Task3 RL Result Assets

This folder receives synced Task3 materials for final report authoring.

- Figures: docs/report/figures
- Tables: docs/report/tables
- Seeds: 42, 123, 2026, 9, 20
- Methods: Random, DQN, PPO
- Reward schemes: A (sparse), B (dense heuristic)

## Retuning checkpoint (2026-04-19, quick matrix)
- Added Reward C (potential-based shaping) and exploration-strengthened settings.
- Quick matrix scope: seed42, methods={DQN,PPO}, rewards={A,B,C}, total_timesteps=200k, eval_episodes=100.
- Break-zero result: 0/6 combinations reached non-zero success rate.

Per-group metrics snapshot:
- DQN-A: success_rate=0.0, episode_return_mean=-1.000
- DQN-B: success_rate=0.0, episode_return_mean=-2.306
- DQN-C: success_rate=0.0, episode_return_mean=-1.000
- PPO-A: success_rate=0.0, episode_return_mean=-3.240
- PPO-B: success_rate=0.0, episode_return_mean=-2.306
- PPO-C: success_rate=0.0, episode_return_mean=-3.064

Failure-mode audit highlights:
- Best-trajectory replay classification shows SPIN_NEAR_START and wall-heavy stagnation dominate.
- LOOP_TWO_CELL is not the primary mode in this version.
- For 200k eval rows (seed42): DQN-A/C are nearly all SPIN_NEAR_START; PPO A/B/C remain mixed stagnation (SPIN_NEAR_START + DEEP_WALL_CRASH + OTHER_STAGNATION).

Key interpretation:
- Potential-only shaping with small scale did not break the local optimum.
- Next iteration should couple stronger potential shaping with explicit anti-wall and anti-stagnation penalties.

## C2 checkpoint (2026-04-19, post-fix rerun)
- C2 change set: stronger Reward C potential scale plus explicit wall-hit/dead-loop penalties.
- PPO implementation fix: replaced unsupported callable `ent_coef` with phase-wise float updates to recover SB3 compatibility.
- Re-run scope: seed42, methods={DQN,PPO}, rewards={A,B,C}, total_timesteps=200k, eval_episodes=100.
- Break-zero result: still 0/6.

Per-group metrics snapshot (C2):
- DQN-A: success_rate=0.0, episode_return_mean=-1.000
- DQN-B: success_rate=0.0, episode_return_mean=-2.306
- DQN-C: success_rate=0.0, episode_return_mean=-100.900
- PPO-A: success_rate=0.0, episode_return_mean=-3.240
- PPO-B: success_rate=0.0, episode_return_mean=-2.306
- PPO-C: success_rate=0.0, episode_return_mean=-101.581

Delta vs C1:
- Largest degradation: DQN-C (delta episode_return_mean = -99.900), PPO-C (delta = -98.517).
- No group shows success-rate improvement.

Current interpretation:
- C2 penalties are too strong and dominate optimization without improving goal-reaching behavior.
- Next iteration (C3) should rebalance shaping magnitude and apply penalties only for repeated ineffective collisions/stalls.

## Full-matrix completion snapshot (2026-04-18)
- Full run status: completed (30/30 combinations).
- Strict export status: passed.
- Aggregation status: passed (`metrics_multi_seed_summary.csv` generated).

## Aggregated highlights
- Reward A: Random=-201.8620, DQN=-1.0000, PPO=-2.9600 (episode_return_mean)
- Reward B: Random=-202.5044, DQN=-1.5940, PPO=-1.9244 (episode_return_mean)
- Success rate means: all method x reward groups are 0.0 in this run set.

## Interpretation note
- This run set is complete and reproducible, but it does not yet demonstrate successful goal-reaching policies.
- Use these results as baseline diagnostics and support follow-up tuning experiments before making optimal-path claims.

## Latest controlled results after Reward G freeze

The active post-freeze comparison family is no longer the old A/B/C matrix. The current controlled line is Reward H physics with three algorithmic views:

| Setting | Algorithm | Environment / Control | success_rate | return_mean | dead_loop_rate | wall_collision_case_rate | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Stage 6 v4 | PPO | Reward H stealth, CNN, non-recurrent | 0.32 | -49.31 | 0.04 | 0.02 | complete |
| Stage 7 v5 | PPO + LSTM | Reward H stealth, same curriculum, recurrent memory | 0.41 | -44.34 | 0.15 | 0.05 | complete |
| Stage 7 baseline | DQN | Reward H stealth, same curriculum, no LSTM | TBD | TBD | TBD | TBD | running |

Current interpretation:
- Reward H is the first Task3 line that yields a stable non-zero standard-start PPO policy under the CNN pipeline.
- Adding recurrent memory on top of the same Reward H physics improves success from `0.32` to `0.41` on seed42, which is the clearest evidence so far that partial observability is part of the bottleneck.
- The memory gain is not free: v5 solves more episodes but takes longer paths and shows higher dead-loop/wall-contact rates, so the final report should describe it as a success-vs-control tradeoff rather than a uniform dominance claim.
- The fair DQN conclusion must wait for the Stage 7 baseline to finish because the comparison is only valid when all three methods share Reward H physics and the same curriculum schedule.

## Stage 7 layered PPO+LSTM evaluation (seed42)

| Layer | success_rate | steps_mean | dead_loop_rate | wall_collision_case_rate |
| --- | ---: | ---: | ---: | ---: |
| k1 | 0.91 | 4.67 | 0.00 | 0.00 |
| k2 | 0.81 | 10.07 | 0.01 | 0.01 |
| k3 | 0.73 | 16.15 | 0.03 | 0.02 |
| standard_start | 0.33 | 36.26 | 0.11 | 0.08 |

Interpretation:
- Recurrent memory strongly helps short-to-mid-range navigation and still transfers partially to the standard start condition.
- The residual gap between `k3` and `standard_start` shows that long-horizon planning remains the dominant unresolved problem.

## Latest report-ready visuals

- Marker overlay GIF: `task3_reinforcement/outputs/figures/stage7_lstm_standard_markers.gif`
- Sprite overlay GIF: `task3_reinforcement/outputs/figures/stage7_lstm_standard_sprites.gif`

Both GIFs are exported from the same successful Stage 7 LSTM rollout so they can be used as paired visual evidence in the report.
