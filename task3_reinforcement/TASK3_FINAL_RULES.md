# Task 3 Final Frozen Environment Rules

Update: 2026-04-30

This file defines the frozen Reward G contract that was used as the controlled reference point before the later stealth and memory extensions. Earlier A/B/C/E/F/F2 runs remain design-history and failure-diagnosis evidence. Stage 6 Reward H and Stage 7 "Reward J" are documented below as explicit post-freeze extensions on top of this contract.

## Map And Perception

- Grid: 16x16 DungeonMazeWorld maze.
- Topology: after the baseline DFS single-path maze is generated, the wrapper carves walls until the passable-cell graph gains exactly 5 loops (cycle-rank +5). Candidate openings that do not add a real loop are rejected, which avoids "opened but still no new ring" corner cases.
- Observation: ego-centric 4x7x7 `uint8` CNN radar. The top row of the matrix is always the HeroBot's forward direction.
- Channel 0: terrain and boundary layer. Walls and out-of-bounds cells are encoded as 1; free cells are 0.
- Channel 1: Task2 entity-cluster radar. Empty cells are 0; entity cells are encoded as cluster id + 1, producing values 1-6 for six K-Means clusters.
- Channel 2: A* true-path downhill mask. Local cells with true shortest-path distance no worse than the current robot cell are encoded as 1.
- Channel 3: breadcrumb memory. Recently visited local cells are encoded as 1 to help the policy escape repeated lock states.

## Entity Interaction

Entities are spawned as Task1 species labels, mapped to their Task2 cluster id, and then routed through cluster-level faction sets. If a cluster is mixed across hostile and bribable prototypes, the rule falls back to the original species semantics to avoid ambiguous punishment.

- Hostile species: `orc`, `lizard`, `wingedrat`.
- Bribable / neutral species: `human`, `halfling`.
- On contact, the entity is removed from the map to avoid repeated-cost loops.

Reward G interaction scores:

- Hostile contact: `-6.0`. This is large enough that a short loop detour should be preferable to walking through a dangerous entity.
- Bribable contact: dynamic cost from Task2-style physical features, using species prototype `weight` and `height`; range `-0.08` to `-0.35`. This is intentionally small so crossing a neutral entity can be better than a long detour.

## Reward G

Reward G is the frozen final reward scheme.

- Base environment step cost: inherited from DungeonMazeWorld (`-1` for valid actions, `0` for blocked forward collision).
- True-path progress shaping: `+1.2 * (previous_true_distance - current_true_distance)`.
- Additional step penalty: `-0.03` per step under path-shaping schemes.
- Wall hit: `-1.0` when `move_forwards` fails against a wall.
- Short stagnation: `-0.25` when the recent 5-position window has only 1-2 unique cells.
- Dead loop: `-1.5` when the 24-position window has 5 or fewer unique cells.
- Target reached: `+100.0`, episode success and termination.

## Termination

- Success: the HeroBot reaches the target cell.
- Truncation: max episode length reaches 256 steps.

## Main Training Command For The Frozen Reward G Line

Use the frozen final curriculum launcher:

```bash
cd /Users/ziyanlei/Desktop/AIR/task3_reinforcement
bash src/run_task3_final_curriculum.sh 42 2000000 100 finalG_curriculum
```

The launcher uses Python 3.11 from `/Users/ziyanlei/Desktop/AIR/.venv311/bin/python`, Reward G, `env_final_rewardG.json`, PPO, TinyCNN, curriculum learning, mixed standard starts, and deterministic evaluation.

## Post-Freeze Extension: Stage 6 Reward H (Stealth)

Reward H keeps the braid-maze topology, Task1/Task2 integration, and local A* / breadcrumb logic from the frozen Reward G line, but intentionally changes the environment dynamics:

- Observation grows from `4x7x7` to `5x7x7`; channel 4 encodes weapon state.
- Frame stacking is enabled with `frame_stack=3`, so the effective CNN input becomes `15x7x7`.
- Dynamic patrol enemies create moving kill zones.
- A kill-zone encounter is terminal with `kill_zone_penalty_h=-50.0`.
- Reward shaping keeps the small-step regime used for stealth navigation: `path_scale_e=0.05`, `step_penalty_e=-0.03`.

Stage 6 command:

```bash
cd /Users/ziyanlei/Desktop/AIR/task3_reinforcement
bash src/run_task3_stage6_curriculum.sh 42
```

Stage 6 seed42 result after 6M steps:

- PPO v4 success rate: `0.32`
- Episode return mean: `-49.31`
- Dead-loop rate: `0.04`
- Wall-collision case rate: `0.02`

## Post-Freeze Extension: Stage 7 "Reward J" (Architecture-Controlled Comparison)

Reward J is not a new reward formula. It means "hold Reward H physics fixed and compare algorithms under the same curriculum".

- PPO branch: `sb3-contrib RecurrentPPO` with `CnnLstmPolicy + TinyCNN`, `lstm_hidden_size=256`, `n_lstm_layers=1`.
- DQN branch: same Reward H environment and same curriculum schedule, but without recurrence.
- Runtime exception: recurrent PPO is forced to CPU because the Apple MPS backend hits an LSTM backward assertion during training.

Stage 7 recurrent PPO command:

```bash
cd /Users/ziyanlei/Desktop/AIR/task3_reinforcement
bash src/run_task3_stage7_lstm.sh 42 6000000 100 stage7J_lstm
```

Stage 7 DQN fairness baseline command:

```bash
cd /Users/ziyanlei/Desktop/AIR/task3_reinforcement
bash src/run_task3_dqn_curriculum.sh 42 6000000 100 stage7J_dqn_baseline
```

Current Stage 7 status:

- PPO v5 + LSTM success rate: `0.41`
- PPO v5 layered eval: `k1=0.91`, `k2=0.81`, `k3=0.73`, `standard_start=0.33`
- DQN baseline: still running; final three-way comparison is intentionally pending until convergence

Paired visual exports from the same successful v5 rollout:

- Marker style: `outputs/figures/stage7_lstm_standard_markers.gif`
- Sprite style: `outputs/figures/stage7_lstm_standard_sprites.gif`
