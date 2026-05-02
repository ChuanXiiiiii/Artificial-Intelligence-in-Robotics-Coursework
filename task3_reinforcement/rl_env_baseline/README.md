# RL Baseline Environment Note

This folder is reserved for baseline environment integration notes.

Current recommendation:
- Keep using ../SEMTM0016_DungeonMazeWorld-main/ as the source baseline.
- Avoid editing baseline files directly unless needed.
- If changes are required, document every change and reason in task3_reinforcement/README_task3.md and project_management/decisions.md.
- Stage 6 Reward H and Stage 7 recurrent/DQN experiments still follow the same wrapper-only rule: all stealth, kill-zone, curriculum, and recurrent-policy logic lives under task3_reinforcement/src rather than inside the baseline repo.

Optional workflow:
- Copy a working snapshot here only when you need isolated experiments.
- Keep a clear changelog when snapshot differs from source baseline.
