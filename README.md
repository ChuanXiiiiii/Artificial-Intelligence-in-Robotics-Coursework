# Artificial Intelligence in Robotics

This repository contains the working materials for the SEMTM0016 DungeonMazeWorld coursework. It combines three task pipelines, the shared project management notes, the raw datasets/assets, and the final integrated report notebook.

## What is here

- Task 1: supervised image classification of dungeon species
- Task 2: unsupervised clustering of dungeon sensor statistics
- Task 3: reinforcement learning for maze navigation and stealth
- Final submission notebook: `final_report.ipynb`
- Project planning and risk tracking: `project_management/`
- Report assets and submission staging: `docs/`
- Raw coursework assets and datasets: `SEMTM0016_DungeonMazeWorld-main/`, `dungeon_images_colour80/`, `dungeon_sensorstats.csv`

## Repository layout

```text
AIR/
├── task1_supervised/
├── task2_unsupervised/
├── task3_reinforcement/
├── SEMTM0016_DungeonMazeWorld-main/
├── docs/
├── project_management/
├── final_report.ipynb
├── final_report.html
├── final_report.pdf
└── README.md
```

## Environment

The canonical Python environment for this workspace is:

- Python 3.11.15
- Interpreter: `/Users/ziyanlei/Desktop/AIR/.venv311/bin/python`

If you need to recreate the environment for the task folders, install the relevant requirements file in the matching subdirectory.

## Quick start

### Task 1

```bash
cd task1_supervised
/Users/ziyanlei/Desktop/AIR/.venv311/bin/pip install -r requirements_task1.txt
bash src/run_task1.sh
```

### Task 2

```bash
cd task2_unsupervised
/Users/ziyanlei/Desktop/AIR/.venv311/bin/pip install -r requirements_task2.txt
bash src/run_task2.sh 42 5 preprocessed_main
```

### Task 3

```bash
cd task3_reinforcement
/Users/ziyanlei/Desktop/AIR/.venv311/bin/pip install -r requirements_task3.txt
bash src/run_task3_stage6_curriculum.sh 42
bash src/run_task3_stage7_lstm.sh 42 6000000 100 stage7J_lstm
bash src/run_task3_dqn_curriculum.sh 42 6000000 100 stage7J_dqn_baseline
```

### Final report

The submission notebook is [`final_report.ipynb`](final_report.ipynb). It integrates the three task reports into a single narrative and can be exported to PDF once the notebook outputs are up to date.

## Data and generated artifacts

Keep these raw assets in place:

- `SEMTM0016_DungeonMazeWorld-main/`
- `dungeon_images_colour80/`
- `dungeon_sensorstats.csv`

Generated experiment outputs are stored inside each task's `outputs/` directory. These include figures, tables, logs, model checkpoints, and GIFs. They are intentionally excluded from version control by `.gitignore`.

Common generated files you may see locally:

- `task1_supervised/outputs/`
- `task2_unsupervised/outputs/`
- `task2_unsupervised/outputs_bribe/`
- `task3_reinforcement/outputs/`
- `final_report.html`
- `final_report.pdf`

## Notes for submission

- Use the task subdirectories for experimentation and artifact generation.
- Keep the final report notebook at the repository root.
- Sync only the figures, tables, and report files you actually want to submit.
- Avoid committing large model checkpoints, logs, and cache directories.

## Project management

Useful planning files live in:

- `project_management/decisions.md`
- `project_management/progress_tracker.md`
- `project_management/risks.md`

## Recommended next steps

1. Review the task README files before running experiments.
2. Rebuild the needed task outputs if you want the final report to reflect new results.
3. Export `final_report.ipynb` to PDF after the notebook has been fully executed.
