# Task 1 - Supervised Species Recognition

## Scope and responsibility
This Task 1 workspace is only for implementation, model training, evaluation, visualization, and artifact export.
Report writing is out of scope.

## Data source
- Use only: ../dungeon_images_colour80/
- Label space: halfling, human, lizard, orc, wingedrat
- Task type: single-label multi-class classification

## Runtime baseline
- Python version: 3.11.x
- Recommended interpreter path pattern: ../.venv311/bin/python

## Required methods
At least two supervised methods must be implemented and compared fairly.

- Method A: HOG + Linear SVM
- Method B: ResNet18 transfer learning

Method A tuned solver baseline:

- max_iter=30000
- dual=False
- tol=1e-4

## Fair comparison protocol
Both methods must use:

- The same train/val/test split manifest
- The same random seed policy
- The same metric definitions and evaluator
- The same held-out test split

Recommended split:

- Group-stratified split by class + filename prefix group
- Robust setting: bucket-based grouping with minimum validation/test support per class
- Ratio: 70/15/15 (train/val/test)
- Seed set: 42 as main run, 123 and 2026 for stability check

Recommended split command:

- ../.venv311/bin/python src/build_split_manifest.py --seed 42 --bucket-size 20 --min-val-support 30 --min-test-support 30

## Required metrics and analysis
For each method, export:

- Accuracy
- Macro-F1
- Confusion matrix (raw counts)
- Confusion matrix (row-normalized)

Error analysis is mandatory:

- Per-class precision/recall/F1
- Typical misclassification pairs
- Long-tail class error behavior (especially wingedrat)

## Directory contract
- notebooks/: VS Code Journal notebooks for experiment records and run notes
- src/: reusable scripts for split, train, evaluate, visualize, sync
- configs/: run config files
- outputs/figures/: local figures for Task1 runs
- outputs/tables/: local tables for Task1 runs
- outputs/models/: saved model weights and checkpoints
- outputs/logs/: logs and run summaries

## Report-material sync rules
All figures and tables are first generated under outputs, then synced to report folders:

- Figure source: outputs/figures/
- Figure report copy: ../docs/report/figures/
- Table source: outputs/tables/
- Table report copy: ../docs/report/tables/

Keep names stable and versioned, for example:

- cm_hog_svm_seed42.png
- cm_resnet18_seed42.png
- metrics_compare_main.csv

## Progress tracking rule
After every meaningful run, update:

- ../project_management/progress_tracker.md

Suggested run-level update fields:

Date | Task | What was done | Key result | Blocker | Next step
---- | ---- | ------------- | ---------- | ------- | ---------
YYYY-MM-DD | Task1 | Method A/B run with seed | Acc and Macro-F1 | blocker if any | next action

## Standard implementation order
1. Build split manifest
2. Train and evaluate Method A
3. Train and evaluate Method B
4. Generate figures and tables
5. Sync report-material copies
6. Update progress tracker

Multi-seed aggregation command:

- ../.venv311/bin/python src/aggregate_multi_seed.py --seeds 42,123,2026

Convergence before/after comparison for Method A:

- Before: ../.venv311/bin/python src/train_hog_svm.py --seed 42 --max-iter 8000 --svm-dual true --svm-tol 1e-4
- After: ../.venv311/bin/python src/train_hog_svm.py --seed 42 --max-iter 30000 --svm-dual false --svm-tol 1e-4
- Build 3-seed comparison table: ../.venv311/bin/python src/make_hog_comparison.py

Current conclusion snapshot (2026-04-11):

- Across seeds 42/123/2026, Method A warning count is not fully eliminated after tuning (total 2 -> 2), but mean performance is stable or slightly better (Accuracy 0.9641 -> 0.9652, Macro-F1 0.9643 -> 0.9663).
- For Task1 scope, this supports the statement that ConvergenceWarning does not change the final model-comparison conclusion.
- No further max_iter=50000 expansion is planned in the current iteration cycle.

## Output checklist
- Two implemented methods with fair comparison
- Accuracy and Macro-F1 for both methods
- Confusion matrices for both methods
- Error analysis tables and examples
- Figures and tables in outputs and synced copies in docs/report
