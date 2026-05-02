# AIR Workspace Organization Guide

## Raw assets (keep in place)
- SEMTM0016_2526_AssessmentOverview.pdf
- SEMTM0016_2526_assignment.pdf
- SEMTM0016_DungeonMazeWorld-main/
- dungeon_images_colour80/
- dungeon_sensorstats.csv

Do not rename or move these raw assets during experiments.

## Task directories
- task1_supervised/: supervised learning experiments (image classification)
- task2_unsupervised/: unsupervised learning experiments (sensor clustering)
- task3_reinforcement/: reinforcement learning experiments (maze navigation)

## Reporting and submission
- docs/report/: final figures, tables, references used in report writing
- docs/submission/final_report_pdf/: final PDF versions
- docs/submission/supporting_zip_staging/: staging area before creating submission zip

## Project management
- project_management/progress_tracker.md: weekly status and completion
- project_management/decisions.md: major design/algorithm decisions with reasons
- project_management/risks.md: known risks and mitigation

## Naming convention
- Notebook: tX_mY_topic_vNN.ipynb
- Script: tX_component_action.py
- Figure: fig_tX_mY_topic.png
- Table: tab_tX_topic.csv
- Model/checkpoint: mdl_tX_mY_epochNNN_seedS.ext or ckpt_tX_mY_epNNN_seedS.ext
- Log: log_tX_mY_YYYYMMDD_HHMM.txt

## Reproducibility checklist
- Fixed random seeds are recorded in configs and logs.
- Data split strategy is documented in each task README.
- Every report figure/table has a source notebook or script path.
- Final results are copied to docs/report/ before report writing.
