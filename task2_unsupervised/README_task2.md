# Task 2 - Unsupervised Learning

## Goal
Build and compare two unsupervised methods for clustering entities from sensor/stat features.

## Current implementation status (2026-04-14)
- Implemented end-to-end Task2 pipeline with shared preprocessing, K-Means, GMM, fair comparison, visualization, and multi-seed aggregation.
- Completed multi-seed run for seeds 42, 123, and 2026 with fixed-k fairness view (k=5) and method-best view.
- Main experiment uses feature set without `species` and without `bribe` (high-risk feature reserved for ablation).

## Input data
- ../dungeon_sensorstats.csv

## Suggested methods
- Method 1: K-Means
- Method 2: Gaussian Mixture Model (GMM)

## Fair comparison protocol
- Shared preprocessing artifact for both methods.
- Shared search range (k/components = 2..8).
- Shared seeds (42, 123, 2026).
- Shared model selection rule: sort by Silhouette desc, Davies-Bouldin asc, Calinski-Harabasz desc.
- Dual-view reporting:
	- `method_best`: each method uses its own best cluster count.
	- `fixed_same_k`: both methods are compared at fixed k=5.

## Why k can be 6 when there are 5 species
Task2 is unsupervised clustering. The algorithm does not use species labels during training, so the best internal cluster count does not have to equal the number of known species. In this project, we report both `method_best` (often k=6) and `fixed_same_k=5` to align with assignment semantics and preserve fairness.

## Required metrics
- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz index
- Optional interpretation metric: ARI/NMI (for post-hoc analysis)

## Folder usage
- notebooks/: clustering exploration and visualization
- src/: preprocessing, clustering, evaluation scripts
- configs/: cluster/search settings
- outputs/figures/: PCA/UMAP plots, cluster profile plots
- outputs/tables/: cluster statistics and metric tables
- outputs/logs/: run logs

## Pipeline logic
1. `src/preprocess_data.py`
	- Load `../dungeon_sensorstats.csv`
	- Remove label leakage (`species` from training features)
	- Impute missing values and standardize continuous features
	- Save shared artifacts for both methods
2. `src/train_kmeans.py`
	- Scan k in config range
	- Save metrics, assignments, centers, cluster profiles
3. `src/train_gmm.py`
	- Scan n_components in config range
	- Save metrics, assignments, means, covariance diagonals, cluster profiles
4. `src/compare_methods.py`
	- Produce dual-view comparison (`method_best` and `fixed_same_k`)
5. `src/visualize_embeddings.py`
	- PCA/UMAP side-by-side embeddings and profile heatmaps
6. `src/aggregate_multi_seed.py`
	- Aggregate per-seed comparison into mean/std tables and summary figure

## How to run
Single seed:
```bash
cd task2_unsupervised
bash src/run_task2.sh 42 5 preprocessed_main
```

Multi seed:
```bash
cd task2_unsupervised
bash src/run_task2_multi_seed.sh 5 preprocessed_main
```

Bribe ablation (separate output root, optional):
```bash
cd task2_unsupervised
/Users/ziyanlei/Desktop/AIR/.venv311/bin/python src/preprocess_data.py \
  --outputs-root outputs_bribe \
  --artifact-prefix preprocessed_with_bribe \
  --include-bribe
```

## Minimum deliverables
- 2 methods implemented and compared fairly
- cluster count selection evidence
- cluster interpretation with feature statistics
- conclusion on stability and usefulness for robotics context

## Latest multi-seed summary (main experiment)
Source: `outputs/tables/metrics_multi_seed_summary.csv`

| view | method | silhouette_mean | davies_bouldin_mean | calinski_harabasz_mean |
| ---- | ------ | --------------- | ------------------- | ---------------------- |
| fixed_same_k | kmeans | 0.5578 | 0.7357 | 16845.7989 |
| fixed_same_k | gmm | 0.5571 | 0.7381 | 16726.0665 |
| method_best | kmeans | 0.5671 | 0.7100 | 16317.8444 |
| method_best | gmm | 0.5663 | 0.7136 | 16193.0713 |

Interpretation:
- At fixed k=5 (task-aligned view), K-Means is slightly better on all three internal metrics.
- At method-best view, both methods select k=6 and remain close; K-Means is still marginally better and faster.

## Bribe ablation summary (2026-04-14)
Source: `outputs_bribe/tables/metrics_multi_seed_summary.csv`

| view | method | silhouette_mean | davies_bouldin_mean | calinski_harabasz_mean |
| ---- | ------ | --------------- | ------------------- | ---------------------- |
| fixed_same_k | kmeans | 0.6142 | 0.6058 | 17686.6791 |
| fixed_same_k | gmm | 0.6141 | 0.6061 | 17656.9718 |
| method_best | kmeans | 0.6176 | 0.6156 | 18477.5715 |
| method_best | gmm | 0.6175 | 0.6154 | 18441.5968 |

Main vs ablation comparison (key takeaway):
- Including `bribe` increases apparent separability substantially (higher Silhouette and lower DBI for both methods).
- The method ranking is still consistent (K-Means remains slightly ahead of GMM), but absolute quality scores are inflated.
- Final report should keep no-`bribe` results as primary evidence and present with-`bribe` only as sensitivity analysis.

## Run log template
Date | Method | Seed | k/components | Silhouette | DBI | CHI | Notes
---- | ------ | ---- | ------------ | ---------- | --- | --- | -----
YYYY-MM-DD | M1 | 42 | 5 | 0.000 | 0.000 | 0.000 | note
