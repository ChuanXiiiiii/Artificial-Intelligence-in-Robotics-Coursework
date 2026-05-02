from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common import (
    build_cluster_profile,
    ensure_dirs,
    evaluate_internal_metrics,
    evaluate_optional_external_metrics,
    load_preprocessed_artifacts,
    now_stamp,
    read_json,
    resolve_task2_root,
    select_best_candidate,
    set_global_seed,
    software_versions,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task2 Method A: K-Means clustering.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=resolve_task2_root() / "outputs",
    )
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=resolve_task2_root() / "configs",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="kmeans.json",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default="preprocessed_main",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    outputs_root = args.outputs_root.resolve()
    tables_dir = outputs_root / "tables"
    logs_dir = outputs_root / "logs"
    ensure_dirs(tables_dir, logs_dir)

    config = read_json((args.configs_root / args.config_name).resolve())
    data = load_preprocessed_artifacts(tables_dir=tables_dir, artifact_prefix=args.artifact_prefix)

    X = data.scaled_features.to_numpy(dtype=float)
    y_true = data.species_labels.to_numpy()
    row_ids = np.arange(len(data.scaled_features), dtype=int)

    k_range = [int(v) for v in config["k_range"]]
    n_init = int(config.get("n_init", 10))
    max_iter = int(config.get("max_iter", 300))
    algorithm_raw = str(config.get("algorithm", "lloyd"))
    if algorithm_raw not in {"lloyd", "elkan", "auto", "full"}:
        algorithm_raw = "lloyd"
    algorithm = cast(Literal["lloyd", "elkan", "auto", "full"], algorithm_raw)

    rows = []
    for k in k_range:
        start = time.perf_counter()
        model = KMeans(
            n_clusters=k,
            random_state=args.seed,
            n_init=n_init,
            max_iter=max_iter,
            algorithm=algorithm,
        )
        labels = model.fit_predict(X)
        fit_seconds = time.perf_counter() - start

        internal = evaluate_internal_metrics(X, labels)
        external = evaluate_optional_external_metrics(y_true, labels)

        rows.append(
            {
                "method": "kmeans",
                "seed": int(args.seed),
                "n_clusters": int(k),
                "silhouette": internal["silhouette"],
                "davies_bouldin": internal["davies_bouldin"],
                "calinski_harabasz": internal["calinski_harabasz"],
                "ari": external["ari"],
                "nmi": external["nmi"],
                "fit_seconds": float(fit_seconds),
            }
        )

        assign_df = pd.DataFrame(
            {
                "row_id": row_ids,
                "species": y_true,
                "cluster": labels.astype(int),
            }
        )
        assign_df.to_csv(
            tables_dir / f"cluster_assignments_kmeans_seed{args.seed}_k{k}.csv",
            index=False,
        )

        center_df = pd.DataFrame(model.cluster_centers_, columns=data.feature_columns)
        center_df.insert(0, "cluster", np.arange(k, dtype=int))
        center_df.to_csv(
            tables_dir / f"cluster_centers_kmeans_seed{args.seed}_k{k}.csv",
            index=False,
        )

        profile_df = build_cluster_profile(
            imputed_features=data.imputed_features,
            labels=labels,
            method="kmeans",
            seed=args.seed,
            n_clusters=k,
            top_n=3,
        )
        profile_df.to_csv(
            tables_dir / f"cluster_profile_kmeans_seed{args.seed}_k{k}.csv",
            index=False,
        )

    metrics_df = pd.DataFrame(rows).sort_values("n_clusters").reset_index(drop=True)
    metrics_path = tables_dir / f"metrics_kmeans_seed{args.seed}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best = select_best_candidate(metrics_df)
    best_df = pd.DataFrame([best.to_dict()])
    best_df.to_csv(tables_dir / f"best_kmeans_seed{args.seed}.csv", index=False)

    run_log = {
        "date": now_stamp(),
        "method": "kmeans",
        "seed": int(args.seed),
        "artifact_prefix": args.artifact_prefix,
        "preprocessing_version": data.metadata.get("preprocessing_version", "unknown"),
        "search_space": {"k_range": k_range},
        "seed_set": [42, 123, 2026],
        "feature_columns": data.feature_columns,
        "software_versions": software_versions(),
        "selection_rule": config.get(
            "selection_rule",
            "sort by silhouette desc, davies_bouldin asc, calinski_harabasz desc",
        ),
        "choice_rationale": "K-Means offers an efficient, interpretable centroid baseline for fair comparison with GMM.",
        "best_candidate": {
            "n_clusters": int(best["n_clusters"]),
            "silhouette": float(best["silhouette"]),
            "davies_bouldin": float(best["davies_bouldin"]),
            "calinski_harabasz": float(best["calinski_harabasz"]),
        },
    }
    write_json(run_log, logs_dir / f"run_kmeans_seed{args.seed}.json")

    print("K-Means training complete.")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
