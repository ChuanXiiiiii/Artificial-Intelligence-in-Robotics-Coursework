from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

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
    parser = argparse.ArgumentParser(description="Task2 Method B: Gaussian Mixture Model clustering.")
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
        default="gmm.json",
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

    components_range = [int(v) for v in config["components_range"]]
    covariance_raw = str(config.get("covariance_type", "full"))
    if covariance_raw not in {"full", "tied", "diag", "spherical"}:
        covariance_raw = "full"
    covariance_type = cast(Literal["full", "tied", "diag", "spherical"], covariance_raw)
    n_init = int(config.get("n_init", 1))
    max_iter = int(config.get("max_iter", 200))
    reg_covar = float(config.get("reg_covar", 1e-6))

    rows = []
    for k in components_range:
        start = time.perf_counter()
        model = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=args.seed,
            n_init=n_init,
            max_iter=max_iter,
            reg_covar=reg_covar,
        )
        labels = model.fit_predict(X)
        probs = model.predict_proba(X)
        max_posterior = probs.max(axis=1)
        fit_seconds = time.perf_counter() - start

        internal = evaluate_internal_metrics(X, labels)
        external = evaluate_optional_external_metrics(y_true, labels)

        rows.append(
            {
                "method": "gmm",
                "seed": int(args.seed),
                "n_clusters": int(k),
                "silhouette": internal["silhouette"],
                "davies_bouldin": internal["davies_bouldin"],
                "calinski_harabasz": internal["calinski_harabasz"],
                "ari": external["ari"],
                "nmi": external["nmi"],
                "avg_max_posterior": float(np.mean(max_posterior)),
                "fit_seconds": float(fit_seconds),
            }
        )

        assign_df = pd.DataFrame(
            {
                "row_id": row_ids,
                "species": y_true,
                "cluster": labels.astype(int),
                "max_posterior": max_posterior.astype(float),
            }
        )
        assign_df.to_csv(
            tables_dir / f"cluster_assignments_gmm_seed{args.seed}_k{k}.csv",
            index=False,
        )

        means_array = np.asarray(model.means_, dtype=float)
        means_df = pd.DataFrame(means_array, columns=data.feature_columns)
        means_df.insert(0, "cluster", np.arange(k, dtype=int))
        means_df.to_csv(
            tables_dir / f"cluster_means_gmm_seed{args.seed}_k{k}.csv",
            index=False,
        )

        covariances = np.asarray(model.covariances_, dtype=float)
        if covariance_type == "full":
            cov_diag = np.stack([np.diag(covariances[idx]) for idx in range(k)], axis=0)
        elif covariance_type == "tied":
            tied_diag = np.diag(covariances)
            cov_diag = np.tile(tied_diag[np.newaxis, :], (k, 1))
        elif covariance_type == "diag":
            cov_diag = covariances
        else:
            cov_diag = np.tile(covariances.reshape(-1, 1), (1, len(data.feature_columns)))

        cov_df = pd.DataFrame(cov_diag, columns=[f"covdiag_{c}" for c in data.feature_columns])
        cov_df.insert(0, "cluster", np.arange(k, dtype=int))
        cov_df.to_csv(
            tables_dir / f"cluster_covdiag_gmm_seed{args.seed}_k{k}.csv",
            index=False,
        )

        profile_df = build_cluster_profile(
            imputed_features=data.imputed_features,
            labels=labels,
            method="gmm",
            seed=args.seed,
            n_clusters=k,
            top_n=3,
        )
        profile_df.to_csv(
            tables_dir / f"cluster_profile_gmm_seed{args.seed}_k{k}.csv",
            index=False,
        )

    metrics_df = pd.DataFrame(rows).sort_values("n_clusters").reset_index(drop=True)
    metrics_path = tables_dir / f"metrics_gmm_seed{args.seed}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best = select_best_candidate(metrics_df)
    best_df = pd.DataFrame([best.to_dict()])
    best_df.to_csv(tables_dir / f"best_gmm_seed{args.seed}.csv", index=False)

    run_log = {
        "date": now_stamp(),
        "method": "gmm",
        "seed": int(args.seed),
        "artifact_prefix": args.artifact_prefix,
        "preprocessing_version": data.metadata.get("preprocessing_version", "unknown"),
        "search_space": {"components_range": components_range},
        "seed_set": [42, 123, 2026],
        "feature_columns": data.feature_columns,
        "software_versions": software_versions(),
        "selection_rule": config.get(
            "selection_rule",
            "sort by silhouette desc, davies_bouldin asc, calinski_harabasz desc",
        ),
        "choice_rationale": "GMM complements K-Means by modeling soft memberships and anisotropic cluster shapes.",
        "best_candidate": {
            "n_clusters": int(best["n_clusters"]),
            "silhouette": float(best["silhouette"]),
            "davies_bouldin": float(best["davies_bouldin"]),
            "calinski_harabasz": float(best["calinski_harabasz"]),
            "avg_max_posterior": float(best.get("avg_max_posterior", np.nan)),
        },
    }
    write_json(run_log, logs_dir / f"run_gmm_seed{args.seed}.json")

    print("GMM training complete.")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
