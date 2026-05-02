from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import (
    ensure_dirs,
    load_preprocessed_artifacts,
    now_stamp,
    resolve_task2_root,
    select_best_candidate,
    software_versions,
    write_json,
)

METHODS = ["kmeans", "gmm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task2 compare K-Means and GMM under fair protocol.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=resolve_task2_root() / "outputs",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default="preprocessed_main",
    )
    return parser.parse_args()


def _load_metrics(tables_dir: Path, method: str, seed: int) -> pd.DataFrame:
    path = tables_dir / f"metrics_{method}_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return pd.read_csv(path)


def _tag_view(row: pd.Series, view: str, fixed_k: int) -> dict:
    out = row.to_dict()
    out["view"] = view
    out["fixed_k"] = int(fixed_k)
    out["selected_clusters"] = int(out["n_clusters"])
    return out


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    tables_dir = outputs_root / "tables"
    figures_dir = outputs_root / "figures"
    logs_dir = outputs_root / "logs"
    ensure_dirs(tables_dir, figures_dir, logs_dir)

    prep = load_preprocessed_artifacts(tables_dir=tables_dir, artifact_prefix=args.artifact_prefix)

    compare_rows = []
    selection_rows = []

    for method in METHODS:
        metrics_df = _load_metrics(tables_dir, method=method, seed=args.seed)

        best_row = select_best_candidate(metrics_df)
        compare_rows.append(_tag_view(best_row, "method_best", fixed_k=args.fixed_k))

        fixed_df = metrics_df.loc[metrics_df["n_clusters"] == args.fixed_k]
        if fixed_df.empty:
            raise ValueError(
                f"Method '{method}' has no result for fixed k={args.fixed_k}. "
                "Expand search range or change --fixed-k."
            )
        fixed_row = fixed_df.iloc[0]
        compare_rows.append(_tag_view(fixed_row, "fixed_same_k", fixed_k=args.fixed_k))

        selection_rows.append(
            {
                "method": method,
                "seed": int(args.seed),
                "best_k": int(best_row["n_clusters"]),
                "fixed_k": int(args.fixed_k),
                "best_silhouette": float(best_row["silhouette"]),
                "fixed_silhouette": float(fixed_row["silhouette"]),
            }
        )

    compare_df = pd.DataFrame(compare_rows)
    compare_df = compare_df[
        [
            "view",
            "method",
            "seed",
            "selected_clusters",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "ari",
            "nmi",
            "fit_seconds",
        ]
    ].sort_values(["view", "method"]).reset_index(drop=True)

    compare_df.to_csv(tables_dir / f"metrics_compare_seed{args.seed}.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(
        tables_dir / f"selection_compare_seed{args.seed}.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    metric_info = [
        ("silhouette", True, "Silhouette (higher better)"),
        ("davies_bouldin", False, "Davies-Bouldin (lower better)"),
        ("calinski_harabasz", True, "Calinski-Harabasz (higher better)"),
    ]

    for ax, (metric, _higher_better, title) in zip(axes, metric_info):
        pivot = compare_df.pivot(index="method", columns="view", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("method")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Task2 fair comparison views (seed {args.seed})", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / f"metrics_compare_seed{args.seed}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    run_log = {
        "date": now_stamp(),
        "stage": "compare",
        "seed": int(args.seed),
        "fixed_k": int(args.fixed_k),
        "preprocessing_version": prep.metadata.get("preprocessing_version", "unknown"),
        "search_space": "uses method metrics generated from k/components ranges",
        "seed_set": [42, 123, 2026],
        "feature_columns": prep.feature_columns,
        "software_versions": software_versions(),
        "selection_rule": "report both views: method_best and fixed_same_k",
        "choice_rationale": "Dual-view reporting prevents fairness criticism from only comparing per-method optimum.",
    }
    write_json(run_log, logs_dir / f"run_compare_seed{args.seed}.json")

    print("Comparison complete.")
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()
