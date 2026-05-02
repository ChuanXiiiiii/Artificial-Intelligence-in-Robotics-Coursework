from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dirs, now_stamp, resolve_task2_root, software_versions, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Task2 comparison outputs across seeds.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=resolve_task2_root() / "outputs",
    )
    parser.add_argument("--seeds", type=str, default="42,123,2026")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    tables_dir = outputs_root / "tables"
    figures_dir = outputs_root / "figures"
    logs_dir = outputs_root / "logs"
    ensure_dirs(tables_dir, figures_dir, logs_dir)

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    frames = []
    for seed in seeds:
        path = tables_dir / f"metrics_compare_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing per-seed compare file: {path}")
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(tables_dir / "metrics_multi_seed_all.csv", index=False)

    summary = (
        all_df.groupby(["view", "method"])
        .agg(
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", "std"),
            davies_bouldin_mean=("davies_bouldin", "mean"),
            davies_bouldin_std=("davies_bouldin", "std"),
            calinski_harabasz_mean=("calinski_harabasz", "mean"),
            calinski_harabasz_std=("calinski_harabasz", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_std=("fit_seconds", "std"),
        )
        .reset_index()
        .sort_values(["view", "method"])
    )

    for col in summary.columns:
        if col.endswith("_std"):
            summary[col] = summary[col].fillna(0.0)

    summary.to_csv(tables_dir / "metrics_multi_seed_summary.csv", index=False)

    plot_data = summary.copy()
    plot_data["view_method"] = plot_data["view"] + " | " + plot_data["method"]
    x = np.arange(len(plot_data))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    axes[0].bar(
        x,
        plot_data["silhouette_mean"],
        yerr=plot_data["silhouette_std"],
        capsize=4,
    )
    axes[0].set_title("Silhouette mean±std")
    axes[0].set_ylabel("silhouette")

    axes[1].bar(
        x,
        plot_data["davies_bouldin_mean"],
        yerr=plot_data["davies_bouldin_std"],
        capsize=4,
    )
    axes[1].set_title("Davies-Bouldin mean±std")
    axes[1].set_ylabel("davies_bouldin")

    axes[2].bar(
        x,
        plot_data["calinski_harabasz_mean"],
        yerr=plot_data["calinski_harabasz_std"],
        capsize=4,
    )
    axes[2].set_title("Calinski-Harabasz mean±std")
    axes[2].set_ylabel("calinski_harabasz")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data["view_method"], rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Task2 multi-seed summary ({','.join(map(str, seeds))})", y=1.03)
    fig.tight_layout()
    fig.savefig(figures_dir / "metrics_multi_seed_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    run_log = {
        "date": now_stamp(),
        "stage": "aggregate",
        "seeds": seeds,
        "search_space": "N/A (reads compare outputs)",
        "seed_set": seeds,
        "feature_columns": "N/A (delegated to method run logs)",
        "software_versions": software_versions(),
        "selection_rule": "aggregate over generated comparison files",
        "choice_rationale": "Use multi-seed mean and std to support stability claims beyond single-seed visuals.",
    }
    write_json(run_log, logs_dir / "run_aggregate_multi_seed.json")

    print("Multi-seed aggregation complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
