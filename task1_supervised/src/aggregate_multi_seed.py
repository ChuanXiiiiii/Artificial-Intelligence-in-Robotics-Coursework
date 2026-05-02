from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed metrics for method comparison.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", type=str, default="42,123,2026")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    figures_dir = outputs_root / "figures"
    tables_dir = outputs_root / "tables"
    ensure_dirs(figures_dir, tables_dir)

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    frames = []

    for seed in seeds:
        path = tables_dir / f"metrics_compare_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing per-seed metrics file: {path}")
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(tables_dir / "metrics_multi_seed_all.csv", index=False)

    summary = (
        all_df.groupby("method")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .reset_index()
        .sort_values("method")
    )

    for col in ["accuracy_std", "macro_f1_std"]:
        summary[col] = summary[col].fillna(0.0)

    summary.to_csv(tables_dir / "metrics_multi_seed_summary.csv", index=False)

    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        summary["accuracy_mean"],
        width,
        yerr=summary["accuracy_std"],
        capsize=4,
        label="Accuracy",
    )
    ax.bar(
        x + width / 2,
        summary["macro_f1_mean"],
        width,
        yerr=summary["macro_f1_std"],
        capsize=4,
        label="Macro-F1",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title(f"Task1 multi-seed summary ({','.join(map(str, seeds))})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "metrics_multi_seed_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Multi-seed aggregation complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
