from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ensure_dirs


METHODS = ["hog_svm", "resnet18"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate method results and create comparison artifacts.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_error_pairs(pred_df: pd.DataFrame, method: str) -> pd.DataFrame:
    err = pred_df[pred_df["label_true"] != pred_df["label_pred"]].copy()
    if err.empty:
        return pd.DataFrame(columns=["method", "label_true", "label_pred", "count", "ratio_in_errors"])

    pairs = err.groupby(["label_true", "label_pred"]).size().reset_index(name="count")
    total_errors = pairs["count"].sum()
    pairs["ratio_in_errors"] = pairs["count"] / total_errors
    pairs["method"] = method
    return pairs.sort_values("count", ascending=False)


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    figures_dir = outputs_root / "figures"
    tables_dir = outputs_root / "tables"
    ensure_dirs(figures_dir, tables_dir)

    metric_rows = []
    per_class_frames = []
    error_frames = []

    for method in METHODS:
        metrics_path = tables_dir / f"metrics_{method}_seed{args.seed}_test.csv"
        per_class_path = tables_dir / f"per_class_{method}_seed{args.seed}_test.csv"
        pred_path = tables_dir / f"predictions_{method}_seed{args.seed}_test.csv"

        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        if not per_class_path.exists():
            raise FileNotFoundError(f"Missing per-class file: {per_class_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")

        metric_df = pd.read_csv(metrics_path)
        metric_rows.append(metric_df)

        class_df = pd.read_csv(per_class_path)
        class_df["method"] = method
        per_class_frames.append(class_df)

        pred_df = pd.read_csv(pred_path)
        error_frames.append(build_error_pairs(pred_df, method))

    metrics_compare = pd.concat(metric_rows, ignore_index=True)
    per_class_compare = pd.concat(per_class_frames, ignore_index=True)
    error_pairs = pd.concat(error_frames, ignore_index=True)

    metrics_compare.to_csv(tables_dir / f"metrics_compare_seed{args.seed}.csv", index=False)
    per_class_compare.to_csv(tables_dir / f"per_class_compare_seed{args.seed}.csv", index=False)
    error_pairs.to_csv(tables_dir / f"error_pairs_compare_seed{args.seed}.csv", index=False)

    # Focused error table for long-tail class analysis.
    long_tail = per_class_compare[per_class_compare["class"] == "wingedrat"].copy()
    long_tail.to_csv(tables_dir / f"wingedrat_error_focus_seed{args.seed}.csv", index=False)

    # Main metrics bar figure.
    fig, ax = plt.subplots(figsize=(7, 4.8))
    pivot = metrics_compare.set_index("method")[["accuracy", "macro_f1"]]
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title(f"Task1 method comparison (seed {args.seed})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"metrics_compare_seed{args.seed}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Comparison artifacts written.")
    print(metrics_compare.to_string(index=False))


if __name__ == "__main__":
    main()
