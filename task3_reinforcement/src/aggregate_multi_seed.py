from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dirs, now_stamp, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Task3 metrics across methods, rewards, and seeds.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", type=str, default="42,123,2026,9,20")
    parser.add_argument("--methods", type=str, default="random,dqn,ppo")
    parser.add_argument("--reward-schemes", type=str, default="A,B")
    return parser.parse_args()


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _read_curve(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["r", "is_success", "episode"])
    df = pd.read_csv(path)
    if "episode" not in df.columns:
        df = df.reset_index().rename(columns={"index": "episode"})
        df["episode"] = df["episode"] + 1
    return df


def _plot_curve_with_ci(curves: List[pd.DataFrame], y_col: str, ax, title: str) -> None:
    aligned = []
    min_len = None
    for df in curves:
        if y_col not in df.columns or df.empty:
            continue
        series = df[y_col].to_numpy(dtype=float)
        if min_len is None or len(series) < min_len:
            min_len = len(series)
        aligned.append(series)

    if not aligned or min_len is None or min_len <= 1:
        ax.set_title(title + " (insufficient data)")
        return

    clipped = np.stack([arr[:min_len] for arr in aligned], axis=0)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0)
    n = clipped.shape[0]
    ci95 = 1.96 * (std / np.sqrt(max(n, 1)))
    x = np.arange(1, min_len + 1)
    ax.plot(x, mean, label="mean")
    ax.fill_between(x, mean - ci95, mean + ci95, alpha=0.25, label="95% CI")
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.grid(alpha=0.3)


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    tables_dir = outputs_root / "tables"
    figures_dir = outputs_root / "figures"
    logs_dir = outputs_root / "logs"
    ensure_dirs(tables_dir, figures_dir, logs_dir)

    seeds = [int(x) for x in _parse_csv_list(args.seeds)]
    methods = _parse_csv_list(args.methods)
    reward_schemes = _parse_csv_list(args.reward_schemes)

    metric_frames: List[pd.DataFrame] = []
    curve_groups: Dict[str, List[pd.DataFrame]] = {}

    for method in methods:
        for reward in reward_schemes:
            key = f"{method}_{reward}"
            curve_groups[key] = []
            for seed in seeds:
                tag = f"{method}_seed{seed}_reward{reward}"
                m_path = tables_dir / f"metrics_{tag}.csv"
                c_path = tables_dir / f"curve_{tag}.csv"
                if m_path.exists():
                    df_m = pd.read_csv(m_path)
                    metric_frames.append(df_m)
                curve_groups[key].append(_read_curve(c_path))

    if not metric_frames:
        raise FileNotFoundError("No metrics_*.csv found to aggregate.")

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_metrics.to_csv(tables_dir / "metrics_multi_seed_all.csv", index=False)

    summary = (
        all_metrics.groupby(["reward_scheme", "method"])  # type: ignore[arg-type]
        .agg(
            n_runs=("success_rate", "count"),
            success_rate_mean=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
            episode_return_mean=("episode_return_mean", "mean"),
            episode_return_std=("episode_return_mean", "std"),
            steps_to_goal_mean=("steps_to_goal_mean", "mean"),
            steps_to_goal_std=("steps_to_goal_mean", "std"),
            sample_efficiency_steps_mean=("sample_efficiency_steps", "mean"),
            sample_efficiency_steps_std=("sample_efficiency_steps", "std"),
        )
        .reset_index()
        .sort_values(["reward_scheme", "method"])
    )
    for col in summary.columns:
        if col.endswith("_std"):
            summary[col] = summary[col].fillna(0.0)

    for metric in ["success_rate", "episode_return", "steps_to_goal", "sample_efficiency_steps"]:
        summary[f"{metric}_ci95"] = 1.96 * (
            summary[f"{metric}_std"] / np.sqrt(summary["n_runs"].clip(lower=1))
        )

    summary.to_csv(tables_dir / "metrics_multi_seed_summary.csv", index=False)

    for method in methods:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i, reward in enumerate(reward_schemes):
            key = f"{method}_{reward}"
            _plot_curve_with_ci(
                curves=curve_groups.get(key, []),
                y_col="r",
                ax=axes[0],
                title=f"{method.upper()} Reward {reward} return",
            )
            _plot_curve_with_ci(
                curves=curve_groups.get(key, []),
                y_col="is_success",
                ax=axes[1],
                title=f"{method.upper()} Reward {reward} success",
            )
        handles0, labels0 = axes[0].get_legend_handles_labels()
        if handles0:
            axes[0].legend()
        handles1, labels1 = axes[1].get_legend_handles_labels()
        if handles1:
            axes[1].legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"learning_curve_{method}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    bar_df = summary.copy()
    labels = [f"{r}|{m}" for r, m in zip(bar_df["reward_scheme"], bar_df["method"]) ]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(x, bar_df["success_rate_mean"], yerr=bar_df["success_rate_ci95"], capsize=4)
    axes[0].set_title("Success Rate mean±95%CI")
    axes[0].set_ylim(0.0, 1.0)

    axes[1].bar(x, bar_df["episode_return_mean"], yerr=bar_df["episode_return_ci95"], capsize=4)
    axes[1].set_title("Episode Return mean±95%CI")

    axes[2].bar(x, bar_df["steps_to_goal_mean"], yerr=bar_df["steps_to_goal_ci95"], capsize=4)
    axes[2].set_title("Steps to Goal mean±95%CI")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "metrics_multi_seed_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    run_log = {
        "date": now_stamp(),
        "stage": "aggregate",
        "seeds": seeds,
        "methods": methods,
        "reward_schemes": reward_schemes,
        "aggregation": "mean/std and 95% confidence interval over available seeds",
    }
    write_json(run_log, logs_dir / "run_aggregate_multi_seed.json")

    print("Aggregation complete")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
