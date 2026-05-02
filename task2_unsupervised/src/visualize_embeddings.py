from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.decomposition import PCA

from common import (
    ensure_dirs,
    load_preprocessed_artifacts,
    now_stamp,
    resolve_task2_root,
    software_versions,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Task2 cluster assignments with PCA and UMAP.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view", type=str, default="method_best")
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


def _scatter(ax: Axes, emb: np.ndarray, labels: np.ndarray, title: str) -> None:
    scatter = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=labels,
        cmap="tab10",
        s=24,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.grid(alpha=0.2)
    legend = ax.legend(*scatter.legend_elements(), title="cluster", loc="best", fontsize=8)
    ax.add_artist(legend)


def _plot_profile_heatmap(profile_df: pd.DataFrame, method: str, seed: int, k: int, output_path: Path) -> None:
    z_cols = [c for c in profile_df.columns if c.startswith("z_")]
    if not z_cols:
        return

    feature_names = [c.replace("z_", "") for c in z_cols]
    matrix = profile_df[z_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 0.7), 4.8))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(profile_df)))
    ax.set_yticklabels([f"cluster {int(c)}" for c in profile_df["cluster"]])
    ax.set_title(f"{method} cluster profile z-scores (seed {seed}, k={k})")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    tables_dir = outputs_root / "tables"
    figures_dir = outputs_root / "figures"
    logs_dir = outputs_root / "logs"
    ensure_dirs(tables_dir, figures_dir, logs_dir)

    data = load_preprocessed_artifacts(tables_dir=tables_dir, artifact_prefix=args.artifact_prefix)
    X = data.scaled_features.to_numpy(dtype=float)

    compare_path = tables_dir / f"metrics_compare_seed{args.seed}.csv"
    if not compare_path.exists():
        raise FileNotFoundError(f"Missing compare file: {compare_path}")

    compare_df = pd.read_csv(compare_path)
    selected = compare_df.loc[compare_df["view"] == args.view].copy()
    if selected.empty:
        raise ValueError(f"No rows found for view='{args.view}' in {compare_path}")

    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(X)
    pca_ratio = pca.explained_variance_ratio_

    pca_table = pd.DataFrame(
        [
            {"component": "PC1", "explained_variance_ratio": float(pca_ratio[0])},
            {"component": "PC2", "explained_variance_ratio": float(pca_ratio[1])},
            {
                "component": "PC1+PC2",
                "explained_variance_ratio": float(pca_ratio[0] + pca_ratio[1]),
            },
        ]
    )
    pca_table.to_csv(tables_dir / f"pca_explained_variance_seed{args.seed}.csv", index=False)

    umap_available = False
    umap_note = "umap-learn is not installed; UMAP subplot falls back to PCA embedding."
    umap_embedding = pca_embedding.copy()
    try:
        import umap  # type: ignore[import-not-found]

        reducer = umap.UMAP(random_state=args.seed)
        umap_embedding = reducer.fit_transform(X)
        umap_available = True
        umap_note = (
            "UMAP uses default hyperparameters in this run. "
            "n_neighbors controls local connectivity: lower values emphasize local structure; "
            "higher values produce smoother global grouping."
        )
    except Exception:
        umap_available = False

    for _, row in selected.iterrows():
        method = str(row["method"])
        k = int(row["selected_clusters"])
        assign_path = tables_dir / f"cluster_assignments_{method}_seed{args.seed}_k{k}.csv"
        profile_path = tables_dir / f"cluster_profile_{method}_seed{args.seed}_k{k}.csv"

        if not assign_path.exists():
            raise FileNotFoundError(f"Missing assignment file: {assign_path}")
        assign_df = pd.read_csv(assign_path)
        labels = assign_df["cluster"].to_numpy(dtype=int)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        _scatter(
            axes[0],
            pca_embedding,
            labels,
            title=(
                f"PCA ({method}, seed {args.seed}, k={k})\\n"
                f"explained variance={pca_ratio[0] + pca_ratio[1]:.3f}"
            ),
        )
        _scatter(
            axes[1],
            umap_embedding,
            labels,
            title=f"UMAP ({method}, seed {args.seed}, k={k})",
        )

        fig.tight_layout()
        fig.savefig(
            figures_dir / f"embedding_{method}_seed{args.seed}_k{k}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        if profile_path.exists():
            profile_df = pd.read_csv(profile_path)
            _plot_profile_heatmap(
                profile_df=profile_df,
                method=method,
                seed=args.seed,
                k=k,
                output_path=figures_dir / f"cluster_profile_{method}_seed{args.seed}_k{k}.png",
            )

    run_log = {
        "date": now_stamp(),
        "stage": "visualize",
        "seed": int(args.seed),
        "view": args.view,
        "preprocessing_version": data.metadata.get("preprocessing_version", "unknown"),
        "search_space": "N/A (reads selected clusters)",
        "seed_set": [42, 123, 2026],
        "feature_columns": data.feature_columns,
        "software_versions": software_versions(),
        "selection_rule": "visualize clusters selected by compare stage",
        "choice_rationale": "Use PCA+UMAP side-by-side to complement quantitative metrics with geometric interpretation.",
        "umap_available": umap_available,
        "umap_note": umap_note,
        "pca_explained_variance_pc1_pc2": float(pca_ratio[0] + pca_ratio[1]),
    }
    write_json(run_log, logs_dir / f"run_visualize_seed{args.seed}.json")

    print("Visualization complete.")
    print(umap_note)


if __name__ == "__main__":
    main()
