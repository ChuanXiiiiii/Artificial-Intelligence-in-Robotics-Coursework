from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    build_preprocessed_data,
    ensure_dirs,
    load_raw_dataset,
    now_stamp,
    resolve_default_data_path,
    resolve_task2_root,
    save_preprocessed_artifacts,
    software_versions,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task2 preprocessing for fair unsupervised comparison.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=resolve_default_data_path(),
        help="Path to dungeon_sensorstats.csv",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=resolve_task2_root() / "outputs",
        help="Task2 outputs directory",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default="preprocessed_main",
        help="Prefix for preprocessing artifacts",
    )
    parser.add_argument(
        "--include-bribe",
        action="store_true",
        help="Include high-risk feature 'bribe' for ablation runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    tables_dir = outputs_root / "tables"
    logs_dir = outputs_root / "logs"
    ensure_dirs(tables_dir, logs_dir)

    raw_df = load_raw_dataset(args.data_path.resolve())
    preprocessed = build_preprocessed_data(raw_df, include_bribe=args.include_bribe)
    save_preprocessed_artifacts(preprocessed, tables_dir=tables_dir, artifact_prefix=args.artifact_prefix)

    missing_summary = (
        raw_df[preprocessed.feature_columns]
        .isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    missing_summary.to_csv(
        tables_dir / f"{args.artifact_prefix}_missing_summary.csv",
        index=False,
    )

    summary_df = pd.DataFrame(
        [
            {
                "artifact_prefix": args.artifact_prefix,
                "n_rows": int(len(raw_df)),
                "n_features": int(len(preprocessed.feature_columns)),
                "include_bribe": bool(args.include_bribe),
                "preprocessing_version": preprocessed.metadata["preprocessing_version"],
                "created_at": now_stamp(),
            }
        ]
    )
    summary_df.to_csv(tables_dir / f"{args.artifact_prefix}_preprocessing_summary.csv", index=False)

    run_log = {
        "date": now_stamp(),
        "stage": "preprocess",
        "artifact_prefix": args.artifact_prefix,
        "data_path": str(args.data_path.resolve()),
        "preprocessing_version": preprocessed.metadata["preprocessing_version"],
        "search_space": "N/A (preprocess stage)",
        "seed_set": [42, 123, 2026],
        "feature_columns": preprocessed.feature_columns,
        "software_versions": software_versions(),
        "selection_rule": "N/A (preprocess stage)",
        "choice_rationale": "Create a single shared preprocessing artifact for both K-Means and GMM.",
    }
    write_json(run_log, logs_dir / f"run_preprocess_{args.artifact_prefix}.json")

    print("Preprocessing complete.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
