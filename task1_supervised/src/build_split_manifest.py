from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common import ensure_dirs, extract_prefix_group, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build robust group-stratified split manifest.")
    parser.add_argument("--data-dir", type=Path, default=Path("../dungeon_images_colour80"))
    parser.add_argument("--output-manifest", type=Path, default=Path("outputs/tables/split_manifest.csv"))
    parser.add_argument("--output-stats", type=Path, default=Path("outputs/tables/split_stats.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--bucket-size", type=int, default=20)
    parser.add_argument("--min-val-support", type=int, default=30)
    parser.add_argument("--min-test-support", type=int, default=30)
    return parser.parse_args()


def extract_prefix_and_number(file_name: str) -> tuple[str, int]:
    prefix = extract_prefix_group(file_name)
    stem = Path(file_name).stem
    token = stem.split("_", 1)[1] if "_" in stem else stem

    # Use the last numeric token as the frame id.
    matches = re.findall(r"\d+", token)
    if not matches:
        raise ValueError(f"Cannot parse numeric id from filename: {file_name}")

    number = int(matches[-1])
    if number <= 0:
        number = 1
    return prefix, number


def collect_image_rows(data_dir: Path, bucket_size: int) -> pd.DataFrame:
    if bucket_size <= 0:
        raise ValueError("--bucket-size must be a positive integer")

    rows: List[Dict] = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for image_path in sorted(class_dir.glob("*.png")):
            prefix, number = extract_prefix_and_number(image_path.name)
            bucket = (number - 1) // bucket_size
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "label": label,
                    "prefix": prefix,
                    "number": number,
                    "bucket": bucket,
                    "group_id": f"{label}_{prefix}_B{bucket:03d}",
                }
            )

    if not rows:
        raise RuntimeError(f"No PNG images found under {data_dir}.")
    return pd.DataFrame(rows)


def build_targets(
    total_samples: int,
    ratios: Dict[str, float],
    min_val_support: int,
    min_test_support: int,
) -> Dict[str, int]:
    target_val = max(int(round(total_samples * ratios["val"])), min_val_support)
    target_test = max(int(round(total_samples * ratios["test"])), min_test_support)
    target_train = total_samples - target_val - target_test

    while target_train < 1:
        if target_val > min_val_support and target_val >= target_test:
            target_val -= 1
        elif target_test > min_test_support:
            target_test -= 1
        else:
            raise RuntimeError(
                "Cannot satisfy minimum support constraints for class with "
                f"total={total_samples}, min_val={min_val_support}, min_test={min_test_support}."
            )
        target_train = total_samples - target_val - target_test

    return {"train": target_train, "val": target_val, "test": target_test}


def feasible_after_assignment(
    split_counts: Dict[str, int],
    remaining_samples: int,
    min_val_support: int,
    min_test_support: int,
) -> bool:
    req_train = max(0, 1 - split_counts["train"])
    req_val = max(0, min_val_support - split_counts["val"])
    req_test = max(0, min_test_support - split_counts["test"])
    return (req_train + req_val + req_test) <= remaining_samples


def assign_groups_for_class(
    class_df: pd.DataFrame,
    rng: np.random.Generator,
    ratios: Dict[str, float],
    min_val_support: int,
    min_test_support: int,
) -> Dict[str, str]:
    class_name = str(class_df["label"].iloc[0])
    group_counts = class_df.groupby("group_id").size().reset_index(name="count")
    if len(group_counts) < 3:
        raise RuntimeError(f"Class '{class_name}' has too few groups for train/val/test split.")

    total_samples = int(group_counts["count"].sum())
    targets = build_targets(total_samples, ratios, min_val_support, min_test_support)

    # Deterministic tie-break by seeded shuffle before stable sort by count.
    group_counts = group_counts.iloc[rng.permutation(len(group_counts))]
    group_counts = group_counts.sort_values("count", ascending=False, kind="stable")

    split_order = ["val", "test", "train"]
    split_rank = {"val": 0, "test": 1, "train": 2}
    split_min = {"train": 1, "val": min_val_support, "test": min_test_support}

    split_map: Dict[str, str] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    assigned = 0

    for _, row in group_counts.iterrows():
        gid = str(row["group_id"])
        gcount = int(row["count"])
        remaining = total_samples - assigned - gcount

        candidates = []
        for split_name in split_order:
            trial = dict(split_counts)
            trial[split_name] += gcount

            if not feasible_after_assignment(trial, remaining, min_val_support, min_test_support):
                continue

            l1_distance = (
                abs(trial["train"] - targets["train"])
                + abs(trial["val"] - targets["val"])
                + abs(trial["test"] - targets["test"])
            )
            unmet_priority = 0 if split_counts[split_name] < split_min[split_name] else 1
            deficit = targets[split_name] - split_counts[split_name]
            score = (float(l1_distance), int(unmet_priority), float(-deficit), split_rank[split_name])
            candidates.append((score, split_name, trial))

        if not candidates:
            raise RuntimeError(f"No feasible split choice for class '{class_name}' at group '{gid}'.")

        candidates.sort(key=lambda x: x[0])
        _, chosen_split, chosen_counts = candidates[0]
        split_map[gid] = chosen_split
        split_counts = chosen_counts
        assigned += gcount

    if split_counts["val"] < min_val_support or split_counts["test"] < min_test_support:
        raise RuntimeError(
            f"Class '{class_name}' failed min support constraints: "
            f"val={split_counts['val']}, test={split_counts['test']}"
        )

    return split_map


def main() -> None:
    args = parse_args()
    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    set_global_seed(args.seed)
    master_rng = np.random.default_rng(args.seed)

    data_dir = args.data_dir.resolve()
    df = collect_image_rows(data_dir, bucket_size=args.bucket_size)

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    split_map: Dict[str, str] = {}

    for label in sorted(df["label"].unique().tolist()):
        class_df = df[df["label"] == label].copy()
        class_rng = np.random.default_rng(int(master_rng.integers(0, 2**32 - 1)))
        class_split_map = assign_groups_for_class(
            class_df=class_df,
            rng=class_rng,
            ratios=ratios,
            min_val_support=args.min_val_support,
            min_test_support=args.min_test_support,
        )
        split_map.update(class_split_map)

    df["split"] = df["group_id"].map(split_map)
    if df["split"].isna().any():
        raise RuntimeError("Some samples were not assigned to a split.")

    class_split_counts = df.groupby(["label", "split"]).size().reset_index(name="count")
    for label, class_df in class_split_counts.groupby("label"):
        present = set(class_df["split"].tolist())
        if present != {"train", "val", "test"}:
            raise RuntimeError(f"Class '{label}' does not cover all splits. Found={sorted(present)}")

    leakage_groups = (
        df.groupby("group_id")["split"].nunique().reset_index(name="n_splits").query("n_splits > 1")
    )
    if not leakage_groups.empty:
        raise RuntimeError("Group leakage detected across splits.")

    ensure_dirs(args.output_manifest.parent, args.output_stats.parent)
    df.to_csv(args.output_manifest, index=False)

    stats = (
        df.groupby(["split", "label"]).size().reset_index(name="count").sort_values(["split", "label"])
    )
    stats.to_csv(args.output_stats, index=False)

    print("Split manifest created.")
    print(f"Manifest: {args.output_manifest}")
    print(f"Stats: {args.output_stats}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
