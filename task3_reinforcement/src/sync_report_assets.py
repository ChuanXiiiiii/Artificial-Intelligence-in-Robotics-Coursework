from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Task3 figures/tables to report folders.")
    parser.add_argument("--source-root", type=Path, default=Path("outputs"))
    parser.add_argument("--report-root", type=Path, default=Path("../docs/report"))
    return parser.parse_args()


def copy_all(src_dir: Path, dst_dir: Path, suffixes: tuple[str, ...]) -> int:
    ensure_dirs(dst_dir)
    copied = 0
    for item in sorted(src_dir.glob("*")):
        if not item.is_file() or item.suffix.lower() not in suffixes:
            continue
        shutil.copy2(item, dst_dir / item.name)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    report_root = args.report_root.resolve()

    src_fig = source_root / "figures"
    src_tbl = source_root / "tables"
    dst_fig = report_root / "figures"
    dst_tbl = report_root / "tables"
    dst_ref = report_root / "references"

    n_fig = copy_all(src_fig, dst_fig, (".png", ".jpg", ".jpeg", ".svg", ".pdf"))
    n_tbl = copy_all(src_tbl, dst_tbl, (".csv", ".tsv", ".json"))

    ensure_dirs(dst_ref)
    summary_md = dst_ref / "task3_results_summary.md"
    summary_md.write_text(
        "# Task3 RL Result Assets\n\n"
        "This folder receives synced Task3 materials for final report authoring.\n\n"
        "- Figures: docs/report/figures\n"
        "- Tables: docs/report/tables\n"
        "- Seeds: 42, 123, 2026, 9, 20\n"
        "- Methods: Random, DQN, PPO\n"
        "- Reward schemes: A (sparse), B (dense heuristic)\n",
        encoding="utf-8",
    )

    print(f"Synced figures: {n_fig} -> {dst_fig}")
    print(f"Synced tables: {n_tbl} -> {dst_tbl}")
    print(f"Updated report summary: {summary_md}")


if __name__ == "__main__":
    main()
