from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Task1 figures/tables into report folders.")
    parser.add_argument("--source-root", type=Path, default=Path("outputs"))
    parser.add_argument("--report-figures", type=Path, default=Path("../docs/report/figures"))
    parser.add_argument("--report-tables", type=Path, default=Path("../docs/report/tables"))
    return parser.parse_args()


def copy_all(src_dir: Path, dst_dir: Path, suffixes: tuple[str, ...]) -> int:
    ensure_dirs(dst_dir)
    copied = 0
    for item in sorted(src_dir.glob("*")):
        if not item.is_file():
            continue
        if item.suffix.lower() not in suffixes:
            continue
        shutil.copy2(item, dst_dir / item.name)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    src_fig = (args.source_root / "figures").resolve()
    src_tbl = (args.source_root / "tables").resolve()
    dst_fig = args.report_figures.resolve()
    dst_tbl = args.report_tables.resolve()

    n_fig = copy_all(src_fig, dst_fig, (".png", ".jpg", ".jpeg", ".svg", ".pdf"))
    n_tbl = copy_all(src_tbl, dst_tbl, (".csv", ".tsv", ".json"))

    print(f"Synced figures: {n_fig} -> {dst_fig}")
    print(f"Synced tables: {n_tbl} -> {dst_tbl}")


if __name__ == "__main__":
    main()
