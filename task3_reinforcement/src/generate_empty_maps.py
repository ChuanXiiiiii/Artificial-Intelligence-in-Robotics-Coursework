from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio

from common import ensure_dirs, resolve_project_root, resolve_task3_root


BASELINE_ROOT = resolve_project_root() / "SEMTM0016_DungeonMazeWorld-main"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.append(str(BASELINE_ROOT))

from envs.simple_dungeonworld_env import DungeonMazeEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate empty maze images (maze + start + target) for a seed range."
    )
    parser.add_argument("--seed-start", type=int, default=1, help="Start seed (inclusive).")
    parser.add_argument("--seed-end", type=int, default=50, help="End seed (inclusive).")
    parser.add_argument("--grid-size", type=int, default=16, help="Maze grid size (must be even >= 6).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/empty_maps_seed1_50",
        help="Output directory relative to task3_reinforcement root unless absolute path is provided.",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: str) -> Path:
    out = Path(output_dir)
    if out.is_absolute():
        return out
    return resolve_task3_root() / out


def main() -> None:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("seed-end must be >= seed-start")
    if args.grid_size < 6 or args.grid_size % 2 != 0:
        raise ValueError("grid-size must be an even integer >= 6")

    output_dir = resolve_output_dir(args.output_dir)
    ensure_dirs(output_dir)

    env = DungeonMazeEnv(render_mode="rgb_array", grid_size=int(args.grid_size))
    try:
        for seed in range(int(args.seed_start), int(args.seed_end) + 1):
            env.reset(seed=seed)
            frame = env.render()
            if frame is None:
                raise RuntimeError("env.render() returned None; expected an RGB frame")

            out_path = output_dir / f"seed_{seed:03d}.png"
            imageio.imwrite(out_path, frame)
            print(f"saved: {out_path}")
    finally:
        env.close()

    print(
        f"Done. Generated {args.seed_end - args.seed_start + 1} maps in {output_dir} "
        f"(seeds {args.seed_start}..{args.seed_end})."
    )


if __name__ == "__main__":
    main()
