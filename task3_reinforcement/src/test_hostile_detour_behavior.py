from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import ensure_dirs, read_json, resolve_project_root, resolve_task3_root
from hero_task3_env import EntityState, HeroTask3Env


BASELINE_ROOT = resolve_project_root() / "SEMTM0016_DungeonMazeWorld-main"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.append(str(BASELINE_ROOT))

from core.dungeonworld_objects import Target, Wall  # noqa: E402


HostileCell = Tuple[int, int]
GridCell = Tuple[int, int]


@dataclass
class LayoutSpec:
    grid_size: int
    start: GridCell
    goal: GridCell
    passable_cells: Set[GridCell]
    common_cells: Set[GridCell]
    short_branch_cells: Set[GridCell]
    safe_branch_cells: Set[GridCell]
    hostile_cells: List[HostileCell]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a controlled dual-route test map: one route contains hostile units, "
            "another route avoids hostiles. Then evaluate HeroBot behavior and export analysis artifacts."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/ppo_seed42_rewardG_curriculum_finalG_curriculum/best_model.zip",
        help="Path to PPO checkpoint .zip (absolute or relative to task3_reinforcement).",
    )
    parser.add_argument("--env-config", type=str, default="env_final_rewardG.json")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/hostile_detour_test",
        help="Output directory (absolute or relative to task3_reinforcement).",
    )
    parser.add_argument("--seed", type=int, default=627)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--stochastic-episodes",
        type=int,
        default=60,
        help="How many stochastic episodes to run on the same test map for behavior tendency.",
    )
    parser.add_argument(
        "--sprite-root",
        type=str,
        default="dungeon_images_colour80",
        help="Sprite root directory (absolute or relative to project root).",
    )
    return parser.parse_args()


def _resolve_under_task3(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return resolve_task3_root() / p


def _resolve_under_project(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return resolve_project_root() / p


def _line(a: GridCell, b: GridCell) -> List[GridCell]:
    x1, y1 = a
    x2, y2 = b
    if x1 == x2:
        ys = range(min(y1, y2), max(y1, y2) + 1)
        return [(x1, y) for y in ys]
    if y1 == y2:
        xs = range(min(x1, x2), max(x1, x2) + 1)
        return [(x, y1) for x in xs]
    raise ValueError(f"Only axis-aligned segments are supported: {a} -> {b}")


def _build_layout(grid_size: int) -> LayoutSpec:
    if grid_size != 16:
        raise ValueError(f"This diagnostic layout is tuned for grid_size=16, got {grid_size}.")

    start = (1, 1)
    goal = (grid_size - 2, grid_size - 2)

    # Junction + merge create two choices:
    # 1) Short branch (contains hostiles)
    # 2) Safe branch (slightly longer, no hostile cells)
    junction = (1, 4)
    merge = (goal[0], 10)

    common = set(_line(start, junction))

    short_branch = set(_line(junction, (goal[0], junction[1])))
    short_branch |= set(_line((goal[0], junction[1]), merge))

    safe_branch = set(_line(junction, (1, 12)))
    safe_branch |= set(_line((1, 12), (10, 12)))
    safe_branch |= set(_line((10, 12), (10, 10)))
    safe_branch |= set(_line((10, 10), merge))

    tail = set(_line(merge, goal))

    passable = set(common) | set(short_branch) | set(safe_branch) | set(tail)

    short_branch_only = set(short_branch) - set(common) - set(tail)
    safe_branch_only = set(safe_branch) - set(common) - set(tail)

    hostile_cells: List[HostileCell] = [(5, 4), (8, 4), (11, 4)]
    for c in hostile_cells:
        if c not in short_branch_only:
            raise ValueError(f"Hostile test cell {c} is not on short branch")

    return LayoutSpec(
        grid_size=grid_size,
        start=start,
        goal=goal,
        passable_cells=passable,
        common_cells=common,
        short_branch_cells=short_branch_only,
        safe_branch_cells=safe_branch_only,
        hostile_cells=hostile_cells,
    )


def _build_env_kwargs(configs_root: Path, env_config_name: str) -> Dict[str, Any]:
    env_cfg = read_json((configs_root / env_config_name).resolve())
    return {
        "grid_size": int(env_cfg.get("grid_size", 16)),
        "max_steps": int(env_cfg.get("max_steps", 256)),
        "render_mode": "rgb_array",
        "reward_scheme": str(env_cfg.get("reward_scheme", "G")),
        "n_virtual_entities": int(env_cfg.get("n_virtual_entities", 12)),
        "target_braid_loops": int(env_cfg.get("target_braid_loops", 5)),
        "task1_model_type": str(env_cfg.get("task1_model_type", "hog_svm")),
        "task1_seed": int(env_cfg.get("task1_seed", 42)),
        "task2_cluster_seed": int(env_cfg.get("task2_cluster_seed", 42)),
        "task2_fixed_k": int(env_cfg.get("task2_fixed_k", 6)),
        "hostile_collision_penalty_b": float(env_cfg.get("hostile_collision_penalty_b", -8.0)),
        "bribable_contact_bonus": float(env_cfg.get("bribable_contact_bonus", 2.0)),
        "goal_reward": float(env_cfg.get("goal_reward", 100.0)),
        "distance_scale_b": float(env_cfg.get("distance_scale_b", 1.0)),
        "approach_bribable_scale_b": float(env_cfg.get("approach_bribable_scale_b", 0.2)),
        "potential_scale_c": float(env_cfg.get("potential_scale_c", 0.2)),
        "path_scale_e": float(env_cfg.get("path_scale_e", 1.0)),
        "step_penalty_e": float(env_cfg.get("step_penalty_e", -0.01)),
        "wall_hit_penalty_c": float(env_cfg.get("wall_hit_penalty_c", -0.3)),
        "dead_loop_penalty_c": float(env_cfg.get("dead_loop_penalty_c", -0.1)),
        "stagnation_penalty_c": float(env_cfg.get("stagnation_penalty_c", 0.0)),
        "hostile_collision_penalty_g": float(env_cfg.get("hostile_collision_penalty_g", -6.0)),
        "kill_zone_penalty_h": float(env_cfg.get("kill_zone_penalty_h", -15.0)),
        "wingedrat_kill_bonus_h": float(env_cfg.get("wingedrat_kill_bonus_h", 1.0)),
        "stealth_wait_steps_h": int(env_cfg.get("stealth_wait_steps_h", 3)),
        "bribe_cost_min_g": float(env_cfg.get("bribe_cost_min_g", 0.08)),
        "bribe_cost_max_g": float(env_cfg.get("bribe_cost_max_g", 0.35)),
        "include_astar_hint": bool(env_cfg.get("include_astar_hint", False)),
        "include_astar_ego_hint": bool(env_cfg.get("include_astar_ego_hint", False)),
    }


def _load_policy(checkpoint_path: Path):
    from stable_baselines3 import PPO

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return PPO.load(str(checkpoint_path), device="cpu")


def _apply_layout_to_env(env: HeroTask3Env, layout: LayoutSpec) -> Dict[str, Any]:
    start = layout.start
    goal = layout.goal

    # Fill interior as walls first.
    for x in range(1, env.grid_size - 1):
        for y in range(1, env.grid_size - 1):
            env.base_env.maze.add_cell_item(x, y, Wall(pos=np.array([x, y], dtype=int)))

    # Carve passable corridors.
    for x, y in sorted(layout.passable_cells):
        env.base_env.maze.add_cell_item(x, y, None)

    # Restore target at fixed goal cell.
    env.base_env.maze.add_cell_item(goal[0], goal[1], Target(pos=np.array([goal[0], goal[1]], dtype=int)))

    env.base_env.robot_position = np.array([start[0], start[1]], dtype=int)
    env.base_env.robot_direction = 2  # south
    env.base_env.robot_camera_view = env.base_env.get_robot_camera_view()

    hostile_species = ["orc", "lizard", "wingedrat"]
    env.entities = []
    env.entity_map = {}
    env.next_entity_id = 1
    for i, pos in enumerate(layout.hostile_cells):
        species = hostile_species[i % len(hostile_species)]
        ent = EntityState(
            entity_id=env.next_entity_id,
            species=species,
            pos=pos,
            entity_type="static",
            direction=(0, 0),
        )
        env.next_entity_id += 1
        env.entities.append(ent)
        env.entity_map[pos] = ent

    env.step_count = 0
    env.episode_return = 0.0
    env.wall_collision_count = 0
    env.dead_loop_events = 0
    env.hostile_collision_count = 0
    env.bribable_contact_count = 0
    env.bribe_cost_total = 0.0
    env.stagnation_events = 0
    env.trajectory = [start]
    env.position_window.clear()
    env.position_window.append(start)
    env.short_position_window.clear()
    env.short_position_window.append(start)

    env.true_distance_map = env._build_true_distance_map(goal)
    env.prev_target_dist = float(env._manhattan(start, goal))
    env.prev_true_target_dist = env._true_distance_to_target(start, goal)
    env.prev_bribable_dist = env._nearest_entity_distance(start, {"human", "halfling"})

    return {
        "reward_scheme": env.reward_scheme,
        "entity_count": len(env.entity_map),
        "braid_opened_walls": float(0.0),
        "braid_loops_added": float(0.0),
        "braid_loop_target": float(0.0),
        "is_custom_test_map": True,
    }


def _bfs_distance(passable: Set[GridCell], start: GridCell, goal: GridCell, blocked: Set[GridCell]) -> int:
    if start in blocked or goal in blocked:
        return 10**9
    q: deque[Tuple[GridCell, int]] = deque([(start, 0)])
    visited: Set[GridCell] = {start}
    while q:
        (x, y), d = q.popleft()
        if (x, y) == goal:
            return d
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (x + dx, y + dy)
            if nb in visited:
                continue
            if nb not in passable:
                continue
            if nb in blocked:
                continue
            visited.add(nb)
            q.append((nb, d + 1))
    return 10**9


def _resample_filter() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _load_species_icons(sprite_root: Path, icon_size: int) -> Dict[str, Image.Image]:
    icons: Dict[str, Image.Image] = {}
    for species in ["orc", "lizard", "wingedrat", "human", "halfling"]:
        folder = sprite_root / species
        paths = sorted(folder.glob("*.png"))
        if not paths:
            continue
        icon = Image.open(paths[0]).convert("RGBA")
        icon = icon.resize((icon_size, icon_size), _resample_filter())
        icons[species] = icon
    return icons


def _draw_map_frame(
    layout: LayoutSpec,
    entity_map: Dict[GridCell, EntityState],
    trajectory: List[GridCell],
    step_text: str,
    subtitle: str,
    icons: Dict[str, Image.Image],
    cell_px: int = 36,
) -> Image.Image:
    grid = layout.grid_size
    width = grid * cell_px + 340
    height = grid * cell_px

    img = Image.new("RGBA", (width, height), (22, 24, 30, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()

    def rect_xy(cell: GridCell) -> Tuple[int, int, int, int]:
        x, y = cell
        left = x * cell_px
        top = y * cell_px
        return (left, top, left + cell_px - 1, top + cell_px - 1)

    # Base tiles
    for x in range(grid):
        for y in range(grid):
            c = (x, y)
            if c in layout.passable_cells:
                color = (246, 248, 252, 255)
            else:
                color = (20, 20, 20, 255)
            draw.rectangle(rect_xy(c), fill=color, outline=(34, 34, 34, 255))

    # Highlight branch choices
    for c in layout.short_branch_cells:
        draw.rectangle(rect_xy(c), fill=(255, 226, 178, 255), outline=(120, 90, 40, 255))
    for c in layout.safe_branch_cells:
        draw.rectangle(rect_xy(c), fill=(209, 247, 220, 255), outline=(45, 110, 60, 255))

    # Draw hostile entities as stickers.
    for pos, ent in entity_map.items():
        left, top, right, bottom = rect_xy(pos)
        icon = icons.get(ent.species)
        if icon is not None:
            ox = left + (cell_px - icon.width) // 2
            oy = top + (cell_px - icon.height) // 2
            img.alpha_composite(icon, dest=(ox, oy))
        else:
            draw.ellipse([left + 8, top + 8, right - 8, bottom - 8], fill=(220, 70, 52, 255))

    # Start and goal markers
    s = layout.start
    g = layout.goal
    draw.rectangle(rect_xy(s), outline=(61, 126, 244, 255), width=3)
    draw.rectangle(rect_xy(g), outline=(235, 56, 73, 255), width=3)

    # Trajectory line
    if len(trajectory) >= 2:
        points = [
            (int((x + 0.5) * cell_px), int((y + 0.5) * cell_px))
            for x, y in trajectory
        ]
        draw.line(points, fill=(0, 205, 255, 220), width=4)

    # Robot marker
    rx, ry = trajectory[-1]
    left, top, right, bottom = rect_xy((rx, ry))
    draw.ellipse([left + 7, top + 7, right - 7, bottom - 7], fill=(33, 104, 255, 245))

    # Side panel
    panel_x = grid * cell_px + 14
    y = 14
    lh = 18

    draw.text((panel_x, y), "Hostile Detour Test Map", fill=(245, 245, 245), font=font)
    y += lh + 2
    draw.text((panel_x, y), step_text, fill=(220, 230, 245), font=font)
    y += lh
    draw.text((panel_x, y), subtitle, fill=(170, 190, 215), font=font)

    y += lh + 6
    draw.text((panel_x, y), "Legend", fill=(245, 245, 245), font=font)
    y += lh
    draw.rectangle([panel_x, y + 2, panel_x + 18, y + 16], fill=(255, 226, 178), outline=(120, 90, 40))
    draw.text((panel_x + 26, y), "short branch (hostile)", fill=(220, 220, 220), font=font)
    y += lh
    draw.rectangle([panel_x, y + 2, panel_x + 18, y + 16], fill=(209, 247, 220), outline=(45, 110, 60))
    draw.text((panel_x + 26, y), "safe branch (no hostile)", fill=(220, 220, 220), font=font)
    y += lh
    draw.line([(panel_x, y + 9), (panel_x + 20, y + 9)], fill=(0, 205, 255), width=3)
    draw.text((panel_x + 26, y), "HeroBot trajectory", fill=(220, 220, 220), font=font)
    y += lh
    draw.ellipse([panel_x + 2, y + 2, panel_x + 16, y + 16], fill=(33, 104, 255))
    draw.text((panel_x + 26, y), "HeroBot current cell", fill=(220, 220, 220), font=font)
    y += lh
    draw.ellipse([panel_x + 2, y + 2, panel_x + 16, y + 16], fill=(220, 70, 52))
    draw.text((panel_x + 26, y), "hostile unit", fill=(220, 220, 220), font=font)

    return img


def _run_episode_with_layout(
    policy: Any,
    env_kwargs: Dict[str, Any],
    layout: LayoutSpec,
    seed: int,
    deterministic: bool,
    icons: Dict[str, Image.Image],
    collect_frames: bool,
) -> Dict[str, Any]:
    env = HeroTask3Env(seed=seed, **env_kwargs)
    env.reset(seed=seed)
    reset_info = _apply_layout_to_env(env, layout)

    obs = env._build_observation()

    trajectory: List[GridCell] = [layout.start]
    frames: List[np.ndarray] = []

    if collect_frames:
        first = _draw_map_frame(
            layout=layout,
            entity_map=dict(env.entity_map),
            trajectory=trajectory,
            step_text=f"step 0 / {env.max_steps}",
            subtitle="episode start",
            icons=icons,
        )
        frames.append(np.asarray(first.convert("RGB"), dtype=np.uint8))

    done = False
    episode_return = 0.0
    last_info: Dict[str, Any] = {}
    step_idx = 0

    while not done and step_idx < env.max_steps:
        step_idx += 1
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        episode_return += float(reward)
        pos = env._pos_tuple(env.base_env.robot_position)
        trajectory.append(pos)
        done = bool(terminated or truncated)
        last_info = info

        if collect_frames:
            subtitle = "reached target" if bool(terminated) else ("truncated" if bool(truncated) else "running")
            frame = _draw_map_frame(
                layout=layout,
                entity_map=dict(env.entity_map),
                trajectory=trajectory,
                step_text=f"step {step_idx} / {env.max_steps}",
                subtitle=subtitle,
                icons=icons,
            )
            frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))

    success = bool(trajectory[-1] == layout.goal)
    hostile_contacts = int(env.hostile_collision_count)
    wall_hits = int(env.wall_collision_count)
    dead_loop_events = int(env.dead_loop_events)

    env.close()

    return {
        "seed": int(seed),
        "success": bool(success),
        "steps": int(step_idx),
        "episode_return": float(episode_return),
        "hostile_contacts": int(hostile_contacts),
        "wall_hits": int(wall_hits),
        "dead_loop_events": int(dead_loop_events),
        "trajectory": trajectory,
        "touched_hostile": bool(hostile_contacts > 0),
        "last_info": last_info,
        "reset_info": reset_info,
        "frames": frames,
    }


def _save_video(video_path: Path, frames: List[np.ndarray], fps: int) -> None:
    try:
        imageio.mimsave(video_path, frames, fps=max(1, int(fps)), codec="libx264")
    except Exception:
        imageio.mimsave(video_path, frames, fps=max(1, int(fps)))


def _contains_any(path: Iterable[GridCell], cells: Set[GridCell]) -> bool:
    return any(p in cells for p in path)


def main() -> None:
    args = parse_args()

    checkpoint_path = _resolve_under_task3(args.checkpoint)
    output_dir = _resolve_under_task3(args.output_dir)
    sprite_root = _resolve_under_project(args.sprite_root)
    ensure_dirs(output_dir)

    env_kwargs = _build_env_kwargs(resolve_task3_root() / "configs", args.env_config)
    layout = _build_layout(grid_size=int(env_kwargs["grid_size"]))
    policy = _load_policy(checkpoint_path)

    icons = _load_species_icons(sprite_root=sprite_root, icon_size=26)

    deterministic_result = _run_episode_with_layout(
        policy=policy,
        env_kwargs=env_kwargs,
        layout=layout,
        seed=int(args.seed),
        deterministic=True,
        icons=icons,
        collect_frames=True,
    )

    # Stochastic tendency on the same map shape.
    stochastic_runs: List[Dict[str, Any]] = []
    for i in range(max(1, int(args.stochastic_episodes))):
        run = _run_episode_with_layout(
            policy=policy,
            env_kwargs=env_kwargs,
            layout=layout,
            seed=int(args.seed) + i,
            deterministic=False,
            icons=icons,
            collect_frames=False,
        )
        stochastic_runs.append(run)

    dist_with_hostile = _bfs_distance(layout.passable_cells, layout.start, layout.goal, blocked=set())
    dist_avoid_hostile = _bfs_distance(
        layout.passable_cells,
        layout.start,
        layout.goal,
        blocked=set(layout.hostile_cells),
    )

    det_traj = deterministic_result["trajectory"]
    det_touched = bool(deterministic_result["touched_hostile"])
    det_used_safe = _contains_any(det_traj, layout.safe_branch_cells)
    det_used_short = _contains_any(det_traj, layout.short_branch_cells)

    stochastic_touch_rate = float(np.mean([1.0 if r["touched_hostile"] else 0.0 for r in stochastic_runs]))
    stochastic_success_rate = float(np.mean([1.0 if r["success"] else 0.0 for r in stochastic_runs]))

    behavior_label = "detour_avoid_hostile" if not det_touched else "direct_through_hostile"

    if not det_touched:
        reason = (
            "HeroBot chose the safe branch: learned hostile penalty in Reward G is large enough "
            "that extra route length is preferable to hostile contact on this map."
        )
    else:
        reason = (
            "HeroBot stepped through hostile units: current policy value estimate favors shorter "
            "A*-guided progress over detour cost on this map, despite hostile penalties."
        )

    # Export artifacts.
    map_layout_img = _draw_map_frame(
        layout=layout,
        entity_map={c: "orc" for c in layout.hostile_cells},
        trajectory=[layout.start],
        step_text="layout preview",
        subtitle="orange=hostile branch, green=safe branch",
        icons=icons,
    )

    traj_img = _draw_map_frame(
        layout=layout,
        entity_map={c: "orc" for c in layout.hostile_cells},
        trajectory=deterministic_result["trajectory"],
        step_text=f"deterministic run: {behavior_label}",
        subtitle=f"steps={deterministic_result['steps']} return={deterministic_result['episode_return']:.3f}",
        icons=icons,
    )

    layout_path = output_dir / f"hostile_detour_layout_seed{args.seed}.png"
    traj_path = output_dir / f"hostile_detour_trajectory_seed{args.seed}.png"
    video_path = output_dir / f"hostile_detour_deterministic_seed{args.seed}.mp4"
    summary_path = output_dir / f"hostile_detour_summary_seed{args.seed}.json"

    map_layout_img.save(layout_path)
    traj_img.save(traj_path)
    _save_video(video_path, deterministic_result["frames"], int(args.fps))

    summary = {
        "seed": int(args.seed),
        "checkpoint": str(checkpoint_path),
        "env_config": args.env_config,
        "behavior_label": behavior_label,
        "deterministic": {
            "success": bool(deterministic_result["success"]),
            "steps": int(deterministic_result["steps"]),
            "episode_return": float(deterministic_result["episode_return"]),
            "hostile_contacts": int(deterministic_result["hostile_contacts"]),
            "wall_hits": int(deterministic_result["wall_hits"]),
            "dead_loop_events": int(deterministic_result["dead_loop_events"]),
            "used_safe_branch": bool(det_used_safe),
            "used_short_branch": bool(det_used_short),
            "touched_hostile": bool(det_touched),
            "trajectory": [f"{x}_{y}" for x, y in deterministic_result["trajectory"]],
        },
        "stochastic_tendency": {
            "episodes": int(len(stochastic_runs)),
            "success_rate": float(stochastic_success_rate),
            "touch_hostile_rate": float(stochastic_touch_rate),
            "mean_steps": float(np.mean([r["steps"] for r in stochastic_runs])),
            "mean_hostile_contacts": float(np.mean([r["hostile_contacts"] for r in stochastic_runs])),
        },
        "route_distances": {
            "shortest_with_hostile_passable": int(dist_with_hostile),
            "shortest_avoiding_hostile_cells": int(dist_avoid_hostile),
            "detour_extra_steps": int(dist_avoid_hostile - dist_with_hostile),
        },
        "reason": reason,
        "artifacts": {
            "layout_image": str(layout_path),
            "trajectory_image": str(traj_path),
            "deterministic_video": str(video_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"behavior={behavior_label}")
    print(
        f"deterministic: success={summary['deterministic']['success']} "
        f"steps={summary['deterministic']['steps']} "
        f"hostile_contacts={summary['deterministic']['hostile_contacts']}"
    )
    print(
        f"stochastic: success_rate={summary['stochastic_tendency']['success_rate']:.3f} "
        f"touch_hostile_rate={summary['stochastic_tendency']['touch_hostile_rate']:.3f}"
    )
    print(f"summary: {summary_path}")
    print(f"layout: {layout_path}")
    print(f"trajectory: {traj_path}")
    print(f"video: {video_path}")


if __name__ == "__main__":
    main()
