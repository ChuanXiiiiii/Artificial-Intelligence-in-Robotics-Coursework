from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import ensure_dirs, read_json, resolve_project_root, resolve_task3_root
from hero_task3_env import HeroTask3Env


def _to_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a full HeroBot exploration video on a random 5-loop map, "
            "with entity stickers, A* visualization, and breadcrumb trajectory overlays."
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
        default="outputs/exploration_loop5_demo",
        help="Output directory (absolute or relative to task3_reinforcement).",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="",
        help="Optional output video name. Defaults to herobot_exploration_loop5_seedXXXX.mp4.",
    )
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument(
        "--max-seed-attempts",
        type=int,
        default=200,
        help="Try seed_start..seed_start+N-1 until one episode reaches target.",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--deterministic", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--sprite-root",
        type=str,
        default="dungeon_images_colour80",
        help="Sprite root directory (absolute or relative to project root).",
    )
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--obs-cell-size", type=int, default=16)
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
        "frame_stack": int(env_cfg.get("frame_stack", 1)),
    }


def _stack_obs(history: deque[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(history), axis=0)


def _load_policy(checkpoint_path: Path):
    from stable_baselines3 import PPO

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return PPO.load(str(checkpoint_path), device="cpu")


def _species_sprite_paths(sprite_root: Path) -> Dict[str, List[Path]]:
    species_names = ["halfling", "human", "lizard", "orc", "wingedrat"]
    mapping: Dict[str, List[Path]] = {}
    for species in species_names:
        folder = sprite_root / species
        paths = sorted(folder.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No sprite PNG files found in {folder}")
        mapping[species] = paths
    return mapping


def _resample_filter() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _load_sprite(path: Path, icon_size: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    img = img.resize((icon_size, icon_size), _resample_filter())
    return img


def _cell_center(cell: Tuple[int, int], tile_size: float) -> Tuple[float, float]:
    return ((cell[0] + 0.5) * tile_size, (cell[1] + 0.5) * tile_size)


def _draw_astar_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: Tuple[int, int, int, int],
) -> None:
    draw.line([start, end], fill=color, width=5)
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    norm = max(math.hypot(vx, vy), 1e-6)
    ux, uy = vx / norm, vy / norm
    lx, ly = -uy, ux
    tip = end
    left = (end[0] - ux * 12 + lx * 7, end[1] - uy * 12 + ly * 7)
    right = (end[0] - ux * 12 - lx * 7, end[1] - uy * 12 - ly * 7)
    draw.polygon([tip, left, right], fill=color)


def _draw_binary_grid(
    draw: ImageDraw.ImageDraw,
    top_left: Tuple[int, int],
    matrix: np.ndarray,
    cell: int,
    on_color: Tuple[int, int, int],
    off_color: Tuple[int, int, int],
    border_color: Tuple[int, int, int],
) -> None:
    x0, y0 = top_left
    h, w = matrix.shape
    for iy in range(h):
        for ix in range(w):
            val = int(matrix[iy, ix])
            fill = on_color if val > 0 else off_color
            left = x0 + ix * cell
            top = y0 + iy * cell
            right = left + cell - 1
            bottom = top + cell - 1
            draw.rectangle([left, top, right, bottom], fill=fill, outline=border_color)


@dataclass
class EpisodeRenderResult:
    seed: int
    success: bool
    steps: int
    episode_return: float
    reset_info: Dict[str, float]
    frames: List[np.ndarray]


def _compose_frame(
    env: HeroTask3Env,
    obs: np.ndarray,
    reset_info: Dict[str, float],
    sprite_for_entity_id: Dict[int, Image.Image],
    step_idx: int,
    action_name: str,
    reward: float,
    panel_width: int,
    obs_cell_size: int,
) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None; expected RGB frame")

    base = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGBA")
    map_w, map_h = base.size
    tile = map_w / float(env.grid_size)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d_overlay = ImageDraw.Draw(overlay, "RGBA")

    # Draw recent trajectory with fading brightness.
    trail = env.trajectory[-120:]
    if len(trail) >= 2:
        points = [_cell_center(p, tile) for p in trail]
        n = len(points)
        for i in range(1, n):
            age = i / float(n)
            alpha = int(40 + 180 * age)
            d_overlay.line([points[i - 1], points[i]], fill=(0, 220, 255, alpha), width=3)

    # Draw A* next-step arrow from current position to recommended next cell.
    robot = env._pos_tuple(env.base_env.robot_position)
    target = env._pos_tuple(env.base_env.target_position)
    dx, dy = env._next_astar_step(robot, target)
    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        src = _cell_center(robot, tile)
        dst_cell = (robot[0] + int(dx), robot[1] + int(dy))
        dst = _cell_center(dst_cell, tile)
        _draw_astar_arrow(d_overlay, src, dst, (255, 215, 0, 230))

    # Draw all currently alive entity stickers on their world positions.
    for pos, ent in env.entity_map.items():
        sprite = sprite_for_entity_id.get(ent.entity_id)
        if sprite is None:
            continue
        x = int(pos[0] * tile + (tile - sprite.width) / 2.0)
        y = int(pos[1] * tile + (tile - sprite.height) / 2.0)
        base.alpha_composite(sprite, dest=(x, y))

        faction = env._entity_faction(ent.species)
        if faction == "hostile":
            ring = (235, 72, 52, 220)
        elif faction == "bribable":
            ring = (84, 205, 108, 220)
        else:
            ring = (180, 180, 180, 220)
        d_overlay.ellipse(
            [
                int(pos[0] * tile + 3),
                int(pos[1] * tile + 3),
                int((pos[0] + 1) * tile - 3),
                int((pos[1] + 1) * tile - 3),
            ],
            outline=ring,
            width=2,
        )

    composed = Image.alpha_composite(base, overlay)

    panel = Image.new("RGBA", (panel_width, map_h), (18, 22, 28, 255))
    d_panel = ImageDraw.Draw(panel, "RGBA")
    font = ImageFont.load_default()

    rx, ry = robot
    dir_name = {0: "north", 1: "east", 2: "south", 3: "west"}.get(int(env.base_env.robot_direction), "?")
    entities_total = int(reset_info.get("entity_count", len(env.entity_map)))
    loops_added = int(reset_info.get("braid_loops_added", 0))
    loop_target = int(reset_info.get("braid_loop_target", env.target_braid_loops))

    y = 16
    line_h = 18
    d_panel.text((16, y), "HeroBot Dungeon Exploration", fill=(245, 245, 245), font=font)
    y += line_h + 4
    d_panel.text((16, y), f"step: {step_idx:03d}/{env.max_steps}", fill=(200, 220, 255), font=font)
    y += line_h
    d_panel.text((16, y), f"pos: ({rx},{ry}) dir: {dir_name}", fill=(200, 220, 255), font=font)
    y += line_h
    d_panel.text((16, y), f"action: {action_name}", fill=(200, 220, 255), font=font)
    y += line_h
    d_panel.text((16, y), f"reward: {reward:+.3f}", fill=(200, 220, 255), font=font)
    y += line_h
    d_panel.text((16, y), f"loops: {loops_added}/{loop_target}", fill=(200, 220, 255), font=font)
    y += line_h
    d_panel.text((16, y), f"entities alive: {len(env.entity_map)}/{entities_total}", fill=(200, 220, 255), font=font)

    y += line_h + 6
    d_panel.text((16, y), "Legend", fill=(245, 245, 245), font=font)
    y += line_h
    d_panel.line([(18, y + 8), (70, y + 8)], fill=(0, 220, 255, 255), width=3)
    d_panel.text((78, y), "recent trajectory", fill=(210, 210, 210), font=font)
    y += line_h
    d_panel.line([(18, y + 8), (70, y + 8)], fill=(255, 215, 0, 255), width=4)
    d_panel.text((78, y), "A* next-step arrow", fill=(210, 210, 210), font=font)
    y += line_h
    d_panel.rectangle([18, y + 2, 34, y + 18], outline=(235, 72, 52), width=2)
    d_panel.text((42, y), "hostile entity", fill=(210, 210, 210), font=font)
    y += line_h
    d_panel.rectangle([18, y + 2, 34, y + 18], outline=(84, 205, 108), width=2)
    d_panel.text((42, y), "bribable entity", fill=(210, 210, 210), font=font)

    # Explain the A* design in observation space.
    y += line_h + 6
    d_panel.text((16, y), "A* in model input:", fill=(245, 245, 245), font=font)
    y += line_h
    d_panel.text((16, y), "channel-2 is a downhill mask", fill=(180, 200, 220), font=font)
    y += line_h
    d_panel.text((16, y), "channel-3 is breadcrumb memory", fill=(180, 200, 220), font=font)

    # Draw channel-2 and channel-3 mini-grids.
    grid_top = y + line_h + 4
    d_panel.text((16, grid_top), "obs ch2 (A* mask)", fill=(230, 230, 230), font=font)
    g2_top = grid_top + line_h
    _draw_binary_grid(
        d_panel,
        (16, g2_top),
        np.asarray(obs[2], dtype=np.uint8),
        obs_cell_size,
        on_color=(255, 198, 92),
        off_color=(45, 48, 58),
        border_color=(70, 74, 87),
    )

    g3_title_top = g2_top + 7 * obs_cell_size + 10
    d_panel.text((16, g3_title_top), "obs ch3 (breadcrumb)", fill=(230, 230, 230), font=font)
    g3_top = g3_title_top + line_h
    _draw_binary_grid(
        d_panel,
        (16, g3_top),
        np.asarray(obs[3], dtype=np.uint8),
        obs_cell_size,
        on_color=(92, 220, 255),
        off_color=(45, 48, 58),
        border_color=(70, 74, 87),
    )

    canvas = Image.new("RGBA", (map_w + panel_width, map_h), (0, 0, 0, 255))
    canvas.alpha_composite(composed, dest=(0, 0))
    canvas.alpha_composite(panel, dest=(map_w, 0))
    return np.asarray(canvas.convert("RGB"), dtype=np.uint8)


def _run_episode(
    policy: Any,
    env_kwargs: Dict[str, Any],
    sprite_bank: Dict[str, List[Path]],
    seed: int,
    deterministic: bool,
    panel_width: int,
    obs_cell_size: int,
) -> EpisodeRenderResult:
    frame_stack = int(env_kwargs.get("frame_stack", 1))
    env_init_kwargs = dict(env_kwargs)
    env_init_kwargs.pop("frame_stack", None)
    env = HeroTask3Env(seed=seed, **env_init_kwargs)
    rng = np.random.default_rng(seed + 20260427)

    obs, reset_info = env.reset(seed=seed)
    frame = env.render()
    if frame is None:
        env.close()
        raise RuntimeError("env.render() returned None; expected RGB frame")

    tile = frame.shape[1] / float(env.grid_size)
    icon_size = max(14, int(tile * 0.78))

    sprite_for_entity_id: Dict[int, Image.Image] = {}
    for ent in env.entities:
        if not ent.alive:
            continue
        paths = sprite_bank.get(ent.species)
        if not paths:
            continue
        idx = int(rng.integers(0, len(paths)))
        sprite_for_entity_id[ent.entity_id] = _load_sprite(paths[idx], icon_size)

    frames: List[np.ndarray] = []
    obs_history: deque[np.ndarray] = deque(maxlen=max(1, frame_stack))
    for _ in range(max(1, frame_stack)):
        obs_history.append(obs.copy())
    frames.append(
        _compose_frame(
            env=env,
            obs=obs,
            reset_info=dict(reset_info),
            sprite_for_entity_id=sprite_for_entity_id,
            step_idx=0,
            action_name="reset",
            reward=0.0,
            panel_width=panel_width,
            obs_cell_size=obs_cell_size,
        )
    )

    episode_return = 0.0
    done = False
    step_idx = 0
    action_names = {0: "turn_right", 1: "turn_left", 2: "move_forwards"}

    while not done and step_idx < env.max_steps:
        step_idx += 1
        stacked_obs = _stack_obs(obs_history) if frame_stack > 1 else obs
        action, _ = policy.predict(stacked_obs, deterministic=deterministic)
        action_int = int(action)
        obs, reward, terminated, truncated, _info = env.step(action_int)
        episode_return += float(reward)
        obs_history.append(obs.copy())

        frames.append(
            _compose_frame(
                env=env,
                obs=obs,
                reset_info=dict(reset_info),
                sprite_for_entity_id=sprite_for_entity_id,
                step_idx=step_idx,
                action_name=action_names.get(action_int, str(action_int)),
                reward=float(reward),
                panel_width=panel_width,
                obs_cell_size=obs_cell_size,
            )
        )
        done = bool(terminated or truncated)

    success = bool(env._pos_tuple(env.base_env.robot_position) == env._pos_tuple(env.base_env.target_position))
    result = EpisodeRenderResult(
        seed=seed,
        success=success,
        steps=step_idx,
        episode_return=float(episode_return),
        reset_info=dict(reset_info),
        frames=frames,
    )
    env.close()
    return result


def _save_video(video_path: Path, frames: List[np.ndarray], fps: int) -> None:
    try:
        imageio.mimsave(video_path, frames, fps=max(1, int(fps)), codec="libx264")
    except Exception:
        imageio.mimsave(video_path, frames, fps=max(1, int(fps)))


def main() -> None:
    args = parse_args()

    configs_root = resolve_task3_root() / "configs"
    env_kwargs = _build_env_kwargs(configs_root=configs_root, env_config_name=args.env_config)

    checkpoint_path = _resolve_under_task3(args.checkpoint)
    output_dir = _resolve_under_task3(args.output_dir)
    sprite_root = _resolve_under_project(args.sprite_root)

    ensure_dirs(output_dir)

    policy = _load_policy(checkpoint_path=checkpoint_path)
    sprite_bank = _species_sprite_paths(sprite_root=sprite_root)

    deterministic = _to_bool(args.deterministic)
    attempts = max(1, int(args.max_seed_attempts))

    best_failure: EpisodeRenderResult | None = None
    chosen: EpisodeRenderResult | None = None

    for offset in range(attempts):
        seed = int(args.seed_start) + offset
        episode = _run_episode(
            policy=policy,
            env_kwargs=env_kwargs,
            sprite_bank=sprite_bank,
            seed=seed,
            deterministic=deterministic,
            panel_width=int(args.panel_width),
            obs_cell_size=int(args.obs_cell_size),
        )
        print(
            f"seed={episode.seed} success={episode.success} steps={episode.steps} "
            f"return={episode.episode_return:.3f} loops={int(episode.reset_info.get('braid_loops_added', -1))}"
        )

        if episode.success:
            chosen = episode
            break

        if best_failure is None or episode.episode_return > best_failure.episode_return:
            best_failure = episode

    if chosen is None:
        best_seed = best_failure.seed if best_failure is not None else int(args.seed_start)
        raise RuntimeError(
            "No successful episode found in the requested seed range. "
            f"Best failure seed={best_seed}. Increase --max-seed-attempts or use stochastic eval."
        )

    video_name = args.video_name.strip() if args.video_name else ""
    if not video_name:
        video_name = f"herobot_exploration_loop5_seed{chosen.seed:04d}.mp4"
    if not video_name.lower().endswith(".mp4"):
        video_name = f"{video_name}.mp4"

    video_path = output_dir / video_name
    first_frame_path = output_dir / video_name.replace(".mp4", "_first_frame.png")
    meta_path = output_dir / video_name.replace(".mp4", "_meta.json")

    _save_video(video_path=video_path, frames=chosen.frames, fps=int(args.fps))
    imageio.imwrite(first_frame_path, chosen.frames[0])

    meta = {
        "seed": chosen.seed,
        "success": chosen.success,
        "steps": chosen.steps,
        "episode_return": chosen.episode_return,
        "reset_info": chosen.reset_info,
        "checkpoint": str(checkpoint_path),
        "env_config": args.env_config,
        "deterministic": deterministic,
        "fps": int(args.fps),
        "output_video": str(video_path),
    }
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"video: {video_path}")
    print(f"first_frame: {first_frame_path}")
    print(f"meta: {meta_path}")


if __name__ == "__main__":
    main()
