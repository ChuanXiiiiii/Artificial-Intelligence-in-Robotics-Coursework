from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from common import ensure_dirs, read_json, resolve_project_root, resolve_task3_root
from entity_inference import BRIBABLE_CLASSES, HOSTILE_CLASSES
from hero_task3_env import HeroTask3Env
from train_agents import TinyCNN  # noqa: F401


@dataclass
class EpisodeRenderResult:
    seed: int
    success: bool
    episode_return: float
    steps: int
    frames_by_style: Dict[str, list[np.ndarray]]


def _to_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y"}


def _resolve_under_task3(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return resolve_task3_root() / path


def _resolve_under_project(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return resolve_project_root() / path


def _resample_filter() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _species_sprite_paths(sprite_root: Path) -> Dict[str, list[Path]]:
    species_names = ["halfling", "human", "lizard", "orc", "wingedrat"]
    mapping: Dict[str, list[Path]] = {}
    for species in species_names:
        folder = sprite_root / species
        paths = sorted(folder.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No sprite PNG files found in {folder}")
        mapping[species] = paths
    return mapping


def _load_sprite(path: Path, icon_size: int) -> Image.Image:
    sprite = Image.open(path).convert("RGBA")
    return sprite.resize((icon_size, icon_size), _resample_filter())


def _overlay_entity_markers(frame: np.ndarray, env: HeroTask3Env) -> np.ndarray:
    frame = frame.copy()
    height, width = frame.shape[:2]
    grid_size = int(env.grid_size)
    cell_h = max(1, height // grid_size)
    cell_w = max(1, width // grid_size)

    hostile_color = np.array([244, 114, 182], dtype=np.uint8)
    bribable_color = np.array([34, 211, 238], dtype=np.uint8)
    neutral_color = np.array([180, 180, 30], dtype=np.uint8)

    for (gx, gy), ent in env.entity_map.items():
        if not ent.alive:
            continue
        if ent.species in HOSTILE_CLASSES:
            color = hostile_color
        elif ent.species in BRIBABLE_CLASSES:
            color = bribable_color
        else:
            color = neutral_color

        px0 = gx * cell_w
        py0 = gy * cell_h
        px1 = min(width, px0 + cell_w)
        py1 = min(height, py0 + cell_h)
        roi = frame[py0:py1, px0:px1]
        if roi.size == 0:
            continue
        frame[py0:py1, px0:px1] = (
            roi.astype(np.float32) * 0.42 + color.astype(np.float32) * 0.58
        ).clip(0, 255).astype(np.uint8)

        dx, dy = int(ent.direction[0]), int(ent.direction[1])
        dot_x = int((gx + 0.5 + dx * 0.3) * cell_w)
        dot_y = int((gy + 0.5 + dy * 0.3) * cell_h)
        radius = max(1, min(cell_w, cell_h) // 6)
        dot_x = max(radius, min(width - radius - 1, dot_x))
        dot_y = max(radius, min(height - radius - 1, dot_y))
        frame[dot_y - radius : dot_y + radius + 1, dot_x - radius : dot_x + radius + 1] = [255, 255, 255]

    return frame


def _overlay_entity_sprites(
    frame: np.ndarray,
    env: HeroTask3Env,
    sprite_bank: Dict[str, list[Path]],
    sprite_cache: Dict[int, Image.Image],
) -> np.ndarray:
    base = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(base, "RGBA")
    width, height = base.size
    tile_w = width / float(env.grid_size)
    tile_h = height / float(env.grid_size)
    icon_size = max(14, int(min(tile_w, tile_h) * 0.82))
    ring_width = max(2, int(min(tile_w, tile_h) * 0.08))
    ring_pad = max(2, int(min(tile_w, tile_h) * 0.10))

    for (gx, gy), ent in env.entity_map.items():
        if not ent.alive:
            continue
        if ent.entity_id not in sprite_cache:
            sprite_paths = sprite_bank.get(ent.species)
            if sprite_paths:
                sprite_idx = int(ent.entity_id % len(sprite_paths))
                sprite_cache[ent.entity_id] = _load_sprite(sprite_paths[sprite_idx], icon_size)

        sprite = sprite_cache.get(ent.entity_id)
        if sprite is not None:
            px = int(gx * tile_w + (tile_w - sprite.width) / 2.0)
            py = int(gy * tile_h + (tile_h - sprite.height) / 2.0)
            base.alpha_composite(sprite, dest=(px, py))

        if ent.species in HOSTILE_CLASSES:
            ring_color = (235, 72, 52, 230)
        elif ent.species in BRIBABLE_CLASSES:
            ring_color = (34, 211, 238, 230)
        else:
            ring_color = (220, 220, 120, 230)

        draw.ellipse(
            [
                int(gx * tile_w + ring_pad),
                int(gy * tile_h + ring_pad),
                int((gx + 1) * tile_w - ring_pad),
                int((gy + 1) * tile_h - ring_pad),
            ],
            outline=ring_color,
            width=ring_width,
        )

    return np.asarray(base.convert("RGB"), dtype=np.uint8)


def _build_frame(
    frame: np.ndarray,
    env: HeroTask3Env,
    render_style: str,
    sprite_bank: Dict[str, list[Path]] | None,
    sprite_cache: Dict[int, Image.Image] | None,
) -> np.ndarray:
    if render_style == "sprites":
        if sprite_bank is None or sprite_cache is None:
            raise ValueError("sprite rendering requires sprite_bank and sprite_cache")
        return _overlay_entity_sprites(frame, env, sprite_bank, sprite_cache)
    return _overlay_entity_markers(frame, env)


def _load_model(method: str, checkpoint_path: Path, use_recurrent: bool):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if method == "dqn":
        return DQN.load(str(checkpoint_path), device="cpu")
    if use_recurrent:
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(str(checkpoint_path), device="cpu")
    return PPO.load(str(checkpoint_path), device="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a GIF from a Task3 checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/ppo_seed42_rewardH_curriculum_stage6H_curriculum/best_model.zip",
    )
    parser.add_argument("--env-config", type=str, default="env_stage6_rewardH.json")
    parser.add_argument("--output-path", type=str, default="outputs/figures/task3_best_model.gif")
    parser.add_argument("--method", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--use-recurrent", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--render-style", type=str, default="markers", choices=["markers", "sprites"])
    parser.add_argument("--paired-render-style", type=str, default="", choices=["", "markers", "sprites"])
    parser.add_argument("--sprite-root", type=str, default="dungeon_images_colour80")
    parser.add_argument("--paired-output-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seed-attempts", type=int, default=1)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--deterministic", type=str, default="true", choices=["true", "false"])
    return parser.parse_args()


def _run_episode(
    model: Any,
    env_cfg: Dict[str, Any],
    render_style: str,
    paired_render_style: str,
    seed: int,
    deterministic: bool,
    max_steps: int,
    use_recurrent: bool,
    sprite_bank: Dict[str, list[Path]] | None,
) -> EpisodeRenderResult:
    frame_stack = int(env_cfg.get("frame_stack", 1))
    env_kwargs = dict(env_cfg)
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs.pop("frame_stack", None)

    base_env = HeroTask3Env(seed=seed, **env_kwargs)
    if frame_stack > 1:
        env = DummyVecEnv([lambda: base_env])
        env = VecFrameStack(env, n_stack=frame_stack, channels_order="first")
        obs = env.reset()
    else:
        obs, _ = base_env.reset(seed=seed)
        env = base_env

    styles = [render_style]
    if paired_render_style and paired_render_style not in styles:
        styles.append(paired_render_style)

    sprite_cache: Dict[int, Image.Image] = {}
    frames_by_style: Dict[str, list[np.ndarray]] = {style: [] for style in styles}
    first_frame = base_env.render()
    if first_frame is not None:
        first_frame_array = np.asarray(first_frame, dtype=np.uint8)
        for style in styles:
            frames_by_style[style].append(
                _build_frame(
                    first_frame_array,
                    base_env,
                    style,
                    sprite_bank,
                    sprite_cache if style == "sprites" else None,
                )
            )

    done = False
    steps = 0
    episode_return = 0.0
    lstm_states: Any = None
    episode_starts = np.ones((1,), dtype=bool)
    final_info: Dict[str, Any] = {}

    while not done and steps < max_steps:
        if use_recurrent:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            episode_starts = np.zeros((1,), dtype=bool)
            if frame_stack <= 1:
                action = int(action)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
            if frame_stack <= 1:
                action = int(action)

        if frame_stack > 1:
            obs, reward, done_vec, info = env.step(action)
            episode_return += float(reward[0])
            done = bool(done_vec[0])
            final_info = info[0] if info else {}
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            done = bool(terminated or truncated)
            final_info = info

        frame = base_env.render()
        if frame is not None:
            frame_array = np.asarray(frame, dtype=np.uint8)
            for style in styles:
                frames_by_style[style].append(
                    _build_frame(
                        frame_array,
                        base_env,
                        style,
                        sprite_bank,
                        sprite_cache if style == "sprites" else None,
                    )
                )
        steps += 1

    success = bool(final_info.get("is_success", 0.0))
    if not success:
        success = bool(base_env._pos_tuple(base_env.base_env.robot_position) == base_env._pos_tuple(base_env.base_env.target_position))

    env.close()
    return EpisodeRenderResult(
        seed=seed,
        success=success,
        episode_return=float(episode_return),
        steps=steps,
        frames_by_style=frames_by_style,
    )


def main() -> None:
    args = parse_args()
    task3_root = resolve_task3_root()
    env_cfg = read_json((task3_root / "configs" / args.env_config).resolve())
    checkpoint_path = _resolve_under_task3(args.checkpoint)
    output_path = _resolve_under_task3(args.output_path)
    paired_output_path = _resolve_under_task3(args.paired_output_path) if args.paired_output_path else None
    deterministic = _to_bool(args.deterministic)
    use_recurrent = _to_bool(args.use_recurrent)

    sprite_bank = None
    if args.render_style == "sprites" or args.paired_render_style == "sprites":
        sprite_bank = _species_sprite_paths(_resolve_under_project(args.sprite_root))

    model = _load_model(method=args.method, checkpoint_path=checkpoint_path, use_recurrent=use_recurrent)

    chosen: EpisodeRenderResult | None = None
    best_failure: EpisodeRenderResult | None = None
    attempts = max(1, int(args.max_seed_attempts))

    for seed_offset in range(attempts):
        candidate_seed = int(args.seed) + seed_offset
        result = _run_episode(
            model=model,
            env_cfg=env_cfg,
            render_style=args.render_style,
            paired_render_style=args.paired_render_style,
            seed=candidate_seed,
            deterministic=deterministic,
            max_steps=int(args.max_steps),
            use_recurrent=use_recurrent,
            sprite_bank=sprite_bank,
        )
        print(
            f"seed={result.seed} success={result.success} steps={result.steps} "
            f"episode_return={result.episode_return:.3f} style={args.render_style}"
        )
        if result.success:
            chosen = result
            break
        if best_failure is None or result.episode_return > best_failure.episode_return:
            best_failure = result

    if chosen is None:
        chosen = best_failure
        print("No successful episode found in requested seed range; exporting best failure instead.")

    if chosen is None or not chosen.frames_by_style.get(args.render_style):
        raise RuntimeError("No renderable frames were generated.")

    ensure_dirs(output_path.parent)
    imageio.mimsave(output_path, chosen.frames_by_style[args.render_style], fps=max(1, int(args.fps)))
    print(f"saved gif: {output_path}")
    if paired_output_path is not None:
        paired_style = args.paired_render_style
        paired_frames = chosen.frames_by_style.get(paired_style, [])
        if not paired_style or not paired_frames:
            raise RuntimeError("paired render requested but no paired frames were generated")
        ensure_dirs(paired_output_path.parent)
        imageio.mimsave(paired_output_path, paired_frames, fps=max(1, int(args.fps)))
        print(f"saved paired gif: {paired_output_path}")
    print(f"chosen_seed: {chosen.seed}")
    print(f"chosen_success: {chosen.success}")


if __name__ == "__main__":
    main()
