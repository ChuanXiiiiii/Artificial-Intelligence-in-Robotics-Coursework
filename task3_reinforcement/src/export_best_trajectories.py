from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from common import ensure_dirs, read_json
from hero_task3_env import HeroTask3Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export best trajectories as GIF/MP4 for each method and reward scheme.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--env-config", type=str, default="env.json")
    parser.add_argument("--methods", type=str, default="random,dqn,ppo")
    parser.add_argument("--reward-schemes", type=str, default="A,B")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--strict", type=str, default="false", choices=["true", "false"])
    return parser.parse_args()


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _to_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y"}


def _build_env_kwargs(configs_root: Path, env_config_name: str, reward_scheme: str, render_mode: str | None) -> Dict[str, Any]:
    if env_config_name == "env.json" and reward_scheme == "F2":
        env_config_name = "env_stage5_rewardF2.json"
    elif env_config_name == "env.json" and reward_scheme == "G":
        env_config_name = "env_final_rewardG.json"
    env_cfg = read_json((configs_root / env_config_name).resolve())
    return {
        "grid_size": int(env_cfg.get("grid_size", 16)),
        "max_steps": int(env_cfg.get("max_steps", 256)),
        "render_mode": render_mode,
        "reward_scheme": reward_scheme,
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
        "bribe_cost_min_g": float(env_cfg.get("bribe_cost_min_g", 0.08)),
        "bribe_cost_max_g": float(env_cfg.get("bribe_cost_max_g", 0.35)),
        "include_astar_hint": bool(env_cfg.get("include_astar_hint", False)),
        "include_astar_ego_hint": bool(env_cfg.get("include_astar_ego_hint", False)),
    }


def _load_model(method: str, model_stem: Path):
    if method == "random":
        return None

    from stable_baselines3 import DQN, PPO

    if method == "dqn":
        if not model_stem.with_suffix(".zip").exists():
            return None
        return DQN.load(str(model_stem), device="cpu")
    if method == "ppo":
        if not model_stem.with_suffix(".zip").exists():
            return None
        return PPO.load(str(model_stem), device="cpu")
    return None


def _render_episode(method: str, model: Any, env_kwargs: Dict[str, Any], seed: int) -> List[np.ndarray]:
    env = HeroTask3Env(seed=seed, **env_kwargs)
    obs, _ = env.reset(seed=seed)
    frames: List[np.ndarray] = []

    first_frame = env.render()
    if first_frame is not None:
        frames.append(np.asarray(first_frame, dtype=np.uint8))

    done = False
    while not done:
        if method == "random":
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        obs, _reward, terminated, truncated, _info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))
        done = bool(terminated or truncated)

    env.close()
    return frames


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    tables_dir = outputs_root / "tables"
    figures_dir = outputs_root / "figures"
    checkpoints_dir = outputs_root / "checkpoints"
    ensure_dirs(tables_dir, figures_dir)

    methods = _parse_csv_list(args.methods)
    reward_schemes = _parse_csv_list(args.reward_schemes)
    strict_mode = _to_bool(args.strict)

    best_rows: List[Dict[str, Any]] = []
    strict_violations: List[str] = []

    for reward in reward_schemes:
        for method in methods:
            metric_paths = sorted(tables_dir.glob(f"metrics_{method}_seed*_reward{reward}.csv"))
            if not metric_paths:
                if strict_mode:
                    strict_violations.append(f"missing metrics for method={method}, reward={reward}")
                continue

            frames_df = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
            ranked = frames_df.sort_values(
                by=["success_rate", "episode_return_mean", "seed"],
                ascending=[False, False, True],
            ).reset_index(drop=True)
            best = ranked.iloc[0].to_dict()
            best_seed = int(best["seed"])
            run_tag = f"{method}_seed{best_seed}_reward{reward}"
            model_stem = checkpoints_dir / run_tag / "best_model"

            env_kwargs = _build_env_kwargs(
                configs_root=args.configs_root,
                env_config_name=args.env_config,
                reward_scheme=reward,
                render_mode="rgb_array",
            )
            model = _load_model(method=method, model_stem=model_stem)
            if method != "random" and model is None:
                violation = f"missing model artifact for run_tag={run_tag}"
                if strict_mode:
                    strict_violations.append(violation)
                best_rows.append(
                    {
                        "method": method,
                        "reward_scheme": reward,
                        "best_seed": best_seed,
                        "success_rate": float(best["success_rate"]),
                        "episode_return_mean": float(best["episode_return_mean"]),
                        "steps_to_goal_mean": float(best["steps_to_goal_mean"]),
                        "gif_path": "",
                        "mp4_path": "",
                        "note": "model artifact missing, skipped export",
                    }
                )
                continue

            frames = _render_episode(method=method, model=model, env_kwargs=env_kwargs, seed=best_seed + 20_000)

            if not frames:
                continue

            gif_path = figures_dir / f"best_trajectory_{method}_reward{reward}.gif"
            imageio.mimsave(gif_path, frames, fps=max(args.fps, 1))

            mp4_path = figures_dir / f"best_trajectory_{method}_reward{reward}.mp4"
            mp4_saved = True
            try:
                imageio.mimsave(mp4_path, frames, fps=max(args.fps, 1), codec="libx264")
            except Exception:
                mp4_saved = False

            best_rows.append(
                {
                    "method": method,
                    "reward_scheme": reward,
                    "best_seed": best_seed,
                    "success_rate": float(best["success_rate"]),
                    "episode_return_mean": float(best["episode_return_mean"]),
                    "steps_to_goal_mean": float(best["steps_to_goal_mean"]),
                    "gif_path": str(gif_path),
                    "mp4_path": str(mp4_path) if mp4_saved else "",
                    "note": "ok",
                }
            )

    if not best_rows:
        raise FileNotFoundError("No per-run metrics found for trajectory export.")

    out_df = pd.DataFrame(best_rows).sort_values(["reward_scheme", "method"]).reset_index(drop=True)
    summary_path = tables_dir / "best_trajectory_summary.csv"
    out_df.to_csv(summary_path, index=False)

    if strict_mode and strict_violations:
        msg = "Strict export failed:\n- " + "\n- ".join(strict_violations)
        raise RuntimeError(msg)

    print("Best trajectory export complete")
    print(f"strict_mode={strict_mode}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
