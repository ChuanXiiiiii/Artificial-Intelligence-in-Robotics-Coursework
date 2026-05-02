from __future__ import annotations

import argparse
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from common import (
    ensure_dirs,
    now_stamp,
    read_json,
    select_torch_device,
    set_global_seed,
    software_versions,
    write_json,
)
from curriculum_wrapper import CurriculumLevelUpCallback, CurriculumWrapper
from hero_task3_env import HeroTask3Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task3 RL training entry for Random/DQN/PPO.")
    parser.add_argument("--method", type=str, choices=["random", "dqn", "ppo"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reward-scheme", type=str, choices=["A", "B", "C", "E", "F", "F2", "G", "H"], required=True)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--env-config", type=str, default="env.json")
    parser.add_argument("--dqn-config", type=str, default="dqn.json")
    parser.add_argument("--ppo-config", type=str, default="ppo.json")
    parser.add_argument("--use-curriculum", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--curriculum-initial-level", type=int, default=1)
    parser.add_argument("--curriculum-train-fixed-level", type=int, default=0)
    parser.add_argument("--curriculum-mixed-sampling", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--curriculum-standard-start-ratio", type=float, default=0.2)
    parser.add_argument("--curriculum-standard-potential-multiplier", type=float, default=1.5)
    parser.add_argument("--curriculum-radius-step", type=int, default=5)
    parser.add_argument("--curriculum-success-window", type=int, default=50)
    parser.add_argument("--curriculum-levelup-threshold", type=float, default=0.7)
    parser.add_argument("--curriculum-smooth-bridge-stages", type=int, default=4)
    parser.add_argument("--curriculum-ent-coef", type=float, default=0.05)
    parser.add_argument("--curriculum-ent-coef-end", type=float, default=0.005)
    parser.add_argument("--curriculum-ent-coef-phases", type=int, default=3)
    parser.add_argument("--curriculum-export-level-gifs", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--curriculum-gif-levels", type=str, default="1,2,3")
    parser.add_argument("--curriculum-gif-fps", type=int, default=6)
    parser.add_argument("--init-model-path", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--eval-only", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--run-tag-suffix", type=str, default="")
    parser.add_argument("--eval-deterministic", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--use-recurrent", type=str, default="false", choices=["true", "false"],
                        help="Use sb3_contrib RecurrentPPO with CnnLstmPolicy (PPO method only).")
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--n-lstm-layers", type=int, default=1)
    return parser.parse_args()


def _to_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y"}


def _parse_int_csv(text: str) -> List[int]:
    vals: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    return vals


def build_env_kwargs(args: argparse.Namespace, render_mode: str | None = None) -> Dict[str, Any]:
    env_cfg = read_json((args.configs_root / args.env_config).resolve())
    kwargs: Dict[str, Any] = {
        "grid_size": int(env_cfg.get("grid_size", 16)),
        "max_steps": int(env_cfg.get("max_steps", 256)),
        "render_mode": render_mode,
        "reward_scheme": args.reward_scheme,
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
    return kwargs


def _strip_env_kwargs(env_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    stripped = dict(env_kwargs)
    stripped.pop("frame_stack", None)
    return stripped


def _wrap_frame_stack_env(
    env: gym.Env,
    frame_stack: int,
    monitor_path: Optional[Path] = None,
    info_keys: Tuple[str, ...] = (),
):
    if frame_stack <= 1:
        return env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="first")
    if monitor_path is not None:
        vec_env = VecMonitor(vec_env, filename=str(monitor_path), info_keywords=info_keys)
    return vec_env


def _stack_obs(history: deque[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(history), axis=0)


def linear_decay_schedule(start: float, end: float) -> Callable[[float], float]:
    def _schedule(progress_remaining: float) -> float:
        progress = float(np.clip(progress_remaining, 0.0, 1.0))
        return float(end + (start - end) * progress)

    return _schedule


def compute_sample_efficiency(
    curve_df: pd.DataFrame,
    threshold: float,
    success_col: str = "is_success",
    length_col: str = "l",
    window: int = 50,
) -> Tuple[int, int]:
    if curve_df.empty or success_col not in curve_df.columns:
        return -1, -1

    success = curve_df[success_col].astype(float)
    rolling = success.rolling(window=window, min_periods=window).mean()
    reached = np.where(rolling.to_numpy() >= threshold)[0]
    if reached.size == 0:
        return -1, -1

    idx = int(reached[0])
    episodes = idx + 1
    if length_col in curve_df.columns:
        steps = int(curve_df[length_col].iloc[:episodes].sum())
    else:
        steps = -1
    return episodes, steps


def _extract_loss_frame(progress_df: pd.DataFrame) -> pd.DataFrame:
    if progress_df.empty:
        return pd.DataFrame(columns=["train/loss"])

    keep_cols = [
        col
        for col in progress_df.columns
        if ("loss" in col.lower()) or col in {"time/total_timesteps", "train/n_updates", "phase"}
    ]
    if not keep_cols:
        return pd.DataFrame(columns=["train/loss"])
    return progress_df[keep_cols].copy()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


class TinyCNN(BaseFeaturesExtractor):
    """Lightweight CNN features extractor for stacked 7x7 radar observations."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = int(observation_space.shape[0])

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = int(self.cnn(sample).shape[1])

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float()))


def _build_dqn_decay_plan(config: Dict[str, Any], total_timesteps: int) -> List[Dict[str, Any]]:
    phases = int(config.get("decay_phases", 4))
    phases = max(1, phases)

    lr_start = float(config.get("learning_rate_start", config.get("learning_rate", 1e-4)))
    lr_end = float(config.get("learning_rate_end", lr_start))

    buffer_start = int(config.get("buffer_size_start", config.get("buffer_size", 200_000)))
    buffer_end = int(config.get("buffer_size_end", buffer_start))
    batch_size = int(config.get("batch_size", 256))

    steps = [total_timesteps // phases] * phases
    for i in range(total_timesteps % phases):
        steps[i] += 1

    plan: List[Dict[str, Any]] = []
    for i in range(phases):
        alpha = float(i / (phases - 1)) if phases > 1 else 0.0
        phase_lr = lr_start + (lr_end - lr_start) * alpha
        phase_buffer = int(round(buffer_start + (buffer_end - buffer_start) * alpha))
        phase_buffer = max(phase_buffer, batch_size)
        plan.append(
            {
                "phase": i + 1,
                "timesteps": int(steps[i]),
                "learning_rate": float(phase_lr),
                "buffer_size": int(phase_buffer),
            }
        )
    return plan


def _save_random_policy_artifact(run_checkpoint_dir: Path, run_tag: str, seed: int, reward_scheme: str) -> Path:
    artifact_path = run_checkpoint_dir / "best_model.zip"
    with zipfile.ZipFile(artifact_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "random_policy_manifest.json",
            (
                "{\n"
                f"  \"method\": \"random\",\n"
                f"  \"run_tag\": \"{run_tag}\",\n"
                f"  \"seed\": {seed},\n"
                f"  \"reward_scheme\": \"{reward_scheme}\"\n"
                "}\n"
            ),
        )
    return artifact_path


def _mps_runtime_status() -> Dict[str, Any]:
    try:
        import torch

        return {
            "torch_mps_built": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built()),
            "torch_mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        }
    except Exception:
        return {"torch_mps_built": False, "torch_mps_available": False}


def run_random_training(env_kwargs: Dict[str, Any], seed: int, total_timesteps: int) -> pd.DataFrame:
    env_init_kwargs = _strip_env_kwargs(env_kwargs)
    env = HeroTask3Env(seed=seed, **env_init_kwargs)
    max_steps = int(env_kwargs.get("max_steps", 256))
    n_episodes = int(np.ceil(total_timesteps / max_steps))

    rows: List[Dict[str, float]] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_steps = 0
        final_info: Dict[str, Any] = {}
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_steps += 1
            final_info = info
        rows.append(
            {
                "episode": float(ep + 1),
                "r": float(ep_return),
                "l": float(ep_steps),
                "is_success": float(final_info.get("is_success", 0.0)),
                "dead_loop_events": float(final_info.get("dead_loop_events", 0.0)),
                "wall_collision_count": float(final_info.get("wall_collision_count", 0.0)),
            }
        )
    env.close()
    return pd.DataFrame(rows)


def run_sb3_training(
    method: str,
    env_kwargs: Dict[str, Any],
    seed: int,
    total_timesteps: int,
    config: Dict[str, Any],
    curriculum_cfg: Dict[str, Any],
    monitor_path: Path,
    run_checkpoint_dir: Path,
    logs_dir: Path,
    run_tag: str,
    init_model_path: Optional[Path] = None,
    eval_only: bool = False,
    use_recurrent: bool = False,
    lstm_hidden_size: int = 256,
    n_lstm_layers: int = 1,
) -> Dict[str, Any]:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    if use_recurrent:
        from sb3_contrib import RecurrentPPO

    # MPS backend has a known LSTM gradient bug (GPURNNOps.mm assertion);
    # fall back to CPU when training a RecurrentPPO model.
    device = select_torch_device(prefer_mps=not use_recurrent)
    if use_recurrent and str(device) != "cpu":
        device = "cpu"

    monitor_csv = Path(str(monitor_path) + ".monitor.csv")
    if monitor_csv.exists():
        monitor_csv.unlink()

    env_init_kwargs = _strip_env_kwargs(env_kwargs)
    base_env = HeroTask3Env(seed=seed, **env_init_kwargs)
    callback = None
    if bool(curriculum_cfg.get("enabled", False)):
        base_env = CurriculumWrapper(
            env=base_env,
            initial_level=int(curriculum_cfg.get("initial_level", 1)),
            train_fixed_level=int(curriculum_cfg.get("train_fixed_level", 0)),
            mixed_sampling_enabled=bool(curriculum_cfg.get("mixed_sampling_enabled", False)),
            standard_start_ratio=float(curriculum_cfg.get("standard_start_ratio", 0.2)),
            standard_potential_multiplier=float(curriculum_cfg.get("standard_potential_multiplier", 1.5)),
            radius_step=int(curriculum_cfg.get("radius_step", 5)),
            success_window=int(curriculum_cfg.get("success_window", 50)),
            levelup_threshold=float(curriculum_cfg.get("levelup_threshold", 0.7)),
            smooth_bridge_from_level=int(curriculum_cfg.get("smooth_bridge_from_level", 2)),
            smooth_bridge_to_level=int(curriculum_cfg.get("smooth_bridge_to_level", 3)),
            smooth_bridge_stages=int(curriculum_cfg.get("smooth_bridge_stages", 4)),
        )
        callback = CurriculumLevelUpCallback(verbose=1)

    info_keys: Tuple[str, ...] = ("is_success", "dead_loop_events", "wall_collision_count", "bribe_cost_total")
    if bool(curriculum_cfg.get("enabled", False)):
        info_keys = info_keys + (
            "curriculum_level",
            "curriculum_recent_success_rate",
            "curriculum_fixed_start",
            "curriculum_sample_standard_start",
            "curriculum_standard_start_ratio",
        )

    frame_stack = int(env_kwargs.get("frame_stack", 1))
    if frame_stack > 1:
        env = _wrap_frame_stack_env(base_env, frame_stack, monitor_path=monitor_path, info_keys=info_keys)
    else:
        env = Monitor(base_env, filename=str(monitor_path), info_keywords=info_keys)

    progress_frames: List[pd.DataFrame] = []
    decay_info: Dict[str, Any] = {}

    if method == "dqn":
        if init_model_path is not None and not eval_only:
            raise ValueError("--init-model-path is currently supported for PPO only, unless --eval-only is true.")

        decay_plan = _build_dqn_decay_plan(config=config, total_timesteps=total_timesteps)
        dqn_policy_kwargs = {
            "features_extractor_class": TinyCNN,
            "features_extractor_kwargs": {"features_dim": 512},
            "normalize_images": False,
        }
        decay_info = {
            "buffer_size_decay": True,
            "learning_rate_decay": True,
            "dqn_decay_plan": decay_plan,
        }

        model = None
        if eval_only:
            if init_model_path is None:
                raise ValueError("--eval-only for DQN requires --resume-from or --init-model-path.")
            checkpoint_path = Path(init_model_path).resolve()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"init model not found: {checkpoint_path}")
            model = DQN.load(str(checkpoint_path), env=env, device=device)

        for phase_cfg in decay_plan:
            if eval_only:
                break
            phase_id = int(phase_cfg["phase"])
            phase_steps = int(phase_cfg["timesteps"])
            phase_lr = float(phase_cfg["learning_rate"])
            phase_buffer = int(phase_cfg["buffer_size"])

            model_kwargs = {
                "policy": "CnnPolicy",
                "env": env,
                "seed": seed,
                "policy_kwargs": dqn_policy_kwargs,
                "learning_rate": phase_lr,
                "buffer_size": phase_buffer,
                "batch_size": int(config.get("batch_size", 256)),
                "gamma": float(config.get("gamma", 0.99)),
                "train_freq": int(config.get("train_freq", 4)),
                "target_update_interval": int(config.get("target_update_interval", 5000)),
                "exploration_fraction": float(config.get("exploration_fraction", 0.45)),
                "exploration_final_eps": float(config.get("exploration_final_eps", 0.08)),
                "gradient_steps": int(config.get("gradient_steps", 1)),
                "device": device,
                "verbose": 0,
            }

            if model is None:
                model = DQN(**model_kwargs)
            else:
                next_model = DQN(**model_kwargs)
                next_model.policy.load_state_dict(model.policy.state_dict())
                model = next_model

            logger_dir = logs_dir / f"sb3_{run_tag}_phase{phase_id}"
            ensure_dirs(logger_dir)
            model.set_logger(configure(str(logger_dir), ["csv", "tensorboard"]))

            model.learn(
                total_timesteps=phase_steps,
                reset_num_timesteps=(phase_id == 1),
                progress_bar=False,
                callback=callback,
            )

            progress_path = logger_dir / "progress.csv"
            phase_df = _safe_read_csv(progress_path)
            if not phase_df.empty:
                phase_df["phase"] = phase_id
                phase_df["phase_learning_rate"] = phase_lr
                phase_df["phase_buffer_size"] = phase_buffer
                progress_frames.append(phase_df)

        if model is None:
            raise RuntimeError("DQN model was not initialized.")

        model_load_stem = run_checkpoint_dir / "best_model"
        model.save(str(model_load_stem))
    else:
        lr_start = float(config.get("learning_rate_start", config.get("learning_rate", 3e-4)))
        lr_end = float(config.get("learning_rate_end", lr_start))
        lr_schedule = linear_decay_schedule(start=lr_start, end=lr_end)
        ent_start = float(config.get("ent_coef_start", config.get("ent_coef", 0.08)))
        ent_end = float(config.get("ent_coef_end", ent_start))
        ent_phases = max(1, int(config.get("ent_coef_phases", 2)))

        phase_steps = [total_timesteps // ent_phases] * ent_phases
        for i in range(total_timesteps % ent_phases):
            phase_steps[i] += 1

        decay_info = {
            "buffer_size_decay": False,
            "learning_rate_decay": True,
            "ppo_lr_start": lr_start,
            "ppo_lr_end": lr_end,
            "ppo_ent_coef_start": ent_start,
            "ppo_ent_coef_end": ent_end,
            "ppo_ent_coef_phases": ent_phases,
        }

        loaded_from_checkpoint = False
        policy_kwargs: Dict[str, Any] = {
            "features_extractor_class": TinyCNN,
            "features_extractor_kwargs": {"features_dim": 512},
            "normalize_images": False,
        }
        if use_recurrent:
            policy_kwargs["lstm_hidden_size"] = int(lstm_hidden_size)
            policy_kwargs["n_lstm_layers"] = int(n_lstm_layers)
            policy_kwargs["enable_critic_lstm"] = True
            decay_info["recurrent"] = True
            decay_info["lstm_hidden_size"] = int(lstm_hidden_size)
            decay_info["n_lstm_layers"] = int(n_lstm_layers)

        algo_cls = RecurrentPPO if use_recurrent else PPO
        algo_policy = "CnnLstmPolicy" if use_recurrent else "CnnPolicy"
        if init_model_path is not None:
            checkpoint_path = Path(init_model_path).resolve()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"init model not found: {checkpoint_path}")
            custom_objects = {
                "policy_kwargs": policy_kwargs,
            }
            model = algo_cls.load(str(checkpoint_path), env=env, device=device, custom_objects=custom_objects)
            loaded_from_checkpoint = True
        else:
            model = algo_cls(
                algo_policy,
                env,
                seed=seed,
                policy_kwargs=policy_kwargs,
                learning_rate=lr_schedule,
                n_steps=int(config.get("n_steps", 2048)),
                batch_size=int(config.get("batch_size", 256)),
                n_epochs=int(config.get("n_epochs", 10)),
                gamma=float(config.get("gamma", 0.99)),
                gae_lambda=float(config.get("gae_lambda", 0.95)),
                clip_range=float(config.get("clip_range", 0.2)),
                ent_coef=ent_start,
                device=device,
                verbose=0,
            )

        for phase_id, phase_steps_i in enumerate(phase_steps, start=1):
            if eval_only:
                break
            alpha = float((phase_id - 1) / (ent_phases - 1)) if ent_phases > 1 else 0.0
            phase_ent = float(ent_start + (ent_end - ent_start) * alpha)
            model.ent_coef = phase_ent

            logger_dir = logs_dir / f"sb3_{run_tag}_phase{phase_id}"
            ensure_dirs(logger_dir)
            model.set_logger(configure(str(logger_dir), ["csv"]))

            model.learn(
                total_timesteps=int(phase_steps_i),
                reset_num_timesteps=(phase_id == 1 and not loaded_from_checkpoint),
                progress_bar=False,
                callback=callback,
            )

            progress_path = logger_dir / "progress.csv"
            ppo_df = _safe_read_csv(progress_path)
            if not ppo_df.empty:
                ppo_df["phase"] = phase_id
                ppo_df["phase_learning_rate_start"] = lr_start
                ppo_df["phase_learning_rate_end"] = lr_end
                ppo_df["phase_ent_coef"] = phase_ent
                progress_frames.append(ppo_df)

        model_load_stem = run_checkpoint_dir / "best_model"
        model.save(str(model_load_stem))

    env.close()

    if monitor_csv.exists():
        curve_df = pd.read_csv(monitor_csv, skiprows=1)
    else:
        curve_df = pd.DataFrame(columns=["r", "l", "is_success", "dead_loop_events", "wall_collision_count"])

    progress_df = pd.concat(progress_frames, ignore_index=True) if progress_frames else pd.DataFrame()
    loss_df = _extract_loss_frame(progress_df)

    curriculum_status: Dict[str, Any] = {}
    if isinstance(base_env, CurriculumWrapper):
        curriculum_status = base_env.status()

    return {
        "curve_df": curve_df,
        "progress_df": progress_df,
        "loss_df": loss_df,
        "model_load_stem": model_load_stem,
        "device": device,
        "decay_info": decay_info,
        "curriculum_status": curriculum_status,
    }


def evaluate_policy(
    method: str,
    model: Any,
    env_kwargs: Dict[str, Any],
    seed: int,
    n_eval_episodes: int,
    eval_deterministic: bool = True,
    is_recurrent: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    cases: List[Dict[str, Any]] = []

    for ep in range(n_eval_episodes):
        env_init_kwargs = _strip_env_kwargs(env_kwargs)
        frame_stack = int(env_kwargs.get("frame_stack", 1))
        env = HeroTask3Env(seed=seed + 10_000 + ep, **env_init_kwargs)
        if frame_stack > 1:
            env = _wrap_frame_stack_env(env, frame_stack)
            obs = env.reset()
        else:
            obs, _ = env.reset(seed=seed + 10_000 + ep)
        # --- LSTM hidden state initialisation (cleared per episode) ---
        lstm_states: Any = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        ep_return = 0.0
        final_info: Dict[str, Any] = {}
        while not done:
            if method == "random":
                if frame_stack > 1:
                    action = np.array([env.action_space.sample()])
                else:
                    action = env.action_space.sample()
            elif is_recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=eval_deterministic,
                )
                # After the first call, episode is no longer fresh until done.
                episode_starts = np.zeros((1,), dtype=bool)
                if frame_stack <= 1:
                    action = int(action)
            else:
                action, _ = model.predict(obs, deterministic=eval_deterministic)
                if frame_stack <= 1:
                    action = int(action)

            if frame_stack > 1:
                obs, reward, done_vec, info = env.step(action)
                ep_return += float(reward[0])
                done = bool(done_vec[0])
                final_info = info[0] if info else {}
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                done = bool(terminated or truncated)
                final_info = info

        is_success = float(final_info.get("is_success", 0.0))
        ep_len = float(final_info.get("episode_length", 0.0))
        dead_loops = float(final_info.get("dead_loop_events", 0.0))
        wall_hits = float(final_info.get("wall_collision_count", 0.0))
        trajectory = str(final_info.get("trajectory", ""))
        is_killed = float(final_info.get("is_killed", 0.0))
        killed_by = str(final_info.get("killed_by", ""))
        rows.append(
            {
                "episode": float(ep + 1),
                "is_success": is_success,
                "episode_return": float(ep_return),
                "steps": ep_len,
                "dead_loop_events": dead_loops,
                "wall_collision_count": wall_hits,
                "is_killed": is_killed,
                "killed_by": killed_by,
                "trajectory": trajectory,
            }
        )

        if dead_loops > 0 or wall_hits >= 5:
            cases.append(
                {
                    "episode": int(ep + 1),
                    "case_type": "dead_loop" if dead_loops > 0 else "wall_collision",
                    "dead_loop_events": dead_loops,
                    "wall_collision_count": wall_hits,
                    "trajectory": trajectory,
                }
            )
        env.close()

    return pd.DataFrame(rows), pd.DataFrame(cases)


def evaluate_curriculum_layers(
    model: Any,
    env_kwargs: Dict[str, Any],
    seed: int,
    n_eval_episodes: int,
    curriculum_cfg: Dict[str, Any],
    levels: Sequence[int] = (1, 2, 3),
    eval_deterministic: bool = True,
    is_recurrent: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    layer_rows: List[Dict[str, Any]] = []

    layer_specs: List[Tuple[str, Optional[int], bool]] = []
    for lvl in levels:
        if int(lvl) < 1:
            continue
        layer_specs.append((f"k{int(lvl)}", int(lvl), True))
    layer_specs.append(("standard_start", None, False))

    for layer_idx, (layer_name, forced_level, use_curriculum_layer) in enumerate(layer_specs):
        for ep in range(n_eval_episodes):
            eval_seed = int(seed + 40_000 + layer_idx * 1_000 + ep)

            if use_curriculum_layer:
                env_init_kwargs = _strip_env_kwargs(env_kwargs)
                frame_stack = int(env_kwargs.get("frame_stack", 1))
                c_env = CurriculumWrapper(
                    HeroTask3Env(seed=eval_seed, **env_init_kwargs),
                    initial_level=int(curriculum_cfg.get("initial_level", 1)),
                    train_fixed_level=int(curriculum_cfg.get("train_fixed_level", 0)),
                    mixed_sampling_enabled=bool(curriculum_cfg.get("mixed_sampling_enabled", False)),
                    standard_start_ratio=float(curriculum_cfg.get("standard_start_ratio", 0.2)),
                    standard_potential_multiplier=float(curriculum_cfg.get("standard_potential_multiplier", 1.5)),
                    radius_step=int(curriculum_cfg.get("radius_step", 5)),
                    success_window=int(curriculum_cfg.get("success_window", 50)),
                    levelup_threshold=float(curriculum_cfg.get("levelup_threshold", 0.7)),
                    smooth_bridge_from_level=int(curriculum_cfg.get("smooth_bridge_from_level", 2)),
                    smooth_bridge_to_level=int(curriculum_cfg.get("smooth_bridge_to_level", 3)),
                    smooth_bridge_stages=int(curriculum_cfg.get("smooth_bridge_stages", 4)),
                )
                obs, _ = c_env.reset(
                    seed=eval_seed,
                    options={"curriculum_level": int(forced_level), "freeze_curriculum": True},
                )
                env = c_env
            else:
                env_init_kwargs = _strip_env_kwargs(env_kwargs)
                frame_stack = int(env_kwargs.get("frame_stack", 1))
                env = HeroTask3Env(seed=eval_seed, **env_init_kwargs)
                obs, _ = env.reset(seed=eval_seed)

            obs_history: deque[np.ndarray] = deque(maxlen=max(1, frame_stack))
            for _ in range(max(1, frame_stack)):
                obs_history.append(np.asarray(obs, dtype=np.uint8))

            done = False
            ep_return = 0.0
            final_info: Dict[str, Any] = {}
            # --- LSTM hidden state initialisation (cleared per episode) ---
            lstm_states: Any = None
            episode_starts = np.ones((1,), dtype=bool)

            while not done:
                stacked_obs = _stack_obs(obs_history) if frame_stack > 1 else obs
                if is_recurrent:
                    action, lstm_states = model.predict(
                        stacked_obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=eval_deterministic,
                    )
                    episode_starts = np.zeros((1,), dtype=bool)
                else:
                    action, _ = model.predict(stacked_obs, deterministic=eval_deterministic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_return += float(reward)
                done = bool(terminated or truncated)
                final_info = info
                obs_history.append(np.asarray(obs, dtype=np.uint8))

            layer_rows.append(
                {
                    "layer": layer_name,
                    "episode": float(ep + 1),
                    "eval_seed": float(eval_seed),
                    "is_success": float(final_info.get("is_success", 0.0)),
                    "episode_return": float(ep_return),
                    "steps": float(final_info.get("episode_length", 0.0)),
                    "dead_loop_events": float(final_info.get("dead_loop_events", 0.0)),
                    "wall_collision_count": float(final_info.get("wall_collision_count", 0.0)),
                    "is_killed": float(final_info.get("is_killed", 0.0)),
                    "killed_by": str(final_info.get("killed_by", "")),
                }
            )

            env.close()

    layer_df = pd.DataFrame(layer_rows)
    if layer_df.empty:
        return layer_df, pd.DataFrame()

    summary = (
        layer_df.groupby("layer", as_index=False)
        .agg(
            success_rate=("is_success", "mean"),
            episode_return_mean=("episode_return", "mean"),
            episode_return_std=("episode_return", lambda x: float(np.std(x.to_numpy(dtype=float), ddof=0))),
            steps_mean=("steps", "mean"),
            dead_loop_rate=("dead_loop_events", lambda x: float((x.to_numpy(dtype=float) > 0).mean())),
            wall_collision_case_rate=("wall_collision_count", lambda x: float((x.to_numpy(dtype=float) >= 5).mean())),
            n_eval_episodes=("episode", "count"),
        )
        .sort_values("layer")
        .reset_index(drop=True)
    )

    return layer_df, summary


def export_curriculum_level_gifs(
    model: Any,
    env_kwargs: Dict[str, Any],
    curriculum_cfg: Dict[str, Any],
    seed: int,
    levels: Sequence[int],
    figures_dir: Path,
    run_tag: str,
    fps: int = 6,
    is_recurrent: bool = False,
) -> List[str]:
    import imageio.v2 as imageio

    saved_paths: List[str] = []
    if not levels:
        return saved_paths

    curriculum_kwargs = {
        "initial_level": int(curriculum_cfg.get("initial_level", 1)),
        "train_fixed_level": int(curriculum_cfg.get("train_fixed_level", 0)),
        "mixed_sampling_enabled": bool(curriculum_cfg.get("mixed_sampling_enabled", False)),
        "standard_start_ratio": float(curriculum_cfg.get("standard_start_ratio", 0.2)),
        "standard_potential_multiplier": float(curriculum_cfg.get("standard_potential_multiplier", 1.5)),
        "radius_step": int(curriculum_cfg.get("radius_step", 5)),
        "success_window": int(curriculum_cfg.get("success_window", 50)),
        "levelup_threshold": float(curriculum_cfg.get("levelup_threshold", 0.7)),
        "smooth_bridge_from_level": int(curriculum_cfg.get("smooth_bridge_from_level", 2)),
        "smooth_bridge_to_level": int(curriculum_cfg.get("smooth_bridge_to_level", 3)),
        "smooth_bridge_stages": int(curriculum_cfg.get("smooth_bridge_stages", 4)),
    }

    for level in levels:
        render_env_kwargs = _strip_env_kwargs(env_kwargs)
        render_env_kwargs["render_mode"] = "rgb_array"
        frame_stack = int(env_kwargs.get("frame_stack", 1))

        env = CurriculumWrapper(HeroTask3Env(seed=seed + 30_000 + int(level), **render_env_kwargs), **curriculum_kwargs)
        obs, _ = env.reset(
            seed=seed + 30_000 + int(level),
            options={"curriculum_level": int(level), "freeze_curriculum": True},
        )

        obs_history: deque[np.ndarray] = deque(maxlen=max(1, frame_stack))
        for _ in range(max(1, frame_stack)):
            obs_history.append(np.asarray(obs, dtype=np.uint8))

        frames: List[np.ndarray] = []
        first_frame = env.render()
        if first_frame is not None:
            if isinstance(first_frame, list):
                first_frame = first_frame[0] if first_frame else None
            if first_frame is not None:
                frames.append(np.asarray(first_frame, dtype=np.uint8))

        done = False
        # --- LSTM hidden state initialisation (cleared per episode) ---
        lstm_states: Any = None
        episode_starts = np.ones((1,), dtype=bool)
        while not done:
            stacked_obs = _stack_obs(obs_history) if frame_stack > 1 else obs
            if is_recurrent:
                action, lstm_states = model.predict(
                    stacked_obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(stacked_obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(int(action))
            done = bool(terminated or truncated)
            obs_history.append(np.asarray(obs, dtype=np.uint8))

            frame = env.render()
            if frame is not None:
                if isinstance(frame, list):
                    frame = frame[0] if frame else None
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))

        env.close()

        if not frames:
            continue

        out_path = figures_dir / f"curriculum_trajectory_{run_tag}_k{int(level)}.gif"
        imageio.mimsave(out_path, frames, fps=max(1, int(fps)))
        saved_paths.append(str(out_path))

    return saved_paths


def summarize_eval(eval_df: pd.DataFrame, method: str, seed: int, reward_scheme: str) -> Dict[str, Any]:
    success_rate = float(eval_df["is_success"].mean()) if not eval_df.empty else 0.0
    episode_return_mean = float(eval_df["episode_return"].mean()) if not eval_df.empty else 0.0
    episode_return_std = float(eval_df["episode_return"].std(ddof=0)) if not eval_df.empty else 0.0

    success_steps = eval_df[eval_df["is_success"] > 0.5]["steps"] if not eval_df.empty else pd.Series(dtype=float)
    steps_to_goal_mean = float(success_steps.mean()) if not success_steps.empty else float("nan")
    steps_to_goal_std = float(success_steps.std(ddof=0)) if not success_steps.empty else float("nan")

    return {
        "method": method,
        "seed": int(seed),
        "reward_scheme": reward_scheme,
        "success_rate": success_rate,
        "episode_return_mean": episode_return_mean,
        "episode_return_std": episode_return_std,
        "steps_to_goal_mean": steps_to_goal_mean,
        "steps_to_goal_std": steps_to_goal_std,
        "dead_loop_rate": float((eval_df["dead_loop_events"] > 0).mean()) if not eval_df.empty else 0.0,
        "wall_collision_case_rate": float((eval_df["wall_collision_count"] >= 5).mean()) if not eval_df.empty else 0.0,
        "n_eval_episodes": int(len(eval_df)),
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    use_curriculum = _to_bool(args.use_curriculum)
    export_curriculum_gifs = _to_bool(args.curriculum_export_level_gifs)
    eval_only = _to_bool(args.eval_only)
    eval_deterministic = _to_bool(args.eval_deterministic)
    use_recurrent = _to_bool(args.use_recurrent)
    if use_recurrent and args.method != "ppo":
        raise ValueError("--use-recurrent is only supported for method=ppo.")

    curriculum_cfg: Dict[str, Any] = {
        "enabled": use_curriculum,
        "initial_level": int(args.curriculum_initial_level),
        "train_fixed_level": int(args.curriculum_train_fixed_level),
        "mixed_sampling_enabled": _to_bool(args.curriculum_mixed_sampling),
        "standard_start_ratio": float(np.clip(args.curriculum_standard_start_ratio, 0.0, 1.0)),
        "standard_potential_multiplier": float(max(1.0, args.curriculum_standard_potential_multiplier)),
        "radius_step": int(args.curriculum_radius_step),
        "success_window": int(args.curriculum_success_window),
        "levelup_threshold": float(args.curriculum_levelup_threshold),
        "smooth_bridge_from_level": 2,
        "smooth_bridge_to_level": 3,
        "smooth_bridge_stages": int(args.curriculum_smooth_bridge_stages),
    }

    outputs_root = args.outputs_root.resolve()
    figures_dir = outputs_root / "figures"
    tables_dir = outputs_root / "tables"
    logs_dir = outputs_root / "logs"
    checkpoints_dir = outputs_root / "checkpoints"
    ensure_dirs(figures_dir, tables_dir, logs_dir, checkpoints_dir)

    env_kwargs = build_env_kwargs(args=args, render_mode=None)

    dqn_cfg = read_json((args.configs_root / args.dqn_config).resolve()) if (args.configs_root / args.dqn_config).exists() else {}
    ppo_cfg = read_json((args.configs_root / args.ppo_config).resolve()) if (args.configs_root / args.ppo_config).exists() else {}

    run_tag = f"{args.method}_seed{args.seed}_reward{args.reward_scheme}"
    if use_curriculum:
        run_tag = f"{run_tag}_curriculum"
    suffix = args.run_tag_suffix.strip()
    if suffix:
        safe_suffix = "_".join(suffix.split())
        run_tag = f"{run_tag}_{safe_suffix}"
    monitor_path = logs_dir / f"monitor_{run_tag}"
    run_checkpoint_dir = checkpoints_dir / run_tag
    ensure_dirs(run_checkpoint_dir)

    if args.method == "random":
        if eval_only:
            raise ValueError("--eval-only is not supported for method=random.")
        curve_df = run_random_training(env_kwargs=env_kwargs, seed=args.seed, total_timesteps=args.total_timesteps)
        model = None
        progress_df = pd.DataFrame()
        loss_df = pd.DataFrame(columns=["train/loss"])
        selected_device = select_torch_device(prefer_mps=True)
        decay_info: Dict[str, Any] = {
            "learning_rate_decay": False,
            "buffer_size_decay": False,
        }
        model_artifact_path = _save_random_policy_artifact(
            run_checkpoint_dir=run_checkpoint_dir,
            run_tag=run_tag,
            seed=args.seed,
            reward_scheme=args.reward_scheme,
        )
    else:
        resume_path_text = args.resume_from.strip()
        init_path_text = args.init_model_path.strip()
        if resume_path_text and init_path_text:
            print("Both --resume-from and --init-model-path are set; using --resume-from.")
        active_init_path = resume_path_text if resume_path_text else init_path_text
        if eval_only and not active_init_path:
            raise ValueError("--eval-only requires --resume-from (or --init-model-path).")

        cfg = dqn_cfg if args.method == "dqn" else ppo_cfg
        if use_curriculum and args.method == "ppo":
            cfg = dict(cfg)
            cfg["ent_coef_start"] = float(args.curriculum_ent_coef)
            cfg["ent_coef_end"] = float(args.curriculum_ent_coef_end)
            cfg["ent_coef_phases"] = max(1, int(args.curriculum_ent_coef_phases))

        if use_curriculum and args.reward_scheme == "C":
            # Curriculum quick runs use potential shaping without C2 heavy penalties.
            env_kwargs["potential_scale_c"] = 0.05
            env_kwargs["wall_hit_penalty_c"] = 0.0
            env_kwargs["dead_loop_penalty_c"] = 0.0

        sb3_result = run_sb3_training(
            method=args.method,
            env_kwargs=env_kwargs,
            seed=args.seed,
            total_timesteps=args.total_timesteps,
            config=cfg,
            curriculum_cfg=curriculum_cfg,
            monitor_path=monitor_path,
            run_checkpoint_dir=run_checkpoint_dir,
            logs_dir=logs_dir,
            run_tag=run_tag,
            init_model_path=Path(active_init_path).resolve() if active_init_path else None,
            eval_only=eval_only,
            use_recurrent=use_recurrent,
            lstm_hidden_size=int(args.lstm_hidden_size),
            n_lstm_layers=int(args.n_lstm_layers),
        )

        curve_df = sb3_result["curve_df"]
        progress_df = sb3_result["progress_df"]
        loss_df = sb3_result["loss_df"]
        selected_device = str(sb3_result["device"])
        decay_info = dict(sb3_result["decay_info"])
        curriculum_status = dict(sb3_result.get("curriculum_status", {}))

        from stable_baselines3 import DQN, PPO

        model_load_stem = Path(sb3_result["model_load_stem"])
        if args.method == "dqn":
            model = DQN.load(str(model_load_stem), device=selected_device)
        elif use_recurrent:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(str(model_load_stem), device=selected_device)
        else:
            model = PPO.load(str(model_load_stem), device=selected_device)
        model_artifact_path = model_load_stem.with_suffix(".zip")

    if args.method == "random":
        curriculum_status = {}

    curve_path = tables_dir / f"curve_{run_tag}.csv"
    progress_path = tables_dir / f"progress_{run_tag}.csv"
    loss_path = tables_dir / f"loss_{run_tag}.csv"

    curve_df.to_csv(curve_path, index=False)
    progress_df.to_csv(progress_path, index=False)
    loss_df.to_csv(loss_path, index=False)

    eval_df, cases_df = evaluate_policy(
        method=args.method,
        model=model,
        env_kwargs=env_kwargs,
        seed=args.seed,
        n_eval_episodes=args.n_eval_episodes,
        eval_deterministic=eval_deterministic,
        is_recurrent=use_recurrent,
    )

    eval_summary = summarize_eval(eval_df, method=args.method, seed=args.seed, reward_scheme=args.reward_scheme)
    episodes_to_thr, steps_to_thr = compute_sample_efficiency(
        curve_df=curve_df,
        threshold=args.success_threshold,
        success_col="is_success",
        length_col="l",
    )
    eval_summary["sample_efficiency_episodes"] = int(episodes_to_thr)
    eval_summary["sample_efficiency_steps"] = int(steps_to_thr)

    eval_summary_df = pd.DataFrame([eval_summary])
    eval_path = tables_dir / f"metrics_{run_tag}.csv"
    cases_path = tables_dir / f"reflection_cases_{run_tag}.csv"
    eval_rows_path = tables_dir / f"eval_rows_{run_tag}.csv"
    eval_summary_df.to_csv(eval_path, index=False)
    eval_df.to_csv(eval_rows_path, index=False)
    cases_df.to_csv(cases_path, index=False)

    layered_eval_path = tables_dir / f"eval_rows_layers_{run_tag}.csv"
    layered_metrics_path = tables_dir / f"metrics_layers_{run_tag}.csv"
    if use_curriculum and args.method == "ppo":
        layered_eval_df, layered_metrics_df = evaluate_curriculum_layers(
            model=model,
            env_kwargs=env_kwargs,
            seed=args.seed,
            n_eval_episodes=args.n_eval_episodes,
            curriculum_cfg=curriculum_cfg,
            levels=(1, 2, 3),
            eval_deterministic=eval_deterministic,
            is_recurrent=use_recurrent,
        )
        layered_eval_df.to_csv(layered_eval_path, index=False)
        layered_metrics_df.to_csv(layered_metrics_path, index=False)

    curriculum_gif_paths: List[str] = []
    if use_curriculum and args.method == "ppo" and export_curriculum_gifs:
        level_up_events = list(curriculum_status.get("level_up_events", []))
        max_allowed_level = int(curriculum_status.get("max_level", args.curriculum_initial_level))
        if len(level_up_events) > 0:
            requested_levels = [x for x in _parse_int_csv(args.curriculum_gif_levels) if x >= 1]
            levels_to_export = [lvl for lvl in requested_levels if lvl <= max_allowed_level]
            curriculum_gif_paths = export_curriculum_level_gifs(
                model=model,
                env_kwargs=env_kwargs,
                curriculum_cfg=curriculum_cfg,
                seed=args.seed,
                levels=levels_to_export,
                figures_dir=figures_dir,
                run_tag=run_tag,
                fps=args.curriculum_gif_fps,
                is_recurrent=use_recurrent,
            )

    mps_status = _mps_runtime_status()
    run_log = {
        "date": now_stamp(),
        "method": args.method,
        "seed": int(args.seed),
        "reward_scheme": args.reward_scheme,
        "total_timesteps": int(args.total_timesteps),
        "n_eval_episodes": int(args.n_eval_episodes),
        "success_threshold": float(args.success_threshold),
        "eval_only": bool(eval_only),
        "eval_deterministic": bool(eval_deterministic),
        "selected_device": selected_device,
        "mps_status": mps_status,
        "decay_info": decay_info,
        "curriculum": curriculum_status,
        "init_model_path": str(Path(args.init_model_path).resolve()) if args.init_model_path.strip() else "",
        "resume_from": str(Path(args.resume_from).resolve()) if args.resume_from.strip() else "",
        "env_kwargs": env_kwargs,
        "curve_path": str(curve_path),
        "progress_path": str(progress_path),
        "loss_path": str(loss_path),
        "eval_metrics_path": str(eval_path),
        "eval_rows_path": str(eval_rows_path),
        "reflection_cases_path": str(cases_path),
        "layered_eval_rows_path": str(layered_eval_path) if (use_curriculum and args.method == "ppo") else "",
        "layered_eval_metrics_path": str(layered_metrics_path) if (use_curriculum and args.method == "ppo") else "",
        "model_artifact_path": str(model_artifact_path),
        "curriculum_gif_paths": curriculum_gif_paths,
        "software_versions": software_versions(),
        "selection_rule": "Compare Random vs DQN vs PPO under the frozen Reward G environment and aggregate over selected seeds.",
        "choice_rationale": "DQN represents value-based RL while PPO represents policy-gradient RL; Random is mandatory baseline.",
    }
    write_json(run_log, logs_dir / f"run_{run_tag}.json")

    print("Training/evaluation complete:")
    print(eval_summary_df.to_string(index=False))
    print(f"best_model: {model_artifact_path}")
    print(f"loss_log: {loss_path}")


if __name__ == "__main__":
    main()
