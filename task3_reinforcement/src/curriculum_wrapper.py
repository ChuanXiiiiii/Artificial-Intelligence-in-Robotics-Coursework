from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from entity_inference import BRIBABLE_CLASSES
from hero_task3_env import HeroTask3Env


class CurriculumWrapper(gym.Wrapper):
    def __init__(
        self,
        env: HeroTask3Env,
        initial_level: int = 1,
        radius_step: int = 5,
        success_window: int = 50,
        levelup_threshold: float = 0.7,
        train_fixed_level: int = 0,
        mixed_sampling_enabled: bool = False,
        standard_start_ratio: float = 0.2,
        standard_potential_multiplier: float = 1.5,
        smooth_bridge_from_level: int = 2,
        smooth_bridge_to_level: int = 3,
        smooth_bridge_stages: int = 4,
        max_level: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self.initial_level = max(1, int(initial_level))
        self.level = self.initial_level
        self.radius_step = max(1, int(radius_step))
        self.success_window = max(1, int(success_window))
        self.levelup_threshold = float(levelup_threshold)
        self.train_fixed_level = max(0, int(train_fixed_level))
        self.mixed_sampling_enabled = bool(mixed_sampling_enabled)
        self.standard_start_ratio = float(np.clip(float(standard_start_ratio), 0.0, 1.0))
        self.current_standard_start_ratio = float(self.standard_start_ratio)
        self.standard_potential_multiplier = max(1.0, float(standard_potential_multiplier))
        self.smooth_bridge_from_level = int(smooth_bridge_from_level)
        self.smooth_bridge_to_level = int(smooth_bridge_to_level)
        self.smooth_bridge_stages = max(0, int(smooth_bridge_stages))

        goal = (self.env.grid_size - 2, self.env.grid_size - 2)
        start = (1, 1)
        full_start_distance = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        computed_max_level = int(np.ceil(full_start_distance / float(self.radius_step)))
        self.max_level = int(max_level) if max_level is not None else max(1, computed_max_level)

        if self.train_fixed_level > 0:
            self.level = int(np.clip(self.train_fixed_level, 1, self.max_level))
        self.fixed_start_mode = self.level >= self.max_level and self.train_fixed_level <= 0
        self.recent_success: deque[float] = deque(maxlen=self.success_window)
        self.episodes_seen = 0
        self.level_up_events: List[Dict[str, Any]] = []
        self.standard_ratio_boost_events: List[Dict[str, Any]] = []
        self.bridge_stage = 0
        self.last_sampled_standard_start = False

    def _effective_radius(self) -> int:
        base_radius = int(self.level * self.radius_step)
        if (
            self.smooth_bridge_stages > 0
            and self.level == self.smooth_bridge_from_level
            and self.smooth_bridge_to_level == self.smooth_bridge_from_level + 1
        ):
            max_bonus = max(0, self.radius_step - 1)
            bonus = int(np.clip(self.bridge_stage, 0, max_bonus))
            return base_radius + bonus
        return base_radius

    def _curriculum_info(self) -> Dict[str, float]:
        return {
            "curriculum_level": float(self.level),
            "curriculum_bridge_stage": float(self.bridge_stage),
            "curriculum_effective_radius": float(self._effective_radius()),
            "curriculum_recent_success_rate": float(np.mean(self.recent_success)) if self.recent_success else 0.0,
            "curriculum_fixed_start": float(1.0 if self.fixed_start_mode else 0.0),
            "curriculum_sample_standard_start": float(1.0 if self.last_sampled_standard_start else 0.0),
            "curriculum_standard_start_ratio": float(self.current_standard_start_ratio),
        }

    def _all_spawn_candidates(self, radius: int) -> List[Tuple[int, int]]:
        target = tuple(map(int, self.env.base_env.target_position.tolist()))
        candidates: List[Tuple[int, int]] = []

        for x in range(1, self.env.grid_size - 1):
            for y in range(1, self.env.grid_size - 1):
                if (x, y) == target:
                    continue
                cell = self.env.base_env.maze.get_cell_item(x, y)
                if cell is not None:
                    continue
                dist = abs(x - target[0]) + abs(y - target[1])
                if dist <= radius:
                    candidates.append((x, y))

        return candidates

    def _set_spawn(self, spawn_pos: Tuple[int, int]) -> None:
        x, y = spawn_pos
        self.env.base_env.robot_position = np.array([x, y], dtype=int)
        self.env.base_env.robot_direction = 2
        self.env.base_env.robot_camera_view = self.env.base_env.get_robot_camera_view()

        # Ensure spawned cell is not occupied by a virtual entity.
        if hasattr(self.env, "_remove_entity_at"):
            self.env._remove_entity_at((x, y))
        else:
            self.env.entity_map.pop((x, y), None)

        target = tuple(map(int, self.env.base_env.target_position.tolist()))
        self.env.prev_target_dist = float(abs(x - target[0]) + abs(y - target[1]))
        if hasattr(self.env, "_true_distance_to_target"):
            self.env.prev_true_target_dist = float(self.env._true_distance_to_target((x, y), target))
        self.env.prev_bribable_dist = self.env._nearest_entity_distance((x, y), BRIBABLE_CLASSES)

        self.env.trajectory = [(x, y)]
        self.env.position_window.clear()
        self.env.position_window.append((x, y))
        self.env.short_position_window.clear()
        self.env.short_position_window.append((x, y))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = options or {}
        forced_level = options.get("curriculum_level")
        freeze_curriculum = bool(options.get("freeze_curriculum", False))
        force_standard_start = bool(options.get("force_standard_start", False))

        obs, info = self.env.reset(seed=seed, options=options)

        # Force fixed-level curriculum for pretraining stage.
        if forced_level is None and self.train_fixed_level > 0:
            forced_level = int(np.clip(self.train_fixed_level, 1, self.max_level))
            freeze_curriculum = True
            self.fixed_start_mode = False

        sampled_standard_start = False
        if forced_level is None and force_standard_start:
            sampled_standard_start = True
        elif (
            forced_level is None
            and self.mixed_sampling_enabled
            and self.train_fixed_level <= 0
            and not self.fixed_start_mode
        ):
            sampled_standard_start = bool(self.env.rng.random() < self.current_standard_start_ratio)

        if sampled_standard_start:
            self.last_sampled_standard_start = True
            self.env.potential_scale_multiplier = float(self.standard_potential_multiplier)
            info.update(self._curriculum_info())
            return obs, info

        # Switch to standard starting-point mode when level is high enough.
        if self.fixed_start_mode and forced_level is None:
            self.last_sampled_standard_start = True
            self.env.potential_scale_multiplier = float(self.standard_potential_multiplier)
            info.update(self._curriculum_info())
            return obs, info

        level_to_use = int(forced_level) if forced_level is not None else int(self.level)
        if level_to_use >= self.max_level and not freeze_curriculum:
            self.fixed_start_mode = True
            self.level = self.max_level
            self.last_sampled_standard_start = True
            self.env.potential_scale_multiplier = float(self.standard_potential_multiplier)
            info.update(self._curriculum_info())
            return obs, info

        radius = int(level_to_use * self.radius_step)
        if (
            forced_level is None
            and self.smooth_bridge_stages > 0
            and level_to_use == self.smooth_bridge_from_level
            and self.smooth_bridge_to_level == self.smooth_bridge_from_level + 1
        ):
            radius = self._effective_radius()
        candidates = self._all_spawn_candidates(radius=radius)
        spawn = (1, 1)
        if candidates:
            idx = int(self.env.rng.integers(low=0, high=len(candidates)))
            spawn = candidates[idx]

        self._set_spawn(spawn)
        obs = self.env._build_observation()
        self.last_sampled_standard_start = False
        self.env.potential_scale_multiplier = 1.0

        info.update(
            {
                "curriculum_level": float(level_to_use),
                "curriculum_bridge_stage": float(self.bridge_stage),
                "curriculum_spawn_x": float(spawn[0]),
                "curriculum_spawn_y": float(spawn[1]),
                "curriculum_radius": float(radius),
                "curriculum_effective_radius": float(radius),
                "curriculum_recent_success_rate": float(np.mean(self.recent_success)) if self.recent_success else 0.0,
                "curriculum_fixed_start": float(1.0 if self.fixed_start_mode else 0.0),
            }
        )
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.update(self._curriculum_info())
        return obs, reward, terminated, truncated, info

    def update_from_episode(self, is_success: float) -> Dict[str, Any]:
        self.episodes_seen += 1
        self.recent_success.append(float(is_success))

        recent_mean = float(np.mean(self.recent_success)) if self.recent_success else 0.0
        leveled_up = False
        switched_to_fixed_start = False

        if self.train_fixed_level > 0:
            return {
                "episodes_seen": int(self.episodes_seen),
                "level": int(self.level),
                "recent_success_rate": float(recent_mean),
                "leveled_up": False,
                "fixed_start_mode": False,
                "switched_to_fixed_start": False,
            }

        if not self.fixed_start_mode and len(self.recent_success) >= self.success_window:
            if recent_mean > self.levelup_threshold:
                if (
                    self.smooth_bridge_stages > 0
                    and self.level == self.smooth_bridge_from_level
                    and self.smooth_bridge_to_level == self.smooth_bridge_from_level + 1
                    and self.bridge_stage < self.smooth_bridge_stages
                ):
                    self.bridge_stage += 1
                    self.level_up_events.append(
                        {
                            "episode": int(self.episodes_seen),
                            "new_level": int(self.level),
                            "recent_success_rate": float(recent_mean),
                            "fixed_start_mode": bool(self.fixed_start_mode),
                            "bridge_stage": int(self.bridge_stage),
                            "effective_radius": int(self._effective_radius()),
                        }
                    )
                else:
                    self.level += 1
                    self.bridge_stage = 0
                    leveled_up = True
                    if self.level >= self.max_level:
                        self.level = self.max_level
                        self.fixed_start_mode = True
                        switched_to_fixed_start = True
                    self.level_up_events.append(
                        {
                            "episode": int(self.episodes_seen),
                            "new_level": int(self.level),
                            "recent_success_rate": float(recent_mean),
                            "fixed_start_mode": bool(self.fixed_start_mode),
                            "bridge_stage": int(self.bridge_stage),
                            "effective_radius": int(self._effective_radius()),
                        }
                    )
                self.recent_success.clear()

        # Stall-triggered weighting: if k2 stays very strong without progression, increase standard sampling by +10%.
        if (
            self.mixed_sampling_enabled
            and self.level == 2
            and not self.fixed_start_mode
            and not leveled_up
            and self.bridge_stage == 0
            and len(self.recent_success) >= self.success_window
            and recent_mean >= 0.8
        ):
            prev_ratio = float(self.current_standard_start_ratio)
            self.current_standard_start_ratio = float(np.clip(prev_ratio + 0.1, 0.0, 1.0))
            self.standard_ratio_boost_events.append(
                {
                    "episode": int(self.episodes_seen),
                    "level": int(self.level),
                    "recent_success_rate": float(recent_mean),
                    "ratio_before": float(prev_ratio),
                    "ratio_after": float(self.current_standard_start_ratio),
                    "reason": "k2_stall_high_success_no_progress",
                }
            )
            self.recent_success.clear()

        return {
            "episodes_seen": int(self.episodes_seen),
            "level": int(self.level),
            "recent_success_rate": float(recent_mean),
            "leveled_up": bool(leveled_up),
            "fixed_start_mode": bool(self.fixed_start_mode),
            "switched_to_fixed_start": bool(switched_to_fixed_start),
            "standard_start_ratio": float(self.current_standard_start_ratio),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "initial_level": int(self.initial_level),
            "train_fixed_level": int(self.train_fixed_level),
            "mixed_sampling_enabled": bool(self.mixed_sampling_enabled),
            "standard_start_ratio": float(self.standard_start_ratio),
            "current_standard_start_ratio": float(self.current_standard_start_ratio),
            "standard_potential_multiplier": float(self.standard_potential_multiplier),
            "level": int(self.level),
            "max_level": int(self.max_level),
            "radius_step": int(self.radius_step),
            "bridge_stage": int(self.bridge_stage),
            "smooth_bridge_from_level": int(self.smooth_bridge_from_level),
            "smooth_bridge_to_level": int(self.smooth_bridge_to_level),
            "smooth_bridge_stages": int(self.smooth_bridge_stages),
            "effective_radius": int(self._effective_radius()),
            "success_window": int(self.success_window),
            "levelup_threshold": float(self.levelup_threshold),
            "fixed_start_mode": bool(self.fixed_start_mode),
            "episodes_seen": int(self.episodes_seen),
            "recent_success_rate": float(np.mean(self.recent_success)) if self.recent_success else 0.0,
            "level_up_events": list(self.level_up_events),
            "standard_ratio_boost_events": list(self.standard_ratio_boost_events),
        }


def unwrap_curriculum_env(env: Any) -> Optional[CurriculumWrapper]:
    current = env
    max_depth = 16
    for _ in range(max_depth):
        if isinstance(current, CurriculumWrapper):
            return current
        if hasattr(current, "env"):
            current = current.env
            continue
        return None
    return None


class CurriculumLevelUpCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.curriculum_env: Optional[CurriculumWrapper] = None

    def _on_training_start(self) -> None:
        vec_env = self.training_env
        if hasattr(vec_env, "envs") and len(vec_env.envs) > 0:
            self.curriculum_env = unwrap_curriculum_env(vec_env.envs[0])

    def _on_step(self) -> bool:
        if self.curriculum_env is None:
            return True

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not bool(done):
                continue
            success = float(info.get("is_success", 0.0))
            state = self.curriculum_env.update_from_episode(success)
            print(
                f"[Curriculum] episode={state['episodes_seen']} k={state['level']} "
                f"bridge_stage={self.curriculum_env.bridge_stage} "
                f"radius={self.curriculum_env._effective_radius()} "
                f"std_ratio={state.get('standard_start_ratio', self.curriculum_env.current_standard_start_ratio):.2f} "
                f"recent{self.curriculum_env.success_window}_success={state['recent_success_rate']:.3f} "
                f"fixed_start={state['fixed_start_mode']}"
            )
            if state["leveled_up"]:
                print(
                    f"[Curriculum] level-up -> k={state['level']} "
                    f"(recent_success={state['recent_success_rate']:.3f})"
                )
                if state["switched_to_fixed_start"]:
                    print("[Curriculum] reached max level, switching to standard fixed start mode")
            elif self.curriculum_env.bridge_stage > 0:
                print(
                    f"[Curriculum] smooth-transition k={state['level']} stage={self.curriculum_env.bridge_stage}/"
                    f"{self.curriculum_env.smooth_bridge_stages} radius={self.curriculum_env._effective_radius()}"
                )

            boost_events = self.curriculum_env.standard_ratio_boost_events
            if boost_events:
                last_boost = boost_events[-1]
                if int(last_boost.get("episode", -1)) == int(state["episodes_seen"]):
                    print(
                        "[Curriculum] stall-trigger ratio boost "
                        f"{last_boost.get('ratio_before', 0.0):.2f} -> {last_boost.get('ratio_after', 0.0):.2f} "
                        f"at k={last_boost.get('level', state['level'])}"
                    )

        return True
