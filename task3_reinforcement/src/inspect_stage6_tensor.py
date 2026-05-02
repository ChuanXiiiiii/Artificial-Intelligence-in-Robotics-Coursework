from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from common import read_json, resolve_task3_root
from hero_task3_env import HeroTask3Env
from train_agents import TinyCNN


class DebugTinyCNN(TinyCNN):
    def __init__(self, observation_space, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self._printed = False

    def forward(self, observations):
        if not self._printed:
            print(f"TinyCNN input shape: {tuple(observations.shape)}")
            self._printed = True
        return super().forward(observations)


def main() -> None:
    task3_root = resolve_task3_root()
    env_cfg = read_json(Path(task3_root) / "configs" / "env_stage6_rewardH.json")
    frame_stack = int(env_cfg.get("frame_stack", 1))

    env_kwargs = dict(env_cfg)
    env_kwargs["render_mode"] = None
    env_kwargs.pop("frame_stack", None)

    env = HeroTask3Env(seed=0, **env_kwargs)
    if frame_stack > 1:
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=frame_stack, channels_order="first")

    policy_kwargs = {
        "features_extractor_class": DebugTinyCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "normalize_images": False,
    }

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        gamma=0.99,
        device="cpu",
        verbose=0,
    )

    obs = env.reset()
    for _ in range(3):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, done, _info = env.step(action)
        if isinstance(done, np.ndarray) and bool(done[0]):
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
