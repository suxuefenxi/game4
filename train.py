import logging
import warnings
import os

warnings.filterwarnings("ignore", module="litellm")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
os.environ["LITELLM_LOG"] = "ERROR"

import numpy as np
import random as _random

from heuristic_agent import heuristic_agent
from connect_four_gym import ConnectFourGym

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


SEED = 42


def _random_agent(obs, config):
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    valid = [c for c in range(config.columns) if grid[0][c] == 0]
    return _random.choice(valid) if valid else -1


def make_mixed_opponent(agents_weights=None):
    if agents_weights is None:
        agents_weights = [
            (heuristic_agent, 0.5),
            (_random_agent, 0.5),
        ]

    agents, weights = zip(*agents_weights)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    def opponent(obs, config):
        selected_agent = np.random.choice(agents, p=weights)
        return selected_agent(obs, config)

    return opponent


def make_env(rank):
    """每个子进程创建自己的环境。"""

    def _init():
        mixed_opp = make_mixed_opponent()

        env = ConnectFourGym(agent2=mixed_opp)
        env = Monitor(env)

        # 不同 worker 使用不同随机种子
        env.reset(seed=SEED + rank)
        return env

    return _init


if __name__ == "__main__":
    n_envs = 4
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=SEED,

        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    print("开始训练 V2：random first/second + sparse terminal reward ...")

    model.learn(
        total_timesteps=1_000_000,
        progress_bar=False,
    )

    model.save("ppo_connectx_v2_twosided")
    env.close()

    print("训练完成：ppo_connectx_v2_masked.zip")