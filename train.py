import os
import numpy as np
import random as _random

from heuristic_agent import heuristic_agent
from connect_four_gym import ConnectFourGym
from minimax_agent import make_minimax_agent
from pathlib import Path
from league import (
    make_maskable_ppo_agent,
    build_opponent_selector,
    find_league_models,
)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


SEED = 42
N_ENVS = 4

RUN_NAME = "v6_gen001"
MODEL_DIR = f"runs/{RUN_NAME}/models"
LOG_DIR = f"runs/{RUN_NAME}/tensorboard"
EVAL_LOG_DIR = f"runs/{RUN_NAME}/eval"


def _random_agent(obs, config):
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    valid = [c for c in range(config.columns) if grid[0][c] == 0]
    return _random.choice(valid) if valid else -1


# 训练专用：时间上限必须短。
# 它的目的不是当最强 Minimax，而是为 PPO 制造更高质量局面。
minimax_d3_fast = make_minimax_agent(
    depth=3,
    time_limit=0.03,
)


def make_episode_opponent_selector():
    """
    返回一个“每局调用一次”的对手选择器。

    selector(rng) -> (Kaggle agent, 名称)
    """

    opponents = [
        ("random", _random_agent, 0.40),
        ("heuristic", heuristic_agent, 0.45),
        ("minimax_d3_fast", minimax_d3_fast, 0.15),
    ]

    names = [item[0] for item in opponents]
    agents = [item[1] for item in opponents]

    weights = np.asarray([item[2] for item in opponents], dtype=np.float64)
    weights /= weights.sum()

    def select_opponent(rng):
        # 使用环境自己的 rng，保证 seed 时可复现。
        index = int(rng.choice(len(agents), p=weights))
        return agents[index], names[index]

    return select_opponent


LEAGUE_DIR = Path("league")


def make_league_selector():
    """
    每个环境 worker 创建时调用一次。

    历史模型在这里加载并冻结；每一个 episode 再由 selector
    以给定权重抽取其中一个对手。
    """
    specs = [
        ("random", _random_agent, 0.20),
        ("heuristic", heuristic_agent, 0.30),
        ("minimax_d3_fast", minimax_d3_fast, 0.15),
    ]

    history_paths = find_league_models(
        league_dir=LEAGUE_DIR,
        max_models=3,
    )

    if not history_paths:
        raise RuntimeError(
            "league/ 中没有历史模型。"
            "请先复制一个 benchmark 最强的 .zip 到 league/。"
        )

    # 历史模型总共占 35%。
    historical_total_weight = 0.35
    each_weight = historical_total_weight / len(history_paths)

    for path in history_paths:
        historical_agent = make_maskable_ppo_agent(path)

        # 例如 gen_000_v4_best.zip -> gen_000_v4_best
        name = path.stem

        specs.append((name, historical_agent, each_weight))

    return build_opponent_selector(specs)

def make_env(rank, seed=SEED):
    def _init():
        env = ConnectFourGym(
            opponent_selector=make_league_selector()
        )
        env = Monitor(env)

        env.reset(seed=seed + rank)
        return env

    return _init


def make_eval_env():
    """
    独立评估环境。

    不与训练环境共享实例，避免评估过程污染训练 episode。
    """
    return DummyVecEnv([make_env(rank=10_000)])


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(EVAL_LOG_DIR, exist_ok=True)

    train_env = SubprocVecEnv(
        [make_env(rank=i) for i in range(N_ENVS)]
    )

    eval_env = make_eval_env()

    CHAMPION_PATH = "league/gen_000_v4_best.zip"

    if Path(CHAMPION_PATH).exists():
        # 加载冠军模型继续训练（延续之前的策略）
        model = MaskablePPO.load(
            CHAMPION_PATH,
            env=train_env,
            device="auto",
            tensorboard_log=LOG_DIR,
        )
        model.set_random_seed(SEED)
        print(f"已加载冠军模型：{CHAMPION_PATH}")
    else:
        print("未找到冠军模型，从零开始训练...")
        # 4 个环境 × 512 步 = 每轮 PPO 更新收集 2048 个样本。
        # batch_size=256 能整除 2048，避免不完整 minibatch。
        model = MaskablePPO(
            policy="MlpPolicy",
            env=train_env,
            seed=SEED,
            verbose=1,

            learning_rate=3e-4,
            n_steps=512,
            batch_size=256,
            n_epochs=4,

            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,

            policy_kwargs=dict(
                net_arch=dict(
                    pi=[128, 128],
                    vf=[128, 128],
                )
            ),

            tensorboard_log=LOG_DIR,
            device="auto",
        )

    # 必须使用 MaskableEvalCallback，而不是 SB3 标准 EvalCallback。
    # 否则评估时不会将 action mask 传给策略。
    eval_callback = MaskableEvalCallback(
        eval_env=eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=EVAL_LOG_DIR,
        eval_freq=20_000,
        n_eval_episodes=100,
        deterministic=True,
        render=False,
    )

    # 每 100,000 个总环境步保存一次。
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // N_ENVS,
        save_path=MODEL_DIR,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("开始训练...")

    model.learn(
        total_timesteps=200_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False,
        tb_log_name=RUN_NAME,
    )

    model.save(f"{MODEL_DIR}/final_model")
    train_env.close()
    eval_env.close()

    print(f"训练完成。最佳模型：{MODEL_DIR}/best_model.zip")
    print(f"最终模型：{MODEL_DIR}/final_model.zip")