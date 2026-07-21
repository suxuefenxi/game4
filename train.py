import os
import numpy as np
import random as _random

from heuristic_agent import heuristic_agent
from connect_four_gym import ConnectFourGym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


SEED = 42
N_ENVS = 4

RUN_NAME = "v4_tactical_ppo"
MODEL_DIR = f"runs/{RUN_NAME}/models"
LOG_DIR = f"runs/{RUN_NAME}/tensorboard"
EVAL_LOG_DIR = f"runs/{RUN_NAME}/eval"


def _random_agent(obs, config):
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    valid = [c for c in range(config.columns) if grid[0][c] == 0]
    return _random.choice(valid) if valid else -1


def make_mixed_opponent(agents_weights=None):
    """
    本课暂时仍是 random + heuristic。
    后续课程会把这里升级为 Minimax 与 self-play 对手池。
    """
    if agents_weights is None:
        agents_weights = [
            (heuristic_agent, 0.5),
            (_random_agent, 0.5),
        ]

    agents, weights = zip(*agents_weights)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    def opponent(obs, config):
        chosen = np.random.choice(agents, p=weights)
        return chosen(obs, config)

    return opponent


def make_env(rank, seed=SEED):
    """
    SubprocVecEnv 使用的环境工厂。

    rank 用于让不同进程使用不同 seed。
    """
    def _init():
        env = ConnectFourGym(agent2=make_mixed_opponent())
        env = Monitor(env)

        # 注意：这里只是初始化随机数状态；
        # 真正每局的先后手由 ConnectFourGym.reset() 随机决定。
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
        eval_freq=25_000,
        n_eval_episodes=100,
        deterministic=True,
        render=False,
    )

    # 每 100,000 个总环境步保存一次。
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // N_ENVS,
        save_path=MODEL_DIR,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("开始训练 V4：tactical mask + PPO experiment framework ...")

    model.learn(
        total_timesteps=2_000_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        tb_log_name=RUN_NAME,
    )

    model.save(f"{MODEL_DIR}/final_model")
    train_env.close()
    eval_env.close()

    print(f"训练完成。最佳模型：{MODEL_DIR}/best_model.zip")
    print(f"最终模型：{MODEL_DIR}/final_model.zip")