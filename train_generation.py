#!/usr/bin/env python
"""
训练一代 PPO agent。

从 --init-model 加载权重继续训练，保存 best_model.zip 和 final_model.zip。

用法：
    python train_generation.py \
        --init-model league/champion.zip \
        --output-dir runs/gen_001 \
        --timesteps 200000 \
        --league-dir league
"""

import argparse
import numpy as np
import random as _random
from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from connect_four_gym import ConnectFourGym
from heuristic_agent import heuristic_agent
from minimax_agent import make_minimax_agent
from league import (
    make_maskable_ppo_agent,
    build_opponent_selector,
    find_league_models,
)

SEED = 42
N_ENVS = 4


def _random_agent(obs, config):
    """随机合法动作。"""
    board = np.asarray(obs.board).reshape(config.rows, config.columns)
    valid = [c for c in range(config.columns) if board[0][c] == 0]
    return _random.choice(valid) if valid else -1


# 训练专用：时间上限必须短，只为 PPO 制造更高质量局面。
_minimax_d3_fast = make_minimax_agent(depth=3, time_limit=0.05)
_minimax_d4_fast = make_minimax_agent(depth=4, time_limit=0.10)
_minimax_d5_fast = make_minimax_agent(depth=5, time_limit=0.25)


def make_league_selector(league_dir):
    """
    构建训练环境的对手池。

    - 固定对手占 65%（random / heuristic / minimax_d3_fast）
    - 历史晋升模型占 35%（按权重均分）
    """
    league_dir = Path(league_dir)

    specs = [
        ("random", _random_agent, 0.05),
        ("heuristic", heuristic_agent, 0.10),
        ("minimax_d3_fast", _minimax_d3_fast, 0.40),
        ("minimax_d4_fast", _minimax_d4_fast, 0.40),
        ("minimax_d5_fast", _minimax_d5_fast, 0.05),
    ]

    history_paths = find_league_models(league_dir, max_models=3)

    if not history_paths:
        raise RuntimeError(
            f"{league_dir}/ 中没有 gen_*.zip 历史模型。\n"
            "请先运行初始化：cp league/gen_000_v4_best.zip league/gen_000.zip"
        )

    # 历史模型总共占 35%
    historical_total_weight = 0.35
    each_weight = historical_total_weight / len(history_paths)

    for path in history_paths:
        frozen_agent = make_maskable_ppo_agent(path)
        specs.append((path.stem, frozen_agent, each_weight))

    return build_opponent_selector(specs)


def make_env(rank, league_dir, seed=SEED):
    """
    SubprocVecEnv 使用的环境工厂。
    """
    def _init():
        env = ConnectFourGym(
            opponent_selector=make_league_selector(league_dir)
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(
        description="训练一代 PPO agent（从 champion 继续训练）"
    )
    parser.add_argument(
        "--init-model", type=str, required=True,
        help="初始模型路径（通常指向 league/champion.zip）",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="输出目录（含 models / tensorboard / eval 子目录）",
    )
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="本代训练总步数（默认 200000）",
    )
    parser.add_argument(
        "--league-dir", type=str, default="league",
        help="league 目录路径（默认 league）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    log_dir = output_dir / "tensorboard"
    eval_dir = output_dir / "eval"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"初始化模型: {args.init_model}")
    print(f"输出目录:   {output_dir}")
    print(f"训练步数:   {args.timesteps}")
    print(f"league 目录: {args.league_dir}")

    # --- 构建环境（这一步最慢：启动 4 个子进程，各加载 PyTorch/SB3） ---
    print("\n正在启动训练环境（4 个子进程）...")
    train_env = SubprocVecEnv(
        [make_env(rank=i, league_dir=args.league_dir) for i in range(N_ENVS)]
    )
    print("  训练环境就绪")

    print("正在启动评估环境...")
    eval_env = DummyVecEnv(
        [make_env(rank=10_000, league_dir=args.league_dir)]
    )
    print("  评估环境就绪")

    # --- 加载模型 ---
    print("正在加载 champion 模型...")
    model = MaskablePPO.load(
        args.init_model,
        env=train_env,
        device="auto",
        tensorboard_log=str(log_dir),
    )
    model.set_random_seed(SEED)
    print("  模型加载完成")

    # --- Callbacks ---
    eval_callback = MaskableEvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(eval_dir),
        eval_freq=20_000 // N_ENVS,  # 每 20k 步评估一次
        n_eval_episodes=100,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // N_ENVS,
        save_path=str(model_dir),
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # --- 训练（每 20k 步会打印一次 eval 结果） ---
    print("\n开始训练...")
    print("（每 20000 步会自动评估一次，打印当前胜率）\n")

    model.learn(
        total_timesteps=args.timesteps,
        # total_timesteps 始终是"增量"——不管之前训练了多少步，
        # 每次调用 learn 都会新训练 args.timesteps 步。
        #
        # reset_num_timesteps=True：内部 num_timesteps 从 0 开始，
        # 这样 _current_progress_remaining 在 [0,1] 之间，LR 调度正确。
        # 模型权重保留之前的训练成果，只是计数器重启。
        reset_num_timesteps=True,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False,
        tb_log_name=output_dir.name,
    )

    model.save(str(model_dir / "final_model"))
    train_env.close()
    eval_env.close()

    print(f"训练完成。")
    print(f"  最佳模型: {model_dir / 'best_model.zip'}")
    print(f"  最终模型: {model_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
