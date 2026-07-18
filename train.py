import logging
import warnings
import os

# ── 屏蔽 Python warnings 模块的警告（LiteLLM 主要用这个） ──
warnings.filterwarnings("ignore", module="litellm")

# ── 同时屏蔽 logging 模块的警告（双保险） ──
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
# 某些子 logger（如 litellm.main）也需要抑制
for name in logging.root.manager.loggerDict:
    if "litellm" in name.lower():
        logging.getLogger(name).setLevel(logging.ERROR)

# 关闭 LiteLLM 自身的 verbosity 开关（如果有同步日志流）
os.environ["LITELLM_LOG"] = "ERROR"

import numpy as np
import random as _random
from heuristic_agent import heuristic_agent
from connect_four_gym import ConnectFourGym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


def _random_agent(obs, config):
    """纯随机 AI：从合法列中随机选一个"""
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    valid = [c for c in range(config.columns) if grid[0][c] == 0]
    return _random.choice(valid) if valid else -1


def make_mixed_opponent(agents_weights=None):
    """
    创建混合对手函数，每次被调用时按权重随机选一个策略出棋

    防止 PPO 背板单一对手——随机性和 heuristic 交叉出现，
    让策略学到更通用的下法。

    参数
    ----------
    agents_weights : [(agent_fn, weight), ...], 可选
        默认 50% heuristic + 50% random
    """
    if agents_weights is None:
        agents_weights = [(heuristic_agent, 0.5), (_random_agent, 0.5)]

    agents, weights = zip(*agents_weights)
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    def opponent(obs, config):
        agent = np.random.choice(agents, p=weights)
        return agent(obs, config)

    return opponent


def make_env():
    """创建单个环境（SubprocVecEnv 的 worker 工厂）"""
    # 每个 worker 独立创建混合对手，避免共享 np.random 状态
    mixed_opp = make_mixed_opponent()

    def _init():
        env = ConnectFourGym(agent2=mixed_opp)
        env = Monitor(env)
        return env
    return _init

if __name__ == '__main__':
    env = SubprocVecEnv([make_env() for _ in range(4)])

    # 2. 实例化 PPO 智能体
    # 使用 MLP 策略，网络层结构采用 SB3 默认架构 (pi=[64, 64], vf=[64, 64])
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003, 
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99
    )

    # 3. 开始训练
    print("开始训练...")
    model.learn(total_timesteps=1_000_000) 
    print("训练完成！")

    # 4. 保存模型
    model.save("ppo_connectx")