import gymnasium as gym
from gymnasium import spaces
import numpy as np
from kaggle_environments import make

class ConnectFourGym(gym.Env):
    def __init__(self, agent2="random"):
        super(ConnectFourGym, self).__init__()
        # 创建 Kaggle 原生环境
        ks_env = make("connectx", debug=True)
        # 用指定对手（默认random）初始化训练环境
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        
        # 动作空间：选择 7 列中的一列
        self.action_space = spaces.Discrete(self.columns)
        
        # 观测空间：扁平化的 42 维向量，更适合基础 MLP 策略
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.rows * self.columns,), dtype=np.float32
        )
        self.obs = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.obs = self.env.reset()
        
        # 兼容不同版本 kaggle_environments 的取值方式
        board_list = self.obs.board if hasattr(self.obs, 'board') else self.obs['board']
        board = np.array(board_list, dtype=np.float32)
        info = {}
        return board, info
        
    def step(self, action):
        board_list = self.obs.board if hasattr(self.obs, 'board') else self.obs['board']
        # 验证落子是否合法
        is_valid = (board_list[int(action)] == 0)
        
        if is_valid:
            self.obs, old_reward, done, info = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:
            # 非法落子给予极高惩罚并直接终止
            reward = -10.0
            done = True
            
        board_list = self.obs.board if hasattr(self.obs, 'board') else self.obs['board']
        board = np.array(board_list, dtype=np.float32)
        
        terminated = done
        truncated = False
        info_dict = {}
        
        return board, reward, terminated, truncated, info_dict
        
    def change_reward(self, old_reward, done):
        """简单的奖励塑造（Reward Shaping）"""
        if done:
            if old_reward == 1:    # 获胜
                return 1.0
            elif old_reward == -1:  # 落败
                return -1.0
            else:                  # 平局
                return 0.0
        else:
            # 基础步长奖励：每活下来一步给予 1/42 的微小正奖励，稳定训练
            return 1.0 / 42.0