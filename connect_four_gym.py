import gymnasium as gym
from gymnasium import spaces
import numpy as np
from kaggle_environments import make


class ConnectFourGym(gym.Env):
    """
    ConnectX -> Gymnasium 环境适配器。

    关键约定：
    - 返回给 RL 模型的观测始终是相对棋盘：
        自己 = +1
        对手 = -1
        空位 =  0
    - action_masks() 返回合法动作掩码：
        True  = 合法列
        False = 已满列
    """

    metadata = {"render_modes": []}

    def __init__(self, agent2="random"):
        super().__init__()

        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])

        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns

        self.action_space = spaces.Discrete(self.columns)

        # 相对编码：0 / -1 / +1
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.rows * self.columns,),
            dtype=np.float32,
        )

        self.obs = None

    @staticmethod
    def _get_value(obs, key):
        """兼容 Kaggle Observation 对象与普通 dict。"""
        return getattr(obs, key) if hasattr(obs, key) else obs[key]

    def _relative_board(self, obs):
        """
        将 Kaggle 原始棋盘编码 0/1/2 转换成当前行动者视角的 0/+1/-1。

        例如：
        当前玩家 mark=1：
            1 -> +1, 2 -> -1

        当前玩家 mark=2：
            2 -> +1, 1 -> -1
        """
        board = np.asarray(self._get_value(obs, "board"), dtype=np.float32)
        my_mark = self._get_value(obs, "mark")
        opp_mark = 3 - my_mark

        state = np.zeros_like(board, dtype=np.float32)
        state[board == my_mark] = 1.0
        state[board == opp_mark] = -1.0
        return state

    def action_masks(self):
        """
        给 MaskablePPO 使用。

        返回长度为 7 的 bool 数组：
        True  表示可落子；
        False 表示该列已满。

        ConnectX 的棋盘是 row-major 展平；
        前 7 个元素正好是棋盘第一行。
        """
        if self.obs is None:
            return np.ones(self.columns, dtype=bool)

        board = self._get_value(self.obs, "board")
        return np.asarray(
            [board[col] == 0 for col in range(self.columns)],
            dtype=bool,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.obs = self.env.reset()
        board = self._relative_board(self.obs)

        info = {
            "action_mask": self.action_masks(),
        }
        return board, info

    def step(self, action):
        action = int(action)

        # MaskablePPO 正常情况下永远不会进入这里。
        # 保留保护代码，方便定位环境或推理调用错误。
        if not self.action_masks()[action]:
            raise ValueError(
                f"非法动作：第 {action} 列已满。"
                "请确认训练/推理时正确使用了 action_masks。"
            )

        self.obs, old_reward, done, _ = self.env.step(action)

        # 本课先保持你的原奖励逻辑，下一课再处理 reward shaping。
        reward = self.change_reward(old_reward, done)

        board = self._relative_board(self.obs)

        terminated = bool(done)
        truncated = False
        info = {
            "action_mask": self.action_masks(),
        }

        return board, reward, terminated, truncated, info

    @staticmethod
    def change_reward(old_reward, done):
        """暂时保留 V0 奖励设计；第 2 课会改为更合理的终局奖励。"""
        if done:
            if old_reward == 1:
                return 1.0
            elif old_reward == -1:
                return -1.0
            return 0.0

        return 1.0 / 42.0