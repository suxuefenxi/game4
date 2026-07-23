import gymnasium as gym
from gymnasium import spaces
import numpy as np
from kaggle_environments import make

from tactical import tactical_action_mask


class ConnectFourGym(gym.Env):
    """
    ConnectX Gymnasium 训练环境。

    V2 改动：
    1. 每局随机决定训练 agent 是先手还是后手；
    2. 仅使用终局奖励：
         赢 +1，输 -1，平 0，非终局 0；
    3. 继续使用相对棋盘表示和 action mask。
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_selector=None):
        super().__init__()

        # 保留一个 Kaggle 原生环境实例。
        self.ks_env = make("connectx", debug=True)

        self.rows = self.ks_env.configuration.rows
        self.columns = self.ks_env.configuration.columns

        # opponent_selector(rng) -> (agent, opponent_name)
        # 如果没传入，则始终使用 random。
        self.opponent_selector = opponent_selector
        self.current_opponent = "random"
        self.current_opponent_name = "random"
        self.env = None
        self.obs = None

        # 0~6：选择落子列
        self.action_space = spaces.Discrete(self.columns)

        # 相对棋盘编码：自己=+1，对手=-1，空位=0
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.rows * self.columns,),
            dtype=np.float32,
        )

        # 当前这局训练 agent 是否先手。
        self.is_first_player = True

    @staticmethod
    def _get_value(obs, key):
        """兼容 Kaggle 的 Observation 对象和 dict。"""
        return getattr(obs, key) if hasattr(obs, key) else obs[key]

    def _make_trainer(self):
        """按本局先后手创建 Kaggle trainer。"""
        if self.is_first_player:
            self.env = self.ks_env.train([
                None,
                self.current_opponent,
            ])
        else:
            self.env = self.ks_env.train([
                self.current_opponent,
                None,
            ])

    def _relative_board(self, obs):
        """
        固定以“当前待行动的训练 agent”视角编码棋盘：

        自己：+1
        对手：-1
        空位： 0
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
        V3：合法性 + 基础战术动作 mask。

        True：
            该动作可以由策略选择。
        False：
            已满列，或该动作会在存在安全走法时让对手下一手直接获胜。
        """
        if self.obs is None:
            return np.ones(self.columns, dtype=bool)

        board = self._get_value(self.obs, "board")
        my_mark = self._get_value(self.obs, "mark")

        return tactical_action_mask(
            board=board,
            mark=my_mark,
            config=self.ks_env.configuration,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. 每局随机决定训练 agent 的先后手。
        self.is_first_player = bool(self.np_random.integers(0, 2))

        # 2. 每局只抽取一次对手，整局保持不变。
        if self.opponent_selector is None:
            self.current_opponent = "random"
            self.current_opponent_name = "random"
        else:
            self.current_opponent, self.current_opponent_name = (
                self.opponent_selector(self.np_random)
            )

        # 3. 用当前对手创建这一局的 trainer。
        self._make_trainer()

        # 若训练者是后手，Kaggle 会先让对手走一步，
        # 再返回轮到训练者行动时的观测。
        self.obs = self.env.reset()

        board = self._relative_board(self.obs)

        info = {
            "action_mask": self.action_masks(),
            "is_first_player": self.is_first_player,
            "agent_mark": self._get_value(self.obs, "mark"),
            "opponent_name": self.current_opponent_name,
        }
        return board, info

    def step(self, action):
        action = int(action)

        # MaskablePPO 正常不会选择非法动作。
        # 此处保留报错，便于尽早发现训练或部署代码问题。
        if not self.action_masks()[action]:
            raise ValueError(
                f"非法动作：列 {action} 已满。"
                "请检查 action_masks 是否正确传递。"
            )

        self.obs, old_reward, done, _ = self.env.step(action)

        # V2：纯终局奖励。
        # 非终局奖励为 0；Kaggle 原始终局奖励为 1 / -1 / 0。
        reward = self.change_reward(old_reward, done)

        board = self._relative_board(self.obs)

        terminated = bool(done)
        truncated = False

        info = {
            "action_mask": self.action_masks(),
            "is_first_player": self.is_first_player,
            "opponent_name": self.current_opponent_name,
        }

        return board, reward, terminated, truncated, info

    @staticmethod
    def change_reward(old_reward, done):
        """
        纯终局奖励：

        胜：+1
        负：-1
        平： 0
        未结束：0
        """
        if not done:
            return 0.0

        if old_reward == 1:
            return 1.0
        if old_reward == -1:
            return -1.0
        return 0.0