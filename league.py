from pathlib import Path
import numpy as np

from sb3_contrib import MaskablePPO
from tactical import tactical_action_mask


# 当前冠军的固定路径。下一代训练自动从它初始化。
# 晋升时会更新此文件以及 league/gen_XXX.zip。
CHAMPION_PATH = "league/champion.zip"


def make_maskable_ppo_agent(model_path):
    """
    将已冻结的 MaskablePPO 模型包装为 Kaggle ConnectX agent。

    注意：
    - 模型只在创建 opponent pool 时加载一次；
    - 训练期间它不会改变；
    - 这正是历史快照对手的含义。
    """
    model = MaskablePPO.load(str(model_path), device="cpu")

    def agent(obs, config):
        board = np.asarray(obs.board, dtype=np.float32)

        my_mark = obs.mark
        opp_mark = 3 - my_mark

        # 必须与训练环境、部署脚本完全一致。
        state = np.zeros_like(board, dtype=np.float32)
        state[board == my_mark] = 1.0
        state[board == opp_mark] = -1.0

        action_mask = tactical_action_mask(
            board=board,
            mark=my_mark,
            config=config,
        )

        action, _ = model.predict(
            state,
            deterministic=False,   # 训练对手加随机性，增加局面多样性
            action_masks=action_mask,
        )
        return int(action)

    return agent


def build_opponent_selector(opponent_specs):
    """
    根据配置创建“每局抽样一次”的选择器。

    opponent_specs 格式：
    [
        ("random", random_agent, 0.20),
        ("heuristic", heuristic_agent, 0.30),
        ("v4_best", frozen_ppo_agent, 0.35),
    ]
    """
    names = [name for name, _, _ in opponent_specs]
    agents = [agent for _, agent, _ in opponent_specs]

    weights = np.asarray(
        [weight for _, _, weight in opponent_specs],
        dtype=np.float64,
    )
    weights /= weights.sum()

    def selector(rng):
        index = int(rng.choice(len(agents), p=weights))
        return agents[index], names[index]

    return selector


def find_league_models(league_dir, max_models=3):
    """
    从 league/ 目录读取历史晋升模型。

    只返回 gen_*.zip 的快照模型，排除当前 champion.zip，
    因为 champion 是当前别名，不是历史对手。
    按修改时间选最近的 max_models 个模型。
    """
    league_dir = Path(league_dir)
    paths = sorted(
        [p for p in league_dir.glob("*.zip") if p.name != "champion.zip"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return paths[:max_models]