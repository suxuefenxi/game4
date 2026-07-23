import numpy as np
import torch
import torch.nn as nn

from tactical import tactical_action_mask


class BCPolicy(nn.Module):
    """
    与 V4 PPO actor 对齐的策略网络：

    42 -> 128 -> 128 -> 7

    输出为每个列动作的 logits，不在 forward 内部做 softmax。
    """

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(42, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(128, 7)

    def forward(self, x):
        return self.action_net(self.mlp(x))


def encode_relative_board(obs):
    """将 Kaggle 原始 0/1/2 棋盘编码成当前行动方视角的 -1/0/+1。"""
    board = np.asarray(obs.board, dtype=np.float32)

    my_mark = obs.mark
    opponent_mark = 3 - my_mark

    state = np.zeros_like(board, dtype=np.float32)
    state[board == my_mark] = 1.0
    state[board == opponent_mark] = -1.0

    return state


def load_bc_agent(weights_path, device="cpu"):
    """
    加载 BC checkpoint，返回 Kaggle 兼容 agent。

    训练、评测、提交均使用：
    - 相对状态编码；
    - tactical_action_mask。
    """
    checkpoint = torch.load(
        weights_path,
        map_location=torch.device(device),
    )

    model = BCPolicy().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def bc_agent(obs, config):
        state = encode_relative_board(obs)

        tactical_mask = tactical_action_mask(
            board=obs.board,
            mark=obs.mark,
            config=config,
        )

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            logits = model(state_tensor).squeeze(0)

            # 无论 BC 网络输出什么，均禁止不允许的动作。
            mask_tensor = torch.as_tensor(
                tactical_mask,
                dtype=torch.bool,
                device=device,
            )
            logits = logits.masked_fill(~mask_tensor, -torch.inf)

            action = int(torch.argmax(logits).item())

        return action

    return bc_agent