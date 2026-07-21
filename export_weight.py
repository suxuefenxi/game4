import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO

# 1. 声明一个与当前 train.py 模型结构（[128, 128]）一致的纯 PyTorch 模型
class PurePyTorchPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = nn.Flatten()
        self.mlp_extractor = nn.Sequential(
            nn.Linear(42, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.action_net = nn.Linear(128, 7)

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.mlp_extractor(x)
        return self.action_net(x)

# 2. 加载训练好的 SB3 模型并转换权重
sb3_model = MaskablePPO.load("runs/v4_tactical_ppo/models/best_model.zip")
sb3_policy = sb3_model.policy

# 3. 映射权重数据
state_dict = {
    'mlp_extractor.0.weight': sb3_policy.mlp_extractor.policy_net[0].weight,
    'mlp_extractor.0.bias':   sb3_policy.mlp_extractor.policy_net[0].bias,
    'mlp_extractor.2.weight': sb3_policy.mlp_extractor.policy_net[2].weight,
    'mlp_extractor.2.bias':   sb3_policy.mlp_extractor.policy_net[2].bias,
    'action_net.weight':      sb3_policy.action_net.weight,
    'action_net.bias':        sb3_policy.action_net.bias,
}

# 4. 保存为纯 PyTorch 的权重文件
torch.save(state_dict, "pure_policy_weights_v4.pth")
print("权重成功提取并保存至 pure_policy_weights_v4.pth！")