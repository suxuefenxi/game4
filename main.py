import os
import torch
import torch.nn as nn
import numpy as np

# 1. 声明完全一致的模型结构
class PurePyTorchPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = nn.Flatten()
        self.mlp_extractor = nn.Sequential(
            nn.Linear(42, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.action_net = nn.Linear(64, 7)
        
    def forward(self, x):
        x = self.features_extractor(x)
        x = self.mlp_extractor(x)
        return self.action_net(x)

# 2. 预先加载模型（安全路径兼容）
if "__file__" in globals():
    # 本地运行时，使用 __file__
    model_path = os.path.join(os.path.dirname(__file__), "pure_policy_weights.pth")
else:
    # Kaggle 评测机上运行时（__file__ 未定义），使用固定的解压绝对路径
    model_path = "/kaggle_simulations/agent/pure_policy_weights.pth"

model = PurePyTorchPolicy()

if os.path.exists(model_path):
    # 显式指定加载到 CPU 上，防止评测机环境由于设备不匹配报错
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def agent(observation, configuration):
    # 状态转换：将 0/1/2 转换为 1(我方) / -1(对手) / 0(空位)
    board_list = observation.board if hasattr(observation, 'board') else observation['board']
    my_mark = observation.mark
    opp_mark = 3 - my_mark
    
    board_arr = np.array(board_list, dtype=np.float32)
    state = np.zeros_like(board_arr, dtype=np.float32)
    state[board_arr == my_mark] = 1.0
    state[board_arr == opp_mark] = -1.0
    
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    # 网络推理
    with torch.no_grad():
        logits = model(state_tensor).squeeze(0)
        
    # 动作屏蔽（Action Masking）：如果该列第一行不为 0，说明已满
    for col in range(configuration.columns):
        if board_list[col] != 0:
            logits[col] = -float('inf') # 设为负无穷防止被选中
            
    # 选择概率（或Logits）最大的合法列
    action = int(torch.argmax(logits).item())
    return action