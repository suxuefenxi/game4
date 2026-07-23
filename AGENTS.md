# AGENTS.md

## 项目概览

这是一个基于 Connect Four 的 PPO self-play 联赛训练项目。
主要流程是：
1. 从 league/champion.zip 初始化模型；
2. 训练一代候选模型；
3. 用 benchmark 对候选模型与 champion / 基线对手进行对战评估；
4. 若达到晋升条件，则把候选模型保存为新的 champion。

## 关键文件

- [train_generation.py](train_generation.py)：训练一代 PPO。默认使用 4 个子进程环境，输出到 runs/gen_XXX/models、runs/gen_XXX/tensorboard、runs/gen_XXX/eval。
- [run_league.py](run_league.py)：联赛主循环，串联训练、benchmark 和晋升逻辑。
- [agent_battle.py](agent_battle.py)：本地多局对战评估脚本，用于比较不同 agent 的强度。
- [connect_four_gym.py](connect_four_gym.py)：Gymnasium 环境包装，负责与 Kaggle ConnectX 环境交互。
- [league.py](league.py)：联赛对手选择、历史模型读取和 PPO agent 封装。
- [tactical.py](tactical.py)：战术动作 mask，训练/评估时都应使用它，避免非法或明显不安全的走法。

## 运行方式

常用命令：

- 训练一代：
  `python train_generation.py --init-model league/champion.zip --output-dir runs/gen_001 --timesteps 200000 --league-dir league`
- 运行联赛主循环：
  `python run_league.py --generations 1 --timesteps 10000`
- 评估两个 agent 的对战：
  `python agent_battle.py heuristic random -n 20`
- 对 PPO 使用采样风格评估：
  `python agent_battle.py candidate.zip champion.zip -n 16 --stochastic`

## 约定与注意事项

- 这里的训练和评估依赖 `stable-baselines3`、`sb3-contrib`、`kaggle-environments`。
- 训练前请确认 `league/champion.zip` 存在；否则联赛脚本会直接报错。
- 训练输出目录约定为 `runs/gen_XXX/`，不要随意改动主流程依赖的路径结构。
- 对战逻辑和训练环境要保持一致：状态需使用当前行动方视角的相对棋盘编码；动作 mask 也要通过 tactical 逻辑生成。
- 如果修改晋升规则，请同时保持与 benchmark 结果解析格式兼容；`run_league.py` 依赖 `agent_battle.py` 输出的胜/平/负统计。
- 代码和脚本应尽量保留中文注释，便于后续阅读和维护。
- 代码在conda虚拟环境pyai运行（已创建，不用你再新建虚拟环境！），已安装必要的库。

## 修改建议

- 改动训练或评估逻辑时，优先保持现有 CLI 参数和输出文件结构不变。
- 新增评估指标时，最好同时更新 `run_league.py` 的汇总和记录逻辑。
- 对 self-play / 晋升规则的修改，建议同时考虑：
  - 多局统计而非单局硬判；
  - 采样式评估而非仅 deterministic 推理；
  - 兼顾胜率和鲁棒性，而不是只看“是否 2:0”。
