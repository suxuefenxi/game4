#!/usr/bin/env python
"""
Agent 对战测试脚手架

在本地运行多场比赛，比较不同 AI 的实力，无需提交到 Kaggle。
自动处理先手/后手轮换，确保公平比较。

支持的 agent 类型：
  random      - Kaggle 内置随机 AI
  heuristic   - 一阶启发式 AI（heuristic_agent.py）
  minimax_d3  - Minimax depth=3 agent
  minimax_d4  - Minimax depth=4 agent
  minimax_d5  - Minimax depth=5 agent
  <模型路径>  - SB3 训练好的 PPO 模型文件（例如 ppo_connectx.zip）

用法示例：
  # heuristic vs random（各10局，先手轮换）
  python agent_battle.py heuristic random -n 20

  # 已训练的 PPO 模型 vs heuristic
  python agent_battle.py ppo_connectx.zip heuristic -n 20

  # 查看详细对局信息
  python agent_battle.py heuristic random -v
"""

import argparse
import sys
import numpy as np
from kaggle_environments import make


def build_ppo_agent(model_path, stochastic=False):
    from sb3_contrib import MaskablePPO
    from tactical import tactical_action_mask

    model = MaskablePPO.load(model_path)

    def ppo_agent(obs, config):
        board = np.asarray(obs.board, dtype=np.float32)

        # 与训练环境完全一致：当前行动方视角的相对编码。
        my_mark = obs.mark
        opp_mark = 3 - my_mark

        state = np.zeros_like(board, dtype=np.float32)
        state[board == my_mark] = 1.0
        state[board == opp_mark] = -1.0

        # 与训练完全一致的战术 mask。
        action_masks = tactical_action_mask(
            board=board,
            mark=my_mark,
            config=config,
        )

        # 默认保持确定性，便于做基准；
        # 若开启 stochastic，则让 PPO 按策略分布采样，产生更有多样性的对局。
        action, _ = model.predict(
            state,
            deterministic=not stochastic,
            action_masks=action_masks,
        )

        return int(action)

    return ppo_agent

def resolve_agent(spec, stochastic=False):
    """
    将 agent 规格字符串解析为可调用对象

    规则：
    - spec 是已注册的短名称 -> 返回对应的 agent 函数
    - spec 以 .zip 结尾     -> 视为 SB3 模型路径
    - 其他                  -> 直接作为字符串传给 Kaggle（如 "random"）
    """
    # 已知 agent 注册表
    registry = {}

    # 内置 agent（懒导入，避免不必要的开销）
    if spec == "heuristic":
        from heuristic_agent import heuristic_agent
        return heuristic_agent
    
    if spec == "minimax_d3":
        from minimax_agent import minimax_d3_agent
        return minimax_d3_agent

    if spec == "minimax_d4":
        from minimax_agent import minimax_d4_agent
        return minimax_d4_agent
    
    if spec == "minimax_d5":
        from minimax_agent import minimax_d5_agent
        return minimax_d5_agent

    if spec == "minimax_d3_fixed":
        from minimax_agent import minimax_d3_deterministic
        return minimax_d3_deterministic

    if spec == "minimax_d4_fixed":
        from minimax_agent import minimax_d4_deterministic
        return minimax_d4_deterministic

    # SB3 模型文件（路径以 .zip 结尾）
    if spec.endswith(".zip"):
        return build_ppo_agent(spec, stochastic=stochastic)

    if spec.endswith(".pt") or spec.endswith(".pth"):
        from bc_policy import load_bc_agent
        return load_bc_agent(spec)

    # 其他情况：当作 Kaggle 内置 agent 名称字符串
    return spec


def run_match(agent1, agent2):
    """
    运行一局比赛，返回 (winner, steps_count)

    参数：
      agent1/agent2: Kaggle 兼容的 agent（字符串名或可调用对象）
      agent1 先手（玩家0），agent2 后手（玩家1）

    返回值：
      winner: 1=agent1胜, 2=agent2胜, 0=平局
      steps_count: 对局步数
    """
    env = make("connectx", debug=True)
    steps = env.run([agent1, agent2])

    # 解析最终状态
    final = steps[-1]
    r1, r2 = final[0].reward, final[1].reward

    if r1 == 1 and r2 == -1:
        winner = 1
    elif r2 == 1 and r1 == -1:
        winner = 2
    else:
        winner = 0  # 平局（满盘或双方都无获胜）

    return winner, len(steps)


def main():
    parser = argparse.ArgumentParser(
        description="Agent 对战测试脚手架 —— 本地多局对战评估 AI 实力"
    )
    parser.add_argument("agent1", help="Agent 1（先手），支持: random / heuristic / <模型.zip>")
    parser.add_argument("agent2", help="Agent 2（后手），同上")
    parser.add_argument("-n", "--num-games", type=int, default=10,
                        help="比赛总局数（默认 10，会自动轮换先手）")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="显示每局详细结果")
    parser.add_argument("--stochastic", action="store_true",
                        help="对 PPO agent 使用采样模式（非 deterministic），增加对局多样性")
    args = parser.parse_args()

    # 注册 agent
    a1 = resolve_agent(args.agent1, stochastic=args.stochastic)
    a2 = resolve_agent(args.agent2, stochastic=args.stochastic)

    # 显示对战信息
    print(f"{'='*50}")
    print(f"Agent 对战测试")
    print(f"{'='*50}")
    print(f"Agent 1 : {args.agent1}  （先手局：奇数局）")
    print(f"Agent 2 : {args.agent2}  （先手局：偶数局）")
    print(f"总局数  : {args.num_games}（含{'先手轮换' if args.num_games > 1 else '不轮换'})")
    print()

    # 运行比赛
    wins = {1: 0, 2: 0, 0: 0}        # agent1胜 / agent2胜 / 平局
    first_win_count = 0               # 先手获胜次数
    steps_record = []
    details = []                      # 记录每局详情

    for i in range(1, args.num_games + 1):
        # 先手轮换：奇数局 agent1 先手，偶数局 agent2 先手
        is_agent1_first = (i % 2 == 1)
        if is_agent1_first:
            winner, steps = run_match(a1, a2)
        else:
            winner, steps = run_match(a2, a1)
            # 调整 winner 编号（因为swap了agent传入顺序）
            winner = {1: 2, 2: 1, 0: 0}[winner]

        wins[winner] += 1
        if winner != 0 and ((winner == 1 and is_agent1_first) or (winner == 2 and not is_agent1_first)):
            first_win_count += 1
        steps_record.append(steps)
        details.append((i, is_agent1_first, winner, steps))

        if args.verbose:
            label = {1: f"{args.agent1} 胜",
                     2: f"{args.agent2} 胜",
                     0: "平局"}[winner]
            first_name = args.agent1 if is_agent1_first else args.agent2
            print(f"  第{i:2d}局 | 先手: {first_name:>10} | {label} | {steps}步")

    # ── 汇总统计 ──
    print()
    print(f"{'='*50}")
    print(f"  对战结果汇总")
    print(f"{'='*50}")
    print(f"  {args.agent1:>20} : {wins[1]:4d} 胜  {wins[0]:2d} 平  {wins[2]:2d} 负")
    print(f"  {args.agent2:>20} : {wins[2]:4d} 胜  {wins[0]:2d} 平  {wins[1]:2d} 负")
    print(f"  {'─'*40}")

    total = args.num_games
    print(f"  {args.agent1:>20} 胜率: {wins[1]/total:.1%}")
    print(f"  {args.agent2:>20} 胜率: {wins[2]/total:.1%}")
    print(f"  平局率    : {wins[0]/total:.1%}")
    print(f"  平均步数  : {np.mean(steps_record):.1f}")

    # 先手优势分析
    non_draw = wins[1] + wins[2]
    if non_draw > 0:
        first_win_rate = first_win_count / non_draw
        print(f"  先手胜率  : {first_win_rate:.1%}（{first_win_count}/{non_draw}局）")
    print(f"{'='*50}")

    return 0 if wins[1] >= wins[2] else 1


if __name__ == "__main__":
    sys.exit(main())
