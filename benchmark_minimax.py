#!/usr/bin/env python
"""
Minimax 耗时基准测试

让 minimax d3/d4 与自己（同深度）对局，不设 time_limit，
测出每步搜索的真实耗时分布，为训练时设置合理的 time_limit 提供依据。

用法：
    python benchmark_minimax.py          # 默认各跑 20 局
    python benchmark_minimax.py -n 10    # 各跑 10 局
    python benchmark_minimax.py --d3-only
    python benchmark_minimax.py --d4-only
"""

import argparse
import sys
import time
import numpy as np
from kaggle_environments import make

# 引入 make_minimax_agent 工厂函数
from minimax_agent import make_minimax_agent


def make_timed_agent(depth):
    """
    创建不设 time_limit 的 minimax agent，同时记录每步耗时。
    返回 (agent_fn, timing_list)，timing_list 会在每次走棋时 append 耗时。
    """
    # time_limit 设为一个极大值，确保搜索不会被提前截断
    agent = make_minimax_agent(depth=depth, time_limit=1e9, seed=None)

    timings = []

    def timed_agent(obs, config):
        t0 = time.perf_counter()
        action = agent(obs, config)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        return action

    return timed_agent, timings


def run_match(agent1, agent2):
    """运行一局，返回步数。"""
    env = make("connectx", debug=True)
    steps = env.run([agent1, agent2])
    return len(steps)


def main():
    parser = argparse.ArgumentParser(
        description="Minimax 耗时基准测试 —— 测量 d3/d4 每步搜索耗时"
    )
    parser.add_argument("-n", "--num-games", type=int, default=20,
                        help="每个 depth 自我对弈的局数（默认 20）")
    parser.add_argument("--d3-only", action="store_true",
                        help="只测 d3")
    parser.add_argument("--d4-only", action="store_true",
                        help="只测 d4")
    parser.add_argument("--depth", type=int, nargs="*",
                        help="自定义搜索深度列表，如 --depth 3 5 7")
    parser.add_argument("--full", action="store_true",
                        help="输出每步的原始耗时（用于绘图/分析）")
    args = parser.parse_args()

    if args.depth is not None:
        depths = sorted(set(args.depth))
    else:
        depths = []
        if not args.d4_only:
            depths.append(3)
        if not args.d3_only:
            depths.append(4)
        depths = sorted(set(depths))

    all_results = {}

    for depth in depths:
        print(f"\n{'='*55}")
        print(f"  Minimax depth={depth} 自我对弈 {args.num_games} 局")
        print(f"{'='*55}")

        game_timings = []  # 每局的每步耗时列表

        for i in range(1, args.num_games + 1):
            # 每局创建独立的 timed agent，确保计时数据干净
            a1, t1 = make_timed_agent(depth)
            a2, t2 = make_timed_agent(depth)

            # 先手轮换
            if i % 2 == 1:
                steps = run_match(a1, a2)
            else:
                steps = run_match(a2, a1)

            # 收集双方耗时
            all_t = np.array(t1 + t2)
            game_timings.append(all_t)

            # 该局统计
            avg_t = np.mean(all_t)
            max_t = np.max(all_t)
            min_t = np.min(all_t)
            print(f"  第{i:2d}局 | {steps:2d}步 | "
                  f"平均 {avg_t*1000:.1f}ms | "
                  f"最慢 {max_t*1000:.1f}ms | "
                  f"最快 {min_t*1000:.1f}ms")

        # ── 全局汇总 ──
        all_flat = np.concatenate(game_timings)
        all_results[depth] = {
            "raw": all_flat if args.full else None,
            "count": len(all_flat),
            "mean": np.mean(all_flat),
            "std": np.std(all_flat),
            "min": np.min(all_flat),
            "max": np.max(all_flat),
            "p50": np.percentile(all_flat, 50),
            "p95": np.percentile(all_flat, 95),
            "p99": np.percentile(all_flat, 99),
            "game_stats": {
                "avg_of_avg": np.mean([np.mean(g) for g in game_timings]),
                "avg_of_max": np.mean([np.max(g) for g in game_timings]),
                "max_of_max": np.max([np.max(g) for g in game_timings]),
            },
        }

        r = all_results[depth]
        print(f"\n  {'─'*50}")
        print(f"  depth={depth} 汇总（共 {r['count']} 步）")
        print(f"  {'─'*50}")
        print(f"  平均耗时  : {r['mean']*1000:.1f} ms")
        print(f"  标准差    : ±{r['std']*1000:.1f} ms")
        print(f"  最快步    : {r['min']*1000:.1f} ms")
        print(f"  最慢步    : {r['max']*1000:.1f} ms")
        print(f"  中位数    : {r['p50']*1000:.1f} ms")
        print(f"  P95       : {r['p95']*1000:.1f} ms")
        print(f"  P99       : {r['p99']*1000:.1f} ms")
        print(f"  每局局均  : {r['game_stats']['avg_of_avg']*1000:.1f} ms")
        print(f"  每局最慢平均 : {r['game_stats']['avg_of_max']*1000:.1f} ms")
        print(f"  所有局中最慢一步 : {r['game_stats']['max_of_max']*1000:.1f} ms")

        if args.full:
            print(f"\n  原始耗时（ms）: ", end="")
            np.set_printoptions(linewidth=100, precision=1, suppress=True)
            print(r["raw"] * 1000)

    # ── 建议 ──
    print(f"\n\n{'='*55}")
    print(f"  训练 time_limit 建议")
    print(f"{'='*55}")
    for depth in depths:
        r = all_results[depth]
        p95 = r["p95"]
        p99 = r["p99"]
        max_t = r["max"]

        # 不同策略的推荐值
        conservative = max_t * 1.1  # 宽松：最慢步 + 10% 余量
        balanced = p99 * 1.2        # 平衡：P99 + 20% 余量
        aggressive = p95 * 1.5      # 激进：P95 + 50% 余量（少数步会中断）

        print(f"  ── depth={depth} ──")
        print(f"    保守（绝不超时）  : time_limit={conservative:.3f}s "
              f"（最慢{r['max']*1000:.0f}ms + 10%）")
        print(f"    平衡（推荐）      : time_limit={balanced:.3f}s "
              f"（P99 {p99*1000:.0f}ms + 20%）")
        print(f"    激进（偶尔截断）  : time_limit={aggressive:.3f}s "
              f"（P95 {p95*1000:.0f}ms + 50%）")

    # ── 对当前设置的评价 ──
    print(f"\n  {'─'*50}")
    print(f"  当前代码默认设置评价")
    print(f"  {'─'*50}")
    if 3 in all_results:
        r3 = all_results[3]
        current_d3 = 0.5  # minimax_agent.py 中 d3 的 time_limit 是 0.5
        p99_3, max_3 = r3["p99"], r3["max"]
        print(f"  d3 当前 time_limit={current_d3}s" +
              (" ✅ 够用" if current_d3 >= max_3 else f" ⚠️  建议 ≥{max_3:.3f}s（最慢步需要 {max_3*1000:.0f}ms）"))

    if 4 in all_results:
        r4 = all_results[4]
        current_d4 = 1.0  # minimax_agent.py 中 d4 的 time_limit 是 1.0
        p99_4, max_4 = r4["p99"], r4["max"]
        print(f"  d4 当前 time_limit={current_d4}s" +
              (" ✅ 够用" if current_d4 >= max_4 else f" ⚠️  建议 ≥{max_4:.3f}s（最慢步需要 {max_4*1000:.0f}ms）"))

    print(f"\n{'='*55}")
    return 0


# ── 辅助函数 ──

def make_timed_agent(depth):
    """
    创建不设 time_limit 的 minimax agent，同时记录每步耗时。
    返回 (agent_fn, timing_list)，每次走棋会 append 耗时。
    """
    agent = make_minimax_agent(depth=depth, time_limit=1e9, seed=None)
    timings = []

    def timed_agent(obs, config):
        t0 = time.perf_counter()
        action = agent(obs, config)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        return action

    return timed_agent, timings


def run_match(agent1, agent2):
    """运行一局，返回步数。"""
    env = make("connectx", debug=True)
    steps = env.run([agent1, agent2])
    return len(steps)


if __name__ == "__main__":
    sys.exit(main())
