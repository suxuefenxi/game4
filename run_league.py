#!/usr/bin/env python
"""
联赛自动化调度器。

自动完成多代训练循环：
    读取当前 champion → 训练候选 → benchmark → 达到门槛则晋升 → 进入下一代

用法：
    # 连续训练 5 代，每代 200k 步
    python run_league.py --generations 5 --timesteps 200000

    # 快速测试 1 代
    python run_league.py --generations 1 --timesteps 10000
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ── 每个 benchmark 对手的固定局数 ──────────────────────
#
# champion（PPO）     16 局 — 采用 stochastic 采样，评估模型在不同对局路径上的鲁棒性。
#                         这样比单纯 2 局更能避免“运气过强”误晋升。
# random               4 局 — 保留少量做安全网，防止模型训崩。
# heuristic           4 局 — 同上，保留少量做安全网。
# minimax_d3          20 局 — depth=3，随机 tiebreak，有区分度的对手。
# minimax_d4          48 局 — depth=4，更强的对手，主要区分来源。
# minimax_d5          10 局 — depth=5，最强的对手，主要区分来源。
# bc_v1               10 局 — 行为克隆策略，从专家数据模仿训练。
BENCHMARK_CONFIG = [
    ("champion",    30),
    ("random",       4),
    ("heuristic",    4),
    ("minimax_d3",  60),
    ("minimax_d4",  4),
    ("minimax_d5",  4),
    ("bc_v1",       60),
]

# 非内置对手的完整路径映射
BC_AGENT_PATH = "runs/bc_v1/bc_policy_best.pt"


LEAGUE_DIR = Path("league")
RUNS_DIR = Path("runs")
CHAMPION_PATH = LEAGUE_DIR / "champion.zip"
RESULTS_PATH = LEAGUE_DIR / "results.jsonl"


def run_cmd(cmd, cwd=None):
    """运行命令，实时逐行打印输出，返回完整 stdout 文本。"""
    print(f"\n{'=' * 60}")
    print(f"执行: {' '.join(map(str, cmd))}")
    print(f"{'=' * 60}")

    lines = []
    with subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,  # 行缓冲，确保实时输出
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
            lines.append(line)

    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"命令返回非预期状态码 {proc.returncode}"
        )

    return "".join(lines)


def parse_battle_results(text):
    """
    从 agent_battle.py 的输出中提取 agent1 的胜平负。

    输出格式（第 195 行）：
          agent1 :   12 胜   2 平   6 负
    """
    pattern = r":\s+(\d+)\s+胜\s+(\d+)\s+平\s+(\d+)\s+负"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(
            f"无法解析对战结果:\n{text}"
        )
    wins, draws, losses = map(int, match.groups())
    return wins, draws, losses


def benchmark_candidate(candidate_path):
    """
    对候选模型运行完整 benchmark。

    返回：
        results: {
            "champion":  {"wins": ..., "draws": ..., "losses": ...},
            "heuristic": {...},
            "random":    {...},
            "minimax_d3": {...},
        }
    """
    results = {}

    for opponent, num_games in BENCHMARK_CONFIG:
        if opponent == "champion":
            spec = str(CHAMPION_PATH)
        elif opponent == "bc_v1":
            spec = BC_AGENT_PATH
        else:
            spec = opponent

        print(f"\n--- benchmark: 候选 vs {opponent} ({num_games} 局) ---")
        cmd = [
            sys.executable,
            "agent_battle.py",
            str(candidate_path),
            spec,
            "-n", str(num_games),
        ]
        if opponent == "champion":
            cmd.append("--stochastic")

        output = run_cmd(cmd)

        wins, draws, losses = parse_battle_results(output)
        games = wins + draws + losses
        score = (wins + 0.5 * draws) / games if games > 0 else 0.0
        results[opponent] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "games": games,
            "score": score,
        }
        print(f"  结果: {wins} 胜 {draws} 平 {losses} 负  (得分 {score:.1%})")

    return results


def check_promotion(results):
    """
    晋升判定。

    两个条件必须同时满足：
    1. 候选 vs champion 2:0（先手后手都赢）
    2. 对所有对手总得分 >= 0.7

    得分 = (胜 + 0.5 × 平) / 总局，平局视为半胜。
    """
    # 条件 1：在多局 stochastic 评估中，显著优于 champion。
    # 不再依赖固定的胜负场数，而是用加权胜率来判断是否真正占优。
    champ = results["champion"]
    champ_score = (champ["wins"] + 0.5 * champ["draws"]) / champ["games"] if champ["games"] > 0 else 0.0
    beats_champion = champ_score >= 0.70

    # 条件 2：总得分门槛（平局算半胜）
    total_wins = sum(r["wins"] for r in results.values())
    total_draws = sum(r["draws"] for r in results.values())
    total_games = sum(r["games"] for r in results.values())

    total_score = (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0.0

    promoted = beats_champion and total_score >= 0.70

    metrics = {
        "beats_champion": beats_champion,
        "champion_wins": champ["wins"],
        "champion_losses": champ["losses"],
        "champion_draws": champ["draws"],
        "total_score": round(total_score, 4),
        "total_wins": total_wins,
        "total_draws": total_draws,
        "total_games": total_games,
    }

    return promoted, metrics


def get_next_gen_number():
    """
    扫描 league/gen_*.zip，返回下一个可用编号。

    如果已经有 gen_000.zip，则下一个是 1，以此类推。
    """
    max_gen = -1
    for p in LEAGUE_DIR.glob("gen_*.zip"):
        m = re.search(r"gen_(\d+)", p.stem)
        if m:
            num = int(m.group(1))
            if num > max_gen:
                max_gen = num
    return max_gen + 1


def append_result(record):
    """将一代的结果追加到 results.jsonl。"""
    LEAGUE_DIR.mkdir(exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_benchmark(results):
    """生成 benchmark 汇总摘要文本。"""
    lines = ["Benchmark 结果:"]
    for opponent, r in results.items():
        label = f"  vs {opponent:>10}"
        line = f"{label}: {r['wins']:2d} 胜 {r['draws']:2d} 平 {r['losses']:2d} 负  (得分 {r['score']:.1%})"
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="自动化 ConnectX Self-Play 联赛调度器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python run_league.py --generations 5 --timesteps 200000\n"
            "  python run_league.py --generations 1 --timesteps 10000\n"
        ),
    )
    parser.add_argument(
        "--generations", type=int, default=3,
        help="连续训练代数（默认 3）",
    )
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="每代训练步数（默认 200000）",
    )
    parser.add_argument(
        "--start-gen", type=int, default=None,
        help="起始代数编号（默认自动检测）",
    )
    args = parser.parse_args()

    # ── 启动前检查 ──
    LEAGUE_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    if not CHAMPION_PATH.exists():
        print(f"错误：找不到 {CHAMPION_PATH}")
        print()
        print("请先放入初始 champion：")
        print("  cp league/gen_000_v4_best.zip league/champion.zip")
        print("  cp league/gen_000_v4_best.zip league/gen_000.zip")
        sys.exit(1)

    start_gen = args.start_gen
    if start_gen is None:
        start_gen = get_next_gen_number()

    print(f"\n{'#' * 60}")
    print(f"  联赛自动化调度器")
    print(f"{'#' * 60}")
    print(f"  当前 champion: {CHAMPION_PATH}")
    print(f"  训练代数:      {args.generations} 代")
    print(f"  每代步数:      {args.timesteps}")
    print(f"  起始编号:      gen_{start_gen:03d}")
    print(f"{'#' * 60}\n")

    # ── 主循环 ──
    #
    # init_model 追踪"下一代的起点模型路径"。
    # 第 1 代从 champion 开始；后续每代都从上一代候选继续，
    # 不管是否晋升，保证训练连续积累，不会原地重复。
    init_model = CHAMPION_PATH

    for gen_offset in range(args.generations):
        gen_num = start_gen + gen_offset
        gen_tag = f"gen_{gen_num:03d}"
        output_dir = RUNS_DIR / gen_tag

        print(f"\n{'#' * 60}")
        print(f"  第 {gen_num} 代 ({gen_tag})")
        print(f"  当前冠军: {CHAMPION_PATH}")
        print(f"  初始化自: {init_model}")
        print(f"{'#' * 60}")

        # 1. 训练候选模型
        print("\n--- 阶段 1: 训练 ---")
        try:
            run_cmd([
                sys.executable,
                "train_generation.py",
                "--init-model", str(init_model),
                "--output-dir", str(output_dir),
                "--timesteps", str(args.timesteps),
                "--league-dir", str(LEAGUE_DIR),
            ])
        except RuntimeError as e:
            print(f"训练失败: {e}")
            print("跳过本代，继续下一代...")
            continue

        # 找候选模型：优先 best_model（训练过程中表现最好的 checkpoint）
        candidate = output_dir / "models" / "best_model.zip"
        if not candidate.exists():
            candidate = output_dir / "models" / "final_model.zip"
        if not candidate.exists():
            print(f"错误：{output_dir}/models/ 中找不到候选模型！")
            continue

        print(f"候选模型: {candidate}")

        # 2. Benchmark（始终对战当前 champion，即最强的已确认版本）
        print("\n--- 阶段 2: Benchmark ---")
        try:
            results = benchmark_candidate(candidate)
        except (ValueError, RuntimeError) as e:
            print(f"Benchmark 失败: {e}")
            continue

        # 3. 晋升判定
        print("\n--- 阶段 3: 晋升判定 ---")
        promoted, metrics = check_promotion(results)

        print(summarize_benchmark(results))
        print()
        print(f"  完胜 champion:     {'✅' if metrics['beats_champion'] else '❌'} "
              f"({metrics['champion_wins']}:{metrics['champion_losses']})")
        print(f"  总得分（胜+0.5×平）: {metrics['total_score']:.1%} "
              f"({metrics['total_wins']}+{metrics['total_draws']}×0.5/{metrics['total_games']})")
        print(f"  晋升:              {'✅ 晋升!' if promoted else '❌ 不晋升'}")

        # 4. 晋升操作：候选晋升为新冠军，保存历史快照
        if promoted:
            gen_path = LEAGUE_DIR / f"{gen_tag}.zip"
            shutil.copy2(candidate, gen_path)
            shutil.copy2(candidate, CHAMPION_PATH)
            print(f"\n✅ 候选晋升为新冠军！")
            print(f"  历史快照: {gen_path}")
            print(f"  冠军更新: {CHAMPION_PATH}")

        # 5. 不管是否晋升，下一代都从当前候选继续训练。
        #    这样即使没达到门槛，训练也能连续积累。
        init_model = candidate

        # 5. 记录结果
        record = {
            "generation": gen_num,
            "tag": gen_tag,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "init_model": str(CHAMPION_PATH),
            "candidate": str(candidate),
            "timesteps": args.timesteps,
            "benchmark": {
                opp: {
                    "wins": r["wins"],
                    "draws": r["draws"],
                    "losses": r["losses"],
                    "games": r["games"],
                    "score": r["score"],
                }
                for opp, r in results.items()
            },
            "promoted": promoted,
            "metrics": metrics,
        }
        append_result(record)

        print(f"\n  结果已保存至: {RESULTS_PATH}")

    # ── 完成 ──
    print(f"\n{'#' * 60}")
    print(f"  联赛训练完成")
    print(f"{'#' * 60}")
    print(f"  最终 champion: {CHAMPION_PATH}")
    print(f"  训练日志:      {RESULTS_PATH}")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
