import time
import numpy as np


WIN_SCORE = 1_000_000

# Connect Four 中间列通常更有价值。
# 先搜中间，Alpha-Beta 更容易剪枝。
MOVE_ORDER_7 = [3, 2, 4, 1, 5, 0, 6]


def _cfg(config, name):
    """兼容 Kaggle 的 Struct 和普通 dict。"""
    return getattr(config, name) if hasattr(config, name) else config[name]


def valid_moves(grid):
    """返回未满列。grid 是二维棋盘。"""
    return [c for c in range(grid.shape[1]) if grid[0, c] == 0]


def ordered_moves(grid):
    """优先搜索中间列，提高 Alpha-Beta 剪枝效率。"""
    cols = grid.shape[1]
    valid = set(valid_moves(grid))

    if cols == 7:
        return [c for c in MOVE_ORDER_7 if c in valid]

    # 兼容非标准 ConnectX 配置。
    center = (cols - 1) / 2
    return sorted(valid, key=lambda c: abs(c - center))


def drop_piece(grid, col, mark):
    """在 col 列落子，返回新棋盘；满列时返回 None。"""
    next_grid = grid.copy()

    for row in range(next_grid.shape[0] - 1, -1, -1):
        if next_grid[row, col] == 0:
            next_grid[row, col] = mark
            return next_grid

    return None


def is_win(grid, mark, inarow):
    """检查某一方是否已经连成 inarow。"""
    rows, cols = grid.shape

    # 横向
    for r in range(rows):
        for c in range(cols - inarow + 1):
            if all(grid[r, c + i] == mark for i in range(inarow)):
                return True

    # 纵向
    for r in range(rows - inarow + 1):
        for c in range(cols):
            if all(grid[r + i, c] == mark for i in range(inarow)):
                return True

    # 对角线 \
    for r in range(rows - inarow + 1):
        for c in range(cols - inarow + 1):
            if all(grid[r + i, c + i] == mark for i in range(inarow)):
                return True

    # 对角线 /
    for r in range(rows - inarow + 1):
        for c in range(inarow - 1, cols):
            if all(grid[r + i, c - i] == mark for i in range(inarow)):
                return True

    return False


def winning_moves(grid, mark, inarow):
    """找出 mark 当前一步能直接获胜的全部列。"""
    result = []

    for col in ordered_moves(grid):
        child = drop_piece(grid, col, mark)
        if child is not None and is_win(child, mark, inarow):
            result.append(col)

    return result


def score_window(window, mark):
    """
    从 mark 视角评价一个长度为 inarow 的窗口。

    对手棋子和己方棋子同时出现，窗口不可发展，得分为 0。
    """
    opp = 3 - mark

    own_count = int(np.count_nonzero(window == mark))
    opp_count = int(np.count_nonzero(window == opp))
    empty_count = int(np.count_nonzero(window == 0))

    if own_count > 0 and opp_count > 0:
        return 0

    # 已被对手占据的窗口：从 mark 的角度扣分。
    if opp_count > 0:
        if opp_count == 3 and empty_count == 1:
            return -1_000
        if opp_count == 2 and empty_count == 2:
            return -40
        if opp_count == 1 and empty_count == 3:
            return -2
        return 0

    # 只有自己棋子和空格的窗口。
    if own_count == 3 and empty_count == 1:
        return 1_000
    if own_count == 2 and empty_count == 2:
        return 40
    if own_count == 1 and empty_count == 3:
        return 2

    return 0


def evaluate_position(grid, mark, inarow):
    """
    非终局局面估值；正数表示 mark 更占优。

    注意：这只是叶节点启发式估值。
    必胜/必败由 Minimax 搜索中的 WIN_SCORE 处理。
    """
    rows, cols = grid.shape
    score = 0

    # 中心列奖励。
    center_cols = [cols // 2]
    if cols % 2 == 0:
        center_cols.append(cols // 2 - 1)

    for col in center_cols:
        score += 6 * int(np.count_nonzero(grid[:, col] == mark))
        score -= 6 * int(np.count_nonzero(grid[:, col] == (3 - mark)))

    # 所有长度为 inarow 的横向窗口。
    for r in range(rows):
        for c in range(cols - inarow + 1):
            score += score_window(grid[r, c:c + inarow], mark)

    # 纵向窗口。
    for r in range(rows - inarow + 1):
        for c in range(cols):
            score += score_window(grid[r:r + inarow, c], mark)

    # 对角线 \ 窗口。
    for r in range(rows - inarow + 1):
        for c in range(cols - inarow + 1):
            window = np.array([grid[r + i, c + i] for i in range(inarow)])
            score += score_window(window, mark)

    # 对角线 / 窗口。
    for r in range(rows - inarow + 1):
        for c in range(inarow - 1, cols):
            window = np.array([grid[r + i, c - i] for i in range(inarow)])
            score += score_window(window, mark)

    return score


def negamax(grid, mark, depth, alpha, beta, inarow, deadline):
    """
    Negamax 形式的 Minimax + Alpha-Beta 剪枝。

    返回值始终从“当前待走方 mark”视角解释：
      正数：当前方占优
      负数：当前方劣势
    """
    opponent = 3 - mark

    # grid 是对方刚走完后的局面；
    # 若对方已获胜，则当前方已败。
    if is_win(grid, opponent, inarow):
        return -WIN_SCORE - depth

    legal = ordered_moves(grid)

    if not legal:
        return 0  # 平局

    if depth == 0 or time.perf_counter() >= deadline:
        return evaluate_position(grid, mark, inarow)

    best_value = -float("inf")

    for col in legal:
        child = drop_piece(grid, col, mark)

        value = -negamax(
            child,
            opponent,
            depth - 1,
            -beta,
            -alpha,
            inarow,
            deadline,
        )

        best_value = max(best_value, value)
        alpha = max(alpha, value)

        # Alpha-Beta 剪枝。
        if alpha >= beta:
            break

    return best_value


def make_minimax_agent(depth=3, time_limit=0.8, seed=None):
    rng = np.random.default_rng(seed)
    """
    生成 Kaggle 兼容 agent。

    参数
    ----
    depth:
        搜索深度。推荐：
        - depth=3：速度较快，适合作为第一个强基线；
        - depth=4：更强，但本地大量对局会明显变慢；
        - depth>=5：纯 Python 下可能较慢，暂不建议用于训练。

    time_limit:
        单次思考的软时间上限（秒）。
        到时间会返回已有搜索结果，避免超时。
    """

    def minimax_agent(obs, config):
        rows = _cfg(config, "rows")
        cols = _cfg(config, "columns")
        inarow = _cfg(config, "inarow")

        board = np.asarray(obs.board, dtype=np.int8)
        grid = board.reshape(rows, cols)

        mark = obs.mark
        opponent = 3 - mark

        legal = ordered_moves(grid)
        if not legal:
            return -1

        # 先处理一步必胜：既更快，也保证战术正确。
        my_wins = winning_moves(grid, mark, inarow)
        if my_wins:
            return my_wins[0]

        # 如果对手存在一步赢，优先考虑堵住。
        # Minimax 也能发现，但显式处理可大幅减少搜索量。
        opp_wins = winning_moves(grid, opponent, inarow)
        if opp_wins:
            blockers = [c for c in legal if c in opp_wins]
            if blockers:
                return blockers[0]

        deadline = time.perf_counter() + time_limit

        best_value = -float("inf")
        best_cols = []
        alpha = -float("inf")
        beta = float("inf")

        for col in legal:
            if time.perf_counter() >= deadline:
                break

            child = drop_piece(grid, col, mark)

            value = -negamax(
                child,
                opponent,
                depth - 1,
                -beta,
                -alpha,
                inarow,
                deadline,
            )

            if value > best_value:
                best_value = value
                best_cols = [col]
            elif value == best_value:
                # 多个动作在当前搜索深度和评价函数下同样好。
                best_cols.append(col)

            alpha = max(alpha, value)

        # 若计时提前结束，至少从合法动作中选一个。
        if not best_cols:
            best_cols = legal

        return int(rng.choice(best_cols))

    return minimax_agent


# 供 agent_battle.py / train.py 直接导入的固定难度版本。
minimax_d3_agent = make_minimax_agent(
    depth=3,
    time_limit=0.5,
    seed=None,
)

minimax_d4_agent = make_minimax_agent(
    depth=4,
    time_limit=1.0,
    seed=None,
)

minimax_d3_deterministic = make_minimax_agent(
    depth=3,
    time_limit=0.5,
    seed=42,
)

minimax_d4_deterministic = make_minimax_agent(
    depth=4,
    time_limit=1.0,
    seed=42,
)