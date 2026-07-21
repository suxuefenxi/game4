import numpy as np


def valid_moves(board, config):
    """
    board: 一维 42 格 Kaggle 原始棋盘，值为 0/1/2。
    返回未满列。
    """
    return [col for col in range(config.columns) if board[col] == 0]


def drop_piece(board, col, mark, config):
    """
    在 col 列模拟落下一枚 mark。

    返回新的二维棋盘；如果列满则返回 None。
    """
    grid = np.asarray(board, dtype=np.int8).reshape(
        config.rows, config.columns
    ).copy()

    for row in range(config.rows - 1, -1, -1):
        if grid[row, col] == 0:
            grid[row, col] = mark
            return grid

    return None


def is_win(grid, mark, config):
    """检查 mark 是否已达成 inarow 连子。"""
    rows = config.rows
    cols = config.columns
    n = config.inarow

    # 水平
    for r in range(rows):
        for c in range(cols - n + 1):
            if all(grid[r, c + i] == mark for i in range(n)):
                return True

    # 垂直
    for r in range(rows - n + 1):
        for c in range(cols):
            if all(grid[r + i, c] == mark for i in range(n)):
                return True

    # 对角线 \
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            if all(grid[r + i, c + i] == mark for i in range(n)):
                return True

    # 对角线 /
    for r in range(rows - n + 1):
        for c in range(n - 1, cols):
            if all(grid[r + i, c - i] == mark for i in range(n)):
                return True

    return False


def winning_moves(board, mark, config):
    """返回当前玩家一步即可获胜的所有列。"""
    wins = []

    for col in valid_moves(board, config):
        next_grid = drop_piece(board, col, mark, config)
        if next_grid is not None and is_win(next_grid, mark, config):
            wins.append(col)

    return wins


def tactical_action_mask(board, mark, config):
    """
    返回与动作空间等长的 bool mask。

    优先级：
    1. 自己有一步必胜：仅允许必胜动作；
    2. 否则，若存在不会让对方下一手直接获胜的动作：仅允许安全动作；
    3. 若所有动作都会送给对手一步必胜：保留所有合法动作。

    True = 允许 PPO / 推理选择
    False = 被战术规则排除
    """
    board = np.asarray(board, dtype=np.int8)
    legal = valid_moves(board, config)

    mask = np.zeros(config.columns, dtype=bool)

    if not legal:
        return mask

    # 1) 自己立即获胜：不需要网络判断。
    my_wins = winning_moves(board, mark, config)
    if my_wins:
        mask[my_wins] = True
        return mask

    # 2) 模拟自己走每个合法动作后，
    #    检查对手是否存在一步获胜。
    opponent = 3 - mark
    safe_moves = []

    for col in legal:
        next_grid = drop_piece(board, col, mark, config)

        # 当前落子若已终局，本应已在 my_wins 中处理；
        # 保留这个判断仅作保护。
        if is_win(next_grid, mark, config):
            safe_moves.append(col)
            continue

        next_board = next_grid.reshape(-1)
        opp_wins = winning_moves(next_board, opponent, config)

        if not opp_wins:
            safe_moves.append(col)

    # 存在安全走法：排除一步送杀走法。
    if safe_moves:
        mask[safe_moves] = True
    else:
        # 没有安全动作，说明是强制败局或双重威胁；
        # 不制造“全 False mask”，保留所有合法列。
        mask[legal] = True

    return mask