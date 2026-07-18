import numpy as np
import random

# ── 窗口评分常量 ──
# 数值越大代表该局面越有利，4连必胜设为极大值确保优先选择
SCORE_4 = 100000   # 四连（必胜）
SCORE_3 = 100      # 三连（强力威胁）
SCORE_2 = 10       # 两连（发展潜力）
SCORE_1 = 1        # 一连（基础铺垫）
CENTER_WEIGHT = 5  # 中心列偏置权重，让AI更偏好中间列


def drop_piece(grid, col, piece, config):
    """模拟在某列落子，返回 (落子后的网格, 落子行号)"""
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            next_grid[row][col] = piece
            return next_grid, row
    return next_grid, None  # 列已满


def is_win(grid, piece, config):
    """检查指定玩家是否已在棋盘上获胜（完整四方向检测）"""
    rows, cols, inarow = config.rows, config.columns, config.inarow

    # 水平方向检查
    for r in range(rows):
        for c in range(cols - inarow + 1):
            if all(grid[r][c + i] == piece for i in range(inarow)):
                return True

    # 垂直方向检查
    for r in range(rows - inarow + 1):
        for c in range(cols):
            if all(grid[r + i][c] == piece for i in range(inarow)):
                return True

    # 正对角线（\）检查
    for r in range(rows - inarow + 1):
        for c in range(cols - inarow + 1):
            if all(grid[r + i][c + i] == piece for i in range(inarow)):
                return True

    # 反对角线（/）检查
    for r in range(rows - inarow + 1):
        for c in range(inarow - 1, cols):
            if all(grid[r + i][c - i] == piece for i in range(inarow)):
                return True

    return False


def score_window(window, piece):
    """
    评估一个连续4格窗口对指定玩家的价值

    规则：
    - 窗口内同时包含双方棋子 → 被阻挡，价值为0
    - 窗口内只有己方棋子 → 按棋子数量递增得分
    """
    opp = 1 if piece == 2 else 2
    count_piece = sum(1 for x in window if x == piece)
    count_opp = sum(1 for x in window if x == opp)

    if count_opp > 0:
        return 0  # 被对手棋子阻挡，该窗口无潜力

    # 按己方棋子数量映射分值
    score_map = {
        0: 0,
        1: SCORE_1,
        2: SCORE_2,
        3: SCORE_3,
        4: SCORE_4,
    }
    return score_map.get(count_piece, 0)


def score_move(grid, col, piece, config):
    """
    评估在指定列落子后的局面分数

    策略：只检查所有「包含新落子」的4格窗口，避免扫全盘
    ——检测4个方向上、4种起始偏移，共最多 4×4=16 个窗口
    """
    new_grid, row = drop_piece(grid, col, piece, config)
    if row is None:
        return -1  # 列已满，不可下

    score = 0
    inarow = config.inarow

    # 四个方向：(行增量, 列增量)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        # 在每个方向上，枚举「包含(row, col)」的4格窗口的起始偏移
        for offset in range(inarow):
            start_r = row - offset * dr
            start_c = col - offset * dc
            end_r = start_r + (inarow - 1) * dr
            end_c = start_c + (inarow - 1) * dc

            # 检查窗口是否在棋盘范围内
            if not (0 <= start_r < config.rows and 0 <= start_c < config.columns):
                continue
            if not (0 <= end_r < config.rows and 0 <= end_c < config.columns):
                continue

            # 提取窗口内容
            window = [new_grid[start_r + i * dr][start_c + i * dc]
                      for i in range(inarow)]
            score += score_window(window, piece)

    # 中心列偏置：离中间越近加分越多，打破平局
    center = (config.columns - 1) / 2
    score += (CENTER_WEIGHT - abs(col - center))

    return score


def heuristic_agent(obs, config):
    """
    启发式规则AI —— 完整一阶启发式（one-ply lookahead）

    决策流程：
    1. 进攻：如果能一步获胜，直接下
    2. 防守：如果对手下一步能赢，堵住
    3. 评分：对每个合法列计算局面评分，选最高分（同分随机）
    4. 利用中心偏置打破平局
    """
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    piece = obs.mark
    opp = 1 if piece == 2 else 2

    # 合法列：该列顶端为空（即该列未满）
    valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]

    if not valid_moves:
        return -1  # 无棋可下（理论上不会发生）

    # ═══ 1. 进攻：能赢就下 ═══
    for col in valid_moves:
        new_grid, _ = drop_piece(grid, col, piece, config)
        if is_win(new_grid, piece, config):
            return col

    # ═══ 2. 防守：堵住对手的必胜棋 ═══
    for col in valid_moves:
        new_grid, _ = drop_piece(grid, col, opp, config)
        if is_win(new_grid, opp, config):
            return col

    # ═══ 3. 局面评分：选择最优列 ═══
    best_score = float('-inf')
    candidates = []
    for col in valid_moves:
        s = score_move(grid, col, piece, config)
        if s > best_score:
            best_score = s
            candidates = [col]
        elif s == best_score:
            candidates.append(col)

    # 同分时随机选择，避免确定性行为
    return random.choice(candidates)
