from pathlib import Path
import numpy as np
import pandas as pd


# ---------- 路径与固定配置 ----------

CSV_PATH = Path("data/good_move/connect4_training_data.csv")
OUTPUT_DIR = Path("data/bc")

ROWS = 6
COLS = 7
SEED = 42
VAL_RATIO = 0.10


# ---------- 数据转换 ----------

def flat_to_kaggle_grid(flat_boards):
    """
    将数据集的 column-major-top-down 格式还原为 Kaggle 标准棋盘。

    原数据每个样本的 42 格排列方式：
        第 0 列的 6 行（从上到下）
        第 1 列的 6 行（从上到下）
        ...
        第 6 列的 6 行（从上到下）

    输出 shape:
        (N, 6, 7)

    输出棋盘与 Kaggle 的 obs.board.reshape(6, 7) 一致：
        第 0 行为顶部，第 5 行为底部。
    """
    return flat_boards.reshape(-1, COLS, ROWS).transpose(0, 2, 1)


def infer_current_mark(grids):
    """
    根据 ConnectX 严格轮流落子的规则推断轮到谁下：

      count(1) == count(2)     -> 轮到 player 1
      count(1) == count(2) + 1 -> 轮到 player 2

    返回 shape=(N,) 的 mark，值为 1 或 2。
    非法局面返回 -1。
    """
    count_1 = np.sum(grids == 1, axis=(1, 2))
    count_2 = np.sum(grids == 2, axis=(1, 2))

    marks = np.full(len(grids), -1, dtype=np.int8)
    marks[count_1 == count_2] = 1
    marks[count_1 == count_2 + 1] = 2

    return marks


def to_relative_states(grids, current_marks):
    """
    转换为当前行动者视角：

      当前玩家棋子 -> +1
      对手棋子     -> -1
      空位         ->  0

    最终展平为 (N, 42)，并使用 Kaggle row-major 格式。
    """
    states = np.zeros_like(grids, dtype=np.float32)

    # current_marks[:, None, None] 广播到每一格
    states[grids == current_marks[:, None, None]] = 1.0
    states[
        (grids != 0)
        & (grids != current_marks[:, None, None])
    ] = -1.0

    return states.reshape(-1, ROWS * COLS)


def legal_masks_from_grids(grids):
    """
    Kaggle ConnectX 中，第 0 行是顶部。
    某列顶部为空，表示这列还能落子。

    输出 shape=(N, 7)，True=合法。
    """
    return (grids[:, 0, :] == 0)


def horizontal_mirror(states, actions, legal_masks):
    """
    水平镜像增强。

    states 原本是 (N, 42)，按 (6, 7) row-major 展平。
    翻转列维度后重新展平。

    动作 c 映射为：6 - c。
    """
    grids = states.reshape(-1, ROWS, COLS)

    mirrored_states = np.flip(grids, axis=2).reshape(-1, ROWS * COLS)
    mirrored_actions = (COLS - 1 - actions).astype(np.int64)
    mirrored_masks = np.flip(legal_masks, axis=1)

    return (
        mirrored_states.astype(np.float32),
        mirrored_actions,
        mirrored_masks.astype(bool),
    )


def validate_dataset(states, actions, legal_masks, name):
    """输出并检查准备好的数据。"""
    print(f"\n--- 校验 {name} ---")
    print(f"states shape      : {states.shape}")
    print(f"actions shape     : {actions.shape}")
    print(f"legal_masks shape : {legal_masks.shape}")

    assert states.dtype == np.float32
    assert actions.dtype == np.int64
    assert legal_masks.dtype == bool

    assert states.ndim == 2 and states.shape[1] == 42
    assert actions.ndim == 1
    assert legal_masks.shape == (len(states), COLS)

    unique_values = np.unique(states)
    print(f"state 取值        : {unique_values}")
    print(f"action 范围       : {actions.min()} ~ {actions.max()}")

    # 每条标签动作必须被 legal mask 允许。
    target_legal = legal_masks[np.arange(len(actions)), actions]
    legal_rate = target_legal.mean()

    print(f"标签动作合法率    : {legal_rate:.4%}")

    assert np.all(np.isin(unique_values, [-1.0, 0.0, 1.0]))
    assert np.all((0 <= actions) & (actions < COLS))
    assert np.all(target_legal), f"{name} 中存在标签指向满列。"

    # 每个局面至少应有一个合法列。
    assert np.all(legal_masks.any(axis=1)), f"{name} 存在无合法动作局面。"


def save_npz(path, states, actions, legal_masks):
    """压缩保存。"""
    np.savez_compressed(
        path,
        states=states.astype(np.float32),
        actions=actions.astype(np.int64),
        legal_masks=legal_masks.astype(bool),
    )
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"已保存: {path} ({size_mb:.2f} MB)")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到原始数据：{CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("读取 CSV（header=None）...")
    df = pd.read_csv(CSV_PATH, header=None)

    if df.shape[1] != 43:
        raise ValueError(
            f"预期 43 列（42 棋盘格 + 1 动作），实际是 {df.shape[1]} 列。"
        )

    raw = df.to_numpy(dtype=np.int8)
    flat_boards = raw[:, :42]
    actions = raw[:, 42].astype(np.int64)

    print(f"原始样本数: {len(actions)}")

    # 1. column-major-top-down -> Kaggle 标准 (N, 6, 7)
    grids = flat_to_kaggle_grid(flat_boards)

    # 2. 推断当前轮到哪一方
    current_marks = infer_current_mark(grids)

    # 3. 过滤异常局面（理论上应为 0）
    valid_turn = current_marks != -1
    valid_action_range = (actions >= 0) & (actions < COLS)

    keep = valid_turn & valid_action_range

    print(f"轮次合法样本数: {valid_turn.sum()} / {len(actions)}")
    print(f"动作范围合法数: {valid_action_range.sum()} / {len(actions)}")
    print(f"保留样本数    : {keep.sum()} / {len(actions)}")

    grids = grids[keep]
    actions = actions[keep]
    current_marks = current_marks[keep]

    # 4. 相对编码 + 合法 mask
    states = to_relative_states(grids, current_marks)
    legal_masks = legal_masks_from_grids(grids)

    # 5. 再次检查“标签动作是否真的合法”
    action_is_legal = legal_masks[np.arange(len(actions)), actions]

    if not np.all(action_is_legal):
        bad_count = int((~action_is_legal).sum())
        raise ValueError(
            f"发现 {bad_count} 条标签动作指向已满列。"
            "棋盘解析方向或 action 解释存在问题。"
        )

    # 6. 先随机划分原始局面，再做训练集镜像增强
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(actions))

    val_size = int(len(indices) * VAL_RATIO)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_states = states[train_idx]
    train_actions = actions[train_idx]
    train_masks = legal_masks[train_idx]

    val_states = states[val_idx]
    val_actions = actions[val_idx]
    val_masks = legal_masks[val_idx]

    # 7. 只增强训练集，避免 validation 数据泄漏
    mirror_states, mirror_actions, mirror_masks = horizontal_mirror(
        train_states,
        train_actions,
        train_masks,
    )

    train_states = np.concatenate(
        [train_states, mirror_states],
        axis=0,
    )
    train_actions = np.concatenate(
        [train_actions, mirror_actions],
        axis=0,
    )
    train_masks = np.concatenate(
        [train_masks, mirror_masks],
        axis=0,
    )

    # 8. 打乱增强后的训练集
    permutation = rng.permutation(len(train_actions))
    train_states = train_states[permutation]
    train_actions = train_actions[permutation]
    train_masks = train_masks[permutation]

    # 9. 最终检查与保存
    validate_dataset(train_states, train_actions, train_masks, "train")
    validate_dataset(val_states, val_actions, val_masks, "validation")

    print("\n动作分布（训练集）：")
    counts = np.bincount(train_actions, minlength=COLS)
    for col, count in enumerate(counts):
        print(f"  column {col}: {count:7d} ({count / len(train_actions):.2%})")

    save_npz(
        OUTPUT_DIR / "train.npz",
        train_states,
        train_actions,
        train_masks,
    )
    save_npz(
        OUTPUT_DIR / "val.npz",
        val_states,
        val_actions,
        val_masks,
    )

    print("\n完成。")
    print(
        f"训练集: {len(train_actions)} 条（含镜像增强）\n"
        f"验证集: {len(val_actions)} 条（不做镜像）"
    )


if __name__ == "__main__":
    main()