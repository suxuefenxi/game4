from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from bc_policy import BCPolicy


# ---------- 配置 ----------

TRAIN_PATH = Path("data/bc/train.npz")
VAL_PATH = Path("data/bc/val.npz")
OUTPUT_DIR = Path("runs/bc_v1")

SEED = 42
BATCH_SIZE = 1024
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
PATIENCE = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Connect4BCDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)

        self.states = data["states"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        self.legal_masks = data["legal_masks"].astype(bool)

        assert len(self.states) == len(self.actions)
        assert self.legal_masks.shape == (len(self.states), 7)

        # 数据准备阶段已经验证过，这里再保护一次。
        assert np.all(
            self.legal_masks[np.arange(len(self.actions)), self.actions]
        ), "训练集中存在标签指向非法动作。"

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.states[index]),
            torch.tensor(self.actions[index], dtype=torch.long),
            torch.from_numpy(self.legal_masks[index]),
        )


def masked_logits(logits, legal_masks):
    """
    将非法列设为极小值。

    logits:      (batch, 7)
    legal_masks: (batch, 7), bool
    """
    return logits.masked_fill(~legal_masks, -1e9)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    for states, actions, legal_masks in loader:
        states = states.to(DEVICE, non_blocking=True)
        actions = actions.to(DEVICE, non_blocking=True)
        legal_masks = legal_masks.to(DEVICE, non_blocking=True).bool()

        logits = model(states)
        logits = masked_logits(logits, legal_masks)

        loss = criterion(logits, actions)

        batch_size = states.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == actions).sum().item()

    return {
        "loss": total_loss / total_samples,
        "accuracy": correct / total_samples,
    }


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"device: {DEVICE}")
    print("加载数据...")

    train_dataset = Connect4BCDataset(TRAIN_PATH)
    val_dataset = Connect4BCDataset(VAL_PATH)

    print(f"训练集: {len(train_dataset):,}")
    print(f"验证集: {len(val_dataset):,}")

    pin_memory = (DEVICE == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # Windows 下先保持 0，最稳定
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = BCPolicy().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 分类目标：专家动作列。
    criterion = torch.nn.CrossEntropyLoss()

    # 验证损失连续 PATIENCE 轮不改进则停止。
    best_val_loss = float("inf")
    no_improve_epochs = 0

    best_path = OUTPUT_DIR / "bc_policy_best.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        total_train_samples = 0
        total_train_correct = 0

        for states, actions, legal_masks in train_loader:
            states = states.to(DEVICE, non_blocking=True)
            actions = actions.to(DEVICE, non_blocking=True)
            legal_masks = legal_masks.to(DEVICE, non_blocking=True).bool()

            optimizer.zero_grad()

            logits = model(states)
            logits = masked_logits(logits, legal_masks)

            loss = criterion(logits, actions)
            loss.backward()

            # 防止偶发梯度爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            batch_size = states.shape[0]
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

            predictions = torch.argmax(logits, dim=1)
            total_train_correct += (predictions == actions).sum().item()

        train_loss = total_train_loss / total_train_samples
        train_acc = total_train_correct / total_train_samples

        val_metrics = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.2%} | "
            f"val loss={val_metrics['loss']:.4f}, "
            f"acc={val_metrics['accuracy']:.2%}"
        )

        # 用 validation loss 保存最佳模型，而不是最后一轮模型。
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "architecture": "42-128-128-7-tanh",
                },
                best_path,
            )

            print(f"  ✓ 保存最佳模型: {best_path}")
        else:
            no_improve_epochs += 1

            if no_improve_epochs >= PATIENCE:
                print(
                    f"验证集连续 {PATIENCE} 轮未提升，提前停止。"
                )
                break

    print("\n训练完成。")

    checkpoint = torch.load(best_path, map_location="cpu")
    print(f"最佳 epoch: {checkpoint['epoch']}")
    print(f"最佳 val loss: {checkpoint['val_loss']:.4f}")
    print(f"最佳 val accuracy: {checkpoint['val_accuracy']:.2%}")
    print(f"模型文件: {best_path}")


if __name__ == "__main__":
    main()