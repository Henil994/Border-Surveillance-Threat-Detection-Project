import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from collections import Counter

from dataset import AudioDataset
from models.ast_model import ASTModel
from config import DEVICE


def train():

    print("\n🚀 FINAL TRAINING (HIGH ACCURACY MODE)...\n")

    dataset = AudioDataset("train")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=8
    )

    model = ASTModel().to(DEVICE)

    # 🔥 Dynamic class weights
    labels = dataset.labels.numpy()

    counts = Counter(labels)
    total = sum(counts.values())

    weights = torch.tensor(
        [total / counts[i] for i in range(4)],
        dtype=torch.float32
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5,               # 🔥 Lower LR
        weight_decay=1e-4      # 🔥 Better regularization
    )

    best_val = 0

    patience = 4
    no_improve = 0

    for epoch in range(10):

        model.train()

        correct = total_count = 0

        for data, labels in train_loader:

            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(data)

            loss = criterion(outputs, labels)

            loss.backward()

            # 🔥 Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total_count += labels.size(0)

        train_acc = 100 * correct / total_count

        # ==========================
        # VALIDATION
        # ==========================
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for data, labels in val_loader:

                data = data.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(data)

                preds = outputs.argmax(1)

                val_correct += (
                    preds == labels
                ).sum().item()

                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        print(
            f"Epoch {epoch+1} | "
            f"Train={train_acc:.2f}% | "
            f"Val={val_acc:.2f}%"
        )

        # 🔥 SAVE BEST MODEL
        if val_acc > best_val:

            best_val = val_acc
            no_improve = 0

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

            print("✅ BEST MODEL SAVED")

        else:
            no_improve += 1

        # 🔥 EARLY STOPPING
        if no_improve >= patience:

            print("⛔ Early stopping")
            break

    print("\n🏁 FINAL BEST VALIDATION:", best_val)
    print("🏁 Training Done")


if __name__ == "__main__":
    train()