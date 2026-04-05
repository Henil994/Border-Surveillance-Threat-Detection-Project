import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.ast_temporal_model import ASTTemporalModel
import config
from torch.cuda.amp import autocast, GradScaler


class SoundDataset(Dataset):
    def __init__(self):
        self.dir = "data/processed"
        self.labels = torch.load(f"{self.dir}/labels.pt")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = torch.load(f"{self.dir}/{idx}.pt")

        # 🔥 2 SEGMENTS (FAST)
        spec = spec.unsqueeze(0).repeat(2, 1, 1, 1)

        return spec, torch.tensor(self.labels[idx])


def train():

    dataset = SoundDataset()

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    model = ASTTemporalModel().to(config.DEVICE)

    # class weights
    class_counts = np.bincount(dataset.labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum()
    weights = weights.to(config.DEVICE)

    criterion = torch.nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=1e-4
    )

    scaler = GradScaler()

    print("\n🚀 Fast Temporal Training...\n")

    for epoch in range(config.EPOCHS):

        correct, total = 0, 0

        for x, y in loader:

            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            # 🔥 LIGHT AUGMENT
            if torch.rand(1).item() > 0.5:
                x = x + torch.randn_like(x) * 0.02

            optimizer.zero_grad()

            with autocast():
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Acc: {(correct/total)*100:.2f}%")

    torch.save(model.state_dict(), "models/ast_model.pth")
    print("\n✅ Model Saved")


if __name__ == "__main__":
    train()