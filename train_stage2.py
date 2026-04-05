import torch
from torch.utils.data import Dataset, DataLoader
from models.stage2_model import Stage2Model


class Stage2Dataset(Dataset):
    def __init__(self):
        self.dir = "data/processed"
        self.labels = torch.load(f"{self.dir}/labels.pt")

        self.indices = []
        self.new_labels = []

        for i, label in enumerate(self.labels):
            if label in [1, 3]:
                self.indices.append(i)
                self.new_labels.append(0 if label == 1 else 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        spec = torch.load(f"{self.dir}/{i}.pt")
        return spec, torch.tensor(self.new_labels[idx])


def train():

    dataset = Stage2Dataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Stage2Model().to(device)

    # ✅ BALANCED LOSS
    labels = torch.tensor(dataset.new_labels)
    counts = torch.bincount(labels)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum()
    weights = weights.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("\n🚀 Stage-2 Training...\n")

    for epoch in range(8):

        correct, total = 0, 0

        for x, y in loader:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        print(f"Epoch {epoch+1} | Acc: {(correct/total)*100:.2f}%")

    torch.save(model.state_dict(), "models/stage2_model.pth")
    print("✅ Stage2 Model Saved")


if __name__ == "__main__":
    train()