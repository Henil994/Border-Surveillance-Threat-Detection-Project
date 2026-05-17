import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, mode="train"):

        if mode == "train":
            self.dir = "data/processed"
        else:
            self.dir = "data/test"

        print("⚡ Loading dataset into RAM...")

        self.labels = torch.load(f"{self.dir}/labels.pt", weights_only=True)
        self.labels = torch.tensor(self.labels)

        self.data = [
            torch.load(f"{self.dir}/{i}.pt", weights_only=True)
            for i in range(len(self.labels))
        ]

        print(f"✅ Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]