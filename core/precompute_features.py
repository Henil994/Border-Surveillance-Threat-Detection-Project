import sys, os
from tqdm import tqdm
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from dataset import load_training_data
from core.feature_extractor import extract_features

SAVE_DIR = "data/processed"
os.makedirs(SAVE_DIR, exist_ok=True)

files, labels = load_training_data()

print("\n⚡ Precomputing features...\n")

new_labels = []
idx = 0

for i, path in enumerate(tqdm(files)):

    label = labels[i]

    spec = extract_features(path)
    torch.save(spec, os.path.join(SAVE_DIR, f"{idx}.pt"))
    new_labels.append(label)
    idx += 1

    # 🔥 Human boost
    if label == 0:
        for _ in range(2):
            spec_dup = extract_features(path)
            torch.save(spec_dup, os.path.join(SAVE_DIR, f"{idx}.pt"))
            new_labels.append(label)
            idx += 1

torch.save(new_labels, os.path.join(SAVE_DIR, "labels.pt"))

print("\n✅ Feature caching done!")