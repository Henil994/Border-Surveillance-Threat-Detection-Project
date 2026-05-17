import os
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

try:
    from config import SAMPLE_RATE
except:
    SAMPLE_RATE = 16000


# ===============================
# FEATURE EXTRACTION (AST FORMAT)
# ===============================
def extract_ast(path):
    waveform, sr = torchaudio.load(path)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Light augmentation
    if torch.rand(1).item() < 0.4:
        waveform = waveform + 0.005 * torch.randn_like(waveform)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=128
    )(waveform)

    mel = torch.log(mel + 1e-6)
    mel = mel.squeeze(0)

    # Normalize
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    # Resize to AST expected
    mel = F.interpolate(
        mel.unsqueeze(0).unsqueeze(0),
        size=(128, 1024),
        mode="bilinear",
        align_corners=False
    ).squeeze()

    # Final shape: [1024, 128]
    mel = mel.permute(1, 0)

    return mel


# ===============================
# CLASS MAPPING
# ===============================
def map_classid(class_id):
    if class_id in [2, 1]:
        return 0  # Human
    elif class_id in [4, 5, 7]:
        return 1  # Drone
    elif class_id == 3:
        return 2  # Animal
    else:
        return 3  # Noise


# ===============================
# MAIN PRECOMPUTE
# ===============================
def run():
    print("🚀 ROBUST PRECOMPUTE (AUTO FOLDS + SPLIT)")

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    train_labels, test_labels = [], []
    train_idx = test_idx = 0

    base_dir = "data/UrbanSound8K/audio"

    # 🔥 AUTO-DETECT AVAILABLE FOLDS
    all_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if len(all_dirs) == 0:
        raise RuntimeError("❌ No folds found in data/UrbanSound8K/audio")

    print("📁 Found folds:", [os.path.basename(d) for d in all_dirs])

    # 🔥 SIMPLE SPLIT: last folder = test, rest = train
    all_dirs.sort()
    test_dir = all_dirs[-1]
    train_dirs = all_dirs[:-1]

    print("🧪 Test fold:", os.path.basename(test_dir))
    print("🏋️ Train folds:", [os.path.basename(d) for d in train_dirs])

    # ======================
    # COUNT FOR BALANCING (TRAIN ONLY)
    # ======================
    raw_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    train_files = []
    for d in train_dirs:
        for f in os.listdir(d):
            train_files.append((d, f))

    for dir_path, file in train_files:
        try:
            class_id = int(file.split("-")[1])
            label = map_classid(class_id)
            raw_counts[label] += 1
        except:
            continue

    print("📊 RAW COUNTS:", raw_counts)

    min_count = min(raw_counts.values())
    print("🎯 BALANCING EACH CLASS TO:", min_count)

    # ======================
    # CREATE BALANCED TRAIN SET
    # ======================
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for dir_path, file in tqdm(train_files):
        path = f"{dir_path}/{file}"

        try:
            class_id = int(file.split("-")[1])
            label = map_classid(class_id)
        except:
            continue

        if class_counts[label] >= min_count:
            continue

        mel = extract_ast(path)

        torch.save(mel, f"data/processed/{train_idx}.pt")
        train_labels.append(label)

        class_counts[label] += 1
        train_idx += 1

    print("✅ FINAL TRAIN DISTRIBUTION:", class_counts)

    # ======================
    # CREATE TEST SET (NO BALANCE)
    # ======================
    for file in tqdm(os.listdir(test_dir)):
        path = f"{test_dir}/{file}"

        try:
            class_id = int(file.split("-")[1])
            label = map_classid(class_id)
        except:
            continue

        mel = extract_ast(path)

        torch.save(mel, f"data/test/{test_idx}.pt")
        test_labels.append(label)

        test_idx += 1

    torch.save(train_labels, "data/processed/labels.pt")
    torch.save(test_labels, "data/test/labels.pt")

    print("✅ DATA READY (TRAIN + TEST)")


if __name__ == "__main__":
    run()