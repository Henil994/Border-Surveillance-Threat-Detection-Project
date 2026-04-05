import torch
import numpy as np
import threading
import os

from core.feature_extractor import extract_sequence
from models.ast_temporal_model import ASTTemporalModel
from ui.evaluation_ui import EvaluatorUI
import config


# =========================
# 🔥 CLASS NAMES
# =========================
CLASS_NAMES = {
    0: "Human Intrusion",
    1: "Drone Threat",
    2: "Animal Threat",
    3: "Environmental Noise"
}


# =========================
# 🔥 CALIBRATION
# =========================
def calibrated_predict(probs):

    h, d, a, n = probs

    if np.max(probs) > 0.6:
        return int(np.argmax(probs))

    if d > 0.30 and n > 0.30:
        return 1 if d > n else 3

    if h > 0.25:
        return 0

    if a > 0.30:
        return 2

    return int(np.argmax(probs))


# =========================
# 🔥 LOAD DATA (NO CSV)
# =========================
def load_fold2():

    folder = "data/UrbanSound8K/audio/fold2"

    files = [f for f in os.listdir(folder) if f.endswith(".wav")]

    paths = []
    labels = []

    for f in files:

        path = os.path.join(folder, f)

        try:
            class_id = int(f.split("-")[1])
        except:
            continue

        # 🔥 CLASSID → YOUR CLASS
        if class_id == 2:
            mapped = 0   # Human
        elif class_id == 3:
            mapped = 2   # Animal
        elif class_id in [4, 5, 7, 8]:
            mapped = 1   # Drone
        else:
            mapped = 3   # Noise

        paths.append(path)
        labels.append(mapped)

    print(f"\nUrban fold2 Loaded: {len(paths)}")

    return paths, labels


# =========================
# 🔥 EVALUATION
# =========================
def run_eval(ui):

    device = config.DEVICE

    model = ASTTemporalModel().to(device)
    model.load_state_dict(
        torch.load("models/ast_model.pth", map_location=device, weights_only=True)
    )
    model.eval()

    paths, labels = load_fold2()

    total = len(paths)
    correct = 0

    for i, (path, actual) in enumerate(zip(paths, labels), 1):

        try:
            spec_seq = extract_sequence(path, segments=1).to(device)

            with torch.no_grad():
                out = model(spec_seq.unsqueeze(0))
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]

            pred = calibrated_predict(probs)

        except:
            continue

        if pred == actual:
            correct += 1

        acc = (correct / i) * 100

        # 🔥 SEND CLASS NAMES TO UI
        if i % 5 == 0 or i == total:
            ui.update(
                i,
                total,
                acc,
                CLASS_NAMES[pred],
                CLASS_NAMES[actual]
            )

    final_acc = (correct / total) * 100
    ui.show_final(final_acc)


# =========================
# 🔥 START UI
# =========================
if __name__ == "__main__":

    print("\n⚡ Evaluation Started (FINAL)...")

    ui = EvaluatorUI("UrbanSound8K Fold2")

    thread = threading.Thread(target=run_eval, args=(ui,))
    thread.start()

    ui.start()