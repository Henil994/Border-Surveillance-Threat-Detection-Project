# evaluate_model.py (FINAL UI CONNECTED VERSION)

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F
import threading

from dataset import AudioDataset
from models.ast_model import ASTModel
from config import DEVICE, CLASS_NAMES

# 🔥 IMPORT UI
from ui.evaluation_ui import EvaluatorUI


CONF_THRESH = 0.50


def run_evaluation(ui):

    print("\n⚡ FINAL EVALUATION (UI MODE)...\n")

    dataset = AudioDataset("test")
    loader = DataLoader(dataset, batch_size=1)

    total_samples = len(dataset)

    model = ASTModel().to(DEVICE)

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=DEVICE,
            weights_only=True
        ),
        strict=False
    )

    model.eval()

    all_preds = []
    all_labels = []
    all_confs = []

    processed = 0

    with torch.no_grad():

        for data, labels in loader:

            data = data.to(DEVICE)

            outputs = model(data)

            probs = F.softmax(outputs, dim=1)

            confs, preds = probs.max(dim=1)

            pred = preds.item()
            actual = labels.item()
            conf = confs.item()

            all_preds.append(pred)
            all_labels.append(actual)
            all_confs.append(conf)

            processed += 1

            # 🔥 LIVE ACCURACY
            current_acc = (
                accuracy_score(all_labels, all_preds) * 100
            )

            # 🔥 UPDATE UI
            ui.update(
                processed,
                total_samples,
                current_acc,
                CLASS_NAMES[pred],
                CLASS_NAMES[actual]
            )

    # =========================
    # FINAL METRICS
    # =========================
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs = np.array(all_confs)

    full_acc = accuracy_score(all_labels, all_preds)

    mask = all_confs >= CONF_THRESH

    filt_preds = all_preds[mask]
    filt_labels = all_labels[mask]

    if len(filt_preds) > 0:
        filt_acc = accuracy_score(filt_labels, filt_preds)
        coverage = 100 * len(filt_preds) / len(all_preds)
    else:
        filt_acc = 0.0
        coverage = 0.0

    # =========================
    # TERMINAL REPORT
    # =========================
    print(f"\n📊 Full Accuracy: {full_acc*100:.2f}%")
    print(f"🔥 High-Confidence Accuracy: {filt_acc*100:.2f}%")
    print(f"📈 Coverage: {coverage:.2f}%")

    print("\n📊 FULL REPORT:\n")

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=CLASS_NAMES
        )
    )

    if len(filt_preds) > 0:

        print("\n📊 HIGH-CONFIDENCE REPORT:\n")

        print(
            classification_report(
                filt_labels,
                filt_preds,
                target_names=CLASS_NAMES
            )
        )

    # 🔥 FINAL UI RESULT
    ui.show_final(full_acc * 100)


def evaluate():

    ui = EvaluatorUI("UrbanSound8K")

    # 🔥 RUN EVALUATION IN THREAD
    thread = threading.Thread(
        target=run_evaluation,
        args=(ui,)
    )

    thread.daemon = True
    thread.start()

    ui.start()


if __name__ == "__main__":
    evaluate()