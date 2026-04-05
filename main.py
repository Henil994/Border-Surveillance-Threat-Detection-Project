import tkinter as tk
import random
import os
from ui.dashboard import SurveillanceDashboard
from core.predictor import ThreatPredictor


root = tk.Tk()

dashboard = SurveillanceDashboard(root)

predictor = ThreatPredictor()


# Folder containing ESC-50 audio samples
DATASET_FOLDER = "data/ESC-50/audio"


def monitor():

    # Pick random audio file for demo
    audio_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".wav")]

    audio_file = os.path.join(DATASET_FOLDER, random.choice(audio_files))

    label, confidence, probs = predictor.predict_from_audio(audio_file)

    dashboard.update_dashboard(label, confidence, probs)

    root.after(3000, monitor)


monitor()

root.mainloop()