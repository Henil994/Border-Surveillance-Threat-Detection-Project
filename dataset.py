import os
import pandas as pd
import random


def load_esc50(path):

    df = pd.read_csv(os.path.join(path, "meta", "esc50.csv"))
    audio_dir = os.path.join(path, "audio")

    files, labels = [], []

    for _, row in df.iterrows():

        category = row["category"]
        file_path = os.path.join(audio_dir, row["filename"])

        if category in ["speech", "footsteps"]:
            label = 0
        elif category in ["engine", "helicopter"]:
            label = 1
        elif category in ["dog", "cat"]:
            label = 2
        elif category in ["rain", "wind"]:
            label = 3
        else:
            continue

        files.append(file_path)
        labels.append(label)

    while len(files) < 1000:
        i = random.randint(0, len(files) - 1)
        files.append(files[i])
        labels.append(labels[i])

    print("ESC Loaded:", len(files))
    return files, labels


def load_urban_train(path, fold="fold1", split=0.7):

    folder = os.path.join(path, "audio", fold)

    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    random.seed(42)
    random.shuffle(files)

    selected = files[:int(split * len(files))]

    out_files, labels = [], []

    for f in selected:

        class_id = int(f.split("-")[1])

        # ✅ FINAL CORRECT MAPPING
        if class_id in [2,6,8]:   # gunshot → human intrusion proxy
            label = 0
        elif class_id in [4,5,7]:
            label = 1
        elif class_id == 3:
            label = 2
        elif class_id in [0,1,9]:
            label = 3
        else:
            continue

        out_files.append(os.path.join(folder, f))
        labels.append(label)

    print("Urban fold1 (70%) Loaded:", len(out_files))
    return out_files, labels


def load_urban_test(path, fold="fold2"):

    folder = os.path.join(path, "audio", fold)

    files, labels = [], []

    for f in os.listdir(folder):

        if not f.endswith(".wav"):
            continue

        class_id = int(f.split("-")[1])

        if class_id in [2,6,8]:
            label = 0
        elif class_id in [4,5,7]:
            label = 1
        elif class_id == 3:
            label = 2
        elif class_id in [0,1,9]:
            label = 3
        else:
            continue

        files.append(os.path.join(folder, f))
        labels.append(label)

    print("Urban fold2 Loaded:", len(files))
    return files, labels


def load_training_data():
    esc_f, esc_l = load_esc50("data/ESC-50")
    urb_f, urb_l = load_urban_train("data/UrbanSound8K")

    files = esc_f + urb_f
    labels = esc_l + urb_l

    print("Total Training:", len(files))
    return files, labels


def load_test_data():
    return load_urban_test("data/UrbanSound8K")