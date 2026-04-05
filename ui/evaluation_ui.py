import tkinter as tk
from tkinter import ttk
from collections import defaultdict


class EvaluatorUI:

    def __init__(self, dataset_name):

        self.root = tk.Tk()
        self.root.title("Threat Evaluation Console")
        self.root.geometry("820x620")
        self.root.configure(bg="#0b1d2a")

        self.class_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        title = tk.Label(
            self.root,
            text="AI BORDER THREAT EVALUATION",
            font=("Consolas", 18, "bold"),
            fg="#00ff9c",
            bg="#0b1d2a"
        )
        title.pack(pady=10)

        dataset = tk.Label(
            self.root,
            text=f"Dataset: {dataset_name}",
            font=("Consolas", 12),
            fg="#00c8ff",
            bg="#0b1d2a"
        )
        dataset.pack()

        self.progress_label = tk.Label(
            self.root,
            text="0/0",
            font=("Consolas", 12),
            fg="white",
            bg="#0b1d2a"
        )
        self.progress_label.pack(pady=5)

        self.acc_label = tk.Label(
            self.root,
            text="Accuracy: Stabilizing...",
            font=("Consolas", 14, "bold"),
            fg="#ffd166",
            bg="#0b1d2a"
        )
        self.acc_label.pack()

        self.progress_bar = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=750,
            mode="determinate"
        )
        self.progress_bar.pack(pady=10)

        # ===== CLASS PERFORMANCE PANEL =====
        self.class_panel = tk.Label(
            self.root,
            text="Class Accuracy → Initializing...",
            font=("Consolas", 11),
            fg="#ffffff",
            bg="#0b1d2a"
        )
        self.class_panel.pack(pady=5)

        self.log = tk.Text(
            self.root,
            height=20,
            bg="#050a14",
            fg="#00ff9c",
            font=("Consolas", 10)
        )
        self.log.pack(fill="both", expand=True, padx=10, pady=10)

    def update(self, processed, total, accuracy, pred, actual):

        self.progress_label.config(text=f"{processed}/{total}")
        self.progress_bar["value"] = (processed / total) * 100

        if processed < total * 0.1:
            self.acc_label.config(text="Accuracy: Stabilizing...")
        else:
            self.acc_label.config(text=f"Accuracy: {accuracy:.2f}%")

        self.class_stats[actual]["total"] += 1
        if pred == actual:
            self.class_stats[actual]["correct"] += 1

        class_text = " | ".join([
            f"{cls}: { (v['correct']/v['total']*100 if v['total'] else 0):.1f}%"
            for cls, v in self.class_stats.items()
        ])

        self.class_panel.config(text="Class Accuracy → " + class_text)

        status = "✅" if pred == actual else "❌"

        self.log.insert(
            "end",
            f"[{processed:04}] {actual} → {pred} {status}\n"
        )
        self.log.see("end")

    def show_final(self, final_acc):

        self.acc_label.config(
            text=f"FINAL ACCURACY: {final_acc:.2f}%",
            fg="#00ff9c"
        )

        self.log.insert(
            "end",
            f"\n===== FINAL ACCURACY: {final_acc:.2f}% =====\n"
        )
        self.log.see("end")

    def start(self):
        self.root.mainloop()