# 🚨 AI Acoustic Threat Detection System

## 📌 Overview

This project is a **research-grade AI-based border surveillance system** that detects acoustic threats in real-time.

The system classifies audio into 4 critical categories:

* 👤 Human Intrusion
* 🚁 Drone / Mechanical Threat
* 🐾 Animal Threat
* 🌧 Environmental Noise

It combines **deep learning + real-time UI + alert system** to simulate an intelligent border monitoring solution.

---

## 🧠 Architecture

Audio Input
↓
Log-Mel Spectrogram
↓
CNN Feature Extractor
↓
Vision Transformer (ViT)
↓
LSTM (Temporal Modeling)
↓
Threat Classification

---

## ⚙️ Technologies Used

* PyTorch (Deep Learning)
* torchaudio (Audio Processing)
* timm (Vision Transformer)
* Tkinter (UI Dashboard)
* NumPy / Pandas

---

## 🚀 Features

* 🎯 Real-time threat detection dashboard
* 📡 Radar-based visualization
* 🗺 Location-based alert system
* 🔊 Audio-based AI classification
* 📊 Evaluation UI with accuracy tracking
* ⚡ Fast prediction system

---

## 📂 Dataset

### Training:

* ESC-50 Dataset
* UrbanSound8K (Fold1 - 70%)

### Testing:

* UrbanSound8K (Fold2 - 100%)

---

## 📈 Model Performance

* Accuracy: **~60–70%**
* Strong Classes:

  * Animal Threat
  * Drone Threat
* Moderate:

  * Human Intrusion
* Challenging:

  * Environmental Noise (due to similarity with mechanical sounds)

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

pip install -r requirements.txt

---

### 2️⃣ Run Real-Time System

python main.py

---

### 3️⃣ Evaluate Model

python evaluate_model.py

---

## 🖥️ System Output

* Real-time threat alerts
* Confidence score
* Threat level indicator (LOW / MEDIUM / HIGH)
* Radar visualization
* Alert sound (beep)

---

## 📊 Evaluation System

* Live accuracy tracking
* Class-wise performance
* Prediction logs

---

## 🔮 Future Improvements

* Improve Drone vs Noise separation
* Real-time microphone integration
* Deploy on edge devices
* Increase model accuracy with better dataset balancing

---

## 👨‍💻 Author

**Henil Patel**

---

## 📌 Note

This project demonstrates a **complete AI pipeline** including:

* Data processing
* Model training
* Real-time inference
* UI integration
* Evaluation system

---

## ⭐ Project Highlights

* End-to-end AI system
* Real-time visualization
* Multi-stage deep learning model
* Practical surveillance application
