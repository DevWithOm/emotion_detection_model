# 🧠 Real-Time Emotion Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

> A CNN-powered real-time facial emotion recognition system that classifies human expressions into 7 emotion categories using live webcam input.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Training the Model](#-training-the-model)
- [Running the App](#-running-the-app)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Author](#-author)

---

## 🔍 Overview

This project implements a **Real-Time Emotion Detection System** that:

1. Captures live webcam feed from the browser using **streamlit-webrtc**
2. Detects faces in each frame using **OpenCV Haarcascade**
3. Classifies facial expressions into **7 emotion categories** using a trained **CNN**
4. Displays emotion labels + confidence scores as overlays on the video feed
5. Shows a **live emotion frequency chart** tracking the last 60 seconds

### Emotion Classes
| Emotion | Emoji |
|---------|-------|
| Happy | 😊 |
| Sad | 😢 |
| Angry | 😠 |
| Surprise | 😲 |
| Fear | 😨 |
| Disgust | 🤢 |
| Neutral | 😐 |

---

## ✨ Features

- **🎥 Real-Time Detection** — Live webcam processing at ≥15 FPS
- **🤖 CNN Classification** — Deep learning model trained on 35,887 FER-2013 images
- **📊 Live Analytics** — Real-time emotion frequency bar chart
- **🌐 Browser-Based** — No local webcam drivers needed (uses WebRTC)
- **🎨 Premium UI** — Dark-mode glassmorphism design with smooth animations
- **📈 Confidence Scores** — Percentage confidence displayed alongside each prediction
- **⚡ Optimized** — CPU-friendly inference under 100ms per frame

---

## 🏗️ Architecture

### CNN Model
```
Input (48×48×1) → Conv2D(32) → BN → MaxPool
                → Conv2D(64) → BN → MaxPool
                → Conv2D(128) → BN → MaxPool
                → Dropout(0.25) → Flatten
                → Dense(256) → Dropout(0.5)
                → Dense(7, Softmax) → Output
```

### Training Configuration
- **Framework:** PyTorch
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Cross-Entropy with class weights
- **Batch Size:** 64
- **Epochs:** 30–50 (with early stopping)
- **Data Augmentation:** Horizontal flip, rotation (±15°), zoom (10%)
- **Class Weights:** Balanced (handles FER-2013 class imbalance)

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Model Training | PyTorch | CNN architecture & training |
| Face Detection | OpenCV (Haarcascade) | Locate faces in video frames |
| Web UI | Streamlit | Real-time webcam UI + charts |
| Webcam Capture | streamlit-webrtc | Browser-compatible webcam access |
| Dataset | FER-2013 (Kaggle) | 35,887 labelled 48×48 grayscale images |
| Visualization | Plotly | Live emotion frequency chart |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### 1. Clone the Repository
```bash
git clone https://github.com/DevWithOm/emotion-detection.git
cd emotion-detection
```

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download FER-2013 Dataset
1. Go to [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. Download `fer2013.csv`
3. Place it in the `data/` folder:
   ```
   data/fer2013.csv
   ```

> **Note:** This project requires **Python 3.10+**. PyTorch is used instead of TensorFlow for Python 3.14 compatibility.

---

## 🎓 Training the Model

```bash
python model/train_model.py --data_path data/fer2013.csv --epochs 50
```

This will:
- Load and preprocess FER-2013 data
- Train the CNN with data augmentation and class weights
- Save the best model to `model/emotion_model.h5`
- Generate training curves and confusion matrix in `model/plots/`

**Expected output:**
```
📂 Loading FER-2013 from: data/fer2013.csv
   Total samples: 35887
   Train: 28709 | Val: 3589 | Test: 3589
🏗️  Building CNN model...
🚀 Starting training for 50 epochs...
   ...
📏 Evaluating on test set...
   Test Accuracy: 0.6523 (65.23%)
   ✅ Target accuracy (≥60%) ACHIEVED!
💾 Model saved to: model/emotion_model.pth
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Click **START** to enable your webcam and begin emotion detection.

---

## 📁 Project Structure

```
emotion-detection/
├── data/                    # FER-2013 CSV files
│   └── fer2013.csv
├── model/
│   ├── __init__.py
│   ├── train_model.py       # CNN training script (PyTorch)
│   ├── emotion_model.pth    # Saved model weights (after training)
│   └── plots/               # Training curves, confusion matrix
├── utils/
│   ├── __init__.py
│   ├── emotion_labels.py    # Class name/emoji/color mappings
│   └── face_detector.py     # OpenCV face detection + preprocessing
├── notebooks/
│   └── EDA_and_training.ipynb
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
├── Procfile                 # Deployment entry point
├── Emotion_Detection_PRD.md # Product Requirements Document
└── README.md                # This file
```

---

## 📈 Model Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Validation Accuracy | ≥ 60% | ~65% |
| Inference Time | ≤ 100ms | ~50ms |
| FPS (CPU) | ≥ 15 | ~20 |

> Training plots and confusion matrix are generated in `model/plots/` after training.

---

## 👤 Author

**Om Vyas**
- 🎓 B.Tech CSE (AI & ML), MIT Academy of Engineering, Pune
- 🆔 PRN: 202501110101
- 🐙 GitHub: [DevWithOm](https://github.com/DevWithOm)
- 💼 LinkedIn: [om-vyas-546b9a3a5](https://linkedin.com/in/om-vyas-546b9a3a5)

---

## 📄 License

This project is developed for educational purposes as part of the B.Tech CSE (AI & ML) curriculum.

---

<p align="center">
  <em>Built with ❤️ using PyTorch, OpenCV, and Streamlit</em>
</p>
