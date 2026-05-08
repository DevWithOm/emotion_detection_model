**PRODUCT REQUIREMENTS DOCUMENT**

Real-Time Emotion Detection System

*Powered by AntiGravity · CNN \+ OpenCV · Streamlit*

| Version | 1.0 |
| :---- | :---- |
| **Author** | Om Vyas — B.Tech CSE (AI & ML), MIT Academy of Engineering, Pune |
| **PRN** | 202501110101 |
| **Status** | Draft — Ready for Review |
| **Date** | 6 May 2026 |
| **Deployment** | AntiGravity (antigravity.pythonanywhere.com) |

# **1\. Executive Summary**

This PRD defines the complete scope, architecture, functional requirements, and delivery plan for a Real-Time Emotion Detection System — a portfolio-grade AI/ML project developed as part of the B.Tech CSE (AI & ML) curriculum at MIT Academy of Engineering, Pune.

The system will detect human facial expressions from a live webcam feed and classify them into seven emotion categories using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset. The final product will be deployed publicly via AntiGravity, making it accessible from any browser without local setup.

# **2\. Problem Statement**

Human emotion recognition is a critical capability in human-computer interaction (HCI), mental health monitoring, and adaptive systems. Existing solutions are either expensive, cloud-locked, or require proprietary hardware. There is a clear need for an open, lightweight, real-time emotion detection system that:

* Runs in real-time on standard consumer webcams

* Is trained on a well-known, reproducible dataset (FER-2013)

* Can be deployed publicly as a web application (no local installation needed)

* Serves as a demonstrable, explainable AI/ML portfolio project

# **3\. Goals & Non-Goals**

## **3.1 Goals**

1. Train a CNN model achieving ≥60% validation accuracy on FER-2013

2. Detect and classify faces in real-time from webcam input at ≥15 FPS

3. Display emotion label \+ confidence score as an overlay on the video feed

4. Show a live emotion history bar chart (last 60 seconds)

5. Deploy publicly on AntiGravity with a shareable URL

6. Document the project so it can be explained in a viva / college presentation

## **3.2 Non-Goals**

* Voice/audio-based emotion detection (v2 scope)

* Multi-person simultaneous emotion tracking (v2 scope)

* Medical-grade accuracy or clinical use

* Mobile app (Android/iOS)

# **4\. Users & Stakeholders**

| Stakeholder | Role | Success Metric |
| ----- | ----- | ----- |
| Om Vyas (Developer) | Builder, presenter, submitter | Project runs live, viva passed |
| College Faculty | Evaluator / Instructor | Working demo, clean code, documented |
| Recruiter / Viewer | Portfolio consumer | App opens, detects emotion via webcam |
| General Public | Casual demo user | Intuitively usable without instructions |

# **5\. Tech Stack**

| Layer | Tool / Library | Purpose |
| ----- | ----- | ----- |
| Model Training | TensorFlow / Keras | CNN architecture & training |
| Face Detection | OpenCV (Haarcascade) | Locate faces in each video frame |
| UI / Web App | Streamlit | Real-time webcam UI \+ charts |
| Deployment | AntiGravity | 1-command public web deployment |
| Dataset | FER-2013 (Kaggle) | 35,887 labelled 48x48 grayscale images |
| Data Processing | NumPy, Pandas | Preprocessing, normalization |
| Visualization | Matplotlib, Plotly | Confusion matrix, accuracy plots, live chart |
| Version Control | GitHub (DevWithOm) | Source code, model weights, README |

# **6\. Functional Requirements**

## **FR-01: Model Training Pipeline**

* Load FER-2013 CSV, split into train/validation/test sets (80/10/10)

* Resize all images to 48x48, normalize pixel values to \[0, 1\]

* Apply data augmentation: horizontal flip, rotation, zoom

* Train CNN for 30–50 epochs, save best weights as emotion\_model.h5

* Log accuracy and loss curves; generate confusion matrix

## **FR-02: Real-Time Face Detection**

* Capture live webcam feed using OpenCV

* Detect faces per frame using OpenCV Haarcascade (frontalface\_default)

* Crop, convert to grayscale, and resize face ROI to 48x48 before inference

* Draw bounding box around each detected face in the video overlay

## **FR-03: Emotion Inference**

* Pass each detected face ROI through the trained CNN model

* Output top emotion label from 7 classes: Happy, Sad, Angry, Disgust, Fear, Surprise, Neutral

* Display confidence score as a percentage alongside the label

* Overlay label \+ confidence on the webcam frame in real time

## **FR-04: Streamlit Web UI**

* Show live webcam feed with bounding box \+ emotion label overlay

* Display current detected emotion and confidence score in a sidebar panel

* Render a live bar chart of emotion frequencies over the last 60 seconds

* Provide a Start / Stop button to control webcam capture

* Show model metadata: architecture, dataset, accuracy

## **FR-05: AntiGravity Deployment**

* Package app with requirements.txt and a one-command launch entry point

* Deploy to AntiGravity using: python \-m antigravity (or configured launch script)

* App must be accessible from a public URL with no local installation

* Webcam stream handled via streamlit-webrtc or equivalent browser-compatible method

# **7\. Non-Functional Requirements**

| ID | Category | Requirement |
| ----- | ----- | ----- |
| NFR-01 | Performance | Inference per frame ≤ 100ms on CPU (no GPU required) |
| NFR-02 | Accuracy | Minimum 60% validation accuracy on FER-2013 test set |
| NFR-03 | Usability | App usable by a non-technical user within 30 seconds of opening the URL |
| NFR-04 | Availability | Deployed app available 24/7 via AntiGravity URL |
| NFR-05 | Reproducibility | Full training code and model weights committed to GitHub (DevWithOm) |
| NFR-06 | Portability | requirements.txt must work on Python 3.9+ without conda |

# **8\. CNN Architecture Specification**

The model follows a standard deep CNN pattern optimised for small 48x48 grayscale inputs:

| \# | Layer Type | Filters / Units | Activation | Notes |
| ----- | ----- | ----- | ----- | ----- |
| 1 | Conv2D | 32, (3x3) | ReLU | Input: (48,48,1) |
| 2 | BatchNorm | — | — | Stabilise training |
| 3 | MaxPooling2D | (2x2) | — | Downsample |
| 4 | Conv2D | 64, (3x3) | ReLU | Deeper features |
| 5 | BatchNorm \+ Pool | (2x2) | — |  |
| 6 | Conv2D | 128, (3x3) | ReLU | High-level features |
| 7 | BatchNorm \+ Pool | (2x2) | — |  |
| 8 | Dropout | 0.25 | — | Prevent overfitting |
| 9 | Flatten | — | — |  |
| 10 | Dense | 256 | ReLU |  |
| 11 | Dropout | 0.5 | — |  |
| 12 | Dense (Output) | 7 | Softmax | 7 emotion classes |

Training config: Loss \= categorical\_crossentropy · Optimizer \= Adam (lr=0.001) · Epochs \= 30–50 · Batch size \= 64

# **9\. AntiGravity Deployment Plan**

## **9.1 What is AntiGravity?**

AntiGravity is a Python library / hosting service that allows developers to deploy Python web applications with a single command — no server configuration, no Docker, no cloud console. It is ideal for student portfolio projects that need a permanent public URL.

## **9.2 Deployment Steps**

7. Install AntiGravity: pip install antigravity

8. Build the Streamlit app (app.py) with streamlit-webrtc for browser-compatible webcam access

9. Create requirements.txt with all dependencies pinned

10. Create a Procfile or entry point compatible with AntiGravity’s runner

11. Run: python \-m antigravity to push to the AntiGravity server

12. Copy the generated public URL — share on LinkedIn & GitHub README

## **9.3 Key Dependency: streamlit-webrtc**

Standard OpenCV webcam capture (cv2.VideoCapture) does NOT work in deployed web apps because servers lack physical cameras. The correct approach for a deployed Streamlit app is:

* Use streamlit-webrtc to capture webcam frames from the user’s browser

* Pass browser frames to the OpenCV \+ model inference pipeline on the server

* Return annotated frames with bounding box and emotion overlay to the browser

# **10\. Project File Structure**

**emotion-detection/**

emotion-detection/

├── data/                  \# FER-2013 CSV files

├── model/

│   ├── train\_model.py     \# CNN training script

│   └── emotion\_model.h5   \# Saved model weights

├── app.py                 \# Streamlit web application

├── utils/

│   ├── face\_detector.py   \# OpenCV face detection

│   └── emotion\_labels.py  \# Class name mapping

├── notebooks/

│   └── EDA\_and\_training.ipynb

├── requirements.txt

├── Procfile               \# AntiGravity entry point

└── README.md              \# Setup, demo GIF, LinkedIn link

# **11\. Milestones & Timeline**

| Phase | Milestone | Deliverable | Duration |
| ----- | ----- | ----- | ----- |
| 1 | Data & Setup | FER-2013 downloaded, EDA notebook done | Day 1 |
| 2 | Model Training | CNN trained, emotion\_model.h5 saved, accuracy ≥60% | Day 2–4 |
| 3 | Real-Time Detection | OpenCV \+ model inference working on local webcam | Day 5 |
| 4 | Streamlit UI | Full app.py with webcam feed, overlay, sidebar chart | Day 6–7 |
| 5 | AntiGravity Deploy | Public URL live, webrtc-based webcam working in browser | Day 8 |
| 6 | Documentation | README, GitHub push, LinkedIn post, viva prep notes | Day 9–10 |

# **12\. Risks & Mitigations**

| Risk | Severity | Mitigation |
| ----- | ----- | ----- |
| cv2.VideoCapture fails in deployed app | High | Use streamlit-webrtc for browser webcam capture |
| Model accuracy below 60% | Medium | Apply data augmentation, use pre-trained weights (transfer learning) as fallback |
| AntiGravity server timeout on heavy model | Medium | Optimise model size; use model quantization if needed |
| Kaggle FER-2013 class imbalance | Low | Use class weights in model.fit() to handle imbalanced labels |
| Virtual environment dependency conflicts | Low | Pin all versions in requirements.txt; test on clean venv before deploy |

# **13\. Success Metrics**

* Model achieves ≥60% validation accuracy on FER-2013 test split

* Real-time inference runs at ≥15 FPS on a standard laptop CPU

* App deployed at a public AntiGravity URL accessible from any browser

* Emotion label \+ confidence score correctly overlaid on webcam feed

* GitHub repository is public with README, demo GIF, and requirements.txt

* Project explained clearly in a 5-minute viva without reading from notes

# **14\. Future Scope (v2)**

* Voice Emotion Detection — classify emotions from speech using librosa \+ LSTM (multi-modal AI)

* Multi-Face Tracking — detect and classify multiple faces simultaneously per frame

* Emotion-Based Music Recommendation — Spotify API integration to suggest songs by mood

* Mental Health Tracker — log emotion trends over a session; generate a wellbeing report

* Mobile Deployment — convert model to TFLite; embed in a Flutter app

# **Document Sign-Off**

| Prepared By | Om Vyas — github.com/DevWithOm · linkedin.com/in/om-vyas-546b9a3a5 |
| :---- | :---- |
| **Institution** | MIT Academy of Engineering, Pune — B.Tech CSE (AI & ML) |
| **PRN** | 202501110101 |
| **Status** | Ready for Implementation |