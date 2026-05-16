# Real-Time Emotion Detection System - Project Overview & Presentation Script

## Part 1: Project Overview

### Introduction
The Real-Time Emotion Detection System is a machine learning web application that classifies human facial expressions into 7 distinct emotion categories (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral) using live webcam feed. 

### Why this project?
Existing solutions for emotion recognition are often expensive, cloud-locked, or require proprietary hardware. This project aims to provide an open-source, lightweight, and real-time solution that runs directly in the browser using standard webcams without any local setup.

### How it Works
1. **Webcam Capture**: Uses `streamlit-webrtc` to access the user's webcam directly from the browser.
2. **Face Detection**: Employs OpenCV's Haarcascade classifier to locate faces in real-time video frames.
3. **Preprocessing**: Crops the detected face, converts it to grayscale, and resizes it to 48x48 pixels to match the model's expected input.
4. **Emotion Inference**: Passes the preprocessed image through a trained Convolutional Neural Network (CNN) built with PyTorch.
5. **Visualization**: Streamlit renders the live feed with bounding boxes, emotion labels, and confidence scores, alongside a live emotion frequency chart tracking the last 60 seconds.

### Tech Stack
- **Model Training**: PyTorch (CNN architecture & training)
- **Face Detection**: OpenCV (Haarcascade `frontalface_default`)
- **Web App UI**: Streamlit
- **Webcam Interface**: `streamlit-webrtc`
- **Dataset**: FER-2013 (Kaggle) - 35,887 labelled 48x48 grayscale images

---

## Part 2: Presentation Script

*This script is designed for a 5-7 minute college presentation or viva. Adjust the pacing as needed.*

### **Slide 1: Title Slide**
**Visuals**: Project Title, Your Name , PRN, College details.
**Script**:
> "Good morning everyone. My name is Om Vyas, and today I'll be presenting my project: a Real-Time Emotion Detection System. This project was developed as part of my B.Tech curriculum in AI & ML. At its core, it's a web-based application that uses a Convolutional Neural Network to detect and classify human emotions through a live webcam feed."

### **Slide 2: Problem Statement**
**Visuals**: Bullet points highlighting existing issues (Expensive, Cloud-locked, Proprietary hardware).
**Script**:
> "Let's start with the problem. Emotion recognition is crucial for human-computer interaction, mental health monitoring, and adaptive systems. However, most existing solutions are expensive, require heavy local processing power, or need proprietary hardware. There is a clear need for a lightweight, accessible solution that can run in real-time on standard consumer webcams without requiring the user to install any software."

### **Slide 3: The Solution & Features**
**Visuals**: High-level features (Real-time detection, Browser-based, Live analytics).
**Script**:
> "To solve this, I built a browser-based application using Streamlit. It captures live video, detects faces using OpenCV, and classifies expressions into one of seven emotions: Happy, Sad, Angry, Surprise, Fear, Disgust, or Neutral. It's completely browser-based using WebRTC, meaning it runs smoothly without any local camera drivers, and it provides real-time confidence scores and analytics."

### **Slide 4: Tech Stack & Architecture**
**Visuals**: Diagram of the tech stack (PyTorch -> OpenCV -> Streamlit).
**Script**:
> "Here is the technology stack I used. For model training and architecture, I chose PyTorch. For the computer vision aspect—specifically face detection—I used OpenCV. The entire user interface and web framework is powered by Streamlit, and `streamlit-webrtc` handles the webcam capture. The system processes the video frame by frame, runs inference on the detected face, and updates the UI in under 100 milliseconds per frame."

### **Slide 5: The CNN Model & Training**
**Visuals**: Model architecture (Conv2D -> MaxPooling -> Dense), Dataset mention (FER-2013).
**Script**:
> "Diving into the machine learning aspect, the brain of this system is a Deep Convolutional Neural Network. I trained this model on the FER-2013 dataset from Kaggle, which contains over 35,000 grayscale face images. The architecture consists of multiple 2D Convolutional layers with Batch Normalization and MaxPooling, followed by dense layers. To prevent overfitting, I used Dropout layers and data augmentation techniques like random rotations and flips. The model successfully achieved around 65% validation accuracy, which is excellent for this specific, challenging dataset."

### **Slide 6: Live Demonstration / Application UI**
**Visuals**: Screenshot of the Streamlit App or a live demo (Webcam feed, bounding box, live bar chart).
**Script**:
> "Here you can see the application in action. The OpenCV Haarcascade rapidly detects the face and draws a bounding box. The PyTorch model then predicts the emotion and overlays the label and confidence percentage. On the side, we have a live analytics dashboard tracking the frequency of detected emotions over the last 60 seconds, providing a continuous emotional profile of the user."

### **Slide 7: Conclusion & Future Scope**
**Visuals**: Summary points, Future scope (Voice detection, Mobile app).
**Script**:
> "To conclude, this project successfully demonstrates a highly optimized, real-time computer vision pipeline running directly in a web environment. It proves that complex AI models can be made highly accessible. For future enhancements, I plan to integrate voice-based emotion detection for a multi-modal approach, and potentially convert the model to TensorFlow Lite to deploy it as a native mobile application using Flutter. Thank you. I'm now open to any questions."  
