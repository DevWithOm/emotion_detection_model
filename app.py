"""
🧠 Real-Time Emotion Detection System
======================================
Streamlit web application that detects facial emotions in real-time
using a CNN model trained on FER-2013 dataset.

Features:
    - Live webcam feed via streamlit-webrtc (browser-compatible)
    - Real-time face detection with OpenCV Haarcascade
    - Emotion classification into 7 categories (PyTorch CNN)
    - Confidence score overlay on video feed
    - Live emotion frequency bar chart (last 60 seconds)
    - Model metadata and architecture display

Author: Om Vyas — B.Tech CSE (AI & ML), MIT Academy of Engineering, Pune
"""

import os
import time
import threading
from collections import deque

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import project modules
from utils.emotion_labels import (
    EMOTION_LABELS, NUM_CLASSES,
    get_emotion_label, get_emotion_emoji,
    get_emotion_color_bgr, get_emotion_color_hex,
    EMOTION_COLORS_HEX
)
from utils.face_detector import detect_faces, preprocess_face, draw_bounding_box


# ─── CNN Model Definition (must match train_model.py) ───────────────────
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(EmotionCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout1 = nn.Dropout(0.25)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout1(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


# ─── Configuration ───────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "emotion_model.pth")
HISTORY_DURATION = 60
MAX_HISTORY_SIZE = 900
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Page Configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Real-Time Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Custom CSS for Premium UI ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .hero-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7, #ff6b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }

    .hero-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1rem;
        font-weight: 300;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
    }

    .metric-label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }

    .metric-value.happy { color: #FFD700; }
    .metric-value.sad { color: #4488FF; }
    .metric-value.angry { color: #FF4444; }
    .metric-value.surprise { color: #FF8C00; }
    .metric-value.fear { color: #AA44AA; }
    .metric-value.disgust { color: #44AA44; }
    .metric-value.neutral { color: #AAAAAA; }

    /* Confidence Bar */
    .confidence-bar-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        height: 8px;
        margin-top: 0.5rem;
        overflow: hidden;
    }

    .confidence-bar {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        transition: width 0.3s ease;
    }

    /* Info Panel */
    .info-panel {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }

    .info-panel h3 {
        color: #58a6ff;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .info-panel p, .info-panel li {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
    }

    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-active {
        background: rgba(0, 255, 128, 0.15);
        color: #00ff80;
        border: 1px solid rgba(0, 255, 128, 0.3);
    }

    .status-inactive {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Architecture Table */
    .arch-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.5rem 0;
    }

    .arch-table th {
        background: rgba(88, 166, 255, 0.1);
        color: #58a6ff;
        padding: 0.5rem;
        font-size: 0.75rem;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }

    .arch-table td {
        padding: 0.4rem 0.5rem;
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.6);
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: rgba(255, 255, 255, 0.7);
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_emotion_model():
    """Load the trained PyTorch CNN model. Cached so it's only loaded once."""
    if os.path.exists(MODEL_PATH):
        try:
            model = EmotionCNN(num_classes=NUM_CLASSES)
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            return model, True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, False
    else:
        return None, False


# ─── WebRTC Video Processor ─────────────────────────────────────────────
class EmotionVideoProcessor(VideoProcessorBase):
    """
    Processes each video frame:
    1. Detect faces using Haarcascade
    2. Run PyTorch emotion inference on each face
    3. Draw bounding box + label overlay
    """

    def __init__(self):
        global model
        self.model = model
        
        # Warm-up inference to prevent lag on the first video frame
        if self.model is not None:
            dummy_input = torch.zeros(1, 1, 48, 48).to(DEVICE)
            with torch.no_grad():
                self.model(dummy_input)
                
        # Warm-up OpenCV face detection
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        detect_faces(dummy_img)
        
        # Thread-safe state for the main Streamlit thread to read
        self.lock = threading.Lock()
        self.history = deque(maxlen=MAX_HISTORY_SIZE)
        self.current_emotion = "None"
        self.current_confidence = 0.0
        self.current_emoji = "❓"
        self.faces_detected = 0
        self.fps = 0
        self._frame_times = deque(maxlen=30)
        self.frame_counter = 0
        self.process_every = 3
        self.last_results = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        try:
            if self.model is None:
                cv2.putText(
                    img, "Model not loaded - Train the model first",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA
                )
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            now = time.time()
            if not hasattr(self, 'last_process_time'):
                self.last_process_time = 0

            self.frame_counter += 1
            # Process at most ~3 times per second (every 0.3s) to reduce lag
            if now - self.last_process_time >= 0.3 or not hasattr(self, 'last_results'):
                self.last_process_time = now
                # Make face detection slightly more lenient for backlit/poor lighting
                faces = detect_faces(img, scale_factor=1.3, min_neighbors=5)
                num_faces = len(faces)
                top_emotion = "None"
                top_confidence = 0.0
                new_results = []

                for (x, y, w, h) in faces:
                    # Preprocess: (1, 48, 48, 1) numpy -> (1, 1, 48, 48) torch tensor
                    face_input = preprocess_face(img, x, y, w, h)
                    # Convert from (1, 48, 48, 1) to (1, 1, 48, 48) for PyTorch
                    face_tensor = torch.tensor(
                        face_input.transpose(0, 3, 1, 2),
                        dtype=torch.float32
                    ).to(DEVICE)

                    with torch.no_grad():
                        outputs = self.model(face_tensor)
                        probabilities = F.softmax(outputs, dim=1)[0]
                        confidence, emotion_idx = torch.max(probabilities, 0)
                        confidence = confidence.item()
                        emotion_idx = emotion_idx.item()

                    emotion_label = get_emotion_label(emotion_idx)

                    if confidence > top_confidence:
                        top_emotion = emotion_label
                        top_confidence = confidence

                    color = get_emotion_color_bgr(emotion_label)
                    new_results.append((x, y, w, h, emotion_label, confidence, color))
                
                self.last_results = new_results

                # Update state safely
                with self.lock:
                    now = time.time()
                    self.current_emotion = top_emotion
                    self.current_confidence = top_confidence
                    self.current_emoji = get_emotion_emoji(top_emotion)
                    self.faces_detected = num_faces
                    if top_emotion != "None":
                        self.history.append((now, top_emotion, top_confidence))

            # Always draw the last results
            for (x, y, w, h, emotion_label, confidence, color) in getattr(self, 'last_results', []):
                draw_bounding_box(img, x, y, w, h, emotion_label, confidence, color)

            with self.lock:
                now = time.time()
                self._frame_times.append(now)
                if len(self._frame_times) > 1:
                    elapsed = self._frame_times[-1] - self._frame_times[0]
                    if elapsed > 0:
                        self.fps = len(self._frame_times) / elapsed

        except Exception as e:
            # Draw the error directly on the frame so we can see what crashed!
            cv2.putText(
                img, f"Error: {str(e)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1, cv2.LINE_AA
            )
            print(f"WebRTC Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")
        
    def get_snapshot(self):
        with self.lock:
            return {
                "emotion": self.current_emotion,
                "confidence": self.current_confidence,
                "emoji": self.current_emoji,
                "faces": self.faces_detected,
                "fps": self.fps
            }
            
    def get_recent_history(self, seconds: int = HISTORY_DURATION) -> dict:
        with self.lock:
            cutoff = time.time() - seconds
            counts = {label: 0 for label in EMOTION_LABELS}
            for timestamp, emotion, _ in self.history:
                if timestamp >= cutoff:
                    counts[emotion] = counts.get(emotion, 0) + 1
            return counts


# ─── Hero Header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🧠 Real-Time Emotion Detection</h1>
    <p>CNN-powered facial emotion recognition • 7 emotion classes • Live webcam analysis</p>
</div>
""", unsafe_allow_html=True)


# ─── Helper for State Access ─────────────────────────────────────────────
def get_current_snapshot(ctx):
    if ctx and ctx.video_processor:
        return ctx.video_processor.get_snapshot()
    return {
        "emotion": "None",
        "confidence": 0.0,
        "emoji": "❓",
        "faces": 0,
        "fps": 0.0
    }

def get_current_history(ctx):
    if ctx and ctx.video_processor:
        return ctx.video_processor.get_recent_history()
    return {label: 0 for label in EMOTION_LABELS}

# Load the model state globally for the page
model, model_loaded = load_emotion_model()

# ─── Main Content Area ──────────────────────────────────────────────────
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("### 📹 Live Webcam Feed")

    if model_loaded:
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionVideoProcessor,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                ]
            },
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 30}
                },
                "audio": False,
            },
            async_processing=True,
        )

        st.markdown("""
        <div class="info-panel" style="margin-top: 1rem;">
            <h3>📋 Instructions</h3>
            <p>1. Click <strong>"START"</strong> to enable your webcam</p>
            <p>2. Allow camera access when prompted by your browser</p>
            <p>3. Position your face in front of the camera</p>
            <p>4. Emotion detection starts automatically</p>
            <p>5. Click <strong>"STOP"</strong> to end the session</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        webrtc_ctx = None
        st.markdown("""
        <div class="info-panel" style="border-color: rgba(255, 68, 68, 0.3);">
            <h3 style="color: #FF4444;">⚠️ Model Not Found</h3>
            <p>The emotion detection model has not been trained yet.</p>
            <p>Please follow these steps:</p>
            <p>1. Download the <a href="https://www.kaggle.com/datasets/msambare/fer2013" style="color: #58a6ff;">FER-2013 dataset</a> from Kaggle</p>
            <p>2. Place <code>fer2013.csv</code> in the <code>data/</code> folder</p>
            <p>3. Run: <code>python model/train_model.py</code></p>
            <p>4. Restart this app: <code>streamlit run app.py</code></p>
        </div>
        """, unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")

    if model_loaded:
        st.markdown('<span class="status-badge status-active">● Model Loaded</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-inactive">○ Model Not Found</span>',
                    unsafe_allow_html=True)
        st.warning("⚠️ Train the model first:\n```\npython model/train_model.py\n```")

    st.markdown("---")

    st.markdown("### 🎭 Current Emotion")
    
    @st.fragment(run_every=2)
    def render_sidebar_metrics():
        snapshot = get_current_snapshot(webrtc_ctx)

        emotion_lower = snapshot["emotion"].lower()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Detected Emotion</div>
            <div class="metric-value {emotion_lower}">{snapshot["emoji"]} {snapshot["emotion"]}</div>
            <div class="confidence-bar-container">
                <div class="confidence-bar" style="width: {snapshot['confidence'] * 100}%"></div>
            </div>
            <div style="color: rgba(255,255,255,0.4); font-size: 0.75rem; margin-top: 0.3rem;">
                Confidence: {snapshot['confidence'] * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Faces Detected</div>
            <div class="metric-value">{snapshot["faces"]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Frame Rate</div>
            <div class="metric-value">{snapshot["fps"]:.1f} FPS</div>
        </div>
        """, unsafe_allow_html=True)

    render_sidebar_metrics()

    st.markdown("---")

    st.markdown("### 🏗️ Model Architecture")
    st.markdown("""
    <div class="info-panel">
        <h3>CNN Specification (PyTorch)</h3>
        <table class="arch-table">
            <tr><th>Layer</th><th>Config</th></tr>
            <tr><td>Conv2D</td><td>32 filters, 3×3, ReLU</td></tr>
            <tr><td>BatchNorm + Pool</td><td>2×2 MaxPool</td></tr>
            <tr><td>Conv2D</td><td>64 filters, 3×3, ReLU</td></tr>
            <tr><td>BatchNorm + Pool</td><td>2×2 MaxPool</td></tr>
            <tr><td>Conv2D</td><td>128 filters, 3×3, ReLU</td></tr>
            <tr><td>BatchNorm + Pool</td><td>2×2 MaxPool</td></tr>
            <tr><td>Dropout</td><td>25%</td></tr>
            <tr><td>Dense</td><td>256 units, ReLU</td></tr>
            <tr><td>Dropout</td><td>50%</td></tr>
            <tr><td>Output</td><td>7 classes, Softmax</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel">
        <h3>Training Details</h3>
        <p>📦 <strong>Dataset:</strong> FER-2013 (35,887 images)</p>
        <p>📐 <strong>Input:</strong> 48×48 grayscale</p>
        <p>⚙️ <strong>Optimizer:</strong> Adam (lr=0.001)</p>
        <p>📉 <strong>Loss:</strong> Cross-Entropy with class weights</p>
        <p>🔧 <strong>Framework:</strong> PyTorch</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.3); font-size: 0.7rem; padding: 1rem 0;">
        Built by Om Vyas<br>
        MIT Academy of Engineering, Pune<br>
        B.Tech CSE (AI & ML)
    </div>
    """, unsafe_allow_html=True)


with col2:
    st.markdown("### 📊 Emotion Distribution")
    st.caption("Live frequency chart — last 60 seconds")

    chart_placeholder = st.empty()

    @st.fragment(run_every=2)
    def render_emotion_chart():
        try:
            emotion_counts = get_current_history(webrtc_ctx)
        except NameError:
            emotion_counts = get_current_history(None)

        labels = list(emotion_counts.keys())
        values = list(emotion_counts.values())
        colors = [EMOTION_COLORS_HEX.get(label, "#FFFFFF") for label in labels]

        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255,255,255,0.1)', width=1),
                    opacity=0.85
                ),
                text=[f"{v}" for v in values],
                textposition='outside',
                textfont=dict(color='rgba(255,255,255,0.7)', size=12),
            )
        ])

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='rgba(255,255,255,0.6)'),
            xaxis=dict(
                tickangle=-45,
                showgrid=False,
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)',
                title=dict(text="Frequency", font=dict(size=11)),
            ),
            margin=dict(l=40, r=20, t=20, b=60),
            height=350,
            bargap=0.15,
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)

    render_emotion_chart()

    st.markdown("### 🎭 Emotion Classes")

    legend_rows = ""
    for label in EMOTION_LABELS:
        emoji = get_emotion_emoji(label)
        color = get_emotion_color_hex(label)
        legend_rows += f"""
        <tr>
            <td>{label}</td>
            <td style="font-size: 1.2rem;">{emoji}</td>
            <td><span style="display: inline-block; width: 14px; height: 14px;
                        background: {color}; border-radius: 3px; vertical-align: middle;"></span>
                {color}</td>
        </tr>"""

    st.markdown(f"""
    <div class="info-panel">
        <table class="arch-table">
            <tr><th>Emotion</th><th>Emoji</th><th>Color</th></tr>
            {legend_rows}
        </table>
    </div>
    """, unsafe_allow_html=True)


# ─── Bottom Section ─────────────────────────────────────────────────────
st.markdown("---")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div class="info-panel">
        <h3>🎯 About This Project</h3>
        <p>This Real-Time Emotion Detection System uses a Convolutional Neural Network
        trained on the FER-2013 dataset to classify facial expressions into 7 emotion
        categories. The system processes live webcam feeds using OpenCV for face detection
        and PyTorch for emotion inference.</p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="info-panel">
        <h3>🛠️ Technology Stack</h3>
        <p>• <strong>Model:</strong> PyTorch (CNN)</p>
        <p>• <strong>Face Detection:</strong> OpenCV Haarcascade</p>
        <p>• <strong>Web UI:</strong> Streamlit + WebRTC</p>
        <p>• <strong>Dataset:</strong> FER-2013 (Kaggle)</p>
        <p>• <strong>Visualization:</strong> Plotly</p>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="info-panel">
        <h3>👤 Developer</h3>
        <p><strong>Om Vyas</strong></p>
        <p>B.Tech CSE (AI & ML)</p>
        <p>MIT Academy of Engineering, Pune</p>
        <p>PRN: 202501110101</p>
        <p>
            <a href="https://github.com/DevWithOm" style="color: #58a6ff;">GitHub</a> •
            <a href="https://linkedin.com/in/om-vyas-546b9a3a5" style="color: #58a6ff;">LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Auto-Refresh for Live UI ──────────────────────────────────────────
# Removed global st.rerun() to prevent UI lagging. Using st.fragment instead.

