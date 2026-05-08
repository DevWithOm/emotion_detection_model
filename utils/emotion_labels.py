"""
Emotion Labels Module
Maps FER-2013 class indices to human-readable emotion labels.

When using torchvision.datasets.ImageFolder, classes are sorted alphabetically:
    0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy,
    4 = Neutral, 5 = Sad, 6 = Surprise
"""

# Ordered list matching ImageFolder alphabetical class indices
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

# Emoji mapping for UI display
EMOTION_EMOJIS = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐",
}

# Color mapping for bounding box / chart (BGR for OpenCV, hex for Plotly)
EMOTION_COLORS_BGR = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 128, 0),
    "Fear": (128, 0, 128),
    "Happy": (0, 255, 255),
    "Sad": (255, 0, 0),
    "Surprise": (0, 165, 255),
    "Neutral": (200, 200, 200),
}

EMOTION_COLORS_HEX = {
    "Angry": "#FF4444",
    "Disgust": "#44AA44",
    "Fear": "#AA44AA",
    "Happy": "#FFD700",
    "Sad": "#4488FF",
    "Surprise": "#FF8C00",
    "Neutral": "#AAAAAA",
}

NUM_CLASSES = len(EMOTION_LABELS)


def get_emotion_label(index: int) -> str:
    """Return the emotion label for a given class index."""
    if 0 <= index < NUM_CLASSES:
        return EMOTION_LABELS[index]
    return "Unknown"


def get_emotion_emoji(label: str) -> str:
    """Return the emoji for a given emotion label."""
    return EMOTION_EMOJIS.get(label, "❓")


def get_emotion_color_bgr(label: str) -> tuple:
    """Return the BGR color tuple for OpenCV drawing."""
    return EMOTION_COLORS_BGR.get(label, (255, 255, 255))


def get_emotion_color_hex(label: str) -> str:
    """Return the hex color string for Plotly/web charts."""
    return EMOTION_COLORS_HEX.get(label, "#FFFFFF")
    