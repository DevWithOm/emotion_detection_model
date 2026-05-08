"""
Face Detector Module
Uses OpenCV Haarcascade classifier to detect faces in video frames.
Provides utilities to crop, resize, and preprocess face ROIs for
the emotion classification CNN (48x48 grayscale input).
"""

import cv2
import numpy as np
import os

# Path to the Haarcascade XML file (bundled with OpenCV)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load the cascade classifier once at module level
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# CNN input dimensions
FACE_SIZE = (48, 48)


def detect_faces(frame: np.ndarray, scale_factor: float = 1.3,
                 min_neighbors: int = 5, min_size: tuple = (30, 30)) -> list:
    """
    Detect faces in a BGR frame using Haarcascade.

    Args:
        frame: BGR image (numpy array from OpenCV).
        scale_factor: How much the image size is reduced at each scale.
        min_neighbors: Minimum number of neighbors for a detection to be kept.
        min_size: Minimum face size in pixels (width, height).

    Returns:
        List of (x, y, w, h) bounding boxes for detected faces.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Downscale for much faster detection if image is large
    height, width = gray.shape
    max_width = 320
    
    if width > max_width:
        scale = max_width / width
        small_gray = cv2.resize(gray, (max_width, int(height * scale)))
        small_min_size = (int(min_size[0] * scale), int(min_size[1] * scale))
        
        faces = face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=small_min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        
        # Scale back the bounding boxes to original image coordinates
        scaled_faces = []
        for (x, y, w, h) in faces:
            scaled_faces.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale)))
        return scaled_faces
    else:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        # Convert to a regular Python list
        return list(faces) if len(faces) > 0 else []


def preprocess_face(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Crop a face ROI from the frame, convert to grayscale, resize to 48x48,
    and normalize pixel values to [0, 1] for CNN inference.

    Args:
        frame: BGR image.
        x, y, w, h: Bounding box coordinates of the detected face.

    Returns:
        Preprocessed face as a numpy array of shape (1, 48, 48, 1).
    """
    # Crop the face region
    face_roi = frame[y:y + h, x:x + w]

    # Convert to grayscale
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Resize to model input size
    resized_face = cv2.resize(gray_face, FACE_SIZE, interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    normalized_face = resized_face.astype("float32") / 255.0

    # Reshape to (1, 48, 48, 1) — batch dimension + channel dimension
    return normalized_face.reshape(1, 48, 48, 1)


def draw_bounding_box(frame: np.ndarray, x: int, y: int, w: int, h: int,
                      label: str, confidence: float, color: tuple = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
    """
    Draw a bounding box and emotion label on the frame.

    Args:
        frame: BGR image to annotate.
        x, y, w, h: Bounding box coordinates.
        label: Emotion label string.
        confidence: Confidence score (0-1).
        color: BGR color tuple for the box and text.
        thickness: Line thickness.

    Returns:
        Annotated frame.
    """
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Prepare label text
    text = f"{label} ({confidence * 100:.1f}%)"

    # Get text size for background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, 1)

    # Draw filled rectangle behind text
    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 5, y), color, -1)

    # Draw text
    cv2.putText(frame, text, (x + 2, y - 5), font, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA)

    return frame
