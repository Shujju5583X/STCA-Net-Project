"""
Shared Face Detection Utilities

Consolidated face detection logic used by both the prediction and video
processing pipelines. Uses Haar cascades for CPU-efficient detection.
"""
import cv2
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)

# Haar cascade path (ships with OpenCV)
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Module-level singleton for the cascade classifier
_face_cascade = None


def get_face_cascade():
    """Lazily load and cache the Haar cascade classifier."""
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if _face_cascade.empty():
            logger.error("Failed to load Haar cascade classifier.")
            raise FileNotFoundError("Haar cascade XML file not found.")
    return _face_cascade


def extract_face(image_input, cascade=None):
    """
    Detect and crop the largest face from an image.

    Args:
        image_input: Either a PIL Image or a numpy array (RGB).
        cascade: Optional pre-loaded CascadeClassifier. If None, uses the
                 module-level cached instance.

    Returns:
        tuple: (cropped_face_pil, face_found_bool)
               If no face is found, returns a center crop.
    """
    if cascade is None:
        cascade = get_face_cascade()

    # Convert to numpy array if PIL Image
    if isinstance(image_input, Image.Image):
        img_array = np.array(image_input)
    else:
        img_array = image_input

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        # No face found — center crop as fallback
        h, w = img_array.shape[:2]
        size = min(h, w)
        y1, x1 = (h - size) // 2, (w - size) // 2
        cropped = img_array[y1:y1 + size, x1:x1 + size]
        return Image.fromarray(cropped), False

    # Find the largest face by area
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    # Add 20% margin around the face for context
    margin_w, margin_h = int(w * 0.2), int(h * 0.2)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_array.shape[1], x + w + margin_w)
    y2 = min(img_array.shape[0], y + h + margin_h)

    cropped = img_array[y1:y2, x1:x2]
    return Image.fromarray(cropped), True
