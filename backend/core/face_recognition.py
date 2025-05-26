import os
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from core.config import settings
from scipy.spatial.distance import cosine
import time
import uuid

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load YOLOv8 model
yolo_model = YOLO("yolov8n-face.pt")
yolo_model.to(DEVICE)

# Directory setup
CROPPED_FACE_DIR = "./face"
os.makedirs(CROPPED_FACE_DIR, exist_ok=True)

thresh = settings.FACE_DET_THRESH

# In-memory unknown face gallery
unknown_gallery = {}
UNKNOWN_SIMILARITY_THRESHOLD = settings.UNKNOWN_SIMILARITY_THRESHOLD

# Load frontal face Haar cascade
frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def is_frontal_face(face_img: np.ndarray) -> bool:
    """Check if the given face image is likely a frontal face."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=1)
    return len(faces) > 0

def load_known_embeddings(path=settings.EMBEDDING_PATH):
    """
    Load known face embeddings from a .npz file,
    where each key is a person name and each value is a numpy array of embeddings.
    """
    if not os.path.exists(path):
        print(f"Embedding file {path} not found.")
        return {}

    data = np.load(path, allow_pickle=True)
    known_faces = {key: data[key] for key in data.files}
    return known_faces


def extract_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Extract a specified rectangular region from the frame.
    Coordinates are clamped to frame dimensions.
    """
    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros(
            (1, 1, 3), dtype=np.uint8
        )  # Return empty black image on invalid crop

    return frame[y1:y2, x1:x2]


def detect_faces_yolov8(frame: np.ndarray, x_offset: int = 0, y_offset: int = 0):
    """
    Detect faces using YOLOv8 on input frame or ROI.
    Returns list of dicts with bbox (offset to original frame) and cropped face.
    """
    results = yolo_model.predict(
        source=frame, imgsz=320, conf=0.55, verbose=False, device=DEVICE
    )[0]
    face_entries = []

    h, w = frame.shape[:2]

    for det in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, det)

        # Clamp within ROI
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # Filter by frontal face
        if not is_frontal_face(face_crop):
            continue

        # Offset bbox to match original full frame
        face_entries.append(
            {
                "bbox": (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset),
                "face_crop": face_crop,
            }
        )

    return face_entries


def get_facenet_embedding(face_img: np.ndarray):
    """
    Get embedding from face image NumPy array using DeepFace Facenet512 model.
    If direct numpy array input fails, fallback to saving temp file.
    """
    try:
        embedding = DeepFace.represent(
            img_path=face_img,
            model_name=settings.FACE_MODEL_NAME,
            enforce_detection=False,
        )[0]["embedding"]

    except Exception as e:
        try:
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face_img)
            embedding = DeepFace.represent(
                img_path=temp_path,
                model_name=settings.FACE_MODEL_NAME,
                enforce_detection=False,
            )[0]["embedding"]
            os.remove(temp_path)
        except Exception as e2:
            print(f"Embedding error (fallback): {e2}")
            return None

    embedding = np.array(embedding)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm


def recognize_face_from_embedding(test_embedding, known_faces, threshold=thresh):
    """
    Compare test embedding against known_faces dictionary.
    Returns best matched name or None.
    """
    best_match = None
    best_similarity = threshold

    for name, embeddings_list in known_faces.items():
        for embedding in embeddings_list:
            embedding = np.asarray(embedding)
            if embedding.ndim != 1 or test_embedding.ndim != 1:
                continue
            similarity = 1 - cosine(embedding, test_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

    return best_match


def assign_unknown_id(test_embedding):
    """
    Assign a persistent unknown ID if the face is not already recognized.
    """
    for uid, entry in unknown_gallery.items():
        sim = 1 - cosine(test_embedding, entry["embedding"])
        if sim >= UNKNOWN_SIMILARITY_THRESHOLD:
            unknown_gallery[uid]["last_seen"] = time.time()
            return uid

    # New unknown face
    new_id = f"unknown_{str(uuid.uuid4())[:8]}"
    unknown_gallery[new_id] = {
        "embedding": test_embedding,
        "last_seen": time.time()
    }
    return new_id


