import os
import cv2
import numpy as np
from datetime import datetime

# Base directory to save cropped faces
BASE_DIR = "./face"


def save_unique_cropped_face(frame: np.ndarray, bbox: tuple, name: str):
    """
    Crop the face from the frame and save it only once per recognized person.
    Creates a date-wise subfolder inside the 'face' directory.
    """
    # Create a subfolder named by current date
    date_str = datetime.now().strftime("%Y-%m-%d")
    date_folder = os.path.join(BASE_DIR, date_str)
    os.makedirs(date_folder, exist_ok=True)

    # Define the output file path
    file_path = os.path.join(date_folder, f"{name}.jpg")

    # If face already saved, skip
    if os.path.exists(file_path):
        return

    # Crop face using bounding box (x1, y1, x2, y2)
    x1, y1, x2, y2 = bbox
    face_crop = frame[y1:y2, x1:x2]

    # Save face crop
    cv2.imwrite(file_path, face_crop)
