# app/api/endpoints/face.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Dict
import numpy as np
import cv2

from core.embeddings import save_embeddings
from core.face_recognition import load_known_embeddings
# from core.face_recognition import face_app

router = APIRouter()

# Load known faces
known_faces = load_known_embeddings()

face_data_log = []  # Temporary storage (optional to persist in DB)


@router.get("/get_face_data", response_model=List[Dict[str, str]])
async def get_face_data():
    return face_data_log


@router.post("/upload_face")
async def upload_face(name: str = Form(...), file: UploadFile = File(...)):
    try:
        image_data = np.frombuffer(file.file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        faces = face_app.get(image)

        if not faces:
            raise HTTPException(
                status_code=400, detail="No face detected in the image."
            )

        face_embedding = faces[0].embedding
        known_faces[name] = face_embedding
        save_embeddings(known_faces)

        return {"message": f"Face for {name} registered successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete_face")
async def delete_face(name: str = Form(...)):
    try:
        if name in known_faces:
            del known_faces[name]
            save_embeddings(known_faces)
            return {"message": f"Face for {name} deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail="Face not found.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
