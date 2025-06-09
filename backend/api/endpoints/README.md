# SmartLog API Endpoints

This folder houses the FastAPI route definitions that power SmartLog’s face-detection, recognition, tracking, and logging functionality. Everything here is mounted under the `/api/face` router prefix.

---

## 📦 Features

- **Face Detection** (`POST /detect`)  
  - Accepts an image frame  
  - Runs YOLOv8 to locate faces  
  - Returns bounding boxes with confidence scores

- **Face Recognition** (`POST /recognize`)  
  - Accepts one or more cropped face images  
  - Computes ArcFace embeddings (via DeepFace)  
  - Matches against a “known” gallery (cosine similarity or Faiss)  
  - Labels each face as known (with `user_id`) or unknown (with generated `uuid`)

- **Multi‐Object Tracking** (`POST /track`)  
  - Ingests frame ID and detected bounding boxes  
  - Uses DeepSORT (or ByteTrack) to maintain persistent track IDs across frames

- **Entry/Exit Logging** (`POST /log`)  
  - Receives a track ID + centroid coordinates  
  - Applies “left” and “right” line‐cross logic to determine **IN** vs **OUT**  
  - Persists an `action` event in the `SmartLog` database table

---

## ⚙️ Prerequisites

Before working here, ensure the main backend environment is ready:

1. **Python** ≥ 3.8  
2. **Virtual environment** (venv, conda, etc.)  
3. **Dependencies installed** (`pip install -r requirements.txt`)  
4. **.env configured** with database URL, model paths, thresholds, and line positions  
5. **Database migrated/initialized** (using Alembic or SQLAlchemy metadata)  
6. **YOLOv8** weights available (path set via `.env`)  
7. **ArcFace** model downloaded (via DeepFace configuration)

---

## 🚀 Setup Instructions

1. **Navigate to endpoints folder**  
   ```bash
   cd SmartLog/backend/api/endpoints

2. **Ensure environment is activated**

   ```bash
   # from repository root
   source .venv/bin/activate
   ```

3. **Install (or verify) dependencies**

   ```bash
   pip install --upgrade pip
   pip install fastapi pydantic sqlalchemy asyncpg ultralytics deepface opencv-python scipy
   ```

4. **Confirm `.env` values**
   Your root `.env` should include entries like:

   ```
   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/smartlog
   YOLO_MODEL_PATH=/absolute/path/to/yolov8.pt
   UNKNOWN_SIM_THRESHOLD=0.65
   LEFT_LINE_X=0.3
   RIGHT_LINE_X=0.7
   ```

5. **Run a quick smoke test**
   With the backend (root) running via:

   ```bash
   uvicorn main:app --reload
   ```

   ping the detect endpoint:

   ```bash
   curl -F "frame=@/path/to/test.jpg" http://127.0.0.1:8000/api/face/detect
   ```

---

## 🔍 Code Overview

```
backend/api/endpoints/
├── __init__.py       # Registers the router with prefix="/api/face"
└── face.py           # All face-related POST routes
```

### `__init__.py`

* Imports the FastAPI `APIRouter`
* Includes:

  ```python
  from fastapi import APIRouter
  from .face import router as face_router

  router = APIRouter(prefix="/api/face", tags=["face"])
  router.include_router(face_router)
  ```

### `face.py`

1. **Imports & Dependencies**

   ```python
   from fastapi import APIRouter, UploadFile, File, Depends
   from sqlalchemy.ext.asyncio import AsyncSession
   from core.database import get_db
   from core.face_recognition import detect_faces, compute_embeddings, match_embedding
   from core.tracking import TrackingContext
   from models.models import Smartlog
   ```

2. **Router Declaration**

   ```python
   router = APIRouter()
   ```

3. **POST `/detect`**

   * **Signature**: `async def detect(frame: UploadFile = File(...))`
   * **Flow**:

     1. Read image bytes
     2. Convert to array (OpenCV)
     3. `detect_faces()` → list of boxes
     4. Return JSON list of `{ x1, y1, x2, y2, confidence }`

4. **POST `/recognize`**

   * **Signature**: `async def recognize(crops: List[UploadFile] = File(...), db: AsyncSession = Depends(get_db))`
   * **Flow**:

     1. For each crop: read & convert to array
     2. `compute_embeddings()` → list of embeddings
     3. `match_embedding()` vs known gallery
     4. Persist new unknown crops (if any) and reload gallery
     5. Return list of `{ id, is_known, similarity }`

5. **POST `/track`**

   * **Signature**: `async def track(frame_id: str, bboxes: List[BoundingBox], db: AsyncSession = Depends(get_db))`
   * **Flow**:

     1. Pass `bboxes` to a shared `TrackingContext` instance
     2. Update tracks via DeepSORT
     3. Return list of `{ track_id, bbox }`

6. **POST `/log`**

   * **Signature**: `async def log_event(track_id: int, centroid: Centroid, db: AsyncSession = Depends(get_db))`
   * **Flow**:

     1. Retrieve previous centroid for `track_id`
     2. Compare to `LEFT_LINE_X` & `RIGHT_LINE_X` from config
     3. Determine `action = "I"` or `"O"`
     4. Insert new `Smartlog` row with timestamp
     5. Return `{ entry_id, action, timestamp }`

---

*Now you have everything you need to understand, develop, and extend the face-API endpoints in SmartLog’s backend!*
