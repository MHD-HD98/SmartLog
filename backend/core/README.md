# SmartLog Core Modules

The **`backend/core`** directory contains the core “microservice” modules that power SmartLog’s face-detection, recognition, tracking, cropping, and utility functions. These modules are consumed by the FastAPI endpoints to provide the application’s main business logic.

---

## 📦 Features

- **Centralized configuration** via Pydantic settings  
- **Async database engine & session management**  
- **YOLOv8-based face detection**  
- **ArcFace embeddings & matching**  
- **Multi-object tracking** (DeepSORT or ByteTrack)  
- **Unique-face cropping & gallery management**  
- **General-purpose utilities** for image conversion, UUIDs, and more  

---

## ⚙️ Prerequisites

1. **Python** ≥ 3.8  
2. **Virtual environment** activated (venv, conda, etc.)  
3. **Dependencies installed** (see root `requirements.txt`):  
   - `fastapi`, `uvicorn`  
   - `sqlalchemy`, `asyncpg`  
   - `pydantic`  
   - `ultralytics` (YOLOv8)  
   - `deepface` (ArcFace)  
   - `opencv-python`, `numpy`, `scipy`  
   - `torch` / `cuda` (optional GPU)  
4. **Environment variables** configured in root `.env`  
5. **YOLOv8 weights** downloaded and path set (`YOLO_MODEL_PATH`)  
6. (Optional) **Alembic** if using migrations  

---

## 🚀 Setup Instructions

1. **Activate your environment**  
   ```bash
   cd SmartLog/backend
   source .venv/bin/activate

2. **Install (or verify) core dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure `.env`** with at minimum:

   ```env
   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/smartlog
   YOLO_MODEL_PATH=/absolute/path/to/yolov8.pt
   UNKNOWN_SIM_THRESHOLD=0.65
   LEFT_LINE_X=0.3
   RIGHT_LINE_X=0.7
   ```

4. **Run database migrations** (if using Alembic)

   ```bash
   cd backend
   alembic upgrade head
   ```

---

## 🗂 Code Overview

### 1. `config.py`

* **Purpose**: Centralizes all runtime settings via Pydantic’s `BaseSettings`.
* **Key classes / variables**:

  ```python
  class Settings(BaseSettings):
      DATABASE_URL: str
      YOLO_MODEL_PATH: str
      UNKNOWN_SIM_THRESHOLD: float = 0.65
      LEFT_LINE_X: float = 0.3
      RIGHT_LINE_X: float = 0.7
      # … other hyperparameters …
  settings = Settings()
  ```
* **Usage**:

  ```python
  from core.config import settings
  db_url = settings.DATABASE_URL
  threshold = settings.UNKNOWN_SIM_THRESHOLD
  ```

---

### 2. `database.py`

* **Purpose**: Sets up the async SQLAlchemy engine and session dependency for FastAPI.
* **Key components**:

  ```python
  from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
  from sqlalchemy.orm import sessionmaker

  engine = create_async_engine(settings.DATABASE_URL, echo=True)
  AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

  async def get_db() -> AsyncGenerator[AsyncSession, None]:
      async with AsyncSessionLocal() as session:
          yield session
  ```
* **Usage**:

  * Inject `db: AsyncSession = Depends(get_db)` into endpoints
  * Use `await db.execute(...)`, `await db.commit()`, etc.

---

### 3. `face_recognition.py`

* **Purpose**: Wraps YOLOv8 detector and DeepFace’s ArcFace embedder; handles embedding computation and matching.
* **Key functions**:

  ```python
  def load_models():
      detector = YOLO(settings.YOLO_MODEL_PATH)
      embedder = DeepFace.build_model("ArcFace")
      return detector, embedder

  def detect_faces(frame: np.ndarray) -> List[BoundingBox]:
      # runs YOLO inference → returns boxes + confidences

  def compute_embeddings(crops: List[np.ndarray]) -> np.ndarray:
      # preprocess crops → feed to ArcFace → return L2-normalized embeddings

  def match_embedding(emb: np.ndarray,
                      gallery: np.ndarray,
                      threshold: float = settings.UNKNOWN_SIM_THRESHOLD) -> Tuple[bool, float]:
      # compute cosine similarities → return (is_known, max_similarity)
  ```
* **Usage**:

  ```python
  from core.face_recognition import detect_faces, compute_embeddings, match_embedding
  boxes = detect_faces(frame)
  embs = compute_embeddings(crop_images)
  known, sim = match_embedding(embs[i], gallery)
  ```

---

### 4. `tracking.py`

* **Purpose**: Maintains a global tracker instance (DeepSORT or ByteTrack) and last‐seen centroids.
* **Key class**:

  ```python
  class TrackingContext:
      def __init__(self):
          self.tracker = DeepSort(... )  # or ByteTrack
          self.last_positions: Dict[int, Tuple[float, float]] = {}

      def update(self, frame_id: str, bboxes: List[BoundingBox]) -> List[Track]:
          # run tracker → update self.last_positions → return tracks
  ```
* **Usage**:

  ```python
  from core.tracking import TrackingContext
  ctx = TrackingContext()
  tracks = ctx.update(frame_id, detected_boxes)
  ```

---

### 5. `crop.py`

* **Purpose**: Saves unique “unknown” face crops to disk and reloads the gallery embeddings on each save.
* **Key functions**:

  ```python
  def save_unknown_gallery(crop: np.ndarray) -> Path:
      filename = f"{uuid4()}.jpg"
      path = UNKNOWN_DIR / filename
      cv2.imwrite(str(path), crop)
      return path

  def load_unknown_gallery() -> np.ndarray:
      # load all images from UNKNOWN_DIR → compute embeddings → return as array
  ```
* **Usage**:

  ```python
  from core.crop import save_unknown_gallery, load_unknown_gallery
  path = save_unknown_gallery(crop_img)
  gallery = load_unknown_gallery()
  ```

---

### 6. `utils.py`

* **Purpose**: General helper functions for image I/O and ID generation.
* **Examples**:

  ```python
  def bytes_to_image(data: bytes) -> np.ndarray:
      # convert JPEG/PNG bytes → OpenCV BGR array

  def generate_uuid() -> str:
      return str(uuid4())
  ```
* **Usage**:

  ```python
  from core.utils import bytes_to_image, generate_uuid
  frame = bytes_to_image(await file.read())
  uid = generate_uuid()
  ```

---

With these core modules in place, the `api/endpoints` layer can simply import and orchestrate detection, embedding, tracking, cropping, and logging logic—allowing for a clean separation of concerns and easy extensibility.

```
```
