# app/core/config.py

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Face Recognition"
    API_V1_STR: str = "/api"

    # Video source
    # VIDEO_SOURCE: int = 0
    VIDEO_SOURCE: str = "rtsp://lenokcam:lenokcam@192.168.29.70:554/stream1"
    # VIDEO_SOURCE: str = "vid.avi"

    # Face recognition
    FACE_MODEL_NAME: str = "SFace"
    FACE_DET_SIZE: tuple = (320, 320)
    FACE_DET_THRESH: float = 0.5
    UNKNOWN_SIMILARITY_THRESHOLD: float = 0.1

    # CORS settings
    ALLOWED_ORIGINS: list[str] = ["*"]  # Replace with frontend URL in prod

    # CSV paths
    CSV_PATH: str = "face_detections.csv"
    UNKNOWN_CSV_PATH: str = "unknown_count.csv"

    # Buffers
    BUFFER_TIME: int = 5
    UNKNOWN_BUFFER_TIME: int = 5

    # Embeddings path
    EMBEDDING_PATH: str = "embed.npz"

    # Context ID for GPU (set -1 for CPU)
    CONTEXT_ID: int = 0

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
