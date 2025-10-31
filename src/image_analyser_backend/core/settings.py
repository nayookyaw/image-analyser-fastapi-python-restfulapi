from pydantic_settings import BaseSettings
from typing import Dict, Tuple

class Settings(BaseSettings):
    APP_NAME: str = "Image Analyser Backend API (FastAPI)"
    APP_VERSION: str = "0.1.0"
    RULE_CONFIG_PATH: str = "config/rules.yaml"

    # detector settings
    YOLO_ONNX_PATH: str = "yolo_model/yolov8n.onnx"
    # YOLO_ONNX_PATH: str = "yolo_model/nomodel.onnx"
    YOLO_IMGSZ: int = 640 # pixels (YOLOv8 default)
    CONFIDENCE_THRESHOLD: float = 0.7
    IOU_THRESHOLD: float = 0.3
    MAX_DETS: int = 300

    # Rule engine constants
    DEFAULT_PROXIMITY_THRESHOLD: int = 120  # pixels
    DEFAULT_MISSING_PPE_LABELS: Tuple[str, ...] = ("helmet", "vest")
    DEFAULT_RULE_WEIGHTS: Dict[str, float] = {
        "proximity": 0.6,
        "ppe": 0.4,
    }
    DEFAULT_RULE_CAPS: Dict[str, float] = {
        "proximity": 10.0,
        "ppe": 7.0,
    }
    
    CORS_ORIGINS: str = "*"
    MAX_UPLOAD_MB: int = 10

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()
