import io
import os
from typing import List, Optional
from pydantic import BaseModel
from PIL import Image

from image_analyser_backend.schemas.analysis_result import Detection as ApiDetection, BBox
from image_analyser_backend.services.onnx_detector import OnnxYoloV8Detector, COCO80

from image_analyser_backend.core.settings import settings

class DetectorConfig(BaseModel):
    model_path: str = settings.YOLO_ONNX_PATH
    imgsz: int = settings.YOLO_IMGSZ
    confidence_threshold: float = settings.CONFIDENCE_THRESHOLD
    iou_threshold: float = settings.IOU_THRESHOLD
    max_dets: int = settings.MAX_DETS
    providers: Optional[list[str]] = None  # e.g. ["CPUExecutionProvider"] or ["CUDAExecutionProvider","CPUExecutionProvider"]

class Detector:
    def __init__(self, name: str, backend: str, config: DetectorConfig, impl: OnnxYoloV8Detector):
        self.name = name
        self.backend = backend
        self.config = config
        self._impl = impl

    def class_names(self) -> list[str]:
        return list(COCO80)

    def detect(self, image_bytes: bytes, min_confidence: Optional[float] = None) -> List[ApiDetection]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if min_confidence is not None:
            self._impl.conf_threshold = float(min_confidence)

        res = self._impl.infer(img, size=(self.config.imgsz, self.config.imgsz))

        out: List[ApiDetection] = []
        for detection in res.detections:
            out.append(ApiDetection(
                label=str(detection.label),
                class_id=int(detection.class_id),
                score=float(detection.score),
                bbox=BBox(x1=float(detection.x1), y1=float(detection.y1), x2=float(detection.x2), y2=float(detection.y2)),
            ))
        return out

class DetectorFactory:
    @staticmethod
    def load_default_detector() -> Detector:
        detector_config = DetectorConfig()
        providers = detector_config.providers or ["CPUExecutionProvider"]

        impl = OnnxYoloV8Detector(
            model_path=detector_config.model_path,
            class_names=COCO80,
            providers=providers,
            input_size=(detector_config.imgsz, detector_config.imgsz),
            conf_threshold=detector_config.confidence_threshold,
            nms_iou_threshold=detector_config.iou_threshold,
            max_dets=detector_config.max_dets,
            prefer_sigmoid=True,
            fallback_mock=True,  # allows running without model for dev
        )

        backend = impl._provider
        name = os.path.basename(detector_config.model_path) if os.path.exists(detector_config.model_path) else f"mock_model {detector_config.model_path}"
        print (f"Loaded detector model '{name}' with backend '{backend}'")
        print (f"Model Path : {detector_config.model_path} config: imgsz={detector_config.imgsz}, conf={detector_config.confidence_threshold}, iou={detector_config.iou_threshold}, max_dets={detector_config.max_dets}")
        return Detector(name=name, backend=backend, config=detector_config, impl=impl)