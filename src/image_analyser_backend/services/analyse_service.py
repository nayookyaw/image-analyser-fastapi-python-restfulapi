import hashlib
import time
from typing import Optional, List
from fastapi import UploadFile
from starlette.status import HTTP_400_BAD_REQUEST

from image_analyser_backend.core.settings import settings
from image_analyser_backend.schemas.analysis_result import (
    AnalysisResult, RiskScale, Detection
)
from image_analyser_backend.utils.image_util import ImageUtil
from image_analyser_backend.services.detector import DetectorFactory
from image_analyser_backend.services.rule_engine import RuleEngineFactory
from image_analyser_backend.response_handlers.json_response_handler import Response_SUCCESS

class AnalyseService:
    """Minimal analysis service: validate → detect → rules → package result."""

    def __init__(self, max_upload_mb: Optional[int] = None) -> None:
        self.max_bytes = (max_upload_mb or settings.MAX_UPLOAD_MB) * 1024 * 1024

    async def run_analyse(
        self,
        *,
        input_img: UploadFile,
        is_return_visualisation: bool,
        min_confidence: float,
    ):
        """Run the full pipeline and return an Analysis Result."""

        content_type: Optional[str] = input_img.content_type
        image_bytes: Optional[bytes] = await input_img.read()
        detector = DetectorFactory.load_default_detector()
        rules = RuleEngineFactory.load_default_rule_engine()

        if not self._validate_image_payload(content_type, image_bytes):
            return Response_SUCCESS(status_code=HTTP_400_BAD_REQUEST, message="Invalid image payload!")

        t0 = time.perf_counter()
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        # Detect
        t_det = time.perf_counter()
        detections: List[Detection] = detector.detect(
            image_bytes, min_confidence=min_confidence
        )
        det_ms = (time.perf_counter() - t_det) * 1000.0

        # Rules
        t_rule = time.perf_counter()
        breaches, risk_score = rules.evaluate(detections)
        rule_ms = (time.perf_counter() - t_rule) * 1000.0

        # Scale + optional visual
        scale = self._compute_scale(risk_score)
        annotated_b64 = None
        if is_return_visualisation:
            annotated_b64 = ImageUtil.annotate_image_base64(image_bytes, detections, breaches)

        total_ms = (time.perf_counter() - t0) * 1000.0

        analysis_result = AnalysisResult(
            image_sha256=image_hash,
            objects=detections,
            breaches=breaches,
            overall_risk=round(risk_score, 2),
            overall_scale=scale,
            perf_ms={
                "total": round(total_ms, 2),
                "detection": round(det_ms, 2),
                "rules": round(rule_ms, 2),
            },
            detector={"name": detector.name, "backend": detector.backend},
            config={"min_confidence": min_confidence, "rule_weights": getattr(rules.config, "weights", {})},
            annotated_image_b64=annotated_b64,
        )
    
        return Response_SUCCESS(
            status_code=200,
            message="Image has been analysed successfully.",
            data=analysis_result
        )

    # ---------- private helpers ----------
    def _validate_image_payload(self, content_type: Optional[str], image_bytes: bytes) -> bool:
        cct = (content_type or "").lower()
        is_valid: bool = True

        allowed_types = {"image/jpeg", "image/png"}
        if cct not in allowed_types:
            # raise AnalyseError("Only JPEG and PNG image formats are supported.")
            is_valid = False
        if not image_bytes or len(image_bytes) < 64:
            # raise AnalyseError("Empty or invalid image.")
            is_valid = False
        if len(image_bytes) > self.max_bytes:
            # raise AnalyseError(f"Payload too large (>{self.max_bytes // 1024 // 1024} MB).")
            is_valid = False
        return is_valid

    @staticmethod
    def _compute_scale(risk_score: float) -> RiskScale:
        if risk_score < 3:
            return RiskScale.LOW
        if risk_score < 6:
            return RiskScale.MEDIUM
        return RiskScale.HIGH
