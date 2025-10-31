from typing import List, Optional, Dict
from pydantic import BaseModel
from enum import Enum

class RiskScale(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

class Detection(BaseModel):
    label: str
    class_id: int
    score: float
    bbox: BBox

class RuleBreach(BaseModel):
    rule_id: str
    description: str
    severity: float            # keep float, rules use non-integers
    involved_indices: Optional[List[int]] = None  # indices into AnalysisResult.objects

class AnalysisResult(BaseModel):
    image_sha256: str
    objects: List[Detection]
    breaches: List[RuleBreach]
    overall_risk: float
    overall_scale: RiskScale
    perf_ms: Dict[str, float]
    detector: Dict[str, str]             # {"name": "...", "backend": "..."}
    config: Dict[str, object]            # {"min_confidence": ..., "rule_weights": {...}}
    annotated_image_b64: Optional[str] = None