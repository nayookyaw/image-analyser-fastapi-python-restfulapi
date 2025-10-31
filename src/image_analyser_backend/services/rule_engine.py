
from dataclasses import field
from typing import List, Dict, Tuple
import math
import os
import yaml
from pydantic import BaseModel

from image_analyser_backend.schemas.analysis_result import Detection, RuleBreach, BBox
from image_analyser_backend.core.settings import settings

class RuleConfig(BaseModel):
    # Distances in pixels for now (could be meters if camera calibration known)
    proximity_threshold: int = settings.DEFAULT_PROXIMITY_THRESHOLD # pixels
    missing_ppe_labels: Tuple[str, ...] = settings.DEFAULT_MISSING_PPE_LABELS
    weights: Dict[str, float] = field(default_factory=lambda: settings.DEFAULT_RULE_WEIGHTS)
    # Severity caps per rule on a 0..10 scale
    caps: Dict[str, float] = field(default_factory=lambda: settings.DEFAULT_RULE_CAPS)

class RuleEngine:
    """Applies safety rules to a set of detections and computes an overall risk score.

    Overall risk is a weighted sum of normalized rule severities, clipped to 0..10:
        score = 10 * ( w_p * prox_norm + w_e * ppe_norm )
    where each norm is 0..1 derived from rule-specific severity/cap.
    """

    def __init__(self, config: RuleConfig):
        self.config = config

    @staticmethod
    def _center(b: BBox) -> Tuple[float, float]:
        return ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)

    def _proximity_rule(self, dets: List[Detection]) -> Tuple[List[RuleBreach], float]:
        """Person near forklift/heavy machinery within a pixel threshold increases severity."""
        person_idxs = [i for i, d in enumerate(dets) if d.label == "person"]
        machine_idxs = [i for i, d in enumerate(dets) if d.label in ("forklift", "truck", "machinery")]
        breaches: List[RuleBreach] = []
        max_severity = 0.0

        for pi in person_idxs:
            for mi in machine_idxs:
                pc = self._center(dets[pi].bbox)
                mc = self._center(dets[mi].bbox)
                dist = math.dist(pc, mc)
                if dist <= self.config.proximity_threshold:
                    # Inverse mapping: closer -> higher severity up to cap
                    severity = min(self.config.caps["proximity"], (self.config.proximity_threshold - dist) / self.config.proximity_threshold * self.config.caps["proximity"])
                    max_severity = max(max_severity, severity)
                    breaches.append(RuleBreach(
                        rule_id="proximity.person_machinery",
                        description=f"Person too close to machinery (≈{int(dist)}px).",
                        severity=round(severity, 2),
                        involved_indices=[pi, mi],
                    ))
        return breaches, max_severity

    def _ppe_rule(self, dets: List[Detection]) -> Tuple[List[RuleBreach], float]:
        """If there is at least one person but NO PPE items detected globally, raise severity."""
        has_person = any(d.label == "person" for d in dets)
        has_ppe_any = any(d.label in self.config.missing_ppe_labels for d in dets)
        breaches: List[RuleBreach] = []
        severity = 0.0
        if has_person and not has_ppe_any:
            severity = self.config.caps["ppe"] * 0.8  # default
            # Attach to all persons for visibility
            involved = [i for i, d in enumerate(dets) if d.label == "person"]
            breaches.append(RuleBreach(
                rule_id="ppe.missing_global",
                description="People detected but no PPE (helmet/vest) visible.",
                severity=round(severity, 2),
                involved_indices=involved,
            ))
        return breaches, severity

    def evaluate(self, dets: List[Detection]) -> Tuple[List[RuleBreach], float]:
        """Returns (all_breaches, overall_risk[0..10])."""
        all_breaches: List[RuleBreach] = []
        prox_breaches, prox_sev = self._proximity_rule(dets)
        ppe_breaches, ppe_sev = self._ppe_rule(dets)

        all_breaches.extend(prox_breaches)
        all_breaches.extend(ppe_breaches)

        # Normalize per cap then weight
        prox_norm = (prox_sev / self.config.caps["proximity"]) if self.config.caps["proximity"] > 0 else 0
        ppe_norm = (ppe_sev / self.config.caps["ppe"]) if self.config.caps["ppe"] > 0 else 0

        score01 = self.config.weights["proximity"] * prox_norm + self.config.weights["ppe"] * ppe_norm
        score10 = max(0.0, min(10.0, 10.0 * score01))
        return all_breaches, round(score10, 2)

class RuleEngineFactory:
    @staticmethod
    def load_default_rule_engine() -> RuleEngine:
        rule_config_path = settings.RULE_CONFIG_PATH
        if rule_config_path and os.path.exists(rule_config_path):
            with open(rule_config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            cfg = RuleConfig(
                proximity_threshold=int(raw.get("proximity_threshold", 120)),
                missing_ppe_labels=tuple(raw.get("missing_ppe_labels", ["helmet", "vest"])),
                weights=raw.get("weights", {"proximity": 0.6, "ppe": 0.4}),
            )
            return RuleEngine(cfg)
        return RuleEngine(RuleConfig())
