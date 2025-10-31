from fastapi import Request

from image_analyser_backend.services.detector import Detector
from image_analyser_backend.services.rule_engine import RuleEngine

def get_detector(request: Request) -> Detector:
    return request.app.state.detector

def get_rules(request: Request) -> RuleEngine:
    return request.app.state.rules
