from contextlib import asynccontextmanager
from fastapi import FastAPI

from image_analyser_backend.services.detector import DetectorFactory
from image_analyser_backend.services.rule_engine import RuleEngineFactory

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = DetectorFactory.load_default_detector()
    app.state.rules = RuleEngineFactory.load_default_rule_engine()
    yield # app runs while paused here
    # teardown if needed later
