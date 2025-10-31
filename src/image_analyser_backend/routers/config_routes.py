from fastapi import APIRouter
from image_analyser_backend.request_schemas.config_request import ConfigUpdateRequestBody
from image_analyser_backend.services.config_service import ConfigService

from image_analyser_backend.services.detector import DetectorFactory
from image_analyser_backend.services.rule_engine import RuleEngineFactory

routers = APIRouter(prefix="/config", tags=["config"])

@routers.put("")
def update(payload: ConfigUpdateRequestBody):
    config_service: ConfigService = ConfigService()
    return config_service.config_update(input_data=payload)