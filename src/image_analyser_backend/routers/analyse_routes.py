from fastapi import APIRouter
from fastapi import APIRouter, UploadFile, File, Query

from image_analyser_backend.services.analyse_service import AnalyseService

routers = APIRouter(prefix="/analyse", tags=["analyse"])

@routers.post("")
async def analyse_image(
    input_img: UploadFile = File(..., description="Image is required"),
    is_return_visualisation: bool = Query(True),
    min_confidence: float = Query(0.25, ge=0.0, le=1.0, description="Minimum confidence threshold value"),
):
    """Endpoint to analyse an image for detections and response detection breaches."""
    analyse_service: AnalyseService = AnalyseService()
    return await AnalyseService.run_analyse(
        analyse_service, input_img=input_img, is_return_visualisation=is_return_visualisation, min_confidence=min_confidence, 
    )
