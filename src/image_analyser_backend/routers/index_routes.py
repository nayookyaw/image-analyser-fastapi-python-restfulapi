from fastapi import APIRouter
from image_analyser_backend.routers import config_routes, analyse_routes

api_v1 = APIRouter(prefix="") # versioned API group (/v1)
api_v1.include_router(config_routes.routers)
api_v1.include_router(analyse_routes.routers)

router = APIRouter()
router.include_router(api_v1)
