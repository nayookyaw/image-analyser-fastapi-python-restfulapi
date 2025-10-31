from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

from image_analyser_backend.core.settings import settings
from image_analyser_backend.core.lifespan import lifespan
from image_analyser_backend.routers.index_routes import router as api_router
from image_analyser_backend.response_handlers.json_error_handler import custom_exception_handler

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS (meant for dev/testing; adjust in production as needed)
origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount a single aggregated router
app.include_router(api_router)

# Global exception handler
app.add_exception_handler(Exception,custom_exception_handler)
