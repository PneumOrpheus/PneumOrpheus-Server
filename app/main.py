from fastapi import FastAPI

from app.config import get_settings
from app.routes.cancer import router as cancer_router
from app.routes.inference import router as inference_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(cancer_router)
app.include_router(inference_router)
