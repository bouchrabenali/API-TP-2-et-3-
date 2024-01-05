"""API Router for Fast API."""
from fastapi import APIRouter
from src.api.routes import hello
from src.api.routes.data import router as data_router

router = APIRouter()

router.include_router(hello.router, tags=["Hello"])
router.include_router(data_router, tags=["Data"])
