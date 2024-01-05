from fastapi import FastAPI, Request, HTTPException, status
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from src.api.router import router
from src.services.utils import setup_logger

def get_application() -> FastAPI:
    application = FastAPI(
        title="epf-flower-data-science",
        description="Fast API",
        version="1.0.0",
        redoc_url=None,
    )

    logger = setup_logger()

    @application.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "Oops! This resource does not exist."},
            )
        # Handle other HTTP exceptions here if needed
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.detail}
        )

    @application.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc} - Path: {request.url.path}")
        return JSONResponse(status_code=500, content={"message": "Internal server error"})

    @application.on_event("startup")
    async def startup_event():
        logger.info("Starting up the application")

    @application.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down the application")

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)
    return application
