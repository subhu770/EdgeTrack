"""FastAPI application entrypoint for EdgeTrack."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import get_detector, get_tracker, router
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for model pre-warming and cleanup."""
    logger.info(f"Initializing {settings.PROJECT_NAME} v{settings.VERSION}...")
    try:
        # Pre-warm detector and session
        detector = get_detector()
        _ = get_tracker()

        # Execute dummy warmup inference
        dummy_frame = np.zeros(
            (settings.INPUT_HEIGHT, settings.INPUT_WIDTH, 3), dtype=np.uint8
        )
        detector.detect(dummy_frame)
        logger.info("Inference engine pre-warmed successfully.")
    except Exception as exc:
        logger.error(f"Error during engine startup/pre-warming: {exc}", exc_info=True)

    yield

    logger.info(f"Shutting down {settings.PROJECT_NAME}...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API endpoints
    app.include_router(router)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": str(exc),
                "path": request.url.path,
            },
        )

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.DEBUG,
    )

