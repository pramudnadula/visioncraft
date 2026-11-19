"""FastAPI application factory, lifespan management, and server entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import jobs, media, models, system
from backend.app.api.routes import settings as settings_routes
from backend.app.api.websocket import ws_router
from backend.app.core.config import Settings, get_settings
from backend.app.core.dependencies import AppStateContainer
from backend.app.core.exceptions import FaceVisionError
from backend.app.core.logging import configure_logging, get_logger
from backend.app.database.session import DatabaseManager
from backend.app.detection.registry import create_default_registry
from backend.app.jobs.events import JobEventBus
from backend.app.jobs.manager import JobManager
from backend.app.storage.factory import create_storage_adapter

logger = get_logger("main")


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    """Create and configure the FaceVision FastAPI application."""
    cfg = custom_settings or get_settings()
    cfg.ensure_directories()
    configure_logging(debug=cfg.debug)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
        db = DatabaseManager(cfg.resolved_database_url)
        db.create_tables()

        storage = create_storage_adapter(cfg)
        registry = create_default_registry(models_dir=cfg.resolved_models_dir)
        event_bus = JobEventBus()
        job_manager = JobManager(
            settings=cfg,
            db=db,
            storage=storage,
            registry=registry,
            event_bus=event_bus,
        )
        job_manager.recover_on_startup()

        app_instance.state.container = AppStateContainer(
            settings=cfg,
            db=db,
            storage=storage,
            registry=registry,
            event_bus=event_bus,
            job_manager=job_manager,
        )
        logger.info("FaceVision API started (%s mode)", cfg.execution_mode)
        try:
            yield
        finally:
            job_manager.shutdown()
            logger.info("FaceVision API stopped")

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(FaceVisionError)
    async def facevision_error_handler(_request: Request, exc: FaceVisionError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error_code, "message": exc.message},
        )

    prefix = cfg.api_prefix
    app.include_router(system.router, prefix=prefix)
    app.include_router(media.router, prefix=prefix)
    app.include_router(jobs.router, prefix=prefix)
    app.include_router(models.router, prefix=prefix)
    app.include_router(settings_routes.router, prefix=prefix)
    app.include_router(ws_router)

    return app


app = create_app()


def run_server() -> None:
    """CLI entrypoint for `facevision-api`."""
    cfg = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=cfg.host,
        port=cfg.port,
        reload=cfg.debug,
    )


if __name__ == "__main__":
    run_server()
