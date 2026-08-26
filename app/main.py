"""FastAPI application factory.

Factory pattern so tests can spin up isolated app instances with overridden
settings/dependencies — no module-level global state.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    yield
    # shutdown cleanup goes here (engine disposal, etc.) in later steps


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mozhi",
        description="AI multilingual dubbing platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    from app.api_job_status import router as job_status_router
    from app.api_jobs import router as jobs_router

    app.include_router(jobs_router)
    app.include_router(job_status_router)

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
