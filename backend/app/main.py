"""FastAPI app entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import get_settings
from app.models.registry import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("sorai.backend")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    log.info("=" * 60)
    log.info("SOR-AI Backend starting")
    log.info("  models_dir : %s", settings.models_dir)
    log.info("  feature_set: %s", settings.feature_set)
    log.info("=" * 60)

    registry.load_from_dir(settings.models_dir)

    if not registry.models:
        log.error("No models loaded! /api/v1/predict will return 503.")
    else:
        log.info("Ready — %d model(s): %s", len(registry.models), registry.loaded_ids)

    yield

    log.info("Shutting down")


app = FastAPI(
    title="SOR-AI Triage API",
    version="1.0.0",
    description="ML-powered Manchester Triage decision support",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def root() -> dict:
    return {"service": "SOR-AI Triage API", "version": "1.0.0", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
