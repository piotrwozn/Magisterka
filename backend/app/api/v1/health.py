"""GET /api/v1/health endpoint — reports model & Ollama status."""
from __future__ import annotations

import time

from fastapi import APIRouter

from app.config import get_settings
from app.inference.ollama_client import get_ollama_client
from app.models.registry import registry
from app.models.schemas import HealthStatus

router = APIRouter()
_START_TIME = time.time()


@router.get("/health", response_model=HealthStatus, response_model_by_alias=True)
async def health() -> HealthStatus:
    s = get_settings()

    ollama_ready = False
    if s.ollama_enabled:
        client = get_ollama_client(s.ollama_base_url, s.ollama_timeout_s)
        ollama_ready = await client.health()

    return HealthStatus(
        status="ok" if registry.models else "degraded",
        models_loaded=registry.loaded_ids,
        ollama_ready=ollama_ready,
        uptime_seconds=int(time.time() - _START_TIME),
    )
