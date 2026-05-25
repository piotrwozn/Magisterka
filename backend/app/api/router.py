"""Aggregates v1 routes."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import health, predict

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(predict.router, tags=["inference"])
api_router.include_router(health.router, tags=["health"])
