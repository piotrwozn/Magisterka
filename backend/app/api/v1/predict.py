"""POST /api/v1/predict endpoint."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.registry import registry
from app.models.schemas import PredictRequest, PredictResponse
from app.pipeline.orchestrator import predict

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse, response_model_by_alias=True)
async def predict_endpoint(request: PredictRequest) -> PredictResponse:
    if not registry.models:
        raise HTTPException(status_code=503, detail="No ML models loaded")
    try:
        return await predict(request, registry)
    except Exception as exc:  # noqa: BLE001
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
