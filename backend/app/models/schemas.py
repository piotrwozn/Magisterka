"""Pydantic schemas — must match frontend/src/lib/types.ts exactly (camelCase aliases)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Base — emits camelCase keys but accepts both casings on input."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ── Request ──────────────────────────────────────────────────────────

class Vitals(BaseModel):
    age: int = Field(ge=0, le=120)
    temp: float = Field(ge=30.0, le=45.0)
    hr: int = Field(ge=20, le=300)
    sbp: int = Field(ge=40, le=300)
    dbp: int = Field(ge=20, le=200)
    rr: int = Field(ge=5, le=80)
    o2: float = Field(ge=50.0, le=100.0)


class PredictRequest(CamelModel):
    vitals: Vitals
    clinical_note: str = Field("", max_length=2000)


# ── Response ─────────────────────────────────────────────────────────

class ModelPrediction(CamelModel):
    model_name: str
    category: int = Field(ge=0, le=4)
    probabilities: list[float]
    confidence: float


class ShapValue(CamelModel):
    feature: str
    value: float
    direction: Literal["positive", "negative"]


class MedGemmaAssessment(CamelModel):
    category: int = Field(ge=0, le=4)
    confidence: float
    reasoning: str
    risk_flags: list[str]
    key_findings: list[str]


class ConflictInfo(CamelModel):
    detected: bool
    severity: Literal["low", "high"]
    alert_doctor: bool
    message: str


class PredictResponse(CamelModel):
    final_category: int = Field(ge=0, le=4)
    confidence: float
    model_predictions: list[ModelPrediction]
    medgemma: MedGemmaAssessment
    shap_top5: list[ShapValue]
    conflict: ConflictInfo
    processing_time_ms: int


class HealthStatus(CamelModel):
    status: Literal["ok", "degraded", "down"]
    models_loaded: list[str]
    ollama_ready: bool
    uptime_seconds: int
