"""Hardcoded safety rules — deterministic, applied OUTSIDE the LLM."""
from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import MedGemmaAssessment, ModelPrediction, Vitals


@dataclass
class SafetyResult:
    final_category: int
    alert_doctor: bool
    severity: str            # "low" | "high"
    detected: bool           # conflict detected
    message: str


def apply_safety_rules(
    final_category: int,
    confidence: float,
    model_predictions: list[ModelPrediction],
    medgemma: MedGemmaAssessment,
    vitals: Vitals,
    confidence_threshold: float = 0.6,
    conflict_threshold: int = 2,
) -> SafetyResult:

    messages: list[str] = []
    alert_doctor = False
    detected = False
    severity = "low"

    # Rule 1: Never downgrade below the minimum (most-urgent) of all models
    if model_predictions:
        min_category = min(p.category for p in model_predictions)
        if final_category > min_category:
            messages.append(f"Kategoria podniesiona do minimum modeli ({min_category}).")
            final_category = min_category

    # Rule 2: Spread ≥ conflict_threshold between models → alert doctor
    if model_predictions:
        cats = [p.category for p in model_predictions]
        max_diff = max(cats) - min(cats)
        if max_diff >= conflict_threshold:
            detected = True
            severity = "high"
            alert_doctor = True
            messages.append(f"Rozbieżność {max_diff} stopni MTS między modelami.")
        elif max_diff >= 1:
            detected = True
            severity = "low"
            messages.append("Niewielka rozbieżność między modelami.")

    # Rule 3: Low MedGemma confidence → escalate
    if medgemma.confidence < confidence_threshold:
        alert_doctor = True
        detected = True
        severity = "high"
        messages.append(f"Niska pewność oceny klinicznej ({medgemma.confidence:.2f}).")

    # Rule 4: Critical vitals → force at least Orange
    critical_vitals = (
        vitals.sbp < 90
        or vitals.o2 < 88
        or vitals.hr > 150
        or vitals.rr > 32
    )
    if critical_vitals and final_category > 1:
        messages.append("Krytyczne parametry życiowe → kategoria Orange minimum.")
        final_category = 1
        alert_doctor = True

    # Rule 5: ML says Red → always alert
    if model_predictions and min(p.category for p in model_predictions) == 0:
        alert_doctor = True

    # Rule 6: ML vs MedGemma differ by ≥ 2 → escalate
    if model_predictions:
        ml_cat = min(p.category for p in model_predictions)
        if abs(ml_cat - medgemma.category) >= 2:
            detected = True
            severity = "high"
            alert_doctor = True
            messages.append("Rozbieżność między ML a oceną kliniczną.")

    if not messages:
        message = "Wszystkie modele zgodne"
    else:
        message = " ".join(messages)

    return SafetyResult(
        final_category=final_category,
        alert_doctor=alert_doctor,
        severity=severity,
        detected=detected,
        message=message,
    )
