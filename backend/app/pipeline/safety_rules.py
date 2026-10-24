"""Hardcoded safety rules — deterministic, applied OUTSIDE the ML models and the LLM.

Class index 0–4 = ESI 1–5; TOPSOR colours (red … blue) are only shown in messages.
The ESI rules mirror training/src/explain/esi_rules.py (ESI Implementation
Handbook v4; tests/test_safety_rules.py checks the thresholds against it):

  Point A — life-saving intervention needed now → ESI 1:
      SpO2 < 85 %, SBP < 80 mmHg, HR < 40 or > 150/min, RR < 8 or > 35/min,
      T < 32 °C or ≥ 41 °C; complaints: cardiac arrest, unresponsive.
  Point B — high-risk situation → at least ESI 2:
      SBP < 90 mmHg, pain ≥ 7/10; complaints: respiratory distress, stroke alert,
      trauma activation, suicidal / homicidal ideation, intentional overdose.
  Point D — danger-zone vitals (adults ≥ 18 y) → at least ESI 2:
      HR > 100/min, RR > 20/min, SpO2 < 92 %.

Rules never lower the urgency — they only escalate and alert the doctor.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from app.models.schemas import MedGemmaAssessment, ModelPrediction, Vitals

# ── ESI levels (display) ─────────────────────────────────────────────

TOPSOR_COLORS_PL: tuple[str, ...] = ("czerwony", "pomarańczowy", "żółty", "zielony", "niebieski")


def esi_label(category: int) -> str:
    """Class index 0–4 → 'ESI 2 (pomarańczowy)'."""
    return f"ESI {category + 1} ({TOPSOR_COLORS_PL[category]})"


# ── ESI rules ─────────────────────────────────────────────────────────

# Danger zone (point D) for adults — strict thresholds (> / <)
ESI_DANGER_ZONE_ADULT: dict[str, float] = {"hr_above": 100.0, "rr_above": 20.0, "spo2_below": 92.0}
ADULT_AGE: int = 18


@dataclass(frozen=True)
class VitalRule:
    vital: str                        # Vitals attribute
    test: Callable[[float], bool]
    esi: int                          # ESI level 1 or 2
    decision_point: str               # "A" | "B" | "D"
    description: str


VITAL_RULES: tuple[VitalRule, ...] = (
    # Point A → ESI 1
    VitalRule("o2", lambda x: x < 85, 1, "A", "krytyczna hipoksja (SpO2 < 85%)"),
    VitalRule("sbp", lambda x: x < 80, 1, "A", "ciężka hipotensja / wstrząs (SBP < 80 mmHg)"),
    VitalRule("hr", lambda x: x > 150, 1, "A", "skrajna tachykardia (HR > 150/min)"),
    VitalRule("hr", lambda x: x < 40, 1, "A", "skrajna bradykardia (HR < 40/min)"),
    VitalRule("rr", lambda x: x > 35, 1, "A", "skrajna tachypnoe (RR > 35/min)"),
    VitalRule("rr", lambda x: x < 8, 1, "A", "bradypnoe (RR < 8/min)"),
    VitalRule("temp", lambda x: x >= 41.0, 1, "A", "hipertermia zagrażająca życiu (T ≥ 41 °C)"),
    VitalRule("temp", lambda x: x < 32.0, 1, "A", "głęboka hipotermia (T < 32 °C)"),
    # Point B → ESI 2
    VitalRule("sbp", lambda x: x < 90, 2, "B", "hipotensja (SBP < 90 mmHg)"),
)

DANGER_ZONE_RULES: tuple[VitalRule, ...] = (
    VitalRule("hr", lambda x: x > ESI_DANGER_ZONE_ADULT["hr_above"], 2, "D",
              "strefa zagrożenia: HR > 100/min"),
    VitalRule("rr", lambda x: x > ESI_DANGER_ZONE_ADULT["rr_above"], 2, "D",
              "strefa zagrożenia: RR > 20/min"),
    VitalRule("o2", lambda x: x < ESI_DANGER_ZONE_ADULT["spo2_below"], 2, "D",
              "strefa zagrożenia: SpO2 < 92%"),
)

SEVERE_PAIN: float = 7.0   # point B: pain ≥ 7/10 (if reported in the note)

# Chief complaints (Layer 0 cc_* flags) that indicate point A or B by themselves
COMPLAINT_RULES: dict[str, tuple[int, str]] = {
    "cc_cardiacarrest": (1, "zatrzymanie krążenia"),
    "cc_unresponsive": (1, "brak reakcji na bodźce"),
    "cc_respiratorydistress": (2, "niewydolność oddechowa"),
    "cc_strokealert": (2, "alert udarowy"),
    "cc_trauma_activation": (2, "aktywacja zespołu urazowego"),
    "cc_suicidal": (2, "myśli samobójcze"),
    "cc_homicidal": (2, "zagrożenie dla innych"),
    "cc_overdose_intentional": (2, "celowe przedawkowanie"),
}


@dataclass(frozen=True)
class EsiTrigger:
    esi: int                          # ESI level indicated (1 or 2)
    decision_point: str               # "A" | "B" | "D"
    description: str


def esi_rule_triggers(
    vitals: Vitals,
    cc_flags: Mapping[str, float] | None = None,
    pain: float | None = None,
) -> list[EsiTrigger]:
    """All ESI rules (points A, B, D) met by the patient."""
    triggers: list[EsiTrigger] = []
    for code, (esi, desc) in COMPLAINT_RULES.items():
        if (cc_flags or {}).get(code, 0) > 0:
            triggers.append(EsiTrigger(esi, "A" if esi == 1 else "B", desc))
    rules = VITAL_RULES + (DANGER_ZONE_RULES if vitals.age >= ADULT_AGE else ())
    for rule in rules:
        if rule.test(float(getattr(vitals, rule.vital))):
            triggers.append(EsiTrigger(rule.esi, rule.decision_point, rule.description))
    if pain is not None and pain >= SEVERE_PAIN:
        triggers.append(EsiTrigger(2, "B", f"silny ból ({pain:g}/10)"))
    return triggers


def esi_rule_floor(triggers: list[EsiTrigger]) -> int | None:
    """Most urgent class index required by the rules (0 = ESI 1, 1 = ESI 2); None = no rule."""
    return min(t.esi for t in triggers) - 1 if triggers else None


# ── Safety layer ──────────────────────────────────────────────────────

@dataclass
class SafetyResult:
    final_category: int
    alert_doctor: bool
    severity: str            # "low" | "high"
    detected: bool           # conflict detected
    message: str
    escalations: list[str] = field(default_factory=list)   # ESI-rule escalations (also in message)


def apply_safety_rules(
    final_category: int,
    confidence: float,
    model_predictions: list[ModelPrediction],
    medgemma: MedGemmaAssessment,
    vitals: Vitals,
    confidence_threshold: float = 0.6,
    conflict_threshold: int = 2,
    cc_flags: Mapping[str, float] | None = None,
    pain: float | None = None,
) -> SafetyResult:

    messages: list[str] = []
    escalations: list[str] = []
    alert_doctor = False
    detected = False
    severity = "low"

    # Rule 1: Never downgrade below the most urgent model
    if model_predictions:
        min_category = min(p.category for p in model_predictions)
        if final_category > min_category:
            messages.append(f"Poziom podniesiony do najpilniejszego wskazania modeli: {esi_label(min_category)}.")
            final_category = min_category

    # Rule 2: Spread ≥ conflict_threshold ESI levels between models → alert doctor
    if model_predictions:
        cats = [p.category for p in model_predictions]
        max_diff = max(cats) - min(cats)
        if max_diff >= conflict_threshold:
            detected = True
            severity = "high"
            alert_doctor = True
            messages.append(f"Rozbieżność {max_diff} poziomów ESI między modelami.")
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

    # Rule 4: ESI rules — point A → ESI 1, points B / D → at least ESI 2
    triggers = esi_rule_triggers(vitals, cc_flags, pain)
    floor = esi_rule_floor(triggers)
    if floor is not None:
        if floor == 0:
            alert_doctor = True          # life-saving intervention needed now
        if final_category > floor:
            reasons = ", ".join(t.description for t in triggers if t.esi - 1 == floor)
            if floor == 0:
                text = f"Punkt A ESI (natychmiastowa interwencja): {reasons} → {esi_label(0)}."
            else:
                text = f"Sytuacja wysokiego ryzyka ESI: {reasons} → minimum {esi_label(1)}."
            escalations.append(text)
            messages.append(text)
            final_category = floor
            alert_doctor = True

    # Rule 5: ML says ESI 1 → always alert
    if model_predictions and min(p.category for p in model_predictions) == 0:
        alert_doctor = True

    # Rule 6: ML vs MedGemma differ by ≥ 2 ESI levels → escalate
    if model_predictions:
        ml_cat = min(p.category for p in model_predictions)
        if abs(ml_cat - medgemma.category) >= 2:
            detected = True
            severity = "high"
            alert_doctor = True
            messages.append("Rozbieżność między ML a oceną kliniczną.")

    message = " ".join(messages) if messages else "Wszystkie modele zgodne"

    return SafetyResult(
        final_category=final_category,
        alert_doctor=alert_doctor,
        severity=severity,
        detected=detected,
        message=message,
        escalations=escalations,
    )
