"""
MTS Rule Engine — kliniczne reguły protokolu Manchester Triage System.

Sprawdza klasyfikację pod kątem zgodności z 52 flowchartami MTS.
Działa jako:
    1. Sanity check dla predykcji ML (czy model nie pomija oczywistych Red?)
    2. Dodatkowa warstwa eksplanacyjna ("dlaczego pacjent jest Red? bo SBP < 90")
    3. Hard-override w przypadku alarmujących vital signs

Zob. TECHNICAL_ANALYSIS.md §5.3.
"""

from __future__ import annotations

from typing import Any

from src.utils.config import CLASS_NAMES, MTS_MAX_WAIT_MINUTES
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# Definicje progów MTS dla każdej kategorii
# ─────────────────────────────────────────
# UWAGA: Te progi są zachowawcze (najpilniejszy próg w grupie wyznacza klasę).
# Logika: pacjent JEST danej kategorii jeśli SPEŁNIA którąkolwiek regułę.

MTS_VITAL_THRESHOLDS: dict[str, dict[str, Any]] = {
    "Red": {
        "description": "Natychmiastowy",
        "description_en": "Immediate",
        "max_wait_minutes": 0,
        "discriminators": {
            "airway_compromise": True,
            "breathing_inadequate": True,
            "shock": True,
            "unresponsive": True,
            "seizure_active": True,
        },
        "vital_rules": {
            # próg → opis kliniczny
            ("triage_sbp",   lambda x: x < 90):              "wstrząs (SBP < 90 mmHg)",
            ("triage_o2sat", lambda x: x < 85):              "krytyczna hipoksja (SpO2 < 85%)",
            ("triage_pulse", lambda x: x > 150):             "ekstremalna tachykardia (HR > 150)",
            ("triage_pulse", lambda x: x < 40):              "ekstremalna bradykardia (HR < 40)",
            ("triage_resp",  lambda x: x > 35):              "ciężka tachypnea (RR > 35)",
            ("triage_resp",  lambda x: x < 8):               "bradypnea/hipoatypia (RR < 8)",
            ("triage_temp",  lambda x: x > 41.0):            "hipertermia ekstremalna (T > 41°C)",
            ("triage_temp",  lambda x: x < 32.0):            "hipotermia ekstremalna (T < 32°C)",
            ("triage_pain",  lambda x: x >= 10):             "ból nieznośny (10/10)",
        },
    },
    "Orange": {
        "description": "Bardzo pilny",
        "description_en": "Very Urgent",
        "max_wait_minutes": 10,
        "vital_rules": {
            ("triage_sbp",   lambda x: x < 100):             "hipotensja (SBP < 100 mmHg)",
            ("triage_o2sat", lambda x: x < 92):              "hipoksja (SpO2 < 92%)",
            ("triage_pulse", lambda x: x > 130):             "tachykardia (HR > 130)",
            ("triage_pulse", lambda x: x < 50):              "bradykardia (HR < 50)",
            ("triage_resp",  lambda x: x > 30):              "tachypnea (RR > 30)",
            ("triage_temp",  lambda x: x > 40.0):            "wysoka gorączka (T > 40°C)",
            ("triage_temp",  lambda x: x < 34.0):            "hipotermia (T < 34°C)",
            ("triage_pain",  lambda x: x >= 8):              "ból bardzo silny (≥ 8/10)",
        },
    },
    "Yellow": {
        "description": "Pilny",
        "description_en": "Urgent",
        "max_wait_minutes": 60,
        "vital_rules": {
            ("triage_sbp",   lambda x: 100 <= x < 110):      "graniczna hipotensja (100–110 mmHg)",
            ("triage_o2sat", lambda x: 92 <= x < 95):        "saturacja graniczna (92–95%)",
            ("triage_pulse", lambda x: 100 < x <= 130):      "umiarkowana tachykardia (100–130)",
            ("triage_temp",  lambda x: 38.5 <= x < 40.0):    "gorączka (38,5–40°C)",
            ("triage_pain",  lambda x: 5 <= x < 8):          "ból umiarkowany (5–7/10)",
        },
    },
    "Green": {
        "description": "Standardowy",
        "description_en": "Standard",
        "max_wait_minutes": 120,
        "vital_rules": {
            ("triage_pain", lambda x: 2 <= x < 5):           "ból łagodny (2–4/10)",
        },
    },
    "Blue": {
        "description": "Niepilny",
        "description_en": "Non-Urgent",
        "max_wait_minutes": 240,
        "vital_rules": {
            ("triage_pain", lambda x: x < 2):                "minimalny ból lub jego brak",
        },
    },
}


# ─────────────────────────────────────────
# Aliases — różne nazwy kolumn vital signs
# ─────────────────────────────────────────
VITAL_COL_ALIASES: dict[str, list[str]] = {
    "triage_o2sat": ["triage_vital_o2", "o2sat", "spo2", "triage_o2sat"],
    "triage_sbp":   ["triage_vital_sbp", "sbp", "triage_sbp"],
    "triage_dbp":   ["triage_vital_dbp", "dbp", "triage_dbp"],
    "triage_pulse": ["triage_vital_hr", "pulse", "triage_pulse", "heart_rate"],
    "triage_resp":  ["triage_vital_rr", "resp", "triage_resp", "respiratory_rate"],
    "triage_temp":  ["triage_vital_temp", "temp", "triage_temp", "temperature"],
    "triage_pain":  ["triage_pain", "pain"],
}


def _resolve_vital(patient_data: dict, canonical: str) -> float | None:
    """Pobiera wartość vital sign sprawdzając kilka możliwych nazw kolumn."""
    aliases = VITAL_COL_ALIASES.get(canonical, [canonical])
    for alias in aliases:
        if alias in patient_data:
            value = patient_data[alias]
            if value is None:
                continue
            try:
                f = float(value)
                if f != f:  # NaN check
                    continue
                return f
            except (ValueError, TypeError):
                continue
    return None


# ─────────────────────────────────────────
# Główna funkcja
# ─────────────────────────────────────────
def rule_based_triage(
    patient_vitals: dict,
    discriminators: dict | None = None,
) -> dict[str, Any]:
    """
    Reguły MTS na vital signs — zwraca sugerowaną kategorię + uzasadnienie.

    Algorytm:
        1. Sprawdź Red — jeśli którakolwiek reguła spełniona, zwróć Red.
        2. Sprawdź Orange — analogicznie.
        3. ... aż do Blue.

    Parameters
    ----------
    patient_vitals : dict
        np. {'triage_sbp': 88, 'triage_pulse': 142, ...}
    discriminators : dict, optional
        Dodatkowe dyskryminatory binarne (airway_compromise, etc.).

    Returns
    -------
    dict z kluczami:
        - suggested_category    (str)
        - suggested_class_idx   (int)
        - max_wait_minutes      (int)
        - triggered_rules       (list[dict] — które reguły zadziałały i dlaczego)
        - all_violations        (list — wszystkie pogwałcenia, nie tylko tej klasy)
    """
    discriminators = discriminators or {}
    all_violations: list[dict] = []

    # Sprawdź każdą kategorię od najpilniejszej
    for category in ["Red", "Orange", "Yellow", "Green", "Blue"]:
        rules = MTS_VITAL_THRESHOLDS[category].get("vital_rules", {})
        cat_violations: list[dict] = []

        # 1. Hard discriminators (Red only)
        if category == "Red":
            disc = MTS_VITAL_THRESHOLDS[category].get("discriminators", {})
            for disc_name, expected in disc.items():
                if discriminators.get(disc_name) == expected:
                    cat_violations.append({
                        "type": "discriminator",
                        "name": disc_name,
                        "category": category,
                        "description": f"Dyskryminator MTS '{disc_name}'",
                    })

        # 2. Vital sign rules
        for (vital_name, rule_fn), description in rules.items():
            value = _resolve_vital(patient_vitals, vital_name)
            if value is None:
                continue
            try:
                if rule_fn(value):
                    cat_violations.append({
                        "type": "vital",
                        "vital": vital_name,
                        "value": value,
                        "category": category,
                        "description": description,
                    })
            except (TypeError, ValueError):
                continue

        all_violations.extend(cat_violations)

        # Pierwsza kategoria z violation = nasza klasyfikacja (best-of)
        if cat_violations and category != "Blue":  # Blue to fallback
            class_idx = CLASS_NAMES.index(category)
            return {
                "suggested_category": category,
                "suggested_class_idx": class_idx,
                "max_wait_minutes": MTS_MAX_WAIT_MINUTES[category],
                "triggered_rules": cat_violations,
                "all_violations": all_violations,
                "description": MTS_VITAL_THRESHOLDS[category]["description"],
            }

    # Fallback — wszystkie vital signs w normie
    return {
        "suggested_category": "Green",
        "suggested_class_idx": CLASS_NAMES.index("Green"),
        "max_wait_minutes": MTS_MAX_WAIT_MINUTES["Green"],
        "triggered_rules": [],
        "all_violations": all_violations,
        "description": "Wszystkie vital signs w normie",
    }


def check_consistency(
    ml_prediction: int,
    rule_prediction: int,
) -> dict[str, Any]:
    """
    Porównuje predykcję ML z regułową — zwraca poziom zgodności.

    Returns
    -------
    dict z 'agree' (bool), 'distance' (int), 'safer' (str — która strona jest bezpieczniejsza).
    """
    distance = ml_prediction - rule_prediction  # ujemne = ML pilniejszy
    if distance == 0:
        verdict = "Pełna zgodność — model i reguły MTS się zgadzają."
        safer = "both"
    elif distance < 0:
        verdict = "Model jest BARDZIEJ pilny niż reguły — overtriage (bezpieczniejszy)."
        safer = "ml"
    else:
        verdict = "⚠️ Model jest MNIEJ pilny niż reguły — POTENCJALNY UNDERTRIAGE."
        safer = "rule"

    return {
        "agree": distance == 0,
        "distance": distance,
        "safer_side": safer,
        "verdict": verdict,
        "ml_class": CLASS_NAMES[ml_prediction] if 0 <= ml_prediction < len(CLASS_NAMES) else "?",
        "rule_class": CLASS_NAMES[rule_prediction] if 0 <= rule_prediction < len(CLASS_NAMES) else "?",
    }


def explain_rule_decision(rule_result: dict) -> str:
    """Formatuje wynik regułowy jako tekst po polsku."""
    cat = rule_result["suggested_category"]
    desc = rule_result["description"]
    wait = rule_result["max_wait_minutes"]

    parts = [
        f"Klasyfikacja regułowa MTS: **{cat}** ({desc})",
        f"Max. czas oczekiwania: {wait} min",
    ]

    if rule_result["triggered_rules"]:
        parts.append("\nZadziałane reguły kliniczne:")
        for rule in rule_result["triggered_rules"]:
            if rule["type"] == "vital":
                parts.append(f"  • {rule['description']}: {rule['value']}")
            else:
                parts.append(f"  • {rule['description']}")
    else:
        parts.append("\nŻadna reguła krytyczna nie zadziałała — vital signs w normie.")

    return "\n".join(parts)
