"""
Regułowa weryfikacja parametrów życiowych zgodna z ESI (sanity check dla modelu).

Nazwa modułu jest historyczna (wcześniejsza wersja projektu sprawdzała reguły
Manchester Triage System) — zachowana dla kompatybilności importów. Obecnie
moduł NIE wykonuje żadnego mapowania ESI ↔ MTS: wynik to poziom ESI 1–5
(indeks klasy 0–4), a kolor TOPSOR służy wyłącznie do wyświetlania.

Co sprawdzamy (algorytm ESI v4, tylko elementy możliwe do oceny z danych triażu):
    Punkt A — natychmiastowe zagrożenie życia → ESI 1
        skrajne parametry życiowe (SpO2 < 85%, SBP < 80, HR < 40 lub > 150,
        RR < 8 lub > 35, T < 32 °C lub ≥ 41 °C) oraz jawne dyskryminatory
        (zatrzymanie krążenia, brak reakcji, niewydolność oddechowa …).
    Punkt B — sytuacja wysokiego ryzyka → ESI 2
        hipotensja (SBP < 90), silny ból (≥ 7/10, jeśli podany), skargi
        wysokiego ryzyka (alert udarowy, pełny alert urazowy, myśli samobójcze …).
    Punkt D — strefa zagrożenia parametrów (dorośli) → rozważ ESI 2
        HR > 100, RR > 20, SpO2 < 92%.
    Punkt C (ESI 3 / 4 / 5) zależy od przewidywanej liczby zasobów — z samych
    parametrów życiowych NIE da się go ustalić, więc reguły nie wskazują wtedy
    konkretnego poziomu (wynik `suggested_class_idx = None`).

Zastosowania:
    1. Sanity check predykcji ML — czy model nie zaniża pilności pacjenta
       z alarmującymi parametrami (potencjalne niedotriażowanie).
    2. Warstwa wyjaśnień ("dlaczego rozważyć ESI 2? bo SpO2 < 92%").

Progi są kliniczne (ESI Implementation Handbook v4, NEWS2), NIE dopasowane
do danych. W Yale większość pacjentów ze skrajnymi parametrami ma ESI 2 —
decyzja o ESI 1 wymaga oceny potrzeby natychmiastowej interwencji, więc
rozbieżność "reguły ESI 1 vs model ESI 2" to sygnał do weryfikacji, nie błąd.

Wejście: słownik / wiersz w schemacie kanonicznym (triage_vital_hr, _sbp, _dbp,
_rr, _o2, _temp w °C, cc_*), akceptowane są też stare nazwy (triage_pulse,
triage_o2sat, …). Temperatura > 50 traktowana jest jako °F i przeliczana.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.utils.config import (
    CLASS_NAMES,
    CLASS_NAMES_PL,
    N_CLASSES,
    TOPSOR_COLORS,
    TOPSOR_COLORS_PL,
    TOPSOR_MAX_WAIT_MINUTES,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# Reguły
# ─────────────────────────────────────────
@dataclass(frozen=True)
class VitalRule:
    """Pojedyncza reguła progowa na parametrze życiowym."""

    vital: str                          # nazwa kanoniczna (triage_vital_*)
    test: Callable[[float], bool]
    description: str
    decision_point: str                 # punkt decyzyjny ESI: A | B | D
    advisory: bool = False              # True = "rozważ" (strefa zagrożenia), nie twardy próg


ESI_DANGER_ZONE_ADULT: dict[str, float] = {
    "hr_above": 100.0,
    "rr_above": 20.0,
    "spo2_below": 92.0,
}

# Reguły zgrupowane wg poziomu ESI (klucz = nazwa klasy z config.CLASS_NAMES)
_ESI1_RULES: list[VitalRule] = [
    VitalRule("triage_vital_o2", lambda x: x < 85, "krytyczna hipoksja (SpO2 < 85%)", "A"),
    VitalRule("triage_vital_sbp", lambda x: x < 80, "ciężka hipotensja / wstrząs (SBP < 80 mmHg)", "A"),
    VitalRule("triage_vital_hr", lambda x: x > 150, "skrajna tachykardia (HR > 150/min)", "A"),
    VitalRule("triage_vital_hr", lambda x: x < 40, "skrajna bradykardia (HR < 40/min)", "A"),
    VitalRule("triage_vital_rr", lambda x: x > 35, "skrajna tachypnoe (RR > 35/min)", "A"),
    VitalRule("triage_vital_rr", lambda x: x < 8, "bradypnoe / niewydolność oddychania (RR < 8/min)", "A"),
    VitalRule("triage_vital_temp", lambda x: x >= 41.0, "hipertermia zagrażająca życiu (T ≥ 41 °C)", "A"),
    VitalRule("triage_vital_temp", lambda x: x < 32.0, "głęboka hipotermia (T < 32 °C)", "A"),
]

_ESI2_RULES: list[VitalRule] = [
    VitalRule("triage_vital_sbp", lambda x: x < 90, "hipotensja (SBP < 90 mmHg) — sytuacja wysokiego ryzyka", "B"),
    VitalRule("triage_pain", lambda x: x >= 7, "silny ból (≥ 7/10)", "B"),
    VitalRule(
        "triage_vital_hr", lambda x: x > ESI_DANGER_ZONE_ADULT["hr_above"],
        "strefa zagrożenia: HR > 100/min", "D", advisory=True,
    ),
    VitalRule(
        "triage_vital_rr", lambda x: x > ESI_DANGER_ZONE_ADULT["rr_above"],
        "strefa zagrożenia: RR > 20/min", "D", advisory=True,
    ),
    VitalRule(
        "triage_vital_o2", lambda x: x < ESI_DANGER_ZONE_ADULT["spo2_below"],
        "strefa zagrożenia: SpO2 < 92%", "D", advisory=True,
    ),
]

# Jawne dyskryminatory (np. z formularza triażowego): nazwa → (poziom ESI, opis)
DISCRIMINATORS: dict[str, tuple[int, str]] = {
    "cardiac_arrest": (1, "zatrzymanie krążenia"),
    "airway_compromise": (1, "zagrożona drożność dróg oddechowych"),
    "breathing_inadequate": (1, "niewydolny oddech / bezdech"),
    "unresponsive": (1, "brak reakcji na bodźce (AVPU = U)"),
    "shock": (1, "objawy wstrząsu / hipoperfuzji"),
    "seizure_active": (1, "czynny napad drgawek"),
    "confusion": (2, "nowe zaburzenia świadomości / splątanie"),
    "severe_distress": (2, "silny ból lub cierpienie"),
    "high_risk": (2, "sytuacja wysokiego ryzyka w ocenie triażysty"),
}

# Skargi główne (cc_* z Yale), które same w sobie wskazują punkt A lub B
COMPLAINT_DISCRIMINATORS: dict[str, tuple[int, str]] = {
    "cc_cardiacarrest": (1, "skarga: zatrzymanie krążenia"),
    "cc_unresponsive": (1, "skarga: brak reakcji"),
    "cc_respiratorydistress": (2, "skarga: niewydolność oddechowa (rozważ ESI 1)"),
    "cc_strokealert": (2, "skarga: alert udarowy"),
    "cc_fulltrauma": (2, "skarga: pełny alert urazowy (rozważ ESI 1)"),
    "cc_suicidal": (2, "skarga: myśli samobójcze"),
    "cc_homicidal": (2, "skarga: zagrożenie dla innych"),
    "cc_overdose_intentional": (2, "skarga: celowe przedawkowanie"),
}


def _level_info(class_idx: int) -> dict[str, Any]:
    color = TOPSOR_COLORS[class_idx]
    return {
        "esi": class_idx + 1,
        "class_idx": class_idx,
        "class_name": CLASS_NAMES[class_idx],
        "label_pl": CLASS_NAMES_PL[class_idx],
        "color": color,
        "color_pl": TOPSOR_COLORS_PL[class_idx],
        "max_wait_minutes": TOPSOR_MAX_WAIT_MINUTES[color],
    }


# Opis poziomów ESI z regułami (nazwa zachowana dla kompatybilności).
# Klucze = config.CLASS_NAMES ("ESI1" … "ESI5"); kolor TOPSOR tylko do wyświetlania.
MTS_VITAL_THRESHOLDS: dict[str, dict[str, Any]] = {
    CLASS_NAMES[0]: {
        **_level_info(0),
        "decision_point": "A",
        "description": "Natychmiastowa interwencja ratująca życie",
        "vital_rules": _ESI1_RULES,
    },
    CLASS_NAMES[1]: {
        **_level_info(1),
        "decision_point": "B/D",
        "description": "Sytuacja wysokiego ryzyka lub parametry w strefie zagrożenia",
        "vital_rules": _ESI2_RULES,
    },
    CLASS_NAMES[2]: {
        **_level_info(2),
        "decision_point": "C",
        "description": "≥ 2 przewidywane zasoby (nie do ustalenia z parametrów życiowych)",
        "vital_rules": [],
    },
    CLASS_NAMES[3]: {
        **_level_info(3),
        "decision_point": "C",
        "description": "1 przewidywany zasób (nie do ustalenia z parametrów życiowych)",
        "vital_rules": [],
    },
    CLASS_NAMES[4]: {
        **_level_info(4),
        "decision_point": "C",
        "description": "Brak przewidywanych zasobów (nie do ustalenia z parametrów życiowych)",
        "vital_rules": [],
    },
}
ESI_VITAL_RULES = MTS_VITAL_THRESHOLDS


# ─────────────────────────────────────────
# Aliasy nazw parametrów (kanoniczne + stare nazwy)
# ─────────────────────────────────────────
VITAL_COL_ALIASES: dict[str, list[str]] = {
    "triage_vital_hr": ["triage_vital_hr", "triage_pulse", "pulse", "heart_rate", "hr"],
    "triage_vital_sbp": ["triage_vital_sbp", "triage_sbp", "sbp"],
    "triage_vital_dbp": ["triage_vital_dbp", "triage_dbp", "dbp"],
    "triage_vital_rr": ["triage_vital_rr", "triage_resp", "resp", "respiratory_rate", "rr"],
    "triage_vital_o2": ["triage_vital_o2", "triage_o2sat", "o2sat", "spo2"],
    "triage_vital_temp": ["triage_vital_temp", "triage_temp", "temp", "temperature"],
    "triage_pain": ["triage_pain", "pain"],
}
# Parametry wymagane do pełnej oceny (ból jest opcjonalny — brak w Yale)
CORE_VITALS: list[str] = [
    "triage_vital_hr", "triage_vital_sbp", "triage_vital_rr", "triage_vital_o2", "triage_vital_temp",
]

FAHRENHEIT_THRESHOLD: float = 50.0   # T > 50 nie istnieje w °C u żywego pacjenta → to °F


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f   # NaN → None


def _resolve_vital(patient_data: Mapping, canonical: str) -> float | None:
    """Wartość parametru (pierwszy niepusty alias); temperatura zawsze w °C."""
    for alias in VITAL_COL_ALIASES.get(canonical, [canonical]):
        if alias in patient_data:
            value = _as_float(patient_data[alias])
            if value is None:
                continue
            if canonical == "triage_vital_temp" and value > FAHRENHEIT_THRESHOLD:
                value = (value - 32.0) * 5.0 / 9.0
            return value
    return None


def _to_mapping(patient: Any) -> Mapping:
    """dict / pd.Series / 1-wierszowy DataFrame → mapping."""
    if isinstance(patient, Mapping):
        return patient
    if hasattr(patient, "iloc") and hasattr(patient, "columns"):     # DataFrame
        return patient.iloc[0].to_dict()
    if hasattr(patient, "to_dict"):                                  # Series
        return patient.to_dict()
    raise TypeError(f"Nieobsługiwany typ danych pacjenta: {type(patient)}")


# ─────────────────────────────────────────
# Główna funkcja
# ─────────────────────────────────────────
def rule_based_triage(
    patient_vitals: Any,
    discriminators: dict | None = None,
) -> dict[str, Any]:
    """
    Reguły ESI (punkty A, B, D) na parametrach życiowych i skargach głównych.

    Zwraca NAJPILNIEJSZY poziom wskazany przez reguły. Gdy żadna reguła nie
    zadziałała, parametry nie wymuszają ESI 1–2, a poziom 3–5 zależy od
    przewidywanych zasobów — wtedy `suggested_class_idx` = None.

    Parameters
    ----------
    patient_vitals : dict | pd.Series | pd.DataFrame (1 wiersz)
        np. {'triage_vital_sbp': 88, 'triage_vital_hr': 142, 'cc_chestpain': 1, ...}
    discriminators : dict, optional
        Jawne dyskryminatory {nazwa: True}, nazwy z `DISCRIMINATORS`.

    Returns
    -------
    dict z kluczami:
        - suggested_esi         (int 1–5 | None)
        - suggested_class_idx   (int 0–4 | None)
        - suggested_category    (str | None — kolor TOPSOR, do wyświetlania)
        - label_pl, color_pl
        - max_wait_minutes      (int | None — wg TOPSOR)
        - decision_point        ('A' | 'B' | 'D' | None)
        - advisory              (bool — True gdy wskazanie wynika wyłącznie ze strefy zagrożenia)
        - triggered_rules       (list[dict] — reguły wyznaczające poziom)
        - all_violations        (list[dict] — wszystkie spełnione reguły)
        - missing_vitals        (list[str]), notes (list[str]), description (str)
    """
    patient = _to_mapping(patient_vitals)
    discriminators = discriminators or {}
    violations: list[dict] = []

    # 1. Jawne dyskryminatory i skargi główne
    for name, flag in discriminators.items():
        if not flag:
            continue
        if name not in DISCRIMINATORS:
            log.warning(f"Nieznany dyskryminator '{name}' — pomijam")
            continue
        level, desc = DISCRIMINATORS[name]
        violations.append({
            "type": "discriminator", "name": name, "esi": level, "class_idx": level - 1,
            "decision_point": "A" if level == 1 else "B", "advisory": False, "description": desc,
        })
    for col, (level, desc) in COMPLAINT_DISCRIMINATORS.items():
        value = _as_float(patient.get(col))
        if value is not None and value > 0:
            violations.append({
                "type": "complaint", "name": col, "esi": level, "class_idx": level - 1,
                "decision_point": "A" if level == 1 else "B", "advisory": False, "description": desc,
            })

    # 2. Reguły progowe na parametrach życiowych
    for class_name in CLASS_NAMES:
        level_cfg = MTS_VITAL_THRESHOLDS[class_name]
        for rule in level_cfg["vital_rules"]:
            value = _resolve_vital(patient, rule.vital)
            if value is None:
                continue
            if rule.test(value):
                violations.append({
                    "type": "vital", "vital": rule.vital, "value": round(value, 2),
                    "esi": level_cfg["esi"], "class_idx": level_cfg["class_idx"],
                    "decision_point": rule.decision_point, "advisory": rule.advisory,
                    "description": rule.description,
                })

    # 3. Braki pomiarów i uwagi
    missing = [v for v in CORE_VITALS if _resolve_vital(patient, v) is None]
    notes: list[str] = []
    if len(missing) == len(CORE_VITALS):
        notes.append(
            "Brak wszystkich parametrów życiowych — u pacjentów ESI 1 częste "
            "(pacjent od razu na salę resuscytacyjną). Reguły nie mogą tego ocenić."
        )
    elif missing:
        notes.append("Brak pomiaru: " + ", ".join(missing))
    age = _as_float(patient.get("age"))
    if age is not None and age < 18:
        notes.append("Progi strefy zagrożenia ESI dotyczą dorosłych — u dzieci zależą od wieku.")

    if not violations:
        return {
            "suggested_esi": None,
            "suggested_class_idx": None,
            "suggested_category": None,
            "label_pl": "ESI 3–5 (zależnie od przewidywanych zasobów)",
            "color_pl": None,
            "max_wait_minutes": None,
            "decision_point": None,
            "advisory": False,
            "triggered_rules": [],
            "all_violations": [],
            "missing_vitals": missing,
            "notes": notes,
            "description": (
                "Parametry życiowe nie wskazują na ESI 1–2; poziom 3–5 zależy "
                "od przewidywanych zasobów (punkt decyzyjny C)."
            ),
        }

    best_idx = min(v["class_idx"] for v in violations)
    triggered = [v for v in violations if v["class_idx"] == best_idx]
    info = _level_info(best_idx)
    advisory = all(v["advisory"] for v in triggered)
    decision_point = sorted({v["decision_point"] for v in triggered})[0]
    return {
        "suggested_esi": info["esi"],
        "suggested_class_idx": best_idx,
        "suggested_category": info["color"],
        "label_pl": info["label_pl"],
        "color_pl": info["color_pl"],
        "max_wait_minutes": info["max_wait_minutes"],
        "decision_point": decision_point,
        "advisory": advisory,
        "triggered_rules": triggered,
        "all_violations": violations,
        "missing_vitals": missing,
        "notes": notes,
        "description": (
            "Strefa zagrożenia parametrów — rozważ ESI 2" if advisory
            else MTS_VITAL_THRESHOLDS[CLASS_NAMES[best_idx]]["description"]
        ),
    }


def check_consistency(
    ml_prediction: int,
    rule_prediction: int | None,
) -> dict[str, Any]:
    """
    Porównuje predykcję ML (indeks 0–4) z wynikiem reguł.

    `rule_prediction` = None oznacza, że reguły nie wskazały ESI 1–2
    (każda predykcja ESI 3–5 jest wtedy z nimi zgodna).

    Returns
    -------
    dict z 'agree' (bool), 'distance' (int | None, ML − reguły; dodatnie = model mniej pilny),
    'safer_side' ('both' | 'ml' | 'rule'), 'undertriage' (bool), 'verdict' (str),
    'ml_class', 'rule_class'.
    """
    ml_prediction = int(ml_prediction)
    ml_class = CLASS_NAMES[ml_prediction] if 0 <= ml_prediction < N_CLASSES else "?"

    if rule_prediction is None:
        if ml_prediction >= 2:
            verdict = (
                "Zgodność — parametry nie wymuszają ESI 1–2, a ESI 3–5 zależy "
                "od przewidywanych zasobów."
            )
            agree, safer = True, "both"
        else:
            verdict = (
                "Model jest BARDZIEJ pilny niż wynika z samych parametrów "
                "(np. skarga wysokiego ryzyka) — strona bezpieczna."
            )
            agree, safer = False, "ml"
        return {
            "agree": agree,
            "distance": None,
            "safer_side": safer,
            "undertriage": False,
            "verdict": verdict,
            "ml_class": ml_class,
            "rule_class": "ESI3–5",
        }

    rule_prediction = int(rule_prediction)
    distance = ml_prediction - rule_prediction   # ujemne = ML pilniejszy
    if distance == 0:
        verdict = "Pełna zgodność — model i reguły ESI wskazują ten sam poziom."
        safer = "both"
    elif distance < 0:
        verdict = "Model jest BARDZIEJ pilny niż reguły — przetriażowanie (strona bezpieczna)."
        safer = "ml"
    else:
        verdict = (
            f"⚠️ Model jest MNIEJ pilny niż reguły o {distance} poziom(y) — "
            "POTENCJALNE NIEDOTRIAŻOWANIE. Zweryfikuj pacjenta."
        )
        safer = "rule"

    return {
        "agree": distance == 0,
        "distance": distance,
        "safer_side": safer,
        "undertriage": distance > 0,
        "verdict": verdict,
        "ml_class": ml_class,
        "rule_class": CLASS_NAMES[rule_prediction] if 0 <= rule_prediction < N_CLASSES else "?",
    }


def explain_rule_decision(rule_result: dict) -> str:
    """Formatuje wynik regułowy jako tekst po polsku."""
    if rule_result.get("suggested_class_idx") is None:
        parts = [
            f"Weryfikacja regułowa ESI: **{rule_result['label_pl']}**",
            rule_result["description"],
        ]
    else:
        prefix = "rozważ " if rule_result.get("advisory") else ""
        parts = [
            f"Weryfikacja regułowa ESI: **{prefix}{rule_result['label_pl']}** "
            f"(punkt decyzyjny {rule_result['decision_point']})",
            f"Max. czas do kontaktu z lekarzem (TOPSOR): {rule_result['max_wait_minutes']} min",
        ]

    violations = rule_result.get("all_violations") or rule_result.get("triggered_rules") or []
    if violations:
        parts.append("\nSpełnione reguły:")
        for rule in sorted(violations, key=lambda r: r["class_idx"]):
            value = f": {rule['value']}" if rule.get("type") == "vital" else ""
            parts.append(f"  • [ESI {rule['esi']}, {rule['decision_point']}] {rule['description']}{value}")
    else:
        parts.append("\nŻadna reguła nie zadziałała — parametry poza strefą zagrożenia.")

    for note in rule_result.get("notes", []):
        parts.append(f"Uwaga: {note}")

    return "\n".join(parts)
