"""
Schemat danych Yale EMMLC — jawne listy kolumn i mapowania kategorii.

Zamiast zgadywać grupy cech po prefiksach (stare heurystyki nie trafiały
w prawdziwe nazwy: PMH nie ma prefiksu, leki to `meds_*`, historia vitals
to `pulse_last` itd.), ten moduł opisuje dokładnie, co jest czym.

Słownik danych: https://github.com/yaleemmlc/admissionprediction
"""

from __future__ import annotations

import re

import pandas as pd

# ─────────────────────────────────────────
# Kolumny meta (nigdy nie są cechami)
# ─────────────────────────────────────────
ESI_RAW_COL = "esi"
SITE_RAW_COL = "dep_name"          # oddział Yale: A / B / C
DISPOSITION_RAW_COL = "disposition"  # Admit / Discharge — wynik wizyty (przeciek!)

# Zmienne specyficzne dla USA — wykluczone: niedostępne w polskim SOR
# (rasa, wyznanie, ubezpieczenie, zatrudnienie …) i ryzyko uprzedzeń modelu.
US_SPECIFIC_COLS: list[str] = [
    "ethnicity",
    "race",
    "lang",
    "religion",
    "maritalstatus",
    "employstatus",
    "insurance_status",
]

# ─────────────────────────────────────────
# Triaż
# ─────────────────────────────────────────
TRIAGE_VITAL_COLS: list[str] = [
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_o2_device",
    "triage_vital_temp",
]

# Sposób przybycia → kategorie, które mają odpowiednik w polskim SOR
ARRIVAL_MODE_MAP: dict[str, str] = {
    "ambulance": "ambulance",              # ZRM / LPR
    "Car": "self",                         # transport własny
    "Walk-in": "self",
    "Public Transportation": "self",
    "Police": "police",
    "Wheelchair": "other",
    "Other": "other",
}
ARRIVAL_CATEGORIES: list[str] = ["ambulance", "self", "police", "other", "unknown"]

MONTH_MAP: dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

WEEKDAY_MAP: dict[str, int] = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
    "Friday": 5, "Saturday": 6, "Sunday": 7,
}

# Przedziały godzin przybycia w Yale → porządek chronologiczny 0–5
HOUR_BIN_MAP: dict[str, int] = {
    "23-02": 0,
    "03-06": 1,
    "07-10": 2,
    "11-14": 3,
    "15-18": 4,
    "19-22": 5,
}


def hour_to_bin(hour: int) -> int:
    """Godzina 0–23 → przedział Yale 0–5 (do inferencji w backendzie)."""
    hour = int(hour) % 24
    if hour >= 23 or hour <= 2:
        return 0
    return (hour - 3) // 4 + 1


# ─────────────────────────────────────────
# Historia pacjenta (zestaw 'full')
# ─────────────────────────────────────────
ED_USAGE_COLS: list[str] = ["n_edvisits", "n_admissions", "n_surgeries"]
PREVIOUS_DISPO_RAW_COL = "previousdispo"

IMAGING_COLS: list[str] = [
    "cxr_count", "echo_count", "ekg_count", "headct_count", "mri_count",
    "otherct_count", "otherimg_count", "otherus_count", "otherxr_count",
]

HISTORY_VITAL_PREFIXES: tuple[str, ...] = (
    "pulse_", "resp_", "spo2_", "temp_", "sbp_", "dbp_", "o2_device_",
)
HISTORY_STAT_SUFFIXES: tuple[str, ...] = ("_last", "_min", "_max", "_median")
LAB_SUFFIXES: tuple[str, ...] = ("_last", "_min", "_max", "_median", "_npos", "_count")

# eGFR liczony z rasą (USA, równania sprzed 2021) — wykluczony.
RACE_BASED_LAB_PATTERN = re.compile(r"^egfr")


def chief_complaint_cols(columns) -> list[str]:
    return [c for c in columns if c.startswith("cc_")]


def medication_cols(columns) -> list[str]:
    return [c for c in columns if c.startswith("meds_")]


def history_vital_cols(columns) -> list[str]:
    return [
        c for c in columns
        if c.startswith(HISTORY_VITAL_PREFIXES) and c.endswith(HISTORY_STAT_SUFFIXES)
    ]


def lab_cols(columns) -> list[str]:
    """Wyniki badań z poprzednich wizyt (bez vitals i licznika obrazowania)."""
    hist_vitals = set(history_vital_cols(columns))
    imaging = set(IMAGING_COLS)
    return [
        c for c in columns
        if c.endswith(LAB_SUFFIXES)
        and c not in hist_vitals
        and c not in imaging
        and not c.startswith("triage_vital")
    ]


def race_based_lab_cols(columns) -> list[str]:
    return [c for c in lab_cols(columns) if RACE_BASED_LAB_PATTERN.match(c)]


def pmh_cols(columns) -> list[str]:
    """
    Historia chorób (kategorie AHRQ CCS, binarne) — w Yale bez prefiksu.

    Wyznaczana przez wykluczenie wszystkich pozostałych znanych grup.
    """
    known = (
        {ESI_RAW_COL, SITE_RAW_COL, DISPOSITION_RAW_COL, PREVIOUS_DISPO_RAW_COL,
         "age", "gender", "arrivalmode", "arrivalmonth", "arrivalday", "arrivalhour_bin"}
        | set(US_SPECIFIC_COLS)
        | set(TRIAGE_VITAL_COLS)
        | set(ED_USAGE_COLS)
        | set(IMAGING_COLS)
        | set(chief_complaint_cols(columns))
        | set(medication_cols(columns))
        | set(history_vital_cols(columns))
        | set(lab_cols(columns))
    )
    return [c for c in columns if c not in known]


# ─────────────────────────────────────────
# Zakresy wiarygodności (jednostki kanoniczne; poza zakresem → brak)
# ─────────────────────────────────────────
VITAL_BOUNDS: dict[str, tuple[float, float]] = {
    "hr": (20.0, 300.0),      # /min
    "sbp": (40.0, 300.0),     # mmHg
    "dbp": (15.0, 200.0),     # mmHg
    "rr": (4.0, 80.0),        # /min
    "o2": (50.0, 100.0),      # SpO2 %
    "temp": (30.0, 44.0),     # °C
}

# Prefiks kolumny historii vitals → klucz w VITAL_BOUNDS
HISTORY_VITAL_KIND: dict[str, str] = {
    "pulse_": "hr",
    "resp_": "rr",
    "spo2_": "o2",
    "temp_": "temp",
    "sbp_": "sbp",
    "dbp_": "dbp",
}


def fahrenheit_to_celsius(values: pd.Series) -> pd.Series:
    return (values - 32.0) * 5.0 / 9.0


# ─────────────────────────────────────────
# Nazwy kolumn
# ─────────────────────────────────────────
_INVALID_CHARS = re.compile(r"[^0-9a-z_]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


def sanitize_name(name: str) -> str:
    """
    Nazwa bezpieczna dla XGBoost/LightGBM/CatBoost (bez , [ ] < > : itp.).

    'baseexcess,venous(poc)_last' → 'baseexcess_venous_poc_last'
    'cc_fall>65'                  → 'cc_fall_65'
    """
    out = _INVALID_CHARS.sub("_", str(name).lower())
    out = _MULTI_UNDERSCORE.sub("_", out).strip("_")
    return out


def sanitize_columns(columns) -> dict[str, str]:
    """Mapa oryginał → nazwa bezpieczna; gwarantuje unikalność."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in columns:
        base = sanitize_name(col)
        name = base
        i = 2
        while name in used:
            name = f"{base}_{i}"
            i += 1
        used.add(name)
        mapping[col] = name
    return mapping
