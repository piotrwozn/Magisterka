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
    "Police": "other",                     # 99 wizyt (0,02%) — osobna kolumna byłaby stała
    "Wheelchair": "other",
    "Other": "other",
}
ARRIVAL_CATEGORIES: list[str] = ["ambulance", "self", "other"]

# Brak sposobu przybycia w Yale (≈20,8 tys. wizyt, 99,7% z oddziału A) to ścieżka
# typu fast-track dla studentów i pracowników — wyznaczona ubezpieczeniem
# i zatrudnieniem w USA, nie kategoria kliniczna. Koduje się ją jak backend
# (DEFAULT_ARRIVAL_MODE): brak → transport własny.
ARRIVAL_MISSING_AS: str = "self"

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


# Skargi główne: aktywacje zespołu urazowego (pełna / zmodyfikowana) to dwustopniowy
# protokół oddziału A Yale (98% wizyt), nieodtwarzalny z polskich danych → jedna flaga.
TRAUMA_ACTIVATION_COL = "cc_trauma_activation"
TRAUMA_ACTIVATION_SOURCES: list[str] = ["cc_fulltrauma", "cc_modifiedtrauma"]


# ─────────────────────────────────────────
# Historia pacjenta (zestaw 'full')
# ─────────────────────────────────────────
ED_USAGE_COLS: list[str] = ["n_edvisits", "n_admissions", "n_surgeries"]
PREVIOUS_DISPO_RAW_COL = "previousdispo"

# Dyspozycja POPRZEDNIEJ wizyty (Yale `3_dispo_timeseries.R`) → 5 przenośnych kategorii.
# 'Observation' (status rozliczeniowy Medicare) i 'Send to L&D' (ścieżka położnicza USA)
# = przyjęcie; AMA / ucieczka / LWBS = opuszczenie SOR przed końcem (rzadkie poziomy,
# silnie skorelowane z ubezpieczeniem — scalone).
PREV_DISPO_MISSING = "No previous dispo"
PREV_DISPO_MAP: dict[str, str] = {
    "Admit": "admitted",
    "Observation": "admitted",
    "Send to L&D": "admitted",
    "Discharge": "discharged",
    "Transfer to Another Facility": "transferred",
    "AMA": "left_early",
    "Eloped": "left_early",
    "LWBS before Triage": "left_early",
    "LWBS after Triage": "left_early",
    PREV_DISPO_MISSING: "none",
}
PREV_DISPO_CATEGORIES: list[str] = ["admitted", "discharged", "transferred", "left_early", "none"]

# Historia chorób (CCS) wykluczona z zestawu 'full'
# Kody V „Administrative/social admission” (mieszkanie, sytuacja materialna, zatrudnienie):
# 71% Medicaid/self-pay vs 35% ogółem — wskaźnik statusu socjoekonomicznego.
SOCIOECONOMIC_PMH_COLS: list[str] = ["socialadmin"]
# Kody V opisujące powód wizyty (badanie kontrolne, przesiewowe, szczepienie, opieka
# po leczeniu), a nie chorobę.
ADMIN_CODE_PMH_COLS: list[str] = ["exameval", "otherscreen", "immunizscrn", "otaftercare"]
# < 50 pozytywnych w train (kody okołoporodowe/noworodkowe u dorosłych, kody E).
# Stała lista (nie liczona z danych) → ten sam zestaw cech przy każdym podziale.
LOW_SUPPORT_COLS: list[str] = [
    "ecodesmachinery", "ecodesfireburn", "septicemiaexceptinlabor",
    "ecodesdrowningsubmersion", "ecodesplaceofoccurrence", "umbilcord",
    "obrelatedperintrauma", "birthasphyx", "ecodessuffocation", "ecodespoisoning",
    "perintjaund", "liveborn", "abortcompl", "longpregncy", "fetaldistrs",
    "ecodesstruckbyagainst", "respdistres", "malposition", "pelvicobstr", "otrespirca",
    "ecodespedalcyclistnotmvt", "rehab", "birthtrauma", "ecodesnaturalenvironment",
    "ecodesadverseeffectsofmedicalcare", "ecodestransportnotmvt", "malgenitca",
    "poisonpsych",
]

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

# Badania bezużyteczne w Yale (wszystkie NaN, stałe, śmieciowe wartości, mieszane skale).
BROKEN_LAB_BASES: frozenset[str] = frozenset({
    "phencyclidine(pcp)screen,urine,noconf.",   # 100% NaN w train/val/test
    "benzodiazepinesscreen,urine,noconf.",      # 1–10 wartości, wszystkie od −512 do −55
    "epithelialcells",                          # ≤ 28 wartości; `_last` puste w val i test
    "acetonebld",                               # ≤ 1,8 tys. wartości w różnych skalach
    "pocph",                                    # pH moczu (paski) zmieszane z pH krwi
})
BROKEN_LAB_COLS: list[str] = [
    "bloodculture,routine_last", "bloodculture,routine_npos",   # stałe 0 mimo `_count > 0`
    "pregtestur_last", "pregtestur_npos",                       # pozytywne niekodowane
]
# Mieszane jednostki w jednej kolumnie (µg/mL FEU obok ng/mL; FEU ≈ 2 × DDU) — wykluczone.
UNIT_MIXED_LAB_BASES: frozenset[str] = frozenset({"d-dimer"})

# Badania jakościowe (`_last`, `_npos`, `_count`): Yale wpisał 0 także, gdy badania
# nie wykonano (`20_cleanmerged.R`) → `_last`/`_npos` = NaN, gdy `_count == 0`.
CATEGORICAL_LABS: list[str] = [
    "bloodua", "glucoseua", "ketonesua", "leukocytesua", "nitriteua", "proteinua",
    "urineculture,routine",
]

# Część wyników w komórkach/µL zamiast 10³/µL: wartość > próg → podzielona przez 1000.
LAB_UNIT_HARMONIZE: dict[str, float] = {
    "monosabs": 20.0,   # monocyty: norma 0,2–1,0 × 10³/µL
    "basosabs": 5.0,    # bazofile: norma 0–0,2 × 10³/µL
}

# Jednostki konwencjonalne USA w Yale (dokumentacja; zestaw 'triage' ich nie używa).
# Przed podaniem polskich wyników modelowi 'full' trzeba je przeliczyć do tych jednostek.
LAB_UNITS_US: dict[str, str] = {
    "glucose": "mg/dL (mmol/L = mg/dL / 18,0)",
    "creatinine": "mg/dL (µmol/L = mg/dL × 88,4)",
    "bun": "mg/dL azotu mocznikowego (mocznik mg/dL = BUN × 2,14; mmol/L = BUN × 0,357)",
    "calcium": "mg/dL (mmol/L = mg/dL × 0,25)",
    "magnesium": "mg/dL (mmol/L = mg/dL × 0,411)",
    "phosphorus": "mg/dL (mmol/L = mg/dL × 0,323)",
    "hemoglobin": "g/dL (mmol/L = g/dL × 0,621)",
    "troponin": "ng/mL, POC TnI i TnT — nieporównywalne z polskim hs-cTn (ng/L); nie mapować",
    "monosabs, basosabs": "10³/µL (po ujednoliceniu jednostek)",
}


def lab_base(col: str) -> str:
    """'d-dimer_last' → 'd-dimer' (nazwa badania bez sufiksu statystyki)."""
    for suffix in LAB_SUFFIXES:
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


def chief_complaint_cols(columns) -> list[str]:
    return [c for c in columns if c.startswith("cc_")]


def canonical_chief_complaints(columns) -> list[str]:
    """Skargi główne po scaleniu aktywacji urazowej (nazwy surowe, przed `sanitize_name`)."""
    raw = chief_complaint_cols(columns)
    out = [c for c in raw if c not in TRAUMA_ACTIVATION_SOURCES]
    if any(c in raw for c in TRAUMA_ACTIVATION_SOURCES):
        out.append(TRAUMA_ACTIVATION_COL)
    return out


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


def excluded_pmh_cols(columns) -> dict[str, list[str]]:
    """Kolumny PMH wykluczone z zestawu 'full' (powód → kolumny obecne w danych)."""
    pmh = set(pmh_cols(columns))
    return {
        "socioeconomic": [c for c in SOCIOECONOMIC_PMH_COLS if c in pmh],
        "admin_codes": [c for c in ADMIN_CODE_PMH_COLS if c in pmh],
        "low_support": [c for c in LOW_SUPPORT_COLS if c in pmh],
    }


def excluded_lab_cols(columns) -> dict[str, list[str]]:
    """Badania wykluczone z zestawu 'full' (powód → kolumny obecne w danych)."""
    labs = lab_cols(columns)
    return {
        "race_based": [c for c in labs if RACE_BASED_LAB_PATTERN.match(c)],
        "broken": [c for c in labs if lab_base(c) in BROKEN_LAB_BASES or c in BROKEN_LAB_COLS],
        "unit_mixed": [c for c in labs if lab_base(c) in UNIT_MIXED_LAB_BASES],
    }


# ─────────────────────────────────────────
# Zakresy wiarygodności (jednostki kanoniczne; poza zakresem → brak)
# ─────────────────────────────────────────
# Filtr Yale (`9_edvitalsclean.R`, x < min lub x ≥ max → NA) zastosowany PRZED
# udostępnieniem danych — dokumentacja do raportu z audytu.
YALE_UPSTREAM_VITAL_BOUNDS: dict[str, tuple[float, float]] = {
    "HR /min": (30.0, 300.0),
    "SBP mmHg": (30.0, 400.0),
    "DBP mmHg": (25.0, 300.0),
    "SpO2 %": (60.0, 100.0),
    "RR /min": (8.0, 70.0),
    "temperatura °F": (90.0, 110.0),
}
# Filtr Yale usunął każdy odczyt SpO2 = 100% (x ≥ 100 → NA). Wartość przywracana
# w triażu, gdy zmierzono HR i SBP (pulsoksymetr podaje też HR).
SPO2_RESTORED_VALUE: float = 100.0

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
