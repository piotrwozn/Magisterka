"""
Preprocessing Yale EMMLC: feature engineering, imputacja, definicja zestawów cech.

Yale EMMLC ma 972 kolumny, które dzielimy na grupy semantyczne:
    - TRIAGE_VITALS  — 8 vital signs przy triażu
    - DEMOGRAPHICS   — wiek, płeć, rasa, ubezpieczenie
    - ARRIVAL        — tryb i czas przybycia
    - CHIEF_COMPLAINTS — 200 binarnych flag głównych skarg
    - PAST_MEDICAL   — historia chorób (AHRQ CCS, binarne)
    - MEDICATIONS    — leki w 48 kategoriach terapeutycznych
    - HISTORICAL_VITALS — vital signs z poprzednich wizyt
    - HISTORICAL_LABS  — wyniki badań laboratoryjnych
    - ED_USAGE       — historia wizyt na SOR

Zestawy cech do eksperymentów:
    - FEATURES_TRIAGE_ONLY — tylko to, co ma pielęgniarka na triażu
    - FEATURES_FULL        — pełny obraz (z historią EHR)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# Definicje grup cech
# ─────────────────────────────────────────
@dataclass
class FeatureGroups:
    """Grupy cech zidentyfikowane w datasecie."""

    triage_vitals: list[str] = field(default_factory=list)
    demographics: list[str] = field(default_factory=list)
    arrival: list[str] = field(default_factory=list)
    chief_complaints: list[str] = field(default_factory=list)
    past_medical: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    historical_vitals: list[str] = field(default_factory=list)
    historical_labs: list[str] = field(default_factory=list)
    ed_usage: list[str] = field(default_factory=list)
    imaging_history: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)

    @property
    def triage_only(self) -> list[str]:
        """Cechy dostępne w momencie triażu (bez historii EHR)."""
        return list(
            self.triage_vitals
            + self.demographics
            + self.arrival
            + self.chief_complaints
        )

    @property
    def full(self) -> list[str]:
        """Pełny zestaw cech."""
        return list(
            self.triage_only
            + self.past_medical
            + self.medications
            + self.historical_vitals
            + self.historical_labs
            + self.ed_usage
            + self.imaging_history
        )

    def summary(self) -> dict[str, int]:
        return {
            "triage_vitals": len(self.triage_vitals),
            "demographics": len(self.demographics),
            "arrival": len(self.arrival),
            "chief_complaints": len(self.chief_complaints),
            "past_medical": len(self.past_medical),
            "medications": len(self.medications),
            "historical_vitals": len(self.historical_vitals),
            "historical_labs": len(self.historical_labs),
            "ed_usage": len(self.ed_usage),
            "imaging_history": len(self.imaging_history),
            "other": len(self.other),
            "TOTAL_TRIAGE_ONLY": len(self.triage_only),
            "TOTAL_FULL": len(self.full),
        }


# ─────────────────────────────────────────
# Detekcja grup cech (heurystyki nazw kolumn Yale EMMLC)
# ─────────────────────────────────────────
TRIAGE_VITAL_HINTS = (
    "triage_sbp",
    "triage_dbp",
    "triage_pulse",
    "triage_resp",
    "triage_o2",
    "triage_temp",
    "triage_pain",
    # alternatywne nazwy
    "triage_vital_",
)

DEMOGRAPHIC_HINTS = (
    "age",
    "sex",
    "gender",
    "race",
    "ethnic",
    "lang",
    "insurance",
    "employment",
    "marital",
)

ARRIVAL_HINTS = (
    "arrivalmode",
    "arrivalmonth",
    "arrivalday",
    "arrivalhour",
    "arrival_year",
    "dep_name",
    "ed_location",
)

CHIEF_COMPLAINT_HINTS = ("cc_",)

PAST_MEDICAL_HINTS = ("pmh_", "pmhx_")

MEDICATION_HINTS = ("med_", "medication_")

HISTORICAL_VITAL_HINTS = ("prev_sbp", "prev_dbp", "prev_pulse", "prev_resp", "prev_o2", "prev_temp")

HISTORICAL_LAB_HINTS = ("lab_", "prev_lab")

ED_USAGE_HINTS = ("n_ed", "n_admit", "prev_dispo", "prev_visit", "n_surgeries")

IMAGING_HINTS = ("n_xray", "n_ct", "n_mri", "n_ekg", "n_us", "n_imaging")

EXCLUDE_FROM_FEATURES = {
    # ID i targety
    "esi",
    "mts_color",
    "mts_numeric",
    "disposition",
    "patient_id",
    "person_id",
    "encounter_id",
    "admit",
    "outcome",
}


def build_feature_groups(df: pd.DataFrame) -> FeatureGroups:
    """
    Identyfikuje grupy cech w DataFrame na podstawie prefiksów nazw kolumn.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    FeatureGroups
    """
    groups = FeatureGroups()
    columns = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]

    for col in columns:
        col_lower = col.lower()

        if col_lower.startswith(TRIAGE_VITAL_HINTS):
            groups.triage_vitals.append(col)
        elif col_lower.startswith(CHIEF_COMPLAINT_HINTS):
            groups.chief_complaints.append(col)
        elif col_lower.startswith(PAST_MEDICAL_HINTS):
            groups.past_medical.append(col)
        elif col_lower.startswith(MEDICATION_HINTS):
            groups.medications.append(col)
        elif col_lower.startswith(HISTORICAL_VITAL_HINTS):
            groups.historical_vitals.append(col)
        elif col_lower.startswith(HISTORICAL_LAB_HINTS):
            groups.historical_labs.append(col)
        elif col_lower.startswith(ED_USAGE_HINTS):
            groups.ed_usage.append(col)
        elif col_lower.startswith(IMAGING_HINTS):
            groups.imaging_history.append(col)
        elif col_lower.startswith(ARRIVAL_HINTS):
            groups.arrival.append(col)
        elif col_lower.startswith(DEMOGRAPHIC_HINTS):
            groups.demographics.append(col)
        else:
            groups.other.append(col)

    log.info("Zidentyfikowano grupy cech:")
    for name, n in groups.summary().items():
        log.info(f"  {name:25s}: {n:>5}")

    return groups


# Globalna instancja po preprocessing (wypełniana przez build_feature_groups)
FEATURE_GROUPS: FeatureGroups = FeatureGroups()


# ─────────────────────────────────────────
# Imputacja
# ─────────────────────────────────────────
def impute_missing(
    df: pd.DataFrame,
    groups: FeatureGroups,
    numeric_strategy: str = "median",
) -> pd.DataFrame:
    """
    Imputacja braków:
        - Vital signs: mediana (vital signs mają sens tylko numerycznie)
        - Cechy binarne (CC, PMH, MED): 0 (brak informacji = brak choroby)
        - Cechy numeryczne ciągłe (lab, historical vitals): mediana
        - Cechy kategoryczne: tryb (most frequent)
    """
    df = df.copy()

    # --- Vital signs — mediana ---
    if groups.triage_vitals:
        present = [c for c in groups.triage_vitals if c in df.columns]
        if present:
            log.debug(f"Imputacja vital signs (mediana): {len(present)} kolumn")
            num_imp = SimpleImputer(strategy=numeric_strategy)
            df[present] = num_imp.fit_transform(df[present])

    # --- Cechy binarne — 0 (brak informacji ≈ brak choroby) ---
    binary_cols = (
        groups.chief_complaints + groups.past_medical + groups.medications
    )
    binary_cols = [c for c in binary_cols if c in df.columns]
    if binary_cols:
        log.debug(f"Imputacja cech binarnych (0): {len(binary_cols)} kolumn")
        df[binary_cols] = df[binary_cols].fillna(0)

    # --- Historical vitals i labs — mediana ---
    historical = groups.historical_vitals + groups.historical_labs
    historical = [c for c in historical if c in df.columns]
    if historical:
        # Wybierz tylko kolumny numeryczne
        num_hist = [c for c in historical if pd.api.types.is_numeric_dtype(df[c])]
        if num_hist:
            log.debug(f"Imputacja historycznych (mediana): {len(num_hist)} kolumn")
            num_imp = SimpleImputer(strategy=numeric_strategy)
            df[num_hist] = num_imp.fit_transform(df[num_hist])

    # --- ED usage i imaging — 0 (brak wizyt = 0) ---
    counts = groups.ed_usage + groups.imaging_history
    counts = [c for c in counts if c in df.columns]
    if counts:
        log.debug(f"Imputacja licznikowych (0): {len(counts)} kolumn")
        df[counts] = df[counts].fillna(0)

    # --- Demographics, arrival — kategoryczne → most frequent ---
    cat_cols = groups.demographics + groups.arrival
    cat_cols = [c for c in cat_cols if c in df.columns]
    if cat_cols:
        for col in cat_cols:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

    return df


# ─────────────────────────────────────────
# Encoding kategoryczny
# ─────────────────────────────────────────
def encode_categoricals(
    df: pd.DataFrame,
    groups: FeatureGroups,
    method: str = "onehot",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Koduje cechy kategoryczne.

    Parameters
    ----------
    method : str
        'onehot' lub 'label'. Dla XGBoost z `enable_categorical=True`
        można pozostawić jako kategorie.

    Returns
    -------
    df_encoded : pd.DataFrame
    encoding_map : dict
        Mapowanie oryginalnej kolumny → nowych kolumn (do śledzenia).
    """
    df = df.copy()
    encoding_map: dict[str, list[str]] = {}

    cat_cols = groups.demographics + groups.arrival
    # Tylko obiektowe lub kategoryczne
    cat_cols = [
        c for c in cat_cols
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]

    if not cat_cols:
        log.debug("Brak kolumn kategorycznych do enkodowania")
        return df, encoding_map

    log.info(f"Enkodowanie kategorycznych ({method}): {len(cat_cols)} kolumn")

    if method == "onehot":
        before_cols = set(df.columns)
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False, dtype=np.uint8)
        after_cols = set(df.columns)

        # Mapowanie original → new
        for col in cat_cols:
            new_cols = [c for c in (after_cols - before_cols) if c.startswith(f"{col}_")]
            encoding_map[col] = new_cols

        # Zaktualizuj groups (one-hot powiększa demographics/arrival)
        new_demo = [c for c in df.columns if any(c.startswith(f"{d}_") for d in groups.demographics)]
        new_arr = [c for c in df.columns if any(c.startswith(f"{a}_") for a in groups.arrival)]

        groups.demographics = [c for c in groups.demographics if c in df.columns] + new_demo
        groups.arrival = [c for c in groups.arrival if c in df.columns] + new_arr

    elif method == "label":
        from sklearn.preprocessing import LabelEncoder

        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoding_map[col] = [col]
    else:
        raise ValueError(f"Nieznana metoda enkodowania: {method}")

    return df, encoding_map


# ─────────────────────────────────────────
# Główna funkcja preprocessing
# ─────────────────────────────────────────
def preprocess_dataframe(
    df: pd.DataFrame,
    encode_method: str = "label",
    impute: bool = True,
) -> tuple[pd.DataFrame, FeatureGroups]:
    """
    Pełen preprocessing: identyfikacja grup, imputacja, enkodowanie.

    Parameters
    ----------
    df : pd.DataFrame
        Surowy DataFrame z Yale EMMLC.
    encode_method : str
        'onehot' lub 'label' (XGBoost lubi label, RF/MLP wolą onehot).
    impute : bool
        Czy uzupełnić braki.

    Returns
    -------
    df : pd.DataFrame
        Przetworzony DataFrame.
    groups : FeatureGroups
        Zaktualizowane grupy cech.
    """
    log.info(f"Preprocessing: {df.shape[0]:,} wierszy × {df.shape[1]:,} kolumn")

    # 1. Identyfikuj grupy
    groups = build_feature_groups(df)

    # 2. Imputacja
    if impute:
        df = impute_missing(df, groups)

    # 3. Enkodowanie
    df, _ = encode_categoricals(df, groups, method=encode_method)

    log.info(f"Po preprocessing: {df.shape[0]:,} wierszy × {df.shape[1]:,} kolumn")

    # Zaktualizuj globalną instancję
    global FEATURE_GROUPS
    FEATURE_GROUPS = groups

    return df, groups


def split_features(
    df: pd.DataFrame,
    groups: FeatureGroups,
    feature_set: str = "triage_only",
    target: str = "mts_numeric",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Wybór zestawu cech do eksperymentu.

    Parameters
    ----------
    df : pd.DataFrame
    groups : FeatureGroups
    feature_set : str
        'triage_only' | 'full' | 'top'
    target : str

    Returns
    -------
    X : pd.DataFrame, y : pd.Series, feature_names : list[str]
    """
    if feature_set == "triage_only":
        feature_names = [c for c in groups.triage_only if c in df.columns]
    elif feature_set == "full":
        feature_names = [c for c in groups.full if c in df.columns]
    elif feature_set == "top":
        # ESI + demographics + ED usage + medications (zgodnie z TECHNICAL_ANALYSIS.md §2.6)
        feature_names = (
            groups.demographics + groups.ed_usage + groups.medications
        )
        feature_names = [c for c in feature_names if c in df.columns]
    else:
        raise ValueError(f"Nieznany feature_set: {feature_set}")

    if not feature_names:
        raise ValueError(f"Pusty feature set dla '{feature_set}'")

    if target not in df.columns:
        raise ValueError(f"Brak kolumny target '{target}' w DataFrame")

    X = df[feature_names].copy()
    y = df[target].copy()

    # Konwersja cech bool → int (XGBoost ich nie lubi)
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(np.uint8)

    log.info(f"Wybrano {len(feature_names)} cech (zestaw '{feature_set}')")
    return X, y, feature_names
