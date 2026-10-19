"""
Wspólne fixture'y i konfiguracja pytest.

Dane testowe są w pełni syntetyczne, ale mają DOKŁADNIE te nazwy kolumn
i typy, które pipeline czyta z surowego Yale EMMLC (kategorie tekstowe,
temperatura w °F, nazwy z przecinkami / nawiasami / '>' itd.).

Żaden test nie dotyka prawdziwych katalogów `data/`, `runs/` ani `logs/`:
    - `_isolate_logs` (autouse) przekierowuje logi treningu i JSON-y eksperymentów,
    - `isolated_paths` przekierowuje splity, manifest, raport z audytu, runs/, Optunę,
    - `prepared_paths` = `isolated_paths` + gotowe splity z syntetycznego Yale.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

# Pozwól testom na import `from src...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Testy zawsze na CPU (deterministycznie, także na stacji z GPU); można nadpisać zmienną.
os.environ.setdefault("SOR_AI_DEVICE", "cpu")

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────
# Syntetyczny surowy Yale
# ─────────────────────────────────────────
US_COLUMNS: dict[str, list[str]] = {
    "ethnicity": ["Hispanic or Latino", "Non-Hispanic"],
    "race": ["White or Caucasian", "Black or African American", "Asian"],
    "lang": ["English", "Spanish"],
    "religion": ["Catholic", "None"],
    "maritalstatus": ["Single", "Married"],
    "employstatus": ["Full Time", "Retired"],
    "insurance_status": ["Commercial", "Medicare", "Self pay"],
}

ARRIVAL_VALUES = ["ambulance", "Car", "Walk-in", "Police", "Wheelchair",
                  "Public Transportation", "Other"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOUR_BINS = ["23-02", "03-06", "07-10", "11-14", "15-18", "19-22"]
# Wszystkie 10 wartości z Yale (prepare scala je do 5 kategorii)
PREVIOUS_DISPO = [
    "No previous dispo", "Discharge", "Admit", "Observation", "Send to L&D",
    "Transfer to Another Facility", "AMA", "Eloped", "LWBS before Triage", "LWBS after Triage",
]

# Skargi główne — część wymaga normalizacji nazwy ('>' i '/'); dwie aktywacje urazowe
# są scalane do `cc_trauma_activation`
CC_COLUMNS = [
    "cc_chestpain", "cc_shortnessofbreath", "cc_abdominalpain", "cc_alteredmentalstatus",
    "cc_suture/stapleremoval", "cc_fall>65", "cc_bleeding/bruising",
    "cc_strokealert", "cc_fulltrauma", "cc_modifiedtrauma",
]
# Historia chorób (AHRQ CCS) — w Yale BEZ prefiksu (także nazwa zaczynająca się od cyfry)
PMH_COLUMNS = ["2ndarymalig", "acutemi", "asthma", "chfnonhp"]
# PMH wykluczone: socjoekonomiczne, administracyjne kody V, małe wsparcie
EXCLUDED_PMH_COLUMNS = ["socialadmin", "exameval", "liveborn"]
MEDS_COLUMNS = ["meds_analgesics", "meds_anti-obesitydrugs", "meds_antibiotics"]
IMAGING_COLUMNS = [
    "cxr_count", "echo_count", "ekg_count", "headct_count", "mri_count",
    "otherct_count", "otherimg_count", "otherus_count", "otherxr_count",
]
# Badania z poprzednich wizyt; eGFR liczony z rasą musi zostać wykluczony
LAB_COLUMNS = [
    "baseexcess,venous(poc)_last", "glucose,meter_last", "troponini(poc)_max",
    "egfr_last", "egfr(aframer)_last", "egfr(nonafricanamerican)_median",
]
RACE_LAB_COLUMNS = ["egfr_last", "egfr(aframer)_last", "egfr(nonafricanamerican)_median"]
# Zepsute badania i D-dimer (mieszane jednostki) — wykluczane
BROKEN_LAB_COLUMNS = [
    "pocph_last", "epithelialcells_max", "phencyclidine(pcp)screen,urine,noconf._last",
    "bloodculture,routine_last", "bloodculture,routine_npos",
]
UNIT_MIXED_LAB_COLUMNS = ["d-dimer_median"]
# Badania jakościowe: _last/_npos = 0 także, gdy nie badano (_count == 0)
CATEGORICAL_LAB_COLUMNS = ["bloodua_count", "bloodua_last", "bloodua_npos",
                           "bloodculture,routine_count"]
# Część wyników w komórkach/µL zamiast 10³/µL
UNIT_LAB_COLUMNS = ["monosabs_last", "basosabs_max"]

# Indeksy wierszy, które fixture gwarantuje jako "czyste" (z ESI, bez duplikatów) —
# testy mogą je modyfikować i szukać pod tym samym row_id po przygotowaniu.
N_CLEAN_HEAD_ROWS = 20


def make_raw_yale(n: int = 800, seed: int = 42, n_missing_esi: int = 7, n_duplicates: int = 4) -> pd.DataFrame:
    """
    Mały syntetyczny SUROWY Yale (te same nazwy kolumn i typy co prawdziwy parquet).

    - ESI jako kategoria '1'…'5' z brakami (`n_missing_esi` wierszy o indeksie ≥ 20),
    - na końcu `n_duplicates` dokładnych kopii wcześniejszych wierszy,
    - wiersze 0…19 zawsze mają ESI i nie są duplikatami,
    - temperatura (triaż i historia) w °F, część parametrów życiowych brakująca,
    - jak w Yale: SpO2 = 100 skasowane (filtr x ≥ 100), skargi jako liczniki (0–2),
      brak sposobu przybycia, policja, badania jakościowe z 0 zamiast „nie badano”,
      monocyty/bazofile częściowo w komórkach/µL, kolumny wykluczane (leki, socialadmin…),
    - słaby sygnał: niższy ESI → wyższe tętno / niższa saturacja, częściej karetka.
    """
    rng = np.random.default_rng(seed)
    esi = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.08, 0.25, 0.37, 0.20, 0.10])
    # Każdy poziom ESI na każdym oddziale (stratyfikacja ESI × oddział w podziale)
    site = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])

    df = pd.DataFrame({
        "dep_name": site,
        "esi": esi.astype(str),
        "age": rng.integers(18, 100, n),
        "gender": rng.choice(["Male", "Female"], n),
    })
    for col, values in US_COLUMNS.items():
        df[col] = rng.choice(values, n)
    df["disposition"] = np.where(rng.random(n) < (0.7 - 0.12 * esi), "Admit", "Discharge")

    p_ambulance = np.clip(0.7 - 0.12 * esi, 0.05, 0.9)
    arrival = np.where(
        rng.random(n) < p_ambulance, "ambulance",
        rng.choice(["Car", "Walk-in", "Police", "Wheelchair", "Public Transportation", "Other"],
                   n, p=[0.45, 0.35, 0.05, 0.05, 0.05, 0.05]),
    ).astype(object)
    arrival[rng.random(n) < 0.05] = None
    df["arrivalmode"] = arrival
    df["arrivalmonth"] = rng.choice(MONTHS, n)
    df["arrivalday"] = rng.choice(WEEKDAYS, n)
    df["arrivalhour_bin"] = rng.choice(HOUR_BINS, n)
    df["previousdispo"] = rng.choice(PREVIOUS_DISPO, n)

    for col in PMH_COLUMNS + EXCLUDED_PMH_COLUMNS:
        df[col] = rng.integers(0, 2, n).astype(float)

    # --- Parametry życiowe z triażu (temperatura w °F) ---
    vitals = pd.DataFrame({
        "triage_vital_hr": rng.normal(112 - 7 * esi, 15, n).round(),
        "triage_vital_sbp": rng.normal(130, 20, n).round(),
        "triage_vital_dbp": rng.normal(78, 10, n).round(),
        "triage_vital_rr": rng.normal(23 - 1.2 * esi, 3, n).round(),
        "triage_vital_o2": np.clip(rng.normal(92 + 1.3 * esi, 2.5, n), 70, 100).round(),
        "triage_vital_o2_device": (rng.random(n) < 0.1).astype(float),
        "triage_vital_temp": rng.normal(98.2, 1.0, n).round(1),
    })
    vitals = vitals.mask(rng.random(vitals.shape) < 0.15)
    # Filtr Yale (`9_edvitalsclean.R`): SpO2 ≥ 100 → brak
    vitals.loc[vitals["triage_vital_o2"] >= 100, "triage_vital_o2"] = np.nan
    # ESI 1: częsty brak WSZYSTKICH pomiarów (pacjent od razu na salę resuscytacyjną)
    no_vitals = (esi == 1) & (rng.random(n) < 0.5)
    vitals.loc[no_vitals, :] = np.nan
    df = pd.concat([df, vitals], axis=1)

    # Skargi jak w Yale: liczniki wpisów (czasem 2), czasem brak
    for col in CC_COLUMNS:
        df[col] = (rng.random(n) < 0.15).astype(float) * rng.choice([1.0, 2.0], n, p=[0.9, 0.1])
    df.loc[rng.random(n) < 0.03, CC_COLUMNS[0]] = np.nan

    for col in MEDS_COLUMNS:
        df[col] = rng.integers(0, 3, n).astype(float)
    df["n_edvisits"] = rng.integers(0, 10, n).astype(np.int32)
    df["n_admissions"] = rng.integers(0, 5, n).astype(np.int32)
    df["n_surgeries"] = rng.integers(0, 3, n).astype(np.int32)
    for col in IMAGING_COLUMNS:
        df[col] = rng.integers(0, 3, n).astype(float)

    # --- Historia vitals (temperatura w °F) ---
    df["pulse_last"] = rng.normal(85, 15, n).round()
    df["temp_max"] = rng.normal(99.0, 1.2, n).round(1)
    df["spo2_min"] = np.clip(rng.normal(95, 3, n), 70, 99).round()
    df["sbp_median"] = rng.normal(128, 15, n).round()
    df["o2_device_last"] = (rng.random(n) < 0.08).astype(float)
    for col in ["pulse_last", "temp_max", "spo2_min", "sbp_median", "o2_device_last"]:
        df.loc[rng.random(n) < 0.3, col] = np.nan

    for col in LAB_COLUMNS + BROKEN_LAB_COLUMNS + UNIT_MIXED_LAB_COLUMNS:
        values = rng.normal(50, 20, n)
        values[rng.random(n) < 0.6] = np.nan
        df[col] = values

    # Badania jakościowe: Yale wypełnia NA → 0 (także gdy badania nie wykonano)
    count = rng.choice([0, 1, 2, 3], n, p=[0.6, 0.2, 0.1, 0.1]).astype(float)
    tested = count > 0
    df["bloodua_count"] = count
    df["bloodua_last"] = np.where(tested, (rng.random(n) < 0.3).astype(float), 0.0)
    df["bloodua_npos"] = np.where(tested, np.minimum(count, rng.integers(0, 2, n)), 0.0)
    df["bloodculture,routine_count"] = rng.choice([0.0, 1.0], n, p=[0.9, 0.1])

    # Monocyty / bazofile w 10³/µL, część w komórkach/µL (× 1000)
    in_cells = rng.random(n) < 0.1
    df["monosabs_last"] = np.round(rng.uniform(0.2, 1.0, n) * np.where(in_cells, 1000, 1), 3)
    df["basosabs_max"] = np.round(rng.uniform(0.0, 0.2, n) * np.where(in_cells, 1000, 1), 3)

    # --- Braki ESI i duplikaty ---
    missing_idx = rng.choice(np.arange(N_CLEAN_HEAD_ROWS, n), size=n_missing_esi, replace=False)
    df.loc[missing_idx, "esi"] = None
    candidates = np.setdiff1d(np.arange(N_CLEAN_HEAD_ROWS, n), missing_idx)
    dup_idx = rng.choice(candidates, size=n_duplicates, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Typy jak w parquet z R: kolumny tekstowe jako `category`
    categories = {
        "dep_name": ["A", "B", "C"],
        "esi": ["1", "2", "3", "4", "5"],
        "gender": ["Female", "Male"],
        "disposition": ["Admit", "Discharge"],
        "arrivalmode": ARRIVAL_VALUES,
        "arrivalmonth": MONTHS,
        "arrivalday": WEEKDAYS,
        "arrivalhour_bin": HOUR_BINS,
        "previousdispo": PREVIOUS_DISPO,
        **US_COLUMNS,
    }
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col].astype(object), categories=cats)
    return df


@pytest.fixture
def raw_yale() -> pd.DataFrame:
    """Świeża kopia syntetycznego surowego Yale (800 wizyt + 4 duplikaty)."""
    return make_raw_yale()


@pytest.fixture(scope="session")
def make_raw():
    """Fabryka syntetycznego Yale z parametrami (n, seed, n_missing_esi, n_duplicates)."""
    return make_raw_yale


# ─────────────────────────────────────────
# Mały numeryczny zbiór do testów modeli
# ─────────────────────────────────────────
def make_numeric_xy(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """X (float32, z NaN) + y 0–4 (każda klasa obecna, słaby sygnał)."""
    rng = np.random.default_rng(seed)
    y = np.resize(np.arange(5), n)
    rng.shuffle(y)
    X = pd.DataFrame({
        "triage_vital_hr": rng.normal(110 - 7 * y, 12, n),
        "triage_vital_sbp": rng.normal(130, 20, n),
        "triage_vital_rr": rng.normal(23 - 1.2 * y, 3, n),
        "triage_vital_o2": np.clip(rng.normal(92 + 1.3 * y, 2.5, n), 70, 100),
        "triage_vital_temp": rng.normal(37.0, 0.6, n),
        "age": rng.integers(18, 95, n).astype(float),
        "arrival_ambulance": (rng.random(n) < 0.6 - 0.1 * y).astype(float),
        "cc_chestpain": (rng.random(n) < 0.2).astype(float),
        "cc_fall_65": (rng.random(n) < 0.1).astype(float),
        "news2_score": rng.integers(0, 10, n).astype(float),
    }).astype(np.float32)
    for col in ["triage_vital_hr", "triage_vital_sbp", "triage_vital_rr", "triage_vital_temp"]:
        X.loc[rng.random(n) < 0.15, col] = np.nan
    return X, pd.Series(y, name="target")


@pytest.fixture(scope="session")
def make_xy():
    """Fabryka numerycznego zbioru (n, seed) — do fixture'ów o zasięgu modułu."""
    return make_numeric_xy


@pytest.fixture
def synthetic_X_y() -> tuple[pd.DataFrame, pd.Series]:
    """Numeryczny zbiór 400 × 10 z brakami, klasy 0–4 (= ESI 1–5)."""
    return make_numeric_xy()


# ─────────────────────────────────────────
# Izolacja ścieżek
# ─────────────────────────────────────────
@pytest.fixture(autouse=True)
def _isolate_logs(tmp_path_factory, monkeypatch):
    """Każdy test zapisuje logi treningu, JSON-y eksperymentów i modele do katalogu tymczasowego."""
    from src.models import base
    from src.utils import config

    logs = tmp_path_factory.mktemp("logs")
    monkeypatch.setattr(config, "TRAINING_LOGS_DIR", logs / "training")
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", logs / "experiments")
    monkeypatch.setattr(config, "MODELS_DIR", logs / "models")
    monkeypatch.setattr(base, "MODELS_DIR", logs / "models")


@pytest.fixture(scope="session", autouse=True)
def _single_thread_models():
    """
    Modele w testach na jednym wątku.

    Na kilkuset wierszach wielowątkowość nic nie daje, a przy obciążonym CPU
    (równoległe procesy) OpenMP w LightGBM/XGBoost potrafi zwolnić trening ~100×.
    Jeden wątek daje też deterministyczne prawdopodobieństwa Random Forest.
    """
    from src.utils import config

    with pytest.MonkeyPatch.context() as mp:
        for params, key in [
            (config.XGB_DEFAULT_PARAMS, "n_jobs"),
            (config.LGBM_DEFAULT_PARAMS, "n_jobs"),
            (config.CATBOOST_DEFAULT_PARAMS, "thread_count"),
            (config.RF_DEFAULT_PARAMS, "n_jobs"),
            (config.EBM_DEFAULT_PARAMS, "n_jobs"),
        ]:
            mp.setitem(params, key, 1)
        yield


def _redirect_defaults(monkeypatch, func, mapping: dict) -> None:
    """Domyślne argumenty funkcji (wiązane przy definicji) → nowe ścieżki."""
    defaults = func.__defaults__
    if not defaults:
        return
    new = tuple(mapping.get(d, d) if isinstance(d, Path) else d for d in defaults)
    if new != defaults:
        monkeypatch.setattr(func, "__defaults__", new)


def _patch_paths(monkeypatch, data_dir: Path, work_dir: Path) -> SimpleNamespace:
    """Przekierowuje wszystkie stałe ścieżek używane przez pipeline (patrz moduły źródłowe)."""
    from src.data import dataset, load_data, prepare
    from src.ensemble import runs as ensemble_runs
    from src.models import train, tuning
    from src.utils import config

    paths = SimpleNamespace(
        ESI_DATA_DIR=data_dir,
        TRAIN_PARQUET=data_dir / "train.parquet",
        VAL_PARQUET=data_dir / "val.parquet",
        TEST_PARQUET=data_dir / "test.parquet",
        FEATURE_SETS_JSON=data_dir / "feature_sets.json",
        DATA_MANIFEST_JSON=data_dir / "manifest.json",
        DATA_AUDIT_MD=work_dir / "DATA_AUDIT.md",
        RUNS_DIR=work_dir / "runs",
        LEADERBOARD_CSV=work_dir / "runs" / "leaderboard.csv",
        TRAINING_LOGS_DIR=work_dir / "logs" / "training",
        EXPERIMENTS_DIR=work_dir / "logs" / "experiments",
        OPTUNA_STORAGE_DIR=work_dir / "logs" / "optuna",
    )
    old_to_new = {getattr(config, name): value for name, value in vars(paths).items()
                  if hasattr(config, name)}

    for module in (config, dataset, prepare, train, tuning, ensemble_runs):
        for name, value in vars(paths).items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, value)

    # Słownik splitów i domyślne argumenty są wiązane przy imporcie modułu
    monkeypatch.setattr(dataset, "SPLIT_PATHS", {
        "train": paths.TRAIN_PARQUET, "val": paths.VAL_PARQUET, "test": paths.TEST_PARQUET,
    })
    for func in (prepare.write_audit_report, ensemble_runs.load_run,
                 ensemble_runs.list_runs, ensemble_runs.latest_runs):
        _redirect_defaults(monkeypatch, func, old_to_new)

    # Bezpiecznik: test nigdy nie może wczytać pełnego Yale (~4 GB RAM)
    def _no_full_data(*args, **kwargs):
        raise RuntimeError("Testy nie mogą wczytywać pełnego zbioru Yale")

    monkeypatch.setattr(load_data, "load_dataset", _no_full_data)

    dataset._feature_sets_payload.cache_clear()
    return paths


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Wszystkie ścieżki danych / runów / logów w `tmp_path` (puste — bez splitów)."""
    from src.data import dataset

    paths = _patch_paths(monkeypatch, tmp_path / "data" / "esi", tmp_path)
    yield paths
    dataset._feature_sets_payload.cache_clear()


@pytest.fixture(scope="session")
def prepared_esi_dir(tmp_path_factory) -> Path:
    """Raz na sesję: syntetyczny Yale → splity ESI (podział losowy) w katalogu tymczasowym."""
    from src.data import dataset
    from src.data.prepare import prepare_dataset

    root = tmp_path_factory.mktemp("prepared")
    with pytest.MonkeyPatch.context() as mp:
        _patch_paths(mp, root / "esi", root)
        prepare_dataset(strategy="random", raw=make_raw_yale(n=1000, seed=7))
    dataset._feature_sets_payload.cache_clear()
    return root / "esi"


@pytest.fixture
def prepared_paths(prepared_esi_dir, tmp_path, monkeypatch):
    """Jak `isolated_paths`, ale z gotowymi splitami (kopia — test może je modyfikować)."""
    from src.data import dataset

    data_dir = tmp_path / "data" / "esi"
    shutil.copytree(prepared_esi_dir, data_dir)
    paths = _patch_paths(monkeypatch, data_dir, tmp_path)
    yield paths
    dataset._feature_sets_payload.cache_clear()
