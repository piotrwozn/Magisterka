"""
Konfiguracja globalna projektu SOR-AI.

Wszystkie ścieżki, stałe, hiperparametry i progi kliniczne zdefiniowane
w jednym miejscu, by uniknąć rozproszenia "magic numbers".

Etykieta: natywny Emergency Severity Index (ESI 1–5) — ten sam algorytm,
który polskie SOR-y stosują w systemie TOPSOR (Dz.U. 2021 poz. 1182, § 2 pkt 9).
Brak jakiegokolwiek mapowania na MTS.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────
# ŚCIEŻKI
# ─────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
DATA_EXTERNAL_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"
REPORTS_DIR: Path = RESULTS_DIR / "reports"

# Każdy trening zapisuje komplet artefaktów do runs/<run_name>/
# (model, metryki, predykcje val/test do ręcznego łączenia modeli).
RUNS_DIR: Path = PROJECT_ROOT / "runs"

LOGS_DIR: Path = PROJECT_ROOT / "logs"
TRAINING_LOGS_DIR: Path = LOGS_DIR / "training"     # logi z treningu modeli
EVAL_LOGS_DIR: Path = LOGS_DIR / "evaluation"       # logi z ewaluacji
EXPERIMENTS_DIR: Path = LOGS_DIR / "experiments"    # JSON-owe metadane eksperymentów

# MLflow tracking
MLFLOW_TRACKING_URI: str = f"file://{(PROJECT_ROOT / 'mlruns').as_posix()}"
MLFLOW_EXPERIMENT_NAME: str = "sor-ai-triage-esi"

# Surowe dane Yale EMMLC
RDATA_FILE: Path = DATA_RAW_DIR / "5v_cleandf.rdata"
RAW_PARQUET: Path = DATA_PROCESSED_DIR / "yale_emmlc_raw.parquet"

# Przygotowane dane ESI (wynik `scripts/02_preprocess.py`)
ESI_DATA_DIR: Path = DATA_PROCESSED_DIR / "esi"
TRAIN_PARQUET: Path = ESI_DATA_DIR / "train.parquet"
VAL_PARQUET: Path = ESI_DATA_DIR / "val.parquet"
TEST_PARQUET: Path = ESI_DATA_DIR / "test.parquet"
FEATURE_SETS_JSON: Path = ESI_DATA_DIR / "feature_sets.json"
DATA_MANIFEST_JSON: Path = ESI_DATA_DIR / "manifest.json"

# Raport z audytu danych (wersjonowany w repo — mały plik tekstowy)
DATA_AUDIT_MD: Path = PROJECT_ROOT / "DATA_AUDIT.md"

# Wersja logiki przygotowania danych. Zmień, gdy zmieniasz prepare/engineering —
# trening odmówi użycia splitów przygotowanych starszą wersją.
DATA_VERSION: str = "esi-v1"

# Utwórz katalogi przy imporcie
for _dir in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_EXTERNAL_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    TRAINING_LOGS_DIR,
    EVAL_LOGS_DIR,
    EXPERIMENTS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# REPRODUKOWALNOŚĆ
# ─────────────────────────────────────────
RANDOM_SEED: int = 42

# ─────────────────────────────────────────
# SPRZĘT
# ─────────────────────────────────────────
# 'auto' = GPU jeśli dostępne (nvidia-smi), inaczej CPU. Nadpisz: SOR_AI_DEVICE=cpu|gpu
DEVICE: str = os.environ.get("SOR_AI_DEVICE", "auto").lower()

# ─────────────────────────────────────────
# ETYKIETA — ESI (bez mapowania)
# ─────────────────────────────────────────
ESI_COL: str = "esi"          # oryginalny poziom ESI 1–5
TARGET_COL: str = "target"    # indeks klasy 0–4 (0 = ESI 1, ..., 4 = ESI 5)
N_CLASSES: int = 5

# Nazwy klas w kolejności indeksów 0–4
CLASS_NAMES: list[str] = ["ESI1", "ESI2", "ESI3", "ESI4", "ESI5"]

# Kolory TOPSOR przypisane poziomom ESI (ESI 1 = czerwony, …, ESI 5 = niebieski)
TOPSOR_COLORS: list[str] = ["Red", "Orange", "Yellow", "Green", "Blue"]
TOPSOR_COLORS_PL: list[str] = ["czerwony", "pomarańczowy", "żółty", "zielony", "niebieski"]

CLASS_NAMES_PL: list[str] = [
    "ESI 1 · czerwony (natychmiast)",
    "ESI 2 · pomarańczowy (do 10 min)",
    "ESI 3 · żółty (do 60 min)",
    "ESI 4 · zielony (do 120 min)",
    "ESI 5 · niebieski (do 240 min)",
]

# Maksymalny czas do pierwszego kontaktu z lekarzem w TOPSOR (minuty), wg koloru
TOPSOR_MAX_WAIT_MINUTES: dict[str, int] = {
    "Red": 0,
    "Orange": 10,
    "Yellow": 60,
    "Green": 120,
    "Blue": 240,
}
# Alias dla modułów wyjaśnień, które posługują się kolorami
MTS_MAX_WAIT_MINUTES = TOPSOR_MAX_WAIT_MINUTES

# Klasy wysokiej pilności (do metryk bezpieczeństwa): ESI 1–2
HIGH_ACUITY_CLASSES: tuple[int, ...] = (0, 1)
# Klasy niskiej pilności: ESI 4–5
LOW_ACUITY_CLASSES: tuple[int, ...] = (3, 4)

# ─────────────────────────────────────────
# WAGI KLAS (opcjonalne — domyślnie trenujemy BEZ wag)
# ─────────────────────────────────────────
# Domyślna strategia 'none' daje skalibrowane prawdopodobieństwa, które najlepiej
# nadają się do ręcznego łączenia modeli. Asymetrię kosztów (niedotriażowanie
# gorsze niż przetriażowanie) realizuje reguła decyzyjna z macierzy kosztów
# poniżej — działa tak samo dla pojedynczego modelu i dla ensemble.
DEFAULT_SAMPLE_WEIGHT_STRATEGY: str = "none"

# Strategia 'custom' — ręczne wagi (rozkład ESI w Yale: ~0.9 / 29 / 42 / 22 / 5 %)
CUSTOM_CLASS_WEIGHTS: dict[int, float] = {
    0: 10.0,   # ESI 1 — rzadka, krytyczna
    1: 2.0,    # ESI 2 — pominięcie = niebezpieczne niedotriażowanie
    2: 1.0,    # ESI 3 — baseline
    3: 1.0,    # ESI 4
    4: 2.0,    # ESI 5 — mała klasa
}

# ─────────────────────────────────────────
# MACIERZ KOSZTÓW (decyzja kosztowa na prawdopodobieństwach)
# ─────────────────────────────────────────
# koszt[prawda, predykcja] = |p - t|^2 × (UNDER jeśli p > t, OVER jeśli p < t)
# p > t oznacza przewidzenie NIŻSZEJ pilności niż prawdziwa (niedotriażowanie).
UNDERTRIAGE_COST_MULTIPLIER: float = 3.0
OVERTRIAGE_COST_MULTIPLIER: float = 1.0

# ─────────────────────────────────────────
# PODZIAŁ DANYCH
# ─────────────────────────────────────────
# Yale EMMLC nie ma roku ani identyfikatora pacjenta — podział chronologiczny
# nie jest możliwy. Domyślnie: losowy, stratyfikowany po ESI × oddział.
# Alternatywa: --split site (oddział C = test zewnętrzny).
TEST_SIZE: float = 0.10            # 10% = test (ostateczna ocena)
VAL_SIZE: float = 0.10             # 10% = walidacja (porównanie / łączenie modeli)
EARLY_STOPPING_FRACTION: float = 0.10   # 10% TRAIN wydzielone na early stopping
HOLDOUT_SITE: str = "C"            # oddział trzymany jako test w --split site

# ─────────────────────────────────────────
# HIPERPARAMETRY MODELI (domyślne — sensowne bez tuningu)
# ─────────────────────────────────────────
XGB_DEFAULT_PARAMS: dict = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "max_bin": 256,
    "early_stopping_rounds": 100,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    # "device" ustawiane w runtime (auto: cuda jeśli dostępne)
}

LGBM_DEFAULT_PARAMS: dict = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,           # bez tego LightGBM ignoruje subsample
    "colsample_bytree": 0.7,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}
LGBM_EARLY_STOPPING_ROUNDS: int = 100

# CatBoost — inne nazwy parametrów (iterations, depth, random_seed, thread_count).
CATBOOST_DEFAULT_PARAMS: dict = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "classes_count": N_CLASSES,
    "iterations": 5000,
    "learning_rate": 0.08,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "border_count": 254,
    "grow_policy": "SymmetricTree",
    "random_seed": RANDOM_SEED,
    "thread_count": -1,
    "allow_writing_files": False,
    "od_type": "Iter",
    "od_wait": 200,
    "verbose": 200,
    # "task_type" ustawiane w runtime (auto: GPU jeśli dostępne)
}

# Random Forest — jedyny model baggingowy (dywersyfikacja ensemble).
# Braki obsługiwane natywnie (scikit-learn >= 1.4).
# min_samples_leaf=20: przy ~400 tys. wierszy model ma ~300 MB zamiast kilku GB,
# a prawdopodobieństwa są gładsze (lepiej nadają się do łączenia modeli).
RF_DEFAULT_PARAMS: dict = {
    "n_estimators": 300,
    "criterion": "gini",
    "max_features": "sqrt",
    "min_samples_leaf": 20,
    "max_samples": 0.5,            # bootstrap z 50% wierszy — szybciej i mniejszy model
    "bootstrap": True,
    "class_weight": None,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Explainable Boosting Machine — model "glassbox" (interpretowalność, MDR).
# interactions=0: w wersji wieloklasowej interpret nie wizualizuje interakcji par cech,
# więc czysty GAM zachowuje pełną interpretowalność (i trenuje się szybciej).
EBM_DEFAULT_PARAMS: dict = {
    "max_bins": 256,
    "interactions": 0,
    "learning_rate": 0.015,
    "outer_bags": 4,
    "max_rounds": 5000,
    "early_stopping_rounds": 100,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Regresja logistyczna (multinomialna) — liniowy punkt odniesienia.
LOGREG_DEFAULT_PARAMS: dict = {
    "C": 1.0,
    "max_iter": 2000,
    "tol": 1e-4,
    "random_state": RANDOM_SEED,
}

# ─────────────────────────────────────────
# OPTUNA — TUNING
# ─────────────────────────────────────────
# Domyślne ustawienia dobrane tak, by tuning wszystkich modeli zmieścił się w jedną noc
# na stacji klasy i9 + RTX 4080 (pełne 400 tys. wierszy × 5 foldów × 100 prób = dni).
OPTUNA_N_TRIALS: int = 50
TUNE_CV_FOLDS: int = 3
TUNE_MAX_ROWS: int | None = 150_000     # próbka TRAIN do tuningu; finalny trening na całości
# EBM jest wielokrotnie wolniejszy od pozostałych — mniej prób domyślnie
OPTUNA_N_TRIALS_PER_MODEL: dict[str, int] = {"ebm": 15}
OPTUNA_TIMEOUT_SECONDS: int | None = None  # bez limitu czasu
OPTUNA_STORAGE_DIR: Path = LOGS_DIR / "optuna"

# ─────────────────────────────────────────
# OLLAMA (LLM medyczny)
# ─────────────────────────────────────────
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL: str = "mistral"   # alternatywy: llama3, gemma2:9b, medllama2
OLLAMA_TEMPERATURE: float = 0.3         # niska — spójność medyczna
OLLAMA_MAX_TOKENS: int = 512
OLLAMA_TIMEOUT_S: int = 120

# ─────────────────────────────────────────
# CV (cross-validation)
# ─────────────────────────────────────────
CV_N_SPLITS: int = 5

# ─────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────
SHAP_BACKGROUND_SIZE: int = 200      # próbka tła do KernelSHAP/DeepSHAP
SHAP_TOP_FEATURES: int = 10          # ile najważniejszych cech pokazywać
