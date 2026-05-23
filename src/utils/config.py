"""
Konfiguracja globalna projektu SOR-AI.

Wszystkie ścieżki, stałe, hiperparametry i progi medyczne MTS
zdefiniowane w jednym miejscu, by uniknąć rozproszenia "magic numbers".
"""

from __future__ import annotations

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

LOGS_DIR: Path = PROJECT_ROOT / "logs"
TRAINING_LOGS_DIR: Path = LOGS_DIR / "training"     # logi z treningu modeli
EVAL_LOGS_DIR: Path = LOGS_DIR / "evaluation"       # logi z ewaluacji
EXPERIMENTS_DIR: Path = LOGS_DIR / "experiments"    # JSON-owe metadane eksperymentów

# MLflow tracking
MLFLOW_TRACKING_URI: str = f"file://{(PROJECT_ROOT / 'mlruns').as_posix()}"
MLFLOW_EXPERIMENT_NAME: str = "sor-ai-triage"

# Pliki danych
RDATA_FILE: Path = DATA_RAW_DIR / "5v_cleandf.rdata"
RAW_PARQUET: Path = DATA_PROCESSED_DIR / "yale_emmlc_raw.parquet"
PROCESSED_PARQUET: Path = DATA_PROCESSED_DIR / "yale_emmlc_processed.parquet"

TRAIN_PARQUET: Path = DATA_PROCESSED_DIR / "train.parquet"
VAL_PARQUET: Path = DATA_PROCESSED_DIR / "val.parquet"
TEST_PARQUET: Path = DATA_PROCESSED_DIR / "test.parquet"

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
# KLASYFIKACJA MTS
# ─────────────────────────────────────────
# Mapowanie ESI (Emergency Severity Index) → MTS (Manchester Triage System).
# Zob. Strategia 2 w TECHNICAL_ANALYSIS.md §6 — to mapowanie BAZOWE,
# które dodatkowo wzbogaca się o dyskryminatory parametrów życiowych.
ESI_TO_MTS: dict[int, str] = {
    1: "Red",      # Immediate / Natychmiastowy
    2: "Orange",   # Very Urgent / Bardzo pilny
    3: "Yellow",   # Urgent / Pilny
    4: "Green",    # Standard / Standardowy
    5: "Blue",     # Non-Urgent / Niepilny
}

# Zakodowane numerycznie 0–4 (do treningu modeli, kolejność rosnąca pilności = malejąca)
ESI_TO_MTS_NUMERIC: dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

# Nazwy klas w kolejności indeksów 0–4 (Red, Orange, Yellow, Green, Blue)
CLASS_NAMES: list[str] = ["Red", "Orange", "Yellow", "Green", "Blue"]
CLASS_NAMES_PL: list[str] = [
    "Czerwony (Natychmiastowy)",
    "Pomarańczowy (Bardzo pilny)",
    "Żółty (Pilny)",
    "Zielony (Standardowy)",
    "Niebieski (Niepilny)",
]

# Maksymalny czas oczekiwania (minuty) — zgodnie z protokołem MTS
MTS_MAX_WAIT_MINUTES: dict[str, int] = {
    "Red": 0,
    "Orange": 10,
    "Yellow": 60,
    "Green": 120,
    "Blue": 240,
}

# ─────────────────────────────────────────
# WAGI KLAS (asymetryczne — undertriage > overtriage)
# ─────────────────────────────────────────
# Kategoria Red to ~1-2% danych, ale jej pominięcie kosztuje życie.
# Te wagi forsują model, by NIE pomijał Red/Orange.
CUSTOM_CLASS_WEIGHTS: dict[int, float] = {
    0: 10.0,   # Red    — najwyższy priorytet
    1: 5.0,    # Orange — wysoki priorytet
    2: 1.0,    # Yellow — baseline
    3: 2.0,    # Green  — umiarkowany
    4: 3.0,    # Blue   — umiarkowany (mała klasa)
}

# Macierz kosztów dla ordinal cost-sensitive loss.
# Wiersz = prawdziwa klasa, kolumna = predykcja.
# Undertriage (predykcja niższej pilności) jest 3× droższy niż overtriage.
UNDERTRIAGE_COST_MULTIPLIER: float = 3.0
OVERTRIAGE_COST_MULTIPLIER: float = 1.0

# ─────────────────────────────────────────
# PODZIAŁ DANYCH (TEMPORAL SPLIT)
# ─────────────────────────────────────────
# Zob. TECHNICAL_ANALYSIS.md §2.6 — używamy podziału chronologicznego,
# aby zwalidować model w warunkach zmienności sezonowej.
TEST_SIZE: float = 0.10            # 10% najnowszych danych = test
VAL_SIZE_OF_TRAINVAL: float = 0.111  # ~10% z train+val = walidacja

# ─────────────────────────────────────────
# HIPERPARAMETRY MODELI
# ─────────────────────────────────────────
XGB_DEFAULT_PARAMS: dict = {
    "objective": "multi:softprob",
    "num_class": 5,
    "eval_metric": ["mlogloss", "merror"],
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",   # 'gpu_hist' jeśli masz GPU
    "device": "cuda",
    "random_state": RANDOM_SEED,
    "early_stopping_rounds": 50,
    "n_jobs": -1,
}

LGBM_DEFAULT_PARAMS: dict = {
    "objective": "multiclass",
    "num_class": 5,
    "n_estimators": 1000,
    "max_depth": 8,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

RF_DEFAULT_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

EBM_DEFAULT_PARAMS: dict = {
    "max_bins": 256,
    "interactions": 10,
    "learning_rate": 0.01,
    "min_samples_leaf": 5,
    "random_state": RANDOM_SEED,
}

# CatBoost — multi-threaded CPU (komplementarny boosting do XGB/LGBM).
# Uwaga: CatBoost używa innych nazw:
#   - `random_seed`   (nie random_state)
#   - `thread_count`  (nie n_jobs)
#   - `iterations`    (nie n_estimators)
#   - `depth`         (nie max_depth)
#   - `l2_leaf_reg`   (nie reg_lambda)
#   - `rsm`           (nie colsample_bylevel)
CATBOOST_DEFAULT_PARAMS: dict = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "classes_count": 5,
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "border_count": 254,             # liczba binów dla cech numerycznych (max quality)
    "bootstrap_type": "MVS",         # nowoczesny CPU-default w CatBoost
    "subsample": 0.8,
    "rsm": 0.8,                       # column sampling per level
    "grow_policy": "SymmetricTree",  # CatBoost signature — najbardziej stabilny
    "boosting_type": "Plain",        # szybsza wersja (Ordered tylko dla <50k samples)
    "random_seed": RANDOM_SEED,
    "thread_count": -1,               # wszystkie rdzenie CPU
    "task_type": "GPU",
    "devices": "0",
    "allow_writing_files": False,    # nie twórz folderu catboost_info/
    "od_type": "Iter",                # early stopping mode
    "od_wait": 100,                   # patience
    "verbose": 100,                   # log co 100 iteracji
}

# ExtraTrees — maksymalna randomizacja (komplementarna do RF).
# Większość parametrów jak RF, ale domyślnie bootstrap=False,
# a criterion może być też 'friedman_mse' / 'mae' / 'poisson'.
ET_DEFAULT_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": None,
    "bootstrap": False,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": 1,
}

# HistGradientBoosting — gradient boosting na histogramach (sklearn).
# Inny inductive bias niż XGB/LGBM/CatBoost.
HISTGBT_DEFAULT_PARAMS: dict = {
    "learning_rate": 0.05,
    "max_iter": 1000,
    "max_depth": 6,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": "auto",
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "tol": 1e-7,
    "random_state": RANDOM_SEED,
}

# ─────────────────────────────────────────
# OPTUNA — TUNING
# ─────────────────────────────────────────
OPTUNA_N_TRIALS: int = 300
OPTUNA_TIMEOUT_SECONDS: int | None = None  # bez limitu czasu

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
CV_N_SPLITS: int = 10

# ─────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────
SHAP_BACKGROUND_SIZE: int = 200      # próbka tła do KernelSHAP/DeepSHAP
SHAP_TOP_FEATURES: int = 10          # ile najważniejszych cech pokazywać
