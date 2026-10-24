"""
Konfiguracja globalna projektu SOR-AI.

Wszystkie ścieżki, stałe, hiperparametry i progi kliniczne zdefiniowane
w jednym miejscu, by uniknąć rozproszenia "magic numbers".

Etykieta: natywny Emergency Severity Index (ESI 1–5) — ten sam algorytm,
który polskie SOR-y stosują w systemie TOPSOR (Dz.U. 2021 poz. 1182, § 2 pkt 9).
Etykieta nie jest mapowana na żadną inną skalę triażu.
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
DATA_VERSION: str = "esi-v2"

# ─────────────────────────────────────────
# PATH ENV OVERRIDES (testy end-to-end w katalogu tymczasowym, inne dyski)
# ─────────────────────────────────────────
# Czytane RAZ przy imporcie — podprocesy dziedziczą je ze środowiska. Nazwy stałych bez zmian.
#   SOR_AI_ESI_DATA_DIR → splity ESI, manifest, listy cech (raport z audytu też tam —
#                         raport innych danych nie nadpisuje wersjonowanego DATA_AUDIT.md),
#   SOR_AI_RUNS_DIR     → runs/,
#   SOR_AI_LOGS_DIR     → logs/ (logi treningu, ewaluacji, eksperymentów, badania Optuny).
def _env_path(var: str) -> Path | None:
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


if (_override := _env_path("SOR_AI_ESI_DATA_DIR")) is not None:
    ESI_DATA_DIR = _override
    TRAIN_PARQUET = ESI_DATA_DIR / "train.parquet"
    VAL_PARQUET = ESI_DATA_DIR / "val.parquet"
    TEST_PARQUET = ESI_DATA_DIR / "test.parquet"
    FEATURE_SETS_JSON = ESI_DATA_DIR / "feature_sets.json"
    DATA_MANIFEST_JSON = ESI_DATA_DIR / "manifest.json"
    DATA_AUDIT_MD = ESI_DATA_DIR / "DATA_AUDIT.md"
if (_override := _env_path("SOR_AI_RUNS_DIR")) is not None:
    RUNS_DIR = _override
if (_override := _env_path("SOR_AI_LOGS_DIR")) is not None:
    LOGS_DIR = _override
    TRAINING_LOGS_DIR = LOGS_DIR / "training"
    EVAL_LOGS_DIR = LOGS_DIR / "evaluation"
    EXPERIMENTS_DIR = LOGS_DIR / "experiments"

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
# Urządzenie wybiera src.utils.hardware.device_for(biblioteka), czytając zmienne NA BIEŻĄCO:
#   SOR_AI_DEVICE_XGBOOST / SOR_AI_DEVICE_CATBOOST  >  SOR_AI_DEVICE  >  'auto'
# Wartości auto | cpu | gpu. 'auto' = próbny trening na GPU (porażka → CPU z ostrzeżeniem),
# 'gpu' = wymuszenie (porażka → błąd zamiast cichego CPU). Poniższa stała tylko informacyjnie.
DEVICE: str = os.environ.get("SOR_AI_DEVICE", "auto").lower()

# Wątki CPU dla wszystkich modeli: SOR_AI_N_JOBS (> 0) albo liczba FIZYCZNYCH rdzeni
# (i9-14900K: 24, nie 32 — wątki HT i pętle OpenMP z barierami spowalniają trening).
# Rozwiązywane przy imporcie: proces nadrzędny (run_all, 03_train.py --n-jobs) ustawia
# zmienną PRZED importem. Import tutaj (nie na górze pliku): hardware nie importuje config.
from src.utils.hardware import n_jobs as _resolve_n_jobs  # noqa: E402

N_JOBS: int = _resolve_n_jobs()

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

# Odcienie kolorów TOPSOR do wykresów i UI (klucz = nazwa klasy z CLASS_NAMES).
# Wyłącznie prezentacja — etykietą zawsze jest poziom ESI.
TOPSOR_HEX: dict[str, str] = {
    "ESI1": "#D32F2F",   # czerwony
    "ESI2": "#F57C00",   # pomarańczowy
    "ESI3": "#FBC02D",   # żółty
    "ESI4": "#388E3C",   # zielony
    "ESI5": "#1976D2",   # niebieski
}

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
# REGUŁY DECYZYJNE (src/evaluation/decision.py)
# ─────────────────────────────────────────
# Reguła "undertriage ≤ T" (B1): oczekiwany poziom EL = Σ k·p_k dzielony 4 progami; próg ESI2/ESI3
# wynika z celu T, pozostałe trzy maksymalizują QWK. Parametry dobierane WYŁĄCZNIE na val —
# domyślnie w podgrupie vitals_complete==1 (populacja TOPSOR; dobór na całym val przestrzeliwuje
# w vc o 2–4 pp) — i stosowane bez zmian na teście. Progi zależą od rozkładu ESI: w Polsce
# trzeba je dobrać na nowo na danych lokalnych.
DECISION_TARGET_UNDERTRIAGE: float = 0.15
DECISION_FIT_POPULATION: str = "vc"           # "vc" (vitals_complete == 1) | "all"
DECISION_GUARANTEE_ALPHA: float = 0.05        # B6: jednostronna górna granica Clopper–Pearsona ≤ T
DECISION_CUTPOINT_GRID: int = 1000            # kandydaci na progi = tyle kwantyli EL
# Siatka bezpieczeństwa (B2): P(ESI1) + P(ESI2) ≥ q → najwyżej ESI3. q = NAJWIĘKSZE spełniające
# cel critical miss (ESI1 → ESI4–5) na danych dopasowania; q < SAFETY_NET_MIN_Q nie jest
# rozważane (zbyt wielu pacjentów spychanych do ESI3).
SAFETY_NET_MAX_CRITICAL_MISS: float = 0.005
SAFETY_NET_MIN_Q: float = 0.05
# q dobierane na WSZYSTKICH wierszach val ("all"), nawet gdy progi na vc: w vc jest tylko ~60 pacjentów
# ESI1 (pomiar: dobór na połowie vc czasem nie widzi żadnego pominięcia i siatka znika)
SAFETY_NET_FIT_POPULATION: str = "all"        # "all" | "mask" (te same wiersze co progi)
# Wykres kompromisu undertriage–QWK (C6): cele T reguł i mnożniki k reguły kosztowej
TRADEOFF_TARGETS: tuple[float, ...] = (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40)
TRADEOFF_COST_MULTIPLIERS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)

# ─────────────────────────────────────────
# PODZIAŁ DANYCH
# ─────────────────────────────────────────
# Yale EMMLC nie ma roku ani identyfikatora pacjenta — podział chronologiczny
# nie jest możliwy. Domyślnie: losowy, stratyfikowany po ESI × oddział.
# Alternatywa: --split site (oddział B = test zewnętrzny; C ma tylko ~20 wizyt ESI 1).
TEST_SIZE: float = 0.10            # 10% = test (ostateczna ocena)
VAL_SIZE: float = 0.10             # 10% = walidacja (porównanie / łączenie modeli)
EARLY_STOPPING_FRACTION: float = 0.10   # 10% TRAIN wydzielone na early stopping
HOLDOUT_SITE: str = "B"            # oddział trzymany jako test w --split site (675 × ESI 1)

# ─────────────────────────────────────────
# HIPERPARAMETRY MODELI (domyślne — sensowne bez tuningu)
# ─────────────────────────────────────────
# Limity iteracji boostingu są celowo wysokie: przy małym learning_rate i pełnych ~400 tys.
# wierszy model potrzebuje tysięcy drzew, a trening i tak kończy early stopping. Gdy najlepsza
# iteracja dojdzie do ≥ 95% limitu, wrapper modelu loguje WARNING (model niedotrenowany).
# Wątki: N_JOBS (sekcja SPRZĘT). Testy nadpisują je na 1 (tests/conftest.py).
XGB_DEFAULT_PARAMS: dict = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_estimators": 10_000,
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
    "n_jobs": N_JOBS,
    # "device" ustawiane w runtime: hardware.device_for("xgboost") → "cuda" | "cpu";
    # po treningu na GPU model przełączany na device="cpu" (predykcja na danych z RAM)
}

LGBM_DEFAULT_PARAMS: dict = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "n_estimators": 20_000,
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
    "n_jobs": N_JOBS,              # fizyczne rdzenie (-1 = wszystkie wątki logiczne — wolniej)
    "verbosity": -1,
}
LGBM_EARLY_STOPPING_ROUNDS: int = 100

# CatBoost — inne nazwy parametrów (iterations, depth, random_seed, thread_count).
# Postęp (verbose) ustawia wrapper w czasie treningu — fit_estimator() jest cichy.
CATBOOST_DEFAULT_PARAMS: dict = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "classes_count": N_CLASSES,
    "iterations": 20_000,
    "learning_rate": 0.08,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "border_count": 254,
    "grow_policy": "SymmetricTree",
    "random_seed": RANDOM_SEED,
    "thread_count": N_JOBS,
    "allow_writing_files": False,
    "od_type": "Iter",
    "od_wait": 200,
    # "task_type" ustawiane w runtime: hardware.device_for("catboost") → "GPU" | "CPU"
}
# Na GPU CatBoost rezerwuje ułamek WOLNEJ pamięci karty na cały trening (domyślnie 0.95);
# 0.85 zostawia zapas dla ekranu, gdy karta obsługuje też monitor.
CATBOOST_GPU_RAM_PART: float = 0.85

# Random Forest — jedyny model baggingowy (dywersyfikacja ensemble).
# Braki obsługiwane natywnie (scikit-learn >= 1.4). Liczby drzew się nie stroi (więcej drzew
# nigdy nie pogarsza log loss) — 500 na finalny model. min_samples_leaf=5 jest bliżej optimum
# niż 20, ale model jest większy: przy ~400 tys. wierszy ~1–2 GB w RAM (plik joblib kompresowany).
RF_DEFAULT_PARAMS: dict = {
    "n_estimators": 500,
    "criterion": "gini",
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "max_samples": 0.5,            # bootstrap z 50% wierszy — szybciej i mniejszy model
    "bootstrap": True,
    "class_weight": None,
    "random_state": RANDOM_SEED,
    "n_jobs": N_JOBS,
}

# Explainable Boosting Machine — model "glassbox" (interpretowalność, MDR).
# interactions=0: w wersji wieloklasowej interpret nie wizualizuje interakcji par cech,
# więc czysty GAM zachowuje pełną interpretowalność (i trenuje się szybciej).
# outer_bags=14 (domyślne interpret) na finalny model: EBM zrównolegla się tylko po workach,
# więc 14 worków na 24 rdzeniach kosztuje tyle czasu co 4, a krzywe są gładsze.
# Do tuningu można podać mniej worków (parametr outer_bags).
EBM_DEFAULT_PARAMS: dict = {
    "max_bins": 256,
    "interactions": 0,
    "learning_rate": 0.015,
    "outer_bags": 14,
    "max_rounds": 5000,
    "early_stopping_rounds": 100,
    "random_state": RANDOM_SEED,
    "n_jobs": N_JOBS,
}

# Regresja logistyczna (multinomialna) — liniowy punkt odniesienia.
# Pipeline (logreg_model.py): SimpleImputer(median, add_indicator=True, keep_empty_features=True)
# → StandardScaler → LogisticRegression; wskaźniki braków zachowują informację "nie zmierzono".
# Wątki BLAS ograniczane do N_JOBS w czasie treningu.
LOGREG_DEFAULT_PARAMS: dict = {
    "C": 1.0,
    "max_iter": 2000,
    "tol": 1e-4,
    "random_state": RANDOM_SEED,
}

# ─────────────────────────────────────────
# OPTUNA — TUNING
# ─────────────────────────────────────────
# Budżety tuningu w PROFILACH (03_train.py --profile, run_all.py --profile). Liczba prób i limit
# czasu dotyczą CAŁEGO badania Optuny: wznowienie po awarii dokańcza tylko brakujące próby i czas.
# Klucze modelu: n_trials, tune_rows (0 = cały zbiór dopasowania X_fit), n_folds, timeout_s.
# Klucze profilu: train_sample / eval_sample (None = pełne splity).
# Szacunki czasu (i9-14900K + RTX 4080 Super, zestaw 'triage', "standard"): ~11–12 h sekwencyjnie,
# ~7–8 h w dwóch torach (GPU: xgboost → catboost, CPU: reszta); każdy model ma też limit czasu.
DEFAULT_TUNING_PROFILE: str = "standard"
TUNE_CV_FOLDS: int = 3                  # foldy CV, gdy tune_model wywołano bez profilu


def _budget(n_trials: int, tune_rows: int, n_folds: int, timeout_s: int) -> dict:
    return {"n_trials": n_trials, "tune_rows": tune_rows, "n_folds": n_folds, "timeout_s": timeout_s}


def _uniform_profile(train_sample: int | None, eval_sample: int | None, **budget) -> dict:
    """Profil testowy: ten sam (mały) budżet dla każdego modelu."""
    models = ("logreg", "lightgbm", "xgboost", "catboost", "random_forest", "ebm")
    return {"train_sample": train_sample, "eval_sample": eval_sample,
            **{m: _budget(**budget) for m in models}}


_H = 3600
TUNING_PROFILES: dict[str, dict] = {
    # CI / test instalacji: kilka minut na wszystkie modele
    "smoke": _uniform_profile(6_000, 4_000, n_trials=2, tune_rows=3_000, n_folds=2, timeout_s=120),
    # Test całego pipeline'u w domu przed nocnym przebiegiem (~20–30 min)
    "quick": _uniform_profile(60_000, None, n_trials=3, tune_rows=20_000, n_folds=2, timeout_s=300),
    # Domyślny nocny przebieg. GBDT i logreg strojone na całym X_fit (optimum wielkości liścia
    # zależy od liczby wierszy); RF i EBM na 150 tys. (ich parametry dobrze się przenoszą).
    "standard": {
        "train_sample": None, "eval_sample": None,
        "logreg": _budget(20, 0, 3, 45 * 60),
        "lightgbm": _budget(120, 0, 3, 2 * _H),
        "xgboost": _budget(100, 0, 3, int(3.5 * _H)),
        "catboost": _budget(60, 0, 3, 4 * _H),
        "random_forest": _budget(40, 150_000, 3, int(1.5 * _H)),
        "ebm": _budget(25, 150_000, 3, 4 * _H),
    },
    # ≈ 1,7 × próby i czas "standard"; EBM strojony na 250 tys. wierszy
    "thorough": {
        "train_sample": None, "eval_sample": None,
        "logreg": _budget(25, 0, 3, 75 * 60),
        "lightgbm": _budget(200, 0, 3, int(3.4 * _H)),
        "xgboost": _budget(170, 0, 3, int(6.0 * _H)),
        "catboost": _budget(100, 0, 3, int(6.8 * _H)),
        "random_forest": _budget(70, 150_000, 3, int(2.55 * _H)),
        "ebm": _budget(40, 250_000, 3, int(6.8 * _H)),
    },
}

# Budżet zastępczy, gdy hardware.device_for() zwraca 'cpu' dla biblioteki GPU (brak karty,
# za stary sterownik, wymuszone SOR_AI_DEVICE_*=cpu). Próba na CPU trwa kilka–kilkanaście razy
# dłużej, więc mniej wierszy i prób. Profile bez wpisu (smoke, quick) — budżet bez zmian.
CPU_FALLBACK_BUDGETS: dict[str, dict[str, dict]] = {
    "standard": {
        "xgboost": _budget(50, 150_000, 3, 3 * _H),
        "catboost": _budget(25, 100_000, 3, 4 * _H),
    },
    "thorough": {
        "xgboost": _budget(85, 150_000, 3, int(5.1 * _H)),
        "catboost": _budget(40, 100_000, 3, int(6.8 * _H)),
    },
}

# Wersja przestrzeni hiperparametrów (src/models/tuning.py) — część klucza badania Optuny.
# PODBIJ przy każdej zmianie zakresów, wyborów albo protokołu foldu danego modelu: stare próby
# nie zostaną wtedy wymieszane z nowymi (nowe badanie zamiast cichego łączenia).
SEARCH_SPACE_VERSION: dict[str, int] = {
    "logreg": 2,
    "lightgbm": 2,
    "xgboost": 2,
    "catboost": 2,
    "random_forest": 2,
    "ebm": 2,
}

# Parametry STAŁE w czasie tuningu (finalny model bierze wartości domyślne z sekcji modeli):
#   RF: 300 drzew w CV, 500 w finalnym modelu (więcej drzew nie pogarsza log loss),
#   EBM: 8 worków w CV (3 foldy × 8 = 24 rdzenie), 14 w finalnym; interakcje zawsze 0.
TUNE_FIXED_PARAMS: dict[str, dict] = {
    "random_forest": {"n_estimators": 300},
    "ebm": {"outer_bags": 8, "interactions": 0},
}
# Parametry "liczności" skalowane przed finalnym treningiem o sqrt(n_fit / n_fold_train):
# optimum minimalnej wielkości liścia rośnie ~ √n (pomiar: ×4 danych → ×2 optimum).
TUNE_SQRT_SCALED_PARAMS: dict[str, list[str]] = {
    "lightgbm": ["min_child_samples"],
    "xgboost": ["min_child_weight"],
}
# Strażnik rozmiaru RF: próba, której finalny model (500 drzew) miałby > tylu bajtów, jest przerywana
TUNE_RF_MAX_MODEL_BYTES: float = 1.5e9

OPTUNA_STORAGE_DIR: Path = LOGS_DIR / "optuna"
OPTUNA_HEARTBEAT_INTERVAL_S: int = 60     # "puls" trwającej próby w bazie SQLite
OPTUNA_GRACE_PERIOD_S: int = 180          # brak pulsu dłużej → próba FAIL i ponowienie
OPTUNA_MAX_RETRY: int = 1                 # ile razy ponowić próbę przerwaną awarią procesu
OPTUNA_MAX_CONSECUTIVE_FAILS: int = 5     # tyle błędów z rzędu → koniec badania
OPTUNA_PRUNER_STARTUP_TRIALS: int = 5     # MedianPruner: pierwsze próby zawsze do końca

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
