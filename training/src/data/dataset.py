"""
Dostęp do przygotowanych danych ESI: splity, zestawy cech, X/y.

Użycie:
    from src.data.dataset import load_splits, get_xy
    splits = load_splits()                       # train / val / test
    X, y = get_xy(splits["train"], "triage")
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import (
    DATA_MANIFEST_JSON,
    DATA_VERSION,
    EARLY_STOPPING_FRACTION,
    FEATURE_SETS_JSON,
    RANDOM_SEED,
    TARGET_COL,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

SPLIT_PATHS = {"train": TRAIN_PARQUET, "val": VAL_PARQUET, "test": TEST_PARQUET}

# Stare nazwy z poprzedniej wersji pipeline'u
FEATURE_SET_ALIASES = {"triage_only": "triage", "top": "triage"}

# =1 → brak/nieaktualne splity to błąd zamiast automatycznego przygotowania
# (ustawiane przez orkiestrator: podprocesy treningu nie mogą równolegle przygotowywać danych)
NO_AUTO_PREPARE_ENV = "SOR_AI_NO_AUTO_PREPARE"
SPLIT_STRATEGIES = ("random", "site")


def resolve_feature_set(name: str) -> str:
    name = FEATURE_SET_ALIASES.get(name, name)
    if name not in ("triage", "full"):
        raise ValueError(f"Nieznany zestaw cech '{name}'. Dostępne: triage, full")
    return name


def _read_manifest() -> dict | None:
    """Manifest splitów albo None (brak / uszkodzony plik)."""
    try:
        return json.loads(DATA_MANIFEST_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _not_prepared_reason() -> str | None:
    """Dlaczego splity nie nadają się do użycia (None = są aktualne)."""
    missing = [p.name for p in (*SPLIT_PATHS.values(), FEATURE_SETS_JSON) if not p.exists()]
    if missing:
        return f"brak plików: {', '.join(missing)}"
    meta = _read_manifest()
    if meta is None:
        return f"brak lub uszkodzony {DATA_MANIFEST_JSON.name}"
    version = meta.get("data_version")
    if version != DATA_VERSION:
        return f"wersja danych '{version}', kod oczekuje '{DATA_VERSION}'"
    return None


def is_prepared() -> bool:
    """Czy splity istnieją i zostały przygotowane bieżącą wersją logiki."""
    return _not_prepared_reason() is None


def auto_prepare_disabled() -> bool:
    return os.environ.get(NO_AUTO_PREPARE_ENV, "").strip().lower() in ("1", "true", "yes", "tak")


def ensure_prepared(strategy: str | None = None) -> None:
    """
    Uruchamia przygotowanie danych, jeśli splitów brak lub są nieaktualne.

    strategy=None → strategia podziału z istniejącego manifestu (np. 'site'),
    a bez manifestu 'random'. Przy `SOR_AI_NO_AUTO_PREPARE=1` zamiast przygotowania
    (≈ 11 GB RAM) zgłasza RuntimeError.
    """
    reason = _not_prepared_reason()
    if reason is None:
        return
    if auto_prepare_disabled():
        raise RuntimeError(
            f"Splity ESI nie są gotowe ({reason}), a {NO_AUTO_PREPARE_ENV}=1 wyłącza "
            f"automatyczne przygotowanie. Uruchom: python scripts/02_preprocess.py"
        )
    if strategy is None:
        previous = (_read_manifest() or {}).get("split_strategy")
        strategy = previous if previous in SPLIT_STRATEGIES else "random"
    log.warning(f"Splity ESI nie są gotowe ({reason}) — uruchamiam przygotowanie danych "
                f"(podział '{strategy}', scripts/02_preprocess.py)…")
    from src.data.prepare import prepare_dataset

    prepare_dataset(strategy=strategy)
    _feature_sets_payload.cache_clear()


def manifest() -> dict:
    return json.loads(DATA_MANIFEST_JSON.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _feature_sets_payload() -> dict:
    if not FEATURE_SETS_JSON.exists():
        raise FileNotFoundError(
            f"Brak {FEATURE_SETS_JSON}. Uruchom: python scripts/02_preprocess.py"
        )
    return json.loads(FEATURE_SETS_JSON.read_text(encoding="utf-8"))


def feature_names(feature_set: str = "triage") -> list[str]:
    """Lista cech danego zestawu (kolejność stała — zgodna z modelem)."""
    return list(_feature_sets_payload()["feature_sets"][resolve_feature_set(feature_set)])


def feature_groups() -> dict[str, list[str]]:
    return dict(_feature_sets_payload()["groups"])


def load_split(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    if name not in SPLIT_PATHS:
        raise ValueError(f"Nieznany split '{name}'. Dostępne: {list(SPLIT_PATHS)}")
    path = SPLIT_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"Brak {path}. Uruchom: python scripts/02_preprocess.py")
    if DATA_MANIFEST_JSON.exists():
        version = manifest().get("data_version")
        if version != DATA_VERSION:
            raise RuntimeError(
                f"Splity przygotowane wersją '{version}', kod oczekuje '{DATA_VERSION}'. "
                f"Uruchom ponownie: python scripts/02_preprocess.py"
            )
    return pd.read_parquet(path, columns=columns)


def load_splits(
    names: tuple[str, ...] = ("train", "val", "test"),
    feature_set: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Wczytuje splity; z `feature_set` tylko potrzebne kolumny (oszczędza RAM)."""
    columns = None
    if feature_set is not None:
        from src.data.prepare import META_COLS

        columns = META_COLS + feature_names(feature_set)
    return {name: load_split(name, columns=columns) for name in names}


def get_xy(df: pd.DataFrame, feature_set: str = "triage") -> tuple[pd.DataFrame, pd.Series]:
    """X (float32-zgodne, NaN dozwolone) i y (0–4)."""
    names = feature_names(feature_set)
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise KeyError(f"Brak kolumn w danych: {missing[:10]}")
    X = df[names].astype(np.float32)
    y = df[TARGET_COL].astype(int).rename(TARGET_COL)
    return X, y


def sample_frame(df: pd.DataFrame, n: int | None, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Stratyfikowana próbka (do szybkich testów pipeline'u)."""
    if n is None or n >= len(df):
        return df
    sample, _ = train_test_split(df, train_size=n, random_state=seed, stratify=df[TARGET_COL])
    return sample.sort_index()


def early_stopping_split(
    X: pd.DataFrame,
    y: pd.Series,
    fraction: float = EARLY_STOPPING_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Wydziela z TRAIN część na early stopping (stratyfikacja po klasie).

    Dzięki temu zbiór `val` pozostaje nietknięty przez każdy model i może służyć
    do uczciwego porównania oraz ręcznego łączenia modeli.
    """
    X_fit, X_es, y_fit, y_es = train_test_split(
        X, y, test_size=fraction, random_state=seed, stratify=y
    )
    return X_fit, y_fit, X_es, y_es
