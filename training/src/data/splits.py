"""
Podziały train/val/test.

Dwa typy:
    1. CHRONOLOGICAL — preferowany (TECHNICAL_ANALYSIS.md §2.6).
       Trening na przeszłości, test na przyszłości — symuluje deployment
       i waliduje generalizację w czasie (sezonowość, np. epidemie grypy).
    2. STRATIFIED — tylko gdy nie ma daty/sezonowości; zachowuje rozkład klas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import (
    DATA_PROCESSED_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE_OF_TRAINVAL,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# Sortowanie chronologiczne
# ─────────────────────────────────────────
def _build_temporal_key(df: pd.DataFrame) -> pd.Series | None:
    """
    Próbuje zbudować klucz czasowy z dostępnych kolumn.

    Yale EMMLC zwykle ma:
        - arrival_year, arrival_month, arrival_day, arrival_hour
        lub
        - arrivalmonth_*, arrivalday_*, arrivalhour_*

    Zwraca pd.Series posortowaną rosnąco (od najstarszego do najnowszego)
    lub None jeśli nie da się zbudować klucza.
    """
    candidates_full = ["arrival_datetime", "arrivaltime", "ed_arrival_time"]
    for col in candidates_full:
        if col in df.columns:
            try:
                key = pd.to_datetime(df[col], errors="coerce")
                if key.notna().sum() > 0.5 * len(df):
                    return key
            except Exception:
                continue

    # Próba złożenia z części
    parts = []
    for prefix in ("arrival_", "arrival"):
        year_col = f"{prefix}year" if f"{prefix}year" in df.columns else None
        month_col = f"{prefix}month" if f"{prefix}month" in df.columns else None
        day_col = f"{prefix}day" if f"{prefix}day" in df.columns else None
        hour_col = f"{prefix}hour" if f"{prefix}hour" in df.columns else None
        if year_col or month_col:
            for c in (year_col, month_col, day_col, hour_col):
                if c is not None:
                    parts.append(c)
            break

    if parts:
        try:
            # Złóż datę z części (rok-miesiąc-dzień-godzina)
            df_parts = df[parts].copy()
            base = pd.Timestamp("2014-01-01")
            year = df_parts.iloc[:, 0] if len(parts) > 0 else 2014
            # uproszczony klucz porządkowy: rok*12 + miesiąc itd.
            key_int = np.zeros(len(df), dtype=np.float64)
            multiplier = 12 * 31 * 24
            for col in parts:
                # Wyciągnij liczbowo (np. "Apr" → 4)
                series = df_parts[col]
                if pd.api.types.is_numeric_dtype(series):
                    key_int += series.fillna(0).values * multiplier
                else:
                    # Nie-numeryczne (miesiące jako string) — kodujemy
                    codes = series.astype("category").cat.codes
                    key_int += codes.fillna(0).values * multiplier
                multiplier //= 31 if multiplier > 31 else 1
                if multiplier < 1:
                    multiplier = 1
            return pd.Series(key_int, index=df.index)
        except Exception as e:
            log.debug(f"Nie udało się zbudować klucza czasowego: {e}")
            return None

    return None


# ─────────────────────────────────────────
# Podział chronologiczny
# ─────────────────────────────────────────
def chronological_split(
    df: pd.DataFrame,
    target_col: str = "mts_numeric",
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE_OF_TRAINVAL,
    sort_by: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Podział czasowy: pierwsze X% → train, środek → val, ostatnie Y% → test.

    Trening na przeszłości, test na przyszłości — symuluje deployment.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    test_size : float
        Procent najnowszych danych jako test.
    val_size : float
        Procent z train+val jako walidacja.
    sort_by : str, optional
        Nazwa kolumny do sortowania. Jeśli None — auto-detekcja.

    Returns
    -------
    dict z kluczami 'train', 'val', 'test'
    """
    df = df.copy()

    # Zbuduj/wybierz klucz czasowy
    if sort_by is not None and sort_by in df.columns:
        log.info(f"Sortowanie chronologiczne po: {sort_by}")
        df = df.sort_values(sort_by, kind="stable").reset_index(drop=True)
    else:
        key = _build_temporal_key(df)
        if key is not None:
            log.info("Sortowanie chronologiczne po auto-wykrytym kluczu czasowym")
            df = df.assign(_temp_key=key.values).sort_values("_temp_key", kind="stable")
            df = df.drop(columns=["_temp_key"]).reset_index(drop=True)
        else:
            log.warning(
                "Brak kolumny czasowej — używam kolejności wierszów. "
                "Rozważ podział stratyfikowany (`stratified_split`)."
            )

    n = len(df)
    n_test = int(n * test_size)
    n_train_val = n - n_test
    n_val = int(n_train_val * val_size)
    n_train = n_train_val - n_val

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()

    log.info("Podział chronologiczny:")
    log.info(f"  train: {len(train):>10,} ({100 * len(train) / n:5.1f}%)")
    log.info(f"  val:   {len(val):>10,} ({100 * len(val) / n:5.1f}%)")
    log.info(f"  test:  {len(test):>10,} ({100 * len(test) / n:5.1f}%)")

    if target_col in df.columns:
        log.info(f"\nRozkład klas {target_col}:")
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            counts = split_df[target_col].value_counts(normalize=True).sort_index()
            log.info(f"  {split_name}: {dict(counts.round(3))}")

    return {"train": train, "val": val, "test": test}


# ─────────────────────────────────────────
# Podział stratyfikowany (fallback)
# ─────────────────────────────────────────
def stratified_split(
    df: pd.DataFrame,
    target_col: str = "mts_numeric",
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE_OF_TRAINVAL,
    random_state: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """
    Podział stratyfikowany — zachowuje rozkład klas w każdym zbiorze.
    Używaj tylko gdy nie ma sensownej kolumny czasowej.
    """
    if target_col not in df.columns:
        raise ValueError(f"Brak kolumny target '{target_col}'")

    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=df_train_val[target_col],
    )

    log.info("Podział stratyfikowany:")
    n = len(df)
    log.info(f"  train: {len(df_train):>10,} ({100 * len(df_train) / n:5.1f}%)")
    log.info(f"  val:   {len(df_val):>10,} ({100 * len(df_val) / n:5.1f}%)")
    log.info(f"  test:  {len(df_test):>10,} ({100 * len(df_test) / n:5.1f}%)")

    return {"train": df_train, "val": df_val, "test": df_test}


# ─────────────────────────────────────────
# Zapis splitów
# ─────────────────────────────────────────
def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: Path | str = DATA_PROCESSED_DIR,
    format: str = "parquet",
) -> dict[str, Path]:
    """
    Zapisuje podziały na dysk.

    Parameters
    ----------
    splits : dict
        Wynik `chronological_split` lub `stratified_split`.
    output_dir : Path
    format : str
        'parquet' (domyślne) lub 'csv'.

    Returns
    -------
    dict z mapowaniem split_name → Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, split_df in splits.items():
        if format == "parquet":
            path = output_dir / f"{name}.parquet"
            split_df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        elif format == "csv":
            path = output_dir / f"{name}.csv"
            split_df.to_csv(path, index=False)
        else:
            raise ValueError(f"Nieznany format: {format}")

        paths[name] = path
        log.info(f"Zapisano {name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    return paths
