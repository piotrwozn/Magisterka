"""
Ładowanie datasetu Yale EMMLC.

Yale EMMLC pochodzi z R i jest dostarczony w formacie .RData (560 486 × 972).
Konwertujemy do parquet, by przyspieszyć kolejne ładowania (parquet ~10× szybszy
i ~5× mniejszy rozmiar niż CSV).

Przykład użycia:
    from src.data.load_data import load_dataset
    df = load_dataset()  # automatycznie ładuje parquet jeśli istnieje
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import (
    PROCESSED_PARQUET,
    RAW_PARQUET,
    RDATA_FILE,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# .RData → DataFrame
# ─────────────────────────────────────────
def load_rdata(path: Path | str = RDATA_FILE) -> pd.DataFrame:
    """
    Wczytuje plik .RData i zwraca pierwszy znaleziony DataFrame.

    Wymaga `pyreadr`. Yale EMMLC zawiera jeden obiekt: `df`.

    Parameters
    ----------
    path : Path | str
        Ścieżka do pliku .RData.

    Returns
    -------
    pd.DataFrame
        Surowe dane (560 486 × 972 dla pełnego datasetu Yale EMMLC).
    """
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "Pakiet `pyreadr` nie jest zainstalowany. "
            "Zainstaluj: `pip install pyreadr`."
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Plik .RData nie znaleziony: {path}. "
            f"Pobierz Yale EMMLC z: https://github.com/yaleemmlc/admissionprediction"
        )

    log.info(f"Wczytywanie .RData: {path} ({path.stat().st_size / 1e6:.1f} MB)…")
    result = pyreadr.read_r(str(path))

    if not result:
        raise ValueError(f"Plik .RData nie zawiera żadnych obiektów: {path}")

    # Wyciągnij pierwszy DataFrame (Yale EMMLC zawiera tylko jeden)
    key = next(iter(result.keys()))
    df = result[key]

    log.info(
        f"Wczytano DataFrame '{key}': {df.shape[0]:,} wierszy × {df.shape[1]:,} kolumn"
    )
    return df


# ─────────────────────────────────────────
# Konwersja .RData → parquet
# ─────────────────────────────────────────
def convert_rdata_to_parquet(
    rdata_path: Path | str = RDATA_FILE,
    parquet_path: Path | str = RAW_PARQUET,
    overwrite: bool = False,
) -> Path:
    """
    Konwertuje .RData → parquet. Jednorazowa, kosztowna operacja
    (.RData jest powoli czytany przez pyreadr).

    Parameters
    ----------
    rdata_path : Path | str
        Wejściowy plik .RData.
    parquet_path : Path | str
        Wyjściowy parquet (domyślnie data/processed/yale_emmlc_raw.parquet).
    overwrite : bool
        Czy nadpisać istniejący parquet.

    Returns
    -------
    Path
        Ścieżka do utworzonego pliku parquet.
    """
    parquet_path = Path(parquet_path)

    if parquet_path.exists() and not overwrite:
        log.info(f"Parquet już istnieje: {parquet_path} (użyj overwrite=True)")
        return parquet_path

    df = load_rdata(rdata_path)

    # Najpierw spróbuj rzucić obiektowe kolumny do `category` (oszczędność miejsca)
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        log.debug(f"Konwersja {len(obj_cols)} kolumn typu object → category…")
        for col in obj_cols:
            try:
                df[col] = df[col].astype("category")
            except Exception:
                # Jeśli się nie da (np. typy mieszane), zostaw object
                pass

    log.info(f"Zapisywanie parquet: {parquet_path}…")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)

    size_mb = parquet_path.stat().st_size / 1e6
    log.info(f"Zapisano parquet: {parquet_path} ({size_mb:.1f} MB)")
    return parquet_path


# ─────────────────────────────────────────
# Główna funkcja ładująca
# ─────────────────────────────────────────
def load_dataset(
    use_processed: bool = False,
    parquet_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Ładuje dataset z najszybszego dostępnego źródła.

    Kolejność preferencji:
        1. processed parquet (po preprocessing) — jeśli `use_processed=True`
        2. raw parquet (po `convert_rdata_to_parquet`)
        3. .RData (najwolniejsze)

    Parameters
    ----------
    use_processed : bool
        Czy preferować przetworzony parquet (po preprocessing).
    parquet_path : Path | str, optional
        Niestandardowa ścieżka do parquet.

    Returns
    -------
    pd.DataFrame
    """
    if parquet_path is not None:
        parquet_path = Path(parquet_path)
        log.info(f"Ładowanie z parquet: {parquet_path}")
        return pd.read_parquet(parquet_path)

    target = PROCESSED_PARQUET if use_processed else RAW_PARQUET

    if target.exists():
        log.info(f"Ładowanie z parquet: {target}")
        return pd.read_parquet(target)

    if RAW_PARQUET.exists():
        log.info(f"Brak {target}, używam {RAW_PARQUET}")
        return pd.read_parquet(RAW_PARQUET)

    log.warning("Brak parquet. Ładowanie z .RData (powolne).")
    return load_rdata()
