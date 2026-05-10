"""
Mapowanie ESI (Emergency Severity Index) → MTS (Manchester Triage System).

Problem (zob. TECHNICAL_ANALYSIS.md §6):
    ESI i MTS to RÓŻNE systemy:
        - ESI: oparty na zużyciu zasobów (resource-based)
        - MTS: oparty na objawach klinicznych i 52 flowchartach (symptom-based)
        - Zgodność: Cohen's κ ≈ 0.51

Strategia 2 (zaimplementowana, REKOMENDOWANA):
    Startujemy z prostego mapowania ESI → MTS, ale WERYFIKUJEMY i UPGRADUJEMY
    klasyfikację na podstawie dyskryminatorów parametrów życiowych z MTS.

    Przykład: ESI-3 z SpO₂ < 92% → upgrade do Orange (nie Yellow).

To kluczowy punkt obrony metodologicznej — czyni mapowanie autentycznie "MTS-owym".
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import CLASS_NAMES, ESI_TO_MTS, ESI_TO_MTS_NUMERIC
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# DYSKRYMINATORY MTS (parametry życiowe)
# ─────────────────────────────────────────
# Progi vital signs zdefiniowane w MTS Flowcharts (kategoria → próg).
# Wartości na podstawie Triage Master Doc (MTSm) i protokołów polskich SOR.

# UPGRADE do RED (Immediate, max wait 0 min)
RED_THRESHOLDS: dict[str, callable] = {
    "triage_o2sat": lambda x: x < 85,                  # krytyczna hipoksja
    "triage_sbp":   lambda x: x < 90,                   # wstrząs
    "triage_pulse": lambda x: x > 150 or x < 40,        # tachy/bradykardia ekstremalna
    "triage_resp":  lambda x: x > 35 or x < 8,          # niewydolność oddechowa
    "triage_temp":  lambda x: x > 41.0 or x < 32.0,     # hipertermia/hipotermia
    "triage_pain":  lambda x: x >= 10,                  # ból nieznośny
}

# UPGRADE do ORANGE (Very Urgent, max wait 10 min)
ORANGE_THRESHOLDS: dict[str, callable] = {
    "triage_o2sat": lambda x: x < 92,
    "triage_sbp":   lambda x: x < 100,
    "triage_pulse": lambda x: x > 130 or x < 50,
    "triage_resp":  lambda x: x > 30,
    "triage_temp":  lambda x: x > 40.0 or x < 34.0,
    "triage_pain":  lambda x: x >= 8,
}

# Typowe etykiety kolumn vital signs w Yale EMMLC (sprawdzane case-insensitive)
VITAL_COL_ALIASES: dict[str, list[str]] = {
    "triage_o2sat": ["triage_vital_o2", "o2sat", "spo2", "triage_o2sat"],
    "triage_sbp":   ["triage_vital_sbp", "sbp", "triage_sbp"],
    "triage_dbp":   ["triage_vital_dbp", "dbp", "triage_dbp"],
    "triage_pulse": ["triage_vital_hr", "pulse", "triage_pulse", "heart_rate"],
    "triage_resp":  ["triage_vital_rr", "resp", "triage_resp"],
    "triage_temp":  ["triage_vital_temp", "temp", "triage_temp"],
    "triage_pain":  ["triage_pain", "pain"],
}


# ─────────────────────────────────────────
# Funkcje publiczne
# ─────────────────────────────────────────
def esi_to_mts_color(esi: int | float) -> str | None:
    """ESI 1–5 → kolor MTS (Red/Orange/Yellow/Green/Blue)."""
    try:
        return ESI_TO_MTS[int(esi)]
    except (KeyError, ValueError, TypeError):
        return None


def esi_to_mts_numeric(esi: int | float) -> int | None:
    """ESI 1–5 → numeryczna klasa MTS 0–4 (0=Red, 4=Blue)."""
    try:
        return ESI_TO_MTS_NUMERIC[int(esi)]
    except (KeyError, ValueError, TypeError):
        return None


def _get_vital(row: pd.Series, canonical: str) -> float | None:
    """Pobiera wartość vital sign sprawdzając kilka możliwych nazw kolumn."""
    aliases = VITAL_COL_ALIASES.get(canonical, [canonical])
    for alias in aliases:
        if alias in row.index:
            value = row[alias]
            if pd.notna(value):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue
    return None


def _passes_thresholds(vitals: dict[str, float | None], thresholds: dict[str, callable]) -> bool:
    """Czy pacjent spełnia którąkolwiek z reguł upgrade'u (any-rule logic)."""
    for vital_name, rule_fn in thresholds.items():
        value = vitals.get(vital_name)
        if value is None:
            continue
        try:
            if rule_fn(value):
                return True
        except (TypeError, ValueError):
            continue
    return False


def enhanced_mts_label(row: pd.Series) -> int:
    """
    STRATEGIA 2 — wzbogacone mapowanie ESI → MTS.

    Zasady:
        1. Start od prostego mapowania ESI → MTS (1:1).
        2. Sprawdź dyskryminatory MTS RED na vital signs — jeśli spełnione,
           upgrade do 0 (Red).
        3. Inaczej sprawdź dyskryminatory ORANGE — upgrade do max(min(klasa, 1), 0).
        4. NIGDY nie obniżaj klasy (downgrade) — bezpieczniej zostawić wyższą pilność.

    Parameters
    ----------
    row : pd.Series
        Wiersz DataFrame zawierający kolumnę 'esi' i kolumny vital signs.

    Returns
    -------
    int
        Numeryczna klasa MTS (0=Red, 1=Orange, 2=Yellow, 3=Green, 4=Blue).
    """
    esi = row.get("esi")
    if pd.isna(esi):
        return -1  # nieznane

    base = ESI_TO_MTS_NUMERIC.get(int(esi))
    if base is None:
        return -1

    # Zbierz vital signs raz
    vitals = {key: _get_vital(row, key) for key in VITAL_COL_ALIASES}

    # Upgrade do Red?
    if base > 0 and _passes_thresholds(vitals, RED_THRESHOLDS):
        return 0

    # Upgrade do Orange?
    if base > 1 and _passes_thresholds(vitals, ORANGE_THRESHOLDS):
        return 1

    return base


def map_dataframe_to_mts(
    df: pd.DataFrame,
    use_enhanced: bool = True,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """
    Dodaje kolumny `mts_color` i `mts_numeric` do DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame z kolumną 'esi' (1-5).
    use_enhanced : bool
        True (domyślne) — Strategia 2 (vital sign discriminators).
        False — proste mapowanie 1:1.
    drop_invalid : bool
        Czy odrzucić wiersze z nieprawidłowym/brakującym ESI.

    Returns
    -------
    pd.DataFrame
        Kopia z dodatkowymi kolumnami.
    """
    if "esi" not in df.columns:
        raise ValueError("DataFrame nie zawiera kolumny 'esi'.")

    df = df.copy()
    n_before = len(df)

    if use_enhanced:
        log.info("Mapowanie ESI → MTS (Strategia 2: wzbogacone dyskryminatorami vital signs)")
        df["mts_numeric"] = df.apply(enhanced_mts_label, axis=1)
    else:
        log.info("Mapowanie ESI → MTS (proste 1:1)")
        df["mts_numeric"] = df["esi"].map(ESI_TO_MTS_NUMERIC).fillna(-1).astype(int)

    df["mts_color"] = df["mts_numeric"].map(
        {i: name for i, name in enumerate(CLASS_NAMES)}
    )

    if drop_invalid:
        df = df[df["mts_numeric"] >= 0].copy()
        n_after = len(df)
        if n_after < n_before:
            log.info(f"Odrzucono {n_before - n_after:,} wierszy z brakującym/nieprawidłowym ESI")

    # Statystyki
    if use_enhanced:
        # Pokaz ile wierszy zostało zupgrade'owanych
        baseline = df["esi"].map(ESI_TO_MTS_NUMERIC)
        upgraded = (df["mts_numeric"] < baseline).sum()
        log.info(f"Strategia 2: upgrade'owano {upgraded:,} wierszy ({100 * upgraded / len(df):.2f}%)")

    log.info("Rozkład klas po mapowaniu:")
    counts = df["mts_color"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    for color, n in counts.items():
        pct = 100 * n / len(df)
        log.info(f"  {color:8s}: {n:>10,} ({pct:5.2f}%)")

    return df
