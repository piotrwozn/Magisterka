"""
Przygotowanie danych Yale EMMLC do treningu modeli ESI.

Surowy Yale (972 kolumny, °F, kategorie tekstowe)
    → schemat kanoniczny (°C, jawne kodowania, wykluczone zmienne z USA)
    → cechy kliniczne (`src.features.engineering`)
    → podział train / val / test (+ manifest, listy cech, raport z audytu).

Najważniejsze decyzje (szczegóły w DATA_AUDIT.md po uruchomieniu):
    - Etykieta = natywne ESI 1–5 (target 0–4). Bez mapowania.
    - Temperatura (triaż i historia) °F → °C. SpO2 = 100% przywrócone w triażu
      (filtr Yale skasował każdy odczyt 100%).
    - Wartości fizjologicznie niemożliwe → brak (NaN), NIE mediana.
    - Brak globalnej imputacji: braki zostają NaN (modele drzewiaste je obsługują,
      regresja logistyczna imputuje wewnątrz własnego pipeline'u na zbiorze treningowym).
    - Wykluczone: rasa, etniczność, wyznanie, ubezpieczenie, zatrudnienie, język,
      stan cywilny, oddział (dep_name), `disposition` (wynik wizyty = przeciek),
      leki z bieżącej wizyty (przeciek), kody socjoekonomiczne/administracyjne PMH,
      zepsute badania i D-dimer (mieszane jednostki).
    - `vitals_complete` (meta, NIE cecha): HR, SBP i RR zmierzone w triażu — populacja
      odpowiadająca TOPSOR (pomiary obowiązkowe); ewaluacja raportowana osobno.
    - Nazwy kolumn znormalizowane (bez przecinków/nawiasów — wymóg LightGBM/XGBoost).

Pamięć: surowa ramka (~4,4 GB) nie jest kopiowana — kolumny są czytane pojedynczo
(float32) i składane grupami; pełna ramka powstaje jednym `concat`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import yale_schema as ys
from src.features.engineering import (
    ENGINEERED_FEATURES,
    O2_DEVICE_COL,
    REMOVED_DUPLICATE_FEATURES,
    VITALS,
    engineer_features,
)
from src.utils.config import (
    CLASS_NAMES,
    DATA_AUDIT_MD,
    DATA_MANIFEST_JSON,
    DATA_VERSION,
    ESI_COL,
    ESI_DATA_DIR,
    FEATURE_SETS_JSON,
    HOLDOUT_SITE,
    RANDOM_SEED,
    TARGET_COL,
    TEST_PARQUET,
    TEST_SIZE,
    TOPSOR_COLORS_PL,
    TRAIN_PARQUET,
    VAL_PARQUET,
    VAL_SIZE,
)
from src.utils.fileio import write_text_atomic
from src.utils.logger import get_logger

log = get_logger(__name__)

# Kolumny meta zapisywane w splitach (nigdy nie trafiają do X)
ROW_ID_COL = "row_id"
SITE_COL = "site"
ADMITTED_COL = "admitted"   # wynik wizyty — tylko do analiz, NIE cecha
# 1 = HR, SBP i RR zmierzone w triażu (po czyszczeniu). Populacja TOPSOR — do ewaluacji
# w podgrupie, NIE cecha (brak pomiaru w Yale = decyzja o sali, podjęta po triażu).
VITALS_COMPLETE_COL = "vitals_complete"
VITALS_COMPLETE_SOURCES: list[str] = [VITALS["hr"], VITALS["sbp"], VITALS["rr"]]
META_COLS: list[str] = [ROW_ID_COL, SITE_COL, ESI_COL, TARGET_COL, ADMITTED_COL, VITALS_COMPLETE_COL]

FEATURE_SETS = ("triage", "full")

# Grupy w kolejności cech zestawów (ramka kanoniczna = META_COLS + zestaw 'full')
TRIAGE_GROUPS: tuple[str, ...] = (
    "demographics", "arrival", "time", "triage_vitals", "chief_complaints", "engineered",
)
HISTORY_GROUPS: tuple[str, ...] = (
    "past_medical", "ed_usage", "imaging_history", "history_vitals", "history_labs",
)


@dataclass
class PreparedData:
    frame: pd.DataFrame
    feature_sets: dict[str, list[str]]
    groups: dict[str, list[str]]
    audit: dict = field(default_factory=dict)


# ─────────────────────────────────────────
# Plan kolumn (wyłącznie z listy nazw — bez wczytywania danych)
# ─────────────────────────────────────────
@dataclass
class FeaturePlan:
    """Grupy cech w nazwach kanonicznych (przed normalizacją) + mapa nazw + wykluczenia."""

    raw_groups: dict[str, list[str]]
    rename: dict[str, str]
    excluded: dict[str, list[str]]

    @property
    def groups(self) -> dict[str, list[str]]:
        return {g: [self.rename.get(c, c) for c in cols] for g, cols in self.raw_groups.items()}

    @property
    def feature_sets(self) -> dict[str, list[str]]:
        groups = self.groups
        triage = [c for g in TRIAGE_GROUPS for c in groups[g]]
        full = triage + [c for g in HISTORY_GROUPS for c in groups[g]]
        return {"triage": triage, "full": full}


def plan_features(columns) -> FeaturePlan:
    """
    Które kolumny surowego Yale trafiają do której grupy cech.

    Zależy tylko od listy kolumn, więc liczbę cech można policzyć ze schematu
    parquet (bez wczytywania 4,4 GB danych).
    """
    columns = list(columns)
    pmh_excl = ys.excluded_pmh_cols(columns)
    lab_excl = ys.excluded_lab_cols(columns)
    pmh_drop = {c for cols in pmh_excl.values() for c in cols}
    lab_drop = {c for cols in lab_excl.values() for c in cols}

    raw_groups: dict[str, list[str]] = {
        "demographics": ["age", "is_male"],
        "arrival": [f"arrival_{cat}" for cat in ys.ARRIVAL_CATEGORIES],
        "time": ["arrival_month", "arrival_weekday", "arrival_hour_bin"],
        "triage_vitals": [VITALS[k] for k in VITALS] + [O2_DEVICE_COL],
        "chief_complaints": ys.canonical_chief_complaints(columns),
        "past_medical": [c for c in ys.pmh_cols(columns) if c not in pmh_drop],
        "ed_usage": list(ys.ED_USAGE_COLS)
        + [f"prev_dispo_{cat}" for cat in ys.PREV_DISPO_CATEGORIES],
        "imaging_history": list(ys.IMAGING_COLS),
        "history_vitals": ys.history_vital_cols(columns),
        "history_labs": [c for c in ys.lab_cols(columns) if c not in lab_drop],
        "engineered": list(ENGINEERED_FEATURES),
    }
    # Mapa nazw: meta + grupy (bez inżynieryjnych — ich nazwy są gotowe)
    frame_order = META_COLS + [
        c for g in TRIAGE_GROUPS[:-1] + HISTORY_GROUPS for c in raw_groups[g]
    ]
    excluded = {
        "us_specific": [c for c in ys.US_SPECIFIC_COLS if c in columns],
        "medications": ys.medication_cols(columns),
        "pmh_socioeconomic": pmh_excl["socioeconomic"],
        "pmh_admin_codes": pmh_excl["admin_codes"],
        "pmh_low_support": pmh_excl["low_support"],
        "race_based_labs": lab_excl["race_based"],
        "broken_labs": lab_excl["broken"],
        "unit_mixed_labs": lab_excl["unit_mixed"],
        "chief_complaints_merged": [c for c in ys.TRAUMA_ACTIVATION_SOURCES if c in columns],
        "engineered_duplicates": list(REMOVED_DUPLICATE_FEATURES),
    }
    return FeaturePlan(
        raw_groups=raw_groups,
        rename=ys.sanitize_columns(frame_order),
        excluded=excluded,
    )


# ─────────────────────────────────────────
# Krok 1: surowy Yale → schemat kanoniczny
# ─────────────────────────────────────────
def _clip_to_nan(values: pd.Series, kind: str, audit_key: str, audit: dict) -> pd.Series:
    lo, hi = ys.VITAL_BOUNDS[kind]
    bad = values.notna() & ((values < lo) | (values > hi))
    audit.setdefault("implausible_to_nan", {})[audit_key] = int(bad.sum())
    return values.mask(bad)


def _numeric(raw: pd.DataFrame, rows: np.ndarray, col: str) -> np.ndarray:
    """Jedna surowa kolumna → float32 dla zachowanych wierszy (bez kopii całej ramki)."""
    s = raw[col]
    if pd.api.types.is_numeric_dtype(s.dtype) and not isinstance(s.dtype, pd.CategoricalDtype):
        return s.to_numpy(dtype=np.float32, na_value=np.nan)[rows]
    values = pd.Series(s.astype("object").to_numpy()[rows])
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)


def _numeric_block(
    raw: pd.DataFrame, rows: np.ndarray, cols: list[str], fill: float | None = None,
) -> dict[str, np.ndarray]:
    """Grupa kolumn → {nazwa: float32}, kolumna po kolumnie (niski szczyt pamięci)."""
    out: dict[str, np.ndarray] = {}
    for col in cols:
        values = _numeric(raw, rows, col)
        if fill is not None:
            values[np.isnan(values)] = fill
        out[col] = values
    return out


def _categorical(raw: pd.DataFrame, rows: np.ndarray, col: str) -> pd.Series:
    """Kolumna tekstowa/kategoryczna → Series obiektów (None = brak) dla zachowanych wierszy."""
    return pd.Series(raw[col].astype("object").to_numpy()[rows], dtype="object")


def _arrival_block(raw: pd.DataFrame, rows: np.ndarray, audit: dict) -> pd.DataFrame:
    """Sposób przybycia (one-hot) + miesiąc, dzień tygodnia, przedział godzin."""
    arrival_raw = _categorical(raw, rows, "arrivalmode")
    mode = arrival_raw.map(ys.ARRIVAL_MODE_MAP)
    unmapped = arrival_raw.notna() & mode.isna()
    if unmapped.any():
        log.warning(f"Nieznane wartości arrivalmode → 'other': "
                    f"{sorted(arrival_raw[unmapped].astype(str).unique())}")
        mode = mode.where(~unmapped, "other")
    audit["arrival_missing_imputed"] = int(mode.isna().sum())
    audit["arrival_missing_as"] = ys.ARRIVAL_MISSING_AS
    mode = mode.fillna(ys.ARRIVAL_MISSING_AS)
    audit["arrival_mode_counts"] = mode.value_counts().to_dict()
    audit["arrival_mode_raw_counts"] = arrival_raw.fillna("<brak>").value_counts().to_dict()

    arrival = pd.DataFrame(
        {f"arrival_{cat}": (mode == cat).to_numpy(dtype=np.float32) for cat in ys.ARRIVAL_CATEGORIES}
    )
    for col, raw_col, mapping in [
        ("arrival_month", "arrivalmonth", ys.MONTH_MAP),
        ("arrival_weekday", "arrivalday", ys.WEEKDAY_MAP),
        ("arrival_hour_bin", "arrivalhour_bin", ys.HOUR_BIN_MAP),
    ]:
        arrival[col] = _categorical(raw, rows, raw_col).map(mapping).astype(np.float32)
    return arrival


def _triage_vitals_block(raw: pd.DataFrame, rows: np.ndarray, audit: dict) -> pd.DataFrame:
    """Parametry życiowe z triażu: °F → °C, SpO2 = 100 przywrócone, wartości niemożliwe → NaN."""
    vit = pd.DataFrame(_numeric_block(raw, rows, ys.TRIAGE_VITAL_COLS)).astype(float)
    audit["temp_raw_fahrenheit_median"] = float(vit[VITALS["temp"]].median())
    vit[VITALS["temp"]] = ys.fahrenheit_to_celsius(vit[VITALS["temp"]])

    # Filtr Yale (x ≥ 100 → NA) skasował każdy odczyt SpO2 = 100%. Gdy zmierzono HR
    # (ten sam pulsoksymetr) i SBP, brak SpO2 to prawie zawsze właśnie 100%.
    restored = vit[VITALS["o2"]].isna() & vit[VITALS["hr"]].notna() & vit[VITALS["sbp"]].notna()
    audit["spo2_100_restored"] = int(restored.sum())
    vit.loc[restored, VITALS["o2"]] = ys.SPO2_RESTORED_VALUE

    for kind, col in VITALS.items():
        vit[col] = _clip_to_nan(vit[col], kind, col, audit)
    inverted = (vit[VITALS["dbp"]] >= vit[VITALS["sbp"]]) & vit[VITALS["sbp"]].notna()
    audit["dbp_ge_sbp_to_nan"] = int(inverted.sum())
    vit.loc[inverted, VITALS["dbp"]] = np.nan
    return vit.astype(np.float32)


def _chief_complaint_block(
    raw: pd.DataFrame, rows: np.ndarray, cc_cols: list[str], audit: dict,
) -> pd.DataFrame:
    """Skargi główne jako flagi 0/1; aktywacje urazowe scalone w jedną flagę."""
    sources = [c for c in ys.TRAUMA_ACTIVATION_SOURCES if c in raw.columns]
    data = _numeric_block(raw, rows, [c for c in cc_cols if c != ys.TRAUMA_ACTIVATION_COL])
    if sources:
        data[ys.TRAUMA_ACTIVATION_COL] = np.fmax.reduce([_numeric(raw, rows, c) for c in sources])

    # Kolumna po kolumnie, w miejscu: brak → 0; Yale liczy powtórzone wpisy skargi
    # (dcast(..., length)) → wartości 2–3, a backend wysyła 0/1 → przycięcie do 1
    no_complaint = np.ones(len(rows), dtype=bool)
    clipped = 0
    for values in data.values():
        missing = np.isnan(values)
        no_complaint &= missing
        repeated = values > 1.0
        clipped += int(repeated.sum())
        values[missing] = 0.0
        values[repeated] = 1.0
    audit["rows_without_chief_complaint"] = int(no_complaint.sum())
    audit["chief_complaint_counts_clipped"] = clipped
    return pd.DataFrame(data, columns=cc_cols)


def _prev_dispo_block(raw: pd.DataFrame, rows: np.ndarray, audit: dict) -> pd.DataFrame:
    """Dyspozycja poprzedniej wizyty → 5 kategorii (one-hot); nieznana wartość → NaN."""
    prev_raw = _categorical(raw, rows, ys.PREVIOUS_DISPO_RAW_COL).fillna(ys.PREV_DISPO_MISSING)
    prev = prev_raw.map(ys.PREV_DISPO_MAP)
    unmapped = prev.isna()
    audit["prev_dispo_raw_counts"] = prev_raw.value_counts().to_dict()
    audit["prev_dispo_unmapped"] = int(unmapped.sum())
    if unmapped.any():
        log.warning(f"Nieznane wartości previousdispo → NaN: "
                    f"{sorted(prev_raw[unmapped].astype(str).unique())}")
    out = pd.DataFrame({
        f"prev_dispo_{cat}": (prev == cat).to_numpy(dtype=np.float32)
        for cat in ys.PREV_DISPO_CATEGORIES
    })
    out.loc[unmapped.to_numpy()] = np.nan
    return out


def _history_vitals_block(
    raw: pd.DataFrame, rows: np.ndarray, cols: list[str], audit: dict,
) -> pd.DataFrame:
    """Vitals z poprzednich wizyt: °F → °C, wartości niemożliwe → NaN."""
    hist = pd.DataFrame(_numeric_block(raw, rows, cols)).astype(float)
    for col in cols:
        kind = next((k for p, k in ys.HISTORY_VITAL_KIND.items() if col.startswith(p)), None)
        if kind == "temp":
            hist[col] = ys.fahrenheit_to_celsius(hist[col])
        if kind is not None:
            hist[col] = _clip_to_nan(hist[col], kind, col, audit)
    return hist.astype(np.float32)


def _labs_block(raw: pd.DataFrame, rows: np.ndarray, cols: list[str], audit: dict) -> pd.DataFrame:
    """Badania z poprzednich wizyt: „nie badano” ≠ „ujemny”, ujednolicone jednostki."""
    data = _numeric_block(raw, rows, cols)

    untested_counts: dict[str, int] = {}
    for base in ys.CATEGORICAL_LABS:
        count = data.get(f"{base}_count")
        if count is None:
            continue
        untested = ~(count > 0)          # 0 lub brak → badania nie wykonano
        untested_counts[base] = int(untested.sum())
        for suffix in ("_last", "_npos"):
            if f"{base}{suffix}" in data:
                data[f"{base}{suffix}"][untested] = np.nan
    audit["categorical_labs_untested_to_nan"] = untested_counts

    harmonized: dict[str, int] = {}
    for col, values in data.items():
        threshold = ys.LAB_UNIT_HARMONIZE.get(ys.lab_base(col))
        if threshold is None:
            continue
        cells = values > threshold          # komórki/µL zamiast 10³/µL
        harmonized[col] = int(cells.sum())
        values[cells] = values[cells] / 1000.0
    audit["lab_units_harmonized"] = harmonized
    return pd.DataFrame(data, columns=cols)


def build_canonical_frame(raw: pd.DataFrame) -> PreparedData:
    """
    Surowy DataFrame Yale → kanoniczny DataFrame + grupy cech + liczniki audytu.

    Oszczędnie z pamięcią: surowa ramka (~4,4 GB) nie jest kopiowana w całości,
    kolumny czytane są pojedynczo (float32), a pełna ramka powstaje jednym `concat`.
    """
    audit: dict = {"n_raw_rows": int(len(raw)), "n_raw_cols": int(raw.shape[1])}
    plan = plan_features(raw.columns)
    rg = plan.raw_groups

    # --- Wiersze: bez ESI i bez duplikatów ---
    esi_all = pd.to_numeric(raw[ys.ESI_RAW_COL].astype("object"), errors="coerce")
    has_esi = esi_all.notna().to_numpy()
    audit["dropped_missing_esi"] = int((~has_esi).sum())
    row_hash = pd.util.hash_pandas_object(raw, index=False)
    duplicate = row_hash.duplicated().to_numpy() & has_esi
    audit["dropped_duplicates"] = int(duplicate.sum())
    rows = np.flatnonzero(has_esi & ~duplicate)
    del row_hash

    meta = pd.DataFrame({
        ROW_ID_COL: np.arange(len(rows), dtype=np.int64),
        SITE_COL: raw[ys.SITE_RAW_COL].astype(str).to_numpy()[rows],
        ESI_COL: esi_all.to_numpy()[rows].astype(int),
    })
    meta[TARGET_COL] = meta[ESI_COL] - 1
    meta[ADMITTED_COL] = (
        raw[ys.DISPOSITION_RAW_COL].astype(str).to_numpy()[rows] == "Admit"
    ).astype(int)

    # --- Demografia (bez zmiennych z USA) ---
    gender = raw["gender"].astype("object").to_numpy()[rows]
    demo = pd.DataFrame({
        "age": _numeric(raw, rows, "age"),
        "is_male": np.where(pd.isna(gender), np.nan, gender == "Male").astype(np.float32),
    })
    del gender

    # --- Przybycie i czas; parametry życiowe z triażu ---
    arrival = _arrival_block(raw, rows, audit)
    vit = _triage_vitals_block(raw, rows, audit)[rg["triage_vitals"]]
    meta[VITALS_COMPLETE_COL] = vit[VITALS_COMPLETE_SOURCES].notna().all(axis=1).astype(np.int8)
    audit["vitals_complete_rows"] = int(meta[VITALS_COMPLETE_COL].sum())

    # --- Skargi główne (Epic, flagi 0/1) ---
    cc = _chief_complaint_block(raw, rows, rg["chief_complaints"], audit)

    # --- Cechy inżynieryjne — tylko z bloków triażu (mała ramka, nie pełna) ---
    triage_base = pd.concat([demo, arrival, vit, cc], axis=1).rename(columns=plan.rename)
    engineered = engineer_features(triage_base)[ENGINEERED_FEATURES].astype(np.float32)
    del triage_base

    # --- Historia pacjenta (zestaw 'full'); wykluczenia → plan.excluded ---
    pmh = pd.DataFrame(_numeric_block(raw, rows, rg["past_medical"], fill=0.0))
    non_binary_pmh = [c for c in rg["past_medical"] if not pmh[c].isin([0.0, 1.0]).all()]
    if non_binary_pmh:
        log.warning(f"Kolumny PMH z wartościami spoza {{0,1}}: {non_binary_pmh[:10]}")
    audit["pmh_non_binary"] = non_binary_pmh

    ed_usage = pd.concat([
        pd.DataFrame(_numeric_block(raw, rows, ys.ED_USAGE_COLS)),
        _prev_dispo_block(raw, rows, audit),
    ], axis=1)
    imaging = pd.DataFrame(_numeric_block(raw, rows, ys.IMAGING_COLS, fill=0.0))
    hist = _history_vitals_block(raw, rows, rg["history_vitals"], audit)
    labs = _labs_block(raw, rows, rg["history_labs"], audit)

    # Kolejność kolumn = META_COLS + zestaw 'full' → zapis splitów bez kopii do przestawienia
    pieces = [meta, demo, arrival, vit, cc, engineered, pmh, ed_usage, imaging, hist, labs]
    del meta, demo, arrival, vit, cc, pmh, ed_usage, imaging, hist, labs, engineered
    out = pd.concat(pieces, axis=1)
    del pieces

    # --- Normalizacja nazw (LightGBM nie przyjmuje , : [ ] itp.) ---
    out = out.rename(columns=plan.rename)
    if list(out.columns) != META_COLS + plan.feature_sets["full"]:
        raise RuntimeError("Kolejność kolumn ramki kanonicznej niezgodna z planem cech")
    audit["renamed_columns"] = {k: v for k, v in plan.rename.items() if k != v}
    for reason, cols in plan.excluded.items():
        audit[f"excluded_{reason}"] = cols

    groups = plan.groups
    feature_sets = plan.feature_sets
    for name, cols in feature_sets.items():
        missing = [c for c in cols if c not in out.columns]
        if missing:
            raise RuntimeError(f"Zestaw '{name}': brak kolumn {missing[:10]}")
        if len(set(cols)) != len(cols):
            raise RuntimeError(f"Zestaw '{name}': zduplikowane cechy")
        leaked = set(cols) & set(META_COLS)
        if leaked:
            raise RuntimeError(f"Zestaw '{name}' zawiera kolumny meta: {leaked}")
    if out.columns.duplicated().any():
        raise RuntimeError(f"Zduplikowane kolumny: {out.columns[out.columns.duplicated()].tolist()}")

    audit["n_rows"] = int(len(out))
    return PreparedData(frame=out, feature_sets=feature_sets, groups=groups, audit=audit)


# ─────────────────────────────────────────
# Krok 2: podział
# ─────────────────────────────────────────
def make_splits(
    df: pd.DataFrame,
    strategy: str = "random",
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """
    'random' — losowo, stratyfikacja po ESI × oddział (domyślnie).
    'site'   — oddział HOLDOUT_SITE w całości jako test (walidacja zewnętrzna),
               val losowo z pozostałych oddziałów.
    """
    # Podział na pozycjach wierszy, a ramka kopiowana raz (pamięć: ~1 × ramka, nie ~3 ×)
    strata = (df[TARGET_COL].astype(str) + "_" + df[SITE_COL].astype(str)).to_numpy()
    positions = np.arange(len(df))

    if strategy == "random":
        rest, test = train_test_split(positions, test_size=test_size, random_state=seed, stratify=strata)
        rel_val = val_size / (1.0 - test_size)
    elif strategy == "site":
        held_out = df[SITE_COL].to_numpy() == HOLDOUT_SITE
        test, rest = positions[held_out], positions[~held_out]
        if len(test) == 0:
            raise ValueError(f"Brak wierszy z oddziału '{HOLDOUT_SITE}'")
        rel_val = val_size / (1.0 - len(test) / len(df))
    else:
        raise ValueError(f"Nieznana strategia podziału: '{strategy}' (random | site)")
    train, val = train_test_split(rest, test_size=rel_val, random_state=seed, stratify=strata[rest])

    # Kolejność wierszy jak w ramce (row_id rosnąco)
    splits = {
        name: df.take(np.sort(idx)).reset_index(drop=True)
        for name, idx in (("train", train), ("val", val), ("test", test))
    }
    ids = [set(s[ROW_ID_COL]) for s in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "Splity się nakładają"
    return splits


# ─────────────────────────────────────────
# Krok 3: zapis + raport
# ─────────────────────────────────────────
def save_prepared(
    prepared: PreparedData,
    splits: dict[str, pd.DataFrame],
    strategy: str,
    source: Path | str,
) -> dict[str, Path]:
    """
    Zapis splitów, list cech i manifestu.

    Manifest jest znacznikiem ukończenia: usuwany na starcie i zapisywany jako
    ostatni, więc przerwane przygotowanie nigdy nie wygląda na kompletne.
    """
    ESI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_MANIFEST_JSON.unlink(missing_ok=True)
    keep = META_COLS + prepared.feature_sets["full"]
    paths = {"train": TRAIN_PARQUET, "val": VAL_PARQUET, "test": TEST_PARQUET}
    for name, path in paths.items():
        frame = splits[name]
        if list(frame.columns) != keep:
            frame = frame[keep]
        frame.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        log.info(f"Zapisano {name}: {path} ({len(frame):,} wierszy)")
        del frame

    write_text_atomic(FEATURE_SETS_JSON, json.dumps(
        {"data_version": DATA_VERSION, "feature_sets": prepared.feature_sets,
         "groups": prepared.groups},
        indent=2, ensure_ascii=False,
    ))

    manifest = {
        "data_version": DATA_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "split_strategy": strategy,
        "holdout_site": HOLDOUT_SITE if strategy == "site" else None,
        "label": "ESI 1–5 (natywne, bez mapowania; najnowsze ESI wizyty — po ponownym "
                 "triażu, jeśli był); target = esi - 1",
        "class_names": CLASS_NAMES,
        "n_rows": {name: int(len(s)) for name, s in splits.items()},
        "n_features": {k: len(v) for k, v in prepared.feature_sets.items()},
        "meta_columns": META_COLS,
        "audit": prepared.audit,
    }
    write_text_atomic(
        DATA_MANIFEST_JSON, json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    return paths


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%" if pd.notna(x) else "—"


def _codes(cols, limit: int | None = None) -> str:
    cols = list(cols)
    shown = cols if limit is None else cols[:limit]
    text = ", ".join(f"`{c}`" for c in shown)
    if limit is not None and len(cols) > limit:
        text += f" … (+{len(cols) - limit})"
    return text or "—"


def _esi_counts(df: pd.DataFrame) -> pd.Series:
    return df[ESI_COL].value_counts().reindex(range(1, 6), fill_value=0)


def write_audit_report(
    prepared: PreparedData,
    splits: dict[str, pd.DataFrame],
    strategy: str,
    path: Path = DATA_AUDIT_MD,
) -> Path:
    """Raport z audytu danych (Markdown) — co zostało sprawdzone i zmienione."""
    df = prepared.frame
    a = prepared.audit
    lines: list[str] = []
    w = lines.append

    w("# Audyt danych Yale EMMLC → ESI")
    w("")
    w(f"Wygenerowano automatycznie przez `scripts/02_preprocess.py` "
      f"(wersja danych `{DATA_VERSION}`, podział `{strategy}`). Nie edytuj ręcznie.")
    w("")
    w("## Źródło i czyszczenie")
    w("")
    w(f"- Surowe dane: {a['n_raw_rows']:,} wizyt × {a['n_raw_cols']} kolumn.")
    w(f"- Usunięto {a['dropped_missing_esi']:,} wizyt bez ESI i {a['dropped_duplicates']} duplikaty.")
    w(f"- Po czyszczeniu: **{a['n_rows']:,} wizyt**, wyłącznie dorośli "
      f"(wiek {df['age'].min():.0f}–{df['age'].max():.0f}).")
    w("- Brak roku wizyty i identyfikatora pacjenta → podział chronologiczny i podział "
      "po pacjentach nie są możliwe; ten sam pacjent może wystąpić w train i test.")
    w("- Etykieta = **najnowsze ESI wizyty**: Yale (`20_cleanmerged.R`) zastępuje ESI "
      "z triażu wartością z ponownego triażu (`second_esi`), jeśli był. Część etykiet może "
      "więc uwzględniać informacje zebrane po triażu (szum etykiety); `second_esi` nie jest "
      "udostępnione, więc nie da się tego cofnąć.")
    w("")

    w("## Etykieta: rozkład ESI")
    w("")
    w("| ESI | kolor TOPSOR | liczba | udział | odsetek przyjęć |")
    w("|---|---|---:|---:|---:|")
    for lvl in range(1, 6):
        sub = df[df[ESI_COL] == lvl]
        w(f"| {lvl} | {TOPSOR_COLORS_PL[lvl - 1]} | {len(sub):,} | {_pct(len(sub) / len(df))} "
          f"| {_pct(sub[ADMITTED_COL].mean())} |")
    w("")
    w("ESI wg oddziału (udział w wierszu):")
    w("")
    ct = pd.crosstab(df[SITE_COL], df[ESI_COL], normalize="index")
    w("| oddział | n | " + " | ".join(f"ESI {c}" for c in ct.columns) + " |")
    w("|---|---:|" + "---:|" * len(ct.columns))
    for site, row in ct.iterrows():
        n = int((df[SITE_COL] == site).sum())
        w(f"| {site} | {n:,} | " + " | ".join(_pct(v) for v in row) + " |")
    w("")

    w("## Parametry życiowe z triażu")
    w("")
    w(f"- Temperatura w Yale jest w **°F** (mediana {a['temp_raw_fahrenheit_median']:.1f} °F) "
      "→ przeliczona na °C (także temperatury z historii).")
    w("- Yale usunął wartości odstające jeszcze przed udostępnieniem danych "
      "(`9_edvitalsclean.R`: x < min lub x ≥ max → brak):")
    w("")
    w("| parametr | zachowane przez Yale |")
    w("|---|---|")
    for name, (lo, hi) in ys.YALE_UPSTREAM_VITAL_BOUNDS.items():
        w(f"| {name} | {lo:g} ≤ x < {hi:g} |")
    w("")
    w(f"- **SpO₂ = 100% przywrócone.** Filtr Yale (x ≥ 100 → brak) skasował każdy odczyt "
      f"100%. Przywrócono SpO₂ = {ys.SPO2_RESTORED_VALUE:g} w **{a['spo2_100_restored']:,}** "
      "wizytach, w których zmierzono HR (ten sam pulsoksymetr) i SBP, a SpO₂ brakowało. "
      "Historia `spo2_*` pozostaje ucięta na 99% (NaN, gdy wszystkie wcześniejsze odczyty "
      "wynosiły 100%) — tego nie da się odtworzyć.")
    w("- Nasze zakresy wiarygodności (poza zakresem → NaN, nie mediana). Filtr Yale jest "
      "zwykle ostrzejszy, więc usuwają niewiele:")
    w("")
    w("| parametr | zakres | wyzerowane |")
    w("|---|---|---:|")
    for kind, col in VITALS.items():
        lo, hi = ys.VITAL_BOUNDS[kind]
        w(f"| `{col}` | {lo:g}–{hi:g} | {a['implausible_to_nan'].get(col, 0)} |")
    w(f"| DBP ≥ SBP | — | {a['dbp_ge_sbp_to_nan']} |")
    w("")
    w("| parametr | braki | mediana | 1. percentyl | 99. percentyl |")
    w("|---|---:|---:|---:|---:|")
    for col in list(VITALS.values()) + [O2_DEVICE_COL]:
        s = df[col]
        w(f"| `{col}` | {_pct(s.isna().mean())} | {s.median():.1f} | "
          f"{s.quantile(0.01):.1f} | {s.quantile(0.99):.1f} |")
    w("")
    w("**Braki parametrów odzwierciedlają organizację pracy Yale, nie stan pacjenta.** "
      "Yale uznaje za pomiar z triażu tylko odczyty sprzed przyjęcia na salę "
      "(`RECORDED_TIME <= roomtime`, `9b_gettriagevitals.R`). Pacjent wprowadzony na salę "
      "przed pomiarem — typowo ESI 1, od razu na salę resuscytacyjną — nie ma żadnych "
      "parametrów. Brak jest więc skutkiem decyzji triażowej i zależy od oddziału. "
      "Braki zostają jako NaN plus flagi `*_missing` (nie mediana). Odsetek wizyt bez "
      "ŻADNEGO parametru życiowego:")
    w("")
    w("| ESI | bez vitals |")
    w("|---|---:|")
    for lvl in range(1, 6):
        sub = df[df[ESI_COL] == lvl]
        w(f"| {lvl} | {_pct(sub['all_vitals_missing'].mean())} |")
    w("")
    w(f"W TOPSOR pomiary są obowiązkowe, więc populacją docelową są wizyty z kompletem "
      f"HR, SBP i RR (`{VITALS_COMPLETE_COL} = 1`, kolumna meta — NIE cecha). Metryki "
      "raportujemy osobno dla tej podgrupy; wyniki na całym zbiorze zawyżają wykrywanie "
      "ESI 1, które w Yale opiera się głównie na wzorcu braków.")
    w("")
    w(f"| `{VITALS_COMPLETE_COL}` | n | " + " | ".join(f"ESI {i}" for i in range(1, 6))
      + " | ESI 1 (n) |")
    w("|---|---:|" + "---:|" * 5 + "---:|")
    for flag in (1, 0):
        sub = df[df[VITALS_COMPLETE_COL] == flag]
        counts = _esi_counts(sub)
        total = max(len(sub), 1)
        w(f"| {flag} | {len(sub):,} | " + " | ".join(_pct(c / total) for c in counts)
          + f" | {int(counts[1]):,} |")
    w("")

    w("## Przybycie i czas")
    w("")
    w("| sposób przybycia (Yale) | liczba | kategoria kanoniczna |")
    w("|---|---:|---|")
    for raw_value, n in a["arrival_mode_raw_counts"].items():
        category = (a["arrival_missing_as"] if raw_value == "<brak>"
                    else ys.ARRIVAL_MODE_MAP.get(raw_value, "other"))
        w(f"| {raw_value} | {n:,} | {category} |")
    w("")
    w(f"- Brak sposobu przybycia ({a['arrival_missing_imputed']:,} wizyt) → "
      f"`{a['arrival_missing_as']}`, tak jak w backendzie. W Yale to prawie wyłącznie "
      "oddział A: ścieżka typu fast-track dla studentów i pracowników, wyznaczona "
      "ubezpieczeniem i zatrudnieniem w USA (64% wizyt ESI 5 tego oddziału). Osobna "
      "kategoria `arrival_unknown` uczyłaby model tej ścieżki, a w TOPSOR nigdy nie występuje.")
    w("- Policja → `other` (kilkadziesiąt wizyt — osobna kolumna byłaby praktycznie stała).")
    w("- Miesiąc, dzień tygodnia i przedział godzin zakodowane jawnie (1–12, 1–7 od "
      "poniedziałku, 0–5 od 23:00).")
    w("")

    w("## Skargi główne")
    w("")
    w(f"- Flagi 0/1: Yale zlicza powtórzone wpisy skargi (wartości 2–3) — przycięto "
      f"{a['chief_complaint_counts_clipped']:,} wartości do 1 (backend wysyła 0/1).")
    if a.get("excluded_chief_complaints_merged"):
        w(f"- {_codes(a['excluded_chief_complaints_merged'])} → `{ys.TRAUMA_ACTIVATION_COL}`: "
          "pełna i zmodyfikowana aktywacja zespołu urazowego to dwustopniowy protokół "
          "oddziału A Yale, nieodtwarzalny z polskich danych. `cc_strokealert` zostaje.")
    w(f"- Wizyty bez żadnej skargi głównej: {a['rows_without_chief_complaint']:,} "
      "(skargi uzupełnione zerami).")
    w("")

    w("## Wykluczone kolumny")
    w("")
    w(f"- Specyficzne dla USA lub wrażliwe (niedostępne w polskim SOR, ryzyko uprzedzeń): "
      f"{_codes(a['excluded_us_specific'])}.")
    w("- `dep_name` (oddział) — tylko do stratyfikacji i podziału `--split site`, nie jako cecha.")
    w("- `disposition` (przyjęcie/wypis) — wynik wizyty, zapisany jako `admitted` "
      "tylko do analiz; nigdy nie jest cechą.")
    w(f"- eGFR liczony z rasą: {_codes(a['excluded_race_based_labs'])}.")
    w(f"- Leki `meds_*` (liczba kolumn: {len(a['excluded_medications'])}): lista leków powstaje "
      "głównie przy przyjęciu do szpitala (rekoncyliacja lekowa), czyli po triażu — "
      "przeciek wyniku bieżącej wizyty.")
    w(f"- PMH socjoekonomiczne: {_codes(a['excluded_pmh_socioeconomic'])} (kody V "
      "dotyczące mieszkania, sytuacji materialnej i zatrudnienia — wskaźnik ubezpieczenia).")
    w(f"- PMH administracyjne: {_codes(a['excluded_pmh_admin_codes'])} (powód wizyty, "
      "nie choroba).")
    w(f"- PMH o małym wsparciu ({len(a['excluded_pmh_low_support'])}, stała lista): "
      f"{_codes(a['excluded_pmh_low_support'], limit=8)} — < 50 pozytywnych w train "
      "(kody okołoporodowe u dorosłych, kody E).")
    w(f"- Zepsute badania ({len(a['excluded_broken_labs'])}): "
      f"{_codes(a['excluded_broken_labs'], limit=8)} — puste, stałe, śmieciowe wartości "
      "lub mieszane skale.")
    w(f"- D-dimer ({len(a['excluded_unit_mixed_labs'])}): {_codes(a['excluded_unit_mixed_labs'])}"
      " — mieszane jednostki (µg/mL FEU obok ng/mL).")
    w(f"- Duplikaty cech inżynieryjnych: {_codes(a['excluded_engineered_duplicates'])} "
      "(identyczne z `tachycardia` / `tachypnea`; liczone tylko wewnętrznie dla "
      "`esi_danger_count`).")
    w("")

    w("## Historia pacjenta (zestaw `full`)")
    w("")
    w("- `previousdispo` (dyspozycja POPRZEDNIEJ wizyty) → 5 kategorii: "
      + ", ".join(f"`prev_dispo_{c}`" for c in ys.PREV_DISPO_CATEGORIES)
      + ". Observation (status Medicare) i Send to L&D (ścieżka położnicza USA) → przyjęcie; "
      "AMA, ucieczka i LWBS → `left_early`.")
    if a.get("prev_dispo_unmapped"):
        w(f"- Nieznane wartości `previousdispo`: {a['prev_dispo_unmapped']:,} → NaN.")
    untested = a.get("categorical_labs_untested_to_nan", {})
    if untested:
        w("- Badania jakościowe (`_last`, `_npos`): Yale wpisał 0 także, gdy badania nie "
          "wykonano; przy `_count == 0` → NaN (nie badano ≠ wynik ujemny). Wiersze: "
          + ", ".join(f"`{k}` {v:,}" for k, v in untested.items()) + ".")
    harmonized = a.get("lab_units_harmonized", {})
    if harmonized:
        w("- Ujednolicone jednostki (komórki/µL → 10³/µL, wartość > próg / 1000): "
          + ", ".join(f"`{k}` {v:,}" for k, v in harmonized.items()) + ".")
    w("- Badania są w jednostkach konwencjonalnych USA — przed użyciem modelu `full` "
      "z polskimi wynikami trzeba je przeliczyć:")
    w("")
    w("| badanie | jednostka w Yale (przeliczenie) |")
    w("|---|---|")
    for analyte, unit in ys.LAB_UNITS_US.items():
        w(f"| {analyte} | {unit} |")
    w("")

    w("## Zestawy cech")
    w("")
    w("| grupa | liczba cech | w zestawie |")
    w("|---|---:|---|")
    for name, cols in prepared.groups.items():
        w(f"| {name} | {len(cols)} | {'triage + full' if name in TRIAGE_GROUPS else 'full'} |")
    w(f"| **triage** | **{len(prepared.feature_sets['triage'])}** | dostępne w chwili triażu |")
    w(f"| **full** | **{len(prepared.feature_sets['full'])}** | triaż + historia pacjenta |")
    w("")

    w(f"## Podział `{strategy}`")
    w("")
    w("| zbiór | n | " + " | ".join(f"ESI {i}" for i in range(1, 6)) + " | oddziały |")
    w("|---|---:|" + "---:|" * 5 + "---|")
    for name, s in splits.items():
        dist = s[ESI_COL].value_counts(normalize=True).reindex(range(1, 6), fill_value=0)
        sites = ", ".join(f"{k}: {v:,}" for k, v in s[SITE_COL].value_counts().sort_index().items())
        w(f"| {name} | {len(s):,} | " + " | ".join(_pct(v) for v in dist) + f" | {sites} |")
    w("")
    w("Oddziały w każdym zbiorze (liczba wizyt wg ESI; ESI 1 decyduje o tym, czy metryki "
      "ESI 1 da się w ogóle policzyć):")
    w("")
    w("| zbiór | oddział | n | " + " | ".join(f"ESI {i}" for i in range(1, 6))
      + f" | `{VITALS_COMPLETE_COL}` |")
    w("|---|---|---:|" + "---:|" * 5 + "---:|")
    for name, s in splits.items():
        for site, sub in s.groupby(SITE_COL, sort=True):
            counts = _esi_counts(sub)
            w(f"| {name} | {site} | {len(sub):,} | "
              + " | ".join(f"{int(c):,}" for c in counts)
              + f" | {_pct(sub[VITALS_COMPLETE_COL].mean())} |")
    w("")
    if strategy == "site":
        w(f"- `--split site`: oddział **{HOLDOUT_SITE}** w całości jako test (walidacja "
          "zewnętrzna). Oddział C ma za mało wizyt ESI 1 (≈ 20), by go tak użyć.")
    w("- `train` — trening modeli (10% z niego wydzielane w locie na early stopping).")
    w("- `val` — porównanie modeli i ręczne łączenie (wagi ensemble, progi decyzyjne).")
    w("- `test` — tylko ostateczna ocena; nie dostrajać na nim niczego.")
    w("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"Raport z audytu danych: {path}")
    return path


# ─────────────────────────────────────────
# Całość
# ─────────────────────────────────────────
def prepare_dataset(strategy: str = "random", raw: pd.DataFrame | None = None) -> dict[str, Path]:
    """Pełne przygotowanie: surowy Yale → splity ESI + manifest + raport."""
    from src.data.load_data import load_dataset
    from src.utils.config import RAW_PARQUET

    if raw is None:
        raw = load_dataset()
    log.info(f"Wejście: {raw.shape[0]:,} × {raw.shape[1]:,}")

    prepared = build_canonical_frame(raw)
    del raw
    log.info(f"Kanoniczny zbiór: {prepared.frame.shape[0]:,} wierszy; "
             f"cechy triage={len(prepared.feature_sets['triage'])}, "
             f"full={len(prepared.feature_sets['full'])}")

    splits = make_splits(prepared.frame, strategy=strategy)
    paths = save_prepared(prepared, splits, strategy=strategy, source=RAW_PARQUET)
    write_audit_report(prepared, splits, strategy)
    return paths
