"""
Przygotowanie danych Yale EMMLC do treningu modeli ESI.

Surowy Yale (972 kolumny, °F, kategorie tekstowe)
    → schemat kanoniczny (°C, jawne kodowania, wykluczone zmienne z USA)
    → cechy kliniczne (`src.features.engineering`)
    → podział train / val / test (+ manifest, listy cech, raport z audytu).

Najważniejsze decyzje (szczegóły w DATA_AUDIT.md po uruchomieniu):
    - Etykieta = natywne ESI 1–5 (target 0–4). Bez mapowania.
    - Temperatura (triaż i historia) °F → °C.
    - Wartości fizjologicznie niemożliwe → brak (NaN), NIE mediana.
    - Brak globalnej imputacji: braki zostają NaN (modele drzewiaste je obsługują,
      regresja logistyczna imputuje wewnątrz własnego pipeline'u na zbiorze treningowym).
    - Wykluczone: rasa, etniczność, wyznanie, ubezpieczenie, zatrudnienie, język,
      stan cywilny, oddział (dep_name) oraz `disposition` (wynik wizyty = przeciek).
    - Nazwy kolumn znormalizowane (bez przecinków/nawiasów — wymóg LightGBM/XGBoost).
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
from src.features.engineering import ENGINEERED_FEATURES, VITALS, engineer_features
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
from src.utils.logger import get_logger

log = get_logger(__name__)

# Kolumny meta zapisywane w splitach (nigdy nie trafiają do X)
ROW_ID_COL = "row_id"
SITE_COL = "site"
ADMITTED_COL = "admitted"   # wynik wizyty — tylko do analiz, NIE cecha
META_COLS: list[str] = [ROW_ID_COL, SITE_COL, ESI_COL, TARGET_COL, ADMITTED_COL]

FEATURE_SETS = ("triage", "full")


@dataclass
class PreparedData:
    frame: pd.DataFrame
    feature_sets: dict[str, list[str]]
    groups: dict[str, list[str]]
    audit: dict = field(default_factory=dict)


# ─────────────────────────────────────────
# Krok 1: surowy Yale → schemat kanoniczny
# ─────────────────────────────────────────
def _clip_to_nan(values: pd.Series, kind: str, audit_key: str, audit: dict) -> pd.Series:
    lo, hi = ys.VITAL_BOUNDS[kind]
    bad = values.notna() & ((values < lo) | (values > hi))
    audit.setdefault("implausible_to_nan", {})[audit_key] = int(bad.sum())
    return values.mask(bad)


def build_canonical_frame(raw: pd.DataFrame) -> PreparedData:
    """
    Surowy DataFrame Yale → kanoniczny DataFrame + grupy cech + liczniki audytu.

    Oszczędnie z pamięcią: surowa ramka (~4,4 GB) nie jest kopiowana w całości,
    kolumny wybierane są grupami, a cechy zapisywane jako float32.
    """
    audit: dict = {"n_raw_rows": int(len(raw)), "n_raw_cols": int(raw.shape[1])}

    # --- Wiersze: bez ESI i bez duplikatów ---
    esi_all = pd.to_numeric(raw[ys.ESI_RAW_COL].astype("object"), errors="coerce")
    has_esi = esi_all.notna().to_numpy()
    audit["dropped_missing_esi"] = int((~has_esi).sum())
    row_hash = pd.util.hash_pandas_object(raw, index=False)
    duplicate = row_hash.duplicated().to_numpy() & has_esi
    audit["dropped_duplicates"] = int(duplicate.sum())
    rows = np.flatnonzero(has_esi & ~duplicate)
    del row_hash

    def take(cols) -> pd.DataFrame:
        """Wybrane kolumny, tylko zachowane wiersze (bez kopiowania całej ramki)."""
        return raw[list(cols)].iloc[rows].reset_index(drop=True)

    def take_numeric(cols, fill: float | None = None) -> pd.DataFrame:
        frame = take(cols).apply(pd.to_numeric, errors="coerce").astype(np.float32)
        return frame.fillna(fill) if fill is not None else frame

    pieces: list[pd.DataFrame] = []
    meta = pd.DataFrame({
        ROW_ID_COL: np.arange(len(rows), dtype=np.int64),
        SITE_COL: raw[ys.SITE_RAW_COL].astype(str).to_numpy()[rows],
        ESI_COL: esi_all.to_numpy()[rows].astype(int),
    })
    meta[TARGET_COL] = meta[ESI_COL] - 1
    meta[ADMITTED_COL] = (
        raw[ys.DISPOSITION_RAW_COL].astype(str).to_numpy()[rows] == "Admit"
    ).astype(int)
    pieces.append(meta)

    # --- Demografia (bez zmiennych z USA) ---
    demo = pd.DataFrame(index=meta.index)
    demo["age"] = take_numeric(["age"])["age"]
    gender = raw["gender"].astype("object").to_numpy()[rows]
    demo["is_male"] = np.where(pd.isna(gender), np.nan, gender == "Male").astype(np.float32)
    audit["excluded_us_specific"] = [c for c in ys.US_SPECIFIC_COLS if c in raw.columns]
    pieces.append(demo)

    # --- Przybycie i czas ---
    arrival_raw = pd.Series(raw["arrivalmode"].astype("object").to_numpy()[rows])
    mode = arrival_raw.map(ys.ARRIVAL_MODE_MAP)
    unmapped = arrival_raw.notna() & mode.isna()
    if unmapped.any():
        log.warning(f"Nieznane wartości arrivalmode → 'other': "
                    f"{sorted(arrival_raw[unmapped].astype(str).unique())}")
        mode = mode.where(~unmapped, "other")
    mode = mode.fillna("unknown")
    audit["arrival_mode_counts"] = mode.value_counts().to_dict()
    audit["arrival_mode_raw_counts"] = arrival_raw.fillna("<brak>").value_counts().to_dict()
    arrival = pd.DataFrame(
        {f"arrival_{cat}": (mode == cat).astype(np.float32) for cat in ys.ARRIVAL_CATEGORIES}
    )
    for col, raw_col, mapping in [
        ("arrival_month", "arrivalmonth", ys.MONTH_MAP),
        ("arrival_weekday", "arrivalday", ys.WEEKDAY_MAP),
        ("arrival_hour_bin", "arrivalhour_bin", ys.HOUR_BIN_MAP),
    ]:
        values = pd.Series(raw[raw_col].astype("object").to_numpy()[rows])
        arrival[col] = values.map(mapping).astype(np.float32)
    pieces.append(arrival)

    # --- Parametry życiowe z triażu ---
    vit = take_numeric(ys.TRIAGE_VITAL_COLS).astype(float)
    audit["temp_raw_fahrenheit_median"] = float(vit["triage_vital_temp"].median())
    vit["triage_vital_temp"] = ys.fahrenheit_to_celsius(vit["triage_vital_temp"])
    for kind, col in VITALS.items():
        vit[col] = _clip_to_nan(vit[col], kind, col, audit)
    inverted = (vit[VITALS["dbp"]] >= vit[VITALS["sbp"]]) & vit[VITALS["sbp"]].notna()
    audit["dbp_ge_sbp_to_nan"] = int(inverted.sum())
    vit.loc[inverted, VITALS["dbp"]] = np.nan
    pieces.append(vit.astype(np.float32))

    # --- Skargi główne (Epic, 200 flag) ---
    cc_raw = ys.chief_complaint_cols(raw.columns)
    cc = take_numeric(cc_raw)
    audit["rows_without_chief_complaint"] = int(cc.isna().all(axis=1).sum())
    pieces.append(cc.fillna(0.0))
    del cc

    # --- Historia pacjenta (zestaw 'full') ---
    pmh_raw = ys.pmh_cols(raw.columns)
    meds_raw = ys.medication_cols(raw.columns)
    hist_vitals_raw = ys.history_vital_cols(raw.columns)
    labs_all = ys.lab_cols(raw.columns)
    race_labs = ys.race_based_lab_cols(raw.columns)
    labs_raw = [c for c in labs_all if c not in race_labs]
    audit["excluded_race_based_labs"] = race_labs

    pmh = take_numeric(pmh_raw, fill=0.0)
    non_binary_pmh = [c for c in pmh_raw if not pmh[c].isin([0.0, 1.0]).all()]
    if non_binary_pmh:
        log.warning(f"Kolumny PMH z wartościami spoza {{0,1}}: {non_binary_pmh[:10]}")
    audit["pmh_non_binary"] = non_binary_pmh
    pieces.append(pmh)
    pieces.append(take_numeric(meds_raw, fill=0.0))
    pieces.append(take_numeric(ys.ED_USAGE_COLS))
    pieces.append(take_numeric(ys.IMAGING_COLS, fill=0.0))

    prev = pd.Series(raw[ys.PREVIOUS_DISPO_RAW_COL].astype("object").to_numpy()[rows])
    prev = prev.fillna("No previous dispo")
    prev_cols = [f"prev_dispo_{value}" for value in sorted(prev.unique())]
    pieces.append(pd.DataFrame(
        {col: (prev == col[len("prev_dispo_"):]).astype(np.float32) for col in prev_cols}
    ))

    hist = take_numeric(hist_vitals_raw).astype(float)
    for col in hist_vitals_raw:
        kind = next((k for p, k in ys.HISTORY_VITAL_KIND.items() if col.startswith(p)), None)
        if kind == "temp":
            hist[col] = ys.fahrenheit_to_celsius(hist[col])
        if kind is not None:
            hist[col] = _clip_to_nan(hist[col], kind, col, audit)
    pieces.append(hist.astype(np.float32))
    pieces.append(take_numeric(labs_raw))

    out = pd.concat(pieces, axis=1)
    del pieces

    # --- Normalizacja nazw (LightGBM nie przyjmuje , : [ ] itp.) ---
    rename = ys.sanitize_columns(out.columns)
    out = out.rename(columns=rename)
    audit["renamed_columns"] = {k: v for k, v in rename.items() if k != v}

    def _r(cols: list[str]) -> list[str]:
        return [rename[c] for c in cols]

    groups: dict[str, list[str]] = {
        "demographics": ["age", "is_male"],
        "arrival": [f"arrival_{cat}" for cat in ys.ARRIVAL_CATEGORIES],
        "time": ["arrival_month", "arrival_weekday", "arrival_hour_bin"],
        "triage_vitals": [VITALS[k] for k in VITALS] + ["triage_vital_o2_device"],
        "chief_complaints": _r(cc_raw),
        "engineered": list(ENGINEERED_FEATURES),
        "past_medical": _r(pmh_raw),
        "medications": _r(meds_raw),
        "ed_usage": _r(ys.ED_USAGE_COLS) + _r(prev_cols),
        "imaging_history": _r(ys.IMAGING_COLS),
        "history_vitals": _r(hist_vitals_raw),
        "history_labs": _r(labs_raw),
    }

    # --- Cechy inżynieryjne (na kanonicznych nazwach) ---
    engineered = engineer_features(out[groups["demographics"] + groups["arrival"]
                                       + groups["time"] + groups["triage_vitals"]
                                       + groups["chief_complaints"]])
    out = pd.concat([out, engineered[ENGINEERED_FEATURES].astype(np.float32)], axis=1)
    del engineered

    triage = (
        groups["demographics"] + groups["arrival"] + groups["time"]
        + groups["triage_vitals"] + groups["chief_complaints"] + groups["engineered"]
    )
    full = triage + (
        groups["past_medical"] + groups["medications"] + groups["ed_usage"]
        + groups["imaging_history"] + groups["history_vitals"] + groups["history_labs"]
    )
    feature_sets = {"triage": triage, "full": full}

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
    strata = df[TARGET_COL].astype(str) + "_" + df[SITE_COL].astype(str)

    if strategy == "random":
        rest, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)
        rel_val = val_size / (1.0 - test_size)
        train, val = train_test_split(
            rest, test_size=rel_val, random_state=seed, stratify=strata.loc[rest.index]
        )
    elif strategy == "site":
        test = df[df[SITE_COL] == HOLDOUT_SITE]
        rest = df[df[SITE_COL] != HOLDOUT_SITE]
        if test.empty:
            raise ValueError(f"Brak wierszy z oddziału '{HOLDOUT_SITE}'")
        rel_val = val_size / (1.0 - len(test) / len(df))
        train, val = train_test_split(
            rest, test_size=rel_val, random_state=seed, stratify=strata.loc[rest.index]
        )
    else:
        raise ValueError(f"Nieznana strategia podziału: '{strategy}' (random | site)")

    splits = {
        "train": train.sort_values(ROW_ID_COL).reset_index(drop=True),
        "val": val.sort_values(ROW_ID_COL).reset_index(drop=True),
        "test": test.sort_values(ROW_ID_COL).reset_index(drop=True),
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
    ESI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    keep = META_COLS + prepared.feature_sets["full"]
    paths = {"train": TRAIN_PARQUET, "val": VAL_PARQUET, "test": TEST_PARQUET}
    for name, path in paths.items():
        splits[name][keep].to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        log.info(f"Zapisano {name}: {path} ({len(splits[name]):,} wierszy)")

    FEATURE_SETS_JSON.write_text(json.dumps(
        {"feature_sets": prepared.feature_sets, "groups": prepared.groups},
        indent=2, ensure_ascii=False,
    ))

    manifest = {
        "data_version": DATA_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "split_strategy": strategy,
        "label": "ESI 1–5 (natywne, bez mapowania); target = esi - 1",
        "class_names": CLASS_NAMES,
        "n_rows": {name: int(len(s)) for name, s in splits.items()},
        "n_features": {k: len(v) for k, v in prepared.feature_sets.items()},
        "meta_columns": META_COLS,
        "audit": prepared.audit,
    }
    DATA_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return paths


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


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
    w("- Wartości poza zakresem fizjologicznym zamienione na brak (NaN), nie na medianę:")
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
    for col in list(VITALS.values()) + ["triage_vital_o2_device"]:
        s = df[col]
        w(f"| `{col}` | {_pct(s.isna().mean())} | {s.median():.1f} | "
          f"{s.quantile(0.01):.1f} | {s.quantile(0.99):.1f} |")
    w("")
    w("**Braki są informatywne.** Odsetek wizyt bez ŻADNEGO parametru życiowego:")
    w("")
    w("| ESI | bez vitals |")
    w("|---|---:|")
    for lvl in range(1, 6):
        sub = df[df[ESI_COL] == lvl]
        w(f"| {lvl} | {_pct(sub['all_vitals_missing'].mean())} |")
    w("")
    w("Pacjenci ESI 1 zwykle trafiają od razu na salę resuscytacyjną, bez pomiaru w triażu. "
      "Dlatego braki zostają jako NaN plus flagi `*_missing`, a nie jako mediana. "
      "Uwaga przy wdrożeniu: w TOPSOR pomiary są obowiązkowe, więc ten wzorzec może się różnić.")
    w("")

    w("## Przybycie i czas")
    w("")
    w("| sposób przybycia (Yale) | liczba | kategoria kanoniczna |")
    w("|---|---:|---|")
    for raw_value, n in a["arrival_mode_raw_counts"].items():
        w(f"| {raw_value} | {n:,} | {ys.ARRIVAL_MODE_MAP.get(raw_value, 'unknown')} |")
    w("")
    w("Miesiąc, dzień tygodnia i przedział godzin zakodowane jawnie (1–12, 1–7 od poniedziałku, "
      "0–5 od 23:00). Wcześniej LabelEncoder kodował je alfabetycznie, przez co "
      "`ambulance_flag`, `weekend_flag` i pory dnia liczyły się na złych kategoriach.")
    w("")

    w("## Wykluczone kolumny")
    w("")
    w(f"- Specyficzne dla USA: {', '.join(f'`{c}`' for c in a['excluded_us_specific'])}.")
    w("- `dep_name` (oddział) — tylko do stratyfikacji i podziału `--split site`, nie jako cecha.")
    w("- `disposition` (przyjęcie/wypis) — wynik wizyty, zapisany jako `admitted` "
      "tylko do analiz; nigdy nie jest cechą.")
    w(f"- eGFR liczony z rasą: {', '.join(f'`{c}`' for c in a['excluded_race_based_labs'])}.")
    w(f"- Wizyty bez żadnej skargi głównej: {a['rows_without_chief_complaint']:,} "
      "(skargi uzupełnione zerami).")
    w("")

    w("## Zestawy cech")
    w("")
    w("| grupa | liczba cech | w zestawie |")
    w("|---|---:|---|")
    triage_groups = {"demographics", "arrival", "time", "triage_vitals", "chief_complaints", "engineered"}
    for name, cols in prepared.groups.items():
        w(f"| {name} | {len(cols)} | {'triage + full' if name in triage_groups else 'full'} |")
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
