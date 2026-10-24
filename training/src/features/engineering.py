"""
Inżynieria cech klinicznych dla triażu ESI.

Wejście: ramka w schemacie KANONICZNYM (wynik `src.data.prepare`, albo wiersz
zbudowany przez backend):
    triage_vital_hr / _sbp / _dbp / _rr / _o2 / _o2_device / _temp (°C!)
    age, is_male, arrival_ambulance, arrival_hour_bin (0–5), arrival_weekday (1–7)
    cc_* (skargi główne, nazwy po `sanitize_name`)

Zasady:
    - Temperatura w °C (Yale jest w °F — konwersja dzieje się w prepare).
    - Brak pomiaru ≠ norma: flaga liczona z brakującego parametru jest NaN,
      a o braku informują osobne cechy `*_missing`. Modele drzewiaste (XGBoost,
      LightGBM, CatBoost, RF, EBM) obsługują NaN natywnie.
    - Tylko cechy z uzasadnieniem klinicznym. Przekształcenia monotoniczne
      (x², log x) i przypadkowe iloczyny nic nie wnoszą modelom drzewiastym.
    - Dorośli: Yale zawiera wyłącznie pacjentów ≥ 18 lat.
    - Bez duplikatów: cecha identyczna z inną (np. `esi_danger_hr` ≡ `tachycardia`)
      dzieli ważność SHAP/EBM między dwie nazwy — liczona tylko wewnętrznie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VITALS = {
    "hr": "triage_vital_hr",
    "sbp": "triage_vital_sbp",
    "dbp": "triage_vital_dbp",
    "rr": "triage_vital_rr",
    "o2": "triage_vital_o2",
    "temp": "triage_vital_temp",
}
O2_DEVICE_COL = "triage_vital_o2_device"

# Grupy skarg głównych (nazwy po sanitize_name). Brakujące kolumny = 0.
# `cc_trauma_activation` = scalone cc_fulltrauma + cc_modifiedtrauma (prepare).
CC_CRITICAL = [
    "cc_cardiacarrest", "cc_strokealert", "cc_respiratorydistress",
    "cc_unresponsive", "cc_trauma_activation",
]
CC_GROUPS: dict[str, list[str]] = {
    "cardiac": [
        "cc_cardiacarrest", "cc_chestpain", "cc_chesttightness", "cc_palpitations",
        "cc_irregularheartbeat", "cc_rapidheartrate", "cc_tachycardia",
        "cc_syncope", "cc_nearsyncope", "cc_hypotension",
    ],
    "neuro": [
        "cc_strokealert", "cc_unresponsive", "cc_neurologicproblem",
        "cc_alteredmentalstatus", "cc_confusion", "cc_seizures", "cc_seizure_newonset",
        "cc_seizure_priorhxof", "cc_lossofconsciousness", "cc_extremityweakness",
        "cc_numbness", "cc_headache_newonsetornewsymptoms", "cc_lethargy", "cc_weakness",
    ],
    "respiratory": [
        "cc_respiratorydistress", "cc_shortnessofbreath", "cc_dyspnea",
        "cc_breathingdifficulty", "cc_breathingproblem", "cc_wheezing", "cc_asthma",
        "cc_cough", "cc_hemoptysis",
    ],
    "trauma": [
        "cc_trauma_activation", "cc_trauma", "cc_motorvehiclecrash",
        "cc_motorcyclecrash", "cc_fall", "cc_fall_65", "cc_multiplefalls",
        "cc_headinjury", "cc_assaultvictim",
    ],
    "psych_tox": [
        "cc_suicidal", "cc_homicidal", "cc_psychiatricevaluation", "cc_psychoticsymptoms",
        "cc_hallucinations", "cc_depression", "cc_anxiety", "cc_panicattack",
        "cc_agitation", "cc_alcoholintoxication", "cc_drug_alcoholassessment",
        "cc_withdrawal_alcohol", "cc_overdose_intentional", "cc_overdose_accidental",
        "cc_ingestion", "cc_poisoning", "cc_detoxevaluation", "cc_addictionproblem",
        "cc_alcoholproblem", "cc_drugproblem",
    ],
    "abdominal": [
        "cc_abdominalpain", "cc_abdominalpainpregnant", "cc_abdominalcramping",
        "cc_abdominaldistention", "cc_epigastricpain", "cc_flankpain", "cc_nausea",
        "cc_emesis", "cc_gibleeding", "cc_rectalbleeding", "cc_diarrhea",
    ],
    # Typowe powody wizyt o niskiej pilności (ESI 4–5)
    "low_acuity": [
        "cc_suture_stapleremoval", "cc_woundcheck", "cc_woundre_evaluation",
        "cc_medicationrefill", "cc_dentalpain", "cc_rash", "cc_sorethroat",
        "cc_earpain", "cc_otalgia", "cc_conjunctivitis", "cc_tickremoval",
        "cc_stdcheck", "cc_insectbite", "cc_coldlikesymptoms",
        "cc_follow_upcellulitis", "cc_headachere_evaluation", "cc_eyeredness",
        "cc_skinirritation", "cc_nasalcongestion",
    ],
}


# ─────────────────────────────────────────
# Helpery
# ─────────────────────────────────────────
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Kolumna jako float; brak kolumny → same NaN."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _cc(df: pd.DataFrame, col: str) -> pd.Series:
    """Skarga główna (0/1+); brak kolumny lub NaN → 0."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    return pd.Series(0.0, index=df.index)


def _flag(condition: pd.Series, *sources: pd.Series) -> pd.Series:
    """Warunek → 1.0/0.0, ale NaN tam, gdzie brakuje któregokolwiek źródła."""
    out = condition.astype(float)
    missing = pd.Series(False, index=condition.index)
    for s in sources:
        missing |= s.isna()
    return out.mask(missing)


def _sum_available(parts: list[pd.Series]) -> pd.Series:
    """Suma dostępnych składowych; NaN tylko gdy brakuje wszystkich."""
    frame = pd.concat(parts, axis=1)
    return frame.sum(axis=1, min_count=1)


def _news2_points(value: pd.Series, bins: list[float], points: list[int]) -> pd.Series:
    """Punkty NEWS2 z przedziałów (prawostronnie domkniętych); NaN zostaje NaN."""
    idx = np.digitize(value.to_numpy(dtype=float), bins, right=True)
    pts = np.asarray(points, dtype=float)[np.clip(idx, 0, len(points) - 1)]
    return pd.Series(pts, index=value.index).mask(value.isna())


# ─────────────────────────────────────────
# Główna funkcja
# ─────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca kopię `df` z dołączonymi cechami z `ENGINEERED_FEATURES`."""
    hr = _num(df, VITALS["hr"])
    sbp = _num(df, VITALS["sbp"])
    dbp = _num(df, VITALS["dbp"])
    rr = _num(df, VITALS["rr"])
    o2 = _num(df, VITALS["o2"])
    temp = _num(df, VITALS["temp"])
    o2_device = _num(df, O2_DEVICE_COL)
    age = _num(df, "age")

    c: dict[str, pd.Series] = {}

    # === Hemodynamika ===
    c["pulse_pressure"] = sbp - dbp
    c["map"] = dbp + (sbp - dbp) / 3.0
    c["shock_index"] = hr / sbp.where(sbp > 0)
    c["modified_shock_index"] = hr / c["map"].where(c["map"] > 0)

    c["tachycardia"] = _flag(hr > 100, hr)
    c["severe_tachycardia"] = _flag(hr > 130, hr)
    c["bradycardia"] = _flag(hr < 50, hr)
    c["hypotension"] = _flag(sbp < 100, sbp)
    c["severe_hypotension"] = _flag(sbp < 90, sbp)
    c["severe_hypertension"] = _flag(sbp >= 180, sbp)
    c["hypertensive_crisis"] = _flag((sbp >= 200) | (dbp >= 120), sbp, dbp)
    c["shock_index_critical"] = _flag(c["shock_index"] >= 1.0, c["shock_index"])

    # === Oddech ===
    c["tachypnea"] = _flag(rr > 20, rr)
    c["severe_tachypnea"] = _flag(rr >= 30, rr)
    c["bradypnea"] = _flag(rr < 10, rr)
    c["hypoxia"] = _flag(o2 < 94, o2)
    c["severe_hypoxia"] = _flag(o2 < 90, o2)

    # === Temperatura (°C) ===
    c["fever"] = _flag(temp >= 38.0, temp)
    c["high_fever"] = _flag(temp >= 39.0, temp)
    c["hypothermia"] = _flag(temp < 35.0, temp)           # hipotermia kliniczna (nie próg SIRS)
    c["temp_deviation"] = (temp - 37.0).abs()

    # === Strefa zagrożenia ESI (punkt decyzyjny D, dorośli) ===
    # HR > 100, RR > 20, SpO2 < 92 → rozważ ESI 2. Składowe HR i RR są identyczne
    # z `tachycardia` / `tachypnea`, więc nie są osobnymi cechami.
    c["esi_danger_spo2"] = _flag(o2 < 92, o2)
    c["esi_danger_count"] = _sum_available(
        [c["tachycardia"], c["tachypnea"], c["esi_danger_spo2"]]
    )
    c["esi_danger_zone"] = _flag(c["esi_danger_count"] >= 1, c["esi_danger_count"])

    # === SIRS (bez leukocytów) ===
    sirs_parts = [
        _flag((temp > 38.0) | (temp < 36.0), temp),
        _flag(hr > 90, hr),
        _flag(rr > 20, rr),
    ]
    c["sirs_score"] = _sum_available(sirs_parts)
    c["sirs_positive"] = _flag(c["sirs_score"] >= 2, c["sirs_score"])

    # === qSOFA (zaburzenia świadomości: proxy ze skarg głównych) ===
    ams = ((_cc(df, "cc_alteredmentalstatus") + _cc(df, "cc_confusion")
            + _cc(df, "cc_unresponsive")) > 0).astype(float)
    # Bez RR i SBP wynik byłby samą skargą — brak pomiaru ≠ norma → NaN
    qsofa_parts = [_flag(rr >= 22, rr), _flag(sbp <= 100, sbp), ams]
    c["qsofa_score"] = _sum_available(qsofa_parts).mask(rr.isna() & sbp.isna())
    c["qsofa_positive"] = _flag(c["qsofa_score"] >= 2, c["qsofa_score"])

    # === NEWS2 (bez oceny świadomości; skala SpO2 nr 1) ===
    news_rr = _news2_points(rr, [8, 11, 20, 24], [3, 1, 0, 2, 3])
    news_o2 = _news2_points(o2, [91, 93, 95], [3, 2, 1, 0])
    news_air = (o2_device * 2.0).where(o2_device.notna())
    news_sbp = _news2_points(sbp, [90, 100, 110, 219], [3, 2, 1, 0, 3])
    news_hr = _news2_points(hr, [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
    news_temp = _news2_points(temp, [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
    c["news2_score"] = _sum_available([news_rr, news_o2, news_air, news_sbp, news_hr, news_temp])
    c["news2_high"] = _flag(c["news2_score"] >= 7, c["news2_score"])

    # === Wiek ===
    c["age_65plus"] = _flag(age >= 65, age)
    c["age_80plus"] = _flag(age >= 80, age)

    # === Braki pomiarów (u ESI 1 często brak WSZYSTKICH vitals — pacjent od razu na salę) ===
    missing = {
        name: _num(df, col).isna().astype(float) for name, col in VITALS.items()
    }
    for name, flag in missing.items():
        c[f"{name}_missing"] = flag
    c["n_vitals_missing"] = sum(missing.values())
    c["all_vitals_missing"] = (c["n_vitals_missing"] == len(VITALS)).astype(float)

    # === Czas przybycia ===
    hour_bin = _num(df, "arrival_hour_bin")
    weekday = _num(df, "arrival_weekday")
    c["night_arrival"] = _flag(hour_bin.isin([0, 1]), hour_bin)
    c["weekend_arrival"] = _flag(weekday.isin([6, 7]), weekday)

    # === Skargi główne ===
    # Tylko skargi z danych: pomijamy własne cechy cc_count/cc_critical/cc_group_*,
    # żeby ponowne wywołanie na gotowej ramce dawało ten sam wynik
    cc_cols = [
        col for col in df.columns
        if col.startswith("cc_")
        and col not in ("cc_count", "cc_critical")
        and not col.startswith("cc_group_")
    ]
    if cc_cols:
        cc_frame = df[cc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        c["cc_count"] = (cc_frame > 0).sum(axis=1).astype(float)
    else:
        c["cc_count"] = pd.Series(0.0, index=df.index)
    c["cc_critical"] = sum(_cc(df, col) for col in CC_CRITICAL)
    for group, cols in CC_GROUPS.items():
        c[f"cc_group_{group}"] = sum(_cc(df, col) for col in cols)

    # === Interakcje skarga × parametry ===
    chest = (_cc(df, "cc_chestpain") + _cc(df, "cc_chesttightness")) > 0
    c["chestpain_tachycardia"] = _flag(chest & (hr > 100), hr)
    c["resp_complaint_hypoxia"] = _flag((c["cc_group_respiratory"] > 0) & (o2 < 94), o2)
    c["trauma_shock"] = _flag(
        (c["cc_group_trauma"] > 0) & (c["shock_index"] >= 1.0), c["shock_index"]
    )
    c["ams_fever"] = _flag((ams > 0) & (temp >= 38.0), temp)
    c["elderly_hypotension"] = _flag((age >= 65) & (sbp < 100), age, sbp)
    ambulance = _num(df, "arrival_ambulance")
    c["ambulance_danger_zone"] = _flag(
        (ambulance > 0) & (c["esi_danger_zone"] > 0), ambulance, c["esi_danger_zone"]
    )

    engineered = pd.DataFrame(c, index=df.index)
    base = df.drop(columns=[col for col in engineered.columns if col in df.columns])
    return pd.concat([base, engineered], axis=1)


ENGINEERED_FEATURES: list[str] = [
    # Hemodynamika
    "pulse_pressure", "map", "shock_index", "modified_shock_index",
    "tachycardia", "severe_tachycardia", "bradycardia",
    "hypotension", "severe_hypotension", "severe_hypertension", "hypertensive_crisis",
    "shock_index_critical",
    # Oddech
    "tachypnea", "severe_tachypnea", "bradypnea", "hypoxia", "severe_hypoxia",
    # Temperatura
    "fever", "high_fever", "hypothermia", "temp_deviation",
    # ESI danger zone (składowe HR/RR = tachycardia/tachypnea)
    "esi_danger_spo2", "esi_danger_count", "esi_danger_zone",
    # Skale wczesnego ostrzegania
    "sirs_score", "sirs_positive", "qsofa_score", "qsofa_positive",
    "news2_score", "news2_high",
    # Wiek
    "age_65plus", "age_80plus",
    # Braki
    "hr_missing", "sbp_missing", "dbp_missing", "rr_missing", "o2_missing", "temp_missing",
    "n_vitals_missing", "all_vitals_missing",
    # Czas
    "night_arrival", "weekend_arrival",
    # Skargi główne
    "cc_count", "cc_critical",
    *[f"cc_group_{g}" for g in CC_GROUPS],
    # Interakcje
    "chestpain_tachycardia", "resp_complaint_hypoxia", "trauma_shock",
    "ams_fever", "elderly_hypotension", "ambulance_danger_zone",
]

# Usunięte duplikaty (identyczne z `tachycardia` / `tachypnea`) — do raportu z audytu
REMOVED_DUPLICATE_FEATURES: list[str] = ["esi_danger_hr", "esi_danger_rr"]
