"""Tests for the runtime feature engineering wrapper (canonical ESI triage row)."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.models.schemas import Vitals
from app.pipeline.feature_engineering import (
    ARRIVAL_MODE_ALIASES,
    CHIEF_COMPLAINT_COLUMNS,
    DEFAULT_ARRIVAL_MODE,
    align_features,
    build_feature_row,
    encode_sex,
    normalize_arrival_mode,
)

# Importable only after app.pipeline.feature_engineering put training/ on sys.path
from src.data import yale_schema as ys  # noqa: E402
from src.data.prepare import build_canonical_frame  # noqa: E402
from src.features.engineering import ENGINEERED_FEATURES  # noqa: E402
from src.utils.config import DATA_VERSION  # noqa: E402

FEATURE_SETS_JSON = (
    Path(__file__).resolve().parents[2] / "training" / "data" / "processed" / "esi" / "feature_sets.json"
)

CANONICAL_BASE = [
    "age", "is_male",
    "arrival_ambulance", "arrival_self", "arrival_other",
    "arrival_month", "arrival_weekday", "arrival_hour_bin",
    "triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp", "triage_vital_rr",
    "triage_vital_o2", "triage_vital_o2_device", "triage_vital_temp",
]
ARRIVAL_COLUMNS = ["arrival_ambulance", "arrival_self", "arrival_other"]


def load_trained_feature_sets() -> dict:
    """Real training/data/processed/esi/feature_sets.json — skip if missing or stale."""
    if not FEATURE_SETS_JSON.exists():
        pytest.skip("No training/data/processed/esi/feature_sets.json — run 02_preprocess.py")
    payload = json.loads(FEATURE_SETS_JSON.read_text(encoding="utf-8"))
    version = payload.get("data_version")
    if version != DATA_VERSION:
        pytest.skip(f"feature_sets.json is from data_version {version!r}, the code expects "
                    f"{DATA_VERSION!r} — re-run training/scripts/02_preprocess.py")
    return payload


def legacy_raw_cc_columns() -> list[str]:
    """Raw Yale chief-complaint columns: the model vocabulary before the trauma merge."""
    merged = ys.TRAUMA_ACTIVATION_COL
    return [c for c in CHIEF_COMPLAINT_COLUMNS if c != merged] + list(ys.TRAUMA_ACTIVATION_SOURCES)


def synthetic_raw_yale(cc_columns: list[str], n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Minimal raw Yale frame (real column names/types) for training's prepare."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "esi": pd.Categorical(rng.choice(list("12345"), n), categories=list("12345")),
        "dep_name": rng.choice(["A", "B", "C"], n),
        "disposition": rng.choice(["Admit", "Discharge"], n),
        "age": rng.integers(18, 95, n),
        "gender": rng.choice(["Male", "Female"], n),
        "arrivalmode": rng.choice(["ambulance", "Car", "Police", None], n),
        "arrivalmonth": rng.choice(list(ys.MONTH_MAP), n),
        "arrivalday": rng.choice(list(ys.WEEKDAY_MAP), n),
        "arrivalhour_bin": rng.choice(list(ys.HOUR_BIN_MAP), n),
        "previousdispo": rng.choice(list(ys.PREV_DISPO_MAP), n),
        "triage_vital_hr": rng.normal(85, 15, n),
        "triage_vital_sbp": rng.normal(130, 20, n),
        "triage_vital_dbp": rng.normal(75, 10, n),
        "triage_vital_rr": rng.normal(18, 3, n),
        "triage_vital_o2": np.where(rng.random(n) < 0.3, np.nan, rng.normal(96, 2, n)),
        "triage_vital_o2_device": rng.integers(0, 2, n).astype(float),
        "triage_vital_temp": rng.normal(98.4, 1.0, n),
    })
    extra = {col: rng.choice([0.0, 1.0, 2.0], n, p=[0.9, 0.08, 0.02]) for col in cc_columns}
    extra.update({col: rng.integers(0, 3, n).astype(float)
                  for col in ys.ED_USAGE_COLS + ys.IMAGING_COLS})
    return pd.concat([df, pd.DataFrame(extra)], axis=1)


@pytest.fixture(scope="module")
def synthetic_prepared():
    """Training's prepare on a synthetic raw frame with the full Yale complaint vocabulary."""
    return build_canonical_frame(synthetic_raw_yale(legacy_raw_cc_columns()))


def _vitals(**overrides) -> Vitals:
    values = dict(age=40, temp=36.7, hr=80, sbp=120, dbp=80, rr=16, o2=99)
    values.update(overrides)
    return Vitals(**values)


@pytest.fixture(scope="module")
def triage_features() -> list[str]:
    return load_trained_feature_sets()["feature_sets"]["triage"]


class TestBuildFeatureRow:

    def test_returns_single_row(self):
        df = build_feature_row(_vitals(), {}, {})
        assert df.shape[0] == 1
        assert df.shape[1] == len(CANONICAL_BASE) + len(CHIEF_COMPLAINT_COLUMNS) + len(ENGINEERED_FEATURES)
        assert df.columns.is_unique

    def test_temperature_stays_celsius(self):
        df = build_feature_row(_vitals(temp=37.0), {}, {})
        # No °C → °F conversion: training data are converted to °C in prepare
        assert df["triage_vital_temp"].iloc[0] == pytest.approx(37.0)

    def test_fever_threshold_in_celsius(self):
        df = build_feature_row(_vitals(temp=38.5), {}, {})
        assert df["fever"].iloc[0] == 1.0
        assert df["hypothermia"].iloc[0] == 0.0

    @pytest.mark.parametrize("temp, expected", [(34.9, 1.0), (35.0, 0.0), (35.8, 0.0)])
    def test_hypothermia_below_35(self, temp, expected):
        assert build_feature_row(_vitals(temp=temp), {}, {})["hypothermia"].iloc[0] == expected

    def test_spo2_100_passes_through(self):
        # Training restores Yale's deleted 100% readings → 100 is in-distribution
        row = build_feature_row(_vitals(o2=100), {}, {}).iloc[0]
        assert row["triage_vital_o2"] == 100.0
        assert row["o2_missing"] == 0.0
        assert row["hypoxia"] == 0.0

    def test_contains_canonical_columns(self):
        df = build_feature_row(_vitals(), {}, {})
        for col in CANONICAL_BASE:
            assert col in df.columns, col

    def test_covers_every_trained_triage_feature(self, triage_features):
        df = build_feature_row(_vitals(), {"cc_chestpain": 1}, {})
        missing = [c for c in triage_features if c not in df.columns]
        assert missing == []

    def test_no_legacy_columns(self):
        df = build_feature_row(_vitals(), {}, {})
        for legacy in ("arrivalmode", "arrivalmonth", "arrivalday", "arrivalhour_bin",
                       "arrival_hour", "arrival_dayofweek", "gender"):
            assert legacy not in df.columns

    def test_vitals_passed_through(self):
        df = build_feature_row(_vitals(hr=101, sbp=135, dbp=85, rr=19, o2=96.5), {}, {})
        row = df.iloc[0]
        assert (row["triage_vital_hr"], row["triage_vital_sbp"], row["triage_vital_dbp"],
                row["triage_vital_rr"], row["triage_vital_o2"]) == (101, 135, 85, 19, 96.5)
        assert row["age"] == 40

    def test_implausible_vitals_become_nan(self):
        # Same plausibility bounds as training (temp 30–44 °C; DBP ≥ SBP → missing)
        df = build_feature_row(_vitals(temp=44.5, sbp=90, dbp=95), {}, {})
        row = df.iloc[0]
        assert math.isnan(row["triage_vital_temp"])
        assert math.isnan(row["triage_vital_dbp"])
        assert row["temp_missing"] == 1.0
        # Flags derived from a missing vital are NaN, not 0
        assert math.isnan(row["fever"])

    def test_cc_features_propagated(self):
        df = build_feature_row(_vitals(), {"cc_chestpain": 1}, {})
        assert df["cc_chestpain"].iloc[0] == 1
        assert df["cc_count"].iloc[0] == 1
        assert df["cc_group_cardiac"].iloc[0] == 1

    def test_default_vocabulary_prepopulated_with_zero(self):
        df = build_feature_row(_vitals(), {"cc_chestpain": 1}, {})
        assert len(CHIEF_COMPLAINT_COLUMNS) == 199
        assert "cc_trauma_activation" in CHIEF_COMPLAINT_COLUMNS
        assert not {"cc_fulltrauma", "cc_modifiedtrauma"} & set(CHIEF_COMPLAINT_COLUMNS)
        others = [c for c in CHIEF_COMPLAINT_COLUMNS if c != "cc_chestpain"]
        assert (df[others].iloc[0] == 0).all()

    def test_unknown_cc_codes_dropped(self):
        df = build_feature_row(_vitals(), {"cc_bogus": 1, "cc_seizure": 1}, {})
        assert "cc_bogus" not in df.columns
        assert "cc_seizure" not in df.columns   # real column is cc_seizures
        assert df["cc_count"].iloc[0] == 0

    def test_engineered_cc_names_are_not_complaints(self):
        df = build_feature_row(
            _vitals(),
            {"cc_chestpain": 1, "cc_group_cardiac": 7, "cc_count": 9},  # engineered — ignored
            {},
            all_cc_columns=["cc_chestpain", "cc_group_cardiac", "cc_count", "cc_other"],
        )
        assert df.columns.is_unique
        assert df["cc_group_cardiac"].iloc[0] == 1     # recomputed from cc_chestpain
        assert df["cc_count"].iloc[0] == 1

    def test_all_cc_columns_prepopulated(self):
        cc_list = ["cc_chestpain", "cc_dyspnea", "cc_other"]
        df = build_feature_row(_vitals(), {"cc_chestpain": 1}, {}, all_cc_columns=cc_list)
        assert df["cc_chestpain"].iloc[0] == 1
        assert df["cc_dyspnea"].iloc[0] == 0
        assert df["cc_other"].iloc[0] == 0

    def test_extra_overrides(self):
        df = build_feature_row(_vitals(), {}, {"cc_fever": 1, "triage_pain": 8.0})
        assert df["cc_fever"].iloc[0] == 1
        assert df["triage_pain"].iloc[0] == 8.0   # informational, dropped by align_features

    @pytest.mark.parametrize("cc, extra", [
        ({"cc_fulltrauma": 1}, {}),
        ({}, {"cc_modifiedtrauma": 1}),
        ({"cc_fulltrauma": 1, "cc_modifiedtrauma": 1}, {}),
        ({"cc_trauma_activation": 1}, {}),
    ])
    def test_legacy_trauma_codes_mapped(self, cc, extra):
        row = build_feature_row(_vitals(), cc, extra).iloc[0]
        assert row["cc_trauma_activation"] == 1.0
        assert row["cc_critical"] == 1.0
        assert row["cc_count"] == 1.0
        assert "cc_fulltrauma" not in row.index and "cc_modifiedtrauma" not in row.index

    def test_complaint_values_are_flags(self):
        # Training clips Yale's repeated-complaint counts to 0/1
        row = build_feature_row(_vitals(), {"cc_chestpain": 3}, {}).iloc[0]
        assert row["cc_chestpain"] == 1.0

    def test_engineered_features_computed(self):
        df = build_feature_row(_vitals(hr=120, sbp=90, dbp=60), {}, {})
        assert df["shock_index"].iloc[0] == pytest.approx(120 / 90)
        assert df["tachycardia"].iloc[0] == 1.0
        assert df["esi_danger_count"].iloc[0] == 1.0
        # Duplicates of tachycardia / tachypnea are no longer model features
        assert "esi_danger_hr" not in df.columns and "esi_danger_rr" not in df.columns

    def test_time_features(self):
        # 2026-10-18 = Sunday, 23:15 → night bin 0 ("23-02")
        df = build_feature_row(_vitals(), {}, {}, now=datetime(2026, 10, 18, 23, 15))
        row = df.iloc[0]
        assert (row["arrival_month"], row["arrival_weekday"], row["arrival_hour_bin"]) == (10, 7, 0)
        assert row["weekend_arrival"] == 1.0
        assert row["night_arrival"] == 1.0

        df = build_feature_row(_vitals(), {}, {}, now=datetime(2026, 10, 19, 12, 0))
        row = df.iloc[0]
        assert (row["arrival_weekday"], row["arrival_hour_bin"]) == (1, 3)   # Monday, "11-14"
        assert row["weekend_arrival"] == 0.0

    def test_unknown_request_fields_default(self):
        df = build_feature_row(_vitals(), {}, {})
        row = df.iloc[0]
        # Missing arrival mode → "self", exactly like Yale's missing mode in training
        assert DEFAULT_ARRIVAL_MODE == ys.ARRIVAL_MISSING_AS == "self"
        assert row["arrival_self"] == 1.0
        assert row[["arrival_ambulance", "arrival_other"]].sum() == 0
        assert not {"arrival_unknown", "arrival_police"} & set(row.index)
        assert math.isnan(row["is_male"])
        assert math.isnan(row["triage_vital_o2_device"])

    def test_explicit_request_fields(self):
        df = build_feature_row(_vitals(), {}, {}, is_male="K", arrival_mode="ZRM", o2_device=1)
        row = df.iloc[0]
        assert row["is_male"] == 0.0
        assert row[ARRIVAL_COLUMNS].tolist() == [1.0, 0.0, 0.0]
        assert row["triage_vital_o2_device"] == 1.0

    def test_request_fields_read_from_request_object(self):
        # Forward-compatible: optional fields on the request object are used if present
        req = SimpleNamespace(age=70, temp=36.6, hr=80, sbp=120, dbp=80, rr=16, o2=97,
                              sex="M", arrival_mode="policja", o2_device=0)
        row = build_feature_row(req, {}, {}).iloc[0]
        assert row["is_male"] == 1.0
        assert row[ARRIVAL_COLUMNS].tolist() == [0.0, 0.0, 1.0]     # police → other
        assert row["triage_vital_o2_device"] == 0.0


class TestRequestEncoding:

    @pytest.mark.parametrize("mode, expected", [
        ("ZRM", "ambulance"), ("karetka", "ambulance"), ("Karetka ZRM", "ambulance"),
        ("LPR", "ambulance"), ("ambulance", "ambulance"),
        ("własny", "self"), ("transport własny", "self"), ("Walk-in", "self"), ("Car", "self"),
        ("policja", "other"), ("Police", "other"), ("radiowóz policji", "other"),
        ("Wheelchair", "other"), ("coś zupełnie innego", "other"),
        (None, "self"), ("", "self"), ("nieznany", "self"), ("unknown", "self"),
    ])
    def test_arrival_mode(self, mode, expected):
        assert normalize_arrival_mode(mode) == expected

    @pytest.mark.parametrize("value, expected", [
        ("M", 1.0), ("mężczyzna", 1.0), ("male", 1.0), (True, 1.0), (1, 1.0),
        ("K", 0.0), ("kobieta", 0.0), ("F", 0.0), (False, 0.0), (0, 0.0),
    ])
    def test_sex(self, value, expected):
        assert encode_sex(value) == expected

    @pytest.mark.parametrize("value", [None, "?", 2, float("nan")])
    def test_sex_unknown_is_nan(self, value):
        assert math.isnan(encode_sex(value))


class TestAlignFeatures:

    def test_drops_extra_columns(self):
        df = build_feature_row(_vitals(), {}, {"triage_pain": 5.0})
        aligned = align_features(df.copy(), ["triage_vital_hr", "age"], {})
        assert list(aligned.columns) == ["triage_vital_hr", "age"]
        assert aligned.shape == (1, 2)

    def test_missing_feature_is_nan_not_median(self):
        df = pd.DataFrame([{"hr": 80}])
        aligned = align_features(df, ["hr", "sbp"], {"sbp": 120.0})
        assert math.isnan(aligned["sbp"].iloc[0])

    def test_missing_unknown_feature_is_nan(self):
        df = pd.DataFrame([{"hr": 80}])
        aligned = align_features(df, ["hr", "unknown_feature"], {})
        assert math.isnan(aligned["unknown_feature"].iloc[0])

    def test_missing_chief_complaint_is_zero(self):
        df = pd.DataFrame([{"hr": 80}])
        aligned = align_features(df, ["hr", "cc_chestpain", "cc_group_cardiac"])
        assert aligned["cc_chestpain"].iloc[0] == 0.0           # absent complaint
        assert math.isnan(aligned["cc_group_cardiac"].iloc[0])  # engineered, not a complaint

    def test_medians_only_as_explicit_fallback(self):
        df = pd.DataFrame([{"hr": 80}])
        aligned = align_features(df, ["hr", "sbp", "dbp"], {"sbp": 120.0}, use_medians=True)
        assert aligned["sbp"].iloc[0] == 120.0
        assert math.isnan(aligned["dbp"].iloc[0])

    def test_preserves_order(self):
        df = pd.DataFrame([{"b": 2, "a": 1, "c": 3}])
        aligned = align_features(df, ["a", "b", "c"], {})
        assert list(aligned.columns) == ["a", "b", "c"]

    def test_full_row_aligned_to_trained_features(self, triage_features):
        df = build_feature_row(_vitals(), {"cc_chestpain": 1}, {})
        aligned = align_features(df, triage_features, {})
        assert aligned.shape == (1, len(triage_features))
        # Only request data the frontend does not collect stay missing
        nan_cols = sorted(aligned.columns[aligned.isna().iloc[0]])
        assert nan_cols == ["is_male", "triage_vital_o2_device"]


class TestTrainingContract:
    """build_feature_row vs the feature lists training's prepare builds (no real data needed)."""

    def test_arrival_categories_match_training(self):
        assert ys.ARRIVAL_CATEGORIES == ["ambulance", "self", "other"]
        for alias, category in ARRIVAL_MODE_ALIASES.items():
            assert category in ys.ARRIVAL_CATEGORIES, alias

    def test_chief_complaints_match_prepare(self, synthetic_prepared):
        assert set(synthetic_prepared.groups["chief_complaints"]) == set(CHIEF_COMPLAINT_COLUMNS)

    def test_row_covers_every_triage_feature(self, synthetic_prepared):
        df = build_feature_row(_vitals(), {"cc_chestpain": 1}, {})
        triage = synthetic_prepared.feature_sets["triage"]
        assert [c for c in triage if c not in df.columns] == []

    def test_aligned_row_only_missing_unknown_request_data(self, synthetic_prepared):
        df = build_feature_row(_vitals(), {"cc_chestpain": 1}, {})
        aligned = align_features(df, synthetic_prepared.feature_sets["triage"], {})
        nan_cols = sorted(aligned.columns[aligned.isna().iloc[0]])
        assert nan_cols == ["is_male", "triage_vital_o2_device"]

    def test_meta_columns_never_in_row_features(self, synthetic_prepared):
        assert "vitals_complete" not in synthetic_prepared.feature_sets["triage"]
        assert synthetic_prepared.frame["vitals_complete"].isin([0, 1]).all()
