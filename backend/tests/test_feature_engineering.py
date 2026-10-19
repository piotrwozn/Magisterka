"""Tests for the runtime feature engineering wrapper (canonical ESI triage row)."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.models.schemas import Vitals
from app.pipeline.feature_engineering import (
    CHIEF_COMPLAINT_COLUMNS,
    DEFAULT_ARRIVAL_MODE,
    align_features,
    build_feature_row,
    encode_sex,
    normalize_arrival_mode,
)

FEATURE_SETS_JSON = (
    Path(__file__).resolve().parents[2] / "training" / "data" / "processed" / "esi" / "feature_sets.json"
)

CANONICAL_BASE = [
    "age", "is_male",
    "arrival_ambulance", "arrival_self", "arrival_police", "arrival_other", "arrival_unknown",
    "arrival_month", "arrival_weekday", "arrival_hour_bin",
    "triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp", "triage_vital_rr",
    "triage_vital_o2", "triage_vital_o2_device", "triage_vital_temp",
]


def _vitals(**overrides) -> Vitals:
    values = dict(age=40, temp=36.7, hr=80, sbp=120, dbp=80, rr=16, o2=99)
    values.update(overrides)
    return Vitals(**values)


@pytest.fixture(scope="module")
def triage_features() -> list[str]:
    if not FEATURE_SETS_JSON.exists():
        pytest.skip("No training/data/processed/esi/feature_sets.json — run 02_preprocess.py")
    return json.loads(FEATURE_SETS_JSON.read_text())["feature_sets"]["triage"]


class TestBuildFeatureRow:

    def test_returns_single_row(self):
        df = build_feature_row(_vitals(), {}, {})
        assert df.shape[0] == 1
        assert df.shape[1] >= 276
        assert df.columns.is_unique

    def test_temperature_stays_celsius(self):
        df = build_feature_row(_vitals(temp=37.0), {}, {})
        # No °C → °F conversion: training data are converted to °C in prepare
        assert df["triage_vital_temp"].iloc[0] == pytest.approx(37.0)

    def test_fever_threshold_in_celsius(self):
        df = build_feature_row(_vitals(temp=38.5), {}, {})
        assert df["fever"].iloc[0] == 1.0
        assert df["hypothermia"].iloc[0] == 0.0

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
        assert len(CHIEF_COMPLAINT_COLUMNS) == 200
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

    def test_engineered_features_computed(self):
        df = build_feature_row(_vitals(hr=120, sbp=90, dbp=60), {}, {})
        assert df["shock_index"].iloc[0] == pytest.approx(120 / 90)
        assert df["esi_danger_hr"].iloc[0] == 1.0

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
        # Missing arrival mode → DEFAULT_ARRIVAL_MODE, never arrival_unknown
        # (Yale "unknown" ≈ fast-track population, 90% ESI 4–5 → undertriage)
        assert DEFAULT_ARRIVAL_MODE == "self"
        assert row["arrival_self"] == 1.0
        assert row[["arrival_ambulance", "arrival_police", "arrival_other", "arrival_unknown"]].sum() == 0
        assert math.isnan(row["is_male"])
        assert math.isnan(row["triage_vital_o2_device"])

    def test_explicit_request_fields(self):
        df = build_feature_row(_vitals(), {}, {}, is_male="K", arrival_mode="ZRM", o2_device=1)
        row = df.iloc[0]
        assert row["is_male"] == 0.0
        assert row["arrival_ambulance"] == 1.0
        assert row["arrival_unknown"] == 0.0
        assert row["triage_vital_o2_device"] == 1.0

    def test_request_fields_read_from_request_object(self):
        # Forward-compatible: optional fields on the request object are used if present
        req = SimpleNamespace(age=70, temp=36.6, hr=80, sbp=120, dbp=80, rr=16, o2=97,
                              sex="M", arrival_mode="policja", o2_device=0)
        row = build_feature_row(req, {}, {}).iloc[0]
        assert row["is_male"] == 1.0
        assert row["arrival_police"] == 1.0
        assert row["triage_vital_o2_device"] == 0.0


class TestRequestEncoding:

    @pytest.mark.parametrize("mode, expected", [
        ("ZRM", "ambulance"), ("karetka", "ambulance"), ("Karetka ZRM", "ambulance"),
        ("LPR", "ambulance"), ("ambulance", "ambulance"),
        ("własny", "self"), ("transport własny", "self"), ("Walk-in", "self"), ("Car", "self"),
        ("policja", "police"), ("Police", "police"),
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
