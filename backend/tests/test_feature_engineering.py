"""Tests for the runtime feature engineering wrapper."""
from __future__ import annotations

import pytest

from app.models.schemas import Vitals
from app.pipeline.feature_engineering import align_features, build_feature_row


class TestBuildFeatureRow:

    def test_returns_dataframe(self):
        vitals = Vitals(age=40, temp=36.7, hr=80, sbp=120, dbp=80, rr=16, o2=99)
        df = build_feature_row(vitals, {}, {})
        assert df.shape[0] == 1
        assert df.shape[1] > 50

    def test_celsius_to_fahrenheit_conversion(self):
        vitals = Vitals(age=40, temp=37.0, hr=80, sbp=120, dbp=80, rr=16, o2=99)
        df = build_feature_row(vitals, {}, {})
        # 37.0°C = 98.6°F
        assert abs(df["triage_vital_temp"].iloc[0] - 98.6) < 0.01

    def test_includes_all_vital_columns(self):
        vitals = Vitals(age=40, temp=36.7, hr=80, sbp=120, dbp=80, rr=16, o2=99)
        df = build_feature_row(vitals, {}, {})
        for col in (
            "triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp",
            "triage_vital_rr", "triage_vital_o2", "triage_vital_temp", "age",
        ):
            assert col in df.columns
            assert df[col].iloc[0] is not None

    def test_cc_features_propagated(self):
        vitals = Vitals(age=40, temp=36.7, hr=80, sbp=120, dbp=80, rr=16, o2=99)
        df = build_feature_row(vitals, {"cc_chestpain": 1}, {})
        assert df["cc_chestpain"].iloc[0] == 1

    def test_engineered_features_computed(self):
        vitals = Vitals(age=40, temp=36.7, hr=120, sbp=90, dbp=60, rr=16, o2=99)
        df = build_feature_row(vitals, {}, {})
        # shock_index = HR/SBP = 120/90 ≈ 1.33
        assert "shock_index" in df.columns
        assert abs(df["shock_index"].iloc[0] - 120 / 90) < 0.01

    def test_no_duplicate_columns(self):
        vitals = Vitals(age=40, temp=36.7, hr=80, sbp=120, dbp=80, rr=16, o2=99)
        df = build_feature_row(
            vitals,
            {"cc_chestpain": 1, "cc_cardiac_group": 1},  # group is engineered — must be skipped
            {},
            all_cc_columns=["cc_chestpain", "cc_cardiac_group", "cc_other"],
        )
        assert df.columns.nunique() == df.shape[1]

    def test_all_cc_columns_prepopulated(self):
        vitals = Vitals(age=40, temp=36.7, hr=80, sbp=120, dbp=80, rr=16, o2=99)
        cc_list = ["cc_chestpain", "cc_dyspnea", "cc_other"]
        df = build_feature_row(vitals, {"cc_chestpain": 1}, {}, all_cc_columns=cc_list)
        # cc_chestpain=1, others=0
        assert df["cc_chestpain"].iloc[0] == 1
        assert df["cc_dyspnea"].iloc[0] == 0
        assert df["cc_other"].iloc[0] == 0


class TestAlignFeatures:

    def test_drops_extra_columns(self):
        vitals = Vitals(age=40, temp=36.7, hr=80, sbp=120, dbp=80, rr=16, o2=99)
        df = build_feature_row(vitals, {}, {})
        aligned = align_features(df.copy(), ["triage_vital_hr", "age"], {})
        assert list(aligned.columns) == ["triage_vital_hr", "age"]
        assert aligned.shape == (1, 2)

    def test_fills_missing_with_median(self):
        import pandas as pd
        df = pd.DataFrame([{"hr": 80}])
        aligned = align_features(df, ["hr", "sbp"], {"sbp": 120.0})
        assert aligned["sbp"].iloc[0] == 120.0

    def test_fills_unknown_with_zero(self):
        import pandas as pd
        df = pd.DataFrame([{"hr": 80}])
        aligned = align_features(df, ["hr", "unknown_feature"], {})
        assert aligned["unknown_feature"].iloc[0] == 0.0

    def test_preserves_order(self):
        import pandas as pd
        df = pd.DataFrame([{"b": 2, "a": 1, "c": 3}])
        aligned = align_features(df, ["a", "b", "c"], {})
        assert list(aligned.columns) == ["a", "b", "c"]
