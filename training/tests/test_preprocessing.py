"""Testy preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    FeatureGroups,
    build_feature_groups,
    impute_missing,
    preprocess_dataframe,
    split_features,
)
from src.data.splits import chronological_split, stratified_split


class TestBuildFeatureGroups:
    def test_basic_detection(self, synthetic_dataset):
        groups = build_feature_groups(synthetic_dataset)
        assert isinstance(groups, FeatureGroups)
        assert "triage_sbp" in groups.triage_vitals
        assert "cc_chestpain" in groups.chief_complaints
        assert "pmh_diabetes" in groups.past_medical
        assert "med_betablocker" in groups.medications

    def test_excludes_target_columns(self, synthetic_dataset):
        df = synthetic_dataset.copy()
        df["esi"] = 3
        df["mts_numeric"] = 2
        groups = build_feature_groups(df)

        all_features = (
            groups.triage_vitals + groups.demographics + groups.arrival
            + groups.chief_complaints + groups.past_medical + groups.medications
            + groups.historical_vitals + groups.historical_labs + groups.ed_usage
            + groups.imaging_history + groups.other
        )
        assert "esi" not in all_features
        assert "mts_numeric" not in all_features


class TestImputation:
    def test_imputes_vital_signs(self, synthetic_dataset):
        df = synthetic_dataset.copy()
        df.loc[0:5, "triage_sbp"] = np.nan
        groups = build_feature_groups(df)

        df_imp = impute_missing(df, groups)
        assert df_imp["triage_sbp"].isna().sum() == 0

    def test_imputes_binary_with_zero(self, synthetic_dataset):
        df = synthetic_dataset.copy()
        df.loc[0:5, "cc_chestpain"] = np.nan
        groups = build_feature_groups(df)

        df_imp = impute_missing(df, groups)
        assert df_imp["cc_chestpain"].isna().sum() == 0
        assert (df_imp.loc[0:5, "cc_chestpain"] == 0).all()


class TestPreprocessDataframe:
    def test_returns_no_nans(self, synthetic_dataset):
        df = synthetic_dataset.copy()
        df.loc[0:5, "triage_sbp"] = np.nan
        df.loc[10:15, "cc_chestpain"] = np.nan

        df_processed, groups = preprocess_dataframe(df, encode_method="label")
        # Tylko kolumny używane przez model nie powinny mieć NaN
        for col in groups.triage_vitals + groups.chief_complaints:
            if col in df_processed.columns:
                assert df_processed[col].isna().sum() == 0


class TestSplitFeatures:
    def test_triage_only(self, synthetic_dataset):
        df = synthetic_dataset.copy()
        df["mts_numeric"] = (df["esi"] - 1)
        groups = build_feature_groups(df)

        X, y, names = split_features(df, groups, feature_set="triage_only")
        assert len(names) > 0
        assert len(X) == len(y)
        assert y.between(0, 4).all()

    def test_full_includes_history(self, synthetic_dataset):
        df = synthetic_dataset.copy()
        df["mts_numeric"] = (df["esi"] - 1)
        groups = build_feature_groups(df)

        X_triage, _, names_triage = split_features(df, groups, feature_set="triage_only")
        X_full, _, names_full = split_features(df, groups, feature_set="full")
        assert len(names_full) >= len(names_triage)


class TestSplits:
    def test_chronological_keeps_order(self, synthetic_dataset):
        df = synthetic_dataset.copy()
        df["mts_numeric"] = (df["esi"] - 1)

        splits = chronological_split(df, target_col="mts_numeric")
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(df)

    def test_stratified_preserves_distribution(self, synthetic_dataset):
        df = synthetic_dataset.copy()
        df["mts_numeric"] = (df["esi"] - 1)

        splits = stratified_split(df, target_col="mts_numeric")
        original_dist = df["mts_numeric"].value_counts(normalize=True).sort_index()

        for name, split_df in splits.items():
            split_dist = split_df["mts_numeric"].value_counts(normalize=True).sort_index()
            # Powinny być zbliżone (stratified)
            assert abs(split_dist - original_dist).max() < 0.05
