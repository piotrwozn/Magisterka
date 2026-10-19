"""
Testy dostępu do przygotowanych danych (`src.data.dataset`).

Korzysta z syntetycznych splitów przygotowanych w katalogu tymczasowym
(fixture `prepared_paths` w conftest.py) — prawdziwe dane nie są czytane.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data import dataset
from src.data.dataset import (
    early_stopping_split,
    ensure_prepared,
    feature_groups,
    feature_names,
    get_xy,
    is_prepared,
    load_split,
    load_splits,
    resolve_feature_set,
    sample_frame,
)
from src.data.prepare import META_COLS
from src.utils.config import DATA_VERSION, TARGET_COL


# ─────────────────────────────────────────
# Zestawy cech
# ─────────────────────────────────────────
class TestFeatureSets:
    @pytest.mark.parametrize("alias, expected", [
        ("triage", "triage"), ("full", "full"), ("triage_only", "triage"), ("top", "triage"),
    ])
    def test_resolve_aliases(self, alias, expected):
        assert resolve_feature_set(alias) == expected

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Nieznany zestaw cech"):
            resolve_feature_set("everything")

    def test_feature_names_aliases(self, prepared_paths):
        triage = feature_names("triage")
        assert feature_names("triage_only") == triage
        assert feature_names("top") == triage
        assert len(feature_names("full")) > len(triage)

    def test_feature_names_match_json(self, prepared_paths):
        payload = json.loads(prepared_paths.FEATURE_SETS_JSON.read_text())
        assert feature_names("full") == payload["feature_sets"]["full"]

    def test_feature_names_never_meta(self, prepared_paths):
        for fs in ("triage", "full"):
            assert not set(feature_names(fs)) & set(META_COLS)

    def test_feature_groups(self, prepared_paths):
        groups = feature_groups()
        assert "cc_fall_65" in groups["chief_complaints"]
        assert set(groups["engineered"]) <= set(feature_names("triage"))

    def test_missing_feature_sets_json_raises(self, isolated_paths):
        with pytest.raises(FileNotFoundError):
            feature_names("triage")


# ─────────────────────────────────────────
# Wczytywanie splitów
# ─────────────────────────────────────────
class TestLoadSplits:
    def test_is_prepared(self, prepared_paths):
        assert is_prepared()

    def test_not_prepared_in_empty_dir(self, isolated_paths):
        assert not is_prepared()

    def test_ensure_prepared_noop_when_ready(self, prepared_paths):
        before = prepared_paths.TRAIN_PARQUET.stat().st_mtime_ns
        ensure_prepared()
        assert prepared_paths.TRAIN_PARQUET.stat().st_mtime_ns == before

    def test_load_splits_all(self, prepared_paths):
        splits = load_splits()
        assert set(splits) == {"train", "val", "test"}
        full = feature_names("full")
        for df in splits.values():
            assert list(df.columns) == META_COLS + full
        ids = [set(df["row_id"]) for df in splits.values()]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_load_splits_with_feature_set_selects_columns(self, prepared_paths):
        splits = load_splits(names=("val",), feature_set="triage_only")
        assert list(splits) == ["val"]
        assert list(splits["val"].columns) == META_COLS + feature_names("triage")

    def test_load_split_unknown_name(self, prepared_paths):
        with pytest.raises(ValueError, match="Nieznany split"):
            load_split("holdout")

    def test_load_split_missing_file(self, isolated_paths):
        with pytest.raises(FileNotFoundError):
            load_split("train")

    def test_version_mismatch_raises(self, prepared_paths):
        manifest = json.loads(prepared_paths.DATA_MANIFEST_JSON.read_text())
        assert manifest["data_version"] == DATA_VERSION
        manifest["data_version"] = "stara-wersja-mts"
        prepared_paths.DATA_MANIFEST_JSON.write_text(json.dumps(manifest))

        assert not is_prepared()
        with pytest.raises(RuntimeError, match="stara-wersja-mts"):
            load_split("train")
        with pytest.raises(RuntimeError):
            load_splits(feature_set="triage")


# ─────────────────────────────────────────
# X / y
# ─────────────────────────────────────────
class TestGetXY:
    @pytest.mark.parametrize("fs", ["triage", "full"])
    def test_dtypes_and_shape(self, prepared_paths, fs):
        df = load_split("train")
        X, y = get_xy(df, fs)
        assert X.shape == (len(df), len(feature_names(fs)))
        assert list(X.columns) == feature_names(fs)
        assert (X.dtypes == np.float32).all()
        assert pd.api.types.is_integer_dtype(y)
        assert y.name == TARGET_COL
        assert set(y.unique()) == {0, 1, 2, 3, 4}

    def test_nan_kept(self, prepared_paths):
        X, _ = get_xy(load_split("train"), "triage")
        assert X["triage_vital_hr"].isna().any()       # brak imputacji
        assert X["cc_chestpain"].notna().all()         # skargi: brak = 0

    def test_target_matches_esi(self, prepared_paths):
        df = load_split("val")
        _, y = get_xy(df, "triage")
        assert (y == df["esi"] - 1).all()

    def test_missing_column_raises(self, prepared_paths):
        df = load_split("test").drop(columns=["triage_vital_hr"])
        with pytest.raises(KeyError, match="triage_vital_hr"):
            get_xy(df, "triage")


# ─────────────────────────────────────────
# Próbkowanie i early stopping
# ─────────────────────────────────────────
class TestSampling:
    def test_early_stopping_split_disjoint_and_stratified(self, prepared_paths):
        X, y = get_xy(load_split("train"), "triage")
        X_fit, y_fit, X_es, y_es = early_stopping_split(X, y)
        assert len(X_fit) + len(X_es) == len(X)
        assert not set(X_fit.index) & set(X_es.index)
        assert (X_fit.index == y_fit.index).all()
        assert (X_es.index == y_es.index).all()
        assert len(X_es) == pytest.approx(0.1 * len(X), abs=2)
        assert set(y_es.unique()) == {0, 1, 2, 3, 4}
        overall = y.value_counts(normalize=True).sort_index()
        es_dist = y_es.value_counts(normalize=True).sort_index()
        assert (overall - es_dist).abs().max() < 0.03

    def test_early_stopping_split_deterministic(self, synthetic_X_y):
        X, y = synthetic_X_y
        a = early_stopping_split(X, y)
        b = early_stopping_split(X, y)
        assert list(a[2].index) == list(b[2].index)

    def test_sample_frame(self, prepared_paths):
        df = load_split("train")
        sample = sample_frame(df, 200)
        assert len(sample) == 200
        assert set(sample[TARGET_COL].unique()) == {0, 1, 2, 3, 4}
        assert sample.index.is_monotonic_increasing
        assert sample_frame(df, None) is df
        assert sample_frame(df, len(df) + 10) is df

    def test_feature_sets_cache_cleared_between_tests(self, isolated_paths):
        """Bezpiecznik dla innych testów: cache list cech nie przecieka między katalogami."""
        assert dataset._feature_sets_payload.cache_info().currsize == 0
