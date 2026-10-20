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
from src.data.prepare import META_COLS, VITALS_COMPLETE_COL
from src.utils.config import DATA_VERSION, TARGET_COL


def _set_manifest(paths, **changes) -> None:
    manifest = json.loads(paths.DATA_MANIFEST_JSON.read_text(encoding="utf-8"))
    manifest.update(changes)
    paths.DATA_MANIFEST_JSON.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def prepare_calls(monkeypatch) -> list[str]:
    """Podmienia prepare_dataset (zapamiętuje strategię zamiast przygotowywać dane)."""
    from src.data import prepare

    calls: list[str] = []
    monkeypatch.setattr(prepare, "prepare_dataset", lambda strategy="random", raw=None: calls.append(strategy))
    monkeypatch.delenv(dataset.NO_AUTO_PREPARE_ENV, raising=False)
    return calls


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
        payload = json.loads(prepared_paths.FEATURE_SETS_JSON.read_text(encoding="utf-8"))
        assert feature_names("full") == payload["feature_sets"]["full"]

    def test_feature_names_never_meta(self, prepared_paths):
        for fs in ("triage", "full"):
            assert not set(feature_names(fs)) & set(META_COLS)

    def test_meta_columns_contract(self):
        from src.data import VITALS_COMPLETE_COL as exported

        assert exported == VITALS_COMPLETE_COL == "vitals_complete"
        assert META_COLS == ["row_id", "site", "esi", "target", "admitted", "vitals_complete"]

    def test_feature_sets_json_carries_data_version(self, prepared_paths):
        payload = json.loads(prepared_paths.FEATURE_SETS_JSON.read_text(encoding="utf-8"))
        assert payload["data_version"] == DATA_VERSION

    def test_triage_has_no_removed_columns(self, prepared_paths):
        triage = set(feature_names("triage"))
        full = set(feature_names("full"))
        assert {"arrival_ambulance", "arrival_self", "arrival_other", "cc_trauma_activation"} <= triage
        for removed in ("arrival_unknown", "arrival_police", "esi_danger_hr", "esi_danger_rr",
                        "cc_fulltrauma", "cc_modifiedtrauma", "vitals_complete"):
            assert removed not in full, removed
        assert not [c for c in full if c.startswith("meds_")]

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

    def test_ensure_prepared_noop_when_ready_even_without_auto_prepare(self, prepared_paths, monkeypatch):
        monkeypatch.setenv(dataset.NO_AUTO_PREPARE_ENV, "1")
        ensure_prepared()

    def test_no_auto_prepare_raises_when_missing(self, isolated_paths, monkeypatch, prepare_calls):
        monkeypatch.setenv(dataset.NO_AUTO_PREPARE_ENV, "1")
        with pytest.raises(RuntimeError, match="SOR_AI_NO_AUTO_PREPARE"):
            ensure_prepared()
        assert prepare_calls == []

    def test_no_auto_prepare_raises_when_stale(self, prepared_paths, monkeypatch, prepare_calls):
        _set_manifest(prepared_paths, data_version="esi-v1")
        monkeypatch.setenv(dataset.NO_AUTO_PREPARE_ENV, "1")
        with pytest.raises(RuntimeError, match="esi-v1"):
            ensure_prepared()
        assert prepare_calls == []

    def test_auto_prepare_keeps_manifest_split_strategy(self, prepared_paths, prepare_calls):
        _set_manifest(prepared_paths, data_version="esi-v1", split_strategy="site")
        ensure_prepared()
        assert prepare_calls == ["site"]

    def test_auto_prepare_explicit_strategy_wins(self, prepared_paths, prepare_calls):
        _set_manifest(prepared_paths, data_version="esi-v1", split_strategy="site")
        ensure_prepared("random")
        assert prepare_calls == ["random"]

    def test_auto_prepare_defaults_to_random(self, isolated_paths, prepare_calls):
        ensure_prepared()
        assert prepare_calls == ["random"]

    def test_corrupt_manifest_is_not_prepared(self, prepared_paths):
        prepared_paths.DATA_MANIFEST_JSON.write_text("{nie-json", encoding="utf-8")
        assert not is_prepared()

    def test_load_splits_all(self, prepared_paths):
        splits = load_splits()
        assert set(splits) == {"train", "val", "test"}
        full = feature_names("full")
        for df in splits.values():
            assert list(df.columns) == META_COLS + full
            assert df[VITALS_COMPLETE_COL].isin([0, 1]).all()
            assert df[VITALS_COMPLETE_COL].sum() > 0
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
        manifest = json.loads(prepared_paths.DATA_MANIFEST_JSON.read_text(encoding="utf-8"))
        assert manifest["data_version"] == DATA_VERSION
        _set_manifest(prepared_paths, data_version="stara-wersja")

        assert not is_prepared()
        with pytest.raises(RuntimeError, match="stara-wersja"):
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


# ─────────────────────────────────────────
# Odcisk danych
# ─────────────────────────────────────────
class TestDataFingerprint:
    BASE = {"data_version": DATA_VERSION, "created": "2026-10-19T19:31:16", "source": "/a/raw.parquet",
            "split_strategy": "random", "n_rows": {"train": 800, "val": 100, "test": 100},
            "audit": {"written_at_counts": {"Admit": 3}}}

    def test_volatile_keys_do_not_change_fingerprint(self):
        other = {**self.BASE, "created": "2026-10-20T08:00:00", "source": r"C:\inna\ścieżka\raw.parquet",
                 "prepared_at": "teraz", "created_by_host": "pc", "timestamp": 1.0}
        fp = dataset.data_fingerprint(self.BASE)
        assert fp == dataset.data_fingerprint(other)
        assert len(fp) == 10 and fp != dataset.NO_DATA_FINGERPRINT

    def test_content_changes_fingerprint(self):
        fp = dataset.data_fingerprint(self.BASE)
        assert dataset.data_fingerprint({**self.BASE, "n_rows": {"train": 801, "val": 100, "test": 100}}) != fp
        assert dataset.data_fingerprint({**self.BASE, "split_strategy": "site"}) != fp
        assert dataset.data_fingerprint({**self.BASE, "data_version": "esi-v0"}) != fp
        # Zagnieżdżone klucze (audyt) są danymi — nawet gdy nazwa przypomina znacznik czasu
        assert dataset.data_fingerprint({**self.BASE, "audit": {"written_at_counts": {"Admit": 4}}}) != fp

    def test_path_and_missing_file(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(self.BASE, ensure_ascii=False), encoding="utf-8")
        assert dataset.data_fingerprint(path) == dataset.data_fingerprint(str(path)) == \
            dataset.data_fingerprint(self.BASE)
        assert dataset.data_fingerprint(tmp_path / "brak.json") == dataset.NO_DATA_FINGERPRINT
        (tmp_path / "zly.json").write_text("{nie json", encoding="utf-8")
        assert dataset.data_fingerprint(tmp_path / "zly.json") == dataset.NO_DATA_FINGERPRINT

    def test_reprepare_identical_data_keeps_fingerprint(self, isolated_paths, make_raw):
        """Ponowne przygotowanie tych samych danych (inny czas, inna ścieżka) → ten sam odcisk."""
        from src.data.prepare import prepare_dataset
        from src.models import tuning

        raw = make_raw(n=600, seed=21)
        prepare_dataset(strategy="random", raw=raw)
        first = dataset.data_fingerprint()
        assert first != dataset.NO_DATA_FINGERPRINT
        assert tuning.data_fingerprint is dataset.data_fingerprint          # jeden algorytm w całym kodzie
        _set_manifest(isolated_paths, created="2000-01-01T00:00:00", source="/gdzie/indziej.parquet")
        assert dataset.data_fingerprint() == first
        prepare_dataset(strategy="random", raw=raw)
        assert dataset.data_fingerprint() == first
        prepare_dataset(strategy="site", raw=raw)                            # inny podział = inne dane
        assert dataset.data_fingerprint() != first
