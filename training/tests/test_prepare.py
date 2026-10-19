"""
Testy przygotowania danych (`src.data.prepare`): surowy Yale → schemat kanoniczny ESI.

Sprawdza decyzje opisane w DATA_AUDIT.md: etykieta = natywne ESI, °F → °C,
wartości niewiarygodne → NaN, wykluczenie zmiennych z USA i przecieków,
jawne kodowania kategorii, normalizacja nazw, podziały i zapis artefaktów.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from src.data import yale_schema as ys
from src.data.prepare import (
    ADMITTED_COL,
    META_COLS,
    ROW_ID_COL,
    SITE_COL,
    build_canonical_frame,
    make_splits,
    prepare_dataset,
)
from src.utils.config import DATA_VERSION, ESI_COL, HOLDOUT_SITE, TARGET_COL

SAFE_NAME = re.compile(r"^[a-z0-9_]+$")


@pytest.fixture(scope="module")
def prepared(make_raw):
    """Kanoniczny zbiór z domyślnego syntetycznego Yale (liczony raz na moduł)."""
    return build_canonical_frame(make_raw())


def _all_features(prepared) -> set[str]:
    return set(prepared.feature_sets["triage"]) | set(prepared.feature_sets["full"])


# ─────────────────────────────────────────
# Etykieta i wiersze
# ─────────────────────────────────────────
class TestLabelAndRows:
    def test_target_is_esi_minus_one(self, prepared):
        df = prepared.frame
        assert set(df[ESI_COL].unique()) == {1, 2, 3, 4, 5}
        assert (df[TARGET_COL] == df[ESI_COL] - 1).all()
        assert df[TARGET_COL].between(0, 4).all()

    def test_esi_matches_raw_value(self, raw_yale):
        """Brak mapowania: ESI w zbiorze kanonicznym = ESI z surowych danych (wiersze 0–19)."""
        out = build_canonical_frame(raw_yale).frame
        expected = raw_yale["esi"].iloc[:20].astype(str).astype(int).to_numpy()
        np.testing.assert_array_equal(out[ESI_COL].iloc[:20].to_numpy(), expected)

    def test_missing_esi_dropped(self, raw_yale, prepared):
        n_missing = int(raw_yale["esi"].isna().sum())
        assert n_missing > 0
        assert prepared.audit["dropped_missing_esi"] == n_missing
        assert prepared.frame[ESI_COL].notna().all()

    def test_duplicates_dropped(self, raw_yale, prepared):
        n_dup = int(raw_yale.duplicated().sum())
        assert n_dup > 0
        assert prepared.audit["dropped_duplicates"] == n_dup
        n_expected = len(raw_yale) - prepared.audit["dropped_missing_esi"] - n_dup
        assert len(prepared.frame) == prepared.audit["n_rows"] == n_expected

    def test_row_ids_unique_and_contiguous(self, prepared):
        ids = prepared.frame[ROW_ID_COL].to_numpy()
        np.testing.assert_array_equal(ids, np.arange(len(ids)))

    def test_admitted_from_disposition(self, raw_yale):
        out = build_canonical_frame(raw_yale).frame
        expected = (raw_yale["disposition"].astype(str).iloc[:20] == "Admit").astype(int).to_numpy()
        np.testing.assert_array_equal(out[ADMITTED_COL].iloc[:20].to_numpy(), expected)

    def test_site_kept_as_meta(self, prepared):
        assert set(prepared.frame[SITE_COL].unique()) == {"A", "B", "C"}


# ─────────────────────────────────────────
# Parametry życiowe
# ─────────────────────────────────────────
class TestVitals:
    def test_fahrenheit_to_celsius(self, raw_yale):
        raw = raw_yale
        raw.loc[0, "triage_vital_temp"] = 98.6
        raw.loc[1, "triage_vital_temp"] = 100.4
        raw.loc[0, "temp_max"] = 98.6          # historia vitals też w °F
        out = build_canonical_frame(raw).frame
        assert out.loc[0, "triage_vital_temp"] == pytest.approx(37.0, abs=0.01)
        assert out.loc[1, "triage_vital_temp"] == pytest.approx(38.0, abs=0.01)
        assert out.loc[0, "temp_max"] == pytest.approx(37.0, abs=0.01)

    def test_all_temperatures_in_celsius_range(self, prepared):
        for col in ["triage_vital_temp", "temp_max"]:
            temps = prepared.frame[col].dropna()
            assert temps.between(30, 44).all(), col
            assert 35.5 < temps.median() < 38.5, col

    def test_implausible_values_become_nan(self, raw_yale):
        raw = raw_yale
        raw.loc[0, "triage_vital_hr"] = 500
        raw.loc[1, "triage_vital_o2"] = 30
        raw.loc[2, "triage_vital_temp"] = 120.0      # °F → 48.9 °C
        raw.loc[3, "triage_vital_rr"] = 0
        raw.loc[4, "pulse_last"] = 0
        raw.loc[5, "triage_vital_hr"] = 72
        prepared = build_canonical_frame(raw)
        out = prepared.frame
        assert np.isnan(out.loc[0, "triage_vital_hr"])
        assert np.isnan(out.loc[1, "triage_vital_o2"])
        assert np.isnan(out.loc[2, "triage_vital_temp"])
        assert np.isnan(out.loc[3, "triage_vital_rr"])
        assert np.isnan(out.loc[4, "pulse_last"])
        assert out.loc[5, "triage_vital_hr"] == 72
        nan_counts = prepared.audit["implausible_to_nan"]
        assert nan_counts["triage_vital_hr"] >= 1
        assert nan_counts["pulse_last"] >= 1

    def test_dbp_ge_sbp_sets_dbp_nan(self, raw_yale):
        raw = raw_yale
        raw.loc[0, ["triage_vital_sbp", "triage_vital_dbp"]] = [100, 110]
        raw.loc[1, ["triage_vital_sbp", "triage_vital_dbp"]] = [120, 120]
        raw.loc[2, ["triage_vital_sbp", "triage_vital_dbp"]] = [120, 80]
        prepared = build_canonical_frame(raw)
        out = prepared.frame
        assert np.isnan(out.loc[0, "triage_vital_dbp"])
        assert out.loc[0, "triage_vital_sbp"] == 100     # SBP zostaje
        assert np.isnan(out.loc[1, "triage_vital_dbp"])
        assert out.loc[2, "triage_vital_dbp"] == 80
        valid = out[["triage_vital_sbp", "triage_vital_dbp"]].dropna()
        assert (valid["triage_vital_dbp"] < valid["triage_vital_sbp"]).all()
        assert prepared.audit["dbp_ge_sbp_to_nan"] >= 2

    def test_missing_vitals_stay_nan(self, prepared):
        """Brak globalnej imputacji — braki zostają NaN (i są oznaczone flagami)."""
        df = prepared.frame
        assert df["triage_vital_hr"].isna().any()
        assert (df["hr_missing"] == df["triage_vital_hr"].isna().astype(float)).all()
        assert df["all_vitals_missing"].sum() > 0


# ─────────────────────────────────────────
# Wykluczone kolumny i przecieki
# ─────────────────────────────────────────
class TestExclusions:
    @pytest.mark.parametrize("col", ys.US_SPECIFIC_COLS)
    def test_us_specific_columns_absent(self, prepared, col):
        features = _all_features(prepared)
        assert col not in prepared.frame.columns
        assert not any(f == col or f.startswith(f"{col}_") for f in features)
        assert col in prepared.audit["excluded_us_specific"]

    @pytest.mark.parametrize("col", META_COLS + ["dep_name", "disposition", "gender", "esi"])
    def test_meta_and_leaky_columns_not_features(self, prepared, col):
        for name, cols in prepared.feature_sets.items():
            assert col not in cols, f"'{col}' w zestawie '{name}'"

    def test_meta_columns_present_in_frame(self, prepared):
        for col in META_COLS:
            assert col in prepared.frame.columns

    def test_race_based_egfr_excluded(self, prepared):
        features = _all_features(prepared)
        assert not [f for f in features if f.startswith("egfr")]
        assert not [c for c in prepared.frame.columns if c.startswith("egfr")]
        assert set(prepared.audit["excluded_race_based_labs"]) == {
            "egfr_last", "egfr(aframer)_last", "egfr(nonafricanamerican)_median",
        }


# ─────────────────────────────────────────
# Kodowania kategorii
# ─────────────────────────────────────────
class TestEncodings:
    @pytest.mark.parametrize("raw_value, expected", [
        ("ambulance", "ambulance"),
        ("Car", "self"),
        ("Walk-in", "self"),
        ("Public Transportation", "self"),
        ("Police", "police"),
        ("Wheelchair", "other"),
        (None, "unknown"),
    ])
    def test_arrival_one_hot(self, raw_yale, raw_value, expected):
        raw = raw_yale
        raw.loc[0, "arrivalmode"] = raw_value
        out = build_canonical_frame(raw).frame
        for cat in ys.ARRIVAL_CATEGORIES:
            assert out.loc[0, f"arrival_{cat}"] == float(cat == expected), cat

    def test_arrival_one_hot_sums_to_one(self, prepared):
        cols = [f"arrival_{c}" for c in ys.ARRIVAL_CATEGORIES]
        assert (prepared.frame[cols].sum(axis=1) == 1).all()
        assert prepared.frame["arrival_unknown"].sum() > 0

    def test_month_weekday_hour_bin_explicit(self, raw_yale):
        raw = raw_yale
        raw.loc[0, ["arrivalmonth", "arrivalday", "arrivalhour_bin"]] = ["January", "Monday", "23-02"]
        raw.loc[1, ["arrivalmonth", "arrivalday", "arrivalhour_bin"]] = ["December", "Sunday", "19-22"]
        raw.loc[2, ["arrivalmonth", "arrivalday", "arrivalhour_bin"]] = ["May", "Saturday", "07-10"]
        out = build_canonical_frame(raw).frame
        cols = ["arrival_month", "arrival_weekday", "arrival_hour_bin"]
        assert out.loc[0, cols].tolist() == [1, 1, 0]
        assert out.loc[1, cols].tolist() == [12, 7, 5]
        assert out.loc[2, cols].tolist() == [5, 6, 2]

    def test_time_encodings_match_maps_everywhere(self, raw_yale):
        out = build_canonical_frame(raw_yale).frame
        head = raw_yale.iloc[:20]
        for col, raw_col, mapping in [
            ("arrival_month", "arrivalmonth", ys.MONTH_MAP),
            ("arrival_weekday", "arrivalday", ys.WEEKDAY_MAP),
            ("arrival_hour_bin", "arrivalhour_bin", ys.HOUR_BIN_MAP),
        ]:
            expected = head[raw_col].astype(str).map(mapping).to_numpy(dtype=float)
            np.testing.assert_array_equal(out[col].iloc[:20].to_numpy(dtype=float), expected)

    def test_gender_encoding(self, raw_yale):
        out = build_canonical_frame(raw_yale).frame
        expected = (raw_yale["gender"].astype(str).iloc[:20] == "Male").astype(float).to_numpy()
        np.testing.assert_array_equal(out["is_male"].iloc[:20].to_numpy(), expected)

    def test_chief_complaint_nan_becomes_zero(self, raw_yale):
        raw = raw_yale
        raw.loc[0, "cc_chestpain"] = np.nan
        out = build_canonical_frame(raw).frame
        assert out.loc[0, "cc_chestpain"] == 0.0
        assert out["cc_chestpain"].notna().all()

    def test_previous_dispo_one_hot(self, prepared):
        cols = [c for c in prepared.feature_sets["full"] if c.startswith("prev_dispo_")]
        assert "prev_dispo_no_previous_dispo" in cols
        assert (prepared.frame[cols].sum(axis=1) == 1).all()


# ─────────────────────────────────────────
# Nazwy i zestawy cech
# ─────────────────────────────────────────
class TestFeatureSets:
    def test_names_sanitized(self, prepared):
        bad = [c for c in prepared.frame.columns if not SAFE_NAME.match(c)]
        assert not bad
        for col in prepared.feature_sets["full"]:
            assert not set(col) & set(",()[]<>:/>-. "), col

    def test_specific_sanitized_names(self, prepared):
        triage = set(prepared.feature_sets["triage"])
        full = set(prepared.feature_sets["full"])
        assert {"cc_fall_65", "cc_bleeding_bruising", "cc_suture_stapleremoval"} <= triage
        assert {"baseexcess_venous_poc_last", "meds_anti_obesitydrugs", "2ndarymalig"} <= full
        renamed = prepared.audit["renamed_columns"]
        assert renamed["cc_fall>65"] == "cc_fall_65"
        assert renamed["baseexcess,venous(poc)_last"] == "baseexcess_venous_poc_last"

    def test_sanitize_name_examples(self):
        assert ys.sanitize_name("baseexcess,venous(poc)_last") == "baseexcess_venous_poc_last"
        assert ys.sanitize_name("cc_fall>65") == "cc_fall_65"
        assert ys.sanitize_name("cc_bleeding/bruising") == "cc_bleeding_bruising"

    def test_sanitize_columns_unique(self):
        mapping = ys.sanitize_columns(["a,b", "a(b)", "a_b"])
        assert len(set(mapping.values())) == 3

    def test_triage_subset_of_full_without_history(self, prepared):
        triage = prepared.feature_sets["triage"]
        full = prepared.feature_sets["full"]
        assert full[: len(triage)] == triage
        history = set(prepared.groups["past_medical"] + prepared.groups["medications"]
                      + prepared.groups["ed_usage"] + prepared.groups["imaging_history"]
                      + prepared.groups["history_vitals"] + prepared.groups["history_labs"])
        assert history
        assert not history & set(triage)
        assert history <= set(full)

    def test_groups_detected(self, prepared):
        g = prepared.groups
        assert set(g["past_medical"]) == {"2ndarymalig", "acutemi", "asthma", "chfnonhp"}
        assert set(g["medications"]) == {"meds_analgesics", "meds_anti_obesitydrugs", "meds_antibiotics"}
        assert {"pulse_last", "temp_max", "spo2_min", "sbp_median", "o2_device_last"} == set(g["history_vitals"])
        assert set(g["imaging_history"]) == set(ys.IMAGING_COLS)
        assert {"n_edvisits", "n_admissions", "n_surgeries"} <= set(g["ed_usage"])

    def test_no_duplicate_features(self, prepared):
        for cols in prepared.feature_sets.values():
            assert len(cols) == len(set(cols))
        assert not prepared.frame.columns.duplicated().any()

    def test_engineered_features_in_triage(self, prepared):
        from src.features.engineering import ENGINEERED_FEATURES

        assert set(ENGINEERED_FEATURES) <= set(prepared.feature_sets["triage"])

    def test_no_mts_features(self, prepared):
        features = _all_features(prepared)
        assert not [f for f in features if "mts" in f]


# ─────────────────────────────────────────
# Podziały
# ─────────────────────────────────────────
class TestSplits:
    def test_random_split_disjoint_and_complete(self, prepared):
        splits = make_splits(prepared.frame, strategy="random")
        ids = {name: set(s[ROW_ID_COL]) for name, s in splits.items()}
        assert not ids["train"] & ids["val"]
        assert not ids["train"] & ids["test"]
        assert not ids["val"] & ids["test"]
        assert set().union(*ids.values()) == set(prepared.frame[ROW_ID_COL])

    def test_random_split_sizes_and_stratification(self, prepared):
        df = prepared.frame
        splits = make_splits(df, strategy="random")
        assert len(splits["test"]) == pytest.approx(0.10 * len(df), abs=2)
        assert len(splits["val"]) == pytest.approx(0.10 * len(df), abs=2)
        overall = df[TARGET_COL].value_counts(normalize=True).sort_index()
        for name, s in splits.items():
            assert set(s[TARGET_COL].unique()) == {0, 1, 2, 3, 4}, name
            dist = s[TARGET_COL].value_counts(normalize=True).sort_index()
            assert (dist - overall).abs().max() < 0.03, name
            assert set(s[SITE_COL].unique()) == {"A", "B", "C"}, name

    def test_site_split_holds_out_site(self, prepared):
        splits = make_splits(prepared.frame, strategy="site")
        assert set(splits["test"][SITE_COL].unique()) == {HOLDOUT_SITE}
        assert HOLDOUT_SITE not in set(splits["train"][SITE_COL])
        assert HOLDOUT_SITE not in set(splits["val"][SITE_COL])
        n_total = sum(len(s) for s in splits.values())
        assert n_total == len(prepared.frame)

    def test_unknown_strategy_raises(self, prepared):
        with pytest.raises(ValueError, match="Nieznana strategia"):
            make_splits(prepared.frame, strategy="chronological")


# ─────────────────────────────────────────
# Zapis artefaktów (prepare_dataset)
# ─────────────────────────────────────────
class TestPrepareDataset:
    def test_writes_splits_manifest_and_audit(self, isolated_paths, raw_yale):
        paths = prepare_dataset(strategy="random", raw=raw_yale)
        p = isolated_paths
        for name, path in paths.items():
            assert path == getattr(p, f"{name.upper()}_PARQUET")
            assert path.exists()

        manifest = json.loads(p.DATA_MANIFEST_JSON.read_text())
        assert manifest["data_version"] == DATA_VERSION
        assert manifest["split_strategy"] == "random"
        assert manifest["meta_columns"] == META_COLS
        assert sum(manifest["n_rows"].values()) == manifest["audit"]["n_rows"]

        payload = json.loads(p.FEATURE_SETS_JSON.read_text())
        assert set(payload["feature_sets"]) == {"triage", "full"}
        assert manifest["n_features"] == {k: len(v) for k, v in payload["feature_sets"].items()}
        assert "chief_complaints" in payload["groups"]

        audit = p.DATA_AUDIT_MD.read_text(encoding="utf-8")
        assert audit.startswith("# Audyt danych")
        assert DATA_VERSION in audit
        assert "°F" in audit

    def test_split_files_contain_meta_and_full_features(self, isolated_paths, raw_yale):
        prepare_dataset(strategy="random", raw=raw_yale)
        payload = json.loads(isolated_paths.FEATURE_SETS_JSON.read_text())
        train = pd.read_parquet(isolated_paths.TRAIN_PARQUET)
        assert list(train.columns) == META_COLS + payload["feature_sets"]["full"]

    def test_site_strategy_written_to_manifest(self, isolated_paths, raw_yale):
        prepare_dataset(strategy="site", raw=raw_yale)
        manifest = json.loads(isolated_paths.DATA_MANIFEST_JSON.read_text())
        assert manifest["split_strategy"] == "site"
        test = pd.read_parquet(isolated_paths.TEST_PARQUET, columns=[SITE_COL])
        assert set(test[SITE_COL]) == {HOLDOUT_SITE}
