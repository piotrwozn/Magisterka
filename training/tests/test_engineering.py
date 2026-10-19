"""
Testy inżynierii cech klinicznych (`src.features.engineering`) i pomocników schematu Yale.

Wejście w schemacie kanonicznym: temperatura w °C, braki jako NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import yale_schema as ys
from src.features.engineering import ENGINEERED_FEATURES, VITALS, engineer_features

NORMAL = {
    "triage_vital_hr": 80.0,
    "triage_vital_sbp": 125.0,
    "triage_vital_dbp": 80.0,
    "triage_vital_rr": 16.0,
    "triage_vital_o2": 98.0,
    "triage_vital_o2_device": 0.0,
    "triage_vital_temp": 36.8,
    "age": 40.0,
    "arrival_ambulance": 0.0,
    "arrival_hour_bin": 3.0,
    "arrival_weekday": 3.0,
}


def _row(**changes) -> pd.DataFrame:
    return pd.DataFrame([{**NORMAL, **changes}])


def _only(**values) -> pd.DataFrame:
    """Jeden wiersz z wyłącznie podanymi parametrami (reszta NaN)."""
    base = {col: np.nan for col in VITALS.values()}
    base["triage_vital_o2_device"] = np.nan
    return pd.DataFrame([{**base, **values}])


def _feat(df: pd.DataFrame, name: str) -> float:
    return float(engineer_features(df)[name].iloc[0])


# ─────────────────────────────────────────
# Lista cech
# ─────────────────────────────────────────
class TestFeatureList:
    def test_engineered_features_unique(self):
        assert len(ENGINEERED_FEATURES) == len(set(ENGINEERED_FEATURES))

    def test_all_engineered_features_created(self):
        out = engineer_features(_row())
        added = [c for c in out.columns if c not in NORMAL]
        assert set(added) == set(ENGINEERED_FEATURES)

    def test_no_removed_mts_or_pediatric_features(self):
        for removed in ["temp_mts_bin", "mts_vitals_score", "ambulance_flag", "dbp_risk"]:
            assert removed not in ENGINEERED_FEATURES
        assert not [f for f in ENGINEERED_FEATURES if "pediatric" in f or "mts" in f]

    def test_returns_copy(self):
        df = _row()
        before = df.copy()
        engineer_features(df)
        pd.testing.assert_frame_equal(df, before)


# ─────────────────────────────────────────
# Odporność na brakujące kolumny / jeden wiersz
# ─────────────────────────────────────────
class TestRobustness:
    def test_one_row_with_missing_columns(self):
        out = engineer_features(pd.DataFrame({"triage_vital_hr": [120.0]}))
        assert len(out) == 1
        for col in ENGINEERED_FEATURES:
            assert col in out.columns
        assert out["tachycardia"].iloc[0] == 1.0
        assert np.isnan(out["fever"].iloc[0])           # brak temperatury → NaN, nie 0
        assert out["cc_count"].iloc[0] == 0.0
        assert out["n_vitals_missing"].iloc[0] == 5.0

    def test_empty_columns_frame(self):
        out = engineer_features(pd.DataFrame(index=[0]))
        assert out["all_vitals_missing"].iloc[0] == 1.0
        assert out["cc_critical"].iloc[0] == 0.0

    def test_no_duplicate_columns_when_rerun(self):
        once = engineer_features(_row(cc_chestpain=1.0))
        twice = engineer_features(once)
        assert not once.columns.duplicated().any()
        assert not twice.columns.duplicated().any()
        assert list(twice.columns) == list(once.columns)

    def test_rerun_is_idempotent(self):
        once = engineer_features(_row(cc_chestpain=1.0))
        twice = engineer_features(once)
        pd.testing.assert_frame_equal(once[ENGINEERED_FEATURES], twice[ENGINEERED_FEATURES])

    def test_index_preserved(self):
        df = pd.concat([_row(), _row(triage_vital_hr=140.0)]).set_axis([10, 42])
        out = engineer_features(df)
        assert list(out.index) == [10, 42]
        assert out.loc[42, "severe_tachycardia"] == 1.0


# ─────────────────────────────────────────
# Temperatura (°C) i propagacja braków
# ─────────────────────────────────────────
class TestTemperatureCelsius:
    @pytest.mark.parametrize("temp, fever, high_fever, hypothermia", [
        (36.0, 0.0, 0.0, 0.0),
        (35.9, 0.0, 0.0, 1.0),
        (37.9, 0.0, 0.0, 0.0),
        (38.0, 1.0, 0.0, 0.0),
        (39.0, 1.0, 1.0, 0.0),
    ])
    def test_thresholds(self, temp, fever, high_fever, hypothermia):
        out = engineer_features(_row(triage_vital_temp=temp))
        assert out["fever"].iloc[0] == fever
        assert out["high_fever"].iloc[0] == high_fever
        assert out["hypothermia"].iloc[0] == hypothermia
        assert out["temp_deviation"].iloc[0] == pytest.approx(abs(temp - 37.0))

    def test_fahrenheit_value_is_not_normal(self):
        """98.6 podane omyłkowo w °F to NIE norma w skali °C (konwersja należy do prepare)."""
        assert _feat(_row(triage_vital_temp=98.6), "fever") == 1.0


class TestMissingPropagation:
    @pytest.mark.parametrize("vital, flags", [
        ("triage_vital_temp", ["fever", "high_fever", "hypothermia", "temp_deviation"]),
        ("triage_vital_hr", ["tachycardia", "bradycardia", "esi_danger_hr", "shock_index"]),
        ("triage_vital_sbp", ["hypotension", "severe_hypotension", "shock_index", "map"]),
        ("triage_vital_rr", ["tachypnea", "bradypnea", "esi_danger_rr"]),
        ("triage_vital_o2", ["hypoxia", "severe_hypoxia", "esi_danger_spo2"]),
    ])
    def test_flag_nan_when_vital_missing(self, vital, flags):
        out = engineer_features(_row(**{vital: np.nan}))
        for flag in flags:
            assert np.isnan(out[flag].iloc[0]), flag

    def test_missing_flags(self):
        out = engineer_features(_row(triage_vital_hr=np.nan, triage_vital_temp=np.nan))
        assert out["hr_missing"].iloc[0] == 1.0
        assert out["temp_missing"].iloc[0] == 1.0
        assert out["sbp_missing"].iloc[0] == 0.0
        assert out["n_vitals_missing"].iloc[0] == 2.0
        assert out["all_vitals_missing"].iloc[0] == 0.0

    def test_all_vitals_missing(self):
        out = engineer_features(_only())
        assert out["all_vitals_missing"].iloc[0] == 1.0
        assert out["n_vitals_missing"].iloc[0] == float(len(VITALS))
        assert np.isnan(out["news2_score"].iloc[0])
        assert np.isnan(out["esi_danger_zone"].iloc[0])

    def test_normal_row_has_no_missing(self):
        out = engineer_features(_row())
        assert out["n_vitals_missing"].iloc[0] == 0.0
        assert out["all_vitals_missing"].iloc[0] == 0.0


# ─────────────────────────────────────────
# Strefa zagrożenia ESI (punkt D)
# ─────────────────────────────────────────
class TestEsiDangerZone:
    @pytest.mark.parametrize("changes, expected", [
        ({}, 0.0),
        ({"triage_vital_hr": 101.0}, 1.0),
        ({"triage_vital_hr": 100.0}, 0.0),
        ({"triage_vital_rr": 21.0}, 1.0),
        ({"triage_vital_rr": 20.0}, 0.0),
        ({"triage_vital_o2": 91.0}, 1.0),
        ({"triage_vital_o2": 92.0}, 0.0),
    ])
    def test_single_criterion(self, changes, expected):
        assert _feat(_row(**changes), "esi_danger_zone") == expected

    def test_count(self):
        out = engineer_features(_row(triage_vital_hr=120.0, triage_vital_rr=28.0, triage_vital_o2=88.0))
        assert out["esi_danger_count"].iloc[0] == 3.0
        assert out["esi_danger_zone"].iloc[0] == 1.0

    def test_partial_missing_uses_available(self):
        out = engineer_features(_row(triage_vital_hr=np.nan))
        assert np.isnan(out["esi_danger_hr"].iloc[0])
        assert out["esi_danger_count"].iloc[0] == 0.0
        assert out["esi_danger_zone"].iloc[0] == 0.0

    def test_ambulance_danger_zone(self):
        out = engineer_features(_row(arrival_ambulance=1.0, triage_vital_hr=130.0))
        assert out["ambulance_danger_zone"].iloc[0] == 1.0
        assert _feat(_row(arrival_ambulance=0.0, triage_vital_hr=130.0), "ambulance_danger_zone") == 0.0


# ─────────────────────────────────────────
# NEWS2
# ─────────────────────────────────────────
class TestNews2:
    @pytest.mark.parametrize("values, points", [
        ({"triage_vital_rr": 25.0}, 3),
        ({"triage_vital_rr": 8.0}, 3),
        ({"triage_vital_rr": 10.0}, 1),
        ({"triage_vital_rr": 16.0}, 0),
        ({"triage_vital_rr": 22.0}, 2),
        ({"triage_vital_o2": 92.0}, 2),
        ({"triage_vital_o2": 91.0}, 3),
        ({"triage_vital_o2": 95.0}, 1),
        ({"triage_vital_o2": 96.0}, 0),
        ({"triage_vital_sbp": 95.0}, 2),
        ({"triage_vital_sbp": 90.0}, 3),
        ({"triage_vital_sbp": 105.0}, 1),
        ({"triage_vital_sbp": 220.0}, 3),
        ({"triage_vital_hr": 135.0}, 3),
        ({"triage_vital_hr": 115.0}, 2),
        ({"triage_vital_hr": 95.0}, 1),
        ({"triage_vital_hr": 45.0}, 1),
        ({"triage_vital_hr": 40.0}, 3),
        ({"triage_vital_temp": 39.5}, 2),
        ({"triage_vital_temp": 38.5}, 1),
        ({"triage_vital_temp": 35.5}, 1),
        ({"triage_vital_temp": 35.0}, 3),
        ({"triage_vital_temp": 37.0}, 0),
        ({"triage_vital_o2_device": 1.0}, 2),
        ({"triage_vital_o2_device": 0.0}, 0),
    ])
    def test_component_points(self, values, points):
        assert _feat(_only(**values), "news2_score") == points

    def test_normal_row_scores_zero(self):
        out = engineer_features(_row())
        assert out["news2_score"].iloc[0] == 0.0
        assert out["news2_high"].iloc[0] == 0.0

    def test_sum_and_high(self):
        df = _only(triage_vital_rr=25.0, triage_vital_o2=92.0, triage_vital_sbp=95.0,
                   triage_vital_hr=135.0, triage_vital_temp=39.5, triage_vital_o2_device=1.0)
        out = engineer_features(df)
        assert out["news2_score"].iloc[0] == 3 + 2 + 2 + 3 + 2 + 2
        assert out["news2_high"].iloc[0] == 1.0


# ─────────────────────────────────────────
# Hemodynamika, SIRS, qSOFA, wiek, czas, skargi
# ─────────────────────────────────────────
class TestOtherFeatures:
    def test_shock_index(self):
        out = engineer_features(_row(triage_vital_hr=120.0, triage_vital_sbp=100.0, triage_vital_dbp=70.0))
        assert out["shock_index"].iloc[0] == pytest.approx(1.2)
        assert out["shock_index_critical"].iloc[0] == 1.0
        assert out["map"].iloc[0] == pytest.approx(70 + 30 / 3)
        assert out["pulse_pressure"].iloc[0] == 30.0
        assert out["modified_shock_index"].iloc[0] == pytest.approx(120 / 80)

    def test_shock_index_normal(self):
        out = engineer_features(_row())
        assert out["shock_index"].iloc[0] == pytest.approx(80 / 125)
        assert out["shock_index_critical"].iloc[0] == 0.0

    def test_sirs(self):
        out = engineer_features(_row(triage_vital_temp=38.5, triage_vital_hr=95.0, triage_vital_rr=22.0))
        assert out["sirs_score"].iloc[0] == 3.0
        assert out["sirs_positive"].iloc[0] == 1.0

    def test_qsofa_uses_altered_mental_status_complaint(self):
        out = engineer_features(_row(triage_vital_rr=24.0, cc_alteredmentalstatus=1.0))
        assert out["qsofa_score"].iloc[0] == 2.0
        assert out["qsofa_positive"].iloc[0] == 1.0

    def test_age_flags(self):
        out = engineer_features(_row(age=82.0))
        assert out["age_65plus"].iloc[0] == 1.0
        assert out["age_80plus"].iloc[0] == 1.0
        assert np.isnan(_feat(_row(age=np.nan), "age_65plus"))

    @pytest.mark.parametrize("hour_bin, night", [(0, 1.0), (1, 1.0), (2, 0.0), (5, 0.0)])
    def test_night_arrival(self, hour_bin, night):
        assert _feat(_row(arrival_hour_bin=float(hour_bin)), "night_arrival") == night

    @pytest.mark.parametrize("weekday, weekend", [(1, 0.0), (5, 0.0), (6, 1.0), (7, 1.0)])
    def test_weekend_uses_monday_one_encoding(self, weekday, weekend):
        assert _feat(_row(arrival_weekday=float(weekday)), "weekend_arrival") == weekend

    def test_chief_complaint_groups(self):
        out = engineer_features(_row(cc_chestpain=1.0, cc_fall_65=1.0, cc_cardiacarrest=1.0,
                                     cc_rash=np.nan, triage_vital_hr=110.0))
        assert out["cc_count"].iloc[0] == 3.0
        assert out["cc_critical"].iloc[0] == 1.0
        assert out["cc_group_cardiac"].iloc[0] == 2.0
        assert out["cc_group_trauma"].iloc[0] == 1.0
        assert out["cc_group_low_acuity"].iloc[0] == 0.0
        assert out["chestpain_tachycardia"].iloc[0] == 1.0

    def test_elderly_hypotension(self):
        assert _feat(_row(age=70.0, triage_vital_sbp=95.0), "elderly_hypotension") == 1.0
        assert _feat(_row(age=40.0, triage_vital_sbp=95.0), "elderly_hypotension") == 0.0


# ─────────────────────────────────────────
# Schemat Yale: przedziały godzin
# ─────────────────────────────────────────
class TestHourToBin:
    EXPECTED = {
        **{h: 0 for h in (23, 0, 1, 2)},
        **{h: 1 for h in range(3, 7)},
        **{h: 2 for h in range(7, 11)},
        **{h: 3 for h in range(11, 15)},
        **{h: 4 for h in range(15, 19)},
        **{h: 5 for h in range(19, 23)},
    }

    @pytest.mark.parametrize("hour", range(24))
    def test_all_hours(self, hour):
        assert ys.hour_to_bin(hour) == self.EXPECTED[hour]

    @pytest.mark.parametrize("label, bin_idx", list(ys.HOUR_BIN_MAP.items()))
    def test_consistent_with_yale_labels(self, label, bin_idx):
        start, end = (int(x) for x in label.split("-"))
        hours = [h % 24 for h in range(start, end + 1 + (24 if end < start else 0))]
        assert hours
        assert all(ys.hour_to_bin(h) == bin_idx for h in hours)

    def test_wraps_modulo_24(self):
        assert ys.hour_to_bin(24) == ys.hour_to_bin(0)
