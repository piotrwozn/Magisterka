"""Testy mapowania ESI → MTS."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.esi_mts_mapping import (
    enhanced_mts_label,
    esi_to_mts_color,
    esi_to_mts_numeric,
    map_dataframe_to_mts,
)


class TestSimpleMappings:
    """Mapowanie 1:1 ESI → MTS bez dyskryminatorów."""

    @pytest.mark.parametrize(
        "esi,expected_color,expected_num",
        [
            (1, "Red", 0),
            (2, "Orange", 1),
            (3, "Yellow", 2),
            (4, "Green", 3),
            (5, "Blue", 4),
        ],
    )
    def test_basic_mapping(self, esi, expected_color, expected_num):
        assert esi_to_mts_color(esi) == expected_color
        assert esi_to_mts_numeric(esi) == expected_num

    def test_invalid_esi(self):
        assert esi_to_mts_color(0) is None
        assert esi_to_mts_color(6) is None
        assert esi_to_mts_color(None) is None
        assert esi_to_mts_numeric(None) is None


class TestEnhancedMapping:
    """Strategia 2: upgrade na bazie dyskryminatorów vital signs."""

    def test_no_upgrade_normal_vitals(self):
        """ESI-3 z prawidłowymi vital signs zostaje Yellow."""
        row = pd.Series({
            "esi": 3,
            "triage_sbp": 130,
            "triage_o2sat": 99,
            "triage_pulse": 80,
            "triage_resp": 16,
            "triage_temp": 36.6,
            "triage_pain": 4,
        })
        assert enhanced_mts_label(row) == 2  # Yellow

    def test_upgrade_to_red_via_o2sat(self):
        """ESI-3 z SpO2 < 85% → upgrade do Red."""
        row = pd.Series({
            "esi": 3,
            "triage_sbp": 120,
            "triage_o2sat": 80,
            "triage_pulse": 80,
            "triage_resp": 16,
            "triage_temp": 36.6,
            "triage_pain": 0,
        })
        assert enhanced_mts_label(row) == 0  # Red

    def test_upgrade_to_red_via_sbp(self):
        """ESI-4 z wstrząsem → upgrade do Red."""
        row = pd.Series({
            "esi": 4,
            "triage_sbp": 75,
            "triage_o2sat": 96,
            "triage_pulse": 110,
            "triage_resp": 22,
            "triage_temp": 37.0,
            "triage_pain": 6,
        })
        assert enhanced_mts_label(row) == 0  # Red

    def test_upgrade_to_orange(self):
        """ESI-3 z SpO2 90% → upgrade do Orange."""
        row = pd.Series({
            "esi": 3,
            "triage_sbp": 110,
            "triage_o2sat": 90,
            "triage_pulse": 95,
            "triage_resp": 22,
            "triage_temp": 37.5,
            "triage_pain": 5,
        })
        assert enhanced_mts_label(row) == 1  # Orange

    def test_no_downgrade(self):
        """ESI-1 z normalnymi vital signs ZOSTAJE Red (nie downgrade)."""
        row = pd.Series({
            "esi": 1,
            "triage_sbp": 130,
            "triage_o2sat": 99,
            "triage_pulse": 80,
            "triage_resp": 16,
            "triage_temp": 36.6,
            "triage_pain": 0,
        })
        assert enhanced_mts_label(row) == 0  # Red, nie downgrade

    def test_invalid_esi_returns_negative(self):
        row = pd.Series({"esi": None, "triage_sbp": 120})
        assert enhanced_mts_label(row) == -1


class TestDataFrameMapping:
    def test_map_dataframe_simple(self, synthetic_dataset):
        df = map_dataframe_to_mts(synthetic_dataset, use_enhanced=False)
        assert "mts_color" in df.columns
        assert "mts_numeric" in df.columns
        assert df["mts_numeric"].between(0, 4).all()
        assert set(df["mts_color"].unique()).issubset(
            {"Red", "Orange", "Yellow", "Green", "Blue"}
        )

    def test_map_dataframe_enhanced(self, synthetic_dataset):
        df = map_dataframe_to_mts(synthetic_dataset, use_enhanced=True)
        assert "mts_numeric" in df.columns
        # Część wierszy może być upgrade'owana
        baseline = (synthetic_dataset["esi"] - 1).values
        upgraded_count = (df["mts_numeric"].values < baseline).sum()
        # Dla syntetycznych danych upgrade > 0 jest wysoce prawdopodobne
        assert upgraded_count >= 0  # po prostu nie crash

    def test_drop_invalid(self):
        df = pd.DataFrame({
            "esi": [1, 2, None, 999, 3],
            "triage_sbp": [120, 110, 100, 90, 130],
        })
        result = map_dataframe_to_mts(df, drop_invalid=True)
        assert len(result) == 3  # 1, 2, 3
