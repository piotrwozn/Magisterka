"""Wspólne fixture'y i konfiguracja pytest."""

from __future__ import annotations

import sys
from pathlib import Path

# Pozwól testom na import `from src...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    """
    Generuje mały syntetyczny dataset symulujący Yale EMMLC.

    100 wierszy z:
        - ESI 1-5
        - Vital signs (z realistycznym rozkładem)
        - Kilka chief complaints
    """
    rng = np.random.default_rng(42)
    n = 500  # większy sample by każda klasa miała >= 5 instancji (potrzebne do stratify)

    df = pd.DataFrame({
        "esi": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.45, 0.25, 0.10]),
        "age": rng.integers(0, 100, n),
        "sex": rng.choice(["M", "F"], n),
        "triage_sbp": rng.normal(125, 25, n).clip(60, 200),
        "triage_dbp": rng.normal(80, 15, n).clip(40, 120),
        "triage_pulse": rng.normal(85, 20, n).clip(30, 180),
        "triage_resp": rng.normal(18, 5, n).clip(8, 40),
        "triage_o2sat": rng.normal(96, 4, n).clip(70, 100),
        "triage_temp": rng.normal(36.7, 0.7, n).clip(34, 41),
        "triage_pain": rng.integers(0, 11, n),
        "cc_chestpain": rng.integers(0, 2, n),
        "cc_abdominalpain": rng.integers(0, 2, n),
        "cc_shortnessofbreath": rng.integers(0, 2, n),
        "cc_headache": rng.integers(0, 2, n),
        "pmh_diabetes": rng.integers(0, 2, n),
        "pmh_hypertension": rng.integers(0, 2, n),
        "med_betablocker": rng.integers(0, 2, n),
        "n_ed_visits": rng.integers(0, 10, n),
        "arrival_year": rng.choice([2014, 2015, 2016, 2017], n),
        "arrival_month": rng.integers(1, 13, n),
    })
    return df


@pytest.fixture
def synthetic_X_y(synthetic_dataset) -> tuple[pd.DataFrame, pd.Series]:
    """Wyciąga X, y z syntetycznego datasetu."""
    feature_cols = [c for c in synthetic_dataset.columns if c != "esi" and c != "sex"]
    X = synthetic_dataset[feature_cols].copy()
    y = synthetic_dataset["esi"] - 1  # 0..4
    return X, y
