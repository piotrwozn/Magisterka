"""Runtime feature engineering for a single patient row.

Wraps src.features.engineering.engineer_features() (which works on DataFrames)
and produces a 336-feature row aligned to a model's expected feature_names,
filling missing values from imputation medians.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.models.schemas import Vitals

log = logging.getLogger(__name__)

# Add training/ source tree so we can import src.features.engineering
_settings = get_settings()
_training_root = str(_settings.project_root / "training")
if _training_root not in sys.path:
    sys.path.insert(0, _training_root)

from src.features.engineering import ENGINEERED_FEATURES, engineer_features  # noqa: E402

# Columns that engineer_features() will produce; we MUST NOT pre-populate them
# in the raw row (otherwise pd.concat([df, c_df]) inside engineer_features
# would create duplicate columns).
_ENGINEERED_SET = set(ENGINEERED_FEATURES)

# cc_* columns that engineer_features() reads via df.get(); if absent the call
# returns scalar 0 (not a Series) → .astype() crashes. We always pre-populate
# these with 0 to guarantee a Series.
_REQUIRED_CC_COLS: tuple[str, ...] = (
    "cc_cardiacarrest", "cc_chestpain", "cc_respiratorydistress",
    "cc_strokealert", "cc_fulltrauma", "cc_alteredmentalstatus",
    "cc_suicidal", "cc_alcoholintoxication", "cc_psychiatricevaluation",
    "cc_headache", "cc_abdominalpain",
)


def build_feature_row(
    vitals: Vitals,
    cc_features: dict[str, int],
    extra: dict[str, float] | None = None,
    all_cc_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build single-row DataFrame matching the training schema.

    vitals          : data from frontend form
    cc_features     : {cc_chest_pain: 1, ...} from Layer 0 parser
    extra           : optional manual overrides for arrival_*, demographics
    all_cc_columns  : full list of cc_* columns that existed in training
                      (others are filled with 0 — absence of complaint)
    """
    now = datetime.now()

    # Yale EMMLC training data uses Fahrenheit for body temperature.
    # Frontend collects Celsius, so convert before feeding the model.
    temp_f = vitals.temp * 9.0 / 5.0 + 32.0

    # Pre-populate every known cc_* column with 0 so engineer_features.get()
    # always receives a Series rather than a scalar fallback.
    # Exclude any cc_* names that engineer_features will produce itself
    # (e.g. cc_cardiac_group) to avoid duplicate columns after concat.
    cc_base: dict[str, int] = {c: 0 for c in _REQUIRED_CC_COLS}
    if all_cc_columns:
        for c in all_cc_columns:
            if c not in _ENGINEERED_SET:
                cc_base[c] = 0
    for k, v in cc_features.items():
        if k not in _ENGINEERED_SET:
            cc_base[k] = v

    row: dict[str, float | int] = {
        # Triage vitals (real names used in training)
        "triage_vital_hr":   vitals.hr,
        "triage_vital_sbp":  vitals.sbp,
        "triage_vital_dbp":  vitals.dbp,
        "triage_vital_rr":   vitals.rr,
        "triage_vital_o2":   vitals.o2,
        "triage_vital_temp": temp_f,
        # Demographics — only age comes from form
        "age": vitals.age,
        # Arrival metadata (auto-derived from system clock)
        "arrivalmode":       4,                  # 4 = walk-in (median in train set)
        "arrivalmonth":      now.month,
        "arrivalday":        now.day,
        "arrivalhour_bin":   now.hour,           # raw 0..23 — engineer_features bins it
        "arrival_hour":      now.hour,
        "arrival_dayofweek": now.weekday(),
        # Chief complaint binary flags
        **cc_base,
    }
    if extra:
        row.update(extra)

    df = pd.DataFrame([row])
    df = engineer_features(df)
    return df


def align_features(
    df: pd.DataFrame,
    expected: list[str],
    medians: dict[str, float],
) -> pd.DataFrame:
    """Pad/select columns to match the model's expected feature list.

    Missing columns are filled with the train-set median (or 0.0 if unknown).
    Extra columns are dropped.
    """
    missing = [c for c in expected if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = medians.get(c, 0.0)
    return df[expected].astype(float, errors="ignore")
