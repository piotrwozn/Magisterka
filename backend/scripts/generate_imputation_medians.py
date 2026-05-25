"""Generate imputation medians from train set for missing features at inference time.

Run once locally:
    python backend/scripts/generate_imputation_medians.py

Output: backend/models/imputation_medians.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.preprocessing import build_feature_groups, split_features  # noqa: E402
from src.features.engineering import engineer_features  # noqa: E402
from src.utils.config import TRAIN_PARQUET  # noqa: E402


def main() -> None:
    print(f"Loading train set: {TRAIN_PARQUET}")
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"Rows: {len(df):,}")

    print("Running feature engineering...")
    df = engineer_features(df)
    groups = build_feature_groups(df)
    X, _, feat_names = split_features(df, groups, feature_set="triage_only")
    print(f"Features: {X.shape[1]}")

    print("Computing medians...")
    medians = X.median(numeric_only=True).to_dict()

    # Cast numpy types to plain Python so JSON serialises cleanly
    medians = {k: float(v) if pd.notna(v) else 0.0 for k, v in medians.items()}

    out_path = Path(__file__).resolve().parents[1] / "models" / "imputation_medians.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"feature_set": "triage_only", "n_features": len(medians), "medians": medians}, f, indent=2)

    print(f"\nSaved {len(medians)} feature medians → {out_path}")


if __name__ == "__main__":
    main()
