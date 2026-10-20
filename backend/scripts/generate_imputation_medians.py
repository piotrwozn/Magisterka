"""Generate train-set medians of the model features (backend/models/imputation_medians.json).

The medians are OPTIONAL. Inference does NOT impute them: missing features stay
NaN exactly as in training (tree models / EBM handle NaN natively, logreg
imputes inside its own Pipeline) — see app.pipeline.feature_engineering.align_features
(use_medians=True is an explicit, non-default fallback). The file serves as:
  - diagnostics: typical patient values (temperature in °C) for sanity checks,
  - vocabulary: the orchestrator reads the cc_* keys as the list of
    chief-complaint columns of the trained models,
  - unknown sex: training has no missing is_male, so a request without sex gets
    the train mode — the median of the 0/1 column (feature_engineering.unknown_sex_value).

Regenerate whenever the training data are re-prepared (DATA_VERSION changes):
    python backend/scripts/generate_imputation_medians.py [--feature-set triage]

Requires prepared splits (training/scripts/02_preprocess.py).
Output: backend/models/imputation_medians.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "training"
sys.path.insert(0, str(TRAINING_ROOT))

from src.data.dataset import feature_names, load_split, resolve_feature_set  # noqa: E402
from src.utils.config import DATA_VERSION, TRAIN_PARQUET  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "imputation_medians.json"


def compute_medians(feature_set: str = "triage") -> tuple[dict[str, float | None], int]:
    """Medians of the feature set on the TRAIN split (NaN skipped; all-NaN → None)."""
    names = feature_names(feature_set)
    df = load_split("train", columns=names)
    medians = df[names].astype(float).median()
    # float32 data → round to 4 decimals (36.66666793823242 → 36.6667)
    out = {
        name: (None if math.isnan(value) else round(float(value), 4))
        for name, value in medians.items()
    }
    return out, len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-set", default="triage", help="triage (default) | full")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    feature_set = resolve_feature_set(args.feature_set)
    print(f"Loading train split: {TRAIN_PARQUET} (feature set '{feature_set}')")
    medians, n_rows = compute_medians(feature_set)
    print(f"Rows: {n_rows:,} | features: {len(medians)}")

    temp = medians.get("triage_vital_temp")
    if temp is not None and not 35.0 <= temp <= 38.5:
        raise SystemExit(f"Temperature median {temp:.1f} is not in °C — re-prepare the data")

    payload = {
        "feature_set": feature_set,
        "data_version": DATA_VERSION,
        "n_rows": n_rows,
        "n_features": len(medians),
        "note": "Train-set medians (diagnostics + cc_* vocabulary; is_male = train mode for an "
                "unknown sex). Not used for imputation otherwise — missing features stay NaN. "
                "Temperature in °C.",
        "medians": medians,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(medians)} feature medians → {args.output}")


if __name__ == "__main__":
    main()
