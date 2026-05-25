"""Model registry — loads .joblib models + imputation medians at startup."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

log = logging.getLogger(__name__)


class ModelRegistry:
    """Holds loaded ML models, their expected features, and imputation medians."""

    def __init__(self) -> None:
        self.models: dict[str, dict[str, Any]] = {}
        self.medians: dict[str, float] = {}
        self.feature_set: str = "triage_only"

    def load_from_dir(self, models_dir: Path) -> None:
        """Load latest version of each known model from disk."""
        models_dir = Path(models_dir)
        log.info("Loading models from %s", models_dir)

        # Map of model_id → glob pattern
        targets = {
            "catboost": "catboost_*.joblib",
            "lightgbm": "lightgbm_*.joblib",
            "xgboost":  "xgboost_*.joblib",
            "random_forest": "random_forest_*.joblib",
            "extra_trees":   "extra_trees_*.joblib",
            "hist_gbt":      "hist_gbt_*.joblib",
            "ebm":           "ebm_*.joblib",
        }

        # Look in backend/models first; fall back to project_root/models
        candidates_dirs = [models_dir, models_dir.parent.parent / "models"]

        for model_id, pattern in targets.items():
            found: Path | None = None
            for d in candidates_dirs:
                matches = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                if matches:
                    found = matches[0]
                    break
            if not found:
                log.warning("  %-15s — not found, skipping", model_id)
                continue

            try:
                data = joblib.load(found)
                self.models[model_id] = {
                    "model": data["model"],
                    "feature_names": data.get("feature_names", []),
                    "path": str(found),
                }
                log.info(
                    "  %-15s ✓ loaded (%d features) from %s",
                    model_id,
                    len(self.models[model_id]["feature_names"]),
                    found.name,
                )
            except Exception as exc:  # noqa: BLE001
                log.error("  %-15s ✗ failed: %s", model_id, exc)

        # Load imputation medians
        medians_path = models_dir / "imputation_medians.json"
        if medians_path.exists():
            with medians_path.open() as f:
                payload = json.load(f)
            self.medians = payload["medians"]
            self.feature_set = payload.get("feature_set", "triage_only")
            log.info("Loaded %d imputation medians (%s)", len(self.medians), self.feature_set)
        else:
            log.warning("No imputation_medians.json — missing features will be 0")

    @property
    def loaded_ids(self) -> list[str]:
        return sorted(self.models.keys())


# Singleton
registry = ModelRegistry()
