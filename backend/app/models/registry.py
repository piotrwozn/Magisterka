"""Model registry — loads ESI model artifacts (.joblib) + optional medians at startup.

Search order per model (first location with a valid artifact wins):
  1. backend/models/<model>_*.joblib or <model>.joblib   (explicit deployment, newest file)
  2. <repo>/models/<model>_*.joblib or <model>.joblib
  3. training/runs/<run>/model.joblib                     (newest complete run per model;
     runs trained on the full train split are preferred over --sample runs)

Artifacts are written by `BaseTriageModel.save()` (training/src/models/base.py):
dict with name, model (raw estimator / sklearn Pipeline), feature_names,
feature_set, class_names (["ESI1", …, "ESI5"]), label="esi", run_id, data_version.
Class index 0..4 = ESI 1..5. Legacy artifacts without label="esi" are skipped.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

import joblib

from app.config import get_settings

log = logging.getLogger(__name__)

# Models served by the backend (training/src/models MODEL_REGISTRY)
TARGET_MODELS: tuple[str, ...] = (
    "xgboost", "lightgbm", "catboost", "random_forest", "ebm", "logreg",
)
N_CLASSES = 5

# Old feature-set names (same aliases as src.data.dataset)
_FEATURE_SET_ALIASES = {"triage_only": "triage", "top": "triage"}


def resolve_feature_set(name: str | None) -> str | None:
    if name is None:
        return None
    return _FEATURE_SET_ALIASES.get(name, name)


class _Candidate(NamedTuple):
    path: Path
    source: str
    train_sample: bool = False     # run trained on a --sample of the train split


class ModelRegistry:
    """Holds loaded ML models, their expected features, and (optional) train medians."""

    def __init__(self, feature_set: str | None = None) -> None:
        self.models: dict[str, dict[str, Any]] = {}
        # Optional (imputation_medians.json): diagnostics + chief-complaint vocabulary.
        # NOT used for imputation by default — see feature_engineering.align_features.
        self.medians: dict[str, float] = {}
        self.feature_set: str = resolve_feature_set(feature_set or get_settings().feature_set)

    # ── Discovery ────────────────────────────────────────────────────

    @staticmethod
    def _file_candidates(model_id: str, dirs: list[Path]) -> list[_Candidate]:
        """Joblib files for a model in deployment dirs (dir order, newest file first)."""
        out: list[_Candidate] = []
        for d in dirs:
            if not d.is_dir():
                continue
            matches = {*d.glob(f"{model_id}_*.joblib"), *d.glob(f"{model_id}.joblib")}
            for path in sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True):
                out.append(_Candidate(path, str(d)))
        return out

    def _run_candidates(self, runs_dir: Path) -> dict[str, list[_Candidate]]:
        """training/runs/*/model.joblib per model — full-data runs first, newest first."""
        if not runs_dir.is_dir():
            return {}
        found: dict[str, list[_Candidate]] = {}
        for info_path in runs_dir.glob("*/run_info.json"):
            model_path = info_path.parent / "model.joblib"
            if not model_path.exists():
                continue
            try:
                info = json.loads(info_path.read_text())
            except (OSError, ValueError) as exc:
                log.warning("  skipping run %s — unreadable run_info.json (%s)", info_path.parent.name, exc)
                continue
            if resolve_feature_set(info.get("feature_set")) != self.feature_set:
                continue
            found.setdefault(info.get("model"), []).append(_Candidate(
                model_path,
                f"run {info_path.parent.name}",
                train_sample=info.get("train_sample") is not None,
            ))
        for runs in found.values():
            # Newest run first (directory names start with YYYYmmdd_HHMMSS), then
            # full-data runs before --sample runs (stable sort keeps the date order).
            runs.sort(key=lambda c: c.path.parent.name, reverse=True)
            runs.sort(key=lambda c: c.train_sample)
        return found

    # ── Loading ──────────────────────────────────────────────────────

    def _load_artifact(self, model_id: str, path: Path, source: str) -> dict[str, Any] | None:
        """Load + validate one artifact; None (with a log line) if unusable."""
        try:
            data = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            log.error("  %-15s ✗ failed to load %s: %s", model_id, path, exc)
            return None
        if not isinstance(data, dict) or "model" not in data:
            log.warning("  %-15s ✗ %s is not a SOR-AI artifact — skipping", model_id, path)
            return None
        if data.get("label") != "esi":
            log.warning("  %-15s ✗ %s is a legacy artifact (label=%r, expected 'esi') — skipping",
                        model_id, path, data.get("label"))
            return None
        if data.get("name", model_id) != model_id:
            log.warning("  %-15s ✗ %s contains model %r — skipping", model_id, path, data.get("name"))
            return None
        feature_names = list(data.get("feature_names") or [])
        class_names = list(data.get("class_names") or [])
        if not feature_names:
            log.warning("  %-15s ✗ %s has no feature_names — skipping", model_id, path)
            return None
        if len(class_names) != N_CLASSES:
            log.warning("  %-15s ✗ %s has class_names=%s (expected %d ESI classes) — skipping",
                        model_id, path, class_names, N_CLASSES)
            return None
        artifact_fs = resolve_feature_set(data.get("feature_set"))
        if artifact_fs is not None and artifact_fs != self.feature_set:
            log.warning("  %-15s ✗ %s uses feature set %r (backend serves %r) — skipping",
                        model_id, path, artifact_fs, self.feature_set)
            return None
        return {
            "model": data["model"],
            "feature_names": feature_names,
            "class_names": class_names,
            "feature_set": artifact_fs,
            "run_id": data.get("run_id"),
            "data_version": data.get("data_version"),
            "path": str(path),
            "source": source,
        }

    def load_from_dir(self, models_dir: Path, runs_dir: Path | None = None) -> None:
        """Load the best available artifact of each target model.

        models_dir : deployment dir (backend/models); <models_dir>/../../models
                     (repo-root models/) is searched next
        runs_dir   : training runs dir; default <models_dir>/../../training/runs
        """
        models_dir = Path(models_dir).resolve()
        project_root = models_dir.parent.parent
        runs_dir = Path(runs_dir) if runs_dir is not None else project_root / "training" / "runs"
        log.info("Loading models (feature set %r) from %s, %s, %s",
                 self.feature_set, models_dir, project_root / "models", runs_dir)

        deploy_dirs = list(dict.fromkeys([models_dir, project_root / "models"]))
        run_candidates = self._run_candidates(runs_dir)

        for model_id in TARGET_MODELS:
            candidates = (
                self._file_candidates(model_id, deploy_dirs) + run_candidates.get(model_id, [])
            )
            for cand in candidates:
                bundle = self._load_artifact(model_id, cand.path, cand.source)
                if bundle is None:
                    continue
                bundle["train_sample"] = cand.train_sample
                self.models[model_id] = bundle
                log.info("  %-15s ✓ loaded (%d features) from %s [%s]",
                         model_id, len(bundle["feature_names"]), cand.path, cand.source)
                if cand.train_sample:
                    log.warning("  %-15s ! trained on a --sample of the train split "
                                "(smoke test) — not for clinical evaluation", model_id)
                break
            else:
                log.warning("  %-15s — not found, skipping", model_id)

        self._load_medians(models_dir / "imputation_medians.json")

    def _load_medians(self, medians_path: Path) -> None:
        """Optional train-set medians (scripts/generate_imputation_medians.py)."""
        if not medians_path.exists():
            log.info("No imputation_medians.json — fine: missing features stay NaN")
            return
        with medians_path.open() as f:
            payload = json.load(f)
        self.medians = dict(payload.get("medians", {}))
        medians_fs = resolve_feature_set(payload.get("feature_set"))
        if medians_fs is not None and medians_fs != self.feature_set:
            log.warning("imputation_medians.json is for feature set %r, backend serves %r",
                        medians_fs, self.feature_set)
        log.info("Loaded %d train medians (%s) — diagnostics only, not used for imputation",
                 len(self.medians), medians_fs)

    @property
    def loaded_ids(self) -> list[str]:
        return sorted(self.models.keys())


# Singleton
registry = ModelRegistry()
