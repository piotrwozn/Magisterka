"""Model registry — loads ESI model artifacts (.joblib), optional medians and the
ensemble configuration at startup.

Search order per model (first location with a valid artifact wins):
  1. backend/models/<model>_*.joblib or <model>.joblib   (explicit deployment, newest file)
  2. <repo>/models/<model>_*.joblib or <model>.joblib
  3. training/runs/<run>/model.joblib                     (runs with run_info.json), best
     run first — sort keys in priority order:
       a. trained on the full train split (a --sample / smoke / quick profile run is
          only a fallback, logged as such),
       b. tuned (Optuna) before untuned ("<profile>-notune"),
       c. plain profile runs before custom ones ("<profile>-custom": explicit
          budgets, feature filter, custom params, …),
       d. unweighted before class-weighted (weighted probabilities are not calibrated),
       e. finished (DONE.json) before interrupted,
       f. newest.

Artifacts are written by `BaseTriageModel.save()` (training/src/models/base.py):
dict with name, model (raw estimator / sklearn Pipeline), feature_names,
feature_set, classes_, class_names (["ESI1", …, "ESI5"]), label="esi", run_id,
data_version. Class index 0..4 = ESI 1..5. Legacy artifacts without label="esi"
and runs / artifacts of another data version than the training code's
DATA_VERSION (other feature schema) are skipped with a warning.

Ensemble weights + decision rule: `<models_dir>/ensemble_weights.json`
(format in app/models/ensemble.py; default = equal weights, argmax).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

import joblib

from app.config import ensure_training_on_path, get_settings
from app.models.ensemble import (
    ENSEMBLE_MODELS,
    ENSEMBLE_WEIGHTS_FILE,
    EnsembleConfig,
    load_ensemble_config,
)

log = logging.getLogger(__name__)

# Models served by the backend (training/src/models MODEL_REGISTRY)
TARGET_MODELS: tuple[str, ...] = ENSEMBLE_MODELS
N_CLASSES = 5

# Old feature-set names (same aliases as src.data.dataset)
_FEATURE_SET_ALIASES = {"triage_only": "triage", "top": "triage"}


def resolve_feature_set(name: str | None) -> str | None:
    if name is None:
        return None
    return _FEATURE_SET_ALIASES.get(name, name)


def training_data_version() -> str | None:
    """DATA_VERSION of the training code (src.utils.config) — the feature schema the
    runtime feature row is built for; None when the training sources are missing."""
    ensure_training_on_path()
    try:
        from src.utils.config import DATA_VERSION
    except Exception as exc:  # noqa: BLE001
        log.warning("Cannot import training src.utils.config (%s) — data_version not checked", exc)
        return None
    return DATA_VERSION


def _is_weighted(info: dict[str, Any]) -> bool:
    """Run trained with class weights (as training src.ensemble.runs._is_weighted)."""
    strategy = info.get("weights_strategy") or info.get("sample_weight_strategy")
    return strategy not in (None, "none") or bool(info.get("class_weights"))


def _is_custom(info: dict[str, Any], done: dict[str, Any]) -> bool:
    """Run with non-profile settings (DONE.json "custom"; older runs: tag / custom params)."""
    for source in (done, info):
        if "custom" in source:
            return bool(source["custom"])
    return (str(info.get("tag") or "").endswith("-custom")
            or bool(info.get("custom_params")) or bool(info.get("feature_filter")))


class _Candidate(NamedTuple):
    path: Path
    source: str
    train_sample: bool = False     # run trained on a --sample of the train split
    finished: bool = True          # run wrote DONE.json (its last artifact)
    tuned: bool = False            # hyperparameters tuned with Optuna
    custom: bool = False           # non-profile settings ("<profile>-custom")
    weighted: bool = False         # trained with class weights


class ModelRegistry:
    """Holds loaded ML models, their expected features, (optional) train medians
    and the ensemble configuration (weights + decision rule)."""

    def __init__(self, feature_set: str | None = None) -> None:
        self.models: dict[str, dict[str, Any]] = {}
        # Optional (imputation_medians.json): diagnostics, chief-complaint vocabulary and
        # the train mode of is_male (unknown sex). NOT used for imputation otherwise —
        # see feature_engineering.align_features / unknown_sex_value.
        self.medians: dict[str, float] = {}
        self.feature_set: str = resolve_feature_set(feature_set or get_settings().feature_set)
        self.ensemble: EnsembleConfig = EnsembleConfig()
        # Data version the served artifacts must have (training DATA_VERSION; None = any)
        self.data_version: str | None = None

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
        """training/runs/*/model.joblib per model, best run first (order in the module docstring)."""
        if not runs_dir.is_dir():
            return {}
        found: dict[str, list[_Candidate]] = {}
        for info_path in runs_dir.glob("*/run_info.json"):
            run_dir = info_path.parent
            model_path = run_dir / "model.joblib"
            if not model_path.exists():
                continue
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warning("  skipping run %s — unreadable run_info.json (%s)", run_dir.name, exc)
                continue
            if resolve_feature_set(info.get("feature_set")) != self.feature_set:
                continue
            if self.data_version is not None and info.get("data_version") != self.data_version:
                log.warning("  skipping run %s — data_version %r, the training code expects %r "
                            "(re-train on the current data)", run_dir.name,
                            info.get("data_version"), self.data_version)
                continue
            done_path = run_dir / "DONE.json"
            done: dict[str, Any] = {}
            if done_path.exists():
                try:
                    done = json.loads(done_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    done = {}
                if not isinstance(done, dict):
                    done = {}
            found.setdefault(info.get("model"), []).append(_Candidate(
                model_path,
                f"run {run_dir.name}",
                train_sample=info.get("train_sample") is not None,
                finished=done_path.exists(),
                tuned=bool(done.get("tuned", info.get("tuned", False))),
                custom=_is_custom(info, done),
                weighted=_is_weighted(info),
            ))
        for runs in found.values():
            # Stable sorts — the LAST sort is the primary key: full-data > tuned >
            # plain profile > unweighted > finished (DONE.json) > newest (directory
            # names start with YYYYmmdd_HHMMSS).
            runs.sort(key=lambda c: c.path.parent.name, reverse=True)
            runs.sort(key=lambda c: not c.finished)
            runs.sort(key=lambda c: c.weighted)
            runs.sort(key=lambda c: c.custom)
            runs.sort(key=lambda c: not c.tuned)
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
        if self.data_version is not None and data.get("data_version") != self.data_version:
            log.warning("  %-15s ✗ %s was trained on data_version %r, the training code (feature "
                        "row schema) expects %r — skipping; re-train it on the current data",
                        model_id, path, data.get("data_version"), self.data_version)
            return None
        classes = data.get("classes_")
        return {
            "model": data["model"],
            "feature_names": feature_names,
            "classes": [int(c) for c in classes] if classes is not None else list(range(N_CLASSES)),
            "class_names": class_names,
            "feature_set": artifact_fs,
            "run_id": data.get("run_id"),
            "data_version": data.get("data_version"),
            "path": str(path),
            "source": source,
        }

    def load_from_dir(
        self,
        models_dir: Path,
        runs_dir: Path | None = None,
        data_version: str | None = "current",
    ) -> None:
        """Load the best available artifact of each target model.

        models_dir   : deployment dir (backend/models); <models_dir>/../../models
                       (repo-root models/) is searched next
        runs_dir     : training runs dir; default <models_dir>/../../training/runs
        data_version : runs / artifacts of another data version are skipped;
                       "current" = training DATA_VERSION, None = no check
        """
        models_dir = Path(models_dir).resolve()
        project_root = models_dir.parent.parent
        runs_dir = Path(runs_dir) if runs_dir is not None else project_root / "training" / "runs"
        self.data_version = training_data_version() if data_version == "current" else data_version
        log.info("Loading models (feature set %r, data %s) from %s, %s, %s",
                 self.feature_set, self.data_version or "any version",
                 models_dir, project_root / "models", runs_dir)

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
                bundle["tuned"] = cand.tuned
                self.models[model_id] = bundle
                log.info("  %-15s ✓ loaded (%d features, data %s) from %s [%s]",
                         model_id, len(bundle["feature_names"]), bundle["data_version"],
                         cand.path, cand.source)
                if cand.train_sample:
                    log.warning("  %-15s ! trained on a sample of the train split "
                                "(smoke/quick run) — not for clinical evaluation", model_id)
                if cand.source.startswith("run "):
                    if not cand.tuned:
                        log.warning("  %-15s ! untuned run (default hyperparameters)", model_id)
                    if cand.custom:
                        log.warning("  %-15s ! custom run (non-profile settings)", model_id)
                    if cand.weighted:
                        log.warning("  %-15s ! trained with class weights — probabilities "
                                    "not calibrated", model_id)
                if not cand.finished:
                    log.warning("  %-15s ! run has no DONE.json (unfinished run or older "
                                "training version) — prefer a finished run", model_id)
                break
            else:
                log.warning("  %-15s — not found, skipping", model_id)

        self._load_medians(models_dir / "imputation_medians.json")
        self.ensemble = load_ensemble_config(models_dir / ENSEMBLE_WEIGHTS_FILE)
        active = self.ensemble.active(self.loaded_ids)
        if self.models and not active:
            log.error("No loaded model has a weight > 0 in %s — /api/v1/predict will fail",
                      self.ensemble.source)
        missing = [m for m, w in self.ensemble.weights.items() if w > 0 and m not in self.models]
        if missing and not self.ensemble.is_default:
            log.warning("Weighted models not loaded (weights renormalised over the rest): %s", missing)
        log.info("Ensemble: %s (method=%s, temperature=%s, rule=%s)", active,
                 self.ensemble.method, self.ensemble.temperature, self.ensemble.rule)

    def _load_medians(self, medians_path: Path) -> None:
        """Optional train-set medians (scripts/generate_imputation_medians.py)."""
        if not medians_path.exists():
            log.info("No imputation_medians.json — fine: missing features stay NaN")
            return
        with medians_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        self.medians = dict(payload.get("medians", {}))
        medians_fs = resolve_feature_set(payload.get("feature_set"))
        if medians_fs is not None and medians_fs != self.feature_set:
            log.warning("imputation_medians.json is for feature set %r, backend serves %r",
                        medians_fs, self.feature_set)
        medians_dv = payload.get("data_version")
        if self.data_version is not None and medians_dv != self.data_version:
            log.warning("imputation_medians.json is from data_version %r, the training code expects "
                        "%r — re-run scripts/generate_imputation_medians.py", medians_dv, self.data_version)
        log.info("Loaded %d train medians (%s) — diagnostics, cc_* vocabulary and the is_male mode "
                 "for an unknown sex; not used for imputation", len(self.medians), medians_fs)

    @property
    def loaded_ids(self) -> list[str]:
        return sorted(self.models.keys())


# Singleton
registry = ModelRegistry()
