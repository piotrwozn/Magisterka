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

Tabular foundation models (TFM_MODELS: tabpfn, tabicl) — OFF by default. They are
considered only when `ensemble_weights.json` gives them a weight > 0 AND the operator
opts in with `SOR_AI_BACKEND_TFM=1`; serving then requires:
  * torch with CUDA (no GPU → not served: a TFM context of 25k–100k rows per bag
    would take minutes per request on CPU), the library (tabpfn / tabicl) and the
    training sources on sys.path (the artifact pickles `src.models.tfm_base.TFMPredictor`),
  * a start-up VRAM budget: the summed KV-cache estimate of all bags of all served
    TFMs (`TFMPredictor.estimated_cache_bytes()`) + a reserve
    (`SOR_AI_BACKEND_TFM_RESERVE_MB`, default 2048) must fit the FREE VRAM
    (`torch.cuda.mem_get_info`; the GPU is shared with Ollama),
  * a successful warm-up of every bag at start-up (`keep_fitted_bags="all"`, so a
    request never re-fits a bag).
Any failure — or the default opt-out — means the weighted TFMs are not served:
the registry then switches to `<models_dir>/ensemble_weights_no_tfm.json` (the same
recipe re-fitted on VAL without TFMs, written by training 08_final_report.py) with a
WARNING; without that file the weights are renormalised over the remaining models
and a WARNING says that the temperature / rule parameters were fitted for the full
blend. A TFM that fails during a request is dropped and the same fallback applies.
Note: TabPFN reads `TABPFN_*` settings from a `.env` file in the working directory,
so a `TABPFN_MODEL_CACHE_DIR=...` line in `backend/.env` silently redirects where the
backend looks for the TabPFN weights.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import joblib

from app.config import ensure_training_on_path, get_settings
from app.models.ensemble import (
    ENSEMBLE_MODELS,
    ENSEMBLE_WEIGHTS_FILE,
    ENSEMBLE_WEIGHTS_NO_TFM_FILE,
    EnsembleConfig,
    load_ensemble_config,
)

log = logging.getLogger(__name__)

# Classical models served by the backend (training/src/models MODEL_REGISTRY, CLASSICAL_MODELS)
TARGET_MODELS: tuple[str, ...] = ENSEMBLE_MODELS
N_CLASSES = 5

# Opt-in TFM serving (see the module docstring)
TFM_SERVING_ENV = "SOR_AI_BACKEND_TFM"
TFM_RESERVE_ENV = "SOR_AI_BACKEND_TFM_RESERVE_MB"
DEFAULT_TFM_RESERVE_MB = 2048
_TRUE_VALUES = ("1", "true", "yes", "on")
_MB = 2 ** 20


def tfm_serving_enabled() -> bool:
    """SOR_AI_BACKEND_TFM=1 (true / yes / on) — the operator opted in to serving TFMs (default: off)."""
    return os.environ.get(TFM_SERVING_ENV, "").strip().lower() in _TRUE_VALUES


def tfm_reserve_bytes() -> int:
    """VRAM kept free on top of the summed TFM caches (activations, Ollama spikes) — MB from the env."""
    raw = os.environ.get(TFM_RESERVE_ENV, "").strip()
    try:
        value = float(raw) if raw else float(DEFAULT_TFM_RESERVE_MB)
    except ValueError:
        log.warning("%s=%r is not a number — using %d MB", TFM_RESERVE_ENV, raw, DEFAULT_TFM_RESERVE_MB)
        value = float(DEFAULT_TFM_RESERVE_MB)
    return int(max(0.0, value) * _MB)


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class TFMRuntime:
    """Can this process serve TFMs: torch + CUDA (with free / total VRAM) and the libraries per model."""

    cuda: bool
    reason: str | None = None
    free_vram_bytes: int | None = None
    total_vram_bytes: int | None = None
    libraries: dict[str, bool] = field(default_factory=dict)


def tfm_runtime_status(models: list[str]) -> TFMRuntime:
    """torch / CUDA / free VRAM and the libraries of `models` — imports torch only when called
    (i.e. only after the operator opted in and a TFM is weighted)."""
    libraries = {m: _has_module(m) for m in models}
    if not _has_module("torch"):
        return TFMRuntime(False, "torch is not installed (training/requirements-tfm.txt + a CUDA wheel)",
                          libraries=libraries)
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — broken install (e.g. a missing DLL on Windows)
        return TFMRuntime(False, f"import torch failed: {type(exc).__name__}: {exc}", libraries=libraries)
    try:
        if not torch.cuda.is_available():
            return TFMRuntime(False, f"CUDA is not available to torch {torch.__version__} (CUDA build "
                                     f"{torch.version.cuda}) — TFM serving needs an NVIDIA GPU", libraries=libraries)
        free, total = torch.cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001 — driver / CUDA error
        return TFMRuntime(False, f"CUDA error: {type(exc).__name__}: {exc}", libraries=libraries)
    return TFMRuntime(True, None, int(free), int(total), libraries)


def _release_tfm(model: Any) -> None:
    """Free the fitted bags (GPU memory) of a TFM predictor; never raises."""
    try:
        release = getattr(model, "release", None)
        if callable(release):
            release()
    except Exception as exc:  # noqa: BLE001 — cleanup must never mask the original problem
        log.debug("TFM release failed: %s", exc)

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
        # TFM serving: why each weighted TFM is (not) served, the recipe from ensemble_weights.json and
        # its re-fit without TFMs (ensemble_weights_no_tfm.json) used whenever a weighted TFM is missing
        self.tfm_status: dict[str, str] = {}
        self.ensemble_full: EnsembleConfig | None = None
        self.ensemble_no_tfm: EnsembleConfig | None = None

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
        self.ensemble_full = self.ensemble
        self.tfm_status = {}
        self.ensemble_no_tfm = None
        if self.ensemble.tfm_members:
            self._load_tfm_models(deploy_dirs, run_candidates)
            self._apply_tfm_fallback(models_dir)
        active = self.ensemble.active(self.loaded_ids)
        if self.models and not active:
            log.error("No loaded model has a weight > 0 in %s — /api/v1/predict will fail",
                      self.ensemble.source)
        missing = [m for m, w in self.ensemble.weights.items() if w > 0 and m not in self.models]
        if missing and not self.ensemble.is_default:
            log.warning("Weighted models not loaded (weights renormalised over the rest): %s", missing)
        log.info("Ensemble: %s (method=%s, temperature=%s, rule=%s)", active,
                 self.ensemble.method, self.ensemble.temperature, self.ensemble.rule)

    # ── Foundation models (opt-in) ───────────────────────────────────

    def _tfm_unavailable(self, models: list[str], reason: str) -> None:
        for m in models:
            self.tfm_status[m] = f"not served: {reason}"
            log.warning("  %-15s ✗ not served — %s", m, reason)

    def _load_tfm_models(self, deploy_dirs: list[Path], run_candidates: dict[str, list[_Candidate]]) -> None:
        """Weighted TFMs: opt-in → CUDA → artifacts → VRAM budget → warm-up (all or none are served)."""
        wanted = self.ensemble.tfm_members
        if not tfm_serving_enabled():
            for m in wanted:
                self.tfm_status[m] = f"disabled ({TFM_SERVING_ENV} is not 1)"
            log.info("TFM members %s in %s are not served (opt-in: %s=1, CUDA required)", wanted,
                     self.ensemble.source, TFM_SERVING_ENV)
            return
        runtime = tfm_runtime_status(wanted)
        if not runtime.cuda:
            self._tfm_unavailable(wanted, runtime.reason or "CUDA not available")
            return
        missing_libs = [m for m in wanted if not runtime.libraries.get(m)]
        if missing_libs:
            self._tfm_unavailable(missing_libs, "library not installed (training/requirements-tfm.txt)")
        ensure_training_on_path()             # unpickling needs src.models.tfm_base.TFMPredictor
        loaded: dict[str, tuple[dict[str, Any], _Candidate]] = {}
        for model_id in [m for m in wanted if m not in missing_libs]:
            for cand in self._file_candidates(model_id, deploy_dirs) + run_candidates.get(model_id, []):
                bundle = self._load_artifact(model_id, cand.path, cand.source)
                if bundle is None:
                    continue
                if not callable(getattr(bundle["model"], "warm_up", None)):
                    log.warning("  %-15s ✗ %s is not a foundation-model predictor — skipping", model_id, cand.path)
                    continue
                bundle["model"].keep_fitted_bags = "all"
                loaded[model_id] = (bundle, cand)
                break
            else:
                self._tfm_unavailable([model_id], "no usable artifact (training/runs or backend/models)")
        if len(loaded) < len(wanted):         # the full recipe cannot be served → serve none
            if loaded:
                self._tfm_unavailable(list(loaded), "another weighted TFM is not available — the full recipe "
                                                    "cannot be served")
            for bundle, _ in loaded.values():
                _release_tfm(bundle["model"])
            return

        sizes: dict[str, int | None] = {}
        for model_id, (bundle, _) in loaded.items():
            try:
                sizes[model_id] = bundle["model"].estimated_cache_bytes()
            except Exception as exc:  # noqa: BLE001
                log.warning("  %-15s ! cache size unknown (%s)", model_id, exc)
                sizes[model_id] = None
        known = sum(v for v in sizes.values() if v is not None)
        required = known + tfm_reserve_bytes()
        free = runtime.free_vram_bytes
        log.info("TFM VRAM budget: caches %s MiB + reserve %d MiB vs free %s MiB",
                 {m: None if v is None else round(v / _MB) for m, v in sizes.items()},
                 tfm_reserve_bytes() // _MB, None if free is None else free // _MB)
        if free is not None and required > free:
            self._tfm_unavailable(list(loaded), f"VRAM budget exceeded: {known / _MB:,.0f} MiB of caches + "
                                                f"{tfm_reserve_bytes() / _MB:,.0f} MiB reserve > "
                                                f"{free / _MB:,.0f} MiB free (stop other GPU users, e.g. "
                                                f"`ollama stop <model>`, or serve a smaller TFM recipe)")
            for bundle, _ in loaded.values():
                _release_tfm(bundle["model"])
            return
        for model_id, (bundle, _) in loaded.items():
            try:
                bundle["model"].warm_up()
            except Exception as exc:  # noqa: BLE001 — OOM / CUDA / missing weights → fallback, never a crash
                self._tfm_unavailable(list(loaded), f"warm-up of {model_id} failed: {type(exc).__name__}: {exc}")
                for other, _ in loaded.values():
                    _release_tfm(other["model"])
                return
        for model_id, (bundle, cand) in loaded.items():
            bundle.update({"train_sample": cand.train_sample, "tuned": cand.tuned, "pretrained": True})
            self.models[model_id] = bundle
            self.tfm_status[model_id] = "served"
            log.info("  %-15s ✓ loaded and warmed up (%d bags, %d features) from %s [%s]", model_id,
                     getattr(bundle["model"], "n_bags", 0), len(bundle["feature_names"]), cand.path, cand.source)
            if cand.train_sample:
                log.warning("  %-15s ! trained on a sample of the train split (smoke/quick run) — not for "
                            "clinical evaluation", model_id)

    def _apply_tfm_fallback(self, models_dir: Path) -> None:
        """A weighted TFM not served → ensemble_weights_no_tfm.json (WARNING) or renormalisation (WARNING)."""
        path = models_dir / ENSEMBLE_WEIGHTS_NO_TFM_FILE
        if path.exists():
            config = load_ensemble_config(path)
            if config.tfm_members:
                raise ValueError(f"{path}: the no-TFM recipe gives a weight > 0 to {config.tfm_members}")
            self.ensemble_no_tfm = config
        missing = [m for m in self.ensemble.tfm_members if m not in self.models]
        if not missing:
            return
        reasons = "; ".join(f"{m}: {self.tfm_status.get(m, 'not loaded')}" for m in missing)
        if self.ensemble_no_tfm is not None:
            log.warning("Ensemble members %s are not served (%s) — using %s (the same recipe re-fitted on VAL "
                        "without foundation models)", missing, reasons, path)
            self.ensemble = self.ensemble_no_tfm
        else:
            log.warning("Ensemble members %s are not served (%s) and there is no %s next to %s — weights are "
                        "renormalised over the remaining models, but the temperature and the decision-rule "
                        "parameters were fitted for the FULL blend (calibration may be off); deploy "
                        "backend_ensemble_weights_no_tfm.json from the 08 report as %s", missing, reasons,
                        ENSEMBLE_WEIGHTS_NO_TFM_FILE, ENSEMBLE_WEIGHTS_FILE, path)

    def drop_tfm(self, model_id: str, reason: str) -> bool:
        """A served TFM failed during a request: release it and apply the no-TFM fallback.
        Returns True when the active ensemble configuration changed."""
        bundle = self.models.pop(model_id, None)
        if bundle is None:
            return False
        _release_tfm(bundle.get("model"))
        self.tfm_status[model_id] = f"dropped after a failed request: {reason}"
        log.warning("  %-15s ✗ failed during a request (%s) — no longer served", model_id, reason)
        before = self.ensemble
        if self.ensemble_no_tfm is not None and self.ensemble.tfm_members:
            log.warning("Switching the ensemble to %s (recipe without foundation models)", self.ensemble_no_tfm.source)
            self.ensemble = self.ensemble_no_tfm
        elif self.ensemble.weight(model_id) > 0:
            log.warning("No %s — weights renormalised over the remaining models; temperature and rule "
                        "parameters were fitted for the full blend", ENSEMBLE_WEIGHTS_NO_TFM_FILE)
        return self.ensemble is not before

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
