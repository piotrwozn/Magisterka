"""Ensemble configuration — model weights and the decision rule (Layer 1A).

Defaults: every model in ENSEMBLE_MODELS gets weight 1.0 (plain average of the
probabilities of the loaded models) and the decision rule is "argmax".

Optional override — `backend/models/ensemble_weights.json` (next to the deployed
models; `<models_dir>/ensemble_weights.json`). Paste the blend chosen on the
VALIDATION split (training: `src.ensemble.optimize_weights` / `fit_temperature` /
manual blend):

    {
      "weights": {"xgboost": 0.35, "lightgbm": 0.25, "catboost": 0.40},
      "method": "mean",
      "temperature": 1.12,
      "rule": "cost"
    }

    weights     : model → weight ≥ 0 (normalised to sum 1 over the loaded models).
                  Models NOT listed get weight 0 and are not used at all (same as
                  training `src.ensemble.blend`). Omit "weights" → equal weights.
    method      : "mean" (weighted arithmetic mean, default) or "geometric"
                  (weighted mean of log-probabilities, renormalised) — exactly
                  training `src.ensemble.blend(..., method=...)`.
    temperature : optional float > 0 — temperature scaling of the BLENDED
                  probabilities (p ∝ exp(log p / T), training
                  `src.ensemble.apply_temperature`), fitted on val with
                  `fit_temperature(blend(P_val, ...), y_val)`. Applied before the
                  decision rule; per-model predictions are not rescaled.
    rule        : "argmax" (most probable class, default) or "cost" (Bayes decision
                  with the training cost matrix: undertriage costs 3× more than
                  overtriage, cost[t, p] = (p − t)² × 3 if p > t else (p − t)²).
                  The rule is applied to the blended probabilities and to every
                  per-model prediction.

An invalid file (also an unknown key, e.g. a typo "temprature") raises ValueError
at startup — a mistake in the ensemble must be loud, not silently replaced by
defaults.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Models served by the backend (training/src/models MODEL_REGISTRY)
ENSEMBLE_MODELS: tuple[str, ...] = (
    "xgboost", "lightgbm", "catboost", "random_forest", "ebm", "logreg",
)
# Default weights: equal (no stale test-set scores baked into the code)
MODEL_WEIGHTS: dict[str, float] = dict.fromkeys(ENSEMBLE_MODELS, 1.0)

ENSEMBLE_WEIGHTS_FILE = "ensemble_weights.json"
DEFAULT_SOURCE = "default (equal weights)"
DECISION_RULES: tuple[str, ...] = ("argmax", "cost")
BLEND_METHODS: tuple[str, ...] = ("mean", "geometric")
CONFIG_KEYS: frozenset[str] = frozenset({"weights", "method", "temperature", "rule"})

# Same as training/src/ensemble/runs.py (_EPS): clip before log
_EPS = 1e-12

# Same as training/src/utils/config.py (UNDERTRIAGE_/OVERTRIAGE_COST_MULTIPLIER)
UNDERTRIAGE_COST_MULTIPLIER: float = 3.0
OVERTRIAGE_COST_MULTIPLIER: float = 1.0


def cost_matrix(n_classes: int = 5) -> np.ndarray:
    """cost[t, p] = (p − t)² × (3 if p > t else 1) — as training `src.evaluation.metrics.cost_matrix`."""
    t = np.arange(n_classes)[:, None]
    p = np.arange(n_classes)[None, :]
    dist2 = (p - t).astype(float) ** 2
    return np.where(p > t, UNDERTRIAGE_COST_MULTIPLIER * dist2, OVERTRIAGE_COST_MULTIPLIER * dist2)


def decide(proba: np.ndarray, rule: str = "argmax") -> int:
    """Class index (0–4 = ESI 1–5) for ONE probability vector."""
    proba = np.asarray(proba, dtype=float)
    if rule == "argmax":
        return int(np.argmax(proba))
    if rule == "cost":
        return int(np.argmin(proba @ cost_matrix(len(proba))))
    raise ValueError(f"Unknown decision rule {rule!r} (expected one of {DECISION_RULES})")


def blend(probas: list[np.ndarray], weights: list[float], method: str = "mean") -> np.ndarray:
    """Weighted blend of probability vectors (weights normalised to sum 1) — as training
    `src.ensemble.blend`: "mean" = arithmetic, "geometric" = exp(Σ w·log p), renormalised."""
    w = np.asarray(weights, dtype=float)
    if len(probas) != len(w) or not len(w) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("blend needs one non-negative weight per model and a positive sum")
    w = w / w.sum()
    stack = np.stack([np.asarray(p, dtype=float) for p in probas])
    if method == "mean":
        out = np.tensordot(w, stack, axes=1)
    elif method == "geometric":
        out = np.exp(np.tensordot(w, np.log(np.clip(stack, _EPS, 1.0)), axes=1))
    else:
        raise ValueError(f"Unknown blend method {method!r} (expected one of {BLEND_METHODS})")
    return out / out.sum(axis=-1, keepdims=True)


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature scaling p ∝ exp(log p / T) — as training `src.ensemble.apply_temperature`."""
    z = np.log(np.clip(np.asarray(proba, dtype=float), _EPS, 1.0)) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    out = np.exp(z)
    return out / out.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class EnsembleConfig:
    """Weights per model + blend method + optional temperature + decision rule."""

    weights: dict[str, float] = field(default_factory=lambda: dict(MODEL_WEIGHTS))
    rule: str = "argmax"
    source: str = DEFAULT_SOURCE
    method: str = "mean"
    temperature: float | None = None

    @property
    def is_default(self) -> bool:
        return self.source == DEFAULT_SOURCE

    def weight(self, model_id: str) -> float:
        return float(self.weights.get(model_id, 0.0))

    def active(self, loaded_ids: list[str]) -> list[str]:
        """Loaded models that take part in the ensemble (weight > 0), in the given order."""
        return [m for m in loaded_ids if self.weight(m) > 0]

    def combine(self, probas: dict[str, np.ndarray]) -> np.ndarray:
        """Per-model probabilities → blended (method, weights renormalised over the given
        models) → temperature-scaled when configured. The decision rule comes next."""
        names = list(probas)
        out = blend([probas[m] for m in names], [self.weight(m) for m in names], self.method)
        if self.temperature is not None:
            out = apply_temperature(out, self.temperature)
        return out


def _parse_weights(raw: Any, path: Path) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: \"weights\" must be a non-empty object {{model: weight}}")
    weights: dict[str, float] = {}
    for name, value in raw.items():
        try:
            w = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{path}: weight of {name!r} is not a number: {value!r}") from None
        if math.isnan(w) or math.isinf(w) or w < 0:
            raise ValueError(f"{path}: weight of {name!r} must be a finite number ≥ 0 (got {value!r})")
        if name not in ENSEMBLE_MODELS:
            log.warning("%s: unknown model %r ignored (known: %s)", path, name, ", ".join(ENSEMBLE_MODELS))
            continue
        weights[name] = w
    if sum(weights.values()) <= 0:
        raise ValueError(f"{path}: at least one known model needs a weight > 0")
    return weights


def load_ensemble_config(path: Path) -> EnsembleConfig:
    """ensemble_weights.json → EnsembleConfig; no file → defaults (equal weights, argmax)."""
    path = Path(path)
    if not path.exists():
        log.info("No %s — equal model weights, decision rule 'argmax'", path.name)
        return EnsembleConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected an object {{\"weights\": {{...}}, \"rule\": ...}}")
    unknown = sorted(set(payload) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown} (allowed: {sorted(CONFIG_KEYS)})")

    weights = _parse_weights(payload["weights"], path) if "weights" in payload else dict(MODEL_WEIGHTS)
    rule = payload.get("rule", "argmax")
    if rule not in DECISION_RULES:
        raise ValueError(f"{path}: \"rule\" must be one of {DECISION_RULES} (got {rule!r})")
    method = payload.get("method", "mean")
    if method not in BLEND_METHODS:
        raise ValueError(f"{path}: \"method\" must be one of {BLEND_METHODS} (got {method!r})")
    temperature = _parse_temperature(payload.get("temperature"), path)

    config = EnsembleConfig(weights=weights, rule=rule, source=str(path),
                            method=method, temperature=temperature)
    log.info("Ensemble from %s: method=%s, temperature=%s, rule=%s, weights=%s", path, method,
             temperature, rule, {m: round(w, 4) for m, w in weights.items()})
    return config


def _parse_temperature(raw: Any, path: Path) -> float | None:
    """null / missing → no temperature scaling; otherwise a finite float > 0."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(f"{path}: \"temperature\" must be a number > 0 (got {raw!r})")
    try:
        t = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{path}: \"temperature\" must be a number > 0 (got {raw!r})") from None
    if not math.isfinite(t) or t <= 0:
        raise ValueError(f"{path}: \"temperature\" must be a finite number > 0 (got {raw!r})")
    return t
