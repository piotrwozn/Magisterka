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
      "rule": "target",
      "cutpoints": [1.03, 1.95, 2.35, 3.02],
      "safety_q": 0.13
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
    rule        : decision rule, exactly training `src.evaluation.decision.decide_rule`:
                  "argmax"       — most probable class (default);
                  "cost"         — Bayes decision with the training cost matrix:
                                   undertriage costs 3× more than overtriage,
                                   cost[t, p] = (p − t)² × 3 if p > t else (p − t)²;
                  "target"       — expected level EL = Σ k·p_k (k = 0…4) cut by the 4
                                   "cutpoints" (class = number of cutpoints strictly
                                   exceeded; the ESI2/ESI3 cutpoint was fitted on val to an
                                   undertriage target); optional safety net "safety_q":
                                   P(ESI1) + P(ESI2) ≥ q → at most ESI 3;
                  "bayes_target" — Bayes decision with cost (p − t)² + "lam" ×
                                   1[t ∈ ESI 1–2 and p ∈ ESI 3–5], λ fitted on val to the
                                   undertriage target.
                  The rule is applied to the blended probabilities and to every
                  per-model prediction (its parameters were fitted on the blend).
    cutpoints   : 4 non-decreasing finite numbers — required for "target", not
                  allowed for other rules.
    safety_q    : number in (0, 1] or null — optional, only for "target".
    lam         : number > 0 — required for "bayes_target", not allowed otherwise.

An invalid file (also an unknown key, e.g. a typo "temprature") raises ValueError
at startup — a mistake in the ensemble must be loud, not silently replaced by
defaults.

Tabular foundation models (TFM_MODELS: "tabpfn", "tabicl") may appear in "weights"
(a recipe from training `scripts/15_tfm_blend.py` → `08_final_report.py`), but they
never join the default equal-weight blend and the backend serves them only on
opt-in (`SOR_AI_BACKEND_TFM=1`, CUDA required — app/models/registry.py). For that
case 08 also writes the same recipe re-fitted WITHOUT the TFMs; deploy it next to
this file as `ensemble_weights_no_tfm.json` (ENSEMBLE_WEIGHTS_NO_TFM_FILE): the
registry switches to it (WARNING) whenever a weighted TFM is not served.
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

# Classical models served by the backend (training/src/models MODEL_REGISTRY, config.CLASSICAL_MODELS)
ENSEMBLE_MODELS: tuple[str, ...] = (
    "xgboost", "lightgbm", "catboost", "random_forest", "ebm", "logreg",
)
# Pretrained tabular foundation models (training config.TFM_MODELS) — opt-in serving only, never in defaults
TFM_MODELS: tuple[str, ...] = ("tabpfn", "tabicl")
# Model names accepted in ensemble_weights.json
KNOWN_MODELS: tuple[str, ...] = ENSEMBLE_MODELS + TFM_MODELS
# Default weights: equal over the CLASSICAL models (no stale test-set scores baked into the code)
MODEL_WEIGHTS: dict[str, float] = dict.fromkeys(ENSEMBLE_MODELS, 1.0)

ENSEMBLE_WEIGHTS_FILE = "ensemble_weights.json"
# The same recipe re-fitted on VAL without TFM members (training 08_final_report.py)
ENSEMBLE_WEIGHTS_NO_TFM_FILE = "ensemble_weights_no_tfm.json"
DEFAULT_SOURCE = "default (equal weights)"
DECISION_RULES: tuple[str, ...] = ("argmax", "cost", "target", "bayes_target")
# Parameters each rule accepts (training src/evaluation/decision.py RULE_PARAMS)
RULE_PARAMS: dict[str, frozenset[str]] = {
    "argmax": frozenset(),
    "cost": frozenset(),
    "target": frozenset({"cutpoints", "safety_q"}),
    "bayes_target": frozenset({"lam"}),
}
RULE_KEYS: frozenset[str] = frozenset({"rule", "cutpoints", "safety_q", "lam"})
BLEND_METHODS: tuple[str, ...] = ("mean", "geometric")
CONFIG_KEYS: frozenset[str] = frozenset({"weights", "method", "temperature"}) | RULE_KEYS
N_CUTPOINTS = 4
URGENT_MAX = 1          # index of ESI 2: predictions > 1 (ESI 3–5) of an ESI 1–2 patient = undertriage
SAFETY_NET_CAP = 2      # safety net: at most ESI 3 (index 2)

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


def bayes_cost_matrix(lam: float, n_classes: int = 5) -> np.ndarray:
    """cost[t, p] = (p − t)² + λ·1[t ∈ ESI 1–2 and p ∈ ESI 3–5] — as training
    `src.evaluation.decision.bayes_cost_matrix`."""
    t = np.arange(n_classes)[:, None]
    p = np.arange(n_classes)[None, :]
    return (p - t).astype(float) ** 2 + float(lam) * ((t <= URGENT_MAX) & (p > URGENT_MAX))


def expected_level(proba: np.ndarray) -> float:
    """EL = Σ k·p_k (k = 0…4), summed column by column exactly like training
    `src.evaluation.decision.expected_level` (bit-identical, no BLAS)."""
    el = 0.0
    for k in range(1, len(proba)):
        el = el + k * proba[k]
    return float(el)


def _number(value: Any, name: str, where: str) -> float:
    """A finite number (bools rejected) or ValueError."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{where}: \"{name}\" must be a number (got {value!r})")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: \"{name}\" must be a number (got {value!r})") from None
    if not math.isfinite(out):
        raise ValueError(f"{where}: \"{name}\" must be finite (got {value!r})")
    return out


class DecisionRule(str):
    """Decision rule name WITH its parameters.

    A `str` subclass so that `config.rule == "cost"`, logging and existing callers
    (`decide(proba, config.rule)`) keep working; the parameters ride along as
    attributes. Two rules are equal only if names and parameters match.
    """

    cutpoints: tuple[float, ...] | None
    safety_q: float | None
    lam: float | None

    def __new__(cls, name: str = "argmax", *, cutpoints: Any = None, safety_q: Any = None,
                lam: Any = None) -> DecisionRule:
        spec: dict[str, Any] = {"rule": str(name)}
        if cutpoints is not None:
            spec["cutpoints"] = cutpoints
        if safety_q is not None:
            spec["safety_q"] = safety_q
        if lam is not None:
            spec["lam"] = lam
        return cls.from_spec(spec)

    @classmethod
    def from_spec(cls, spec: dict[str, Any], where: str = "decision rule") -> DecisionRule:
        """{"rule": ..., "cutpoints": ..., "safety_q": ..., "lam": ...} → validated rule
        (same checks as training `src.evaluation.decision.normalize_spec`)."""
        if not isinstance(spec, dict):
            raise ValueError(f"{where}: expected an object {{\"rule\": ...}} (got {spec!r})")
        unknown = sorted(set(spec) - RULE_KEYS)
        if unknown:
            raise ValueError(f"{where}: unknown rule key(s) {unknown} (allowed: {sorted(RULE_KEYS)})")
        name = spec.get("rule", "argmax")
        if name not in DECISION_RULES:
            raise ValueError(f"{where}: \"rule\" must be one of {DECISION_RULES} (got {name!r})")
        foreign = sorted(set(spec) - {"rule"} - RULE_PARAMS[name])
        if foreign:
            raise ValueError(f"{where}: rule {name!r} does not take {foreign}")
        cutpoints = safety_q = lam = None
        if name == "target":
            if "cutpoints" not in spec:
                raise ValueError(f"{where}: rule \"target\" needs \"cutpoints\" (4 numbers)")
            raw = spec["cutpoints"]
            if not isinstance(raw, (list, tuple)) or len(raw) != N_CUTPOINTS:
                raise ValueError(f"{where}: \"cutpoints\" must be a list of {N_CUTPOINTS} numbers (got {raw!r})")
            cutpoints = tuple(_number(c, "cutpoints", where) for c in raw)
            if any(b < a for a, b in zip(cutpoints, cutpoints[1:])):
                raise ValueError(f"{where}: \"cutpoints\" must be sorted ascending (got {raw!r})")
            if spec.get("safety_q") is not None:
                safety_q = _number(spec["safety_q"], "safety_q", where)
                if not 0.0 < safety_q <= 1.0:
                    raise ValueError(f"{where}: \"safety_q\" must be in (0, 1] (got {spec['safety_q']!r})")
        elif name == "bayes_target":
            if "lam" not in spec:
                raise ValueError(f"{where}: rule \"bayes_target\" needs \"lam\" (a number > 0)")
            lam = _number(spec["lam"], "lam", where)
            if lam <= 0:
                raise ValueError(f"{where}: \"lam\" must be > 0 (got {spec['lam']!r})")
        obj = super().__new__(cls, name)
        obj.cutpoints, obj.safety_q, obj.lam = cutpoints, safety_q, lam
        return obj

    def __reduce__(self):
        return (_rule_from_spec, (self.spec(),))

    def spec(self) -> dict[str, Any]:
        """JSON spec (the keys of ensemble_weights.json that describe the rule)."""
        out: dict[str, Any] = {"rule": str(self)}
        if str(self) == "target":
            out["cutpoints"] = list(self.cutpoints or ())
            out["safety_q"] = self.safety_q
        elif str(self) == "bayes_target":
            out["lam"] = self.lam
        return out

    def _key(self) -> tuple:
        return (str(self), self.cutpoints, self.safety_q, self.lam)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DecisionRule):
            return self._key() == other._key()
        return str.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    __hash__ = str.__hash__

    def __repr__(self) -> str:
        params = {k: v for k, v in self.spec().items() if k != "rule"}
        return f"DecisionRule({str(self)!r}{', ' if params else ''}" + \
            ", ".join(f"{k}={v!r}" for k, v in params.items()) + ")"

    def describe(self) -> str:
        """Short text for logs: name + parameters."""
        params = {k: v for k, v in self.spec().items() if k != "rule"}
        return str(self) if not params else f"{self} {params}"


def _rule_from_spec(spec: dict[str, Any]) -> DecisionRule:
    return DecisionRule.from_spec(spec)


def as_rule(rule: str | dict[str, Any] | DecisionRule) -> DecisionRule:
    """str / spec dict / DecisionRule → validated DecisionRule ("target" and
    "bayes_target" need their parameters, so a bare name raises ValueError)."""
    if isinstance(rule, DecisionRule):
        return rule
    if isinstance(rule, dict):
        return DecisionRule.from_spec(rule)
    if isinstance(rule, str):
        return DecisionRule.from_spec({"rule": rule})
    raise ValueError(f"Unknown decision rule {rule!r} (expected one of {DECISION_RULES})")


def decide(proba: np.ndarray, rule: str | dict[str, Any] | DecisionRule = "argmax") -> int:
    """Class index (0–4 = ESI 1–5) for ONE probability vector — the same decision as
    training `src.evaluation.decision.decide_rule` for that row."""
    proba = np.asarray(proba, dtype=float)
    rule = as_rule(rule)
    if rule == "argmax":
        return int(np.argmax(proba))
    if rule == "cost":
        return int(np.argmin(proba @ cost_matrix(len(proba))))
    if rule == "bayes_target":
        return int(np.argmin(proba @ bayes_cost_matrix(rule.lam, len(proba))))
    # "target": number of cutpoints strictly exceeded by EL, then the optional safety net
    el = expected_level(proba)
    category = int(sum(el > c for c in rule.cutpoints))
    if rule.safety_q is not None and proba[0] + proba[1] >= rule.safety_q:
        category = min(category, SAFETY_NET_CAP)
    return category


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
    rule: DecisionRule | str = "argmax"
    source: str = DEFAULT_SOURCE
    method: str = "mean"
    temperature: float | None = None

    def __post_init__(self) -> None:
        # Always a validated DecisionRule (a str): parameters travel with config.rule
        object.__setattr__(self, "rule", as_rule(self.rule))

    @property
    def is_default(self) -> bool:
        return self.source == DEFAULT_SOURCE

    def weight(self, model_id: str) -> float:
        return float(self.weights.get(model_id, 0.0))

    def active(self, loaded_ids: list[str]) -> list[str]:
        """Loaded models that take part in the ensemble (weight > 0), in the given order."""
        return [m for m in loaded_ids if self.weight(m) > 0]

    @property
    def tfm_members(self) -> list[str]:
        """Foundation models (TFM_MODELS) with weight > 0 in this configuration."""
        return [m for m in TFM_MODELS if self.weight(m) > 0]

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
        if name not in KNOWN_MODELS:
            log.warning("%s: unknown model %r ignored (known: %s)", path, name, ", ".join(KNOWN_MODELS))
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
    rule = DecisionRule.from_spec({k: v for k, v in payload.items() if k in RULE_KEYS}, where=str(path))
    method = payload.get("method", "mean")
    if method not in BLEND_METHODS:
        raise ValueError(f"{path}: \"method\" must be one of {BLEND_METHODS} (got {method!r})")
    temperature = _parse_temperature(payload.get("temperature"), path)

    config = EnsembleConfig(weights=weights, rule=rule, source=str(path),
                            method=method, temperature=temperature)
    log.info("Ensemble from %s: method=%s, temperature=%s, rule=%s, weights=%s", path, method,
             temperature, rule.describe(), {m: round(w, 4) for m, w in weights.items()})
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
