"""
Przypadki triażowe do porównania LLM (część "esi" i "tools" porównania modeli językowych).

Każdy przypadek = jeden wiersz VAL z kompletem HR / SBP / RR (vitals_complete == 1):
    - polski opis pacjenta (wiek, płeć, przybycie, parametry z jednostkami, skargi, tlen),
    - DETERMINISTYCZNA decyzja systemu z receptury (połączenie modeli + temperatura + reguła),
      prawdopodobieństwa ESI 1–5,
    - top cechy SHAP jednego modelu drzewiastego receptury (polskie nazwy),
    - flagi reguł bezpieczeństwa ESI (src/explain/esi_rules.py),
    - ESI z triażu (pielęgniarka) — UKRYTE przed LLM, tylko do metryk eskalacji.

Receptura:
    1. `--recipe <runs/final_report__*/recipe.json>` (albo katalog raportu) — dokładnie to,
       co policzył `scripts/08_final_report.py` (wagi, metoda, T, reguła główna);
    2. bez receptury: najnowsze runy po tuningu (`latest_runs`), wagi z minimalnego log loss
       na VAL, temperatura i reguła 'target' dopasowane na VAL (`report_sections.fit_rules`).

`RecipePredictor` liczy decyzję NA ŻYWO z modeli (model.joblib) na wierszu kanonicznym
(cechy inżynieryjne przeliczane przez src.features.engineering) — tak samo jak 08 na
zapisanych predykcjach: prawdopodobieństwa modeli zaokrąglane do float32 (format
predictions_*.parquet), normalizacja, `blend`, `apply_temperature`, `decide_rule`.
Decyzja przypadku jest liczona dla pojedynczego wiersza tą samą ścieżką co what_if (what_if bez
zmian = decyzja przypadku bit w bit), a zgodność z decyzją 08 na zapisanych predykcjach jest
sprawdzana i raportowana (`decision_check`, `live_vs_stored_mismatches`).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ensemble import (
    Run,
    aligned_meta,
    aligned_probas,
    apply_temperature,
    blend,
    fit_temperature,
    latest_runs,
    load_run,
    optimize_weights,
)
from src.evaluation.decision import decide_rule, expected_level, normalize_spec
from src.features.engineering import ENGINEERED_FEATURES, O2_DEVICE_COL, VITALS, engineer_features
from src.llm_eval.cc_labels_pl import (
    ARRIVAL_MODES_PL,
    CC_LABELS_PL,
    HOUR_BIN_LABELS_PL,
    MONTHS_PL,
    WEEKDAYS_PL,
    feature_label,
)
from src.utils import config
from src.utils.config import CLASS_NAMES, N_CLASSES, RANDOM_SEED
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# Stałe
# ─────────────────────────────────────────
VC_COL = "vitals_complete"
# Domyślny rozkład 40 przypadków na ESI 1–5 (ESI 1–2 nadreprezentowane)
DEFAULT_STRATA: dict[int, int] = {1: 8, 2: 12, 3: 10, 4: 6, 5: 4}
# Kolejność uzupełniania, gdy w warstwie brakuje wierszy (najpierw pilne)
REFILL_ORDER: tuple[int, ...] = (2, 1, 3, 4, 5)
SHAP_TOP_N: int = 5
SHAP_TOP_STORED: int = 10
# Preferencja modelu do SHAP (metoda wg src.explain.shap_explainer.detect_method): najpierw drzewiaste
# (TreeSHAP / natywne SHAP CatBoost), potem EBM, na końcu liniowy; w grupie — największa waga
SHAP_METHOD_RANK: dict[str, int] = {"tree": 0, "catboost": 0, "ebm": 1, "linear": 2}
ARRIVAL_COLS: dict[str, str] = {"ambulance": "arrival_ambulance", "self": "arrival_self", "other": "arrival_other"}
# Kolumny bazowe triażu, które zawsze trafiają do wiersza kanonicznego (cc_* — wszystkie z danych)
BASE_GROUPS: tuple[str, ...] = ("demographics", "arrival", "time", "triage_vitals", "chief_complaints")
VITAL_UNITS: dict[str, str] = {
    VITALS["hr"]: "/min", VITALS["sbp"]: " mmHg", VITALS["dbp"]: " mmHg", VITALS["rr"]: "/min",
    VITALS["o2"]: "%", VITALS["temp"]: " °C",
}
RULE_TEXT_PL: dict[str, str] = {
    "argmax": "najbardziej prawdopodobny poziom ESI",
    "cost": "decyzja kosztowa (niedotriażowanie droższe niż nadtriażowanie)",
    "target": ("progi oczekiwanego poziomu ESI dobrane do docelowego odsetka niedotriażowania "
               "(+ siatka bezpieczeństwa: wysokie P(ESI 1) + P(ESI 2) → co najmniej ESI 3)"),
    "bayes_target": "decyzja Bayesa z dodatkową karą za niedotriażowanie pacjentów ESI 1–2",
}


class CasesUnavailable(RuntimeError):
    """Brak wytrenowanych runów / receptury — część esi i tools zostaje pominięta."""


# ─────────────────────────────────────────
# Receptura
# ─────────────────────────────────────────
def _run_keys(runs: list[Run]) -> list[str]:
    """Klucze prawdopodobieństw jak w `aligned_probas`: model albo nazwa runa przy duplikatach."""
    return [r.model if sum(o.model == r.model for o in runs) == 1 else r.name for r in runs]


@dataclass
class Recipe:
    """Deterministyczna receptura decyzji: runy (kolejność łączenia), wagi, metoda, T, reguła."""

    runs: list[Run]
    weights: dict[str, float]
    method: str
    temperature: float | None
    spec: dict[str, Any]
    feature_set: str
    source: str
    data_fingerprint: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def keys(self) -> list[str]:
        return _run_keys(self.runs)

    def active_keys(self) -> list[str]:
        """Modele o dodatniej wadze (tylko one wpływają na decyzję)."""
        return [k for k in self.keys if float(self.weights.get(k, 0.0)) > 0]

    def describe_pl(self) -> str:
        """Jedno zdanie o recepturze (do kontekstu decyzji)."""
        total = sum(float(self.weights.get(k, 0.0)) for k in self.active_keys()) or 1.0
        parts = ", ".join(f"{k} {fmt_num(float(self.weights[k]) / total, 2)}" for k in self.active_keys())
        method = "średnia arytmetyczna" if self.method == "mean" else "średnia geometryczna"
        t_text = (f"kalibracja temperaturą T = {fmt_num(self.temperature, 3)}" if self.temperature is not None
                  else "bez kalibracji temperaturą")
        return (f"połączenie modeli ({method}; wagi: {parts}), {t_text}, reguła decyzyjna: "
                f"{RULE_TEXT_PL.get(self.spec['rule'], self.spec['rule'])}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "feature_set": self.feature_set, "method": self.method,
            "temperature": self.temperature, "spec": self.spec, "weights": dict(self.weights),
            "runs": [{"key": k, "name": r.name, "model": r.model, "path": str(r.path)}
                     for k, r in zip(self.keys, self.runs)],
            "data_fingerprint": self.data_fingerprint, "notes": list(self.notes),
        }


def _resolve_run_path(entry: dict[str, Any], runs_dir: Path | str | None) -> Run:
    """Run z recipe.json: najpierw zapisana ścieżka, potem runs/<nazwa>."""
    path = Path(str(entry.get("path") or ""))
    if entry.get("path") and (path / "run_info.json").exists():
        return load_run(path)
    return load_run(str(entry["name"]), runs_dir)


def load_recipe(path: Path | str, runs_dir: Path | str | None = None) -> Recipe:
    """
    Receptura z raportu końcowego 08: katalog `final_report__*` albo jego `recipe.json`.

    Reguła = `decision.main_spec` (reguła główna, ta sama co w pliku dla backendu); stare
    receptury bez sekcji `decision` → `backend_rule` (argmax / cost).
    """
    path = Path(path)
    if path.is_dir():
        path = path / "recipe.json"
    if not path.exists():
        raise CasesUnavailable(f"Brak pliku receptury {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        runs = [_resolve_run_path(entry, runs_dir) for entry in data["runs"]]
    except FileNotFoundError as exc:
        raise CasesUnavailable(f"Receptura {path} wskazuje run, którego nie ma: {exc}") from exc
    decision = data.get("decision") or {}
    spec = decision.get("main_spec") or {"rule": data.get("backend_rule") or data.get("main_rule") or "argmax"}
    weights = {str(k): float(v) for k, v in data["weights"].items()}
    missing = [k for k in _run_keys(runs) if k not in weights]
    if missing:
        raise CasesUnavailable(f"Receptura {path}: brak wag dla {missing}")
    temperature = data.get("temperature")
    return Recipe(
        runs=runs, weights=weights, method=str(data.get("method", "mean")),
        temperature=None if temperature is None else float(temperature),
        spec=normalize_spec(spec), feature_set=str(data.get("feature_set", "triage")),
        source=f"plik {path.as_posix()}", data_fingerprint=data.get("data_fingerprint"),
    )


def recipe_from_runs(
    feature_set: str = "triage",
    tag: str | None = None,
    runs_dir: Path | str | None = None,
    require_tuned: bool = True,
    target: float = config.DECISION_TARGET_UNDERTRIAGE,
) -> Recipe:
    """
    Receptura bez raportu 08: najnowsze ukończone runy (po tuningu), wagi min. log loss na VAL,
    temperatura i reguła 'target' dopasowane na VAL (populacja vc, jak w 08).

    Brak runów pełnych → próba z runami na próbkach (np. profil smoke) z ostrzeżeniem.
    Brak jakichkolwiek runów → CasesUnavailable.
    """
    from src.evaluation import report_sections as rs

    notes: list[str] = []
    runs = latest_runs(feature_set, tag=tag, runs_dir=runs_dir, require_tuned=require_tuned)
    if not runs:
        runs = latest_runs(feature_set, tag=tag, runs_dir=runs_dir, require_tuned=require_tuned,
                           include_samples=True)
        if runs:
            notes.append("użyto runów na próbkach danych (np. profil smoke) — wyłącznie do testów")
            log.warning("Receptura: brak pełnych runów — używam runów na próbkach (np. smoke)")
    if not runs:
        raise CasesUnavailable(
            f"Brak ukończonych runów{' po tuningu' if require_tuned else ''} zestawu '{feature_set}'"
            f"{f' z tagiem {tag!r}' if tag else ''} — wytrenuj modele (scripts/run_all.py) albo podaj --recipe")
    y, probas, vc = aligned_probas(runs, "val", return_vc=True)
    weights = optimize_weights(probas, y) if len(probas) > 1 else {k: 1.0 for k in probas}
    p_raw = blend(probas, weights, "mean")
    temperature = fit_temperature(p_raw, y)
    proba = apply_temperature(p_raw, temperature)
    try:
        fitted = rs.fit_rules(y, proba, vc, rs.DecisionOptions(main_rule="target", rules=["target"], target=target))
        spec = fitted.specs["target"]
    except ValueError as exc:
        log.warning(f"Receptura: nie da się dopasować reguły 'target' na VAL ({exc}) — używam reguły kosztowej")
        notes.append(f"reguła 'target' niedopasowana ({exc}) — reguła kosztowa")
        spec = {"rule": "cost"}
    return Recipe(
        runs=runs, weights={k: float(v) for k, v in weights.items()}, method="mean",
        temperature=float(temperature), spec=normalize_spec(spec), feature_set=feature_set,
        source="najnowsze runy (" + ", ".join(r.name for r in runs) + "); wagi, T i reguła dopasowane na VAL",
        data_fingerprint=runs[0].info.get("data_fingerprint"), notes=notes,
    )


def resolve_recipe(
    recipe_path: Path | str | None = None,
    runs_dir: Path | str | None = None,
    feature_set: str = "triage",
    tag: str | None = None,
) -> Recipe:
    """--recipe, jeśli podano; inaczej `recipe_from_runs`. Brak runów → CasesUnavailable."""
    if recipe_path:
        return load_recipe(recipe_path, runs_dir)
    return recipe_from_runs(feature_set=feature_set, tag=tag, runs_dir=runs_dir)


# ─────────────────────────────────────────
# Decyzja na żywo (what_if)
# ─────────────────────────────────────────
def as_stored_proba(proba: np.ndarray) -> np.ndarray:
    """Prawdopodobieństwa jak po zapisie do predictions_*.parquet (float32) i `aligned_probas` (normalizacja)."""
    p = np.asarray(proba, dtype=np.float32).astype(np.float64)
    return p / p.sum(axis=1, keepdims=True)


class RecipePredictor:
    """
    Receptura liczona na żywo z modeli: wiersz kanoniczny → cechy → prawdopodobieństwa → decyzja.

    Ładowane są tylko modele o dodatniej wadze (model o wadze 0 nie zmienia połączenia).
    """

    def __init__(self, recipe: Recipe):
        from src.models import load_model

        self.recipe = recipe
        self.models: dict[str, Any] = {}
        for key, run in zip(recipe.keys, recipe.runs):
            if float(recipe.weights.get(key, 0.0)) <= 0:
                continue
            path = run.path / "model.joblib"
            if not path.exists():
                raise CasesUnavailable(f"Run {run.name} nie ma pliku modelu {path} (trenowany z --no-save?)")
            self.models[key] = load_model(path)
        if not self.models:
            raise CasesUnavailable("Receptura nie ma żadnego modelu o dodatniej wadze")
        names: list[str] = []
        for model in self.models.values():
            names += [n for n in model.feature_names if n not in names]
        self.feature_names = names

    @property
    def weights(self) -> dict[str, float]:
        return {k: float(self.recipe.weights[k]) for k in self.models}

    def features(self, base: pd.DataFrame) -> pd.DataFrame:
        """Wiersze kanoniczne (bez cech inżynieryjnych) → cechy modeli (float32, jak `get_xy`)."""
        full = engineer_features(base)
        missing = [c for c in self.feature_names if c not in full.columns]
        if missing:
            raise KeyError(f"Wiersz kanoniczny nie ma cech wymaganych przez modele: {missing[:10]}")
        return full[self.feature_names].astype(np.float32)

    def model_probas(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {k: as_stored_proba(m.predict_proba(X[m.feature_names])) for k, m in self.models.items()}

    def combine(self, probas: dict[str, np.ndarray]) -> np.ndarray:
        """Połączenie + temperatura (dokładnie jak 08)."""
        p = blend(probas, self.weights, self.recipe.method)
        return apply_temperature(p, self.recipe.temperature) if self.recipe.temperature is not None else p

    def predict_features(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Cechy → (prawdopodobieństwa (n, 5), klasy 0–4)."""
        proba = self.combine(self.model_probas(X))
        return proba, decide_rule(proba, self.recipe.spec)

    def predict_base(self, base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Wiersze kanoniczne → (prawdopodobieństwa, klasy 0–4)."""
        return self.predict_features(self.features(base))

    @staticmethod
    def row_frame(row: dict[str, float | None]) -> pd.DataFrame:
        """Wiersz kanoniczny (słownik) → ramka 1 × kolumny, float32 (None = brak pomiaru → NaN)."""
        return pd.DataFrame([{k: (np.nan if v is None else v) for k, v in row.items()}]).astype(np.float32)

    def predict_row(self, row: dict[str, float | None]) -> tuple[np.ndarray, int, pd.DataFrame]:
        """
        Jeden pacjent, jak w backendzie i w what_if: (prawdopodobieństwa (5,), klasa 0–4, cechy X).

        Ta sama ścieżka obliczeń dla przypadku i dla what_if — wynik bez zmian jest identyczny bit
        w bit. (Wsadowe liczenie wielu wierszy może różnić się o 1 ulp float32 przez inną ścieżkę
        BLAS, dlatego przypadki liczymy zawsze pojedynczo.)
        """
        X = self.features(self.row_frame(row))
        proba, pred = self.predict_features(X)
        return proba[0], int(pred[0]), X


def stored_decisions(recipe: Recipe) -> pd.DataFrame:
    """
    Decyzje receptury na VAL z zapisanych predykcji (dokładnie jak 08): indeks row_id,
    kolumny meta (esi, vitals_complete, …), p_esi1..p_esi5 i `decision` (0–4).
    """
    _, probas, _ = aligned_probas(recipe.runs, "val", return_vc=True)
    meta = aligned_meta(recipe.runs, "val")
    weights = {k: float(recipe.weights.get(k, 0.0)) for k in probas}
    p = blend(probas, weights, recipe.method)
    if recipe.temperature is not None:
        p = apply_temperature(p, recipe.temperature)
    out = meta.copy()
    for i, name in enumerate(CLASS_NAMES):
        out[f"p_{name.lower()}"] = p[:, i]
    out["decision"] = decide_rule(p, recipe.spec)
    return out


# ─────────────────────────────────────────
# Formatowanie
# ─────────────────────────────────────────
def fmt_num(value: float | None, digits: int = 1) -> str:
    """Liczba z przecinkiem dziesiętnym (polski zapis); całkowite bez części ułamkowej."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "brak"
    value = float(value)
    if digits == 0 or abs(value - round(value)) < 1e-9 and digits <= 1:
        return str(int(round(value)))
    return f"{value:.{digits}f}".replace(".", ",")


def plural_pl(n: int, one: str, few: str, many: str) -> str:
    """Polska forma liczebnika: 1 przypadek, 2–4 przypadki (bez 12–14), 5+ przypadków."""
    n = int(n)
    if n == 1:
        return f"{n} {one}"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} {few}"
    return f"{n} {many}"


def years_pl(age: float) -> str:
    """Wiek z poprawną formą: 1 rok, 2–4 lata (bez 12–14), pozostałe lat."""
    n = int(round(float(age)))
    if n == 1:
        return "1 rok"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} lata"
    return f"{n} lat"


def _value(row: dict[str, Any], col: str) -> float | None:
    v = row.get(col)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def display_vital(col: str, value: float | None) -> float | None:
    """Wartość parametru tak, jak pokazujemy ją LLM (temperatura z 1 miejscem, reszta całkowite)."""
    if value is None:
        return None
    return round(value, 1) if col == VITALS["temp"] else float(round(value))


def format_feature_value(name: str, value: Any) -> str:
    """Wartość cechy po polsku (jednostki, tak/nie, etykiety pory przybycia)."""
    v = _value({"x": value}, "x")
    if v is None:
        return "brak pomiaru" if name in VITAL_UNITS or name == O2_DEVICE_COL else "brak"
    if name in VITAL_UNITS:
        shown = display_vital(name, v)
        return f"{fmt_num(shown, 1)}{VITAL_UNITS[name]}"
    if name == "age":
        return years_pl(v)
    if name == "is_male":
        return "mężczyzna" if v > 0 else "kobieta"
    if name == "arrival_hour_bin":
        return HOUR_BIN_LABELS_PL.get(int(v), fmt_num(v))
    if name == "arrival_weekday":
        return WEEKDAYS_PL.get(int(v), fmt_num(v))
    if name == "arrival_month":
        return MONTHS_PL.get(int(v), fmt_num(v))
    if name == O2_DEVICE_COL or name in CC_LABELS_PL or name.startswith("arrival_") or (
            name in ENGINEERED_FEATURES and v in (0.0, 1.0) and not name.endswith(("_score", "_count"))
            and name not in ("cc_count", "cc_critical") and not name.startswith("cc_group_")):
        return "tak" if v > 0 else "nie"
    return fmt_num(v, 2) if abs(v - round(v)) > 1e-9 else fmt_num(v, 0)


def arrival_mode(row: dict[str, Any]) -> str | None:
    for mode, col in ARRIVAL_COLS.items():
        v = _value(row, col)
        if v is not None and v > 0:
            return mode
    return None


def active_complaints(row: dict[str, Any]) -> list[str]:
    """Kolumny cc_* skarg obecnych u pacjenta (> 0), w kolejności słownika etykiet."""
    return [c for c in CC_LABELS_PL if (_value(row, c) or 0) > 0]


def describe_patient(row: dict[str, Any]) -> str:
    """Polski opis pacjenta z wiersza kanonicznego (tylko dane z triażu)."""
    age = _value(row, "age")
    sex = _value(row, "is_male")
    sex_text = "płeć nieznana" if sex is None else ("mężczyzna" if sex > 0 else "kobieta")
    lines = [f"Pacjent: {sex_text}, wiek {years_pl(age)}." if age is not None
             else f"Pacjent: {sex_text}, wiek nieznany."]
    mode = arrival_mode(row)
    when = []
    if _value(row, "arrival_hour_bin") is not None:
        when.append(f"godz. {HOUR_BIN_LABELS_PL.get(int(_value(row, 'arrival_hour_bin')), '?')}")
    if _value(row, "arrival_weekday") is not None:
        when.append(WEEKDAYS_PL.get(int(_value(row, "arrival_weekday")), "?"))
    if _value(row, "arrival_month") is not None:
        when.append(f"miesiąc: {MONTHS_PL.get(int(_value(row, 'arrival_month')), '?')}")
    lines.append(f"Przybycie: {ARRIVAL_MODES_PL.get(mode, 'brak danych')}"
                 + (f" ({', '.join(when)})." if when else "."))
    hr, sbp, dbp = (display_vital(VITALS[k], _value(row, VITALS[k])) for k in ("hr", "sbp", "dbp"))
    rr, o2, temp = (display_vital(VITALS[k], _value(row, VITALS[k])) for k in ("rr", "o2", "temp"))
    bp = ("brak pomiaru" if sbp is None else
          f"{fmt_num(sbp)}/{fmt_num(dbp)} mmHg" if dbp is not None else f"skurczowe {fmt_num(sbp)} mmHg "
                                                                         f"(rozkurczowe: brak pomiaru)")
    device = _value(row, O2_DEVICE_COL)
    device_text = "brak danych" if device is None else ("tak" if device > 0 else "nie (powietrze atmosferyczne)")
    lines.append(
        "Parametry życiowe przy triażu: "
        f"tętno {'brak pomiaru' if hr is None else fmt_num(hr) + '/min'}; "
        f"ciśnienie tętnicze {bp}; "
        f"częstość oddechów {'brak pomiaru' if rr is None else fmt_num(rr) + '/min'}; "
        f"SpO2 {'brak pomiaru' if o2 is None else fmt_num(o2) + '%'}; "
        f"temperatura {'brak pomiaru' if temp is None else fmt_num(temp, 1) + ' °C'}; "
        f"tlenoterapia przy pomiarze: {device_text}.")
    complaints = [CC_LABELS_PL[c] for c in active_complaints(row)]
    lines.append("Skargi główne: " + ("; ".join(complaints) if complaints else "brak zakodowanej skargi") + ".")
    return "\n".join(lines)


def vitals_for_check(row: dict[str, Any]) -> dict[str, float | None]:
    """Wartości pokazane LLM (do sprawdzania liczb w odpowiedzi): parametry + wiek."""
    out = {key: display_vital(col, _value(row, col)) for key, col in VITALS.items()}
    out["age"] = None if _value(row, "age") is None else float(round(_value(row, "age")))
    return out


# ─────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────
class ShapHelper:
    """SHAP jednego modelu receptury (preferowany drzewiasty o największej wadze)."""

    def __init__(self, predictor: RecipePredictor):
        from src.explain.shap_explainer import SHAPTriageExplainer, detect_method

        candidates = []
        for key, model in predictor.models.items():
            method = detect_method(model)
            if method in SHAP_METHOD_RANK:
                candidates.append((SHAP_METHOD_RANK[method], -predictor.weights[key], key))
        if not candidates:
            raise CasesUnavailable("Żaden model receptury nie obsługuje SHAP (drzewiasty / EBM / liniowy)")
        _, _, key = min(candidates)
        self.model_key = key
        self.model = predictor.models[key]
        self.explainer = SHAPTriageExplainer(self.model)
        self.explainer.fit()
        self.method = self.explainer.method

    def top_features(self, X: pd.DataFrame, class_idx: int, top_n: int = SHAP_TOP_STORED) -> list[dict[str, Any]]:
        """Top cechy (|SHAP| malejąco) dla klasy `class_idx` pierwszego wiersza X."""
        values, _ = self.explainer.explain_values(X.iloc[[0]])
        sv = values[0, int(class_idx)]
        names = self.explainer.output_feature_names
        frame = self.explainer.feature_frame(X.iloc[[0]])
        order = np.argsort(-np.abs(sv), kind="stable")[:top_n]
        out = []
        for j in order:
            name = str(names[j])
            raw = frame.iloc[0, j] if j < frame.shape[1] else np.nan
            out.append({
                "feature": name,
                "label_pl": feature_label(name),
                "value": None if pd.isna(raw) else float(raw),
                "value_pl": format_feature_value(name, raw) if " & " not in name else "interakcja",
                "shap": float(sv[j]),
                "direction": "za" if sv[j] > 0 else "przeciw",
            })
        return out


# ─────────────────────────────────────────
# Przypadek
# ─────────────────────────────────────────
@dataclass
class EsiCase:
    case_id: str
    row_id: int
    nurse_esi: int                         # ESI z triażu — NIE trafia do promptu
    decision_esi: int                      # decyzja receptury (1–5)
    probabilities: dict[str, float]        # "ESI1" … "ESI5"
    expected_esi: float                    # 1 + Σ k·p_k
    shap: list[dict[str, Any]]             # top cechy (do SHAP_TOP_STORED), polskie nazwy
    shap_model: str | None
    rules: dict[str, Any]                  # flagi reguł bezpieczeństwa ESI
    description_pl: str
    vitals: dict[str, float | None]        # wartości pokazane LLM (hr, sbp, dbp, rr, o2, temp, age)
    complaints: list[str]                  # kolumny cc_* obecne u pacjenta
    base_row: dict[str, float | None]      # wiersz kanoniczny (bez cech inżynieryjnych) — what_if
    decision_check: dict[str, Any] = field(default_factory=dict)

    @property
    def shap_top(self) -> list[dict[str, Any]]:
        return self.shap[:SHAP_TOP_N]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EsiCase:
        return cls(**data)

    # ──────── Teksty do promptów ────────
    def decision_text(self, recipe_text: str | None = None) -> str:
        probs = ", ".join(f"ESI {i + 1}: {fmt_num(100 * self.probabilities[n], 1)}%"
                          for i, n in enumerate(CLASS_NAMES))
        lines = [f"Decyzja systemu: ESI {self.decision_esi} ({config.CLASS_NAMES_PL[self.decision_esi - 1]}).",
                 f"Prawdopodobieństwa poziomów (po kalibracji): {probs}.",
                 f"Oczekiwany poziom ESI: {fmt_num(self.expected_esi, 2)}."]
        if recipe_text:
            lines.append(f"Sposób wyznaczenia: {recipe_text}.")
        return "\n".join(lines)

    def shap_text(self, top_n: int = SHAP_TOP_N) -> str:
        if not self.shap:
            return "Wyjaśnienie SHAP niedostępne."
        head = (f"Najważniejsze czynniki wg SHAP (model {self.shap_model}, wkład w wynik dla ESI "
                f"{self.decision_esi}; „za” = przemawia za tym poziomem, „przeciw” = przeciw):")
        rows = [f"{i + 1}. {f['label_pl']} = {f['value_pl']} ({f['direction']}, SHAP {fmt_num(f['shap'], 3)})"
                for i, f in enumerate(self.shap[:top_n])]
        return "\n".join([head, *rows])

    def rules_text(self) -> str:
        r = self.rules
        lines = []
        if r.get("suggested_esi") is None:
            lines.append("Reguły bezpieczeństwa ESI (parametry i skargi): brak wskazania ESI 1–2 "
                         "(poziom 3–5 zależy od przewidywanych zasobów).")
        else:
            prefix = "rozważ " if r.get("advisory") else ""
            lines.append(f"Reguły bezpieczeństwa ESI: {prefix}ESI {r['suggested_esi']} "
                         f"(punkt decyzyjny {r.get('decision_point')}).")
        for item in r.get("violations", []):
            lines.append(f"- {item}")
        if r.get("verdict"):
            lines.append(f"Porównanie z decyzją systemu: {r['verdict']}")
        return "\n".join(lines)

    def sources(self, recipe_text: str | None = None, with_shap: bool = True, with_rules: bool = True) -> dict[str, str]:
        """Źródła z identyfikatorami do cytowania: S1 pacjent, S2 decyzja, S3 SHAP, S4 reguły."""
        out = {"S1": self.description_pl, "S2": self.decision_text(recipe_text)}
        if with_shap:
            out["S3"] = self.shap_text()
        if with_rules:
            out["S4"] = self.rules_text()
        return out


def _rule_flags(row: dict[str, Any], decision_idx: int) -> dict[str, Any]:
    from src.explain.esi_rules import check_consistency, rule_based_triage

    result = rule_based_triage(row)
    consistency = check_consistency(decision_idx, result["suggested_class_idx"])
    return {
        "suggested_esi": result["suggested_esi"],
        "decision_point": result["decision_point"],
        "advisory": bool(result["advisory"]),
        "violations": [v["description"] + (f" (wartość {fmt_num(v['value'], 1)})" if v.get("type") == "vital" else "")
                       for v in result["all_violations"]],
        "verdict": consistency["verdict"].replace("⚠️ ", ""),
        "undertriage_flag": bool(consistency["undertriage"]),
    }


def _clean_row(series: pd.Series) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k, v in series.items():
        out[str(k)] = None if pd.isna(v) else float(v)
    return out


def allocate_strata(n_cases: int, available: dict[int, int],
                    base: dict[int, int] | None = None) -> dict[int, int]:
    """
    Liczba przypadków na poziom ESI: proporcje `base` (domyślnie 8/12/10/6/4) przeskalowane do
    n_cases (największe reszty), ograniczone dostępnością; niedobór rozdzielany po jednym przypadku
    na poziom, kolejno wg REFILL_ORDER (najpierw ESI 2), dopóki są wiersze.
    """
    base = base or DEFAULT_STRATA
    total = sum(base.values())
    raw = {k: n_cases * v / total for k, v in base.items()}
    alloc = {k: int(math.floor(x)) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - alloc[k]), k))[: n_cases - sum(alloc.values())]:
        alloc[k] += 1
    alloc = {k: min(v, available.get(k, 0)) for k, v in alloc.items()}
    deficit = n_cases - sum(alloc.values())
    while deficit > 0:
        progressed = False
        for k in REFILL_ORDER:
            if deficit > 0 and alloc.get(k, 0) < available.get(k, 0):
                alloc[k] = alloc.get(k, 0) + 1
                deficit -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def base_columns(predictor: RecipePredictor) -> list[str]:
    """Kolumny wiersza kanonicznego: bloki triażu (wszystkie skargi) + nie-inżynieryjne cechy modeli."""
    from src.data.dataset import feature_groups

    groups = feature_groups()
    cols: list[str] = []
    for g in BASE_GROUPS:
        cols += [c for c in groups.get(g, []) if c not in cols]
    engineered = set(ENGINEERED_FEATURES)
    cols += [c for c in predictor.feature_names if c not in engineered and c not in cols]
    return cols


def select_rows(decisions: pd.DataFrame, n_cases: int, seed: int,
                strata: dict[int, int] | None = None) -> list[int]:
    """row_id przypadków: tylko vitals_complete == 1, warstwy wg ESI pielęgniarki, potem tasowanie."""
    pool = decisions
    if VC_COL in pool.columns:
        pool = pool[pd.to_numeric(pool[VC_COL], errors="coerce").fillna(0) == 1]
    else:
        log.warning("Predykcje bez kolumny vitals_complete — losuję ze wszystkich wierszy VAL")
    rng = np.random.default_rng(seed)
    esi = pool["esi"].astype(int)
    available = esi.value_counts().to_dict()
    alloc = allocate_strata(n_cases, available, strata)
    chosen: list[int] = []
    for level in sorted(alloc):
        ids = np.sort(pool.index[esi == level].to_numpy())
        if alloc[level]:
            chosen += [int(i) for i in rng.choice(ids, size=alloc[level], replace=False)]
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    log.info("Przypadki ESI: " + ", ".join(f"ESI {k}: {v}" for k, v in sorted(alloc.items()))
             + f" (z {len(pool):,} wierszy VAL z kompletem HR/SBP/RR)")
    return chosen


def build_cases(
    recipe: Recipe,
    predictor: RecipePredictor | None = None,
    n_cases: int = 40,
    seed: int = RANDOM_SEED,
    strata: dict[int, int] | None = None,
    shap_helper: ShapHelper | None = None,
) -> tuple[list[EsiCase], RecipePredictor, ShapHelper | None]:
    """Przypadki z VAL (patrz docstring modułu). Zwraca (przypadki, predyktor, helper SHAP)."""
    from src.data.dataset import data_fingerprint, load_split

    predictor = predictor or RecipePredictor(recipe)
    current = data_fingerprint()
    if recipe.data_fingerprint and current != recipe.data_fingerprint:
        log.warning(f"Odcisk danych receptury ({recipe.data_fingerprint}) ≠ bieżący ({current}) — wiersze VAL "
                    f"mogą nie odpowiadać predykcjom runów")
    decisions = stored_decisions(recipe)
    row_ids = select_rows(decisions, n_cases, seed, strata)
    cols = base_columns(predictor)
    val = load_split("val", columns=["row_id", *cols])
    val = val[val["row_id"].isin(row_ids)].set_index("row_id").loc[row_ids]
    base = val[cols].astype(np.float32)
    if shap_helper is None:
        try:
            shap_helper = ShapHelper(predictor)
        except Exception as exc:                      # SHAP nie może zatrzymać całego porównania
            log.warning(f"SHAP niedostępny dla receptury: {exc}")
            shap_helper = None

    cases: list[EsiCase] = []
    proba_cols = [f"p_{n.lower()}" for n in CLASS_NAMES]
    for i, row_id in enumerate(row_ids):
        stored = decisions.loc[row_id]
        stored_idx = int(stored["decision"])
        row = _clean_row(base.loc[row_id])
        # Decyzja przypadku = receptura na tym jednym wierszu (jak backend i what_if);
        # zgodność z decyzją 08 na zapisanych predykcjach — w decision_check
        p, decision_idx, X_row = predictor.predict_row(row)
        live_diff = float(np.abs(p - stored[proba_cols].to_numpy(dtype=float)).max())
        if decision_idx != stored_idx:
            log.warning(f"Wiersz {row_id}: decyzja z modeli (ESI {decision_idx + 1}) ≠ z zapisanych predykcji "
                        f"(ESI {stored_idx + 1}); maks. różnica prawdopodobieństw {live_diff:.2e}")
        shap_items: list[dict[str, Any]] = []
        if shap_helper is not None:
            try:
                shap_items = shap_helper.top_features(X_row, decision_idx)
            except Exception as exc:
                log.warning(f"SHAP dla wiersza {row_id} nie powiódł się: {exc}")
        cases.append(EsiCase(
            case_id=f"C{i + 1:02d}",
            row_id=int(row_id),
            nurse_esi=int(stored["esi"]),
            decision_esi=decision_idx + 1,
            probabilities={n: float(p[k]) for k, n in enumerate(CLASS_NAMES)},
            expected_esi=float(1.0 + expected_level(p[None, :])[0]),
            shap=shap_items,
            shap_model=None if shap_helper is None else shap_helper.model_key,
            rules=_rule_flags(row, decision_idx),
            description_pl=describe_patient(row),
            vitals=vitals_for_check(row),
            complaints=active_complaints(row),
            base_row=row,
            decision_check={"stored_esi": stored_idx + 1, "live_esi": decision_idx + 1,
                            "max_abs_proba_diff": live_diff},
        ))
    return cases, predictor, shap_helper


# ─────────────────────────────────────────
# Kontekst części esi / tools
# ─────────────────────────────────────────
@dataclass
class EsiContext:
    """Wszystko, czego potrzebują części esi i tools (budowane raz, współdzielone)."""

    recipe: Recipe
    predictor: RecipePredictor
    cases: list[EsiCase]
    shap: ShapHelper | None

    @property
    def recipe_text(self) -> str:
        return self.recipe.describe_pl()

    def case_by_id(self, case_id: str) -> EsiCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)

    def summary(self) -> dict[str, Any]:
        mismatches = [c.case_id for c in self.cases
                      if c.decision_check.get("stored_esi") != c.decision_check.get("live_esi")]
        return {
            "recipe": self.recipe.to_dict(),
            "n_cases": len(self.cases),
            "nurse_esi_counts": {str(k): int(v) for k, v in
                                 pd.Series([c.nurse_esi for c in self.cases]).value_counts().sort_index().items()},
            "decision_esi_counts": {str(k): int(v) for k, v in
                                    pd.Series([c.decision_esi for c in self.cases]).value_counts().sort_index().items()},
            "shap_model": None if self.shap is None else self.shap.model_key,
            "shap_method": None if self.shap is None else self.shap.method,
            "live_vs_stored_mismatches": mismatches,
        }

    def save(self, path: Path | str) -> Path:
        from src.utils.fileio import write_json_atomic

        return write_json_atomic(path, {"summary": self.summary(), "cases": [c.to_dict() for c in self.cases]})


def prepare_context(
    recipe_path: Path | str | None = None,
    n_cases: int = 40,
    seed: int = RANDOM_SEED,
    runs_dir: Path | str | None = None,
    feature_set: str = "triage",
    tag: str | None = None,
    strata: dict[int, int] | None = None,
) -> EsiContext:
    """Receptura → predyktor → przypadki. Brak runów / receptury / danych → CasesUnavailable."""
    from src.data.dataset import is_prepared

    try:
        recipe = resolve_recipe(recipe_path, runs_dir=runs_dir, feature_set=feature_set, tag=tag)
    except (FileNotFoundError, ValueError) as exc:
        raise CasesUnavailable(f"Nie da się ustalić receptury decyzji: {exc}") from exc
    if not is_prepared():
        raise CasesUnavailable("Brak aktualnych splitów danych ESI (data/processed/esi) — przypadki budujemy z "
                               "wierszy VAL; uruchom python scripts/02_preprocess.py (albo trening run_all.py)")
    predictor = RecipePredictor(recipe)
    cases, predictor, shap = build_cases(recipe, predictor, n_cases=n_cases, seed=seed, strata=strata)
    if not cases:
        raise CasesUnavailable("Brak wierszy VAL z kompletem HR/SBP/RR w predykcjach runów receptury")
    return EsiContext(recipe=recipe, predictor=predictor, cases=cases, shap=shap)


# Kontekst zbudowany w tym procesie (część esi → część tools tego samego przebiegu: jedno
# wczytanie modeli i jedno losowanie przypadków). Klucz obejmuje recepturę, runs/ i odcisk danych.
_CONTEXT_MEMO: dict[tuple, EsiContext] = {}


def _memo_key(kwargs: dict[str, Any]) -> tuple:
    from src.data.dataset import data_fingerprint

    recipe = kwargs.get("recipe_path")
    runs_dir = kwargs.get("runs_dir") or config.RUNS_DIR
    strata = kwargs.get("strata")
    return (None if not recipe else str(Path(recipe).resolve()), str(Path(runs_dir).resolve()),
            int(kwargs.get("n_cases", 40)), int(kwargs.get("seed", RANDOM_SEED)),
            kwargs.get("feature_set", "triage"), kwargs.get("tag"),
            None if strata is None else tuple(sorted(strata.items())), data_fingerprint())


def try_prepare_context(**kwargs: Any) -> tuple[EsiContext | None, str | None]:
    """
    Jak `prepare_context`, ale brak runów → (None, powód) zamiast wyjątku (część pomijana).
    Ten sam zestaw argumentów w jednym procesie → ten sam kontekst (bez ponownego liczenia).
    """
    key = _memo_key(kwargs)
    if key in _CONTEXT_MEMO:
        return _CONTEXT_MEMO[key], None
    try:
        ctx = prepare_context(**kwargs)
    except CasesUnavailable as exc:
        log.warning(f"Części esi/tools pominięte: {exc}")
        return None, str(exc)
    _CONTEXT_MEMO[key] = ctx
    return ctx, None


def cases_status(recipe_path: Path | str | None = None, runs_dir: Path | str | None = None,
                 feature_set: str = "triage", tag: str | None = None) -> tuple[bool, str]:
    """
    Szybkie sprawdzenie do planu (--dry-run), BEZ ładowania modeli i danych: czy części esi/tools
    mają z czego zbudować przypadki. Zwraca (gotowe, opis źródła receptury albo powód pominięcia).
    """
    from src.data.dataset import is_prepared

    try:
        if recipe_path:
            recipe = load_recipe(recipe_path, runs_dir)
            runs = [r for k, r in zip(recipe.keys, recipe.runs) if float(recipe.weights.get(k, 0.0)) > 0]
            source = f"receptura {Path(recipe_path).as_posix()}"
        else:
            runs = latest_runs(feature_set, tag=tag, runs_dir=runs_dir, require_tuned=True)
            source = "najnowsze runy po tuningu"
            if not runs:
                runs = latest_runs(feature_set, tag=tag, runs_dir=runs_dir, require_tuned=True,
                                   include_samples=True)
                source = "runy na próbkach (np. smoke) — wyłącznie test"
            if not runs:
                return False, ("brak ukończonych runów po tuningu — wytrenuj modele (scripts/run_all.py) "
                               "albo podaj --recipe runs/final_report__*/recipe.json")
    except CasesUnavailable as exc:
        return False, str(exc)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return False, f"nie da się wczytać receptury: {exc}"
    no_model = [r.name for r in runs if not (r.path / "model.joblib").exists()]
    if no_model:
        return False, f"runy bez pliku model.joblib: {', '.join(no_model)}"
    if not is_prepared():
        return False, "brak aktualnych splitów danych ESI (data/processed/esi) — uruchom scripts/02_preprocess.py"
    return True, f"{source}: {', '.join(r.model for r in runs)}"


__all__ = [
    "CasesUnavailable",
    "EsiCase",
    "EsiContext",
    "Recipe",
    "RecipePredictor",
    "ShapHelper",
    "allocate_strata",
    "build_cases",
    "cases_status",
    "describe_patient",
    "format_feature_value",
    "load_recipe",
    "prepare_context",
    "recipe_from_runs",
    "resolve_recipe",
    "stored_decisions",
    "try_prepare_context",
]
