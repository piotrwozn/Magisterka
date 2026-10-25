"""
Wspólne elementy analiz (seedy, leave-site-out, krzywa uczenia).

    source_runs(...)        najnowszy UKOŃCZONY, dostrojony run profilu każdego modelu (bez refitów
                            i runów niestandardowych) — źródło parametrów do refitu,
    find_refit_run(...)     gotowy refit danej jednostki (wznawianie po awarii bez ponownego treningu),
    UnitStore               pliki jednostek units/<jednostka>.json, plan.json, results.json,
    merge_units(...)        wyniki kumulatywne: jednostki tego wywołania + wcześniejsze z innych grup
                            (np. inne modele) z tym samym runem źródłowym,
    Eta                     postęp i szacowany czas reszty,
    evaluate(...)           metryki na predykcjach (reguły decyzyjne, prawdopodobieństwa, podgrupa vc),
    analyses_status(...)    postęp analiz z plików (dla `run_all.py --status`; bez treningu i modeli).

Katalog analizy: runs/analyses/<analiza>__<cechy>__<tag>/.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.utils import config
from src.utils.fileio import write_json_atomic, write_text_atomic
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# Stałe
# ─────────────────────────────────────────
ANALYSES_SUBDIR = "analyses"
ANALYSIS_KINDS: tuple[str, ...] = ("seeds", "site_out", "learning_curve")
# Podbij przy zmianie metryk jednostek — jednostki zostaną przeliczone z predykcji gotowych refitów
# (2: O/E = obserwowane / oczekiwane; korekta rozkładu względem rozkładu ważonego wagami klas)
METRICS_VERSION = 2
PROBA_COLS: list[str] = [f"p_{name.lower()}" for name in config.CLASS_NAMES]


class NoSourceRuns(RuntimeError):
    """Brak ukończonych, dostrojonych runów profilu — najpierw trening (run_all.py)."""


class AnalysisArgumentError(ValueError):
    """Złe argumenty analizy (np. oddział, którego nie ma w danych) — skrypt kończy się kodem 2."""


# ─────────────────────────────────────────
# Ścieżki i JSON
# ─────────────────────────────────────────
def analyses_root(runs_dir: Path | str | None = None) -> Path:
    """runs/analyses (config.RUNS_DIR rozwiązywany przy wywołaniu — testy i zmienne środowiska)."""
    return Path(runs_dir if runs_dir is not None else config.RUNS_DIR) / ANALYSES_SUBDIR


def analysis_dir(kind: str, feature_set: str, tag: str, runs_dir: Path | str | None = None) -> Path:
    if kind not in ANALYSIS_KINDS:
        raise ValueError(f"Nieznana analiza '{kind}'. Dostępne: {', '.join(ANALYSIS_KINDS)}")
    return analyses_root(runs_dir) / f"{kind}__{feature_set}__{tag}"


def read_json(path: Path | str) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def clean_json(obj: Any) -> Any:
    """NaN / ±inf → None, typy numpy → typy Pythona (ścisły JSON dla każdego czytelnika)."""
    if isinstance(obj, Mapping):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean_json(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_json_atomic(path, clean_json(payload))


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ─────────────────────────────────────────
# Runy źródłowe i refity
# ─────────────────────────────────────────
def default_tag() -> str:
    """Tag runów źródłowych: profil domyślny (orchestrator: tag = profil przy tuningu)."""
    return config.DEFAULT_TUNING_PROFILE


def source_runs(
    feature_set: str,
    tag: str,
    models: Sequence[str] | None = None,
    runs_dir: Path | str | None = None,
    require_tuned: bool = True,
) -> dict[str, Any]:
    """
    {model: Run} — najnowszy UKOŃCZONY run każdego modelu z tagiem `tag` i bieżącymi danymi
    (wersja + odcisk), w kolejności rejestru modeli.

    Pomijane: refity (`refit_of`), runy niestandardowe (`custom`), bez tuningu (gdy require_tuned),
    z innym odciskiem danych (z ostrzeżeniem — refit na innych danych nie byłby porównywalny).
    Runy na próbkach (profil smoke / quick) są dozwolone: refit zachowa ich próbki.
    """
    from src.data.dataset import data_fingerprint
    from src.ensemble.runs import list_runs, load_run, resolve_feature_set
    from src.models import MODEL_REGISTRY

    fs = resolve_feature_set(feature_set)
    table = list_runs(runs_dir)
    if table.empty:
        return {}
    table = table[(table["feature_set"] == fs) & (table["tag"] == tag) & table["done"]
                  & (table["data_version"] == config.DATA_VERSION)]
    fingerprint = data_fingerprint()
    chosen: dict[str, Any] = {}
    stale: list[str] = []
    for name in table["run"]:                       # kolejność nazw = kolejność czasu → ostatni wygrywa
        run = load_run(name, runs_dir)
        info = run.info
        if info.get("custom") or info.get("refit_of"):
            continue
        if require_tuned and not info.get("tuned"):
            continue
        if models is not None and run.model not in models:
            continue
        if info.get("data_fingerprint") != fingerprint:
            stale.append(name)
            continue
        chosen[run.model] = run
    if stale:
        log.warning(f"Pominięto runy z innych danych (odcisk ≠ {fingerprint}): {', '.join(stale)}")
    return {m: chosen[m] for m in MODEL_REGISTRY if m in chosen}


def require_source_runs(feature_set: str, tag: str, models: Sequence[str] | None,
                        runs_dir: Path | str | None = None) -> dict[str, Any]:
    """Jak `source_runs`, ale brak runów → NoSourceRuns z poleceniem, które je utworzy."""
    runs = source_runs(feature_set, tag, models, runs_dir)
    if not runs:
        wanted = ", ".join(models) if models else "dowolny model"
        raise NoSourceRuns(
            f"Brak ukończonych, dostrojonych runów z tagiem '{tag}' (zestaw '{feature_set}', {wanted}) "
            f"na bieżących danych. Najpierw: python scripts/run_all.py --profile {tag.split('-')[0]}")
    missing = [m for m in (models or []) if m not in runs]
    if missing:
        log.warning(f"Brak runów źródłowych dla: {', '.join(missing)} — pomijam te modele")
    return runs


def find_refit_run(source_run: str, tag: str, unit: str, runs_dir: Path | str | None = None) -> Path | None:
    """
    Najnowszy ukończony refit `runs/*_<tag>/` tej jednostki z tego runa źródłowego, tych samych danych
    i bieżącej wersji procedury refitu (`src.models.train.REFIT_VERSION`).
    """
    from src.data.dataset import data_fingerprint
    from src.models.train import REFIT_VERSION

    root = Path(runs_dir if runs_dir is not None else config.RUNS_DIR)
    fingerprint = data_fingerprint()
    for done_path in sorted(root.glob(f"*_{tag}/DONE.json"), reverse=True):
        done = read_json(done_path)
        if (done and done.get("refit_of") == source_run and done.get("unit") == unit
                and done.get("tag") == tag and done.get("data_fingerprint") == fingerprint
                and done.get("refit_version") == REFIT_VERSION):
            return done_path.parent
    return None


def source_seed(info: Mapping[str, Any]) -> int:
    """Seed modelu runa źródłowego (random_state / random_seed; brak → config.RANDOM_SEED)."""
    params = info.get("params") or {}
    for key in ("random_state", "random_seed"):
        value = params.get(key)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
    return int(config.RANDOM_SEED)


def load_predictions(run_dir: Path | str, name: str) -> tuple[np.ndarray, np.ndarray, Any]:
    """(y, proba (n, 5) float64, ramka predykcji) z `predictions_<name>.parquet`."""
    import pandas as pd

    frame = pd.read_parquet(Path(run_dir) / f"predictions_{name}.parquet")
    proba = frame[PROBA_COLS].to_numpy(dtype=np.float64)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return frame["target"].to_numpy().astype(int), proba, frame


_ADMITTED_CACHE: dict[tuple[str, str], Any] = {}


def admitted_for(row_ids: np.ndarray, splits: Sequence[str]) -> np.ndarray:
    """
    Przyjęcie do szpitala (0/1, NaN = brak) dla `row_id` z podanych splitów — predykcje runów go nie
    zawierają. Czyta tylko kolumny row_id i admitted (pamięć podręczna w procesie).
    """
    import pandas as pd

    from src.data.dataset import data_fingerprint, load_split

    fingerprint = data_fingerprint()
    parts = []
    for split in splits:
        key = (split, fingerprint)
        if key not in _ADMITTED_CACHE:
            frame = load_split(split, columns=["row_id", "admitted"])
            _ADMITTED_CACHE[key] = frame.set_index("row_id")["admitted"].astype(float)
        parts.append(_ADMITTED_CACHE[key])
    lookup = pd.concat(parts) if len(parts) > 1 else parts[0]
    lookup = lookup[~lookup.index.duplicated()]
    return lookup.reindex(np.asarray(row_ids)).to_numpy(dtype=float)


def vc_mask(frame: Any) -> np.ndarray | None:
    """vitals_complete == 1; None, gdy kolumny brak albo same braki (stare dane)."""
    if "vitals_complete" not in frame.columns:
        return None
    values = np.asarray(frame["vitals_complete"], dtype=float)
    if np.isnan(values).all():
        return None
    return values == 1


# ─────────────────────────────────────────
# Jednostki (wznawianie)
# ─────────────────────────────────────────
class UnitStore:
    """
    Pliki analizy: plan.json (jednostki tego wywołania), units/<jednostka>.json (wynik jednostki,
    zapis atomowy po jej zakończeniu), results.json (zbiorczy wynik), summary.md.
    Jednostka jest gotowa, gdy jej plik ma status 'done' i zgadza się z oczekiwaniami (run źródłowy,
    odcisk danych, wersja metryk, konfiguracja).
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.units_dir = self.root / "units"

    def path(self, unit: str) -> Path:
        return self.units_dir / (unit.replace("/", "__") + ".json")

    def load(self, unit: str) -> dict | None:
        return read_json(self.path(unit))

    def valid(self, unit: str, expect: Mapping[str, Any]) -> dict | None:
        payload = self.load(unit)
        if not payload or payload.get("status") != "done":
            return None
        if any(clean_json(payload.get(k)) != clean_json(v) for k, v in expect.items()):
            return None
        return payload

    def save(self, unit: str, payload: Mapping[str, Any]) -> dict:
        record = {**payload, "unit": unit, "status": "done", "finished_at": now_iso()}
        write_json(self.path(unit), record)
        return record

    def write_plan(self, payload: Mapping[str, Any]) -> None:
        write_json(self.root / "plan.json", {**payload, "written_at": now_iso()})

    def planned_units(self) -> list[dict]:
        """Pełne opisy jednostek z poprzedniego plan.json (bez pola 'model' — stary format, pomijane)."""
        plan = read_json(self.root / "plan.json") or {}
        return [dict(u) for u in plan.get("units") or [] if isinstance(u, Mapping) and u.get("unit") and u.get("model")]

    def write_results(self, payload: Mapping[str, Any]) -> Path:
        return write_json(self.root / "results.json", payload)

    def write_summary(self, lines: Sequence[str]) -> Path:
        path = self.root / "summary.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, "\n".join(lines) + "\n")
        return path


def merge_units(
    previous: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    group: Callable[[Mapping[str, Any]], Any],
    current_sources: Mapping[str, str],
) -> list[dict[str, Any]]:
    """
    Wyniki analizy są KUMULATYWNE: jednostki tego wywołania + jednostki z poprzedniego planu
    z grup (`group(u)`, np. model albo oddział/model), których to wywołanie nie liczy — o ile ich
    run źródłowy jest nadal najnowszym runem źródłowym modelu (`current_sources`). Grupa liczona
    teraz zastępuje swoje stare jednostki w całości (np. inna liczba seedów); jednostki starego
    runa źródłowego odpadają. Kolejność: modele jak w rejestrze (w modelu — kolejność jednostek).
    """
    from src.models import MODEL_REGISTRY

    groups = {group(u) for u in current}
    names = {str(u["unit"]) for u in current}
    kept = [dict(u) for u in previous
            if group(u) not in groups and str(u["unit"]) not in names
            and current_sources.get(str(u.get("model"))) == u.get("source_run")]
    order = {m: i for i, m in enumerate(MODEL_REGISTRY)}
    return sorted(kept + [dict(u) for u in current], key=lambda u: order.get(str(u.get("model")), len(order)))


def current_source_names(feature_set: str, tag: str, runs_dir: Path | str | None = None) -> dict[str, str]:
    """{model: nazwa najnowszego runa źródłowego} dla WSZYSTKICH modeli (do łączenia planów)."""
    return {m: r.name for m, r in source_runs(feature_set, tag, None, runs_dir).items()}


def analysis_progress(path: Path) -> dict[str, Any] | None:
    """{total, done, complete, failures, updated_at} z plan.json i plików jednostek; None = brak analizy."""
    plan = read_json(Path(path) / "plan.json")
    if plan is None:
        return None
    store = UnitStore(Path(path))
    units = plan.get("units") or []
    done = 0
    for item in units:
        payload = store.load(str(item.get("unit")))
        if payload and payload.get("status") == "done" and payload.get("source_run") == item.get("source_run"):
            done += 1
    results = read_json(Path(path) / "results.json") or {}
    return {
        "total": len(units),
        "done": done,
        "complete": bool(results.get("complete")) and done == len(units),
        "failures": len(results.get("failures") or []),
        "updated_at": results.get("created") or plan.get("written_at"),
        "results": str(Path(path) / "results.json") if results else None,
    }


def analyses_status(feature_set: str, tag: str, runs_dir: Path | str | None = None) -> dict[str, dict | None]:
    """Postęp każdej analizy (seeds, site_out, learning_curve) dla zestawu cech i tagu."""
    return {kind: analysis_progress(analysis_dir(kind, feature_set, tag, runs_dir)) for kind in ANALYSIS_KINDS}


# ─────────────────────────────────────────
# Postęp i ETA
# ─────────────────────────────────────────
def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "—"
    seconds = int(round(max(0.0, seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def execute_units(
    store: UnitStore,
    units: Sequence[Mapping[str, Any]],
    expect: Callable[[Mapping[str, Any]], dict[str, Any]],
    compute: Callable[[Mapping[str, Any]], dict[str, Any]],
    label: str,
    force: bool = False,
    estimate_s: Callable[[Mapping[str, Any]], float | None] | None = None,
) -> list[dict[str, str]]:
    """
    Pętla jednostek analizy: gotowe (plik jednostki zgodny z `expect(u)`) → pomijane; pozostałe →
    `compute(u)` (refit albo ponowne użycie gotowego refitu + metryki) i zapis pliku jednostki.
    Błąd jednej jednostki nie przerywa pozostałych (lista błędów w wyniku); Ctrl+C — przerywa.
    """
    todo = [u for u in units if force or store.valid(str(u["unit"]), expect(u)) is None]
    skipped = len(units) - len(todo)
    if skipped:
        log.info(f"[{label}] gotowe wcześniej: {skipped}/{len(units)} jednostek — pomijam (--force liczy od nowa)")
    eta = Eta(label, len(todo))
    failures: list[dict[str, str]] = []
    for u in todo:
        unit = str(u["unit"])
        eta.start(unit, estimate_s(u) if estimate_s else None)
        try:
            payload = compute(u)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — jedna jednostka nie zatrzymuje analizy
            log.exception(f"[{label}] {unit}: NIEUDANE — {type(exc).__name__}: {exc}")
            failures.append({"unit": unit, "error": f"{type(exc).__name__}: {exc}"[:500]})
            eta.finish()
            continue
        store.save(unit, {**expect(u), **payload, "duration_s": round(eta.finish(), 1)})
    return failures


class Eta:
    """Licznik jednostek: `start(jednostka, szacunek_s)` loguje [k/n] i szacowany czas reszty."""

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = int(total)
        self.k = 0
        self.durations: list[float] = []
        self._t0: float | None = None

    def start(self, unit: str, estimate_s: float | None = None) -> None:
        self.k += 1
        self._t0 = time.time()
        remaining = self.total - self.k + 1
        per_unit = float(np.mean(self.durations)) if self.durations else estimate_s
        eta = None if per_unit is None else per_unit * remaining
        log.info(f"[{self.label} {self.k}/{self.total}] {unit} — szacowany czas reszty ≈ {format_duration(eta)}")

    def finish(self) -> float:
        duration = time.time() - (self._t0 or time.time())
        self.durations.append(duration)
        return duration


# ─────────────────────────────────────────
# Metryki na predykcjach
# ─────────────────────────────────────────
def rule_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """QWK, niedotriażowanie (ESI 1–2 → 3–5), krytyczne pominięcia (ESI 1 → 4–5), czułość ESI 1."""
    from src.evaluation.metrics import critical_miss_rate, quadratic_weighted_kappa, undertriage_rate

    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qwk = quadratic_weighted_kappa(y, pred) if len(np.unique(np.r_[y, pred])) > 1 else float("nan")
    esi1 = y == 0
    return {
        "qwk": qwk,
        "undertriage": undertriage_rate(y, pred),
        "critical_miss": critical_miss_rate(y, pred),
        "recall_esi1": float((pred[esi1] == 0).mean()) if esi1.any() else float("nan"),
    }


def proba_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Log loss (klasy 0–4) i makro-AUC one-vs-rest (klasy obecne w y)."""
    from sklearn.metrics import log_loss, roc_auc_score

    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=np.float64)
    if len(y) == 0:
        return {"log_loss": float("nan"), "auc_macro": float("nan")}
    aucs = []
    for k in range(proba.shape[1]):
        positives = y == k
        if positives.any() and not positives.all():
            aucs.append(float(roc_auc_score(positives.astype(int), proba[:, k])))
    return {
        "log_loss": float(log_loss(y, proba, labels=list(range(proba.shape[1])))),
        "auc_macro": float(np.mean(aucs)) if aucs else float("nan"),
    }


def admission_auc(admitted: np.ndarray, proba: np.ndarray) -> float:
    """AUC przyjęcia do szpitala z −E[ESI] (niższy oczekiwany poziom = pilniej); NaN bez obu klas."""
    from sklearn.metrics import roc_auc_score

    adm = np.asarray(admitted, dtype=float)
    ok = np.isfinite(adm)
    adm = adm[ok].astype(int)
    if len(adm) == 0 or adm.min() == adm.max():
        return float("nan")
    expected_level = np.asarray(proba)[ok] @ np.arange(np.asarray(proba).shape[1])
    return float(roc_auc_score(adm, -expected_level))


def evaluate(
    y: np.ndarray,
    proba: np.ndarray,
    preds: Mapping[str, np.ndarray],
    vc: np.ndarray | None = None,
    admitted: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Płaski słownik: `<metryka>_<reguła>` (qwk, undertriage, critical_miss, recall_esi1),
    log_loss, auc_macro, admission_auc, n — oraz to samo z prefiksem `vc_` dla vitals_complete == 1.
    """
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=np.float64)
    out: dict[str, float] = {}

    def block(mask: np.ndarray, prefix: str) -> None:
        for rule, pred in preds.items():
            for name, value in rule_metrics(y[mask], np.asarray(pred)[mask]).items():
                out[f"{prefix}{name}_{rule}"] = value
        for name, value in proba_metrics(y[mask], proba[mask]).items():
            out[f"{prefix}{name}"] = value
        if admitted is not None:
            out[f"{prefix}admission_auc"] = admission_auc(np.asarray(admitted)[mask], proba[mask])
        out[f"{prefix}n"] = int(mask.sum())
        out[f"{prefix}n_esi1"] = int((y[mask] == 0).sum())

    block(np.ones(len(y), dtype=bool), "")
    if vc is not None and np.asarray(vc).any():
        block(np.asarray(vc, dtype=bool), "vc_")
    return out


def mean_sd(values: Sequence[float | None]) -> dict[str, Any]:
    """{'mean', 'sd' (n−1; None przy < 2 wartościach), 'min', 'max', 'n', 'values'}."""
    clean = [None if v is None or not math.isfinite(float(v)) else float(v) for v in values]
    finite = np.array([v for v in clean if v is not None], dtype=float)
    return {
        "mean": float(finite.mean()) if finite.size else None,
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "n": int(finite.size),
        "values": clean,
    }


# ─────────────────────────────────────────
# Reguły decyzyjne z src.evaluation.decision (gdy moduł jest dostępny)
# ─────────────────────────────────────────
def decision_module() -> Any | None:
    """src.evaluation.decision albo None (starsza wersja repo — analizy bez reguły „undertriage ≤ T”)."""
    try:
        from src.evaluation import decision
    except ImportError:
        return None
    return decision


def fit_population_mask(vc: np.ndarray | None) -> np.ndarray | None:
    """
    Wiersze do dopasowania reguły decyzyjnej: config.DECISION_FIT_POPULATION == 'vc' → vitals_complete
    (populacja jak w TOPSOR); 'all' albo brak kolumny vc → None (wszystkie wiersze).
    """
    population = getattr(config, "DECISION_FIT_POPULATION", "vc")
    if population == "vc" and vc is not None and np.asarray(vc).any():
        return np.asarray(vc, dtype=bool)
    return None


__all__ = [
    "ANALYSIS_KINDS",
    "METRICS_VERSION",
    "AnalysisArgumentError",
    "Eta",
    "NoSourceRuns",
    "UnitStore",
    "admission_auc",
    "admitted_for",
    "analyses_status",
    "analysis_dir",
    "analysis_progress",
    "clean_json",
    "current_source_names",
    "decision_module",
    "default_tag",
    "evaluate",
    "execute_units",
    "fit_population_mask",
    "find_refit_run",
    "format_duration",
    "load_predictions",
    "mean_sd",
    "merge_units",
    "proba_metrics",
    "read_json",
    "require_source_runs",
    "rule_metrics",
    "source_runs",
    "source_seed",
    "vc_mask",
    "write_json",
]
