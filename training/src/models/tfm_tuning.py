"""
"Tuning" modeli fundamentalnych (TFM) = WYBÓR KONFIGURACJI na holdoucie z TRAIN (zamiast Optuny).

Wołane przez `run_training(..., tune=True)` dla modeli z `pretrained = True`. Kandydaci (n_ctx × n_estimators
× strategia kontekstu × max_features) pochodzą z profilu (`config.TUNING_PROFILES[profil][model]["candidates"]`).

Protokół:
    - holdout: `holdout_rows` wierszy X_fit, stratyfikowanych po ESI × oddział (rzadkie kombinacje → ESI);
      wiersze holdoutu NIGDY nie trafiają do kontekstu kandydata. VAL i TEST nie są używane,
    - etap 1: kandydaci od najtańszego (koszt ~ n_ctx × n_estimators × udział cech), każdy z n_bags=1 →
      log loss holdoutu (kryterium), QWK (argmax), czas, wierszy/s, szczyt VRAM,
    - etap 2 (profil z `bags_stage`, gdy starczy czasu): najlepsza konfiguracja etapu 1 z n_bags="auto"
      (podział rozłączny, max_bags z profilu) vs 1 worek,
    - wybór: najniższy log loss holdoutu; kandydaci w odległości ≤ TFM_SELECTION_TOLERANCE (0,002) od
      najlepszego → wygrywa tańszy,
    - limit czasu `timeout_s` (łącznie z wynikami z poprzednich sesji, jak w badaniu Optuny): pozostali
      kandydaci → "skipped" (WARNING z listą, nigdy po cichu); pierwszy kandydat liczy się zawsze,
    - błąd / brak pamięci kandydata → "failed" / "oom" w tabeli, wybór trwa dalej,
    - wznawialność: wynik każdego kandydata w `<storage_dir>/tfm_<model>_<klucz>.json` ("kind": "tfm_candidate";
      klucz = model, zestaw cech, profil, odcisk danych, efektywna konfiguracja, holdout, biblioteka + wersja,
      identyfikator wag); ponowne uruchomienie używa gotowych ("complete" i "oom"; "failed" liczone od nowa);
      blokada `<badanie>.lock` (jeden proces na wybór); podsumowanie `<badanie>.best.json` ("kind": "tfm_selection").

Wynik ma kształt wyniku `tune_model` (klucze czytane przez train.py / orkiestrator): kind="tfm_config_selection",
metric="holdout_logloss", best_params, best_value, study_name, best_trial, n_trials, n_complete, n_pruned
(= pominięci przez limit czasu), n_fail (błędy + OOM), n_fold_train=None, candidates, throughput, …

`storage_dir` przekazuje train.py (moduł NIE importuje OPTUNA_STORAGE_DIR — testy podmieniają ścieżki per moduł).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, log_loss
from sklearn.model_selection import train_test_split

from src.models import get_model, resolve_model_name
from src.models.tfm_base import adapter_for, free_device_memory, is_tfm_oom, resolve_n_bags
from src.models.tuning import StudyLock, _config_hash, _fold_strata, _json_ready, _safe_file_stem
from src.utils import config as _config
from src.utils.config import N_CLASSES, RANDOM_SEED
from src.utils.fileio import write_json_atomic
from src.utils.logger import get_logger

log = get_logger(__name__)

KIND = "tfm_config_selection"
METRIC = "holdout_logloss"
CANDIDATE_KIND = "tfm_candidate"
SELECTION_KIND = "tfm_selection"
# Klucze ustalane przez maszynę (nie wchodzą do klucza kandydata) — jak REFIT_MACHINE_KEYS w train.py
_MACHINE_KEYS = ("device", "n_jobs")
_REUSABLE = ("complete", "oom")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ─────────────────────────────────────────
# Pomocnicze
# ─────────────────────────────────────────
def candidate_cost(config: Mapping[str, Any], n_features: int, n_rows_pool: int | None = None) -> float:
    """Względny koszt kandydata: worki × n_ctx × n_estimators ("auto" = 8) × udział cech (max_features)."""
    n_est = config.get("n_estimators", 8)
    n_est = 8 if str(n_est).lower() == "auto" else int(n_est)
    n_ctx = int(config.get("n_ctx") or 0)
    if n_rows_pool:
        n_ctx = min(n_ctx, int(n_rows_pool))
    frac = 1.0
    if config.get("max_features") and n_features:
        frac = min(1.0, int(config["max_features"]) / int(n_features))
    bags = config.get("n_bags", 1)
    if isinstance(bags, str):
        bags = resolve_n_bags(bags, int(n_rows_pool or n_ctx or 1), max(n_ctx, 1), int(config.get("max_bags") or 1))
    return float(int(bags) * n_ctx * n_est * frac)


def holdout_split(y: np.ndarray, strata: np.ndarray | None, n_holdout: int,
                  seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray, str]:
    """
    (pozycje puli kontekstu, pozycje holdoutu, opis stratyfikacji): holdout stratyfikowany po ESI × oddział
    (kombinacje < 2 wierszy → ESI), najwyżej połowa wierszy.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    n_holdout = int(min(int(n_holdout), n // 2))
    if n_holdout < 1:
        raise ValueError(f"Za mało wierszy do wyboru konfiguracji TFM (X_fit: {n})")
    positions = np.arange(n)
    labels = _fold_strata(y, strata, min_count=2)
    how = "esi_x_site" if strata is not None and labels is not y else "esi"
    for stratify, name in ((labels, how), (y, "esi"), (None, "none")):
        try:
            pool, hold = train_test_split(positions, test_size=n_holdout, random_state=int(seed), stratify=stratify)
            return np.sort(pool), np.sort(hold), name
        except ValueError:
            continue
    raise RuntimeError("holdout_split: nie udało się podzielić danych")   # nieosiągalne (stratify=None)


def final_bag_budget(throughput: Mapping[str, Any] | None, n_bags: int, n_rows: int,
                     final_budget_s: float | None) -> dict[str, Any]:
    """
    Szacunek czasu finalnej predykcji (VAL + TEST) z przepustowości zmierzonej w wyborze konfiguracji:
    n_bags × (dopasowanie worka + wiersze / wierszy na s). Powyżej `final_budget_s` → mniej worków.

    Returns
    -------
    dict: n_bags (po redukcji), reduce (bool), estimate_s, per_bag_s, final_budget_s.
    """
    out = {"n_bags": int(n_bags), "reduce": False, "estimate_s": None, "per_bag_s": None,
           "final_budget_s": final_budget_s}
    rows_per_s = (throughput or {}).get("rows_per_s")
    if not rows_per_s or not final_budget_s or n_bags < 1:
        return out
    per_bag = float((throughput or {}).get("fit_s_per_bag") or 0.0) + float(n_rows) / float(rows_per_s)
    out["per_bag_s"] = round(per_bag, 1)
    out["estimate_s"] = round(per_bag * int(n_bags), 1)
    if per_bag * int(n_bags) > float(final_budget_s) and n_bags > 1:
        out["n_bags"] = max(1, int(float(final_budget_s) // per_bag))
        out["reduce"] = out["n_bags"] < int(n_bags)
    return out


def _effective_params(model_name: str, fixed: Mapping[str, Any] | None, candidate: Mapping[str, Any]) -> dict:
    """Parametry kandydata: domyślne wrappera + stałe (custom_params) + kandydat (bez kluczy maszyny)."""
    defaults = dict(getattr(_config, f"{model_name.upper()}_DEFAULT_PARAMS"))
    merged = {**defaults, **dict(fixed or {}), **dict(candidate)}
    return {k: v for k, v in merged.items() if k not in _MACHINE_KEYS}


def _weights_key(model_name: str, params: Mapping[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
    """(identyfikator wag do klucza, wersja biblioteki, przekierowanie) — bez budowania estymatora."""
    adapter = adapter_for(model_name)
    weights = adapter.resolve_weights(params)
    ident = {k: weights.get(k) for k in ("file", "size", "sha256")}
    return ident, adapter.library_version(), adapter.weights_override()


def _cache_path(storage_dir: Path, model_name: str, payload: Mapping[str, Any]) -> tuple[str, Path]:
    """Klucz kandydata = sha1[:16] posortowanego JSON-a konfiguracji → plik `tfm_<model>_<klucz>.json`."""
    text = json.dumps(_json_ready(dict(payload)), sort_keys=True, default=str)
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return key, storage_dir / f"tfm_{model_name}_{key}.json"


def _read_cache(path: Path, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("kind") != CANDIDATE_KIND:
        return None
    if _json_ready(data.get("key_payload")) != _json_ready(dict(payload)):
        return None                                               # kolizja skrótu — liczymy od nowa
    return data if data.get("status") in _REUSABLE else None


def _evaluate(model_name: str, params: Mapping[str, Any], X_pool: pd.DataFrame, y_pool: np.ndarray,
              strata_pool: np.ndarray | None, X_ho: pd.DataFrame, y_ho: np.ndarray) -> dict[str, Any]:
    """Jeden kandydat: kontekst z puli → predykcja holdoutu → metryki i pomiary."""
    model = get_model(model_name, params=dict(params))
    try:
        model.context_strata = strata_pool
        t0 = time.time()
        model.fit_estimator(X_pool, y_pool)
        proba = model.predict_proba(X_ho)
        wall = time.time() - t0
        stats = model.model.throughput()
        return {
            "status": "complete",
            "holdout_logloss": float(log_loss(y_ho, proba, labels=list(range(N_CLASSES)))),
            "qwk_argmax": float(cohen_kappa_score(y_ho, proba.argmax(axis=1), weights="quadratic")),
            "wall_s": round(wall, 2),
            "fit_s_per_bag": stats.get("fit_s_per_bag"),
            "predict_s": stats.get("predict_s"),
            "rows_per_s": stats.get("rows_per_s"),
            "peak_vram_mb": stats.get("peak_vram_mb"),
            "n_ctx_used": model.model.max_context(),
            "n_bags_used": model.model.n_bags,
            "runtime": dict(model.model.runtime),
            "events": list(model.tfm_events),
        }
    finally:
        model.release()
        del model
        free_device_memory()


# ─────────────────────────────────────────
# Wybór konfiguracji
# ─────────────────────────────────────────
def select_config(
    model_name: str,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray | pd.Series,
    *,
    strata: np.ndarray | None,
    budget: Mapping[str, Any],
    profile: str,
    feature_set: str,
    storage_dir: Path | str,
    data_fp: str,
    fixed_params: Mapping[str, Any] | None = None,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """
    Wybór konfiguracji TFM na holdoucie z X_fit (zob. opis modułu). `strata` — pozycyjnie względem X_fit
    (ESI × oddział, `train._site_strata`); `budget` — z `train.resolve_budget` (n_trials = liczba kandydatów
    etapu 1, timeout_s, candidates, holdout_rows, bags_stage, max_bags).
    """
    model_name = resolve_model_name(model_name)
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_fit).astype(int).ravel()
    if len(y) != len(X_fit):
        raise ValueError(f"select_config: len(y)={len(y)} ≠ len(X)={len(X_fit)}")
    candidates = [dict(c) for c in (budget.get("candidates") or [])]
    if not candidates:
        raise ValueError(f"{model_name}: profil '{profile}' nie ma kandydatów konfiguracji (candidates)")
    n_features = X_fit.shape[1]
    order = sorted(range(len(candidates)), key=lambda i: (candidate_cost(candidates[i], n_features), i))
    n_target = min(len(candidates), int(budget.get("n_trials") or len(candidates)))
    stage1 = [candidates[i] for i in order[:n_target]]
    dropped = [candidates[i] for i in order[n_target:]]
    timeout = budget.get("timeout_s")
    timeout = None if timeout in (None, 0) else float(timeout)
    max_bags = int(budget.get("max_bags") or 1)
    tolerance = float(_config.TFM_SELECTION_TOLERANCE)

    pool, hold, stratified_by = holdout_split(y, strata, int(budget.get("holdout_rows") or 0) or len(y) // 5, seed)
    X_pool, y_pool = X_fit.iloc[pool], y[pool]
    X_ho, y_ho = X_fit.iloc[hold], y[hold]
    strata_pool = None if strata is None else np.asarray(strata)[pool]
    holdout_spec = {"rows": int(len(hold)), "pool_rows": int(len(pool)), "n_rows": int(len(y)), "seed": int(seed),
                    "stratified_by": stratified_by}

    base_params = _effective_params(model_name, fixed_params, {})
    weights_id, library_version, override = _weights_key(model_name, base_params)
    study_payload = {"model": model_name, "feature_set": feature_set, "profile": profile, "data": data_fp,
                     "holdout": holdout_spec, "library": model_name, "library_version": library_version,
                     "weights": weights_id, "fixed": _json_ready(dict(fixed_params or {})),
                     "candidates": _json_ready(candidates)}
    study_name = _safe_file_stem(f"tfm_{model_name}__{feature_set}__{profile}__data-{data_fp}__"
                                 f"{_config_hash(study_payload)}")
    lock = StudyLock(storage_dir / f"{study_name}.lock", study_name)
    try:
        lock.acquire()
    except RuntimeError as exc:
        raise RuntimeError(f"Wybór konfiguracji TFM '{study_name}' trwa w innym procesie — jeden wybór = jeden "
                           f"proces ({exc})") from None

    log.info("─" * 64)
    log.info(f"Wybór konfiguracji {model_name} (holdout z TRAIN): {len(stage1)} kandydat(ów) etapu 1, holdout "
             f"{len(hold):,} wierszy ({stratified_by}), pula kontekstu {len(pool):,}"
             + (f", limit czasu {timeout / 60:.0f} min" if timeout else "")
             + (f"; poza budżetem n_trials: {len(dropped)}" if dropped else ""))
    if override:
        log.warning(f"Wybór konfiguracji {model_name} z przekierowaniem wag ({override}) — wyniki tylko testowe")
    t_session = time.time()
    entries: list[dict[str, Any]] = []
    elapsed_cached = 0.0

    def elapsed() -> float:
        return elapsed_cached + (time.time() - t_session)

    def run_candidate(candidate: Mapping[str, Any], stage: int) -> dict[str, Any]:
        nonlocal elapsed_cached
        params = _effective_params(model_name, fixed_params, candidate)
        w_id, version, _ = _weights_key(model_name, params)
        payload = {"kind": CANDIDATE_KIND, "model": model_name, "feature_set": feature_set, "profile": profile,
                   "data": data_fp, "config": _json_ready(params), "holdout": holdout_spec,
                   "library": model_name, "library_version": version, "weights": w_id}
        key, path = _cache_path(storage_dir, model_name, payload)
        entry: dict[str, Any] = {"index": len(entries), "stage": stage, "config": dict(candidate), "key": key,
                                 "cache_file": str(path),
                                 "cost": candidate_cost(candidate, n_features, len(pool)), "cached": False}
        cached = _read_cache(path, payload)
        if cached is not None:
            entry.update({k: v for k, v in cached.items() if k not in ("kind", "key_payload", "config")})
            entry["cached"] = True
            elapsed_cached += float(cached.get("wall_s") or 0.0)
            log.info(f"  [{entry['index']}] etap {stage} {_describe(candidate)}: z pamięci ({cached['status']}, "
                     f"log loss {_fmt(cached.get('holdout_logloss'))})")
            return entry
        n_complete = sum(1 for e in entries if e.get("status") == "complete")
        if timeout is not None and n_complete >= 1 and elapsed() >= timeout:
            entry.update({"status": "skipped", "reason": "timeout"})
            return entry
        log.info(f"  [{entry['index']}] etap {stage} {_describe(candidate)}: liczę…")
        t0 = time.time()
        try:
            result = _evaluate(model_name, params, X_pool, y_pool, strata_pool, X_ho, y_ho)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — błąd jednego kandydata nie przerywa wyboru
            status = "oom" if is_tfm_oom(exc) else "failed"
            result = {"status": status, "error": f"{type(exc).__name__}: {str(exc).strip()[:500]}",
                      "wall_s": round(time.time() - t0, 2)}
            log.warning(f"  [{entry['index']}] {_describe(candidate)}: {status.upper()} — {result['error']}")
        entry.update(result)
        write_json_atomic(path, {"kind": CANDIDATE_KIND, "selection": study_name, "key": key,
                                 "key_payload": payload, "config": dict(candidate), "stage": stage,
                                 "model": model_name, "feature_set": feature_set, "profile": profile,
                                 "data_fingerprint": data_fp, "finished_at": _now(), **result},
                          default=str)
        if result["status"] == "complete":
            log.info(f"  [{entry['index']}] {_describe(candidate)}: log loss {result['holdout_logloss']:.5f}, "
                     f"QWK {result['qwk_argmax']:.4f}, {result['wall_s']:.1f} s"
                     + (f", {result['rows_per_s']:.0f} wierszy/s" if result.get("rows_per_s") else ""))
        return entry

    try:
        for candidate in stage1:
            entries.append(run_candidate({**candidate, "n_bags": 1}, 1))
        complete = [e for e in entries if e.get("status") == "complete"]
        stage2: dict[str, Any] | None = None
        if budget.get("bags_stage") and complete:
            best1 = min(complete, key=lambda e: (e["holdout_logloss"], e["cost"], e["index"]))
            cfg = {**best1["config"], "n_bags": "auto", "max_bags": max_bags, "bag_mode": "partition"}
            n_auto = resolve_n_bags("auto", len(pool), int(cfg.get("n_ctx") or 1), max_bags)
            if n_auto > 1:
                entries.append(run_candidate(cfg, 2))
                stage2 = {"from": best1["index"], "n_bags_auto": n_auto, "entry": entries[-1]["index"]}
            else:
                stage2 = {"from": best1["index"], "n_bags_auto": 1, "skipped": "pula mieści się w jednym worku"}
        skipped = [e for e in entries if e.get("status") == "skipped"]
        if skipped:
            log.warning(f"Limit czasu wyboru konfiguracji {model_name} ({timeout / 60:.0f} min) — pominięci "
                        f"kandydaci: " + "; ".join(_describe(e["config"]) for e in skipped))
        complete = [e for e in entries if e.get("status") == "complete"]
        if not complete:
            errors = "; ".join(f"{_describe(e['config'])}: {e.get('error') or e.get('status')}" for e in entries)
            raise RuntimeError(f"{model_name}: żaden kandydat konfiguracji nie zakończył się ({errors})")
        best_value = min(e["holdout_logloss"] for e in complete)
        near = [e for e in complete if e["holdout_logloss"] <= best_value + tolerance]
        chosen = min(near, key=lambda e: (e["cost"], e["holdout_logloss"], e["index"]))
        throughput = {k: chosen.get(k) for k in ("rows_per_s", "fit_s_per_bag", "predict_s", "peak_vram_mb",
                                                 "n_ctx_used", "n_bags_used")}
        throughput["source"] = f"kandydat #{chosen['index']} (etap {chosen['stage']})"
        n_fail = sum(1 for e in entries if e.get("status") in ("failed", "oom"))
        total_elapsed = elapsed()
        result: dict[str, Any] = {
            "kind": KIND,
            "metric": METRIC,
            "direction": "minimize",
            "study_name": study_name,
            "best_params": dict(chosen["config"]),
            "best_value": float(chosen["holdout_logloss"]),
            "best_value_std": None,
            "best_trial": int(chosen["index"]),
            "best_qwk_argmax": chosen.get("qwk_argmax"),
            "best_overall_value": float(best_value),
            "tolerance": tolerance,
            "n_trials": int(sum(1 for e in entries if e.get("status") != "skipped")),
            "n_trials_target": int(n_target),
            "n_candidates_profile": int(len(candidates)),
            "n_complete": int(len(complete)),
            "n_pruned": int(len(skipped)),
            "n_skipped": int(len(skipped)),
            "n_fail": int(n_fail),
            "n_oom": int(sum(1 for e in entries if e.get("status") == "oom")),
            "n_cached": int(sum(1 for e in entries if e.get("cached"))),
            "n_fold_train": None,
            "n_folds": 1,
            "n_rows": int(len(y)),
            "holdout": holdout_spec,
            "candidates": entries,
            "stage2": stage2,
            "throughput": throughput,
            "timeout_s": timeout,
            "elapsed_s": round(total_elapsed, 1),
            "stopped_reason": "timeout" if skipped else "complete",
            "budget_exhausted": bool(skipped) or (timeout is not None and total_elapsed >= timeout),
            "library": model_name,
            "library_version": library_version,
            "weights": weights_id,
            "weights_override": override,
            "fixed_params": dict(fixed_params) if fixed_params else None,
            "storage": str(storage_dir),
            "best_json": None,
        }
        best_json = storage_dir / f"{study_name}.best.json"
        write_json_atomic(best_json, {**result, "kind": SELECTION_KIND, "result_kind": KIND,
                                      "best_json": str(best_json), "model": model_name, "feature_set": feature_set,
                                      "profile": profile, "data_fingerprint": data_fp, "written_at": _now()},
                          default=str)
        result["best_json"] = str(best_json)
        log.info(f"Wybrano {model_name}: {_describe(chosen['config'])} — log loss holdoutu {chosen['holdout_logloss']:.5f}"
                 + (f" (najlepszy {best_value:.5f}; w tolerancji {tolerance} wygrywa tańszy)"
                    if chosen["holdout_logloss"] > best_value else "")
                 + f"; kandydaci: ukończeni {len(complete)}, błędy {n_fail}, pominięci {len(skipped)}")
        return result
    finally:
        lock.release()


def _describe(config: Mapping[str, Any]) -> str:
    parts = [f"n_ctx={config.get('n_ctx')}", f"n_est={config.get('n_estimators')}"]
    for key in ("n_bags", "context_strategy", "max_features"):
        if config.get(key) not in (None, 1, "stratified"):
            parts.append(f"{key}={config[key]}")
    return "{" + ", ".join(parts) + "}"


def _fmt(value: Any) -> str:
    return f"{value:.5f}" if isinstance(value, (int, float)) else "—"


def selection_progress(storage_dir: Path | str, model_name: str, *, feature_set: str | None = None,
                       profile: str | None = None, data_fp: str | None = None) -> dict[str, Any] | None:
    """
    Postęp wyboru konfiguracji z plików kandydatów (`--status` orkiestratora): liczba ukończonych / OOM /
    błędnych kandydatów pasujących do filtrów; None, gdy brak plików.
    """
    counts = {"complete": 0, "oom": 0, "failed": 0}
    found = False
    for path in sorted(Path(storage_dir).glob(f"tfm_{model_name}_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("kind") != CANDIDATE_KIND:
            continue
        if any(v is not None and data.get(k) != v for k, v in
               (("feature_set", feature_set), ("profile", profile), ("data_fingerprint", data_fp))):
            continue
        found = True
        status = str(data.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts if found else None


__all__ = ["CANDIDATE_KIND", "KIND", "METRIC", "SELECTION_KIND", "candidate_cost", "final_bag_budget",
           "holdout_split", "select_config", "selection_progress"]
