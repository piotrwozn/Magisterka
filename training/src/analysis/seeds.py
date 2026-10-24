"""
C1 — zmienność między seedami: refit każdego finalnego modelu z najlepszymi parametrami na kilku
seedach (bez tuningu) → średnia ± SD metryk na val i test.

Po co: dwa runy różniące się tylko seedem potrafią wyjść „istotnie różne” w bootstrapie
(bootstrap opisuje skończony zbiór pacjentów, nie zmienność treningu). Wyższość modelu A nad B
ogłaszaj tylko wtedy, gdy przedział ufności Δ wyklucza 0, |Δ| przekracza ustaloną z góry minimalną
istotną różnicę ORAZ 2 × SD seedów (docs/ULEPSZENIA.md, C1).

Jednostka = (model, seed): refit `runs/<czas>_<model>_<cechy>_<tag>-seed<k>/` (domyślnie seedy 1…N,
zmienia się tylko seed modelu; `vary_es_split` — także podział fit / early stopping) + run
źródłowy jako jeszcze jedna wartość (jego seed, domyślnie 42).

Metryki (pełny zbiór i podgrupa `vc_`): QWK, niedotriażowanie, krytyczne pominięcia, czułość ESI 1
dla reguł argmax, koszt i target (B1 + B2: undertriage ≤ T z siatką bezpieczeństwa; dopasowana na
val tego seeda w populacji config.DECISION_FIT_POPULATION — na val liczba z cross-fittingu, na
teście reguła z całego val), log loss, makro-AUC, AUC przyjęcia.

Wynik: runs/analyses/seeds__<cechy>__<tag>/results.json
    {"models": {model: {"runs": [...], "val": {metryka: {"mean", "sd", "values", …}}, "test": {...}}},
     "metrics": [...], ...}
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.common import (
    METRICS_VERSION,
    UnitStore,
    admitted_for,
    analysis_dir,
    decision_module,
    default_tag,
    evaluate,
    execute_units,
    find_refit_run,
    fit_population_mask,
    load_predictions,
    mean_sd,
    now_iso,
    require_source_runs,
    source_seed,
    vc_mask,
)
from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)

RULES: tuple[str, ...] = ("argmax", "cost", "target")
_BASE = ("qwk", "undertriage", "critical_miss", "recall_esi1")
METRICS: list[str] = (
    [f"{m}_{r}" for r in RULES for m in _BASE] + ["log_loss", "auc_macro", "admission_auc"]
)
METRICS = METRICS + [f"vc_{m}" for m in METRICS]
SUMMARY_METRICS: tuple[str, ...] = ("qwk_argmax", "qwk_cost", "qwk_target", "undertriage_target",
                                    "critical_miss_target", "log_loss", "auc_macro", "admission_auc")


def seed_tag(tag: str, seed: int, vary_es_split: bool = False) -> str:
    """Tag refitu: '<tag>-seed<k>' (z innym podziałem early stopping: '<tag>-seed<k>es')."""
    return f"{tag}-seed{int(seed)}" + ("es" if vary_es_split else "")


def _target_rule_preds(
    y_val: np.ndarray, p_val: np.ndarray, vc_val: np.ndarray | None, p_test: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
    """(val: predykcje cross-fitted, test: reguła z całego val, opis reguły) albo None bez modułu decyzji."""
    decision = decision_module()
    if decision is None:
        return None, None, None
    fitter = functools.partial(decision.fit_target_rule,
                               target_undertriage=config.DECISION_TARGET_UNDERTRIAGE, safety_net=True)
    mask = fit_population_mask(vc_val)
    try:
        pred_val = decision.crossfit_rule(y_val, p_val, fitter, mask=mask)
        rule = fitter(y_val, p_val, mask=mask)
    except ValueError as exc:                       # np. brak ESI 1–2 w małej próbce
        log.warning(f"Reguła 'target' pominięta: {exc}")
        return None, None, None
    pred_test = rule.predict(p_test) if p_test is not None else None
    return pred_val, pred_test, rule.to_dict()


def unit_metrics(run_dir: Path | str) -> dict[str, Any]:
    """{'val': {...}, 'test': {...}, 'target_rule': {...}} z predykcji runa (źródłowego albo refitu)."""
    from src.evaluation.metrics import decide

    run_dir = Path(run_dir)
    y_v, p_v, f_v = load_predictions(run_dir, "val")
    vc_v = vc_mask(f_v)
    has_test = (run_dir / "predictions_test.parquet").exists()
    y_t = p_t = f_t = vc_t = None
    if has_test:
        y_t, p_t, f_t = load_predictions(run_dir, "test")
        vc_t = vc_mask(f_t)
    preds_v = {rule: decide(p_v, rule) for rule in ("argmax", "cost")}
    target_v, target_t, rule_info = _target_rule_preds(y_v, p_v, vc_v, p_t)
    if target_v is not None:
        preds_v["target"] = target_v
    out: dict[str, Any] = {
        "val": evaluate(y_v, p_v, preds_v, vc_v, admitted_for(f_v["row_id"].to_numpy(), ("val",))),
        "target_rule": rule_info,
    }
    if has_test:
        preds_t = {rule: decide(p_t, rule) for rule in ("argmax", "cost")}
        if target_t is not None:
            preds_t["target"] = target_t
        out["test"] = evaluate(y_t, p_t, preds_t, vc_t, admitted_for(f_t["row_id"].to_numpy(), ("test",)))
    return out


def plan_units(runs: Mapping[str, Any], seeds: Sequence[int], include_source: bool,
               vary_es_split: bool) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for model, run in runs.items():
        if include_source:
            units.append({"unit": f"{model}/source", "model": model, "source_run": run.name,
                          "seed": source_seed(run.info), "source": True})
        for seed in seeds:
            name = f"{model}/seed{int(seed)}" + ("es" if vary_es_split else "")
            units.append({"unit": name, "model": model, "source_run": run.name, "seed": int(seed),
                          "source": False})
    return units


def summarize(store: UnitStore, units: Sequence[Mapping[str, Any]], expect: Any) -> dict[str, Any]:
    """Średnia ± SD każdej metryki per model i split z gotowych jednostek."""
    models: dict[str, dict[str, Any]] = {}
    for u in units:
        payload = store.valid(str(u["unit"]), expect(u))
        if payload is None:
            continue
        entry = models.setdefault(u["model"], {"runs": [], "_val": [], "_test": []})
        entry["runs"].append({"unit": u["unit"], "seed": u["seed"], "source": bool(u["source"]),
                              "run": payload.get("run"), "duration_s": payload.get("duration_s")})
        entry["_val"].append(payload.get("val") or {})
        entry["_test"].append(payload.get("test") or {})
    out: dict[str, Any] = {}
    for model, entry in models.items():
        out[model] = {"runs": entry["runs"], "n_runs": len(entry["runs"])}
        for split in ("val", "test"):
            rows = entry[f"_{split}"]
            if any(rows):
                out[model][split] = {m: mean_sd([r.get(m) for r in rows]) for m in METRICS
                                     if any(r.get(m) is not None for r in rows)}
    return out


def summary_lines(results: Mapping[str, Any]) -> list[str]:
    lines = [
        f"# Seedy (C1) — {results['feature_set']} · tag {results['tag']}",
        "",
        f"- seedy refitów: {results['seeds']}" + (" + run źródłowy" if results["includes_source"] else "")
        + ("; zmieniany też podział early stopping" if results.get("vary_es_split") else ""),
        f"- jednostki: {results['n_done']}/{results['n_units']}"
        + (f", błędy: {len(results['failures'])}" if results["failures"] else ""),
        "- reguła `target`: " + str(results.get("target_rule_note")),
        "",
    ]
    for split in ("val", "test"):
        lines += [f"## {split} — średnia ± SD (n runów)", "",
                  "| model | n | " + " | ".join(SUMMARY_METRICS) + " |",
                  "|---|---:|" + "---:|" * len(SUMMARY_METRICS)]
        for model, entry in results["models"].items():
            stats = entry.get(split) or {}
            cells = []
            for m in SUMMARY_METRICS:
                s = stats.get(m) or {}
                if s.get("mean") is None:
                    cells.append("—")
                else:
                    cells.append(f"{s['mean']:.4f}" + (f" ± {s['sd']:.4f}" if s.get("sd") is not None else ""))
            lines.append(f"| {model} | {entry['n_runs']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("Wyższość modelu: CI Δ wyklucza 0 **i** |Δ| > minimalna istotna różnica **i** |Δ| > 2 × SD seedów.")
    return lines


def run_seed_refits(
    feature_set: str = "triage",
    tag: str | None = None,
    models: Sequence[str] | None = None,
    n_seeds: int = 3,
    include_source: bool = True,
    vary_es_split: bool = False,
    force: bool = False,
    save_models: bool = False,
) -> dict[str, Any]:
    """
    Refity na seedach 1…n_seeds dla najnowszych ukończonych runów `tag` (wznawialne) → results.json.
    Zwraca wynik (także `failures`: lista nieudanych jednostek).
    """
    from src.data.dataset import data_fingerprint
    from src.models.train import REFIT_VERSION, refit_from_run

    if n_seeds < 1:
        raise ValueError(f"n_seeds musi być ≥ 1, podano {n_seeds}")
    tag = tag or default_tag()
    runs = require_source_runs(feature_set, tag, models)
    store = UnitStore(analysis_dir("seeds", feature_set, tag))
    seeds = list(range(1, int(n_seeds) + 1))
    units = plan_units(runs, seeds, include_source, vary_es_split)
    fingerprint = data_fingerprint()
    store.write_plan({"analysis": "seeds", "feature_set": feature_set, "tag": tag, "seeds": seeds,
                      "include_source": include_source, "vary_es_split": vary_es_split,
                      "units": [{"unit": u["unit"], "source_run": u["source_run"]} for u in units]})

    def expect(u: Mapping[str, Any]) -> dict[str, Any]:
        return {"source_run": u["source_run"], "data_fingerprint": fingerprint, "metrics_version": METRICS_VERSION,
                "refit_version": REFIT_VERSION}

    def compute(u: Mapping[str, Any]) -> dict[str, Any]:
        run = runs[u["model"]]
        if u["source"]:
            run_dir = run.path
        else:
            refit_tag = seed_tag(tag, u["seed"], vary_es_split)
            run_dir = None if force else find_refit_run(run.name, refit_tag, str(u["unit"]))
            if run_dir is not None:
                log.info(f"{u['unit']}: gotowy refit {run_dir.name} — liczę tylko metryki")
            else:
                result = refit_from_run(run.path, seed=int(u["seed"]), tag=refit_tag,
                                        es_split_seed=int(u["seed"]) if vary_es_split else None,
                                        save_model=save_models, analysis="seeds", unit=str(u["unit"]))
                run_dir = Path(result["run_dir"])
        return {"model": u["model"], "seed": u["seed"], "source": u["source"], "run": run_dir.name,
                **unit_metrics(run_dir)}

    def estimate(u: Mapping[str, Any]) -> float:
        return 2.0 if u["source"] else float(runs[u["model"]].info.get("training_duration_s") or 0.0) * 1.2 + 5.0

    failures = execute_units(store, units, expect, compute, "seeds", force=force, estimate_s=estimate)
    models_out = summarize(store, units, expect)
    n_done = sum(len(m["runs"]) for m in models_out.values())
    results = {
        "analysis": "seeds",
        "feature_set": feature_set,
        "tag": tag,
        "created": now_iso(),
        "data_fingerprint": fingerprint,
        "seeds": seeds,
        "includes_source": include_source,
        "vary_es_split": vary_es_split,
        "source_runs": {m: r.name for m, r in runs.items()},
        "metrics": METRICS,
        "rules": list(RULES),
        "target_rule_note": (f"B1 + B2 (T = {config.DECISION_TARGET_UNDERTRIAGE}, populacja dopasowania "
                             f"'{config.DECISION_FIT_POPULATION}'): val — cross-fitting 2 foldy, test — reguła "
                             f"dopasowana na całym val tego seeda"
                             if decision_module() is not None else "niedostępna (brak src.evaluation.decision)"),
        "n_units": len(units),
        "n_done": n_done,
        "complete": not failures and n_done == len(units),
        "failures": failures,
        "models": models_out,
    }
    path = store.write_results(results)
    store.write_summary(summary_lines(results))
    log.info(f"Seedy: {n_done}/{len(units)} jednostek → {path}")
    return results


__all__ = ["METRICS", "RULES", "plan_units", "run_seed_refits", "seed_tag", "summarize", "unit_metrics"]
