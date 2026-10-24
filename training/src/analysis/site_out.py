"""
C3 — walidacja leave-site-out z korektą rozkładu klas (bez ponownego przygotowania danych).

Dla każdego wyłączonego oddziału S (domyślnie A i B; C ma tylko ~20 wizyt ESI 1) i modelu:
    - refit z najlepszymi parametrami runa źródłowego na wierszach TRAIN POZOSTAŁYCH oddziałów
      (próbka train_sample runa źródłowego, potem filtr oddziału),
    - ocena na wierszach TRAIN + VAL oddziału S (model ich nie widział; NIGDY test),
      z próbką eval_sample runa źródłowego,
    - warianty prawdopodobieństw: `raw`, `prior_known` (znany lokalny rozkład ESI oddziału S —
      w praktyce z rejestru triażu szpitala), `prior_em` (rozkład z EM bez etykiet, Saerens 2002),
    - reguła „undertriage ≤ T” (B1 + B2, src.evaluation.decision): `target_transfer` — dopasowana na
      VAL pozostałych oddziałów (walidacja rozwojowa tego modelu, populacja
      config.DECISION_FIT_POPULATION) i przeniesiona bez zmian; `target_local_*` — dopasowana na
      połowie wierszy oddziału S, wynik na drugiej połowie (lokalna rekalibracja z etykietami),
    - odniesienie: run źródłowy (trenowany także na S) na wierszach VAL oddziału S
      (`in_distribution`) i model przeniesiony na tych samych wierszach (`transported_same_rows`).

Metryki: QWK, niedotriażowanie, krytyczne pominięcia, czułość ESI 1 (reguły argmax, koszt,
target_transfer), log loss, makro-AUC, AUC przyjęcia (−E[ESI]), O/E ESI 1 i ESI 1–2 — pełny zbiór
i podgrupa `vc_`.

Wynik: runs/analyses/site_out__<cechy>__<tag>/results.json
    {"sites": {S: {model: {"raw": {...}, "prior_known": {...}, "prior_em": {...}, "n", "n_esi1", …}}}}

Uwaga do pracy: oddziały jednego amerykańskiego systemu to słabe przybliżenie przejścia USA → Polska;
przenosi się kierunek wniosku (dominuje przesunięcie rozkładu klas), nie liczby.
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
    now_iso,
    read_json,
    require_source_runs,
    rule_metrics,
    source_seed,
    vc_mask,
)
from src.analysis.priors import apply_prior_shift, class_prior, em_prior, normalize_prior
from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_MODELS: tuple[str, ...] = ("lightgbm", "xgboost", "catboost", "logreg")
DEFAULT_SITES: tuple[str, ...] = ("A", "B")
VARIANTS: tuple[str, ...] = ("raw", "prior_known", "prior_em")
HELDOUT, DEV_VAL = "heldout", "dev_val"
SUMMARY_METRICS: tuple[str, ...] = ("qwk_argmax", "undertriage_argmax", "qwk_target_transfer",
                                    "undertriage_target_transfer", "target_local_qwk",
                                    "target_local_undertriage", "log_loss", "admission_auc", "oe_esi1")


def site_tag(tag: str, site: str) -> str:
    return f"{tag}-siteout{site}"


def parse_sites(spec: str | Sequence[str]) -> list[str]:
    values = [s.strip().upper() for s in (spec.split(",") if isinstance(spec, str) else spec) if str(s).strip()]
    if not values:
        raise ValueError("Podaj co najmniej jeden oddział (np. A,B)")
    return list(dict.fromkeys(values))


# ─────────────────────────────────────────
# Ramki ewaluacji
# ─────────────────────────────────────────
class _Frames:
    """Wczytuje TRAIN i VAL (bez testu) raz na proces i buduje ramki dla wyłączonego oddziału."""

    def __init__(self, feature_set: str) -> None:
        self.feature_set = feature_set
        self._splits: dict[str, Any] | None = None

    def splits(self) -> dict[str, Any]:
        if self._splits is None:
            from src.data.dataset import load_splits

            self._splits = load_splits(("train", "val"), feature_set=self.feature_set)
            sites = sorted(set(self._splits["train"]["site"].astype(str)) | set(self._splits["val"]["site"].astype(str)))
            log.info(f"Oddziały w danych: {', '.join(sites)}")
        return self._splits

    def sites(self) -> set[str]:
        s = self.splits()
        return set(s["train"]["site"].astype(str)) | set(s["val"]["site"].astype(str))

    def for_site(self, site: str, eval_sample: int | None) -> dict[str, Any]:
        import pandas as pd

        from src.data.dataset import sample_frame

        s = self.splits()
        train, val = s["train"], s["val"]
        in_train = train["site"].astype(str) == site
        in_val = val["site"].astype(str) == site
        heldout = pd.concat([train.loc[in_train], val.loc[in_val]], ignore_index=True)
        dev_val = val.loc[~in_val].reset_index(drop=True)
        if heldout.empty or dev_val.empty:
            raise ValueError(f"Oddział '{site}': brak wierszy do oceny albo walidacji rozwojowej")
        return {DEV_VAL: sample_frame(dev_val, eval_sample).reset_index(drop=True),
                HELDOUT: sample_frame(heldout, eval_sample).reset_index(drop=True)}


# ─────────────────────────────────────────
# Metryki jednostki
# ─────────────────────────────────────────
def _oe(y: np.ndarray, proba: np.ndarray, classes: Sequence[int]) -> float:
    """Oczekiwane / obserwowane: Σ P(klasy) / liczba wizyt tych klas (O/E < 1 = model zaniża)."""
    observed = int(np.isin(y, classes).sum())
    return float(proba[:, list(classes)].sum() / observed) if observed else float("nan")


def _halves(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    try:
        a, b = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
    except ValueError:
        a, b = train_test_split(idx, test_size=0.5, random_state=seed)
    return np.sort(a), np.sort(b)


def unit_metrics(run_dir: Path | str, source_dir: Path | str, site: str) -> dict[str, Any]:
    """Metryki wariantów (raw / prior_known / prior_em) + reguła przeniesiona i lokalna + odniesienie."""
    from sklearn.metrics import roc_auc_score

    from src.evaluation.metrics import decide

    run_dir, source_dir = Path(run_dir), Path(source_dir)
    info = read_json(run_dir / "run_info.json") or {}
    counts = info.get("class_counts_train")
    if not counts:
        raise ValueError(f"{run_dir.name}: brak class_counts_train w run_info.json (refit sprzed analiz?)")
    train_prior = normalize_prior(counts)

    y, proba, frame = load_predictions(run_dir, HELDOUT)
    vc = vc_mask(frame)
    admitted = admitted_for(frame["row_id"].to_numpy(), ("train", "val"))
    site_prior = class_prior(y)
    prior_em, em_info = em_prior(proba, train_prior)
    variants = {
        "raw": proba,
        "prior_known": apply_prior_shift(proba, site_prior, train_prior),
        "prior_em": apply_prior_shift(proba, prior_em, train_prior),
    }

    decision = decision_module()
    fitter = None
    rule_dev = None
    if decision is not None:
        fitter = functools.partial(decision.fit_target_rule, target_undertriage=config.DECISION_TARGET_UNDERTRIAGE,
                                   safety_net=True)
        y_d, p_d, f_d = load_predictions(run_dir, DEV_VAL)
        try:
            rule_dev = fitter(y_d, p_d, mask=fit_population_mask(vc_mask(f_d)))
        except ValueError as exc:
            log.warning(f"Reguła 'target' na VAL pozostałych oddziałów pominięta: {exc}")
    half_a, half_b = _halves(y, config.RANDOM_SEED)

    out: dict[str, Any] = {}
    for name, p in variants.items():
        preds = {"argmax": decide(p, "argmax"), "cost": decide(p, "cost")}
        if rule_dev is not None:
            preds["target_transfer"] = rule_dev.predict(p)
        metrics = evaluate(y, p, preds, vc, admitted)
        metrics["oe_esi1"] = _oe(y, p, [0])
        metrics["oe_esi12"] = _oe(y, p, [0, 1])
        if name == "prior_em":
            metrics["em_suspect"] = bool(em_info["suspect"])     # EM niestabilne (zob. priors.em_prior)
        if fitter is not None:
            try:
                local = fitter(y[half_a], p[half_a], mask=fit_population_mask(None if vc is None else vc[half_a]))
                pred_b = local.predict(p[half_b])
                for key, value in rule_metrics(y[half_b], pred_b).items():
                    metrics[f"target_local_{key}"] = value
                if vc is not None and vc[half_b].any():
                    for key, value in rule_metrics(y[half_b][vc[half_b]], pred_b[vc[half_b]]).items():
                        metrics[f"vc_target_local_{key}"] = value
                metrics["target_local_n"] = int(len(half_b))
            except ValueError as exc:
                log.warning(f"Lokalna reguła 'target' ({name}) pominięta: {exc}")
        out[name] = metrics

    adm_ok = np.isfinite(admitted)
    nurse_auc = float("nan")
    if adm_ok.any() and len(np.unique(admitted[adm_ok])) == 2:
        nurse_auc = float(roc_auc_score(admitted[adm_ok].astype(int), -y[adm_ok]))

    # Odniesienie: run źródłowy (widział TRAIN oddziału S) na jego wierszach VAL oddziału S
    reference: dict[str, Any] = {}
    y_s, p_s, f_s = load_predictions(source_dir, "val")
    on_site = f_s["site"].astype(str).to_numpy() == site
    if on_site.any():
        rows = f_s["row_id"].to_numpy()[on_site]
        adm_s = admitted_for(rows, ("val",))
        vc_s = vc_mask(f_s)
        vc_s = None if vc_s is None else vc_s[on_site]
        ys, ps = y_s[on_site], p_s[on_site]
        reference["in_distribution"] = evaluate(ys, ps, {"argmax": decide(ps, "argmax"), "cost": decide(ps, "cost")},
                                                vc_s, adm_s)
        position = {rid: i for i, rid in enumerate(frame["row_id"].to_numpy())}
        common = [(j, position[rid]) for j, rid in enumerate(rows) if rid in position]
        if common:
            js = np.array([c[0] for c in common])
            hs = np.array([c[1] for c in common])
            p_same = proba[hs]
            reference["in_distribution_same_rows"] = evaluate(
                ys[js], ps[js], {"argmax": decide(ps[js], "argmax"), "cost": decide(ps[js], "cost")},
                None if vc_s is None else vc_s[js], adm_s[js])
            reference["transported_same_rows"] = evaluate(
                y[hs], p_same, {"argmax": decide(p_same, "argmax"), "cost": decide(p_same, "cost")},
                None if vc is None else vc[hs], admitted[hs])

    return {
        **out,
        **reference,
        "n": int(len(y)),
        "n_esi1": int((y == 0).sum()),
        "n_vc": None if vc is None else int(vc.sum()),
        "n_dev_val": int(info.get(f"n_{DEV_VAL}") or 0),
        "n_fit": info.get("n_fit"),
        "nurse_admission_auc": nurse_auc,
        "priors": {"train": train_prior.tolist(), "site": site_prior.tolist(), "em": prior_em.tolist(),
                   "em_info": em_info},
        "target_rule_dev": None if rule_dev is None else rule_dev.to_dict(),
    }


# ─────────────────────────────────────────
# Analiza
# ─────────────────────────────────────────
def summary_lines(results: Mapping[str, Any]) -> list[str]:
    lines = [f"# Leave-site-out (C3) — {results['feature_set']} · tag {results['tag']}", "",
             f"- oddziały wyłączone: {', '.join(results['sites_requested'])}; jednostki "
             f"{results['n_done']}/{results['n_units']}"
             + (f", błędy: {len(results['failures'])}" if results["failures"] else ""),
             "- ocena: TRAIN + VAL wyłączonego oddziału (nigdy test); `target_transfer` — reguła z VAL "
             "pozostałych oddziałów; `target_local` — dopasowana na połowie oddziału, wynik na drugiej",
             "- ⚠ EM niestabilne: EM wyzerowało klasę (< 0,1 × udział w treningu) albo zawyżyło ją > 20× "
             "(zwykle model nieskalibrowany) — tego wariantu nie interpretuj", ""]
    for site, models in results["sites"].items():
        lines += [f"## Oddział {site} wyłączony", "",
                  "| model | wariant | n | " + " | ".join(SUMMARY_METRICS) + " |",
                  "|---|---|---:|" + "---:|" * len(SUMMARY_METRICS)]
        for model, entry in models.items():
            for variant in (*VARIANTS, "in_distribution"):
                m = entry.get(variant)
                if not m:
                    continue
                cells = ["—" if m.get(k) is None else f"{m[k]:.4f}" for k in SUMMARY_METRICS]
                label = variant + (" ⚠ EM niestabilne" if m.get("em_suspect") else "")
                lines.append(f"| {model} | {label} | {m.get('n', entry.get('n'))} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def run_leave_site_out(
    feature_set: str = "triage",
    tag: str | None = None,
    models: Sequence[str] | None = DEFAULT_MODELS,
    sites: Sequence[str] = DEFAULT_SITES,
    force: bool = False,
) -> dict[str, Any]:
    """Refity bez wyłączonego oddziału + ocena na nim (wznawialne) → results.json, summary.md."""
    from src.data.dataset import data_fingerprint
    from src.models.train import REFIT_VERSION, refit_from_run

    tag = tag or default_tag()
    sites = parse_sites(sites)
    runs = require_source_runs(feature_set, tag, models)
    store = UnitStore(analysis_dir("site_out", feature_set, tag))
    fingerprint = data_fingerprint()
    units = [{"unit": f"{site}/{model}", "site": site, "model": model, "source_run": run.name,
              "seed": source_seed(run.info)} for site in sites for model, run in runs.items()]
    store.write_plan({"analysis": "site_out", "feature_set": feature_set, "tag": tag, "sites": sites,
                      "units": [{"unit": u["unit"], "source_run": u["source_run"]} for u in units]})
    frames = _Frames(feature_set)
    frame_cache: dict[tuple[str, Any], dict[str, Any]] = {}

    def expect(u: Mapping[str, Any]) -> dict[str, Any]:
        return {"source_run": u["source_run"], "data_fingerprint": fingerprint, "metrics_version": METRICS_VERSION,
                "refit_version": REFIT_VERSION}

    def compute(u: Mapping[str, Any]) -> dict[str, Any]:
        run = runs[u["model"]]
        site = str(u["site"])
        refit_tag = site_tag(tag, site)
        run_dir = None if force else find_refit_run(run.name, refit_tag, str(u["unit"]))
        if run_dir is not None:
            log.info(f"{u['unit']}: gotowy refit {run_dir.name} — liczę tylko metryki")
        else:
            if site not in frames.sites():
                raise ValueError(f"Oddziału '{site}' nie ma w danych (dostępne: {', '.join(sorted(frames.sites()))})")
            key = (site, run.info.get("eval_sample"))
            if key not in frame_cache:
                frame_cache.clear()                      # jedna para ramek naraz (RAM)
                frame_cache[key] = frames.for_site(site, run.info.get("eval_sample"))
            result = refit_from_run(
                run.path, seed=int(u["seed"]), tag=refit_tag,
                train_rows_mask=lambda df, s=site: df["site"].astype(str).to_numpy() != s,
                train_rows_label=f"oddziały ≠ {site}", eval_frames=frame_cache[key],
                analysis="site_out", unit=str(u["unit"]),
            )
            run_dir = Path(result["run_dir"])
        return {"model": u["model"], "site": site, "run": run_dir.name, **unit_metrics(run_dir, run.path, site)}

    def estimate(u: Mapping[str, Any]) -> float:
        return float(runs[u["model"]].info.get("training_duration_s") or 0.0) + 10.0

    failures = execute_units(store, units, expect, compute, "site_out", force=force, estimate_s=estimate)

    sites_out: dict[str, dict[str, Any]] = {}
    n_done = 0
    for u in units:
        payload = store.valid(str(u["unit"]), expect(u))
        if payload is None:
            continue
        n_done += 1
        entry = {k: v for k, v in payload.items() if k not in ("unit", "status", "metrics_version", "site", "model")}
        sites_out.setdefault(str(u["site"]), {})[u["model"]] = entry
    results = {
        "analysis": "site_out",
        "feature_set": feature_set,
        "tag": tag,
        "created": now_iso(),
        "data_fingerprint": fingerprint,
        "sites_requested": sites,
        "source_runs": {m: r.name for m, r in runs.items()},
        "variants": list(VARIANTS),
        "target_rule_note": (f"B1 + B2 (T = {config.DECISION_TARGET_UNDERTRIAGE}, populacja "
                             f"'{config.DECISION_FIT_POPULATION}')" if decision_module() is not None
                             else "niedostępna (brak src.evaluation.decision)"),
        "n_units": len(units),
        "n_done": n_done,
        "complete": not failures and n_done == len(units),
        "failures": failures,
        "sites": sites_out,
    }
    path = store.write_results(results)
    store.write_summary(summary_lines(results))
    log.info(f"Leave-site-out: {n_done}/{len(units)} jednostek → {path}")
    return results


__all__ = ["DEFAULT_MODELS", "DEFAULT_SITES", "VARIANTS", "parse_sites", "run_leave_site_out", "unit_metrics"]
