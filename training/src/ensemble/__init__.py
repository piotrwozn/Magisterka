"""
Narzędzia do RĘCZNEGO łączenia modeli na predykcjach zapisanych w runs/.

    from src.ensemble import latest_runs, aligned_probas, blend, evaluate, diversity

    runs = latest_runs("triage", tag="standard")        # najnowszy ukończony run każdego modelu
    y_val, P_val, vc = aligned_probas(runs, "val", return_vc=True)   # dict model → (n, 5)
    p = blend(P_val, {"xgboost": 0.5, "catboost": 0.3, "ebm": 0.2})
    evaluate(y_val, p, rule="cost")
    evaluate(y_val[vc], p[vc], rule="cost")             # podgrupa z kompletem HR/SBP/RR

Protokół: wszystko (wagi, kalibracja, wybór modeli) dobieraj na `val`,
raportuj na `test` dokładnie tą samą recepturą — raz, na końcu.
Runy z wagami klas (prawdopodobieństwa nieskalibrowane) i runy bez tuningu są domyślnie
pomijane (`latest_runs(..., include_weighted=True, require_tuned=False)` je uwzględni).
"""

from src.ensemble.runs import (
    Run,
    aligned_probas,
    apply_temperature,
    blend,
    diversity,
    evaluate,
    fit_temperature,
    latest_runs,
    list_runs,
    load_run,
    optimize_weights,
    resolve_feature_set,
)

__all__ = [
    "Run",
    "aligned_probas",
    "apply_temperature",
    "blend",
    "diversity",
    "evaluate",
    "fit_temperature",
    "latest_runs",
    "list_runs",
    "load_run",
    "optimize_weights",
    "resolve_feature_set",
]
