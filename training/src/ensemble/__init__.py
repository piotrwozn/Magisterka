"""
Narzędzia do RĘCZNEGO łączenia modeli na predykcjach zapisanych w runs/.

    from src.ensemble import latest_runs, aligned_probas, blend, evaluate, diversity

    runs = latest_runs("triage")                        # najnowszy run każdego modelu
    y_val, P_val = aligned_probas(runs, "val")          # dict model → (n, 5)
    p = blend(P_val, {"xgboost": 0.5, "catboost": 0.3, "ebm": 0.2})
    evaluate(y_val, p, rule="cost")

Protokół: wszystko (wagi, kalibracja, wybór modeli) dobieraj na `val`,
raportuj na `test` dokładnie tą samą recepturą.
"""

from src.ensemble.runs import (
    aligned_probas,
    blend,
    diversity,
    evaluate,
    fit_temperature,
    latest_runs,
    list_runs,
    load_run,
    optimize_weights,
    apply_temperature,
)

__all__ = [
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
]
