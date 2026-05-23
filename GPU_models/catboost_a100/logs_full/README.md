# Logi i metadane eksperymentów

Ten katalog gromadzi wszystkie logi z treningu, ewaluacji i metadane eksperymentów.

## Struktura

```
logs/
├── training/                  # Logi tekstowe z treningu (per model + run_id)
│   ├── xgboost/
│   │   ├── 20250510_143000.log
│   │   └── 20250510_154500.log
│   ├── lightgbm/
│   ├── random_forest/
│   ├── ebm/
│   ├── stacking/
│   └── xgboost_tuning/        # Logi z Optuna tuning
│
├── evaluation/                # Logi z ewaluacji (CV, test set)
│   └── cv_stratified/
│       └── 20250510_160000.log
│
└── experiments/               # JSON-y z pełnymi metadanymi eksperymentów
    ├── xgboost_20250510_143000.json
    ├── lightgbm_20250510_154500.json
    └── ...
```

## Plik `.log` — co zawiera

Każdy run treningu ma dedykowany plik tekstowy z formatem:

```
2025-05-10 14:30:00 | INFO | sor_ai.training.xgboost.20250510_143000 | === Logger treningu dla 'xgboost' ===
2025-05-10 14:30:00 | INFO | ...                                     | Run ID:   20250510_143000
2025-05-10 14:30:00 | INFO | ...                                     | Plik log: logs/training/xgboost/20250510_143000.log
2025-05-10 14:30:00 | INFO | ...                                     | Model:    xgboost
2025-05-10 14:30:00 | INFO | ...                                     | Params:   {...}
2025-05-10 14:30:00 | INFO | ...                                     | Dane treningowe:  n=448,000, features=210
2025-05-10 14:30:00 | INFO | ...                                     | Dane walidacyjne: n=56,000
2025-05-10 14:30:01 | INFO | ...                                     | Trenowanie XGBoost (n=448,000, features=210)
2025-05-10 14:30:01 | INFO | ...                                     | Wagi sampli: strategia='custom', min=1.00, max=10.00
2025-05-10 14:30:05 | INFO | ...                                     | Iter     0 | train_mlogloss=1.39521 | val_mlogloss=1.40123
2025-05-10 14:32:10 | INFO | ...                                     | Iter   100 | train_mlogloss=0.45231 | val_mlogloss=0.48721
...
2025-05-10 14:45:00 | INFO | ...                                     | Trening XGBoost zakończony w 894.3s
2025-05-10 14:45:00 | INFO | ...                                     | Top 5 cech (importance):
2025-05-10 14:45:00 | INFO | ...                                     |   esi                    0.4523
2025-05-10 14:45:00 | INFO | ...                                     |   triage_o2sat           0.0832
...
```

## Plik `.json` (experiments/) — co zawiera

Pełne metadane eksperymentu:

```json
{
  "run_id": "20250510_143000",
  "model_name": "xgboost",
  "timestamp_start": "2025-05-10T14:30:00.123",
  "timestamp_end": "2025-05-10T14:45:00.789",
  "duration_seconds": 894.3,
  "environment": {
    "hostname": "MyComputer",
    "platform": "Linux-6.x.x...",
    "python_version": "3.11.5",
    "git_commit": "a3f2b1c"
  },
  "params": {
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    ...
  },
  "data": {
    "feature_set": "triage_only",
    "n_features": 210,
    "feature_names": [...],
    "n_train": 448000,
    "n_val": 56000,
    "n_test": 56000,
    "class_distribution_train": {"0": 0.018, "1": 0.13, "2": 0.49, ...},
    ...
  },
  "training_history": [
    {"iteration": 0, "train_mlogloss": 1.395, "val_mlogloss": 1.401, ...},
    {"iteration": 1, "train_mlogloss": 1.302, "val_mlogloss": 1.310, ...},
    ...
  ],
  "metrics": {
    "best_iteration": 743,
    "test_quadratic_weighted_kappa": 0.872,
    "test_undertriage_rate": 0.023,
    "test_critical_miss_rate": 0.001,
    "test_auc_macro": 0.945,
    ...
  },
  "feature_importances": [
    {"feature": "esi", "importance": 0.4523},
    {"feature": "triage_o2sat", "importance": 0.0832},
    ...
  ],
  "artifacts": {
    "model_file": "/.../models/xgboost_20250510_143000.joblib",
    "training_log": "/.../logs/training/xgboost/20250510_143000.log"
  },
  "notes": [
    "Early stopping na iter 743"
  ]
}
```

## Dlaczego to ważne dla pracy magisterskiej?

1. **Reprodukowalność** — każdy run ma snapshot środowiska (git_commit, python_version, hostname).
2. **Dokumentacja** — każdy eksperyment przechowuje pełne hiperparametry, dane, metryki.
3. **Porównanie modeli** — możesz wczytać wszystkie JSON-y i zrobić tabelę porównawczą.
4. **Wykrywanie problemów** — w razie regresji widać dokładnie kiedy i jak.

## Przykładowy skrypt do agregacji wyników

```python
import json
import pandas as pd
from pathlib import Path

experiments = []
for json_file in Path("logs/experiments").glob("*.json"):
    with open(json_file) as f:
        data = json.load(f)
    experiments.append({
        "run_id": data["run_id"],
        "model": data["model_name"],
        "duration_s": data["duration_seconds"],
        **data.get("metrics", {}),
    })

df = pd.DataFrame(experiments)
df = df.sort_values("test_quadratic_weighted_kappa", ascending=False)
print(df[["model", "test_quadratic_weighted_kappa", "test_undertriage_rate"]].head())
```

## MLflow (opcjonalnie)

Jeśli uruchomisz trening z `--mlflow`, wyniki będą równolegle logowane do `mlruns/`.
Wizualizuj w UI:
```bash
mlflow ui --backend-store-uri file://$PWD/mlruns
```
