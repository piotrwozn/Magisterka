"""
Wspólne narzędzia: konfiguracja, logowanie, sprzęt, zapis plików, tracking eksperymentów.

Importy są LENIWE (PEP 562 `__getattr__`): `import src.utils.config` / `src.utils.hardware`
nie ładuje numpy ani pandas (tylko `ExperimentTracker` ich potrzebuje). Dzięki temu preflight
(`scripts/00_check_env.py`) działa w środowisku bez pakietów i może zgłosić, czego brakuje.
"""

from __future__ import annotations

import importlib
from typing import Any

# Nazwa eksportowana → moduł, z którego jest ładowana przy pierwszym użyciu
_CONFIG_NAMES = (
    "CLASS_NAMES",
    "CLASS_NAMES_PL",
    "CUSTOM_CLASS_WEIGHTS",
    "DATA_PROCESSED_DIR",
    "DATA_RAW_DIR",
    "EVAL_LOGS_DIR",
    "EXPERIMENTS_DIR",
    "LOGS_DIR",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_TRACKING_URI",
    "MODELS_DIR",
    "PROJECT_ROOT",
    "RANDOM_SEED",
    "RESULTS_DIR",
    "RUNS_DIR",
    "TOPSOR_COLORS",
    "TRAINING_LOGS_DIR",
)
_LAZY_NAMES: dict[str, str] = {
    **{name: "src.utils.config" for name in _CONFIG_NAMES},
    "ExperimentTracker": "src.utils.experiment_tracker",
    "close_training_logger": "src.utils.logger",
    "get_logger": "src.utils.logger",
    "get_training_logger": "src.utils.logger",
}
_SUBMODULES = frozenset({"config", "experiment_tracker", "fileio", "hardware", "logger"})

__all__ = sorted(_LAZY_NAMES)


def __getattr__(name: str) -> Any:
    """Leniwy import nazw z `__all__` i podmodułów (`src.utils.config` itd.)."""
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    module = _LAZY_NAMES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
