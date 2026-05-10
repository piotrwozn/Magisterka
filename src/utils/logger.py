"""
Wspólne logowanie z `rich` dla kolorowego, czytelnego outputu.

Dostępne loggery:
    - `get_logger(__name__)`   → ogólny logger (konsola + opcjonalnie plik)
    - `get_training_logger()`  → dedykowany logger treningu modeli (zawsze do pliku)

Użycie:
    from src.utils.logger import get_logger, get_training_logger

    log = get_logger(__name__)
    log.info("Trening rozpoczęty")

    # W treningu:
    train_log, log_path = get_training_logger("xgboost", run_id="20250510_143000")
    train_log.info(f"Iteracja 100: mlogloss=0.452")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    from rich.logging import RichHandler

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ─────────────────────────────────────────
# Formattery
# ─────────────────────────────────────────
_CONSOLE_FORMATTER = logging.Formatter("%(message)s", datefmt="[%X]")
_FILE_FORMATTER = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ─────────────────────────────────────────
# Logger ogólnego przeznaczenia
# ─────────────────────────────────────────
def get_logger(
    name: str = "sor_ai",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """
    Zwraca skonfigurowany logger (kolory na konsoli + opcjonalnie plik).

    Parameters
    ----------
    name : str
        Nazwa loggera (zwykle `__name__` modułu wywołującego).
    level : int
        Poziom logowania (DEBUG, INFO, WARNING, ERROR).
    log_file : Path, optional
        Jeśli podany — dopisuje równolegle logi do pliku.
    """
    logger = logging.getLogger(name)

    # Nie konfiguruj wielokrotnie tego samego loggera
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Konsola
    if _RICH_AVAILABLE:
        console_handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
        console_handler.setFormatter(_CONSOLE_FORMATTER)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_FILE_FORMATTER)

    logger.addHandler(console_handler)

    # Plik (opcjonalnie)
    if log_file is not None:
        _attach_file_handler(logger, log_file)

    return logger


def _attach_file_handler(logger: logging.Logger, log_file: Path) -> None:
    """Dodaje FileHandler do loggera (pomija jeśli już istnieje)."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    abs_path = log_file.resolve()
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == abs_path:
            return  # już dołączony

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(_FILE_FORMATTER)
    logger.addHandler(file_handler)


# ─────────────────────────────────────────
# Dedykowany logger treningu modeli
# ─────────────────────────────────────────
def get_training_logger(
    model_name: str,
    run_id: str | None = None,
    log_dir: Path | None = None,
    level: int = logging.DEBUG,
) -> tuple[logging.Logger, Path]:
    """
    Logger dedykowany dla treningu modelu — zawsze zapisuje do pliku
    `logs/training/<model_name>/<run_id>.log`.

    Parameters
    ----------
    model_name : str
        np. 'xgboost', 'lightgbm', 'random_forest'.
    run_id : str, optional
        Identyfikator runa (timestamp). Domyślnie: bieżąca data/godzina.
    log_dir : Path, optional
        Katalog na logi (domyślnie config.TRAINING_LOGS_DIR).
    level : int
        Poziom logowania.

    Returns
    -------
    logger : logging.Logger
    log_path : Path
        Ścieżka do pliku log dla tego runa.
    """
    # Lokalny import (przez moduł `config`, by monkeypatch w testach działał poprawnie)
    if log_dir is None:
        from src.utils import config as _config
        log_dir = _config.TRAINING_LOGS_DIR

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_log_dir = Path(log_dir) / model_name
    model_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_log_dir / f"{run_id}.log"

    logger_name = f"sor_ai.training.{model_name}.{run_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Konsola
    if not any(isinstance(h, (logging.StreamHandler, RichHandler)) for h in logger.handlers if not isinstance(h, logging.FileHandler)):
        if _RICH_AVAILABLE:
            console_handler: logging.Handler = RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_path=False,
            )
            console_handler.setFormatter(_CONSOLE_FORMATTER)
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(_FILE_FORMATTER)
        logger.addHandler(console_handler)

    # Plik (zawsze)
    _attach_file_handler(logger, log_path)

    logger.info(f"=== Logger treningu dla '{model_name}' ===")
    logger.info(f"Run ID:   {run_id}")
    logger.info(f"Plik log: {log_path}")

    return logger, log_path
