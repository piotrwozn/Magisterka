"""
Wspólne logowanie: konsola (rich w terminalu, zwykły tekst w pliku/potoku) + pliki UTF-8.

Dostępne loggery:
    - `get_logger(__name__)`   → ogólny logger (konsola + opcjonalnie plik)
    - `get_training_logger()`  → dedykowany logger treningu modeli (zawsze do pliku)
    - `close_training_logger()` → zamyka pliki logu treningu (Windows blokuje otwarte pliki)

Import modułu przełącza stdout/stderr na UTF-8: na Windows przekierowany wynik
(`> log.txt`, potok, podproces) używa strony kodowej (cp1250/cp1252), w której nie ma
znaków `═ → ≥ ✓` — bez tego logi giną (`--- Logging error ---`), a `print()` kończy proces.

Użycie:
    from src.utils.logger import get_logger, get_training_logger, close_training_logger

    log = get_logger(__name__)
    log.info("Trening rozpoczęty")

    train_log, log_path = get_training_logger("xgboost", run_id="20250510_143000")
    train_log.info("Iteracja 100: mlogloss=0.452")
    close_training_logger(train_log)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    from rich.logging import RichHandler

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover — rich jest w zależnościach
    RichHandler = None  # type: ignore[assignment,misc]
    _RICH_AVAILABLE = False


# ─────────────────────────────────────────
# UTF-8 na stdout/stderr (Windows: przekierowanie = cp1250)
# ─────────────────────────────────────────
def _force_utf8_stdio() -> None:
    """Przełącza sys.stdout / sys.stderr na UTF-8 (niekodowalne znaki → \\uXXXX zamiast błędu)."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "").replace("_", "")
        if encoding == "utf8":
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError, AttributeError):
            pass


_force_utf8_stdio()


# ─────────────────────────────────────────
# Formattery i handlery
# ─────────────────────────────────────────
_CONSOLE_FORMATTER = logging.Formatter("%(message)s", datefmt="[%X]")
_PLAIN_CONSOLE_FORMATTER = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
_FILE_FORMATTER = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class _StdoutHandler(logging.StreamHandler):
    """StreamHandler piszący do BIEŻĄCEGO sys.stdout (po reconfigure, przy pytest capture)."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property  # type: ignore[override]
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:  # StreamHandler.__init__ przypisuje stream — ignorujemy
        pass


def _stdout_is_tty() -> bool:
    try:
        return bool(sys.stdout is not None and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _make_console_handler() -> logging.Handler:
    """
    Terminal → RichHandler (kolory). Plik/potok → zwykły tekst: rich łamie wiersze na 80
    kolumnach, co czyni logi nocnego runu nieczytelnymi.
    """
    if _RICH_AVAILABLE and _stdout_is_tty():
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            markup=False,          # nawiasy [ ] w komunikatach to dane, nie znaczniki rich
            show_time=True,
            show_path=False,
        )
        handler.setFormatter(_CONSOLE_FORMATTER)
    else:
        handler = _StdoutHandler()
        handler.setFormatter(_PLAIN_CONSOLE_FORMATTER)
    return handler


def _is_console_handler(handler: logging.Handler) -> bool:
    if isinstance(handler, logging.FileHandler):
        return False
    if _RICH_AVAILABLE and isinstance(handler, RichHandler):
        return True
    return isinstance(handler, logging.StreamHandler)


# ─────────────────────────────────────────
# Logger ogólnego przeznaczenia
# ─────────────────────────────────────────
def get_logger(
    name: str = "sor_ai",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """
    Zwraca skonfigurowany logger (konsola + opcjonalnie plik UTF-8).

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
        if log_file is not None:
            _attach_file_handler(logger, log_file)
        return logger

    logger.setLevel(level)
    logger.propagate = False
    logger.addHandler(_make_console_handler())

    if log_file is not None:
        _attach_file_handler(logger, log_file)

    return logger


def _attach_file_handler(logger: logging.Logger, log_file: Path) -> None:
    """
    Dodaje FileHandler UTF-8 (pomija, jeśli ten plik już jest podpięty).

    Bez fsync po każdym wpisie — FileHandler i tak opróżnia bufor po każdym rekordzie,
    a fsync na Windows (z Defenderem) kosztuje milisekundy na linię.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    abs_path = log_file.resolve()
    for h in logger.handlers:
        base = getattr(h, "baseFilename", None)
        if base is not None and Path(base).resolve() == abs_path:
            return  # już dołączony

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
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
    `logs/training/<model_name>/<run_id>.log` (UTF-8). Po treningu zamknij go
    przez `close_training_logger(logger)`.

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

    # Konsola (raz) — poziom INFO, by szczegóły DEBUG trafiały tylko do pliku
    if not any(_is_console_handler(h) for h in logger.handlers):
        console_handler = _make_console_handler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    # Plik (zawsze)
    _attach_file_handler(logger, log_path)

    logger.info(f"=== Logger treningu dla '{model_name}' ===")
    logger.info(f"Run ID:   {run_id}")
    logger.info(f"Plik log: {log_path}")

    return logger, log_path


def close_training_logger(logger: logging.Logger | None) -> None:
    """
    Zamyka i odpina wszystkie handlery loggera treningu.

    Na Windows otwarty plik logu jest zablokowany do końca procesu (nie da się go
    przenieść/usunąć) — dlatego każdy fit() zamyka swój logger po treningu.
    """
    if logger is None:
        return
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:  # noqa: BLE001 — zamykanie nie może przerwać treningu
            pass
        logger.removeHandler(handler)
