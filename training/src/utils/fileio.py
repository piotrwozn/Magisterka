"""
Zapis plików odporny na przerwanie i na Windows (antywirus, indeksowanie, Excel trzymają pliki).

    replace_with_retry(src, dst)    — os.replace z ponowieniami (łącznie ≈ 8 s)
    write_text_atomic(path, text)   — tekst UTF-8 przez plik tymczasowy + replace_with_retry
    write_json_atomic(path, obj)    — JSON przez write_text_atomic

Czytelnik nigdy nie widzi połowy pliku: albo stara wersja, albo nowa.
Moduł używa wyłącznie biblioteki standardowej (preflight działa nawet bez numpy).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

# Windows: ERROR_ACCESS_DENIED (5), ERROR_SHARING_VIOLATION (32) — plik chwilowo otwarty
_RETRY_WINERRORS: frozenset[int] = frozenset({5, 32})
REPLACE_ATTEMPTS: int = 8
REPLACE_TOTAL_S: float = 8.0


def _retryable(exc: OSError) -> bool:
    """Błąd chwilowej blokady pliku (warto ponowić), a nie np. brak katalogu."""
    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in _RETRY_WINERRORS


def _delays(attempts: int, total_s: float) -> list[float]:
    """Rosnące przerwy między próbami (attempts − 1 przerw) sumujące się do total_s."""
    n = max(0, attempts - 1)
    weight = n * (n + 1) / 2
    return [total_s * (i + 1) / weight for i in range(n)] if n else []


def replace_with_retry(
    src: Path | str,
    dst: Path | str,
    attempts: int = REPLACE_ATTEMPTS,
    total_s: float = REPLACE_TOTAL_S,
) -> Path:
    """
    os.replace(src, dst) z ponowieniami, gdy plik docelowy jest chwilowo zablokowany
    (PermissionError albo OSError z winerror 5 / 32). Przerwy rosną, łącznie ≈ total_s.

    Inne błędy i ostatnia nieudana próba → wyjątek (src zostaje — sprząta wywołujący).
    """
    src, dst = Path(src), Path(dst)
    delays = _delays(max(1, int(attempts)), float(total_s))
    for attempt in range(max(1, int(attempts))):
        try:
            os.replace(src, dst)
            return dst
        except OSError as exc:
            if not _retryable(exc) or attempt >= len(delays):
                raise
            time.sleep(delays[attempt])
    return dst


def write_text_atomic(path: Path | str, text: str) -> Path:
    """
    Tekst UTF-8 przez plik tymczasowy w tym samym katalogu + replace_with_retry.

    Przerwany zapis nie zostawia połowy pliku; plik tymczasowy jest zawsze sprzątany
    (best-effort — błąd sprzątania nie przykrywa właściwego błędu).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        replace_with_retry(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return path


def write_json_atomic(path: Path | str, payload: Any, *, indent: int = 2, ensure_ascii: bool = False,
                      default: Any = str) -> Path:
    """JSON (UTF-8) przez write_text_atomic."""
    return write_text_atomic(path, json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii, default=default))


__all__ = ["replace_with_retry", "write_json_atomic", "write_text_atomic"]
