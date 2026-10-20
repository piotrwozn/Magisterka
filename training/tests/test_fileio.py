"""
Testy zapisu plików odpornego na Windows (`src.utils.fileio`).

Blokady plików (antywirus, Excel, backend czytający model) symulowane podmianą os.replace;
przerwy między próbami wyzerowane, żeby testy trwały milisekundy.
"""

from __future__ import annotations

import json
import os

import pytest

from src.utils import fileio


@pytest.fixture
def no_sleep(monkeypatch):
    """Ponowienia bez czekania (sprawdzamy logikę, nie zegar)."""
    sleeps: list[float] = []
    monkeypatch.setattr(fileio.time, "sleep", sleeps.append)
    return sleeps


def _flaky_replace(monkeypatch, errors: list[BaseException]) -> list[tuple]:
    """os.replace rzucający kolejno podane błędy, potem działający normalnie."""
    real = os.replace
    calls: list[tuple] = []

    def replace(src, dst):
        calls.append((src, dst))
        if errors:
            raise errors.pop(0)
        return real(src, dst)

    monkeypatch.setattr(fileio.os, "replace", replace)
    return calls


def _winerror(code: int) -> OSError:
    exc = OSError(22, "plik używany przez inny proces")
    exc.winerror = code
    return exc


def test_delays_sum_to_total():
    delays = fileio._delays(8, 8.0)
    assert len(delays) == 7
    assert sum(delays) == pytest.approx(8.0)
    assert delays == sorted(delays)                       # rosnące przerwy
    assert fileio._delays(1, 8.0) == []


def test_write_text_atomic_utf8_and_no_tmp_left(tmp_path):
    path = tmp_path / "sub" / "plik.json"
    fileio.write_text_atomic(path, "zażółć gęślą jaźń")
    fileio.write_text_atomic(path, "nowa treść — ESI")
    assert path.read_text(encoding="utf-8") == "nowa treść — ESI"
    assert [p.name for p in path.parent.iterdir()] == ["plik.json"]


@pytest.mark.parametrize("error", [PermissionError(13, "odmowa dostępu"), _winerror(32), _winerror(5)])
def test_replace_retries_transient_lock(tmp_path, monkeypatch, no_sleep, error):
    src, dst = tmp_path / "a.tmp", tmp_path / "a.txt"
    src.write_text("x", encoding="utf-8")
    calls = _flaky_replace(monkeypatch, [error, error])
    assert fileio.replace_with_retry(src, dst) == dst
    assert dst.read_text(encoding="utf-8") == "x" and not src.exists()
    assert len(calls) == 3 and len(no_sleep) == 2


def test_replace_gives_up_after_attempts(tmp_path, monkeypatch, no_sleep):
    src, dst = tmp_path / "a.tmp", tmp_path / "a.txt"
    src.write_text("x", encoding="utf-8")
    calls = _flaky_replace(monkeypatch, [PermissionError(13, "zablokowany")] * 10)
    with pytest.raises(PermissionError):
        fileio.replace_with_retry(src, dst, attempts=4, total_s=1.0)
    assert len(calls) == 4 and sum(no_sleep) == pytest.approx(1.0)
    assert src.exists()                                   # sprząta wywołujący (write_text_atomic)


def test_non_transient_error_not_retried(tmp_path, monkeypatch, no_sleep):
    calls = _flaky_replace(monkeypatch, [FileNotFoundError(2, "brak katalogu"), _winerror(3)])
    with pytest.raises(FileNotFoundError):
        fileio.replace_with_retry(tmp_path / "a", tmp_path / "b")
    with pytest.raises(OSError):
        fileio.replace_with_retry(tmp_path / "a", tmp_path / "b")
    assert len(calls) == 2 and no_sleep == []


def test_write_text_atomic_cleans_tmp_on_failure(tmp_path, monkeypatch, no_sleep):
    path = tmp_path / "status.json"
    path.write_text("stara", encoding="utf-8")
    _flaky_replace(monkeypatch, [PermissionError(13, "zablokowany")] * 20)
    with pytest.raises(PermissionError):
        fileio.write_text_atomic(path, "nowa")
    assert path.read_text(encoding="utf-8") == "stara"   # czytelnik widzi starą, całą wersję
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_write_json_atomic(tmp_path):
    path = tmp_path / "x.json"
    fileio.write_json_atomic(path, {"opis": "gęś", "n": 1})
    assert "gęś" in path.read_text(encoding="utf-8")
    fileio.write_json_atomic(path, {"opis": "gęś"}, ensure_ascii=True)
    text = path.read_text(encoding="utf-8")
    assert text.isascii() and json.loads(text) == {"opis": "gęś"}
