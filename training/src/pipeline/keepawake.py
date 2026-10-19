"""
Blokada uśpienia komputera na czas nocnego przebiegu (`scripts/run_all.py`).

    with KeepAwake() as awake:
        print(awake.description)       # co faktycznie zadziałało
        ...                            # długi trening

Platformy:
    Windows — SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) w WĄTKU GŁÓWNYM
              (flaga dotyczy wątku, który ją ustawił — wątek musi żyć przez cały przebieg).
              Monitor może się wyłączyć; system nie zaśnie. Sprawdzenie: `powercfg /requests`.
    Linux   — `systemd-inhibit --what=idle:sleep … tail --pid=<PID> -f /dev/null`
              (blokada znika razem z procesem orkiestratora, także po awarii); bez systemd — nic.
    macOS   — `caffeinate -i -w <PID>`.

Nie chroni przed: zamknięciem pokrywy laptopa, ręcznym uśpieniem, restartem po Windows Update
(wstrzymaj aktualizacje na noc; wznowienie przebiegu i tak obsługuje restart).
Błędy nigdy nie przerywają treningu — `description` mówi, co się udało.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

# Stałe Win32 (winbase.h)
ES_CONTINUOUS: int = 0x80000000
ES_SYSTEM_REQUIRED: int = 0x00000001

_HELPER_START_CHECK_S: float = 0.3   # tyle czekamy, czy proces pomocniczy nie padł od razu


class KeepAwake:
    """
    Kontekst blokujący uśpienie systemu.

    Parameters
    ----------
    enabled : False → nic nie robi (`--no-keep-awake`).
    reason : opis widoczny w `systemd-inhibit --list` / `powercfg /requests`.
    platform : nadpisanie `sys.platform` (testy).
    kernel32 : obiekt z metodą SetThreadExecutionState (testy; domyślnie ctypes.windll.kernel32).
    """

    def __init__(
        self,
        enabled: bool = True,
        reason: str = "SOR-AI: nocny trening modeli",
        platform: str | None = None,
        kernel32: Any | None = None,
    ) -> None:
        self.enabled = enabled
        self.reason = reason
        self.platform = platform or sys.platform
        self._kernel32 = kernel32
        self._proc: subprocess.Popen | None = None
        self._windows_set = False
        self.active = False
        self.description = "wyłączone (--no-keep-awake)" if not enabled else "nieuruchomione"

    # ── kontekst ──
    def __enter__(self) -> KeepAwake:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ── start / stop ──
    def start(self) -> None:
        if not self.enabled:
            return
        try:
            if self.platform == "win32":
                self._start_windows()
            elif self.platform == "darwin":
                self._start_helper(["caffeinate", "-i", "-w", str(os.getpid())], "caffeinate -i")
            elif self.platform.startswith("linux"):
                inhibit, tail = shutil.which("systemd-inhibit"), shutil.which("tail")
                if inhibit and tail:
                    self._start_helper(
                        [inhibit, "--what=idle:sleep", "--who=SOR-AI", f"--why={self.reason}", "--mode=block",
                         tail, f"--pid={os.getpid()}", "-f", "/dev/null"],
                        "systemd-inhibit (idle:sleep)",
                    )
                else:
                    self.description = "brak (systemd-inhibit niedostępny) — wyłącz usypianie ręcznie"
            else:
                self.description = f"brak (platforma {self.platform})"
        except Exception as exc:  # noqa: BLE001 — blokada uśpienia nie może przerwać treningu
            self.active = False
            self.description = f"nie udało się: {type(exc).__name__}: {exc}"
        log.debug(f"Blokada uśpienia: {self.description}")   # komunikat dla użytkownika: orkiestrator

    def stop(self) -> None:
        if self._windows_set:
            try:
                self._win_kernel32().SetThreadExecutionState(ES_CONTINUOUS)   # przywróć normalne zasady
            except Exception:  # noqa: BLE001
                pass
            self._windows_set = False
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
        self.active = False

    # ── Windows ──
    def _win_kernel32(self) -> Any:
        if self._kernel32 is not None:
            return self._kernel32
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
        kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
        self._kernel32 = kernel32
        return kernel32

    def _start_windows(self) -> None:
        previous = self._win_kernel32().SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        if not previous:
            self.description = "SetThreadExecutionState zwróciło błąd — wyłącz usypianie ręcznie"
            return
        self._windows_set = True
        self.active = True
        self.description = "SetThreadExecutionState (system nie zaśnie; monitor może się wyłączyć)"

    # ── Linux / macOS: proces pomocniczy ──
    def _start_helper(self, cmd: list[str], label: str) -> None:
        # Osobna sesja: Ctrl+C w terminalu nie zabija blokady przed końcem sprzątania
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        time.sleep(_HELPER_START_CHECK_S)
        if self._proc.poll() is not None:
            err = (self._proc.stderr.read() if self._proc.stderr else b"") or b""
            self._proc = None
            reason = err.decode("utf-8", errors="replace").strip().splitlines()
            self.description = f"{label} nie działa ({reason[0] if reason else 'proces zakończył się od razu'})"
            return
        self.active = True
        self.description = label


__all__ = ["KeepAwake"]
