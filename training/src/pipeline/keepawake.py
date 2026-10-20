"""
Blokada uśpienia komputera na czas nocnego przebiegu (`scripts/run_all.py`).

    with KeepAwake() as awake:
        print(awake.description)       # co faktycznie zadziałało
        ...                            # długi trening

Platformy:
    Windows — SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) w WĄTKU GŁÓWNYM
              (flaga dotyczy wątku, który ją ustawił — wątek musi żyć przez cały przebieg).
              Monitor może się wyłączyć; system nie zaśnie. Sprawdzenie: `powercfg /requests`.
              Sprawdzenie: `powercfg /requests` — tylko w PowerShell uruchomionym jako administrator.
    Linux   — `systemd-inhibit --what=idle:sleep … tail --pid=<PID> -f /dev/null`
              (blokada znika razem z procesem orkiestratora, także po awarii); bez systemd — nic.
              WSL2: blokada działa tylko w maszynie wirtualnej Linuksa — NIE powstrzyma hosta Windows
              przed uśpieniem (ostrzeżenie w logu; na Windows ustaw uśpienie na „Nigdy”).
    macOS   — `caffeinate -i -w <PID>`.

Nie chroni przed: zamknięciem pokrywy laptopa, ręcznym uśpieniem, restartem po Windows Update
(wstrzymaj aktualizacje na noc; wznowienie przebiegu i tak obsługuje restart).
Błędy nigdy nie przerywają treningu — `description` mówi, co się udało.

`QuickEditGuard` (Windows) wyłącza na czas przebiegu tryb QuickEdit konsoli: kliknięcie w okno
zaznacza tekst i wstrzymuje KAŻDY zapis na konsolę — orkiestrator, a przez potoki także procesy
treningu, stałyby do naciśnięcia klawisza.
"""

from __future__ import annotations

import ctypes
import os
import platform as _platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

# Stałe Win32 (winbase.h)
ES_CONTINUOUS: int = 0x80000000
ES_SYSTEM_REQUIRED: int = 0x00000001
# Konsola (wincon.h)
STD_INPUT_HANDLE: int = -10
ENABLE_QUICK_EDIT_MODE: int = 0x0040
ENABLE_EXTENDED_FLAGS: int = 0x0080

_HELPER_START_CHECK_S: float = 0.3   # tyle czekamy, czy proces pomocniczy nie padł od razu

WSL_HOST_WARNING: str = (
    "WSL2: blokada uśpienia działa tylko wewnątrz maszyny wirtualnej Linuksa i NIE powstrzyma Windows "
    "przed uśpieniem. Na Windows ustaw: Ustawienia → System → Zasilanie (i bateria) → Ekran i uśpienie → "
    "uśpienie przy zasilaniu sieciowym = „Nigdy” (albo w PowerShell: powercfg /change standby-timeout-ac 0)."
)


def is_wsl() -> bool:
    """Linux uruchomiony w WSL (Windows Subsystem for Linux)."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    if "microsoft" in _platform.release().lower():
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False


class KeepAwake:
    """
    Kontekst blokujący uśpienie systemu.

    Parameters
    ----------
    enabled : False → nic nie robi (`--no-keep-awake`).
    reason : opis widoczny w `systemd-inhibit --list` / `powercfg /requests`.
    platform : nadpisanie `sys.platform` (testy).
    kernel32 : obiekt z metodą SetThreadExecutionState (testy; domyślnie ctypes.windll.kernel32).
    wsl : nadpisanie wykrywania WSL (testy; None = `is_wsl()` na Linuksie).

    `host_warning` — ostrzeżenie dla użytkownika (WSL2: host Windows może zasnąć) albo None.
    """

    def __init__(
        self,
        enabled: bool = True,
        reason: str = "SOR-AI: nocny trening modeli",
        platform: str | None = None,
        kernel32: Any | None = None,
        wsl: bool | None = None,
    ) -> None:
        self.enabled = enabled
        self.reason = reason
        self.platform = platform or sys.platform
        self._kernel32 = kernel32
        self._wsl = wsl
        self._proc: subprocess.Popen | None = None
        self._windows_set = False
        self.active = False
        self.host_warning: str | None = None
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
                wsl = is_wsl() if self._wsl is None else self._wsl
                if wsl:
                    self.host_warning = WSL_HOST_WARNING
                    self.description += " — WSL2: NIE chroni hosta Windows przed uśpieniem (ustaw tam „Nigdy”)"
                    log.warning(WSL_HOST_WARNING)
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


class QuickEditGuard:
    """
    Windows: wyłącza tryb QuickEdit konsoli na czas przebiegu i przywraca poprzedni tryb na końcu
    (GetConsoleMode / SetConsoleMode na STD_INPUT_HANDLE: zdjęty bit 0x40 przy ustawionym 0x80).

    Inne systemy, brak konsoli (wejście przekierowane), QuickEdit już wyłączony albo błąd — nic
    nie robi (nigdy nie przerywa treningu); `description` mówi, co się stało.

    Parameters
    ----------
    enabled : False → nic nie robi.
    platform : nadpisanie `sys.platform` (testy).
    kernel32 : obiekt z GetStdHandle / GetConsoleMode / SetConsoleMode (testy).
    """

    def __init__(self, enabled: bool = True, platform: str | None = None, kernel32: Any | None = None) -> None:
        self.enabled = enabled
        self.platform = platform or sys.platform
        self._kernel32 = kernel32
        self._handle: Any = None
        self._original: int | None = None
        self.active = False
        self.description = "nieużywane (nie Windows)" if self.platform != "win32" else "nieuruchomione"

    def __enter__(self) -> QuickEditGuard:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _win_kernel32(self) -> Any:
        if self._kernel32 is not None:
            return self._kernel32
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        kernel32.GetConsoleMode.restype = wintypes.BOOL
        kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetConsoleMode.restype = wintypes.BOOL
        self._kernel32 = kernel32
        return kernel32

    def start(self) -> None:
        if not self.enabled or self.platform != "win32":
            return
        try:
            kernel32 = self._win_kernel32()
            handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
            mode = ctypes.c_uint32(0)
            if not handle or not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):   # brak konsoli
                self.description = "brak konsoli (wejście przekierowane) — QuickEdit bez zmian"
                return
            original = int(mode.value)
            if not original & ENABLE_QUICK_EDIT_MODE:
                self.description = "QuickEdit już wyłączony"
                return
            new_mode = (original | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE
            if not kernel32.SetConsoleMode(handle, new_mode):
                self.description = "SetConsoleMode zwróciło błąd — QuickEdit bez zmian (nie klikaj w okno konsoli)"
                return
            self._handle, self._original, self.active = handle, original, True
            self.description = ("wyłączono QuickEdit na czas przebiegu (kliknięcie w okno nie wstrzyma treningu; "
                                "przywrócę go na końcu)")
        except Exception as exc:  # noqa: BLE001 — tylko wygoda, nie może przerwać treningu
            self.description = f"nie udało się wyłączyć QuickEdit: {type(exc).__name__}: {exc}"
        log.debug(f"QuickEdit: {self.description}")

    def stop(self) -> None:
        if self._original is not None:
            try:
                self._win_kernel32().SetConsoleMode(self._handle, self._original)
            except Exception:  # noqa: BLE001
                pass
            self._original = None
        self.active = False


__all__ = ["KeepAwake", "QuickEditGuard", "WSL_HOST_WARNING", "is_wsl"]
