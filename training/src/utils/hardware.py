"""
Sprzęt — jedno miejsce decydujące o urządzeniu (CPU/GPU) i liczbie wątków.

Urządzenie wybierane OSOBNO dla każdej biblioteki z obsługą GPU (xgboost, catboost) i dla modeli
fundamentalnych TFM (tabpfn, tabicl — próba GPU przez bibliotekę "torch"):

    SOR_AI_DEVICE_XGBOOST / SOR_AI_DEVICE_CATBOOST / SOR_AI_DEVICE_TABPFN / SOR_AI_DEVICE_TABICL
        >  SOR_AI_DEVICE  >  'auto'

Wartości: auto | cpu | gpu.
    auto — PRAWDZIWY próbny trening na GPU (`probe_gpu`); porażka → CPU z głośnym ostrzeżeniem,
    cpu  — zawsze CPU (bez próby),
    gpu  — wymuszenie: porażka próby → RuntimeError (zamiast cichego treningu na CPU).

Samo `nvidia-smi` nie wystarcza: koło xgboost 3.4 jest zbudowane z CUDA 13 i przy sterowniku
starszym niż 580 po cichu trenuje na CPU, a CatBoost rzuca `CUDA error 35`. Torch z PyPI na Windows
jest zbudowany BEZ CUDA — próba "torch" rozpoznaje to i podaje polecenie instalacji koła CUDA.

Wątki CPU: `n_jobs()` = SOR_AI_N_JOBS (> 0) albo liczba FIZYCZNYCH rdzeni (przycięta do maski
affinity procesu).

Moduł nie importuje `src.utils.config` na poziomie modułu (config importuje `n_jobs` stąd).
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from src.utils.fileio import write_text_atomic
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# Stałe
# ─────────────────────────────────────────
GPU_LIBRARIES: tuple[str, ...] = ("xgboost", "catboost", "torch")
DEVICE_CHOICES: tuple[str, ...] = ("auto", "cpu", "gpu")
_LIBRARY_ALIASES: dict[str, str] = {"xgb": "xgboost", "cb": "catboost"}
# Modele, których próba GPU to próba innej biblioteki (TFM → torch). Osobna mapa, NIE alias: zmienna
# SOR_AI_DEVICE_TABPFN / SOR_AI_DEVICE_TABICL musi być czytana pod nazwą modelu.
_PROBE_LIBRARY: dict[str, str] = {"tabpfn": "torch", "tabicl": "torch"}
_MODEL_ALIASES: dict[str, str] = {"pfn": "tabpfn", "icl": "tabicl"}
# Koło torch z CUDA (PyPI na Windows = tylko CPU); cu130 wymaga sterownika NVIDIA >= 580, starszy → cu128
TORCH_CUDA_INDEX_URL: str = "https://download.pytorch.org/whl/cu130"
TORCH_CUDA_INDEX_URL_OLD_DRIVER: str = "https://download.pytorch.org/whl/cu128"

NVIDIA_SMI_TIMEOUT_S: int = 20            # pierwsze wywołanie na Windows (GPU uśpione) bywa wolne
PROBE_CACHE_MAX_AGE_S: int = 12 * 3600    # dodatni wynik próby GPU ważny 12 h (ten sam sterownik i wersja)

# Fragmenty komunikatów błędów pochodzących z GPU (CUDA, OOM, NCCL, sterownik)
_GPU_ERROR_PATTERNS: tuple[str, ...] = ("cuda", "out of memory", "cudaerror", "nccl", "gpu", "device")

# Zmienne środowiskowe sterujące sprzętem (zapisywane w podsumowaniu runa)
_ENV_KEYS: tuple[str, ...] = (
    "SOR_AI_DEVICE", "SOR_AI_DEVICE_XGBOOST", "SOR_AI_DEVICE_CATBOOST", "SOR_AI_N_JOBS",
    "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTHONUTF8", "PYTHONIOENCODING",
    "SOR_AI_DEVICE_TABPFN", "SOR_AI_DEVICE_TABICL",
)

_GPU_HINTS: dict[str, str] = {
    "xgboost": (
        "Koło xgboost >= 3.4 jest zbudowane z CUDA 13 — wymaga sterownika NVIDIA >= 580 "
        "(sprawdź `nvidia-smi`: 'Driver Version' i 'CUDA Version: 13.x')."
    ),
    "catboost": (
        "Sprawdź sterownik NVIDIA (`nvidia-smi`) i czy pamięć GPU nie jest zajęta przez inny proces."
    ),
    "torch": (
        "Modele TFM (TabPFN, TabICL) potrzebują torch z CUDA. Torch z PyPI na Windows jest tylko-CPU — "
        f"zainstaluj koło CUDA: `pip install torch --index-url {TORCH_CUDA_INDEX_URL}` (sterownik NVIDIA >= 580; "
        f"starszy sterownik: `--index-url {TORCH_CUDA_INDEX_URL_OLD_DRIVER}`), potem "
        "`python scripts/13_tfm_setup.py`."
    ),
}

_announced: set[str] = set()   # biblioteki, dla których już zalogowano decyzję o urządzeniu


# ─────────────────────────────────────────
# Pomocnicze
# ─────────────────────────────────────────
def _short(exc: BaseException, limit: int = 400) -> str:
    """Pierwsza linia komunikatu wyjątku (bez wielolinijkowych śladów z bibliotek C++)."""
    text = str(exc).strip().splitlines()
    first = text[0] if text else ""
    return first[:limit]


def _dist_version(dist: str) -> str | None:
    """Wersja zainstalowanego pakietu bez importowania go."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _normalize_library(library: str) -> str:
    lib = str(library).strip().lower()
    lib = _MODEL_ALIASES.get(lib, lib)
    return _LIBRARY_ALIASES.get(lib, lib)


def probe_library(library: str) -> str:
    """Biblioteka, której próba GPU decyduje o urządzeniu (tabpfn / tabicl → torch; reszta bez zmian)."""
    lib = _normalize_library(library)
    return _PROBE_LIBRARY.get(lib, lib)


# ─────────────────────────────────────────
# nvidia-smi
# ─────────────────────────────────────────
def _nvidia_smi_exe() -> str | None:
    """Ścieżka do nvidia-smi (PATH albo standardowa lokalizacja na Windows)."""
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if platform.system() == "Windows":
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidate = base / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        if candidate.exists():
            return str(candidate)
    return None


@lru_cache(maxsize=1)
def nvidia_smi_info() -> dict | None:
    """
    Karta i sterownik według `nvidia-smi` (None, gdy brak sterownika / karty).

    Returns
    -------
    dict: name, driver_version, cuda_version (maks. CUDA obsługiwana przez sterownik),
          memory_total_mb, gpu_count — dla pierwszej karty.
    """
    exe = _nvidia_smi_exe()
    if exe is None:
        return None
    try:
        query = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=NVIDIA_SMI_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line for line in (query.stdout or "").splitlines() if line.strip()]
    if query.returncode != 0 or not lines:
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 3:
        return None
    name, driver, total = parts[0], parts[1], parts[2]
    try:
        memory_total_mb: int | None = int(float(total))
    except ValueError:
        memory_total_mb = None

    cuda_version = None
    try:
        header = subprocess.run(
            [exe], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=NVIDIA_SMI_TIMEOUT_S, check=False,
        )
        match = re.search(r"CUDA Version:\s*([\d.]+)", header.stdout or "")
        cuda_version = match.group(1) if match else None
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        "name": name,
        "driver_version": driver,
        "cuda_version": cuda_version,
        "memory_total_mb": memory_total_mb,
        "gpu_count": len(lines),
    }


# ─────────────────────────────────────────
# Próby GPU (prawdziwy mini-trening)
# ─────────────────────────────────────────
def _probe_xgboost() -> tuple[bool, str]:
    try:
        import numpy as np
        import xgboost as xgb
    except Exception as exc:  # noqa: BLE001 — brak/uszkodzona biblioteka
        return False, f"import xgboost: {type(exc).__name__}: {_short(exc)}"

    info = xgb.build_info()
    cuda = ".".join(str(v) for v in (info.get("CUDA_VERSION") or [])) or "?"
    if not info.get("USE_CUDA"):
        return False, f"xgboost {xgb.__version__} zbudowany bez CUDA (koło tylko-CPU)"

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rng = np.random.default_rng(0)
            X = rng.random((512, 4), dtype=np.float32)
            y = (np.arange(512) % 3).astype(np.float32)
            booster = xgb.train(
                {"device": "cuda", "tree_method": "hist", "objective": "multi:softprob",
                 "num_class": 3, "nthread": 1, "max_depth": 2},
                xgb.DMatrix(X, label=y), num_boost_round=2,
            )
            device = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
    except Exception as exc:  # noqa: BLE001 — XGBoostError (CUDA), cokolwiek innego = brak GPU
        return False, f"xgboost {xgb.__version__} (CUDA {cuda}): {type(exc).__name__}: {_short(exc)}"

    if not str(device).startswith("cuda"):
        details = "; ".join(_short(w.message, 200) for w in caught)
        return False, (
            f"xgboost {xgb.__version__} (CUDA {cuda}) nie widzi GPU — próbny trening przebiegł "
            f"na '{device}'" + (f" ({details})" if details else "")
        )
    return True, f"xgboost {xgb.__version__} (CUDA {cuda}) trenuje na {device}"


def _probe_catboost() -> tuple[bool, str]:
    try:
        import numpy as np
        from catboost import CatBoostClassifier
        from catboost.utils import get_gpu_device_count
    except Exception as exc:  # noqa: BLE001
        return False, f"import catboost: {type(exc).__name__}: {_short(exc)}"

    version = _dist_version("catboost") or "?"
    try:
        n_devices = int(get_gpu_device_count())   # 0 m.in. przy CUDA error 35 (za stary sterownik)
    except Exception as exc:  # noqa: BLE001
        return False, f"catboost {version}: get_gpu_device_count(): {type(exc).__name__}: {_short(exc)}"
    if n_devices < 1:
        return False, f"catboost {version}: get_gpu_device_count() == 0 (brak GPU CUDA albo za stary sterownik)"

    from src.utils import config as _config

    try:
        rng = np.random.default_rng(0)
        X = rng.random((512, 4))
        y = np.arange(512) % 3
        CatBoostClassifier(
            task_type="GPU", devices="0", gpu_ram_part=_config.CATBOOST_GPU_RAM_PART,
            iterations=5, depth=2, loss_function="MultiClass",
            verbose=0, allow_writing_files=False, thread_count=1,
        ).fit(X, y)
    except Exception as exc:  # noqa: BLE001 — CatBoostError (CUDA, OOM)
        return False, f"catboost {version}: {type(exc).__name__}: {_short(exc)}"
    return True, f"catboost {version} trenuje na GPU (urządzeń: {n_devices})"


def torch_info() -> dict[str, Any]:
    """
    Stan torch i CUDA bez wyjątków (preflight, 13_tfm_setup, próba GPU TFM).

    Returns
    -------
    dict: installed, version, cuda_build (wersja CUDA koła albo None = koło tylko-CPU), cuda_available,
          device_name, total_vram_mb, free_vram_mb, error (opis porażki) — klucze zawsze obecne.
    Import torch trwa ~2–3 s — wołaj w procesie treningu / preflight, nie w orkiestratorze.
    """
    info: dict[str, Any] = {
        "installed": _dist_version("torch") is not None, "version": _dist_version("torch"), "cuda_build": None,
        "cuda_available": False, "device_name": None, "total_vram_mb": None, "free_vram_mb": None, "error": None,
    }
    if not info["installed"]:
        info["error"] = "torch nie jest zainstalowany"
        return info
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — uszkodzona instalacja (DLL na Windows)
        info["error"] = f"import torch: {type(exc).__name__}: {_short(exc)}"
        return info
    info["version"] = str(getattr(torch, "__version__", info["version"]))
    info["cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = str(props.name)
            info["total_vram_mb"] = int(props.total_memory // 2**20)
            free, _total = torch.cuda.mem_get_info(0)
            info["free_vram_mb"] = int(free // 2**20)
    except Exception as exc:  # noqa: BLE001 — sterownik / CUDA
        info["cuda_available"] = False
        info["error"] = f"CUDA: {type(exc).__name__}: {_short(exc)}"
    return info


def _probe_torch() -> tuple[bool, str]:
    """torch: CUDA dostępna + mała alokacja na karcie (koło tylko-CPU → wskazówka instalacji)."""
    info = torch_info()
    if not info["installed"] or info["version"] is None:
        return False, "torch nie jest zainstalowany (requirements-tfm.txt + koło CUDA z download.pytorch.org)"
    version = info["version"]
    if info["error"] and str(info["error"]).startswith("import torch"):
        return False, f"torch {version}: {info['error']}"
    if info["cuda_build"] is None:
        return False, f"torch {version} zbudowany BEZ CUDA (koło tylko-CPU, np. z PyPI na Windows)"
    if not info["cuda_available"]:
        reason = info["error"] or "torch.cuda.is_available() == False (brak GPU albo za stary sterownik)"
        return False, f"torch {version} (CUDA {info['cuda_build']}): {reason}"
    try:
        import torch

        tensor = torch.ones(1024, device="cuda") * 2.0
        torch.cuda.synchronize()
        ok = float(tensor.sum().item()) == 2048.0
        del tensor
    except Exception as exc:  # noqa: BLE001 — błąd CUDA przy alokacji
        return False, f"torch {version} (CUDA {info['cuda_build']}): {type(exc).__name__}: {_short(exc)}"
    if not ok:
        return False, f"torch {version}: niepoprawny wynik próbnej operacji na GPU"
    vram = f"{info['total_vram_mb'] / 1024:.1f} GB" if info["total_vram_mb"] else "?"
    return True, f"torch {version} (CUDA {info['cuda_build']}) na {info['device_name']} ({vram} VRAM)"


_PROBES = {"xgboost": _probe_xgboost, "catboost": _probe_catboost, "torch": _probe_torch}


def _probe_cache_path() -> Path:
    from src.utils import config as _config

    return Path(_config.LOGS_DIR) / "hardware_probe.json"


def _probe_cache_key(library: str) -> str:
    smi = nvidia_smi_info() or {}
    return "|".join([library, str(_dist_version(library)), str(smi.get("driver_version")), sys.executable])


def _read_probe_cache() -> dict:
    try:
        data = json.loads(_probe_cache_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_probe_cache(library: str, key: str, ok: bool, reason: str) -> None:
    """Zapis wyniku próby (diagnostyka + ponowne użycie dodatniego wyniku); błędy zapisu ignorowane."""
    path = _probe_cache_path()
    data = _read_probe_cache()
    data[library] = {
        "key": key,
        "ok": bool(ok),
        "reason": reason,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "checked_ts": time.time(),
    }
    try:
        write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False))
    except OSError:
        pass


@lru_cache(maxsize=None)
def probe_gpu(library: str) -> tuple[bool, str]:
    """
    Czy biblioteka NAPRAWDĘ trenuje na GPU — próbny trening na kilkuset wierszach.

    xgboost  — trening z device='cuda' i odczyt urządzenia z `booster.save_config()`
               (xgboost nie rzuca błędu, tylko po cichu przechodzi na CPU),
    catboost — `get_gpu_device_count()` + kilka iteracji z task_type='GPU',
    torch    — `torch.cuda.is_available()` + mała alokacja na karcie (modele TFM: tabpfn, tabicl).

    Wynik zapamiętany w procesie (lru_cache) i w `logs/hardware_probe.json`; z pliku używany
    jest tylko DODATNI wynik (ten sam sterownik, wersja biblioteki i interpreter, < 12 h) —
    ujemny jest zawsze sprawdzany ponownie. Nigdy nie rzuca wyjątku.

    Returns
    -------
    (ok, opis) — opis zawiera powód porażki do komunikatów i preflight.
    """
    lib = probe_library(library)            # tabpfn / tabicl → torch
    if lib not in _PROBES:
        return False, f"{lib}: biblioteka bez obsługi GPU w projekcie"

    try:
        key = _probe_cache_key(lib)
        cached = _read_probe_cache().get(lib)
        if (
            isinstance(cached, dict) and cached.get("ok") and cached.get("key") == key
            and time.time() - float(cached.get("checked_ts", 0)) < PROBE_CACHE_MAX_AGE_S
        ):
            return True, f"{cached.get('reason')} [wynik z {cached.get('checked_at')}]"

        t0 = time.time()
        ok, reason = _PROBES[lib]()
        log.debug(f"Próba GPU ({lib}): ok={ok} w {time.time() - t0:.1f} s — {reason}")
        _write_probe_cache(lib, key, ok, reason)
        return ok, reason
    except Exception as exc:  # noqa: BLE001 — próba nie może przerwać treningu
        return False, f"{lib}: nieoczekiwany błąd próby GPU: {type(exc).__name__}: {_short(exc)}"


# ─────────────────────────────────────────
# Wybór urządzenia
# ─────────────────────────────────────────
def device_preference(library: str) -> tuple[str, str]:
    """
    Preferencja urządzenia dla biblioteki ze zmiennych środowiskowych (czytanych na bieżąco).

    Returns
    -------
    (preferencja 'auto'|'cpu'|'gpu', źródło — nazwa zmiennej albo 'domyślnie')
    """
    lib = _normalize_library(library)
    for var in (f"SOR_AI_DEVICE_{lib.upper()}", "SOR_AI_DEVICE"):
        raw = os.environ.get(var)
        if raw is None or not raw.strip():
            continue
        value = raw.strip().lower()
        if value == "cuda":
            value = "gpu"
        if value not in DEVICE_CHOICES:
            raise ValueError(f"{var}={raw!r} — dozwolone wartości: {' | '.join(DEVICE_CHOICES)}")
        return value, var
    return "auto", "domyślnie"


def device_for(library: str) -> str:
    """
    Urządzenie treningu dla biblioteki: 'gpu' albo 'cpu'.

    Kolejność: SOR_AI_DEVICE_<BIBLIOTEKA> > SOR_AI_DEVICE > auto (próba GPU).
    Biblioteki bez GPU w projekcie (lightgbm, scikit-learn, interpret) → zawsze 'cpu'.
    Modele TFM ('tabpfn', 'tabicl'; aliasy 'pfn', 'icl'): zmienna SOR_AI_DEVICE_TABPFN / _TABICL,
    próba GPU biblioteki "torch" (`probe_library`).

    Raises
    ------
    RuntimeError — wymuszone 'gpu', a próba GPU się nie powiodła.
    """
    lib = _normalize_library(library)
    probe_lib = probe_library(lib)
    if probe_lib not in GPU_LIBRARIES:
        return "cpu"
    preference, source = device_preference(lib)
    if preference == "cpu":
        return "cpu"

    ok, reason = probe_gpu(probe_lib)
    if ok:
        if lib not in _announced:
            _announced.add(lib)
            log.info(f"{lib}: trening na GPU ({reason})")
        return "gpu"

    if preference == "gpu":
        raise RuntimeError(
            f"{source}=gpu, ale {lib} nie może użyć GPU: {reason}. {_GPU_HINTS[probe_lib]} "
            f"Aby trenować na CPU, ustaw {source}=cpu (albo auto)."
        )
    if lib not in _announced:
        _announced.add(lib)
        log.warning(
            f"{lib}: GPU NIEDOSTĘPNE — trening na CPU (znacznie wolniej). Powód: {reason}. "
            f"{_GPU_HINTS[probe_lib]} Wymuś decyzję: SOR_AI_DEVICE_{lib.upper()}=cpu|gpu."
        )
    return "cpu"


def is_gpu_error(exc: BaseException) -> bool:
    """
    Czy wyjątek pochodzi z GPU (CUDA, brak pamięci GPU, NCCL, sterownik) — wtedy warto
    powtórzyć trening na CPU. Rozpoznawane po treści komunikatu i nazwie typu
    (CatBoostError / XGBoostError niosą tekst błędu CUDA).
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(pattern in text for pattern in _GPU_ERROR_PATTERNS)


# ─────────────────────────────────────────
# Wątki CPU
# ─────────────────────────────────────────
def usable_cpus() -> int | None:
    """
    Liczba CPU (logicznych), na których proces może działać — maska affinity
    (os.sched_getaffinity; Windows/macOS: psutil cpu_affinity). None = nie da się ustalić.
    """
    try:
        return len(os.sched_getaffinity(0)) or None
    except (AttributeError, OSError):
        pass
    try:
        import psutil  # opcjonalny

        return len(psutil.Process().cpu_affinity()) or None
    except Exception:  # noqa: BLE001 — brak psutil / brak cpu_affinity na tej platformie
        return None


@lru_cache(maxsize=1)
def physical_cores() -> int:
    """
    Liczba fizycznych rdzeni dostępnych dla procesu.

    Źródło: psutil → joblib (only_physical_cores) → os.cpu_count(); wynik przycięty do liczby
    CPU z maski affinity procesu (`usable_cpus`), np. `taskset`, `start /affinity`, kontener —
    inaczej modele dostałyby więcej wątków niż rdzeni, na których mogą działać.
    """
    count: int | None = None
    try:
        import psutil  # opcjonalny

        count = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001 — brak psutil
        count = None
    if not count:
        try:
            import joblib

            count = joblib.cpu_count(only_physical_cores=True)
        except Exception:  # noqa: BLE001
            count = None
    if not count:
        count = os.cpu_count()
    count = max(1, int(count or 1))
    usable = usable_cpus()
    if usable:
        count = min(count, usable)
    return count


def n_jobs() -> int:
    """
    Liczba wątków CPU dla modeli: SOR_AI_N_JOBS (> 0) albo liczba fizycznych rdzeni.

    Na CPU hybrydowym (i9-14900K: 8P+16E = 24 rdzenie, 32 wątki) pętle OpenMP z barierami
    czekają na najwolniejszy wątek — wątki HT spowalniają LightGBM/XGBoost.
    """
    raw = os.environ.get("SOR_AI_N_JOBS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
        log.warning(f"SOR_AI_N_JOBS={raw!r} zignorowane (oczekiwano liczby > 0) — używam fizycznych rdzeni")
    return physical_cores()


# ─────────────────────────────────────────
# Windows: bez dławienia energii (EcoQoS)
# ─────────────────────────────────────────
_PROCESS_POWER_THROTTLING = 4                    # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_POWER_THROTTLING_CURRENT_VERSION = 1
_POWER_THROTTLING_EXECUTION_SPEED = 0x1


def disable_power_throttling() -> bool:
    """
    Windows 10/11: wyłącza dławienie energii (EcoQoS) bieżącego procesu —
    SetProcessInformation(ProcessPowerThrottling, ControlMask=EXECUTION_SPEED, StateMask=0).

    Bez tego Windows może przenieść proces z zminimalizowaną konsolą na wolne rdzenie E
    z obniżonym taktowaniem (nocny trening kilka razy wolniej). Na innych systemach no-op.

    Returns
    -------
    True — ustawienie przyjęte; False — inny system albo błąd (nigdy nie rzuca wyjątku).
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _PowerThrottlingState(ctypes.Structure):     # PROCESS_POWER_THROTTLING_STATE: 3 × ULONG (32 bity)
            _fields_ = [("Version", ctypes.c_uint32), ("ControlMask", ctypes.c_uint32),
                        ("StateMask", ctypes.c_uint32)]

        state = _PowerThrottlingState(_POWER_THROTTLING_CURRENT_VERSION, _POWER_THROTTLING_EXECUTION_SPEED, 0)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        ok = bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
                                                 ctypes.byref(state), ctypes.sizeof(state)))
        if not ok:
            log.debug(f"SetProcessInformation(ProcessPowerThrottling): błąd {ctypes.get_last_error()}")
        return ok
    except Exception as exc:  # noqa: BLE001 — starszy Windows / brak funkcji: tylko wolniej
        log.debug(f"Nie wyłączono dławienia energii: {type(exc).__name__}: {exc}")
        return False


# ─────────────────────────────────────────
# Podsumowanie sprzętu (run_info, preflight)
# ─────────────────────────────────────────
def _cpu_name() -> str:
    system = platform.system()
    if system == "Windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=5, check=False)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or "?"


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:  # noqa: BLE001
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / 1024**3, 1)
        except Exception:  # noqa: BLE001
            return None
        return None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


_LIBRARY_DISTS: dict[str, tuple[str, ...]] = {
    "xgboost": ("xgboost",),
    "lightgbm": ("lightgbm",),
    "catboost": ("catboost",),
    "interpret": ("interpret-core", "interpret"),
    "scikit-learn": ("scikit-learn",),
    "numpy": ("numpy",),
    "pandas": ("pandas",),
    "optuna": ("optuna",),
    "joblib": ("joblib",),
    "shap": ("shap",),
    "psutil": ("psutil",),
}


@lru_cache(maxsize=1)
def _static_summary() -> dict[str, Any]:
    libraries = {
        name: next((v for v in (_dist_version(d) for d in dists) if v), None)
        for name, dists in _LIBRARY_DISTS.items()
    }
    xgb_cuda, xgb_use_cuda = None, None
    try:
        import xgboost as xgb

        info = xgb.build_info()
        xgb_use_cuda = bool(info.get("USE_CUDA"))
        xgb_cuda = ".".join(str(v) for v in (info.get("CUDA_VERSION") or [])) or None
    except Exception:  # noqa: BLE001
        pass
    return {
        "os": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_name": _cpu_name(),
        "physical_cores": physical_cores(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "nvidia_smi": nvidia_smi_info(),
        "libraries": libraries,
        "xgboost_cuda_build": xgb_cuda,
        "xgboost_use_cuda": xgb_use_cuda,
    }


def hardware_summary() -> dict[str, Any]:
    """
    Opis maszyny i środowiska do run_info / preflight / artefaktu modelu (serializowalny do JSON).

    Bez prób GPU (te są w `probe_gpu`) — szybkie i bezpieczne na maszynie bez karty.
    """
    summary = json.loads(json.dumps(_static_summary()))   # głęboka kopia
    summary["n_jobs"] = n_jobs()
    summary["env"] = {k: os.environ[k] for k in _ENV_KEYS if os.environ.get(k)}
    return summary


__all__ = [
    "DEVICE_CHOICES",
    "GPU_LIBRARIES",
    "TORCH_CUDA_INDEX_URL",
    "device_for",
    "device_preference",
    "disable_power_throttling",
    "hardware_summary",
    "is_gpu_error",
    "n_jobs",
    "nvidia_smi_info",
    "physical_cores",
    "probe_gpu",
    "probe_library",
    "torch_info",
    "usable_cpus",
]
