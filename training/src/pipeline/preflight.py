"""
Kontrola środowiska przed nocnym przebiegiem (preflight).

    python scripts/00_check_env.py          # raport na konsoli + logs/run_all/<czas>/preflight.json
    from src.pipeline.preflight import run_preflight
    report = run_preflight()
    report.ok, report.hard_failures, report.facts["devices"]

Błąd krytyczny (exit 1 w 00_check_env.py, run_all.py przerywa przed treningiem):
    - Python < 3.12 albo 32-bit, brak wymaganego pakietu / wersja spoza requirements.txt,
    - RAM < 16 GB, wolne miejsce < 10 GB,
    - brak danych (ani surowego Yale, ani przygotowanych splitów),
    - wymuszone GPU (SOR_AI_DEVICE*=gpu), a próba GPU się nie powiodła.
Ostrzeżenia (trening ruszy): wersje inne niż requirements-lock.txt, RAM < 32 GB, GPU niedostępne
dla xgboost/catboost (→ CPU z budżetem zastępczym), stary sterownik NVIDIA, ścieżka w OneDrive /
ze znakami spoza ASCII / bardzo długa na Windows, kodowanie konsoli inne niż UTF-8.
"""

from __future__ import annotations

import json
import locale
import os
import platform
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# Progi i listy
# ─────────────────────────────────────────
MIN_PYTHON: tuple[int, int] = (3, 12)
TESTED_PYTHON_MAX: tuple[int, int] = (3, 13)
RAM_FAIL_GB: float = 15.0            # nominalne 16 GB system raportuje jako ~15,6–15,9 GB
RAM_WARN_GB: float = 32.0
DISK_FAIL_GB: float = 10.0
DISK_WARN_GB: float = 20.0
MIN_NVIDIA_DRIVER: int = 580          # koło xgboost 3.4 = CUDA 13 → sterownik R580+
WINDOWS_LONG_PATH: int = 100          # + ~120 znaków nazw runów/logów → blisko limitu MAX_PATH (260)
RAW_MIN_BYTES: int = 1_000_000        # mniejszy "parquet" to raczej wskaźnik Git LFS / uszkodzony plik

REQUIREMENTS_TXT: Path = config.PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_LOCK: Path = config.PROJECT_ROOT / "requirements-lock.txt"

# Pakiety wymagane do treningu (nazwa dystrybucji → alternatywy)
REQUIRED_PACKAGES: dict[str, tuple[str, ...]] = {
    "numpy": ("numpy",),
    "pandas": ("pandas",),
    "pyarrow": ("pyarrow",),
    "scipy": ("scipy",),
    "scikit-learn": ("scikit-learn",),
    "xgboost": ("xgboost",),
    "lightgbm": ("lightgbm",),
    "catboost": ("catboost",),
    "interpret-core": ("interpret-core", "interpret"),
    "optuna": ("optuna",),
    "sqlalchemy": ("SQLAlchemy",),
    "joblib": ("joblib",),
    "threadpoolctl": ("threadpoolctl",),
    "psutil": ("psutil",),
    "shap": ("shap",),
    "matplotlib": ("matplotlib",),
    "seaborn": ("seaborn",),
    "rich": ("rich",),
}

STATUS_OK, STATUS_WARN, STATUS_FAIL, STATUS_INFO = "ok", "warn", "fail", "info"
_LABELS = {STATUS_OK: "[ OK  ]", STATUS_WARN: "[UWAGA]", STATUS_FAIL: "[BŁĄD ]", STATUS_INFO: "[INFO ]"}

_GPU_ADVICE: dict[str, str] = {
    "xgboost": "Zaktualizuj sterownik NVIDIA do >= 580 (Game Ready/Studio): koło xgboost 3.4 jest zbudowane "
               "z CUDA 13. `nvidia-smi` musi pokazać 'Driver Version: 580+' i 'CUDA Version: 13.x'. "
               "CUDA Toolkit NIE jest potrzebny.",
    "catboost": "Sprawdź sterownik (`nvidia-smi`) i czy pamięć karty nie jest zajęta przez inny program "
                "(gry, przeglądarka z akceleracją, drugi trening).",
}


# ─────────────────────────────────────────
# Raport
# ─────────────────────────────────────────
@dataclass
class Check:
    name: str
    status: str
    message: str
    advice: str = ""


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def add(self, name: str, status: str, message: str, advice: str = "") -> Check:
        check = Check(name=name, status=status, message=message, advice=advice)
        self.checks.append(check)
        return check

    @property
    def hard_failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == STATUS_FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == STATUS_WARN]

    @property
    def ok(self) -> bool:
        return not self.hard_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "ok": self.ok,
            "n_fail": len(self.hard_failures),
            "n_warn": len(self.warnings),
            "checks": [asdict(c) for c in self.checks],
            "facts": self.facts,
        }

    def format(self) -> str:
        lines = ["SOR-AI — kontrola środowiska (preflight)", ""]
        for c in self.checks:
            lines.append(f"{_LABELS.get(c.status, c.status)} {c.name}: {c.message}")
            if c.advice and c.status in (STATUS_WARN, STATUS_FAIL):
                lines.append(f"          → {c.advice}")
        lines.append("")
        if self.ok:
            lines.append(f"Wynik: OK ({len(self.warnings)} ostrzeżeń) — można trenować.")
        else:
            lines.append(f"Wynik: {len(self.hard_failures)} błędów krytycznych, {len(self.warnings)} ostrzeżeń "
                         f"— popraw błędy przed treningiem.")
        return "\n".join(lines)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return path


# ─────────────────────────────────────────
# Pomocnicze
# ─────────────────────────────────────────
def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_requirement_lines(path: Path) -> list[Any]:
    """Wymagania z pliku (bez komentarzy, `-r`, pustych linii) jako packaging.Requirement."""
    from packaging.requirements import InvalidRequirement, Requirement

    out = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith(("#", "-")):
            continue
        try:
            out.append(Requirement(line))
        except InvalidRequirement:
            continue
    return out


def _marker_applies(req: Any) -> bool:
    if req.marker is None:
        return True
    try:
        return bool(req.marker.evaluate())
    except Exception:  # noqa: BLE001 — nieznany marker = stosujemy
        return True


# ─────────────────────────────────────────
# Kontrole
# ─────────────────────────────────────────
def check_python(report: PreflightReport, version_info: tuple = tuple(sys.version_info),
                 is_64bit: bool = sys.maxsize > 2**32) -> None:
    version = ".".join(str(v) for v in version_info[:3])
    report.facts["python"] = {"version": version, "executable": sys.executable, "64bit": is_64bit}
    if tuple(version_info[:2]) < MIN_PYTHON:
        report.add("Python", STATUS_FAIL, f"{version} — wymagany >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
                   "Zainstaluj Python 3.13 x64 (python.org) i utwórz venv od nowa (xgboost 3.4, numpy 2.5 i "
                   "shap 0.54 nie mają kół dla starszych wersji).")
        return
    if not is_64bit:
        report.add("Python", STATUS_FAIL, f"{version} 32-bit", "Zainstaluj 64-bitowy Python (x64).")
        return
    if tuple(version_info[:2]) > TESTED_PYTHON_MAX:
        report.add("Python", STATUS_WARN, f"{version} (64-bit) — nowszy niż testowane 3.12/3.13",
                   "Jeśli instalacja pakietów zawiedzie, użyj Pythona 3.13.")
        return
    report.add("Python", STATUS_OK, f"{version} (64-bit)")


def check_packages(
    report: PreflightReport,
    requirements_path: Path = REQUIREMENTS_TXT,
    lock_path: Path = REQUIREMENTS_LOCK,
    version_of: Callable[[str], str | None] = _dist_version,
) -> None:
    """Wymagane pakiety obecne i w zakresach requirements.txt; różnice względem requirements-lock.txt."""
    installed: dict[str, str | None] = {}
    missing: list[str] = []
    for name, dists in REQUIRED_PACKAGES.items():
        version = next((v for v in (version_of(d) for d in dists) if v), None)
        installed[name] = version
        if version is None:
            missing.append(name)
    report.facts["packages"] = installed

    try:
        from packaging.version import InvalidVersion, Version

        ranges = {_normalize(r.name): r for r in _parse_requirement_lines(requirements_path)}
        locked = {_normalize(r.name): r for r in _parse_requirement_lines(lock_path) if _marker_applies(r)}
    except ImportError:
        ranges, locked = {}, {}
        report.add("Pakiety: zakresy", STATUS_INFO, "brak pakietu `packaging` — pomijam porównanie wersji")

    out_of_range: list[str] = []
    for key, req in ranges.items():
        if not _marker_applies(req) or not req.specifier:
            continue
        version = version_of(req.name)
        if version is None:
            continue
        try:
            if Version(version) not in req.specifier:
                out_of_range.append(f"{req.name} {version} (wymagane {req.specifier})")
        except InvalidVersion:
            continue

    if missing or out_of_range:
        parts = []
        if missing:
            parts.append("brak: " + ", ".join(missing))
        if out_of_range:
            parts.append("poza zakresem: " + "; ".join(out_of_range))
        report.add("Pakiety", STATUS_FAIL, " | ".join(parts),
                   "pip install -r requirements-lock.txt (w aktywnym venv)")
    else:
        report.add("Pakiety", STATUS_OK, f"{len(REQUIRED_PACKAGES)} wymaganych pakietów "
                   f"(xgboost {installed['xgboost']}, catboost {installed['catboost']}, "
                   f"lightgbm {installed['lightgbm']}, optuna {installed['optuna']})")

    if not lock_path.exists():
        report.add("Wersje vs lock", STATUS_WARN, f"brak {lock_path.name}", "Plik powinien być w repo (training/).")
        return
    diffs = []
    for key, req in sorted(locked.items()):
        pinned = next((s.version for s in req.specifier if s.operator == "=="), None)
        if pinned is None:
            continue
        version = version_of(req.name)
        if version is None:
            diffs.append(f"{req.name} brak (lock {pinned})")
        elif version != pinned:
            diffs.append(f"{req.name} {version} (lock {pinned})")
    report.facts["lock_differences"] = diffs
    if diffs:
        shown = "; ".join(diffs[:8]) + (f"; … +{len(diffs) - 8}" if len(diffs) > 8 else "")
        report.add("Wersje vs lock", STATUS_WARN, f"{len(diffs)} różnic: {shown}",
                   "Przetestowany zestaw: pip install -r requirements-lock.txt")
    else:
        report.add("Wersje vs lock", STATUS_OK, f"zgodne z {lock_path.name} ({len(locked)} pakietów)")

    for optional, why in (("pyreadr", "konwersja .rdata, gdy brak parquet"),
                          ("requests", "wyjaśnienia Ollama — scripts/05_explain.py")):
        if version_of(optional) is None:
            report.add(f"Pakiet {optional}", STATUS_INFO, f"nie zainstalowany (opcjonalny: {why})")


def check_resources(report: PreflightReport, hardware: dict[str, Any], disk_free_gb: float | None) -> None:
    """RAM, wolne miejsce, rdzenie."""
    ram = hardware.get("ram_gb")
    physical, logical = hardware.get("physical_cores"), hardware.get("logical_cores")
    report.facts.update({"ram_gb": ram, "physical_cores": physical, "logical_cores": logical,
                         "disk_free_gb": disk_free_gb, "n_jobs": hardware.get("n_jobs")})
    if ram is None:
        report.add("RAM", STATUS_WARN, "nie udało się odczytać", "Zalecane 32 GB (przygotowanie danych ~11 GB).")
    elif ram < RAM_FAIL_GB:
        report.add("RAM", STATUS_FAIL, f"{ram:.1f} GB — wymagane co najmniej 16 GB",
                   "Przygotowanie danych potrzebuje ~11 GB, trening RF/EBM na pełnych danych kolejne GB.")
    elif ram < RAM_WARN_GB:
        report.add("RAM", STATUS_WARN, f"{ram:.1f} GB < {RAM_WARN_GB:.0f} GB — jeden tor (bez równoległego GPU)",
                   "Zamknij inne programy na noc; zestaw 'full' wymaga >= 32 GB.")
    else:
        report.add("RAM", STATUS_OK, f"{ram:.1f} GB")

    if disk_free_gb is None:
        report.add("Dysk", STATUS_WARN, "nie udało się odczytać wolnego miejsca")
    elif disk_free_gb < DISK_FAIL_GB:
        report.add("Dysk", STATUS_FAIL, f"wolne {disk_free_gb:.1f} GB < {DISK_FAIL_GB:.0f} GB",
                   "Zwolnij miejsce (model RF może mieć ~2 GB, każdy przebieg dokłada 1–2 GB do runs/).")
    elif disk_free_gb < DISK_WARN_GB:
        report.add("Dysk", STATUS_WARN, f"wolne {disk_free_gb:.1f} GB < {DISK_WARN_GB:.0f} GB",
                   "Starsze runy z runs/ można usunąć albo przenieść.")
    else:
        report.add("Dysk", STATUS_OK, f"wolne {disk_free_gb:.0f} GB")

    report.add("CPU", STATUS_INFO, f"{hardware.get('cpu_name', '?')} — {physical} rdzeni fizycznych / "
               f"{logical} logicznych; wątki modeli (n_jobs): {hardware.get('n_jobs')}")


def _driver_major(driver: str | None) -> int | None:
    match = re.match(r"\s*(\d+)", str(driver or ""))
    return int(match.group(1)) if match else None


def check_gpu(
    report: PreflightReport,
    smi: dict | None,
    probe: Callable[[str], tuple[bool, str]] | None,
    preference: Callable[[str], tuple[str, str]],
    xgboost_cuda_build: str | None = None,
) -> dict[str, str]:
    """
    nvidia-smi + PRAWDZIWA próba treningu na GPU dla xgboost i catboost (`hardware.probe_gpu`).

    probe=None → próba pominięta (urządzenie 'auto' — zdecyduje proces treningu).
    Zwraca urządzenie dla każdej biblioteki: 'gpu' | 'cpu' | 'auto' (zapisywane też w facts).
    """
    report.facts["nvidia_smi"] = smi
    if smi:
        major = _driver_major(smi.get("driver_version"))
        msg = (f"{smi.get('name')}, {smi.get('memory_total_mb')} MB, sterownik {smi.get('driver_version')}, "
               f"CUDA (sterownik) {smi.get('cuda_version')}"
               + (f"; xgboost zbudowany z CUDA {xgboost_cuda_build}" if xgboost_cuda_build else ""))
        if major is not None and major < MIN_NVIDIA_DRIVER:
            report.add("GPU / sterownik", STATUS_WARN, msg + f" — sterownik < {MIN_NVIDIA_DRIVER}",
                       _GPU_ADVICE["xgboost"])
        else:
            report.add("GPU / sterownik", STATUS_OK, msg)
    else:
        report.add("GPU / sterownik", STATUS_INFO,
                   "nvidia-smi niedostępne — brak karty NVIDIA albo sterownika; xgboost/catboost na CPU "
                   "(wolniej, mniejszy budżet tuningu)")

    devices: dict[str, str] = {}
    probes: dict[str, dict[str, Any]] = {}
    for lib in ("xgboost", "catboost"):
        try:
            pref, source = preference(lib)
        except ValueError as exc:
            report.add(f"GPU: {lib}", STATUS_FAIL, str(exc), "Dozwolone wartości: auto | cpu | gpu")
            devices[lib] = "cpu"
            continue
        if pref == "cpu":
            devices[lib] = "cpu"
            report.add(f"GPU: {lib}", STATUS_INFO, f"CPU wymuszone ({source}=cpu) — bez próby GPU")
            continue
        if probe is None:
            devices[lib] = "auto"
            report.add(f"GPU: {lib}", STATUS_INFO, "próba GPU pominięta — decyzję podejmie proces treningu")
            continue
        ok, reason = probe(lib)
        probes[lib] = {"ok": bool(ok), "reason": reason}
        if ok:
            devices[lib] = "gpu"
            report.add(f"GPU: {lib}", STATUS_OK, reason)
        elif pref == "gpu":
            devices[lib] = "gpu"
            report.add(f"GPU: {lib}", STATUS_FAIL, f"{source}=gpu, ale próba GPU nieudana: {reason}",
                       _GPU_ADVICE[lib] + f" Albo ustaw {source}=auto (CPU z ostrzeżeniem).")
        else:
            devices[lib] = "cpu"
            report.add(f"GPU: {lib}", STATUS_WARN, f"GPU niedostępne → CPU (wolniej, budżet zastępczy): {reason}",
                       _GPU_ADVICE[lib])
    report.facts["devices"] = devices
    report.facts["gpu_probe"] = probes
    report.facts["gpu_ok"] = any(p["ok"] for p in probes.values())
    return devices


def check_data(report: PreflightReport) -> None:
    """Surowy Yale (parquet / rdata) albo gotowe splity; stan przygotowanych danych."""
    from src.data.dataset import is_prepared

    raw_parquet, rdata = Path(config.RAW_PARQUET), Path(config.RDATA_FILE)
    prepared = is_prepared()
    info: dict[str, Any] = {"raw_parquet": raw_parquet.exists(), "rdata": rdata.exists(), "prepared": prepared}
    if prepared:
        try:
            manifest = json.loads(Path(config.DATA_MANIFEST_JSON).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        info.update({k: manifest.get(k) for k in ("data_version", "split_strategy", "created_at")})
    report.facts["data"] = info

    small = [p for p in (raw_parquet, rdata) if p.exists() and p.stat().st_size < RAW_MIN_BYTES]
    if small:
        report.add("Dane", STATUS_FAIL, f"podejrzanie mały plik: {', '.join(str(p) for p in small)}",
                   "Plik wygląda na uszkodzony albo wskaźnik Git LFS — sklonuj repo ponownie (pełne pliki).")
        return
    if raw_parquet.exists():
        raw_msg = f"surowy Yale: {raw_parquet.name}"
    elif rdata.exists():
        if _dist_version("pyreadr") is None:
            report.add("Dane", STATUS_FAIL, f"jest tylko {rdata.name}, a brak pakietu pyreadr",
                       "pip install pyreadr  (albo: python scripts/01_convert_rdata.py na innej maszynie)")
            return
        raw_msg = f"surowy Yale: {rdata.name} (wolniejsze wczytanie)"
    elif prepared:
        raw_msg = "brak surowego Yale, ale splity są gotowe"
    else:
        report.add("Dane", STATUS_FAIL, f"brak {raw_parquet} i {rdata}",
                   "Pliki Yale są w repo (training/data/…) — sprawdź klon repozytorium.")
        return
    if prepared:
        report.add("Dane", STATUS_OK, f"{raw_msg}; splity gotowe ({info.get('data_version')}, podział "
                   f"'{info.get('split_strategy')}', {info.get('created_at')})")
    else:
        report.add("Dane", STATUS_OK, f"{raw_msg}; splity zostaną przygotowane (~3–5 min, szczyt ~11 GB RAM)")


def check_paths(report: PreflightReport, root: Path, system: str, prefix: str = sys.prefix) -> None:
    """Ścieżka repo i venv: OneDrive, znaki spoza ASCII, długość na Windows."""
    text, venv = str(root), str(prefix)
    report.facts["paths"] = {"project_root": text, "venv": venv}
    problems: list[str] = []
    advice: list[str] = []
    if "onedrive" in text.lower() or "onedrive" in venv.lower():
        problems.append("w folderze OneDrive (synchronizacja blokuje pliki SQLite/joblib)")
        advice.append("sklonuj repo np. do C:\\dev\\Magisterka, venv do C:\\venvs\\sorai")
    if not text.isascii():
        problems.append("znaki spoza ASCII w ścieżce repo (pyreadr/biblioteki C)")
        advice.append("użyj ścieżki bez polskich znaków")
    if system == "Windows" and len(text) > WINDOWS_LONG_PATH:
        problems.append(f"długa ścieżka ({len(text)} znaków) — ryzyko limitu 260 znaków")
        advice.append("krótsza ścieżka albo włącz LongPathsEnabled w rejestrze")
    if problems:
        report.add("Ścieżki", STATUS_WARN, f"{text}: " + "; ".join(problems), "; ".join(advice))
    else:
        report.add("Ścieżki", STATUS_OK, text)


def check_encoding(report: PreflightReport, system: str = platform.system()) -> None:
    """UTF-8 na konsoli / w trybie Pythona (Windows: cp1250 gubi polskie znaki i ramki w logach)."""
    utf8_mode = bool(sys.flags.utf8_mode)
    preferred = locale.getpreferredencoding(False)
    stdout_enc = getattr(sys.stdout, "encoding", None)
    report.facts["encoding"] = {"utf8_mode": utf8_mode, "preferred": preferred, "stdout": stdout_enc,
                                "PYTHONUTF8": os.environ.get("PYTHONUTF8")}
    if utf8_mode or preferred.lower().replace("-", "") == "utf8":
        report.add("Kodowanie", STATUS_OK, f"UTF-8 (tryb UTF-8: {utf8_mode}, stdout: {stdout_enc})")
    else:
        report.add("Kodowanie", STATUS_WARN if system == "Windows" else STATUS_INFO,
                   f"domyślne kodowanie {preferred} (stdout: {stdout_enc}); run_all.py i tak wymusza UTF-8 "
                   f"w procesach treningu",
                   "setx PYTHONUTF8 1   (nowe okno PowerShell) — dotyczy ręcznie uruchamianych skryptów")


def _disk_free_gb(path: Path) -> float | None:
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return round(shutil.disk_usage(probe).free / 1024**3, 1)
    except OSError:
        return None


# ─────────────────────────────────────────
# Całość
# ─────────────────────────────────────────
def run_preflight(probe_gpus: bool = True, out_dir: Path | None = None) -> PreflightReport:
    """
    Wszystkie kontrole. `out_dir` → zapis `preflight.json` (UTF-8).

    Próby GPU trenują po kilka iteracji na karcie (kilka–kilkanaście sekund przy pierwszym
    uruchomieniu); dodatni wynik jest zapamiętywany w logs/hardware_probe.json i używany
    przez procesy treningu.
    """
    from src.utils import hardware

    report = PreflightReport()
    check_python(report)
    check_packages(report)
    try:
        hw = hardware.hardware_summary()
    except Exception as exc:  # noqa: BLE001 — opis sprzętu nie może przerwać kontroli
        hw = {"error": f"{type(exc).__name__}: {exc}"}
    report.facts["hardware"] = hw
    check_resources(report, hw, _disk_free_gb(config.RUNS_DIR))
    check_gpu(report, hw.get("nvidia_smi"), hardware.probe_gpu if probe_gpus else None,
              hardware.device_preference, hw.get("xgboost_cuda_build"))
    check_data(report)
    check_paths(report, config.PROJECT_ROOT, platform.system())
    check_encoding(report)
    report.facts["platform"] = {"system": platform.system(), "release": platform.release(),
                                "machine": platform.machine()}
    if out_dir is not None:
        path = report.write_json(Path(out_dir) / "preflight.json")
        report.facts["json_path"] = str(path)
    return report


__all__ = [
    "Check",
    "PreflightReport",
    "check_data",
    "check_encoding",
    "check_gpu",
    "check_packages",
    "check_paths",
    "check_python",
    "check_resources",
    "run_preflight",
]
