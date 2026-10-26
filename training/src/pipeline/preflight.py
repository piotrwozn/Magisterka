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
Ostrzeżenia (trening ruszy): wersje inne niż requirements-lock.txt, RAM < 30 GB (komputer z 32 GB
raportuje ~31,8 GB), GPU niedostępne dla xgboost/catboost (→ CPU z budżetem zastępczym), stary
sterownik NVIDIA, ścieżka w OneDrive / ze znakami spoza ASCII / bardzo długa na Windows, kodowanie
konsoli inne niż UTF-8, Windows: plan zasilania inny niż wysoka wydajność.

Modele fundamentalne (TFM: tabpfn, tabicl) — kontrola tylko, gdy są na liście modeli (`check_tfm`;
torch / tabpfn / tabicl NIE są pakietami wymaganymi — główny venv działa bez nich):
    błąd krytyczny — brak torch / biblioteki TFM albo wersja starsza niż tabpfn 9.2.0 / tabicl 2.2.0,
                     brak wag w lokalnej pamięci podręcznej (sprawdzane BEZ sieci; hak testowy
                     SOR_AI_TABPFN_MODEL_PATH / SOR_AI_TABICL_MODEL_PATH / SOR_AI_TFM_FAKE omija kontrolę wag),
                     brak CUDA (koło tylko-CPU, brak karty, nieudana próba torch, wymuszone CPU) — chyba że
                     profil smoke z kontekstem ≤ TFM_CPU_MAX_CONTEXT (wtedy TFM na CPU),
    ostrzeżenie   — WOLNA VRAM < 12 GB (torch.cuda.mem_get_info: Ollama, przeglądarka i pulpit zajmują
                     kartę), mało miejsca w katalogu pamięci podręcznej wag.
Urządzenie TFM z próby torch trafia do facts["devices"] (tory i środowisko procesów treningu).

Kontrola, która sama się wywróci (np. brak pandas w świeżym venv), daje wpis w raporcie zamiast
tracebacku — brak pakietu to błąd krytyczny z poleceniem instalacji.
"""

from __future__ import annotations

import json
import locale
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
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
RAM_WARN_GB: float = 30.0            # nominalne 32 GB system raportuje jako ~31,7–31,9 GB
DISK_FAIL_GB: float = 10.0
DISK_WARN_GB: float = 20.0
MIN_NVIDIA_DRIVER: int = 580          # koło xgboost 3.4 = CUDA 13 → sterownik R580+
WINDOWS_LONG_PATH: int = 100          # + ~120 znaków nazw runów/logów → blisko limitu MAX_PATH (260)
RAW_MIN_BYTES: int = 1_000_000        # mniejszy "parquet" to raczej wskaźnik Git LFS / uszkodzony plik

REQUIREMENTS_TXT: Path = config.PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_LOCK: Path = config.PROJECT_ROOT / "requirements-lock.txt"
INSTALL_ADVICE: str = "python -m pip install -r requirements-lock.txt   (w aktywnym venv, z katalogu training/)"

# Windows: plany zasilania uznawane za wydajne (GUID wbudowanych planów) i nakładka „Najlepsza wydajność”
PERFORMANCE_SCHEMES: dict[str, str] = {
    "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "Wysoka wydajność",
    "e9a42b02-d5df-448d-aa00-03f14749eb61": "Najwyższa wydajność (Ultimate Performance)",
}
BEST_PERFORMANCE_OVERLAY: str = "ded574b5-45a0-4f42-8737-46345c09c238"
_PERFORMANCE_NAME = re.compile(r"high performance|ultimate|wysoka wydajno|najwyższa wydajno|najlepsza wydajno",
                               re.IGNORECASE)
_GUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)

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

# Modele fundamentalne (TFM)
TFM_MIN_VERSIONS: dict[str, str] = {"tabpfn": "9.2.0", "tabicl": "2.2.0"}
TFM_VRAM_RECOMMENDED_GB: float = 12.0   # zalecana WOLNA VRAM (plan pamięci: szczyty profili ≤ ~11 GiB)
TFM_DISK_WARN_GB: float = 2.0           # wolne miejsce w katalogu pamięci podręcznej wag (pobranie ~1 GB)
TFM_CPU_PROFILE: str = "smoke"          # jedyny profil, w którym TFM mogą działać bez CUDA (kontekst ≤ limitu CPU)
TFM_SETUP_COMMAND: str = "python scripts/13_tfm_setup.py"


def tfm_install_advice() -> str:
    """Polecenia instalacji TFM: koło torch z CUDA (cu130 / cu128 dla starszego sterownika) + requirements-tfm."""
    from src.utils.hardware import TORCH_CUDA_INDEX_URL, TORCH_CUDA_INDEX_URL_OLD_DRIVER

    return (f"pip install torch --index-url {TORCH_CUDA_INDEX_URL}  (sterownik NVIDIA < 580: --index-url "
            f"{TORCH_CUDA_INDEX_URL_OLD_DRIVER}), potem pip install -r requirements-tfm.txt -c requirements-lock.txt "
            f"i {TFM_SETUP_COMMAND} (training/docs/TFM.md)")


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
        from src.utils.fileio import write_text_atomic

        return write_text_atomic(Path(path), json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str))


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
        report.add("Pakiety", STATUS_FAIL, " | ".join(parts), INSTALL_ADVICE)
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
                   "Zamknij inne programy na noc; zestaw 'full' wymaga 32 GB.")
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


# ─────────────────────────────────────────
# Modele fundamentalne (TFM)
# ─────────────────────────────────────────
def _version_tuple(text: str) -> tuple[int, ...]:
    """'9.2.0' / '2.2.0.post1' → (9, 2, 0) — porównanie bez pakietu `packaging`."""
    parts = []
    for piece in str(text).split("."):
        match = re.match(r"\d+", piece)
        if not match:
            break
        parts.append(int(match.group(0)))
    return tuple(parts)


def _version_at_least(version: str, minimum: str) -> bool:
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(version) >= Version(minimum)
        except InvalidVersion:
            pass
    except ImportError:
        pass
    return _version_tuple(version) >= _version_tuple(minimum)


def tfm_weights_status(model: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """
    Wagi modelu TFM w LOKALNEJ pamięci podręcznej — bez sieci i bez pobierania:
    {'present', 'path', 'cache_dir', 'size_mb', 'label', 'override'}.

    Hak testowy (SOR_AI_TABPFN_MODEL_PATH / SOR_AI_TABICL_MODEL_PATH / SOR_AI_TFM_FAKE) ma pierwszeństwo:
    sprawdzany jest wtedy tylko wskazany plik. Ścieżki jak w wrapperach: TabPFN — plik wersji
    `model_version` w katalogu TabPFN (TABPFN_MODEL_CACHE_DIR, także z pliku .env w bieżącym katalogu),
    TabICL — `hf_hub_download(..., local_files_only=True)` (HF_HUB_CACHE / HF_HOME). Importuje bibliotekę
    (torch, kilka sekund) — tylko w procesie preflight / 13_tfm_setup; ImportError przechodzi dalej.
    """
    from src.models.tfm_base import FAKE_ENV, fake_mode, weights_override_active

    if model not in config.TFM_MODELS:
        raise ValueError(f"'{model}' nie jest modelem TFM ({', '.join(config.TFM_MODELS)})")
    defaults = dict(getattr(config, f"{model.upper()}_DEFAULT_PARAMS", {}))
    params = {**defaults, **dict(params or {})}
    out: dict[str, Any] = {"present": False, "path": None, "cache_dir": None, "size_mb": None,
                           "label": model, "override": weights_override_active(model)}
    if out["override"]:
        if fake_mode():
            out.update(present=True, label=f"atrapa ({FAKE_ENV}=1)")
            return out
        raw = str(out["override"]).split("=", 1)[1]
        path = Path(raw).expanduser()
        out.update(path=str(path), cache_dir=str(path.parent), label=f"{model} (przekierowanie)")
    elif model == "tabpfn":
        from src.models.tabpfn_model import default_weights_path

        version = str(params.get("model_version") or "v3.5")
        path = default_weights_path(version)
        out.update(path=str(path), cache_dir=str(path.parent), label=f"TabPFN {version}")
    elif model == "tabicl":
        from src.models.tabicl_model import cached_checkpoint
        from src.models.tfm_base import TFMWeightsError

        name = str(params.get("checkpoint_version") or "tabicl-classifier-v2-20260212.ckpt")
        out["label"] = f"TabICL {name}"
        try:
            from huggingface_hub import constants as hf_constants

            out["cache_dir"] = str(hf_constants.HF_HUB_CACHE)
        except Exception:  # noqa: BLE001 — tylko opis katalogu
            pass
        try:
            path = cached_checkpoint(name)
        except TFMWeightsError:
            return out
        out["path"] = str(path)
    else:                                   # nieosiągalne (sprawdzone na początku)
        raise ValueError(f"'{model}' nie jest modelem TFM ({', '.join(config.TFM_MODELS)})")
    if path.is_file() and path.stat().st_size > 0:
        out.update(present=True, size_mb=round(path.stat().st_size / 2**20, 1))
    return out


def _tfm_planned_context(model: str, profile: str, tune: bool) -> int:
    """Największy kontekst worka planowany przez profil (kandydaci wyboru konfiguracji albo n_ctx domyślne)."""
    from src.models.tfm_base import planned_max_context

    prof = config.TUNING_PROFILES[profile]
    budget = {**dict(prof.get(model) or {}), "train_sample": prof.get("train_sample")}
    return int(planned_max_context(model, budget, None, tune))


def check_tfm(
    report: PreflightReport,
    models: Sequence[str],
    profile: str,
    *,
    tune: bool = True,
    torch_info: Callable[[], dict[str, Any]] | None = None,
    probe: Callable[[str], tuple[bool, str]] | None = None,
    preference: Callable[[str], tuple[str, str]] | None = None,
    version_of: Callable[[str], str | None] | None = None,
    weights: Callable[[str], dict[str, Any]] | None = None,
    disk_free: Callable[[Path], float | None] | None = None,
    fake: bool | None = None,
    planned_context: Callable[[str, str, bool], int] | None = None,
) -> dict[str, str]:
    """
    Kontrola TFM (tylko gdy `models` zawiera tabpfn / tabicl): torch + CUDA, wersje bibliotek, wagi offline,
    wolna VRAM, miejsce na dysku. Zwraca urządzenie TFM ('gpu' | 'cpu') — także w facts["devices"].

    Bez CUDA (brak torch, koło tylko-CPU, brak karty / sterownika, nieudana próba torch, wymuszone CPU):
    błąd krytyczny, chyba że profil smoke z kontekstem ≤ config.TFM_CPU_MAX_CONTEXT (wtedy 'cpu' z
    ostrzeżeniem — tor CPU). probe=None (--no-gpu-probe): bez próbnej alokacji — decyduje
    torch.cuda.is_available(). Parametry funkcyjne podstawiają testy (bez torch i karty).
    """
    tfms = [m for m in config.TFM_MODELS if m in models]
    if not tfms:
        return {}
    from src.utils import hardware

    torch_info = torch_info or hardware.torch_info
    preference = preference or hardware.device_preference
    version_of = version_of or _dist_version
    weights = weights or tfm_weights_status
    disk_free = disk_free or _disk_free_gb
    planned_context = planned_context or _tfm_planned_context
    if fake is None:
        fake = os.environ.get("SOR_AI_TFM_FAKE", "").strip().lower() not in ("", "0", "false", "no", "off", "nie")
    limit = int(config.TFM_CPU_MAX_CONTEXT)
    install = tfm_install_advice()
    facts: dict[str, Any] = {"models": tfms, "profile": profile, "fake": fake, "cpu_max_context": limit,
                             "versions": {}, "weights": {}, "devices": {}, "planned_context": {}}
    report.facts["tfm"] = facts
    if fake:
        report.add("TFM: hak testowy", STATUS_WARN, "SOR_AI_TFM_FAKE=1 — atrapa estymatora zamiast TabPFN / TabICL "
                   "(tylko testy): wersje bibliotek i wagi nie są sprawdzane",
                   "Usuń zmienną SOR_AI_TFM_FAKE przed prawdziwym treningiem.")

    # ── torch i CUDA ──
    info = dict(torch_info())
    facts["torch"] = info
    version, build = info.get("version"), info.get("cuda_build")
    cuda_ok, why_not, torch_msg, torch_missing = False, None, "", False
    if not info.get("installed"):
        torch_missing, why_not = True, "torch nie jest zainstalowany"
        if fake:
            report.add("TFM: torch", STATUS_INFO, "torch nie jest zainstalowany — atrapa TFM działa bez niego (CPU)")
        else:
            report.add("TFM: torch", STATUS_FAIL, "torch nie jest zainstalowany (wymagany przez TabPFN i TabICL)",
                       install)
    elif info.get("error") and str(info["error"]).startswith("import torch"):
        torch_missing, why_not = True, f"torch {version}: {info['error']}"
        report.add("TFM: torch", STATUS_FAIL, f"{why_not} (uszkodzona instalacja?)", "Przeinstaluj: " + install)
    elif build is None:
        why_not = (f"torch {version} zbudowany BEZ CUDA (koło tylko-CPU — tak instaluje się torch z PyPI na "
                   f"Windows)")
    elif not info.get("cuda_available"):
        why_not = (f"torch {version} (CUDA {build}): "
                   + (info.get("error") or "torch.cuda.is_available() == False — brak karty NVIDIA albo "
                      "sterownik za stary dla tej wersji CUDA"))
    elif probe is not None:
        ok, reason = probe("torch")
        cuda_ok, torch_msg = bool(ok), reason
        if not ok:
            why_not = f"próba GPU torch nieudana: {reason}"
    else:
        cuda_ok = True
        torch_msg = f"torch {version} (CUDA {build}) na {info.get('device_name')} — bez próbnej alokacji (--no-gpu-probe)"
    if cuda_ok:
        report.add("TFM: torch", STATUS_OK, torch_msg)
    elif not torch_missing:
        report.add("TFM: torch", STATUS_INFO if profile == TFM_CPU_PROFILE else STATUS_WARN,
                   f"{why_not}. Modele TFM potrzebują GPU", install)

    # ── urządzenie każdego TFM ──
    devices: dict[str, str] = {}
    for model in tfms:
        try:
            ctx = int(planned_context(model, profile, tune))
        except Exception:  # noqa: BLE001 — brak budżetu w profilu: zachowawczo ponad limit CPU
            ctx = limit + 1
        facts["planned_context"][model] = ctx
        try:
            pref, source = preference(model)
        except ValueError as exc:
            report.add(f"TFM: {model}", STATUS_FAIL, str(exc), "Dozwolone wartości: auto | cpu | gpu")
            devices[model] = "cpu"
            continue
        if pref != "cpu" and cuda_ok:
            devices[model] = "gpu"
            report.add(f"TFM: {model}", STATUS_OK, f"GPU (kontekst do {ctx:,} wierszy)")
            continue
        devices[model] = "cpu"
        reason = f"CPU wymuszone ({source}=cpu)" if pref == "cpu" else (why_not or "brak CUDA")
        if torch_missing and not fake:
            continue                                 # błąd krytyczny już zgłoszony (TFM: torch)
        if pref == "gpu":
            report.add(f"TFM: {model}", STATUS_FAIL, f"{source}=gpu, ale GPU niedostępne: {reason}", install)
        elif profile == TFM_CPU_PROFILE and ctx <= limit:
            report.add(f"TFM: {model}", STATUS_WARN,
                       f"CPU ({reason}) — profil {profile}, kontekst {ctx:,} ≤ {limit:,} wierszy: trening TFM na "
                       f"CPU (tylko test mechaniki)", install)
        else:
            report.add(f"TFM: {model}", STATUS_FAIL,
                       f"odmowa: {model} bez GPU ({reason}); profil '{profile}' planuje kontekst do {ctx:,} wierszy "
                       f"> TFM_CPU_MAX_CONTEXT={limit:,} (na CPU predykcja trwałaby wiele godzin)",
                       f"{install}. Test bez GPU: --profile {TFM_CPU_PROFILE}; albo usuń tfm z --models.")
    facts["devices"] = devices
    report.facts.setdefault("devices", {}).update(devices)

    # ── wolna VRAM ──
    if any(d == "gpu" for d in devices.values()):
        free, total = info.get("free_vram_mb"), info.get("total_vram_mb")
        if free is not None:
            free_gb, total_gb = float(free) / 1024, float(total or 0) / 1024
            msg = f"wolna VRAM {free_gb:.1f} GB z {total_gb:.1f} GB"
            if free_gb < TFM_VRAM_RECOMMENDED_GB:
                report.add("TFM: VRAM", STATUS_WARN, f"{msg} < {TFM_VRAM_RECOMMENDED_GB:.0f} GB zalecanych",
                           "Zamknij programy używające karty (Ollama: `ollama ps`, `ollama stop <model>`; przeglądarka "
                           "z akceleracją, gry). Przy braku pamięci TFM obniżają porcje same (drabina awaryjna), "
                           "ostatecznie run kończy się błędem z listą parametrów do zmniejszenia.")
            else:
                report.add("TFM: VRAM", STATUS_OK, msg)

    # ── biblioteki i wagi ──
    if fake or torch_missing:
        return devices
    for model in tfms:
        installed = version_of(model)
        facts["versions"][model] = installed
        minimum = TFM_MIN_VERSIONS[model]
        if installed is None:
            report.add(f"TFM: pakiet {model}", STATUS_FAIL, f"{model} nie jest zainstalowany (wymagany >= {minimum})",
                       install)
            continue
        if not _version_at_least(installed, minimum):
            report.add(f"TFM: pakiet {model}", STATUS_FAIL, f"{model} {installed} < {minimum} (requirements-tfm.txt)",
                       "pip install -r requirements-tfm.txt -c requirements-lock.txt")
            continue
        report.add(f"TFM: pakiet {model}", STATUS_OK, f"{model} {installed}")
        try:
            status = weights(model)
        except ImportError as exc:
            report.add(f"TFM: wagi {model}", STATUS_FAIL, f"nie sprawdzono — import {model}: {exc}", install)
            continue
        except Exception as exc:  # noqa: BLE001 — opis błędu zamiast tracebacku
            report.add(f"TFM: wagi {model}", STATUS_FAIL, f"nie sprawdzono — {type(exc).__name__}: {exc}",
                       f"{TFM_SETUP_COMMAND} --offline-check")
            continue
        facts["weights"][model] = status
        label = status.get("label") or model
        if status.get("override"):
            if status.get("present"):
                report.add(f"TFM: wagi {model}", STATUS_WARN,
                           f"PRZEKIEROWANIE WAG (tylko testy — predykcje bezwartościowe): {status['override']}",
                           "Usuń zmienną przed prawdziwym treningiem.")
            else:
                report.add(f"TFM: wagi {model}", STATUS_FAIL,
                           f"przekierowanie {status['override']} wskazuje nieistniejący plik",
                           "Popraw ścieżkę albo usuń zmienną.")
            continue
        if status.get("present"):
            size = f" ({status['size_mb']:,.0f} MB)" if status.get("size_mb") else ""
            report.add(f"TFM: wagi {model}", STATUS_OK, f"{label}: {status.get('path')}{size} — offline")
            continue
        where = status.get("path") or status.get("cache_dir") or "?"
        free = disk_free(Path(status["cache_dir"])) if status.get("cache_dir") else None
        report.add(f"TFM: wagi {model}", STATUS_FAIL,
                   f"{label}: brak w lokalnej pamięci podręcznej ({where}; sprawdzone bez sieci)"
                   + (f"; wolne miejsce {free:.1f} GB" if free is not None else ""),
                   f"Pobierz raz: {TFM_SETUP_COMMAND}"
                   + (" (TabPFN-3.5: akceptacja licencji niekomercyjnej, TABPFN_TOKEN)" if model == "tabpfn" else
                      " (checkpoint z Hugging Face, jingang/TabICL)")
                   + " — potem trening działa offline.")
        if free is not None and free < TFM_DISK_WARN_GB:
            report.add("TFM: dysk", STATUS_WARN,
                       f"wolne {free:.1f} GB w {status['cache_dir']} < {TFM_DISK_WARN_GB:.0f} GB (pobranie wag ~1 GB)",
                       "Zwolnij miejsce albo przenieś pamięć podręczną (TABPFN_MODEL_CACHE_DIR, HF_HOME).")
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
        info.update({k: manifest.get(k) for k in ("data_version", "split_strategy")})
        info["created"] = manifest.get("created") or manifest.get("created_at")
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
                   f"'{info.get('split_strategy')}', {info.get('created')})")
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


def _active_power_scheme() -> str | None:
    """Wyjście `powercfg /getactivescheme` (Windows) albo None — best-effort, 10 s limitu."""
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], stdin=subprocess.DEVNULL, capture_output=True,
                             text=True, errors="replace", timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout or "").strip() or None if out.returncode == 0 else None


def _power_overlay() -> str | None:
    """Windows 10/11: GUID nakładki „Tryb zasilania” przy zasilaniu sieciowym (rejestr) albo None."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes") as key:
            value, _ = winreg.QueryValueEx(key, "ActiveOverlayAcPowerScheme")
        return str(value).strip().lower() or None
    except Exception:  # noqa: BLE001 — brak klucza / nie Windows
        return None


def check_power(report: PreflightReport, system: str, scheme_output: str | None,
                overlay: str | None = None) -> None:
    """
    Windows: plan zasilania. Plan „Zrównoważony” z trybem innym niż „Najlepsza wydajność” pozwala
    systemowi dławić rdzenie (zwłaszcza przy zminimalizowanej konsoli i w nocy) — tylko ostrzeżenie,
    nigdy błąd krytyczny. Inne systemy: brak wpisu.
    """
    if system != "Windows":
        return
    report.facts["power"] = {"scheme": scheme_output, "overlay_ac": overlay}
    advice = ("Ustawienia → System → Zasilanie (i bateria) → Tryb zasilania: „Najlepsza wydajność” (przy zasilaniu "
              "sieciowym) albo plan „Wysoka wydajność” (Panel sterowania → Opcje zasilania); "
              "na noc nie minimalizuj okna konsoli.")
    if not scheme_output:
        report.add("Zasilanie", STATUS_INFO, "nie odczytano planu zasilania (powercfg /getactivescheme)", advice)
        return
    guid_match = _GUID.search(scheme_output)
    guid = guid_match.group(0).lower() if guid_match else None
    name_match = re.search(r"\(([^)]*)\)\s*$", scheme_output)
    name = name_match.group(1).strip() if name_match else (PERFORMANCE_SCHEMES.get(guid or "") or scheme_output)
    if guid in PERFORMANCE_SCHEMES or _PERFORMANCE_NAME.search(name):
        report.add("Zasilanie", STATUS_OK, f"plan: {name}")
    elif overlay and overlay.lower() == BEST_PERFORMANCE_OVERLAY:
        report.add("Zasilanie", STATUS_OK, f"plan: {name}, tryb zasilania: Najlepsza wydajność")
    else:
        report.add("Zasilanie", STATUS_WARN, f"plan: {name} — nie „Wysoka / Najwyższa wydajność” ani tryb "
                   f"„Najlepsza wydajność” (Windows może dławić CPU w nocy)", advice)


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
def _safe_check(report: PreflightReport, name: str, check: Callable[..., Any], *args: Any) -> Any:
    """
    Kontrola, która się wywróci, daje wpis w raporcie zamiast tracebacku: brak pakietu (ImportError)
    → błąd krytyczny z poleceniem instalacji, inny wyjątek → ostrzeżenie.
    """
    try:
        return check(report, *args)
    except ImportError as exc:
        report.add(name, STATUS_FAIL, f"nie sprawdzono — brak pakietu ({type(exc).__name__}: {exc})",
                   INSTALL_ADVICE)
    except Exception as exc:  # noqa: BLE001 — błąd jednej kontroli nie przerywa pozostałych
        report.add(name, STATUS_WARN, f"nie sprawdzono — {type(exc).__name__}: {exc}")
    return None


def run_preflight(probe_gpus: bool = True, out_dir: Path | None = None, *,
                  models: Sequence[str] | None = None, profile: str | None = None,
                  tune: bool = True) -> PreflightReport:
    """
    Wszystkie kontrole. `out_dir` → zapis `preflight.json` (UTF-8).

    Próby GPU trenują po kilka iteracji na karcie (kilka–kilkanaście sekund przy pierwszym
    uruchomieniu); dodatni wynik jest zapamiętywany w logs/hardware_probe.json i używany
    przez procesy treningu.

    models / profile / tune : plan przebiegu (run_all.py) — gdy `models` zawiera TFM (tabpfn, tabicl),
    dochodzi `check_tfm` dla profilu (domyślnie config.DEFAULT_TUNING_PROFILE). None = bez TFM (jak dotąd).
    """
    from src.utils import hardware

    report = PreflightReport()
    system = platform.system()
    _safe_check(report, "Python", check_python)
    _safe_check(report, "Pakiety", check_packages)
    try:
        hw = hardware.hardware_summary()
    except Exception as exc:  # noqa: BLE001 — opis sprzętu nie może przerwać kontroli
        hw = {"error": f"{type(exc).__name__}: {exc}"}
    report.facts["hardware"] = hw
    _safe_check(report, "Zasoby", check_resources, hw, _disk_free_gb(config.RUNS_DIR))
    _safe_check(report, "GPU", check_gpu, hw.get("nvidia_smi"), hardware.probe_gpu if probe_gpus else None,
                hardware.device_preference, hw.get("xgboost_cuda_build"))
    if models is not None and any(m in config.TFM_MODELS for m in models):
        report.facts["models"] = list(models)
        _safe_check(report, "TFM", lambda r: check_tfm(
            r, list(models), profile or config.DEFAULT_TUNING_PROFILE, tune=tune,
            probe=hardware.probe_gpu if probe_gpus else None))
    _safe_check(report, "Dane", check_data)
    _safe_check(report, "Ścieżki", check_paths, config.PROJECT_ROOT, system)
    _safe_check(report, "Kodowanie", check_encoding)
    if system == "Windows":
        _safe_check(report, "Zasilanie", check_power, system, _active_power_scheme(), _power_overlay())
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
    "check_power",
    "check_python",
    "check_resources",
    "check_tfm",
    "run_preflight",
    "tfm_install_advice",
    "tfm_weights_status",
]
