"""
Krok 0: kontrola środowiska przed treningiem (preflight).

Sprawdza: Python (>= 3.12, 64-bit), pakiety i wersje (requirements.txt / requirements-lock.txt),
RAM, wolne miejsce, rdzenie, kartę NVIDIA (nvidia-smi) i PRAWDZIWY próbny trening na GPU dla
xgboost i catboost, obecność danych Yale, ścieżkę repo (OneDrive, polskie znaki, długość),
kodowanie konsoli i (Windows) plan zasilania. Raport: konsola + logs/run_all/<czas>/preflight.json.

Brak pakietów (np. świeży venv bez `pip install`) nie kończy się tracebackiem: skrypt wypisuje,
czego brakuje, i polecenie instalacji.

Użycie:
    python scripts/00_check_env.py
    python scripts/00_check_env.py --no-gpu-probe        # bez próby GPU (szybciej)
    python scripts/00_check_env.py --out-dir logs/x      # własny katalog na preflight.json

Kod wyjścia: 0 = można trenować (ostrzeżenia nie blokują), 1 = błędy krytyczne.
"""

from __future__ import annotations

import sys

# Sprawdzenie wersji PRZED importem pakietu `src` (starszy Python może nie zaimportować kodu)
if sys.version_info < (3, 12):
    print(f"[BŁĄD ] Python {sys.version.split()[0]} — wymagany >= 3.12 (zalecany 3.13 x64). "
          f"Utwórz venv nowszym Pythonem.")
    sys.exit(1)

import argparse  # noqa: E402
import re  # noqa: E402
from datetime import datetime  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

INSTALL_COMMAND = "python -m pip install -r requirements-lock.txt   (w aktywnym venv, z katalogu training/)"


def missing_requirements(requirements: Path = PROJECT_ROOT / "requirements.txt") -> list[str]:
    """Pakiety z requirements.txt, których nie ma w środowisku (tylko biblioteka standardowa)."""
    missing: list[str] = []
    try:
        lines = requirements.read_text(encoding="utf-8").splitlines()
    except OSError:
        return missing
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0]
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    return missing


def report_import_failure(exc: BaseException) -> int:
    """Kod kontroli nie dał się zaimportować — brakujące pakiety i polecenie instalacji zamiast tracebacku."""
    missing = missing_requirements()
    print("SOR-AI — kontrola środowiska (preflight)\n")
    print(f"[BŁĄD ] Nie można uruchomić kontroli: {type(exc).__name__}: {exc}")
    if missing:
        print(f"[BŁĄD ] Brak pakietów: {', '.join(missing)}")
    print(f"          → {INSTALL_COMMAND}")
    print(f"\nPython: {sys.executable}  (czy to venv projektu? sprawdź: python -c \"import sys; print(sys.prefix)\")")
    return 1


try:
    from src.pipeline.orchestrator import (  # noqa: E402
        LANES_MIN_PHYSICAL_CORES,
        LANES_MIN_RAM_GB,
        MODEL_ORDER,
        decide_lanes,
        gpu_planned_models,
    )
    from src.pipeline.preflight import run_preflight  # noqa: E402
    from src.utils import config  # noqa: E402
except ImportError as _import_error:            # brak pakietu w świeżym venv — raport zamiast tracebacku
    if __name__ == "__main__":
        sys.exit(report_import_failure(_import_error))
    raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kontrola środowiska SOR-AI (preflight)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Katalog na preflight.json (domyślnie logs/run_all/<czas>/)")
    parser.add_argument("--no-gpu-probe", action="store_true", help="Pomiń próbny trening na GPU")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or (Path(config.LOGS_DIR) / "run_all" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    try:
        report = run_preflight(probe_gpus=not args.no_gpu_probe, out_dir=out_dir)
    except ImportError as exc:
        return report_import_failure(exc)
    print(report.format())

    facts = report.facts
    gpu_models = gpu_planned_models(MODEL_ORDER, facts.get("devices") or {}, bool(facts.get("gpu_ok")))
    lanes = decide_lanes("auto", gpu_models, facts.get("physical_cores"), facts.get("ram_gb"), MODEL_ORDER)
    print(f"\nrun_all.py --lanes auto → {lanes} tor(y) "
          f"(2 = co najmniej jeden model na GPU [{', '.join(gpu_models) or 'brak'}], "
          f"≥ {LANES_MIN_PHYSICAL_CORES} rdzeni fizycznych, ≥ {LANES_MIN_RAM_GB:.0f} GB RAM)")
    print(f"Raport JSON: {out_dir / 'preflight.json'}")
    if report.ok:
        print("Dalej: python scripts/run_all.py --profile smoke   (5–10 min), potem: python scripts/run_all.py")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
