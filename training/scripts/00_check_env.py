"""
Krok 0: kontrola środowiska przed treningiem (preflight).

Sprawdza: Python (>= 3.12, 64-bit), pakiety i wersje (requirements.txt / requirements-lock.txt),
RAM, wolne miejsce, rdzenie, kartę NVIDIA (nvidia-smi) i PRAWDZIWY próbny trening na GPU dla
xgboost i catboost, obecność danych Yale, ścieżkę repo (OneDrive, polskie znaki, długość)
i kodowanie konsoli. Raport: konsola + logs/run_all/<czas>/preflight.json.

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
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.orchestrator import LANES_MIN_PHYSICAL_CORES, LANES_MIN_RAM_GB, decide_lanes  # noqa: E402
from src.pipeline.preflight import run_preflight  # noqa: E402
from src.utils import config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kontrola środowiska SOR-AI (preflight)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Katalog na preflight.json (domyślnie logs/run_all/<czas>/)")
    parser.add_argument("--no-gpu-probe", action="store_true", help="Pomiń próbny trening na GPU")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or (Path(config.LOGS_DIR) / "run_all" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    report = run_preflight(probe_gpus=not args.no_gpu_probe, out_dir=out_dir)
    print(report.format())

    facts = report.facts
    lanes = decide_lanes("auto", bool(facts.get("gpu_ok")), facts.get("physical_cores"), facts.get("ram_gb"),
                         ["xgboost", "logreg"])
    print(f"\nrun_all.py --lanes auto → {lanes} tor(y) "
          f"(2 = GPU działa, ≥ {LANES_MIN_PHYSICAL_CORES} rdzeni fizycznych, ≥ {LANES_MIN_RAM_GB:.0f} GB RAM)")
    print(f"Raport JSON: {out_dir / 'preflight.json'}")
    if report.ok:
        print("Dalej: python scripts/run_all.py --profile smoke   (5–10 min), potem: python scripts/run_all.py")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
