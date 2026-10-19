"""
Wykrywanie GPU — jedno miejsce decydujące, czy modele trenują na CUDA czy CPU.

Sterowanie: zmienna środowiskowa SOR_AI_DEVICE = auto | cpu | gpu
(domyślnie 'auto': GPU, jeśli `nvidia-smi` widzi kartę).
"""

from __future__ import annotations

import functools
import shutil
import subprocess

from src.utils import config as _config
from src.utils.logger import get_logger

log = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def gpu_count() -> int:
    """Liczba widocznych kart NVIDIA (0, jeśli brak sterownika lub karty)."""
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if result.returncode != 0:
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def use_gpu(preference: str | None = None) -> bool:
    """
    Czy trenować na GPU.

    Parameters
    ----------
    preference : str, optional
        'auto' | 'cpu' | 'gpu'. Domyślnie `config.DEVICE`.
    """
    pref = (preference or _config.DEVICE).lower()
    if pref == "cpu":
        return False
    available = gpu_count() > 0
    if pref == "gpu" and not available:
        log.warning("Wymuszono GPU (SOR_AI_DEVICE=gpu), ale nie wykryto karty — używam CPU.")
    return available
