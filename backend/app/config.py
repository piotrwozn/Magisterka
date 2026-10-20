"""Global configuration loaded from .env."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Models
    models_dir: Path = BACKEND_ROOT / "models"
    feature_set: str = "triage"                     # training feature set of the served models
    project_root: Path = PROJECT_ROOT

    # Arrival time features (arrival_month / _weekday / _hour_bin) use the local
    # time of the ED, not the server clock (containers usually run in UTC).
    # IANA name; on Windows the tz database comes from the `tzdata` package.
    timezone: str = "Europe/Warsaw"

    # Ollama (Layer 0 / 1B / 2)
    ollama_base_url: str = "http://localhost:11434"
    ollama_parser_model: str = "llama3"             # Layer 0
    ollama_nlp_model: str = "medgemma:27b"          # Layer 1B
    ollama_synthesis_model: str = "qwen3.6"         # Layer 2
    ollama_timeout_s: float = 30.0
    ollama_enabled: bool = True                     # auto-detected at runtime via /api/tags

    # CORS
    allowed_origins: list[str] = [
        "https://sorai-triage.com",
        "https://master.sorai-triage.pages.dev",
        "http://localhost:5173",
        "http://localhost:4173",
    ]

    # SHAP
    shap_enabled: bool = True
    shap_top_n: int = 5

    # Safety thresholds
    conflict_threshold: int = 2          # ESI-level spread between models triggering a doctor alert
    confidence_threshold: float = 0.6    # min MedGemma confidence


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_training_on_path() -> Path:
    """Put training/ on sys.path (src.data / src.features / src.utils.config) and return it.

    In Docker the training sources are on PYTHONPATH already (/opt/sorai/src);
    a missing directory is not added.
    """
    training_root = get_settings().project_root / "training"
    if training_root.is_dir() and str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))
    return training_root
