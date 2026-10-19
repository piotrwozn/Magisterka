"""Runtime feature engineering for a single patient row.

Builds one row in the CANONICAL triage schema used in training
(`src.data.prepare` → `src.features.engineering`) and appends the engineered
clinical features with `src.features.engineering.engineer_features()`:

    age (years), is_male (0/1, NaN when unknown),
    arrival_ambulance / _self / _police / _other / _unknown (one-hot;
        missing mode → DEFAULT_ARRIVAL_MODE, never _unknown),
    arrival_month (1–12), arrival_weekday (1–7, Mon=1), arrival_hour_bin (0–5),
    triage_vital_hr / _sbp / _dbp / _rr / _o2 / _o2_device / _temp (°C!),
    cc_* chief-complaint flags (200 Yale/Epic complaints, sanitized names).

Missing values stay NaN — exactly as in training (no global imputation):
tree models / EBM handle NaN natively, logreg imputes inside its own Pipeline.
"""
from __future__ import annotations

import logging
import math
import numbers
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

import pandas as pd

from app.config import get_settings
from app.models.schemas import Vitals
from app.pipeline.layer0_parser import KNOWN_CC_CODES

log = logging.getLogger(__name__)

# Add training/ source tree so we can import src.features / src.data.yale_schema
_settings = get_settings()
_training_root = str(_settings.project_root / "training")
if _training_root not in sys.path:
    sys.path.insert(0, _training_root)

from src.data.yale_schema import (  # noqa: E402
    ARRIVAL_CATEGORIES,
    ARRIVAL_MODE_MAP,
    VITAL_BOUNDS,
    hour_to_bin,
)
from src.features.engineering import ENGINEERED_FEATURES, VITALS, engineer_features  # noqa: E402

_NAN = float("nan")

# Columns that engineer_features() produces. Some of them start with "cc_"
# (cc_count, cc_critical, cc_group_*) but are NOT chief complaints — they must
# never be pre-populated as complaint flags (cc_count counts every cc_* column).
_ENGINEERED_SET = frozenset(ENGINEERED_FEATURES)

# Full chief-complaint vocabulary of the trained models (200 cc_* columns)
CHIEF_COMPLAINT_COLUMNS: tuple[str, ...] = tuple(f"cc_{c}" for c in KNOWN_CC_CODES)

# ── Request value → canonical encoding ───────────────────────────────

# Arrival mode used when the request does not provide one.
# NOT "unknown": in Yale, arrival_unknown=1 marks a distinct (fast-track-like)
# population — 90% ESI 4–5 vs 27% overall — so forcing it on every request
# shifts predictions ~1 ESI level toward low acuity (val, true mode masked:
# undertriage 17% → 56%). "self" is the training majority (61%) and the least
# distorting default (+3 pp undertriage). Collect the real mode (ZRM vs own
# transport) in the form whenever possible — it is a strong feature.
DEFAULT_ARRIVAL_MODE = "self"
_MISSING_ARRIVAL = frozenset({"", "?", "unknown", "nieznany", "brak", "brak danych"})

# Arrival mode (free text from the form / HIS) → Yale arrival category.
# Yale training values + Polish SOR equivalents (ZRM = ambulance).
ARRIVAL_MODE_ALIASES: dict[str, str] = {
    **{k.lower(): v for k, v in ARRIVAL_MODE_MAP.items()},
    **{cat: cat for cat in ARRIVAL_CATEGORIES if cat != "unknown"},
    "zrm": "ambulance", "karetka": "ambulance", "pogotowie": "ambulance",
    "lpr": "ambulance", "śmigłowiec": "ambulance", "smiglowiec": "ambulance",
    "własny": "self", "wlasny": "self", "transport własny": "self",
    "transport wlasny": "self", "samodzielnie": "self", "pieszo": "self",
    "samochód": "self", "samochod": "self",
    "policja": "police",
    "inny": "other", "inne": "other", "wózek": "other", "wozek": "other",
}

_SEX_MAP: dict[str, float] = {
    "m": 1.0, "male": 1.0, "man": 1.0, "mężczyzna": 1.0, "mezczyzna": 1.0,
    "k": 0.0, "f": 0.0, "female": 0.0, "woman": 0.0, "kobieta": 0.0,
}


def normalize_arrival_mode(mode: Any) -> str:
    """Free-text arrival mode → one of ARRIVAL_CATEGORIES.

    Missing / unknown → DEFAULT_ARRIVAL_MODE (see the comment above — never
    'unknown'). Otherwise mirrors training: a known value is mapped, any other
    non-empty value → 'other'.
    """
    text = "" if mode is None else str(mode).strip().lower()
    if text in _MISSING_ARRIVAL:
        return DEFAULT_ARRIVAL_MODE
    if text in ARRIVAL_MODE_ALIASES:
        return ARRIVAL_MODE_ALIASES[text]
    # e.g. "karetka ZRM", "transport własny (rodzina)"
    for alias in sorted(ARRIVAL_MODE_ALIASES, key=len, reverse=True):
        if len(alias) >= 3 and alias in text:
            return ARRIVAL_MODE_ALIASES[alias]
    log.warning("Unknown arrival mode %r → 'other'", mode)
    return "other"


def encode_sex(value: Any) -> float:
    """Sex / is_male from the request → 1.0 (male), 0.0 (female), NaN (unknown)."""
    if value is None:
        return _NAN
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, numbers.Real):
        return float(value) if value in (0, 1) else _NAN
    return _SEX_MAP.get(str(value).strip().lower(), _NAN)


def _binary(value: Any) -> float:
    """0/1 flag (e.g. supplemental O2); None / NaN → NaN."""
    if value is None:
        return _NAN
    value = float(value)
    return _NAN if math.isnan(value) else float(value > 0)


def _vital(value: Any, kind: str) -> float:
    """Vital sign in canonical units; implausible value → NaN (same bounds as training)."""
    if value is None:
        return _NAN
    value = float(value)
    lo, hi = VITAL_BOUNDS[kind]
    return value if lo <= value <= hi else _NAN


def _request_value(source: Any, *names: str) -> Any:
    """First non-None attribute of the request object (forward-compatible optional fields)."""
    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            return value
    return None


def _is_complaint(name: str) -> bool:
    return name.startswith("cc_") and name not in _ENGINEERED_SET


def _cc_vocabulary(all_cc_columns: Iterable[str] | None) -> tuple[str, ...]:
    """Chief-complaint columns to pre-populate with 0 (absent complaint)."""
    if all_cc_columns:
        cols = tuple(dict.fromkeys(c for c in all_cc_columns if _is_complaint(c)))
        if cols:
            return cols
    return CHIEF_COMPLAINT_COLUMNS


# ── Public API ────────────────────────────────────────────────────────

def build_feature_row(
    vitals: Vitals,
    cc_features: Mapping[str, float],
    extra: Mapping[str, Any] | None = None,
    all_cc_columns: Iterable[str] | None = None,
    *,
    is_male: Any = None,
    arrival_mode: Any = None,
    o2_device: Any = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Build a single-row DataFrame in the canonical triage schema + engineered features.

    vitals          : data from the frontend form (temperature in °C — no conversion)
    cc_features     : {"cc_chestpain": 1, ...} from the Layer 0 parser; names that are
                      not chief-complaint columns of the models are dropped
    extra           : optional overrides (cc_* flags or canonical columns)
    all_cc_columns  : chief-complaint columns of the trained models (pre-populated
                      with 0); defaults to the full 200-column vocabulary
    is_male / arrival_mode / o2_device : optional request data; when not given they
                      are read from `vitals` if such fields exist, otherwise
                      is_male=NaN, arrival = DEFAULT_ARRIVAL_MODE ("self" — NOT
                      arrival_unknown, see above), o2_device=NaN
    now             : arrival time (defaults to the system clock)
    """
    now = now or datetime.now()
    if is_male is None:
        is_male = _request_value(vitals, "is_male", "sex", "gender")
    if arrival_mode is None:
        arrival_mode = _request_value(vitals, "arrival_mode", "arrivalmode")
    if o2_device is None:
        o2_device = _request_value(vitals, "o2_device", "triage_vital_o2_device")

    if vitals.age < 18:
        log.warning("Age %s < 18 — models were trained on adults only (Yale ≥ 18 y)", vitals.age)

    # Vitals: °C as collected; implausible values and DBP ≥ SBP → NaN (as in prepare)
    vit = {kind: _vital(getattr(vitals, kind), kind) for kind in VITALS}
    if not (math.isnan(vit["dbp"]) or math.isnan(vit["sbp"])) and vit["dbp"] >= vit["sbp"]:
        vit["dbp"] = _NAN

    mode = normalize_arrival_mode(arrival_mode)
    row: dict[str, Any] = {
        # Demographics
        "age": float(vitals.age),
        "is_male": encode_sex(is_male),
        # Arrival (one-hot) + time (system clock)
        **{f"arrival_{cat}": float(cat == mode) for cat in ARRIVAL_CATEGORIES},
        "arrival_month": float(now.month),
        "arrival_weekday": float(now.isoweekday()),
        "arrival_hour_bin": float(hour_to_bin(now.hour)),
        # Triage vitals
        **{VITALS[kind]: value for kind, value in vit.items()},
        "triage_vital_o2_device": _binary(o2_device),
    }

    # Chief complaints: every known column = 0, then the parsed flags
    vocabulary = _cc_vocabulary(all_cc_columns)
    allowed = set(vocabulary)
    row.update(dict.fromkeys(vocabulary, 0.0))
    flags = dict(cc_features)
    for key, value in (extra or {}).items():
        if key.startswith("cc_"):
            flags[key] = value
        elif key not in _ENGINEERED_SET:
            row[key] = value
    for name, value in flags.items():
        if name in allowed:
            row[name] = float(value)
        else:
            log.debug("Skipping %s — not a chief-complaint column of the models", name)

    return engineer_features(pd.DataFrame([row]))


def align_features(
    df: pd.DataFrame,
    expected: list[str],
    medians: Mapping[str, float] | None = None,
    *,
    use_medians: bool = False,
) -> pd.DataFrame:
    """Select/order columns to match the model's expected feature list.

    Missing columns:
      - cc_* chief-complaint flags → 0.0 (absent complaint, as in training),
      - everything else → NaN (models handle missing values, as in training).
    `medians` (imputation_medians.json) is used ONLY when use_medians=True —
    an explicit fallback; by default nothing is median-imputed.
    Extra columns are dropped.
    """
    missing = [c for c in expected if c not in df.columns]
    out = df.reindex(columns=list(expected))
    fill: dict[str, float] = {c: 0.0 for c in missing if _is_complaint(c)}
    if use_medians and medians:
        fill.update({
            c: float(medians[c]) for c in missing
            if c not in fill and medians.get(c) is not None
        })
    if fill:
        out = out.fillna(value=fill)
    return out.astype(float)
