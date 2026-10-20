"""Runtime feature engineering for a single patient row.

Builds one row in the CANONICAL triage schema used in training
(`src.data.prepare` → `src.features.engineering`) and appends the engineered
clinical features with `src.features.engineering.engineer_features()`:

    age (years), is_male (0/1; unknown → the train mode, see unknown_sex_value — training
        has no missing sex, so NaN would be an input the models never saw),
    arrival_ambulance / _self / _other (one-hot; missing mode → "self" and
        police → "other", exactly as in training — yale_schema.ARRIVAL_*),
    arrival_month (1–12), arrival_weekday (1–7, Mon=1), arrival_hour_bin (0–5) —
        local time of the ED (settings.timezone, default Europe/Warsaw),
    triage_vital_hr / _sbp / _dbp / _rr / _o2 / _o2_device / _temp (°C!;
        SpO2 = 100 passes through — training restores Yale's deleted 100% readings),
    cc_* chief-complaint flags (199 Yale/Epic complaints, sanitized names;
        Yale's full/modified trauma activation merged into cc_trauma_activation).

Missing values stay NaN — exactly as in training (no global imputation):
tree models / EBM handle NaN natively, logreg imputes inside its own Pipeline.
"""
from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from app.config import ensure_training_on_path, get_settings
from app.models.schemas import Vitals
from app.pipeline.layer0_parser import KNOWN_CC_CODES

log = logging.getLogger(__name__)

# Add training/ source tree so we can import src.features / src.data.yale_schema
ensure_training_on_path()

from src.data.yale_schema import (  # noqa: E402
    ARRIVAL_CATEGORIES,
    ARRIVAL_MISSING_AS,
    ARRIVAL_MODE_MAP,
    TRAUMA_ACTIVATION_COL,
    TRAUMA_ACTIVATION_SOURCES,
    VITAL_BOUNDS,
    hour_to_bin,
)
from src.features.engineering import ENGINEERED_FEATURES, VITALS, engineer_features  # noqa: E402

_NAN = float("nan")

# Columns that engineer_features() produces. Some of them start with "cc_"
# (cc_count, cc_critical, cc_group_*) but are NOT chief complaints — they must
# never be pre-populated as complaint flags (cc_count counts every cc_* column).
_ENGINEERED_SET = frozenset(ENGINEERED_FEATURES)

# Full chief-complaint vocabulary of the trained models (199 cc_* columns)
CHIEF_COMPLAINT_COLUMNS: tuple[str, ...] = tuple(f"cc_{c}" for c in KNOWN_CC_CODES)

# Legacy complaint names (Yale's two-tier trauma activation) → merged training column
LEGACY_CC_COLUMNS: dict[str, str] = {src: TRAUMA_ACTIVATION_COL for src in TRAUMA_ACTIVATION_SOURCES}

# ── Request value → canonical encoding ───────────────────────────────

# Arrival mode used when the request does not provide one — the same category
# training assigns to Yale's missing mode (ARRIVAL_MISSING_AS = "self"; in Yale a
# missing mode marks a site-A fast-track pathway, not a clinical category, so it
# is no longer a separate feature). Collect the real mode (ZRM vs own transport)
# in the form whenever possible — it is a strong feature.
DEFAULT_ARRIVAL_MODE = ARRIVAL_MISSING_AS
_MISSING_ARRIVAL = frozenset({"", "?", "unknown", "nieznany", "brak", "brak danych"})

# Arrival mode (free text from the form / HIS) → Yale arrival category.
# Yale training values + Polish SOR equivalents (ZRM = ambulance; police → other).
ARRIVAL_MODE_ALIASES: dict[str, str] = {
    **{k.lower(): v for k, v in ARRIVAL_MODE_MAP.items()},
    **{cat: cat for cat in ARRIVAL_CATEGORIES},
    "zrm": "ambulance", "karetka": "ambulance", "pogotowie": "ambulance",
    "lpr": "ambulance", "śmigłowiec": "ambulance", "smiglowiec": "ambulance",
    "własny": "self", "wlasny": "self", "transport własny": "self",
    "transport wlasny": "self", "samodzielnie": "self", "pieszo": "self",
    "samochód": "self", "samochod": "self",
    "policja": "other", "radiowóz": "other", "radiowoz": "other",
    "inny": "other", "inne": "other", "wózek": "other", "wozek": "other",
}

_SEX_MAP: dict[str, float] = {
    "m": 1.0, "male": 1.0, "man": 1.0, "mężczyzna": 1.0, "mezczyzna": 1.0,
    "k": 0.0, "f": 0.0, "female": 0.0, "woman": 0.0, "kobieta": 0.0,
}


def normalize_arrival_mode(mode: Any) -> str:
    """Free-text arrival mode → one of ARRIVAL_CATEGORIES.

    Missing / unknown → DEFAULT_ARRIVAL_MODE ("self", as in training).
    Otherwise mirrors training: a known value is mapped, any other non-empty
    value → 'other'.
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
    """Sex / is_male from the request → 1.0 (male), 0.0 (female), NaN (unknown).

    NaN is only the "unknown" marker — build_feature_row replaces it with
    unknown_sex_value() before the row reaches the models.
    """
    if value is None:
        return _NAN
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, numbers.Real):
        return float(value) if value in (0, 1) else _NAN
    return _SEX_MAP.get(str(value).strip().lower(), _NAN)


def unknown_sex_value(medians: Mapping[str, Any] | None = None) -> float:
    """is_male used when the request has no (valid) sex: the train mode.

    Training has no missing is_male (Yale records sex for every visit), so NaN
    would be an input the models never saw. imputation_medians.json stores the
    train median of the 0/1 column, which is its mode (≈ 45% male → 0.0);
    without the file → 0.0.
    """
    value = (medians or {}).get("is_male")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value):
        return 0.0
    return 1.0 if value > 0.5 else 0.0


@lru_cache(maxsize=8)
def _zone(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def _utcnow() -> datetime:
    """Current instant (UTC-aware) — a separate function so tests can freeze the clock."""
    return datetime.now(timezone.utc)


def arrival_time(now: datetime | None = None, tz_name: str | None = None) -> datetime:
    """Arrival time in the ED's local time zone (settings.timezone, default Europe/Warsaw).

    now = None → the current instant; an aware datetime is converted to the local
    zone; a naive datetime is taken as already local. The models were trained on
    local arrival hours / weekdays (Yale), so a UTC server clock would shift
    arrival_hour_bin (and the weekday around midnight).
    """
    zone = _zone(tz_name or get_settings().timezone)
    if now is None:
        return _utcnow().astimezone(zone)
    if now.tzinfo is None:
        return now
    return now.astimezone(zone)


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
    medians: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a single-row DataFrame in the canonical triage schema + engineered features.

    vitals          : data from the frontend form (temperature in °C — no conversion)
    cc_features     : {"cc_chestpain": 1, ...} from the Layer 0 parser; legacy names
                      (cc_fulltrauma / cc_modifiedtrauma) → cc_trauma_activation; names
                      that are not chief-complaint columns of the models are dropped
    extra           : optional overrides (cc_* flags or canonical columns)
    all_cc_columns  : chief-complaint columns of the trained models (pre-populated
                      with 0); defaults to the full 199-column vocabulary
    is_male / arrival_mode / o2_device : optional request data; when not given they
                      are read from `vitals` if such fields exist, otherwise
                      is_male = train mode (unknown_sex_value, WARNING in the log),
                      arrival = DEFAULT_ARRIVAL_MODE ("self"), o2_device = NaN
    now             : arrival time (default: now); converted to settings.timezone
                      (see arrival_time)
    medians         : imputation_medians.json "medians" (registry.medians) — source of
                      the train mode of is_male for an unknown sex
    """
    now = arrival_time(now)
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
    sex = encode_sex(is_male)
    if math.isnan(sex):
        sex = unknown_sex_value(medians)
        log.warning("Sex unknown (%r) — is_male = %.0f (train mode; training has no missing "
                    "sex). Send \"sex\": \"M\" | \"F\" for a reliable prediction.", is_male, sex)
    row: dict[str, Any] = {
        # Demographics
        "age": float(vitals.age),
        "is_male": sex,
        # Arrival (one-hot) + local arrival time
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
    flags: dict[str, float] = {}
    complaints = list(cc_features.items()) + [
        (k, v) for k, v in (extra or {}).items() if k.startswith("cc_")
    ]
    for key, value in (extra or {}).items():
        if not key.startswith("cc_") and key not in _ENGINEERED_SET:
            row[key] = value
    for name, value in complaints:
        name = LEGACY_CC_COLUMNS.get(name, name)
        # Flags as in training: 0/1 (several legacy names may map to one column → max)
        flags[name] = max(flags.get(name, 0.0), min(float(value), 1.0))
    for name, value in flags.items():
        if name in allowed:
            row[name] = value
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
