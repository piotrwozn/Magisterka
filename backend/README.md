# SOR-AI Backend (production-ready)

```
                 ┌────────────────────┐
   Frontend ───▶ │  Nginx (80/443)    │  reverse proxy, SSL, rate limit
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │  Spring Boot       │  API gateway, validation,
                 │  Gateway (8080)    │  CORS, audit, circuit breaker
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐       ┌──────────────┐
                 │  FastAPI ML        │ ───▶  │  Ollama LLM  │
                 │  (8000)            │       │  (11434)     │
                 └────────────────────┘       └──────────────┘
                           │
                 ┌─────────▼──────────┐
                 │  Kafka audit log   │  GDPR-safe (no clinical notes)
                 └────────────────────┘
```

## Files

```
backend/
├── app/                                 — FastAPI ML service
├── gateway/                             — Spring Boot API gateway (Java 21)
├── nginx/nginx.conf                     — Nginx config
├── models/                              — deployed joblib models + optional train medians
├── scripts/generate_imputation_medians.py — train medians (diagnostics, cc_* vocabulary)
├── Dockerfile.fastapi
├── Dockerfile.gateway
├── docker-compose.yml
└── requirements.txt
```

## Local development

### 1. FastAPI only (fastest dev loop)

```bash
cd backend
pip install -r requirements.txt
python scripts/generate_imputation_medians.py     # optional; after re-preparing data
uvicorn app.main:app --reload --port 8000
```

Health: <http://localhost:8000/api/v1/health>
Docs: <http://localhost:8000/docs>

### 2. Full production stack (FastAPI + Spring + Nginx + Kafka)

```bash
docker compose -f backend/docker-compose.yml up --build
```

Public API: <http://localhost/api/v1/health>
Metrics: <http://localhost:8080/actuator/prometheus> (internal only)

## Frontend wiring

In `frontend/.env.production`:

```
VITE_API_URL=https://sorai-triage.com
```

In `frontend/src/hooks/usePredict.ts`:

```ts
const USE_MOCK = false;
```

## Layers status

All categories are ESI class indexes: **0–4 = ESI 1–5** (TOPSOR colours red, orange,
yellow, green, blue — display only).

| Layer | Component                | Status |
|-------|--------------------------|--------|
| 0     | Llama 3 parser           | ✓ Ollama structured output; keyword parser when Ollama is offline |
| 1A    | Ensemble ML (6 models)   | ✓ XGBoost, LightGBM, CatBoost, Random Forest, EBM, LogReg (whatever is trained); weights + decision rule from `models/ensemble_weights.json` |
| 1B    | MedGemma 27B reasoning   | ✓ Ollama (answers ESI 1–5); deterministic stub when offline |
| 2     | Qwen3 synthesis          | ✓ only on conflict (answers ESI 1–5); deterministic fallback |
| safety| ESI rules (deterministic)| ✓ live — see below |
| SHAP  | Top-N explainer          | ✓ highest-weight tree model (LightGBM / XGBoost / CatBoost / RF) |

## Request

```json
{
  "vitals": {"age": 67, "temp": 38.2, "hr": 118, "sbp": 95, "dbp": 62, "rr": 22, "o2": 94},
  "clinicalNote": "ból w klatce piersiowej, duszność",
  "sex": "M",                 // optional: "M" | "F"
  "arrivalMode": "ambulance", // optional: "ambulance" | "self" | "other" (also "ZRM", "własny", …)
  "o2Device": false           // optional: supplemental oxygen at triage
}
```

The response (`finalCategory`, `modelPredictions[].category`, `medgemma.category`) uses
class indexes 0–4 = ESI 1–5; `conflict.message` explains escalations in Polish.

## Ensemble weights (`backend/models/ensemble_weights.json`, optional)

Default: **equal weights** for every loaded model and the **argmax** decision rule.
To use weights chosen on the VALIDATION split (training: `src.ensemble.optimize_weights`
or a manual blend), create `backend/models/ensemble_weights.json`:

```json
{
  "weights": {"xgboost": 0.35, "lightgbm": 0.25, "catboost": 0.40},
  "rule": "cost"
}
```

- `weights` — model → weight ≥ 0 (normalised over the loaded models). Models not
  listed get weight 0 and are not run (same as `src.ensemble.blend`). Omit the key
  for equal weights. Known models: `xgboost`, `lightgbm`, `catboost`,
  `random_forest`, `ebm`, `logreg`.
- `rule` — `"argmax"` (default) or `"cost"`: Bayes decision with the training cost
  matrix (undertriage 3× more expensive than overtriage), applied to the blended
  probabilities and to each per-model prediction.

An invalid file (bad JSON, negative weight, unknown rule, no known model with a
weight > 0) stops the backend at startup with a clear error — a typo must not
silently fall back to defaults. The startup log lists the active ensemble.

## Safety rules (`app/pipeline/safety_rules.py`)

Deterministic, applied after the models and the LLMs; they only ever **raise** the
urgency. ESI thresholds are the same as `training/src/explain/esi_rules.py`
(checked by `tests/test_safety_rules.py`):

| ESI point | Condition | Effect |
|---|---|---|
| A — life-saving intervention needed now | SpO2 < 85 %, SBP < 80, HR < 40 or > 150, RR < 8 or > 35, T < 32 °C or ≥ 41 °C; complaint: cardiac arrest, unresponsive | **ESI 1 (czerwony)** + doctor alert |
| B — high-risk situation | SBP < 90, pain ≥ 7/10 (from the note); complaint: respiratory distress, stroke alert, trauma activation, suicidal/homicidal, intentional overdose | **minimum ESI 2 (pomarańczowy)** |
| D — danger-zone vitals (adults) | HR > 100, RR > 20, SpO2 < 92 % | **minimum ESI 2 (pomarańczowy)** |

Plus: the final level is never less urgent than the most urgent model; a spread of
≥ 2 ESI levels between models, ML vs MedGemma disagreement of ≥ 2 levels, low
MedGemma confidence or any model predicting ESI 1 alerts the doctor.

## Models and features

**Label:** native ESI 1–5 (no mapping). Class index 0–4 = ESI 1–5.

**Where models come from** (`app/models/registry.py`, first valid artifact wins per model):

1. `backend/models/<model>_*.joblib` — explicit deployment (copied into the Docker image),
2. `models/<model>_*.joblib` in the repo root,
3. `training/runs/<run>/model.joblib` — runs with `run_info.json` for the served
   feature set: full-data runs before sample runs (`smoke` / `quick` profiles,
   `--sample`; used only as a fallback, with a warning), finished runs (`DONE.json`)
   before interrupted ones, then the newest.

Feature names, classes and class names are read from the artifact
(`BaseTriageModel.save()`); a class missing in training gets probability 0.
Legacy artifacts without `label="esi"` are skipped. The backend serves the `triage`
feature set (`FEATURE_SET` in `.env`; `triage_only` = alias).

**Per-patient row** (`app/pipeline/feature_engineering.py`) — the same canonical schema
as training (`training/src/data/prepare.py` → `src/features/engineering.py`):

| feature | source |
|---|---|
| `triage_vital_hr/sbp/dbp/rr/o2` | form |
| `triage_vital_temp` | form, **°C — no conversion** (Yale °F is converted to °C in training) |
| `age` | form (models are trained on adults ≥ 18 y) |
| `is_male` | request `sex` (`M`/`F`), else NaN |
| `triage_vital_o2_device` | request `o2Device`, else NaN |
| `arrival_ambulance / _self / _other` (one-hot) | request `arrivalMode` (`ZRM`/`karetka` → ambulance, `własny` → self, `policja` → other); missing → **self** (as in training) |
| `arrival_month`, `arrival_weekday` (1–7), `arrival_hour_bin` (0–5) | system clock |
| `cc_*` chief complaints | Layer 0 parser — only real model columns |
| engineered features (NEWS2, qSOFA, shock index, ESI danger zone, …) | `engineer_features()` |

Missing values stay **NaN**, exactly as in training (tree models/EBM handle NaN natively,
logreg imputes inside its pipeline). Train medians (`models/imputation_medians.json`)
are **not** imputed by default — they serve as diagnostics and as the cc_* vocabulary.

A missing arrival mode is encoded as `self`, exactly as training encodes Yale's
missing mode. Collecting the real arrival mode (ZRM vs. own transport) in the form
is the single cheapest accuracy gain.

## Dependencies

`requirements.txt` pins the ML libraries to the training versions (xgboost 3.4,
lightgbm 4.7, catboost 1.2.10+, scikit-learn 1.9, interpret-core 0.7 for EBM,
numpy 2.5, pandas 3.0, shap 0.54) — pickled models are version-sensitive.
