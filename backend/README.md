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

| Layer | Component                | Status |
|-------|--------------------------|--------|
| 0     | Llama 3.2 3B parser      | stub (keyword-based) — swap with Ollama call |
| 1A    | Ensemble ML (6 models)   | ✓ XGBoost, LightGBM, CatBoost, Random Forest, EBM, LogReg (whatever is trained) |
| 1B    | MedGemma 27B reasoning   | stub (deterministic) — swap with Ollama call |
| 2     | Qwen3 32B synthesis      | not yet wired (conditional on conflict) |
| safety| Hardcoded clinical rules | ✓ live |
| SHAP  | Top-N explainer          | ✓ live (best loaded tree model) |

## Models and features

**Label:** native ESI 1–5 (no mapping). Class index 0–4 = ESI 1–5; TOPSOR colours
ESI 1 red … ESI 5 blue are display only.

**Where models come from** (`app/models/registry.py`, first valid artifact wins per model):

1. `backend/models/<model>_*.joblib` — explicit deployment (copied into the Docker image),
2. `models/<model>_*.joblib` in the repo root,
3. `training/runs/<run>/model.joblib` — newest complete run (with `run_info.json`) per
   model for the served feature set; full-data runs win over `--sample` smoke runs
   (a smoke run is used only as a fallback, with a warning in the log).

Feature names and class names are read from the artifact (`BaseTriageModel.save()`).
Artifacts without `label="esi"` (pre-ESI, MTS-mapped) are skipped. The backend serves
the `triage` feature set (276 features, `FEATURE_SET` in `.env`; `triage_only` = alias).

**Per-patient row** (`app/pipeline/feature_engineering.py`) — the same canonical schema
as training (`training/src/data/prepare.py` → `src/features/engineering.py`):

| feature | source |
|---|---|
| `triage_vital_hr/sbp/dbp/rr/o2` | form |
| `triage_vital_temp` | form, **°C — no conversion** (Yale °F is converted to °C in training) |
| `age` | form (models are trained on adults ≥ 18 y) |
| `is_male`, `triage_vital_o2_device` | request if available, else NaN |
| `arrival_*` (one-hot) | request if available (`ZRM`/`karetka` → ambulance, `własny` → self, `policja` → police); missing → **self** |
| `arrival_month`, `arrival_weekday` (1–7), `arrival_hour_bin` (0–5) | system clock |
| `cc_*` (200 chief complaints) | Layer 0 parser — only real model columns |
| 59 engineered features (NEWS2, qSOFA, shock index, ESI danger zone, …) | `engineer_features()` |

Missing values stay **NaN**, exactly as in training (tree models/EBM handle NaN natively,
logreg imputes inside its pipeline). Train medians (`models/imputation_medians.json`)
are **not** imputed by default — they serve as diagnostics and as the cc_* vocabulary.

A missing arrival mode defaults to `self`, not `arrival_unknown`: in Yale the
"unknown" group is ~90% ESI 4–5, and forcing it on every request raised
undertriage from 17% to 56% on the validation split. Collecting the real arrival
mode (ZRM vs. own transport) in the form is the single cheapest accuracy gain.
