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
├── models/                              — joblib models + medians
├── scripts/generate_imputation_medians.py
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
python scripts/generate_imputation_medians.py     # one-off
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
| 1A    | Ensemble ML (7 models)   | ✓ CatBoost + LightGBM live; XGB+RF+ET+HGB+EBM after training |
| 1B    | MedGemma 27B reasoning   | stub (deterministic) — swap with Ollama call |
| 2     | Qwen3 32B synthesis      | not yet wired (conditional on conflict) |
| safety| Hardcoded clinical rules | ✓ live |
| SHAP  | Top-N explainer          | ✓ live (best loaded model) |

## Notes on current behaviour

The frontend collects only **7 vital signs + a clinical note**, but the trained
models expect **336 features** (incl. demographics, dep_name, arrival_mode,
ed_usage, 200 chief-complaint flags). The missing fields are filled with
training-set medians, which biases the model toward the **majority class
(Red, 68.6%)** for ambiguous inputs.

Critical cases (chest pain, stroke, sepsis) classify correctly. Mild cases
tend to **overtriage** — which is the **safer failure mode** for a triage
support system. Plugging in the real Llama 3.2 3B parser will materially
improve the cc_* extraction and reduce this bias.
