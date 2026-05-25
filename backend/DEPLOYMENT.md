# Deployment — SOR-AI Backend

End-to-end production runbook. Targets a single VPS / on-prem host running
Docker. Frontend is served separately by Cloudflare Pages.

---

## Prerequisites

| Component | Version | Why |
|---|---|---|
| Docker Engine | 24+ | container runtime |
| Docker Compose plugin | v2 | `docker compose` CLI |
| (optional) NVIDIA container toolkit | latest | GPU acceleration for Ollama |
| 32 GB RAM, 16 vCPU | recommended | LLM inference + ML models |
| 100 GB SSD | recommended | Ollama models occupy ~50 GB |
| TLS certificate | wildcard or specific | required if exposing publicly |

---

## Quick start (single host, no SSL)

```bash
# 1. Clone the repo on the deployment host
git clone https://github.com/<you>/Magisterka.git /opt/sorai
cd /opt/sorai

# 2. Generate the imputation medians from the train set (one-off)
cd backend
python3 -m pip install -r requirements.txt
python3 scripts/generate_imputation_medians.py
cd ..

# 3. Make sure Ollama models are present on host (or change the volume path)
ollama pull llama3
ollama pull medgemma:27b
ollama pull qwen3.6

# 4. Build & start the stack
docker compose -f backend/docker-compose.yml up -d --build

# 5. Smoke test
curl http://localhost/api/v1/health
```

`http://localhost/api/v1/health` should respond with:

```json
{
  "status": "ok",
  "modelsLoaded": ["catboost", "lightgbm"],
  "ollamaReady": true,
  "uptimeSeconds": 12
}
```

---

## Production checklist

### 1. TLS termination

Drop your fullchain certificate into `backend/nginx/certs/` and uncomment the
443 block at the bottom of `backend/nginx/nginx.conf`.

```bash
cp /etc/letsencrypt/live/sorai-triage.com/fullchain.pem  backend/nginx/certs/
cp /etc/letsencrypt/live/sorai-triage.com/privkey.pem    backend/nginx/certs/
```

### 2. Environment variables

`backend/.env` (loaded by docker-compose):

```bash
OLLAMA_HOST_DIR=/var/lib/ollama         # change to where Ollama keeps its models
AUDIT_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ALLOWED_ORIGINS=https://sorai-triage.com
```

### 3. Spring Boot prod profile

```bash
export SPRING_PROFILES_ACTIVE=prod
```

The actuator endpoints are exposed under `/actuator` but **deny-listed by
default in Nginx** to private networks only. Add a JWT filter if you need
to expose them publicly.

### 4. GPU access for Ollama

Uncomment the `deploy:` block in `docker-compose.yml`. Confirm with:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 5. Persistent volumes

Volumes that *must* survive a re-deploy:

| Volume | Contents |
|---|---|
| `kafka-data`   | audit log retention (default 30 days) |
| `ollama-data`  | LLM weights (~50 GB) |

Mount them on a separate disk in `/etc/docker/daemon.json`:

```json
{ "data-root": "/srv/docker" }
```

---

## Frontend wiring

`frontend/.env.production`:

```bash
VITE_API_URL=https://api.sorai-triage.com
VITE_USE_MOCK=false
```

Build + deploy to Cloudflare Pages:

```bash
cd frontend
npm run build
npx wrangler pages deploy dist --project-name sorai-triage --branch main
```

DNS:

```
api.sorai-triage.com    A    <vps-public-ip>
```

---

## Health monitoring

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/health`       | Public — frontend uptime probe |
| `GET /actuator/health`     | Internal — full Spring health (DB, Kafka, circuit breaker state) |
| `GET /actuator/prometheus` | Internal — metrics scrape for Prometheus |
| `GET http://fastapi:8000/api/v1/health` | Internal — ML service direct |

Recommended Prometheus alert rules:

```yaml
- alert: MlServiceDown
  expr: up{job="gateway"} == 0
  for: 1m

- alert: HighInferenceLatency
  expr: histogram_quantile(0.95, http_server_requests_seconds_bucket{uri="/api/v1/predict"}) > 5

- alert: CircuitBreakerOpen
  expr: resilience4j_circuitbreaker_state{state="open"} > 0
```

---

## Backup & restore

| Item | Frequency | Method |
|---|---|---|
| ML models (`models/*.joblib`) | once per retraining | rsync to S3 |
| Imputation medians (`backend/models/imputation_medians.json`) | once per retraining | git commit |
| Kafka audit topic | continuous | tiered storage to S3 |
| Optuna study DB (`logs/optuna_studies_*.db`) | once per tuning run | rsync to S3 |

---

## Rolling update (zero-downtime)

```bash
# 1. Build new images
docker compose -f backend/docker-compose.yml build

# 2. Restart services one at a time (Nginx → traffic continues)
docker compose -f backend/docker-compose.yml up -d --no-deps fastapi
sleep 30 && curl -fs http://localhost/api/v1/health

docker compose -f backend/docker-compose.yml up -d --no-deps gateway
sleep 30 && curl -fs http://localhost/api/v1/health
```

For full HA add a second `gateway2` and `fastapi2` replica in compose and let
Nginx `upstream { server gateway:8080; server gateway2:8080; }` balance.

---

## Smoke test (after every deploy)

```bash
curl -X POST http://localhost/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "vitals": {"age":67,"temp":38.2,"hr":118,"sbp":95,"dbp":62,"rr":22,"o2":94},
    "clinicalNote":"Pacjent z bólem w klatce piersiowej"
  }' | jq
```

Expected: `finalCategory ≤ 1`, `conflict.alertDoctor=true`.

---

## Rollback

```bash
docker compose -f backend/docker-compose.yml down
git checkout <previous-tag>
docker compose -f backend/docker-compose.yml up -d --build
```

Image tags follow the git short SHA — pin to the last known good in `image:`
fields for deterministic rollback.
