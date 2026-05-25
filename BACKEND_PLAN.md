# Backend SOR-AI — Plan implementacji

> Dokument opisuje pełną architekturę backendu dla systemu wspomagania decyzji
> triażowych. Frontend (`sorai-triage.com`) jest gotowy i oczekuje API opisanego poniżej.

---

## Spis treści

1. [Przegląd architektury](#1-przegląd-architektury)
2. [Stos technologiczny](#2-stos-technologiczny)
3. [Struktura katalogów](#3-struktura-katalogów)
4. [Warstwa 0 — Parser LLM (Llama 3.2 3B)](#4-warstwa-0--parser-llm-llama-32-3b)
5. [Warstwa 1A — Ensemble ML](#5-warstwa-1a--ensemble-ml)
6. [Warstwa 1B — Clinical NLP (MedGemma 27B)](#6-warstwa-1b--clinical-nlp-medgemma-27b)
7. [Warstwa 2 — Synteza (Qwen3 32B)](#7-warstwa-2--synteza-qwen3-32b)
8. [Reguły bezpieczeństwa (hardcoded)](#8-reguły-bezpieczeństwa-hardcoded)
9. [API — kontrakty z frontendem](#9-api--kontrakty-z-frontendem)
10. [Feature engineering w runtime](#10-feature-engineering-w-runtime)
11. [Imputation brakujących cech](#11-imputation-brakujących-cech)
12. [SHAP wyjaśnienia](#12-shap-wyjaśnienia)
13. [Kafka — audit log](#13-kafka--audit-log)
14. [Nginx — reverse proxy](#14-nginx--reverse-proxy)
15. [Kolejność implementacji](#15-kolejność-implementacji)
16. [Zmienne środowiskowe](#16-zmienne-środowiskowe)

---

## 1. Przegląd architektury

```
Pielęgniarka SOR
      │
      ▼
┌─────────────────────────────────────────┐
│           Frontend (React 18)           │
│        sorai-triage.com (Cloudflare)    │
└──────────────┬──────────────────────────┘
               │  POST /api/v1/predict
               │  { vitals, clinicalNote }
               ▼
┌─────────────────────────────────────────┐
│         Nginx (reverse proxy)           │
│  Port 443 → FastAPI ML service          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI ML Service (Python)                │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Endpoint POST /api/v1/predict                       │   │
│  │                                                      │   │
│  │  1. Walidacja (Pydantic)                             │   │
│  │  2. Warstwa 0: Llama 3.2 3B → parse clinical note   │   │
│  │  3. Feature engineering (vitals + cc_* z LLM)       │   │
│  │  4. Imputation brakujących cech (mediana train set)  │   │
│  │  5. Warstwa 1A: Ensemble 7 modeli → predykcja + SHAP │   │
│  │  6. Warstwa 1B: MedGemma 27B → reasoning + flags    │   │
│  │  7. Reguły bezpieczeństwa (hardcoded)                │   │
│  │  8. Warstwa 2: Qwen3 32B → synteza (jeśli konflikt) │   │
│  │  9. Kafka → audit log zdarzenia                      │   │
│  │  10. Response → frontend                             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│  Ollama (lokalny LLM)    │
│  Port 11434              │
│  - Llama 3.2 3B          │
│  - MedGemma 27B          │
│  - Qwen3 32B             │
└──────────────────────────┘
```

> **Kluczowe:** Dane pacjentów NIE opuszczają sieci szpitalnej.
> Ollama działa lokalnie na tej samej maszynie co FastAPI.

---

## 2. Stos technologiczny

| Komponent | Technologia | Wersja |
|---|---|---|
| ML inference API | FastAPI (Python) | 0.115+ |
| Walidacja danych | Pydantic v2 | 2.x |
| Serwer ASGI | Uvicorn | 0.30+ |
| Modele ML | joblib (CatBoost, XGBoost, LightGBM…) | — |
| SHAP | shap | 0.46+ |
| LLM serwer | Ollama | 0.5+ |
| HTTP client (Ollama) | httpx (async) | 0.27+ |
| Feature engineering | pandas + numpy | — |
| Reverse proxy | Nginx | 1.27 |
| Audit log | Apache Kafka | 3.7 |
| Kafka client | aiokafka | 0.11+ |
| Konteneryzacja | Docker + docker-compose | — |

> Spring Boot (Java) opisany w tech stacku frontendu jest opcjonalny —
> FastAPI może być bezpośrednią bramką. Priorytet: FastAPI first.

---

## 3. Struktura katalogów

```
backend/
├── app/
│   ├── main.py                  # FastAPI app, startup/shutdown
│   ├── config.py                # Settings (pydantic-settings)
│   ├── dependencies.py          # Dependency injection (ModelRegistry, OllamaClient)
│   │
│   ├── api/
│   │   ├── v1/
│   │   │   ├── predict.py       # POST /api/v1/predict
│   │   │   └── health.py        # GET  /api/v1/health
│   │   └── router.py
│   │
│   ├── models/
│   │   ├── schemas.py           # Pydantic: PredictRequest, PredictResponse, …
│   │   └── registry.py          # Ładowanie i cache modeli ML z dysku
│   │
│   ├── pipeline/
│   │   ├── orchestrator.py      # Główna logika predict() — łączy wszystkie warstwy
│   │   ├── layer0_parser.py     # Warstwa 0: Llama → parse → cc_* features
│   │   ├── layer1a_ml.py        # Warstwa 1A: ensemble inference + SHAP
│   │   ├── layer1b_nlp.py       # Warstwa 1B: MedGemma → reasoning
│   │   ├── layer2_synthesis.py  # Warstwa 2: Qwen3 → synteza (conditional)
│   │   ├── safety_rules.py      # Hardcoded reguły bezpieczeństwa
│   │   └── feature_engineering.py  # Runtime engineer_features() dla pojedynczego wiersza
│   │
│   ├── inference/
│   │   ├── ollama_client.py     # Async HTTP client do Ollama
│   │   └── shap_explainer.py    # SHAP TreeExplainer wrapper
│   │
│   └── kafka/
│       └── producer.py          # Kafka audit log producer
│
├── models/                      # Skopiowane z ../models/*.joblib
│   ├── catboost_*.joblib
│   ├── lightgbm_*.joblib
│   ├── xgboost_*.joblib
│   └── imputation_medians.json  # Mediany cech z train setu (do imputacji)
│
├── tests/
│   ├── test_predict.py
│   ├── test_feature_engineering.py
│   └── test_safety_rules.py
│
├── nginx/
│   └── nginx.conf
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 4. Warstwa 0 — Parser LLM (Llama 3.2 3B)

### Cel
Zamiana surowej notatki klinicznej pielęgniarki na **strukturyzowany JSON**
z flagami `cc_*` (200 binarnych chief complaint features) potrzebnymi przez modele ML.

### Prompt (structured output)

```python
PARSER_SYSTEM_PROMPT = """
Jesteś precyzyjnym parserem notatek triażowych SOR.
Wyodrębnij z notatki objawy pacjenta i zwróć TYLKO JSON zgodny ze schematem.
Nie dodawaj żadnych komentarzy poza JSON.
"""

PARSER_SCHEMA = {
    "type": "object",
    "properties": {
        "chief_complaints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Lista głównych dolegliwości (angielskie kody: chest_pain, dyspnea, etc.)"
        },
        "pain_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "altered_mental_status": {"type": "boolean"},
        "key_symptoms": {"type": "array", "items": {"type": "string"}},
        "urgency_signals": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["chief_complaints", "pain_score", "altered_mental_status"]
}
```

### Ollama call (structured output, temperatura 0)

```python
# app/pipeline/layer0_parser.py

async def parse_clinical_note(note: str, client: OllamaClient) -> dict:
    response = await client.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": PARSER_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Notatka triażowa: {note}"},
        ],
        format=PARSER_SCHEMA,    # Ollama structured output
        options={"temperature": 0, "seed": 42},
    )
    return response  # zawsze poprawny JSON
```

### Mapowanie na cc_* features

```python
# Lista 200 kodów cc_* pochodzi z train set (data/processed/train.parquet)
CC_CODES = [
    "chest_pain", "dyspnea", "abdominal_pain", "syncope",
    "headache", "altered_mental_status", ...  # 200 total
]

def map_cc_to_features(parsed: dict) -> dict[str, int]:
    """Zwraca {cc_chest_pain: 1, cc_dyspnea: 0, ...} — 200 binarnych cech."""
    result = {f"cc_{code}": 0 for code in CC_CODES}
    for complaint in parsed.get("chief_complaints", []):
        key = f"cc_{complaint.lower().replace(' ', '')}"
        if key in result:
            result[key] = 1
    if parsed.get("altered_mental_status"):
        result["cc_alteredmentalstatus"] = 1
    return result
```

### Latency
- Llama 3.2 3B na RTX 5090: **< 1s**
- Na CPU (fallback): ~3–5s

---

## 5. Warstwa 1A — Ensemble ML

### Modele do załadowania przy starcie

```python
# app/models/registry.py

MODELS_TO_LOAD = [
    "catboost",     # catboost_*.joblib    — status: trained ✓
    "lightgbm",     # lightgbm_*.joblib    — status: trained ✓
    "xgboost",      # xgboost_*.joblib     — status: trained po tuningu ✓
    "random_forest",    # po wytrenowaniu
    "extra_trees",      # po wytrenowaniu
    "hist_gbt",         # po wytrenowaniu
    "ebm",              # po wytrenowaniu
    # "stacking",       # meta-learner LogReg — po wytrenowaniu wszystkich
]
```

### Inference pipeline

```python
# app/pipeline/layer1a_ml.py

def predict_ensemble(features_df: pd.DataFrame, registry: ModelRegistry) -> EnsembleResult:
    predictions = {}
    probabilities = {}

    for model_name, model_data in registry.loaded_models.items():
        model       = model_data["model"]
        feat_names  = model_data["feature_names"]

        # Wyrównaj kolumny — uzupełnij brakujące imputacją
        X = align_features(features_df, feat_names, registry.imputation_medians)

        proba = model.predict_proba(X)
        predictions[model_name]  = int(np.argmax(proba[0]))
        probabilities[model_name] = proba[0].tolist()

    # Stacking (jeśli dostępny meta-learner)
    if "stacking" in registry.loaded_models:
        final_category, final_confidence = stacking_predict(probabilities, registry)
    else:
        # Fallback: ważona średnia (wg QWK każdego modelu)
        final_category, final_confidence = weighted_average(probabilities, MODEL_WEIGHTS)

    return EnsembleResult(
        final_category   = final_category,
        confidence       = final_confidence,
        model_predictions = [
            ModelPrediction(
                model_name   = name,
                category     = predictions[name],
                probabilities = probabilities[name],
                confidence   = max(probabilities[name]),
            )
            for name in predictions
        ],
    )
```

### Wagi modeli (fallback bez stacking)

```python
MODEL_WEIGHTS = {
    "catboost":     0.8729,   # real QWK test
    "lightgbm":     0.8705,   # real QWK test
    "xgboost":      0.876,    # aktualizować po treningu
    "random_forest":  0.84,   # est.
    "extra_trees":    0.83,   # est.
    "hist_gbt":       0.84,   # est.
    "ebm":            0.81,   # est.
}
```

### Latency
- 2–3 wytrenowane modele: **~50ms**
- 7 modeli: **~150ms**
- 7 modeli + stacking: **~200ms**

---

## 6. Warstwa 1B — Clinical NLP (MedGemma 27B)

### Cel
Kliniczne uzasadnienie decyzji — wychwytuje sygnały których dane tabelaryczne
nie zawierają (np. "pacjent blady, spocony, niespokojny").

### Prompt

```python
MEDGEMMA_SYSTEM = """
Jesteś asystentem klinicznym w SOR. Analizujesz notatki triażowe
i oceniasz pilność przypadku medycznego.
Zwróć TYLKO JSON z oceną kliniczną.
"""

MEDGEMMA_SCHEMA = {
    "type": "object",
    "properties": {
        "category":    {"type": "integer", "minimum": 0, "maximum": 4},
        "confidence":  {"type": "number",  "minimum": 0, "maximum": 1},
        "reasoning":   {"type": "string",  "maxLength": 400},
        "risk_flags":  {"type": "array",   "items": {"type": "string"}},
        "key_findings":{"type": "array",   "items": {"type": "string"}},
    },
    "required": ["category", "confidence", "reasoning", "risk_flags", "key_findings"]
}

def build_medgemma_prompt(vitals: Vitals, note: str, ml_category: int) -> str:
    return f"""
Parametry życiowe:
- SBP: {vitals.sbp} mmHg | HR: {vitals.hr} bpm | SpO2: {vitals.o2}%
- Temp: {vitals.temp}°C  | RR: {vitals.rr}/min | Wiek: {vitals.age} lat

Notatka: {note}

Model ML sugeruje kategorię MTS: {ml_category} ({MTS_NAMES[ml_category]})
Czy się zgadzasz? Jeśli nie — podaj uzasadnienie i proponowaną kategorię.
"""
```

### Latency
- MedGemma 27B na RTX 5090: **~3–5s**
- Uruchamiana **równolegle** z Warstwą 1A (asyncio.gather)

---

## 7. Warstwa 2 — Synteza (Qwen3 32B)

### Kiedy się uruchamia
**Tylko gdy** wykryto konflikt (różnica ≥ 2 stopnie MTS między modelami
lub między ML a MedGemma). W przeciwnym razie pomijana → oszczędność latency.

```python
# app/pipeline/layer2_synthesis.py

async def run_synthesis(
    ensemble: EnsembleResult,
    medgemma: MedGemmaAssessment,
    vitals: Vitals,
    note: str,
) -> SynthesisResult | None:
    if not should_run_synthesis(ensemble, medgemma):
        return None  # brak konfliktu → pomiń

    # Wywołaj Qwen3 32B
    ...
```

```python
def should_run_synthesis(ensemble: EnsembleResult, medgemma: MedGemmaAssessment) -> bool:
    """Uruchom syntezę tylko przy konflikcie."""
    ml_cat    = ensemble.final_category
    llm_cat   = medgemma.category
    max_model_diff = max(p.category for p in ensemble.model_predictions) \
                   - min(p.category for p in ensemble.model_predictions)
    return abs(ml_cat - llm_cat) >= 2 or max_model_diff >= 2
```

### Latency
- Qwen3 32B: **~2–4s** (uruchamiana tylko ~5% przypadków)

---

## 8. Reguły bezpieczeństwa (hardcoded)

Implementowane **poza LLM** — deterministyczne, niemożliwe do "zahalucynowania".

```python
# app/pipeline/safety_rules.py

def apply_safety_rules(
    final_category: int,
    model_predictions: list[ModelPrediction],
    medgemma: MedGemmaAssessment,
    vitals: Vitals,
) -> SafetyResult:

    alert_doctor = False
    messages = []

    # Reguła 1: Nigdy nie obniżaj poniżej minimum z modeli
    min_model_category = min(p.category for p in model_predictions)
    if final_category > min_model_category:
        final_category = min_model_category
        messages.append("Kategoria podniesiona do minimum z modeli.")

    # Reguła 2: Różnica ≥ 2 stopnie → alert lekarski
    max_diff = max(p.category for p in model_predictions) \
             - min(p.category for p in model_predictions)
    if max_diff >= 2:
        alert_doctor = True
        messages.append(f"Rozbieżność {max_diff} stopni MTS między modelami.")

    # Reguła 3: Confidence < 0.6 → zawsze flaguj
    if medgemma.confidence < 0.6:
        alert_doctor = True
        messages.append(f"Niska pewność MedGemma ({medgemma.confidence:.2f}).")

    # Reguła 4: Vitale krytyczne → minimum Red/Orange
    if vitals.sbp < 90 or vitals.o2 < 88 or vitals.hr > 150:
        if final_category > 1:
            final_category = 1
            messages.append("Krytyczne parametry życiowe → kategoria Orange minimum.")

    # Reguła 5: ML mówi Red (0) → zawsze alert
    if min_model_category == 0:
        alert_doctor = True

    return SafetyResult(
        final_category = final_category,
        alert_doctor   = alert_doctor,
        messages       = messages,
    )
```

---

## 9. API — kontrakty z frontendem

### `POST /api/v1/predict`

**Request** (dokładnie jak w `frontend/src/lib/types.ts`):
```json
{
  "vitals": {
    "age": 67,
    "temp": 38.2,
    "hr": 118,
    "sbp": 95,
    "dbp": 62,
    "rr": 22,
    "o2": 94
  },
  "clinicalNote": "Pacjent blady, spocony, ból w klatce promieniujący do żuchwy"
}
```

**Response**:
```json
{
  "finalCategory": 1,
  "confidence": 0.87,
  "modelPredictions": [
    {
      "modelName": "catboost",
      "category": 1,
      "probabilities": [0.12, 0.87, 0.01, 0.0, 0.0],
      "confidence": 0.87
    }
  ],
  "medgemma": {
    "category": 1,
    "confidence": 0.91,
    "reasoning": "Parametry życiowe wskazują na stan zagrożenia...",
    "riskFlags": ["hipotensja", "tachykardia"],
    "keyFindings": ["SBP=95", "HR=118"]
  },
  "shapTop5": [
    {"feature": "triage_vital_sbp", "value": 0.342, "direction": "positive"}
  ],
  "conflict": {
    "detected": false,
    "severity": "low",
    "alertDoctor": false,
    "message": "Wszystkie modele zgodne"
  },
  "processingTimeMs": 412
}
```

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "modelsLoaded": ["catboost", "lightgbm", "xgboost"],
  "ollamaReady": true,
  "uptimeSeconds": 3600
}
```

---

## 10. Feature engineering w runtime

Istniejący kod `src/features/engineering.py` musi działać na **pojedynczym wierszu** (1 pacjent),
nie na całym DataFrame. Wrapper:

```python
# app/pipeline/feature_engineering.py

import sys
sys.path.insert(0, "/path/to/Magisterka")  # lub jako pip package

from src.features.engineering import engineer_features
import pandas as pd

def build_feature_row(vitals: Vitals, cc_features: dict, extra: dict = {}) -> pd.DataFrame:
    """
    Buduje pojedynczy wiersz DataFrame z 336 featurami dla modelu.

    vitals      : dane z formularza
    cc_features : {cc_chest_pain: 1, cc_dyspnea: 0, ...} z Layer 0
    extra       : opcjonalne cechy (arrival_hour, arrival_mode, etc.)
    """
    row = {
        # Vitals
        "triage_vital_hr":   vitals.hr,
        "triage_vital_sbp":  vitals.sbp,
        "triage_vital_dbp":  vitals.dbp,
        "triage_vital_rr":   vitals.rr,
        "triage_vital_o2":   vitals.o2,
        "triage_vital_temp": vitals.temp,
        # Demographics
        "age": vitals.age,
        # Czas przybycia (auto)
        "arrival_hour":      pd.Timestamp.now().hour,
        "arrival_dayofweek": pd.Timestamp.now().dayofweek,
        # CC features z LLM parser
        **cc_features,
        # Opcjonalne dodatkowe
        **extra,
    }

    df = pd.DataFrame([row])
    df = engineer_features(df)   # dodaje shock_index, MEWS, SIRS, qSOFA, etc.
    return df
```

---

## 11. Imputation brakujących cech

Modele oczekują 336 cech. Przy wejściu z formularza część będzie brakować
(np. demographics, ed_usage). Uzupełniamy **medianą z train setu**.

### Generowanie pliku medián (jednorazowo)

```bash
# Uruchom raz lokalnie, zapisz do backend/models/
python scripts/generate_imputation_medians.py
```

```python
# scripts/generate_imputation_medians.py
import json, pandas as pd
from src.models.train import load_splits
from src.data.preprocessing import build_feature_groups, split_features
from src.features.engineering import engineer_features

splits = load_splits()
df = engineer_features(splits["train"])
groups = build_feature_groups(df)
X, _, _ = split_features(df, groups)

medians = X.median().to_dict()
with open("backend/models/imputation_medians.json", "w") as f:
    json.dump(medians, f)

print(f"Saved {len(medians)} feature medians")
```

### Użycie w pipeline

```python
def align_features(
    df: pd.DataFrame,
    expected_features: list[str],
    medians: dict[str, float],
) -> pd.DataFrame:
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = medians.get(feat, 0.0)
    return df[expected_features]
```

---

## 12. SHAP wyjaśnienia

```python
# app/inference/shap_explainer.py

import shap, numpy as np

class ShapExplainer:
    def __init__(self, model, feature_names: list[str]):
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names

    def top5(self, X: pd.DataFrame) -> list[dict]:
        shap_values = self.explainer.shap_values(X)

        # Multi-class: użyj klasy z max confidence
        predicted_class = int(np.argmax(
            self.explainer.expected_value if hasattr(...) else shap_values[0].sum(axis=1)
        ))
        values = shap_values[predicted_class][0]  # shape (n_features,)

        # Top 5 absolutnie
        top_idx = np.argsort(np.abs(values))[-5:][::-1]

        return [
            {
                "feature":   self.feature_names[i],
                "value":     float(values[i]),
                "direction": "positive" if values[i] > 0 else "negative",
            }
            for i in top_idx
        ]
```

> **Uwaga:** SHAP dla CatBoost i LightGBM działa natywnie z TreeExplainer.
> Dla XGBoost użyj `shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")`.

---

## 13. Kafka — audit log

Każda predykcja zapisywana asynchronicznie — nie blokuje odpowiedzi.

```python
# app/kafka/producer.py

from aiokafka import AIOKafkaProducer
import json, asyncio

TOPIC = "triage-predictions"

async def log_prediction(producer: AIOKafkaProducer, request, response, duration_ms: int):
    event = {
        "timestamp":     pd.Timestamp.now().isoformat(),
        "vitals":        request.vitals.model_dump(),
        "final_category": response.finalCategory,
        "confidence":    response.confidence,
        "alert_doctor":  response.conflict.alertDoctor,
        "processing_ms": duration_ms,
        # Notatka kliniczna NIE jest logowana (RODO)
    }
    await producer.send(TOPIC, json.dumps(event).encode())
```

> Kafka jest opcjonalna dla MVP — można pominąć na początku i dodać później.

---

## 14. Nginx — reverse proxy

```nginx
# nginx/nginx.conf

upstream ml_service {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl;
    server_name sorai-triage.com;

    # SSL (certbot / Cloudflare origin cert)
    ssl_certificate     /etc/ssl/certs/sorai-triage.pem;
    ssl_certificate_key /etc/ssl/private/sorai-triage.key;

    # CORS (dla Cloudflare Pages)
    add_header Access-Control-Allow-Origin "https://sorai-triage.com";
    add_header Access-Control-Allow-Methods "POST, GET, OPTIONS";

    # Rate limiting
    limit_req zone=api burst=20 nodelay;

    location /api/ {
        proxy_pass         http://ml_service;
        proxy_read_timeout 60s;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
    }
}

server {
    listen 80;
    server_name sorai-triage.com;
    return 301 https://$host$request_uri;
}
```

---

## 15. Kolejność implementacji

### Faza 1 — MVP (2 tygodnie)
```
[ ] 1. FastAPI skeleton + health endpoint
[ ] 2. Model registry — ładowanie CatBoost + LightGBM przy starcie
[ ] 3. Feature engineering wrapper (runtime single-row)
[ ] 4. Imputation medians (generate + load)
[ ] 5. Layer 1A — ensemble inference (bez SHAP)
[ ] 6. Safety rules (hardcoded)
[ ] 7. POST /api/v1/predict — działający endpoint (bez LLM)
[ ] 8. Test z frontendem (wyłącz USE_MOCK = false)
```

### Faza 2 — LLM integration (1 tydzień)
```
[ ] 9.  Ollama client (async httpx)
[ ] 10. Layer 0 — Llama parser + cc_* mapping
[ ] 11. Layer 1B — MedGemma reasoning
[ ] 12. Conflict detection
[ ] 13. Layer 2 — Qwen3 (warunkowe)
```

### Faza 3 — SHAP + stacking (3 dni)
```
[ ] 14. SHAP explainer per model
[ ] 15. Stacking meta-learner (po wytrenowaniu wszystkich 7 modeli)
[ ] 16. Aktualizacja model registry
```

### Faza 4 — Production (1 tydzień)
```
[ ] 17. Nginx config + SSL
[ ] 18. Docker + docker-compose
[ ] 19. Kafka audit log (opcjonalne)
[ ] 20. Load testing
```

---

## 16. Zmienne środowiskowe

```bash
# backend/.env

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2

# Modele
MODELS_DIR=/app/models
FEATURE_SET=triage_only

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_PARSER_MODEL=llama3.2:3b
OLLAMA_NLP_MODEL=medgemma:27b
OLLAMA_SYNTHESIS_MODEL=qwen3:32b
OLLAMA_TIMEOUT_S=30

# Kafka (opcjonalne)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=triage-predictions

# CORS
ALLOWED_ORIGINS=https://sorai-triage.com,http://localhost:5173

# SHAP
SHAP_TOP_N=5
SHAP_ENABLED=true

# Safety
CONFLICT_THRESHOLD=2       # różnica stopni MTS triggering alert
CONFIDENCE_THRESHOLD=0.6   # min confidence MedGemma
```

---

## Frontend — jedyna zmiana po uruchomieniu backendu

W `frontend/src/hooks/usePredict.ts` zmień:

```typescript
// PRZED (mock):
const USE_MOCK = true;

// PO (prawdziwy backend):
const USE_MOCK = false;
// lub ustaw VITE_API_URL=https://api.sorai-triage.com w .env.production
```

Kontrakt API jest już zaimplementowany — frontend jest gotowy.

---

*Dokument wygenerowany: 2026-05-24*
*Wersja: 1.0*
