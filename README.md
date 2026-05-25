# SOR-AI: System Klasyfikacji Triażu MTS oparty na ML

> Praca magisterska — wspomaganie decyzji triażowych na Szpitalnym Oddziale Ratunkowym

## Cel

System klasyfikuje pacjentów do jednej z 5 kategorii Manchester Triage System (Red/Orange/Yellow/Green/Blue) na podstawie danych dostępnych w momencie triażu (vital signs, główna skarga, demografia, historia medyczna). Dla każdej decyzji generowane są **dwie warstwy wyjaśnień**:

1. **Model-level (SHAP/LIME/DiCE)** — które cechy popchnęły model w stronę tej kategorii?
2. **Medical-level (lokalny LLM przez Ollama)** — dlaczego medycznie ta klasyfikacja ma sens?

## Architektura

```
INPUT PACJENTA → [Structured branch (XGBoost)] + [Text branch (BERT, opcjonalnie)]
              → [Late fusion (stacking/MLP)]
              → PREDYKCJA MTS 1-5
              → [SHAP] + [Ollama medical reasoning] + [MTS rule engine]
```

Pełna analiza techniczna: [`TECHNICAL_ANALYSIS.md`](TECHNICAL_ANALYSIS.md).

## Stack

| Warstwa | Technologie |
|---------|-------------|
| ML | scikit-learn, XGBoost, LightGBM, EBM (interpret) |
| DL | PyTorch + HuggingFace Transformers (HerBERT/BioClinBERT) |
| XAI | shap, lime, dice-ml |
| LLM | Ollama (Mistral / Llama3 / MedLlama2) |
| UI | Streamlit (demo), FastAPI (backend) |
| Tracking | MLflow |

## Dataset

**Yale EMMLC** (admissionprediction) — 560 486 wizyt × 972 zmienne, ESI 1–5 jako proxy dla MTS po wzbogaceniu o dyskryminatory parametrów życiowych (Strategia 2 z TECHNICAL_ANALYSIS.md §6).

- Plik: `data/raw/5v_cleandf.rdata`
- Konwersja: `python scripts/01_convert_rdata.py`

## Instalacja

```bash
# 1. Środowisko wirtualne
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# 2. Zależności
pip install -r requirements.txt

# 3. (Opcjonalnie) Ollama dla wyjaśnień medycznych
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
```

## Uruchomienie

```bash
# Krok 1 — konwersja .RData → .parquet
python scripts/01_convert_rdata.py

# Krok 2 — preprocessing + ESI→MTS mapping + zapis splitów
python scripts/02_preprocess.py

# Krok 3 — trening modeli
python scripts/03_train.py --model xgboost --tune

# Krok 4 — ewaluacja na test set
python scripts/04_evaluate.py --model xgboost

# Krok 5 — demo aplikacji
streamlit run app/streamlit_app.py
```

## Struktura projektu

```
sor-ai-triage/
├── training/              # ML training — kod, dane, eksperymenty
│   ├── src/               # modele, tuning, ewaluacja, explainability
│   │   ├── data/          # ładowanie, preprocessing, ESI→MTS, splity
│   │   ├── models/        # XGBoost, LightGBM, RF, EBM, fusion + tuning + train pipeline
│   │   ├── evaluation/    # metryki, CV, wizualizacje
│   │   ├── explain/       # SHAP, LIME, DiCE, MTS rules, Ollama
│   │   └── utils/         # config, logger, experiment_tracker
│   ├── scripts/           # CLI: 01_convert → 06_cross_validate
│   ├── notebooks/         # EDA, training, explainability
│   ├── app/               # Streamlit demo
│   ├── tests/             # testy jednostkowe ML (66+ testów)
│   ├── data/              # dane surowe i przetworzone
│   └── logs/              # logi treningu + JSON-y eksperymentów
├── backend/               # system produkcyjny (FastAPI + Spring Boot + Kafka)
│   ├── app/               # FastAPI ML inference service
│   ├── gateway/           # Spring Boot API gateway (Java 21)
│   ├── tests/             # testy backendu (79 testów, 85% coverage)
│   └── docker-compose.yml
├── frontend/              # React 18 + TypeScript + Tailwind + three.js
│   └── src/               # 93 pliki — journey, demo, i18n PL/EN
├── models/                # wytrenowane .joblib (wspólne dla training/ i backend/)
├── results/               # figury, tabele, raporty do pracy magisterskiej
```

## Logowanie i tracking eksperymentów

**Każdy** run treningu automatycznie generuje:

1. **Plik tekstowy** `logs/training/<model>/<run_id>.log` — pełna historia z konsoli
   (params, per-iteration metrics, czas treningu, top features).
2. **JSON eksperymentu** `logs/experiments/<model>_<run_id>.json` — strukturalne metadane:
   - `params` — wszystkie hiperparametry
   - `data` — info o splitach (n_train, n_val, class_distribution)
   - `training_history` — `mlogloss`/`merror` co iterację
   - `metrics` — końcowe QWK, AUC, undertriage, etc.
   - `feature_importances` — top 30 cech
   - `environment` — git_commit, python_version, hostname (reprodukowalność)
   - `artifacts` — ścieżki do modelu, logu, configu

Szczegóły: [`logs/README.md`](logs/README.md).

Opcjonalnie: trening z `--mlflow` dodatkowo loguje do MLflow (`mlflow ui`).

## Kluczowe metryki

| Metryka | Po co? |
|---------|--------|
| **Quadratic Weighted Kappa** | główna metryka — dystans ordinalny ma znaczenie |
| **Per-class AUC-ROC** | szczególnie dla Red i Orange (rzadkie, krytyczne) |
| **Undertriage rate** | **bezpieczeństwo** — pacjent Red błędnie zaklasyfikowany jako Yellow |
| **Overtriage rate** | efektywność oddziału |
| **5×5 confusion matrix** | wizualizacja błędów ordinalnych |

## Licencja

MIT (kod). Dataset Yale EMMLC: CC-BY (cytowanie wymagane — patrz §11 w `TECHNICAL_ANALYSIS.md`).
