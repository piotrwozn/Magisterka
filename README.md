# SOR-AI: system wspomagania triażu ESI oparty na ML

> Praca magisterska — wspomaganie decyzji triażowych na Szpitalnym Oddziale Ratunkowym

## Cel

System przypisuje pacjenta do jednego z 5 poziomów **Emergency Severity Index (ESI)**. To ten sam
algorytm, którym segregują pacjentów polskie SOR-y w systemie TOPSOR (Dz.U. 2021 poz. 1182,
§ 2 pkt 9), z kolorami TOPSOR:

| ESI | kolor | czas do lekarza |
|---|---|---|
| 1 | czerwony | natychmiast |
| 2 | pomarańczowy | do 10 min |
| 3 | żółty | do 60 min |
| 4 | zielony | do 120 min |
| 5 | niebieski | do 240 min |

Model uczy się na danych dostępnych w momencie triażu: parametry życiowe, skarga główna, wiek,
sposób przybycia (opcjonalnie także historia pacjenta). Każdą decyzję wyjaśniają dwie warstwy:

1. **Model-level (SHAP/LIME/DiCE)** — które cechy popchnęły model w stronę tego poziomu?
2. **Medical-level (lokalny LLM przez Ollama)** — dlaczego medycznie ta klasyfikacja ma sens?

## Architektura

```
INPUT PACJENTA → [modele tabelaryczne: XGBoost, LightGBM, CatBoost, RF, EBM, LogReg]
              + [warstwa tekstowa: notatka pielęgniarki (LLM, opcjonalnie)]
              → ręcznie dobrane połączenie modeli (na zbiorze walidacyjnym)
              → POZIOM ESI 1–5 (reguła argmax albo minimalnego kosztu niedotriażowania)
              → [SHAP] + [Ollama medical reasoning] + [kontrola parametrów życiowych]
```

Pełna analiza techniczna: [`training/SOR_AI_Technical_Analysis.md`](training/SOR_AI_Technical_Analysis.md).

## Stack

| Warstwa | Technologie |
|---------|-------------|
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, EBM (interpret), Optuna |
| XAI | shap, lime, dice-ml |
| LLM | Ollama (lokalnie) |
| UI | Streamlit (demo), React (frontend), FastAPI + Spring Boot (backend) |
| Tracking | JSON + logi per run, opcjonalnie MLflow |

## Dataset

**Yale EMMLC** ([admissionprediction](https://github.com/yaleemmlc/admissionprediction)):
560 486 wizyt × 972 zmienne, 3 oddziały w USA, natywne ESI 1–5.

Przygotowanie danych usuwa elementy typowe dla USA i błędy poprzedniej wersji pipeline'u:
- temperatura °F → °C,
- brak rasy, wyznania i ubezpieczenia wśród cech,
- jawne kodowanie kategorii,
- braki zostają jako NaN zamiast mediany,
- brak przecieku przez `disposition`.

Szczegóły i liczby: [`training/DATA_AUDIT.md`](training/DATA_AUDIT.md).

## Trening

```bash
cd training
pip install -r requirements.txt
python scripts/02_preprocess.py          # dane → splity ESI + raport z audytu
python scripts/03_train.py --model all   # 6 modeli → runs/ + runs/leaderboard.csv
python scripts/07_compare_runs.py        # porównanie + materiał do ręcznego łączenia
```

Pełna instrukcja (modele, tuning, metryki, łączenie modeli, czasy):
[`training/README.md`](training/README.md).

## Struktura projektu

```
Magisterka/
├── training/              # trening modeli ESI
│   ├── src/
│   │   ├── data/          # schemat Yale, przygotowanie danych, splity, dostęp do danych
│   │   ├── features/      # cechy kliniczne (°C, NEWS2, strefa zagrożenia ESI, braki)
│   │   ├── models/        # 6 modeli + trening + tuning Optuną
│   │   ├── ensemble/      # narzędzia do ręcznego łączenia modeli
│   │   ├── evaluation/    # metryki, CV, wykresy, niepewność
│   │   ├── explain/       # SHAP, LIME, DiCE, kontrola parametrów, Ollama
│   │   └── utils/         # konfiguracja, logowanie, wykrywanie GPU
│   ├── scripts/           # 01_convert → 07_compare_runs
│   ├── tests/
│   ├── app/               # demo Streamlit
│   ├── DATA_AUDIT.md      # raport z audytu danych (generowany)
│   └── runs/              # wyniki treningów (lokalnie, poza gitem)
├── backend/               # FastAPI (inferencja) + Spring Boot (gateway) + Kafka
└── frontend/              # React 18 + TypeScript
```

## Kluczowe metryki

| Metryka | Po co? |
|---------|--------|
| **Quadratic Weighted Kappa** | główna metryka — odległość między poziomami ma znaczenie |
| **Undertriage rate** | bezpieczeństwo — ESI 1–2 zaklasyfikowane jako ESI 3–5 |
| **Critical miss rate** | ESI 1 zaklasyfikowane jako ESI 4–5 |
| **Mistriage (under/over)** | błąd o dowolny poziom (standard w literaturze ESI) |
| **AUC, log loss, Brier** | jakość prawdopodobieństw (ważne przy łączeniu modeli) |
| **Expected cost** | średni koszt decyzji (niedotriażowanie 3× droższe) |

## Licencja

MIT (kod). Dataset Yale EMMLC: CC-BY (cytowanie wymagane).
