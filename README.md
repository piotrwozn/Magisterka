# SOR-AI: system wspomagania triażu ESI oparty na ML

> Praca magisterska — wspomaganie decyzji triażowych na Szpitalnym Oddziale Ratunkowym

## Cel

System przypisuje pacjenta do jednego z 5 poziomów **Emergency Severity Index (ESI)**. Tym samym
algorytmem segregują pacjentów polskie SOR-y w systemie TOPSOR (Dz.U. 2021 poz. 1182, § 2 pkt 9).
Kolory TOPSOR służą wyłącznie do prezentacji:

| ESI | kolor | czas do lekarza |
|---|---|---|
| 1 | czerwony | natychmiast |
| 2 | pomarańczowy | do 10 min |
| 3 | żółty | do 60 min |
| 4 | zielony | do 120 min |
| 5 | niebieski | do 240 min |

Model uczy się na danych dostępnych w chwili triażu: parametrach życiowych, skardze głównej,
wieku i sposobie przybycia (opcjonalnie także na historii pacjenta). Decyzję wyjaśniają dwie warstwy:

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

## Dane

**Yale EMMLC** ([admissionprediction](https://github.com/yaleemmlc/admissionprediction)):
560 486 wizyt × 972 zmienne, 3 oddziały w USA, natywne ESI 1–5.

Modele uczą się wyłącznie ogólnej, klinicznej informacji. Przygotowanie danych wyklucza:

- zmienne specyficzne dla USA i socjoekonomiczne: ubezpieczenie, rasę, pochodzenie etniczne,
  wyznanie, język, stan cywilny, zatrudnienie, eGFR liczony z rasą, `socialadmin`,
  „nieznany sposób przybycia” (amerykańska ścieżka szybkiej obsługi);
- przecieki z przyszłości wizyty: dyspozycję (przyjęcie/wypis) i listę leków z bieżącej wizyty;
- oddział szpitala oraz uszkodzone badania laboratoryjne.

Braki parametrów życiowych w Yale odzwierciedlają organizację pracy, więc wyniki raportujemy
także dla podgrupy z kompletem pomiarów (jak w TOPSOR). Szczegóły i uzasadnienia:
[`training/README.md`](training/README.md#10-dane-co-wykluczamy-i-dlaczego); liczby:
[`training/DATA_AUDIT.md`](training/DATA_AUDIT.md).

## Trening — jedno polecenie

```powershell
cd training
pip install -r requirements-lock.txt     # Python 3.12/3.13 x64; sterownik NVIDIA >= 580 dla GPU
python scripts/00_check_env.py           # kontrola środowiska
python scripts/run_all.py --profile smoke   # 5–10 min: test całego pipeline'u
python scripts/run_all.py                # tuning + trening 6 modeli, ewaluacja, porównanie (noc, wznawialne)
```

Instrukcja na domowy komputer (Windows/Linux, czasy, wznawianie, wyniki, łączenie modeli,
rozwiązywanie problemów): **[`training/README.md`](training/README.md)**.

## Struktura projektu

```
Magisterka/
├── training/              # trening modeli ESI
│   ├── src/
│   │   ├── data/          # schemat Yale, przygotowanie danych, splity
│   │   ├── features/      # cechy kliniczne (°C, NEWS2, qSOFA, strefa zagrożenia ESI, braki)
│   │   ├── models/        # 6 modeli + trening + tuning Optuną
│   │   ├── pipeline/      # run_all: orkiestrator, kontrola środowiska, blokada uśpienia
│   │   ├── ensemble/      # narzędzia do ręcznego łączenia modeli
│   │   ├── evaluation/    # metryki, CV, wykresy, niepewność
│   │   ├── explain/       # SHAP, LIME, DiCE, reguły ESI, Ollama
│   │   └── utils/         # konfiguracja, logowanie, sprzęt (CPU/GPU)
│   ├── scripts/           # 00_check_env, run_all, 01_convert … 07_compare_runs
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

Wybór modeli i łączenie odbywa się na zbiorze walidacyjnym; zbiór testowy służy wyłącznie do
ostatecznej oceny.

## Licencja

MIT (kod). Dataset Yale EMMLC: CC-BY (cytowanie wymagane).
