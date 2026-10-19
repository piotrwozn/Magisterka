# Trening modeli ESI — instrukcja

Tu trenujesz modele klasyfikujące pacjentów SOR do poziomów **ESI 1–5**, tak jak robi to polski
system TOPSOR (rozporządzenie MZ, Dz.U. 2021 poz. 1182, § 2 pkt 9). Etykietą jest natywne ESI
z danych Yale. Nie ma mapowania na MTS.

| ESI | kolor TOPSOR | czas do lekarza |
|---|---|---|
| 1 | czerwony | natychmiast |
| 2 | pomarańczowy | do 10 min |
| 3 | żółty | do 60 min |
| 4 | zielony | do 120 min |
| 5 | niebieski | do 240 min |

## Szybki start

```bash
cd training
pip install -r requirements.txt

python scripts/02_preprocess.py          # dane → splity ESI + DATA_AUDIT.md   (1–3 min)
python scripts/03_train.py --model all   # 6 modeli, zestaw 'triage'          (~1–1,5 h)
python scripts/07_compare_runs.py        # porównanie + materiał do łączenia
```

Skrót z Makefile'a: `make prepare`, `make train-all`, `make compare`. Krok 2 uruchamia się też sam,
jeśli splitów brakuje albo są nieaktualne.

Z tuningiem Optuną (ok. jednej nocy na i9 + RTX 4080):

```bash
python scripts/03_train.py --model all --tune
```

## Co robi przygotowanie danych (`scripts/02_preprocess.py`)

Pełny, automatycznie generowany raport: [`DATA_AUDIT.md`](DATA_AUDIT.md).

- Etykieta = ESI 1–5 (`target` 0–4). Odrzucone wizyty bez ESI (2 457) i duplikaty.
- Temperatura z Yale jest w °F i jest przeliczana na °C (dotyczy też temperatur z historii).
- Wartości fizjologicznie niemożliwe są zamieniane na brak, a nie na medianę.
- Braki zostają jako NaN. U 85% pacjentów ESI 1 brak **wszystkich** parametrów życiowych
  (od razu na salę resuscytacyjną), więc imputacja medianą kasowałaby najsilniejszy sygnał.
  Modele drzewiaste obsługują NaN natywnie, a regresja logistyczna imputuje wewnątrz własnego
  pipeline'u, wyłącznie na zbiorze treningowym.
- Wykluczone zmienne specyficzne dla USA: rasa, etniczność, wyznanie, ubezpieczenie,
  zatrudnienie, język, stan cywilny, eGFR liczony z rasą.
- Wykluczone przecieki: `disposition` (przyjęcie lub wypis to wynik wizyty) oraz oddział Yale.
- Kategorie są kodowane jawnie, a nie alfabetycznie:
  - sposób przybycia → `ambulance / self / police / other / unknown`,
  - miesiąc 1–12,
  - dzień tygodnia 1–7,
  - przedział godzin 0–5.
- Nazwy kolumn są znormalizowane, bo LightGBM nie przyjmuje przecinków ani nawiasów.
- Cechy kliniczne: wskaźnik wstrząsowy, MAP, NEWS2, SIRS, qSOFA, strefa zagrożenia ESI
  (HR > 100, RR > 20, SpO₂ < 92), flagi braków i grupy skarg głównych.
  Opis: [`FEATURE_ENGINEERING.md`](FEATURE_ENGINEERING.md).

Podział danych:

| zbiór | udział | do czego |
|---|---|---|
| train | 80% | trening; 10% z niego wydzielane w locie na early stopping |
| val | 10% | porównanie modeli, ręczne łączenie, kalibracja, progi |
| test | 10% | tylko ostateczna ocena — niczego na nim nie dostrajaj |

Domyślnie podział jest losowy, ze stratyfikacją po ESI × oddział. Yale nie ma roku ani
identyfikatora pacjenta, więc podział chronologiczny jest niemożliwy. Alternatywa:
`--split site`, w którym oddział C w całości tworzy test (walidacja zewnętrzna).

Zestawy cech:
- `triage` (276): to, co pielęgniarka ma w chwili triażu. To zestaw domyślny.
- `full` (1022): triaż oraz historia pacjenta (choroby, leki, wcześniejsze wizyty, wyniki badań).

## Modele

| model | sprzęt | dlaczego w zestawie |
|---|---|---|
| `logreg` | CPU | liniowy punkt odniesienia, w pełni przejrzysty |
| `lightgbm` | CPU | boosting, drzewa liść-po-liściu, najszybszy |
| `xgboost` | GPU/CPU | boosting, drzewa poziomowe, TreeSHAP |
| `catboost` | GPU/CPU | boosting, drzewa symetryczne, silna regularyzacja |
| `random_forest` | CPU | bagging, czyli inny mechanizm niż boosting (dywersyfikacja) |
| `ebm` | CPU | Explainable Boosting Machine: interpretowalny GAM |

Usunięte modele:
- `hist_gbt` powielał LightGBM,
- `extra_trees` powielał Random Forest,
- automatyczny `stacking` zastępuje ręczne łączenie (niżej).

GPU wybierane jest automatycznie; wymuszenie: `SOR_AI_DEVICE=cpu|gpu`.

```bash
python scripts/03_train.py --model xgboost                 # jeden model
python scripts/03_train.py --model lightgbm,catboost       # kilka modeli
python scripts/03_train.py --model all --feature-set full  # z historią pacjenta
python scripts/03_train.py --model all --weights balanced  # wagi klas (domyślnie brak)
python scripts/03_train.py --model all --sample 20000 --eval-sample 10000 --tag smoke  # szybki test
```

Domyślnie modele trenują się **bez wag klas**. Dzięki temu prawdopodobieństwa są skalibrowane,
co jest najlepsze do łączenia modeli. Asymetrię kosztów realizuje reguła decyzyjna (niżej).

### Tuning

`--tune` uruchamia Optunę na zbiorze treningowym (k-fold CV); `val` i `test` pozostają nietknięte.

- Domyślnie: 50 prób (EBM 15), 3 foldy, próbka 150 tys. wierszy.
- Cel: log loss, czyli jakość prawdopodobieństw.
- `--tune-metric qwk` optymalizuje QWK i dobiera też wagi klas.
- `--n-trials`, `--cv-folds`, `--tune-rows 0` (cały TRAIN) zmieniają te ustawienia.
- Historia zapisuje się w `logs/optuna/*.db`, więc przerwany tuning wznawia się od miejsca przerwania.

## Wyniki: `runs/`

Każdy trening zapisuje komplet w `runs/<data>_<model>_<zestaw>/`:

| plik | zawartość |
|---|---|
| `report.md` | metryki val/test, macierz pomyłek, top cechy |
| `metrics.json` | wszystkie metryki: `{val,test}_{argmax,cost}_<metryka>` |
| `predictions_val.parquet`, `predictions_test.parquet` | `row_id, site, esi, target, p_esi1…p_esi5, pred_argmax, pred_cost` |
| `model.joblib` | model z listą cech i metadanymi |
| `feature_importance.csv`, `run_info.json` | ważność cech, parametry, rozmiary zbiorów |

`runs/leaderboard.csv` ma jeden wiersz na run i służy do szybkiego porównania.

### Dwie reguły decyzyjne

- **argmax**: najbardziej prawdopodobny poziom ESI.
- **cost**: poziom minimalizujący oczekiwany koszt. Niedotriażowanie kosztuje 3× więcej niż
  przetriażowanie, a koszt rośnie z kwadratem odległości. Ta reguła mocno obniża
  niedotriażowanie ESI 1–2 kosztem nieco większego przetriażowania i niższego QWK, bo QWK karze
  symetrycznie.

### Metryki

- `quadratic_weighted_kappa` — główna metryka porządkowa.
- `undertriage_rate` — ESI 1–2 przewidziane jako ESI 3–5. To metryka bezpieczeństwa.
- `critical_miss_rate` — ESI 1 przewidziane jako ESI 4–5.
- `overtriage_rate` — ESI 4–5 przewidziane jako ESI 1–2.
- `mistriage_under` / `mistriage_over` — błąd o dowolny poziom.
- `auc_macro`, `log_loss`, `brier` — jakość prawdopodobieństw.
- `expected_cost` — średni koszt decyzji.

## Ręczne łączenie modeli

`python scripts/07_compare_runs.py` bierze najnowszy run każdego modelu i pokazuje:
- metryki na `val`,
- różnorodność modeli (korelacja błędów: im niższa, tym więcej daje połączenie),
- punkt odniesienia w postaci równowagowej średniej.

Raport trafia do `runs/comparison_*.md`.

Własne połączenie:

```python
from src.ensemble import latest_runs, aligned_probas, blend, evaluate, optimize_weights
from src.ensemble import fit_temperature, apply_temperature, diversity

runs = latest_runs("triage")
y_val, P_val = aligned_probas(runs, "val")
y_test, P_test = aligned_probas(runs, "test")

# 1. wybór i wagi — WYŁĄCZNIE na val
weights = {"xgboost": 0.4, "catboost": 0.3, "ebm": 0.3}
print(evaluate(y_val, blend(P_val, weights), rule="cost")["quadratic_weighted_kappa"])
print(optimize_weights(P_val, y_val))           # podpowiedź (min. log loss na val)

# 2. kalibracja np. Random Forest — temperatura dopasowana na val
T = fit_temperature(P_val["random_forest"], y_val)
P_test["random_forest"] = apply_temperature(P_test["random_forest"], T)

# 3. raport — ta sama receptura na test
final = evaluate(y_test, blend(P_test, weights), rule="cost")
```

## Ewaluacja, wyjaśnienia, CV

```bash
python scripts/04_evaluate.py --model-name xgboost            # wykresy i metryki runa (val)
python scripts/04_evaluate.py --model-name xgboost --split test
python scripts/05_explain.py --model-name xgboost --n-patients 5   # SHAP + kontrola reguł
python scripts/06_cross_validate.py --model lightgbm          # k-fold CV na TRAIN
streamlit run app/streamlit_app.py                            # demo
```

## Testy

```bash
python -m pytest tests -q            # wszystkie
python -m pytest tests -q -m "not slow"
```

## Ograniczenia danych (do rozdziału o metodologii)

- Yale to dane z USA (3 oddziały, 2014–2017). ESI ma ten sam algorytm co w TOPSOR, ale inna
  jest populacja i organizacja SOR.
- Brak identyfikatora pacjenta: ten sam pacjent może trafić do train i test.
- Wzorzec braków (np. ESI 1 bez parametrów życiowych) wynika z praktyki Yale. W TOPSOR pomiary
  są obowiązkowe, więc przy wdrożeniu trzeba to sprawdzić na polskich danych.
- Skargi główne to lista z systemu Epic (po angielsku). W Polsce trzeba je odwzorować na wywiad
  z karty segregacji (warstwa NLP).
