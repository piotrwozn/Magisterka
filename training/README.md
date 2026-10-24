# Trening modeli ESI — instrukcja na domowy komputer

Tu trenujesz modele, które przypisują pacjenta SOR do jednego z pięciu poziomów **Emergency Severity
Index (ESI 1–5)**. To ten sam algorytm, którym segregują pacjentów polskie SOR-y w systemie TOPSOR
(Dz.U. 2021 poz. 1182, § 2 pkt 9). Etykietą jest natywne ESI z danych Yale EMMLC. Kolory TOPSOR
służą wyłącznie do prezentacji:

| ESI | kolor TOPSOR | czas do lekarza |
|---|---|---|
| 1 | czerwony | natychmiast |
| 2 | pomarańczowy | do 10 min |
| 3 | żółty | do 60 min |
| 4 | zielony | do 120 min |
| 5 | niebieski | do 240 min |

## W skrócie

```powershell
cd C:\dev\Magisterka\training            # w aktywnym venv (instalacja: niżej)
python scripts/00_check_env.py           # 1. kontrola środowiska: GPU, RAM, dysk, wersje, dane
python scripts/run_all.py --profile smoke   # 2. test całego pipeline'u: 5–10 min
python scripts/run_all.py                # 3. właściwy przebieg na noc (profil standard)
```

Po awarii, restarcie albo Ctrl+C uruchom **to samo polecenie** — gotowe modele zostaną pominięte,
a przerwany tuning dokończy tylko brakujące próby. Stan w dowolnej chwili: `python scripts/run_all.py --status`.

Na noc (Windows): **Tryb zasilania „Najlepsza wydajność”**, okno konsoli otwarte (nie minimalizuj go)
i nie klikaj w nie — szczegóły w punkcie 5.

---

## 1. Wymagania

| | |
|---|---|
| System | Windows 11 x64 albo Linux x64 (macOS działa bez GPU) |
| Python | **3.13 x64** (przetestowany) albo 3.12 x64. **Nie** 3.11 i starsze (brak kół xgboost 3.4, numpy 2.5, shap 0.54); 3.14 nieprzetestowany |
| Karta | NVIDIA ze sterownikiem **≥ 580** — koło xgboost 3.4 jest zbudowane z CUDA 13. CUDA Toolkit **nie** jest potrzebny (koła mają własne biblioteki CUDA). Bez karty wszystko działa na CPU, tylko wolniej |
| RAM | **32 GB zalecane**; minimum 16 GB (przygotowanie danych chwilowo zajmuje ~11 GB) |
| Dysk | ≥ 20 GB wolnego (jeden pełny przebieg dokłada 1–2 GB do `runs/`) |
| Ścieżka | repo **poza OneDrive**, bez polskich znaków, krótka — np. `C:\dev\Magisterka` |

Punkt odniesienia dla czasów w tej instrukcji: i9-14900K (24 rdzenie fizyczne) + RTX 4080 Super 16 GB + 64 GB RAM.

## 2. Instalacja

### Windows (PowerShell)

```powershell
nvidia-smi                                   # musi pokazać "Driver Version: 580+" i "CUDA Version: 13.x"
git clone <adres-repo> C:\dev\Magisterka     # NIE do Dokumentów/Pulpitu (OneDrive)

py -3.13 -m venv C:\venvs\sorai              # Python 3.13 x64 z python.org (launcher `py`)
C:\venvs\sorai\Scripts\Activate.ps1          # zablokowane? Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -U pip
cd C:\dev\Magisterka\training
pip install -r requirements-lock.txt         # dokładnie przetestowane wersje
setx PYTHONUTF8 1                            # UTF-8 w Pythonie — działa w NOWYCH oknach PowerShell
```

Za stary sterownik: zaktualizuj go (NVIDIA App / sterownik Game Ready lub Studio) i uruchom ponownie komputer.

### Linux

```bash
nvidia-smi                                   # sterownik >= 580 (koło xgboost dla Linuksa też jest CUDA 13)
sudo apt install git python3.13-venv         # Ubuntu/Debian: moduł venv to osobny pakiet (albo python3.12-venv)
git clone <adres-repo> ~/Magisterka
python3.13 -m venv ~/venvs/sorai && source ~/venvs/sorai/bin/activate   # albo python3.12
python -m pip install -U pip
cd ~/Magisterka/training
pip install -r requirements-lock.txt
```

Ubuntu 24.04 ma w systemie Pythona 3.12 (`python3.12-venv`, `python3.12 -m venv …`) — to wystarczy;
3.13 jest m.in. w PPA deadsnakes. Bez pakietu `python3.X-venv` polecenie `python3.X -m venv` kończy
się błędem „ensurepip is not available”.

WSL2: domyślnie dostaje połowę RAM hosta — ustaw `memory=` w `%UserProfile%\.wslconfig` i trzymaj repo
w systemie plików Linuksa (`~/…`), nie w `/mnt/c`. Blokada uśpienia `run_all.py` działa wtedy tylko
w maszynie wirtualnej i **nie** powstrzyma Windows przed uśpieniem (orkiestrator ostrzega w logu):
na Windows ustaw uśpienie przy zasilaniu sieciowym na „Nigdy”.

### Pliki z zależnościami

| plik | zawartość |
|---|---|
| `requirements-lock.txt` | **przetestowany zestaw** (dokładne wersje: rdzeń + pytest + zależności przechodnie) — instaluj ten |
| `requirements.txt` | rdzeń z zakresami wersji (gdy lock nie pasuje do platformy) |
| `requirements-app.txt` | demo Streamlit, LIME, DiCE (`pip install -r requirements-app.txt` po locku) |
| `requirements-dev.txt` | pytest, ruff, black; MLflow opcjonalnie (zakomentowany) |

## 3. Kontrola środowiska

```powershell
python scripts/00_check_env.py
```

Sprawdza Pythona, pakiety (wersje względem `requirements-lock.txt`), RAM, wolne miejsce, rdzenie,
kartę (`nvidia-smi`) i robi **prawdziwy próbny trening na GPU** dla xgboost i catboost. Sprawdza
też dane Yale, ścieżkę repo (OneDrive, polskie znaki, długość) i kodowanie konsoli. Wynik
trafia na konsolę i do `logs/run_all/<czas>/preflight.json`.

- `[BŁĄD ]` — trzeba poprawić (kod wyjścia 1); `run_all.py` też się wtedy nie uruchomi.
  Brakujące pakiety (np. świeży venv) skrypt wypisuje razem z poleceniem instalacji.
- `[UWAGA]` — trening ruszy. Przykłady: GPU niedostępne → xgboost/catboost na CPU z mniejszym
  budżetem; Windows z planem zasilania innym niż wysoka wydajność.
- Ostatnia linia mówi, czy `run_all.py` użyje dwóch torów (GPU + CPU równolegle).

## 4. Test całości: profil `smoke` (5–10 minut)

```powershell
python scripts/run_all.py --profile smoke
```

Przy pierwszym uruchomieniu najpierw przygotowuje dane (3–5 min, chwilowo ~11 GB RAM). Potem
trenuje wszystkie 6 modeli na małej próbce (6 tys. wierszy, 2 próby Optuny), ocenia je na `val`
i porównuje. Sprawdź podsumowanie `runs/run_all_summary__triage__smoke__<czas>.md`:

- kolumna **urządzenie** dla xgboost i catboost powinna pokazywać GPU (`cuda` / `gpu`), nie `cpu`,
- wszystkie modele mają stan `gotowy`.

Wyniki smoke są oznaczone tagiem `smoke`. Nie nadają się do raportowania.

## 5. Właściwy przebieg

```powershell
python scripts/run_all.py                    # = --profile standard --feature-set triage --models all
```

Kroki (każdy w **osobnym procesie**: awaria jednego modelu, błąd CUDA albo brak pamięci zabija
tylko ten krok, a nie całą noc):

1. **preflight** — jak w punkcie 3;
2. **dane** — `02_preprocess.py`, tylko gdy splity nie są gotowe albo mają inny podział;
3. dla każdego modelu: **tuning Optuną + trening finalny** (`03_train.py --tune --profile standard`),
   potem **ewaluacja na `val`** z wykresami (`04_evaluate.py`);
4. **porównanie** modeli na `val` (`07_compare_runs.py`);
5. **podsumowanie** `runs/run_all_summary__triage__standard__<czas>.md`.

**Szacowany czas** (i9-14900K + RTX 4080 Super, zestaw `triage`): **~7–8 h w dwóch torach**,
~11–12 h sekwencyjnie. To szacunki ±50%. Każdy model ma limit czasu tuningu, więc górną granicę
wyznacza suma limitów; `run_all.py` wypisuje ją na starcie (ETA każdego modelu).

**Dwa tory** (`--lanes auto`) włączają się, gdy co najmniej jeden model idzie na GPU (próba GPU
w preflight się powiodła), CPU ma ≥ 16 rdzeni fizycznych i jest ≥ 30 GB RAM (komputer z 32 GB
pokazuje ~31,8 GB):

- tor GPU: tylko modele na GPU (xgboost → catboost), 6 wątków CPU;
- tor CPU: pozostałe modele (logreg → lightgbm → random_forest → ebm, a także xgboost/catboost,
  jeśli dla nich GPU nie działa), pozostałe rdzenie fizyczne.

Budżety wątków się nie nakładają (nadsubskrypcja OpenMP potrafi spowolnić trening kilkadziesiąt
razy). Jeden tor: `--lanes 1`.

### Przed nocnym przebiegiem (Windows)

- **Zasilanie**: Ustawienia → System → Zasilanie (i bateria) → *Tryb zasilania*: **„Najlepsza wydajność”**
  (przy zasilaniu sieciowym; albo plan „Wysoka wydajność”). Preflight ostrzega, gdy jest inaczej.
- **Nie minimalizuj okna konsoli** — Windows 11 może zdławić zminimalizowany proces (EcoQoS: rdzenie E,
  niższe taktowanie). `run_all.py` i `03_train.py` wyłączają to dla siebie, ale zostaw okno otwarte.
- **Nie klikaj w okno konsoli.** Tryb QuickEdit zaznacza wtedy tekst i wstrzymuje każdy zapis na
  konsolę — trening staje do naciśnięcia klawisza. `run_all.py` wyłącza QuickEdit na czas przebiegu
  i przywraca go na końcu; przy ręcznie uruchamianych skryptach naciśnij Esc, jeśli w tytule okna
  pojawi się „Zaznacz”.

### Profile

| profil | do czego | dane | tuning | czas |
|---|---|---|---|---|
| `smoke` | test instalacji | próbka 6 tys. (val/test 4 tys.) | 2 próby, 2 foldy, ≤ 2 min/model | 5–10 min |
| `quick` | test pipeline'u na większej próbce | próbka 60 tys. | 3 próby na 20 tys. wierszy, 2 foldy, ≤ 5 min/model | 20–30 min |
| `standard` | **właściwy przebieg** (domyślny) | całość | tabela niżej | ~7–8 h (2 tory) |
| `thorough` | dłuższy tuning | całość | ≈ 1,7× próby i czas `standard`; EBM na 250 tys. | ~12–14 h (2 tory) |

Budżet `standard` (próby i limit czasu dotyczą całego badania Optuny, także po wznowieniach):

| model | urządzenie | wiersze do tuningu | foldy | próby | limit tuningu |
|---|---|---|---:|---:|---:|
| `logreg` | CPU | cały zbiór dopasowania | 3 | 20 | 45 min |
| `lightgbm` | CPU | cały | 3 | 120 | 2 h |
| `xgboost` | GPU | cały | 3 | 100 | 3,5 h |
| `catboost` | GPU | cały | 3 | 60 | 4 h |
| `random_forest` | CPU | 150 tys. | 3 | 40 | 1,5 h |
| `ebm` | CPU | 150 tys. | 3 | 25 | 4 h |

Bez GPU xgboost i catboost dostają budżet zastępczy na CPU: xgboost 50 prób na 150 tys. wierszy
w 3 h, catboost 25 prób na 100 tys. w 4 h. Pełne definicje: `TUNING_PROFILES` i
`CPU_FALLBACK_BUDGETS` w `src/utils/config.py`.

Tuning minimalizuje **log loss** w k-krotnej walidacji krzyżowej na części TRAIN. Foldy są
stratyfikowane po ESI × oddział, a early stopping liczony jest na wycinku treningowej części foldu.
Zbiory `val` i `test` nie biorą udziału w tuningu ani w treningu.

### Opcje `run_all.py`

| opcja | znaczenie |
|---|---|
| `--profile smoke\|quick\|standard\|thorough` | budżet (domyślnie `standard`) |
| `--feature-set triage\|full\|triage,full` | zestaw cech (domyślnie `triage`); `full` ≈ 3–4× dłużej na próbę i ≥ 32 GB RAM — osobna noc |
| `--models all\|lista` | np. `--models lightgbm,xgboost` (aliasy: lr, lgbm, xgb, cb, rf) |
| `--split random\|site` | podział danych; `site` = oddział B w całości jako test (walidacja zewnętrzna) |
| `--no-tune` | bez tuningu, parametry domyślne (tag `<profil>-notune`; takich runów nie biorą domyślnie `latest_runs` / `07_compare_runs.py`) |
| `--lanes auto\|1\|2` | tory równoległe |
| `--force` | trenuj ponownie także gotowe modele (ukończone próby Optuny i tak zostaną użyte); po przerwaniu polecenie wznowienia zachowuje `--force` tylko dla niedokończonych modeli |
| `--fail-fast` | pierwszy nieudany model zatrzymuje kolejne (domyślnie pozostałe trenują dalej) |
| `--dry-run` | pokaż plan, ETA i polecenia, nic nie uruchamiaj |
| `--status` | stan z plików-znaczników, nic nie uruchamiaj |
| `--skip-preflight`, `--no-keep-awake` | pomiń kontrolę środowiska / blokadę uśpienia |

Na Linuksie są też skróty z Makefile'a: `make check-env`, `make smoke`, `make all`, `make status`.

## 6. Wznawianie, stan, przerwanie

- **Wznowienie = to samo polecenie.** Model jest pominięty, gdy istnieje
  `runs/*_<model>_<zestaw>_<profil>/DONE.json` z tą samą wersją danych (`esi-v2`) i tym samym
  odciskiem danych (skrót `manifest.json` bez czasu i ścieżki przygotowania), zwykły run profilu
  (nie `custom`), z tym samym trybem tuningu. Model zaplanowany na GPU musi mieć run **z GPU** —
  run, który z powodu błędu karty skończył się na CPU, zostanie powtórzony, gdy GPU znów działa
  (log mówi dlaczego). Brakująca ewaluacja zostanie dorobiona.
- **Przerwany tuning** wznawia się od miejsca przerwania. Badanie Optuny (`logs/optuna/optuna_<model>.db`)
  liczy ukończone próby i zużyty czas. Próba przerwana awarią procesu jest oznaczana jako błąd
  i ponawiana raz.
- **Ctrl+C**: orkiestrator nie uruchamia już nowych kroków i czeka do 2 minut, aż bieżące się
  zakończą (drugie Ctrl+C = natychmiast). Potem zapisuje podsumowanie i wypisuje polecenie wznowienia
  (po `--force`: z `--force` i `--models` zawężonym do niedokończonych, osobno dla każdego zestawu
  cech, a na końcu zwykłe polecenie bez `--force`, które dorobi brakujące ewaluacje i porównanie).
  Kod wyjścia to 130.
- **Twarda awaria kroku** (np. błąd natywny, zabity proces) nie blokuje toru: orkiestrator zabija
  pozostałości kroku (Linux: całą grupę procesów) i przechodzi do następnego modelu.
- **Restart / zanik prądu**: po prostu uruchom to samo polecenie.
- **Co unieważnia gotowe modele**: inne dane (inne wyniki przygotowania → inny odcisk), zmiana
  podziału (`--split`) i nowa wersja logiki danych w kodzie. Samo ponowne przygotowanie tych samych
  danych tym samym kodem (np. po `make clean`) niczego nie unieważnia. Stare runy zostają w `runs/`.
- **Stan**: `python scripts/run_all.py --status` (dla profilu innego niż standard dodaj `--profile …`).
  Pokazuje gotowe modele z QWK na `val`, postęp tuningu (liczbę prób w badaniu) i przerwane runy.
  Gdy proces orkiestratora już nie działa (awaria, restart), niedokończone kroki mają stan
  „przerwany (proces nie działa)” — wznów tym samym poleceniem.
- xgboost/catboost z **błędem GPU** w logu (CUDA, brak pamięci karty) są automatycznie powtarzane
  raz na CPU. Podsumowanie pokazuje to jako „powtórka na CPU”. Powtórka dostaje wątki toru CPU,
  jeśli ten tor skończył już swoje modele; inaczej działa z 6 wątkami toru GPU (log ostrzega, że
  będzie wolniej).

Kody wyjścia `run_all.py`: 0 = wszystko gotowe, 1 = któryś krok nieudany, 2 = błąd preflight
albo przygotowania danych, 130 = przerwano.

## 7. Wyniki: gdzie są i jak je czytać

**Najpierw** `runs/run_all_summary__<zestaw>__<profil>__<czas>.md`. Zawiera jeden wiersz na model:

| kolumna | znaczenie |
|---|---|
| stan | `gotowy`, `gotowy wcześniej` (pominięty przy wznowieniu), `NIEUDANY` (sekcja „Błędy” z końcówką logu), `przerwany`, `nieuruchomiony` |
| urządzenie | faktyczne urządzenie treningu finalnego (`cuda`/`gpu` = karta); „fallback”/„powtórka na CPU” = GPU zawiodło |
| próby | ukończone / przycięte (pruning) / błędne próby Optuny |
| CV | najlepszy log loss z walidacji krzyżowej na TRAIN ± odchylenie między foldami |
| QWK argmax / koszt | Quadratic Weighted Kappa na `val` dla obu reguł decyzyjnych |
| undertriage (koszt) | odsetek ESI 1–2 przewidzianych jako ESI 3–5 (reguła kosztowa) |
| QWK vc | QWK w podgrupie z kompletem HR, SBP i RR przy triażu (`vitals_complete == 1`), czyli populacji jak w TOPSOR |

Podsumowanie i porównanie pokazują **tylko metryki VAL**. Zbiór testowy oceniasz raz, na końcu (punkt 8).

| gdzie | co |
|---|---|
| `runs/<czas>_<model>_<zestaw>_<profil>/` | jeden run: `report.md` (metryki, tuning, macierz pomyłek, top cechy), `metrics.json`, `predictions_val.parquet` i `predictions_test.parquet` (`row_id, site, esi, target, vitals_complete, p_esi1…p_esi5, pred_argmax, pred_cost`), `model.joblib`, `run_info.json` (parametry, urządzenie, sprzęt, tuning), `feature_importance.csv`, `DONE.json` |
| `runs/<run>/figures/` | z ewaluacji: macierze pomyłek (także podgrupa vc), ROC, kalibracja i wykres niezawodności, ważność cech, `metrics_val.json` (metryki per oddział, `calibration`: ECE, `ci`: 95% przedziały ufności) |
| `runs/leaderboard.csv` | jeden wiersz na ukończony run, metryki VAL (+ podgrupa vc) |
| `runs/comparison__<zestaw>__<profil>__<czas>.md` | porównanie modeli na VAL (z ECE i temperaturą T), różnorodność, średnia równowagowa |
| `runs/final_report__<zestaw>__<czas>/` | raport końcowy na TEST (punkt 14): `report.md`, `metrics.json`, `recipe.json`, `tables/`, `figures/`, `backend_ensemble_weights.json` |
| `logs/run_all/<czas>/` | log każdego kroku (`<zestaw>__<model>__train.log`, `…__eval.log`), `preflight.json`, `status.json`, `run_all.log` |
| `logs/optuna/` | badania Optuny (`optuna_<model>.db`) i wynik każdego badania (`<badanie>.best.json`) |
| `logs/training/` | szczegółowe logi treningu i historia prób (`*_trials.csv`) |

### Reguły decyzyjne i metryki

- **argmax**: najbardziej prawdopodobny poziom ESI.
- **koszt** (`cost`): poziom minimalizujący oczekiwany koszt. Niedotriażowanie kosztuje 3× więcej
  niż przetriażowanie, a koszt rośnie z kwadratem odległości. Mocno obniża niedotriażowanie
  ESI 1–2 kosztem większego przetriażowania i niższego QWK (QWK karze symetrycznie).
- `quadratic_weighted_kappa` — główna metryka porządkowa;
- `undertriage_rate` — ESI 1–2 → ESI 3–5 (bezpieczeństwo);
- `critical_miss_rate` — ESI 1 → ESI 4–5;
- `overtriage_rate` — ESI 4–5 → ESI 1–2;
- `mistriage_under` / `mistriage_over` — błąd o dowolny poziom;
- `auc_macro`, `log_loss`, `brier` — jakość prawdopodobieństw; `expected_cost` — średni koszt decyzji.

Klucze w `metrics.json`: `{split}_{reguła}_{metryka}`, a dla podgrupy vc `{split}_vc_{reguła}_{metryka}`.

## 8. Łączenie modeli (na VAL) i test na końcu

Modele trenują się **bez wag klas**, więc ich prawdopodobieństwa są skalibrowane i nadają się do
łączenia. Asymetrię kosztów realizuje reguła decyzyjna. Wszystko (wybór modeli, wagi,
kalibrację, regułę) dobierasz na `val` — jak w `notebooks/02_model_training.ipynb`:

```python
from src.ensemble import latest_runs, aligned_probas, blend, evaluate, optimize_weights
from src.ensemble import fit_temperature, apply_temperature

runs = latest_runs("triage", tag="standard")          # najnowszy ukończony run każdego modelu
y_val, P_val, vc_val = aligned_probas(runs, "val", return_vc=True)

weights = {"xgboost": 0.4, "catboost": 0.3, "ebm": 0.3}   # albo podpowiedź: optimize_weights(P_val, y_val)
T = fit_temperature(blend(P_val, weights), y_val)         # temperatura POŁĄCZONYCH prawdopodobieństw, na val
p_val = apply_temperature(blend(P_val, weights), T)       # ta sama receptura, którą ocenisz test
print(evaluate(y_val, p_val, rule="cost")["quadratic_weighted_kappa"])
print(evaluate(y_val[vc_val], p_val[vc_val], rule="cost")["undertriage_rate"])   # podgrupa vc
```

`latest_runs` domyślnie pomija runy na próbkach, nieukończone, bez pliku modelu, z innej wersji
danych, z wagami klas i bez tuningu (wypisuje notę; `require_tuned=False` albo
`07_compare_runs.py --include-untuned` je uwzględni). `aligned_probas` sprawdza, czy runy mają
te same wiersze i etykiety.

**Test — raz, na końcu**, dokładnie tą samą recepturą (te same wagi i ta sama `T` dopasowana na val):

```python
y_test, P_test = aligned_probas(runs, "test")
final = evaluate(y_test, apply_temperature(blend(P_test, weights), T), rule="cost")
```

Pojedynczy model: `python scripts/04_evaluate.py --run <run> --split test`. Nie wracaj potem do
strojenia — liczby z testu idą tylko do pracy.

Całość tego kroku — z przedziałami ufności, kalibracją, porównaniem z najlepszym pojedynczym
modelem i plikiem wag dla backendu — robi jedno polecenie: `scripts/08_final_report.py` (punkt 14).

## 9. Rozwiązywanie problemów

| objaw | co zrobić |
|---|---|
| komputer usnął w nocy | `run_all.py` blokuje uśpienie (Windows: `SetThreadExecutionState`; sprawdzisz w PowerShell **uruchomionym jako administrator**: `powercfg /requests` — `python.exe` pod SYSTEM). Nie chroni przed zamknięciem pokrywy laptopa ani ręcznym uśpieniem. Laptop: zasilacz podłączony. **WSL2**: blokada działa tylko w maszynie wirtualnej Linuksa — na Windows ustaw uśpienie na „Nigdy” (Ustawienia → System → Zasilanie (i bateria) → Ekran i uśpienie). Wznów tym samym poleceniem |
| trening stoi, w tytule okna „Zaznacz” | kliknięcie w konsolę (QuickEdit) wstrzymało wypisywanie — naciśnij Esc. `run_all.py` wyłącza QuickEdit na czas przebiegu |
| w nocy wyraźnie wolniej niż w dzień | Tryb zasilania „Najlepsza wydajność”, okno konsoli niezminimalizowane (punkt 5); `00_check_env.py` pokazuje plan zasilania |
| restart po Windows Update | na noc: Ustawienia → Windows Update → *Wstrzymaj aktualizacje* (albo godziny aktywne). Po restarcie uruchom to samo polecenie |
| repo w OneDrive | przenieś do `C:\dev\Magisterka`: synchronizacja blokuje pliki (SQLite „database is locked”, `PermissionError`) i wysyła gigabajty `runs/` |
| wolne wczytywanie / zapis | Windows Defender: Zabezpieczenia Windows → Ochrona przed wirusami → Wykluczenia: dodaj folder repo i `C:\venvs\sorai` |
| GPU nieużywane (urządzenie `cpu`) | `python scripts/00_check_env.py` pokaże powód. Najczęściej sterownik < 580: zaktualizuj i zrestartuj. Ręczne sprawdzenie: `python -c "import xgboost as x; print(x.__version__, x.build_info()['CUDA_VERSION'])"` i `python -c "from catboost.utils import get_gpu_device_count as g; print(g())"` (≥ 1). Zamknij programy zajmujące kartę (gry, inny trening) |
| wymuszenie urządzenia | PowerShell: `$env:SOR_AI_DEVICE_XGBOOST="gpu"` (błąd zamiast cichego CPU) albo `"cpu"`; to samo `SOR_AI_DEVICE_CATBOOST`, wspólnie `SOR_AI_DEVICE` |
| „Logging error”, krzaczki w logach | `setx PYTHONUTF8 1` i nowe okno PowerShell (`run_all.py` i tak wymusza UTF-8 w procesach treningu) |
| brak pamięci (RAM) | zamknij inne programy, `--lanes 1`; zestaw `full` wymaga ≥ 32 GB |
| model `NIEUDANY` | sekcja „Błędy” w podsumowaniu ma końcówkę logu; pełny log: `logs/run_all/<czas>/<zestaw>__<model>__train.log`. Po poprawce: to samo polecenie |
| „badanie jest używane przez inny proces” | ten sam model trenuje w innym oknie. Plik blokady `logs/optuna/<badanie>.lock` po martwym procesie jest przejmowany automatycznie |
| `Activate.ps1` zablokowany | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| błędy instalacji pakietów | sprawdź `python --version` (3.12/3.13 x64), krótka ścieżka venv (`C:\venvs\sorai`); ewentualnie włącz długie ścieżki (LongPathsEnabled) |

## 10. Dane: co wykluczamy i dlaczego

Yale EMMLC: 560 486 wizyt z 3 oddziałów ratunkowych w USA (2014–2017), 972 zmienne, natywne
ESI 1–5. Modele mają się uczyć **ogólnej, klinicznej** wiedzy, przenośnej na polski SOR. Dlatego
`scripts/02_preprocess.py` usuwa wszystko, co jest specyficzne dla USA, socjoekonomiczne, zależne
od organizacji szpitala albo przecieka z przyszłości wizyty. Pełny, automatycznie generowany raport
z liczbami: [`DATA_AUDIT.md`](DATA_AUDIT.md).

| wykluczone | dlaczego |
|---|---|
| ubezpieczenie (`insurance_status`) | amerykański system finansowania; w Polsce bez znaczenia klinicznego, a w danych silnie powiązane z dostępem do opieki |
| rasa, pochodzenie etniczne, wyznanie, język, stan cywilny, zatrudnienie | cechy chronione i socjoekonomiczne, nieobecne w karcie triażu TOPSOR; model nie może segregować po nich pacjentów |
| eGFR (wszystkie warianty) | wzór CKD-EPI 2009 używa współczynnika rasy (osobne kolumny „African American / non-AA”) |
| oddział Yale (`dep_name` → `site`) | uczyłby się praktyk konkretnego szpitala; zostaje tylko jako metadana (stratyfikacja, metryki per oddział, `--split site`) |
| `disposition` (przyjęcie / wypis) | to wynik wizyty — przeciek przyszłości; zostaje jako `admitted` wyłącznie do analizy trafności |
| leki (`meds_*`, zestaw `full`) | lista leków jest dokumentowana głównie przy przyjęciu, więc przecieka bieżąca decyzja (wśród pacjentów ≥ 40 lat z ≥ 2 przyjęciami w ostatnim roku leki ma wpisane 9,1% wypisanych vs 68,7% przyjętych) |
| `socialadmin` (historia chorób) | wskaźnik socjoekonomiczny (71% Medicaid / bez ubezpieczenia vs 35% w populacji) |
| administracyjne kody V (badania przesiewowe, szczepienia, opieka po zabiegu) i rzadkie grupy chorób | kody rozliczeniowe, nie stan pacjenta; za mało przypadków na wiarygodny sygnał |
| `arrival_unknown` (brak sposobu przybycia) | 99,7% braków pochodzi z oddziału A, gdzie oznaczają amerykańską ścieżkę szybkiej obsługi (64% wizyt ESI 5 tego oddziału). W TOPSOR takiej kategorii nie ma — brak kodowany jako „przyszedł sam”, policja jako „inny” |
| zepsute badania laboratoryjne (24 kolumny: puste, stałe, śmieciowe) i D-dimer | brak informacji albo mieszane jednostki w jednej kolumnie |
| `esi_danger_hr`, `esi_danger_rr` | dokładne duplikaty `tachycardia` / `tachypnea` |

Przekształcenia: temperatura °F → °C; wartości fizjologicznie niemożliwe → brak (nie mediana);
skargi główne jako flagi 0/1; aktywacje urazowe połączone w `cc_trauma_activation`; poprzednia
dyspozycja zwinięta do `admitted / discharged / transferred / left_early / none`; badania
jakościowe `_last`/`_npos` = brak, gdy badania nie wykonano; część wyników morfologii przeliczona
z komórek/µL na 10³/µL. Braki zostają jako NaN — modele drzewiaste obsługują je natywnie,
a regresja logistyczna imputuje medianą z flagą braku wewnątrz własnego pipeline'u (tylko na TRAIN).

Zestawy cech: `triage` (domyślny, ~270 cech) zawiera to, co pielęgniarka ma w chwili triażu:
parametry życiowe, wiek, płeć, sposób przybycia, czas przybycia, skargi główne i cechy kliniczne
(wskaźnik wstrząsowy, NEWS2, qSOFA, strefa zagrożenia ESI, flagi braków). `full` (~900 cech) to
`triage` + historia pacjenta: choroby przewlekłe, wcześniejsze wizyty, badania obrazowe, parametry
i wyniki badań z **poprzednich** wizyt. Dokładne liczby są w `DATA_AUDIT.md`. Opis cech:
[`FEATURE_ENGINEERING.md`](FEATURE_ENGINEERING.md).

Podział danych:

| zbiór | udział | do czego |
|---|---|---|
| train | 80% | tuning (CV) i trening; 10% wydzielane w locie na early stopping |
| val | 10% | porównanie modeli, łączenie, kalibracja, progi |
| test | 10% | tylko ostateczna ocena — niczego na nim nie dostrajaj |

Domyślnie podział jest losowy, stratyfikowany po ESI × oddział. Yale nie ma roku ani
identyfikatora pacjenta, więc podział chronologiczny jest niemożliwy. `--split site` trzyma
oddział **B** w całości jako test (walidacja zewnętrzna). Oddział C ma tylko 19 wizyt ESI 1 —
za mało, by go tak użyć.

## 11. Zastrzeżenia do metodologii

- **Braki parametrów życiowych odzwierciedlają organizację pracy, a nie stan pacjenta.** Yale
  zapisuje jako „triażowe” pomiary sprzed przyjęcia na salę. 85,2% wizyt ESI 1 nie ma żadnego
  pomiaru, bo pacjenta od razu przeniesiono na salę resuscytacyjną. W TOPSOR pomiar jest
  obowiązkowy, więc metryki na całym `val` przeszacowują wykrywanie ESI 1. Raportuj zawsze także
  **podgrupę vc** (`vitals_complete == 1`: zmierzone HR, SBP i RR) — kolumny `*_vc_*` w metrykach,
  podsumowaniu, porównaniu i ewaluacji.
- **SpO₂ = 100% przywrócone.** Filtr Yale (wartość ≥ 100 → brak) skasował każdy odczyt 100%.
  Gdy zmierzono HR i SBP, brak SpO₂ przywracany jest jako 100%. Historia `spo2_*` pozostaje
  ucięta na 99%.
- Etykieta to **najnowsze ESI wizyty** (po ewentualnym ponownym triażu).
- Brak identyfikatora pacjenta: ten sam pacjent może trafić do różnych zbiorów (lekko optymistyczne
  wyniki dla wszystkich modeli jednakowo).
- Dane z USA: populacja i organizacja SOR są inne niż w Polsce. Skargi główne pochodzą z listy
  systemu Epic (po angielsku) i trzeba je odwzorować na wywiad z karty segregacji. Wyniki badań
  są w jednostkach amerykańskich (zestaw `full` wymaga tabeli przeliczeń).

## 12. Pojedyncze skrypty

`run_all.py` wywołuje je za Ciebie. Ręcznie przydają się do eksperymentów:

```powershell
python scripts/02_preprocess.py [--split site]                       # dane → splity + DATA_AUDIT.md
python scripts/03_train.py --model xgboost --tune --profile quick    # jeden model (tag `quick`)
python scripts/03_train.py --model all                               # 6 modeli bez tuningu (tag `standard-notune`)
python scripts/04_evaluate.py --run <run>                            # wykresy, metryki, ECE i 95% CI na val
python scripts/05_explain.py --model-name xgboost --n-patients 5     # SHAP + kontrola reguł ESI
python scripts/06_cross_validate.py --model lightgbm                 # k-fold CV na TRAIN
python scripts/07_compare_runs.py --tag standard                     # porównanie na val
python scripts/07_compare_runs.py --tag standard --bootstrap 1000     # + 95% przedziały ufności (val)
python scripts/08_final_report.py --tag standard --weights wagi.json  # raport końcowy na TEST (punkt 14)
streamlit run app/streamlit_app.py                                   # demo (requirements-app.txt)
```

Tag runa (dopisek w nazwie katalogu, gdy nie podasz `--tag`): `<profil>` — tuning z budżetem
profilu bez nadpisań (zwykły run profilu, ten sam, który tworzy `run_all.py`); `<profil>-notune` —
bez `--tune`; `<profil>-custom` — z nadpisaniami (`--n-trials`, `--tune-rows`, `--cv-folds`,
`--tune-timeout`, `--sample`, `--eval-sample` inne niż w profilu, `--weights` ≠ none,
`--tune-metric qwk`, `--filter`, `--no-save`). Run niestandardowy ma w `DONE.json` `"custom": true`
i nigdy nie zastępuje kroku profilu w `run_all.py` — nawet z ręcznym `--tag <profil>`.

Modele: `logreg` (liniowy punkt odniesienia), `lightgbm`, `xgboost` (GPU), `catboost` (GPU),
`random_forest` (bagging — dywersyfikacja), `ebm` (Explainable Boosting Machine, interpretowalny
GAM bez interakcji).

## 13. Testy

```powershell
python -m pytest tests -q                    # wszystkie (z testem end-to-end run_all.py, ~1 min)
python -m pytest tests -q -m "not slow"      # bez testów w procesach
```

Testy używają wyłącznie syntetycznych danych w katalogach tymczasowych; nie dotykają `data/`,
`runs/` ani `logs/`.

## 14. Raport końcowy do pracy (przedziały ufności, kalibracja)

Gdy recepturę masz już wybraną na `val`, jedno polecenie liczy wszystkie liczby do pracy na
zbiorze **testowym**, z zapisanych predykcji (bez ponownego treningu):

```powershell
python scripts/07_compare_runs.py --tag standard --bootstrap 1000    # 1. wybór na VAL (z przedziałami)
python scripts/08_final_report.py --feature-set triage --tag standard --weights wagi.json --temperature fit
```

`wagi.json` to zwykły plik, np. `{"xgboost": 0.4, "catboost": 0.3, "ebm": 0.3}`. Może to być też
`backend_ensemble_weights.json` z wcześniejszego raportu — wtedy skrypt przejmuje z niego również
metodę łączenia (`method`), regułę dla backendu (`rule`) i to, czy jest temperatura. Opcja podana
jawnie i sprzeczna z plikiem (np. `--method mean` przy `"method": "geometric"`) kończy się odmową.
Samą wartość T skrypt zawsze dopasowuje na nowo na VAL i ostrzega, gdy wyszła inna niż w pliku.
Na Linuksie JSON można podać wprost:
`--weights '{"xgboost": 0.4, "catboost": 0.3, "ebm": 0.3}'`. W PowerShell lepiej użyć pliku,
bo PowerShell potrafi zgubić cudzysłowy w JSON-ie. Skrót na Linuksie: `make final-report TAG=standard WEIGHTS='{…}'`.

### Zalecana kolejność

1. **Wybór na VAL.** Porównaj modele (`07_compare_runs.py`, z `--bootstrap 1000` dla przedziałów)
   i ewentualnie połączenia (`src.ensemble`, punkt 8). Zdecyduj o modelach, wagach, metodzie
   łączenia (`mean` / `geometric`), temperaturze (`fit` / `none`) i regule decyzyjnej.
2. **Raz — TEST.** Uruchom `08_final_report.py` z tą recepturą. Skrypt na początku i na końcu
   wypisuje ostrzeżenie, że test ocenia się raz. Każde uruchomienie trafia do rejestru
   `runs/final_reports.jsonl` (niezależnie od `--out`). Jeśli dla tych samych runów był już raport,
   skrypt go wylicza (z informacją, czy receptura była ta sama) i zapisuje w `recipe.json`
   (`previous_reports`).
3. **Do pracy** bierzesz tabele z `report.md` (albo `tables/*.csv`) i wykresy z `figures/`.
   **Do backendu** kopiujesz `backend_ensemble_weights.json` jako `backend/models/ensemble_weights.json`
   (wagi, metoda, temperatura, reguła — format `backend/app/models/ensemble.py`).

### Co jest w raporcie (`runs/final_report__<zestaw>__<czas>/`)

| sekcja `report.md` | zawartość |
|---|---|
| Receptura | runy, wagi, temperatura (połączenia i każdego modelu), reguły, najlepszy pojedynczy model (najniższy log loss na VAL), wynik na VAL dla porównania |
| Jak czytać | krótka instrukcja do tabel (poniżej to samo szerzej) |
| 1. Wynik końcowy | połączenie: 15 metryk z 95% CI, cała populacja i podgrupa vc, obie reguły |
| 2. Połączenie i modele | każdy model (z własną temperaturą z VAL) obok połączenia, z CI; wykresy leśne `forest_test_*.png` |
| 3. Różnice sparowane | Δ = połączenie − najlepszy pojedynczy model dla QWK, niedotriażowania i log loss: przedział i p |
| 4. Kalibracja | ECE, MCE, log loss, Brier przed i po temperaturze; ECE idealnie skalibrowanego modelu (szum); wykres `reliability_test_blend.png` |
| 5. Macierze pomyłek | liczby i % wiersza dla każdej reguły (+ wykresy, także dla podgrupy vc) |
| 6. Oddziały | połączenie na każdym oddziale z CI |
| 7. Przyjęcia | AUC pilności względem przyjęcia do szpitala (model vs ESI z triażu) i odsetek przyjęć wg poziomu; gdy kolumny `admitted` nie ma ani w predykcjach, ani w przygotowanym splicie — sekcja pominięta |

Poza tym: `metrics.json` (wszystkie liczby, także porównania połączenia z każdym modelem),
`recipe.json` (receptura do odtworzenia, commit gita, argumenty) i `backend_ensemble_weights.json`.

### Co znaczą liczby

- **Przedział ufności 95%**, zapis `0.712 [0.701; 0.723]`. Zbiór testowy to jedna próbka
  pacjentów. Gdyby wylosować inną z tej samej populacji, wynik byłby trochę inny. Przedział
  pokazuje, jak bardzo. Liczymy go **bootstrapem**: 2000 razy losujemy pacjentów ze zwracaniem
  (osobno w każdej klasie ESI, więc rzadkie ESI 1 jest w każdym losowaniu), za każdym razem
  liczymy metrykę i bierzemy 2,5. i 97,5. percentyl. Przedział **nie** obejmuje zmienności samego
  treningu (inne ziarno, inne hiperparametry).
- **Odsetki pacjentów** (trafność, niedotriażowanie, przetriażowanie, krytyczne pominięcia,
  czułość ESI 1 i ESI 1–2) mają **dokładny przedział Clopper–Pearsona** z liczebności, a nie
  przedział z bootstrapu. Losowanie w klasach ESI ustala mianownik (np. liczbę pacjentów ESI 1), więc
  bootstrap odsetka tylko powtarza zaobserwowaną proporcję: przy 0 zdarzeń daje „przedział” [0; 0],
  a przy kilku zdarzeniach — za wąski. Clopper–Pearson przy 0 krytycznych pominięć na 38 pacjentów
  ESI 1 daje [0; 9,3%] i tyle niepewności rzeczywiście jest. Dlatego przy metrykach ESI raport podaje
  też liczebność, np. `0.0% [0.0; 9.3] (0/38)`. Dla ESI 1 (kilkaset wizyt na teście, w podgrupie vc
  kilkadziesiąt, na oddziale czasem kilka) przedziały są szerokie i tak należy je opisać.
- **Porównanie dwóch modeli.** Nakładające się przedziały w tabeli 2 **nie** oznaczają, że modele są
  równie dobre. O różnicy rozstrzyga sekcja 3: **sparowany bootstrap**, czyli oba modele oceniane
  na tych samych losowaniach pacjentów. Jeśli przedział Δ nie zawiera 0, różnica jest istotna.
  `p` to dwustronna p-wartość z tego samego bootstrapu: 2 × odsetek losowań, w których Δ wypadła
  po „drugiej” stronie zera (losowania z Δ dokładnie 0 liczą się do obu stron — ostrożnie). Jest
  zgodna z przedziałem: przedział bez zera ⇔ p < 0,05. Najmniejsza możliwa p to 1/liczba losowań.
  Dla niedotriażowania i log loss **ujemna** Δ znaczy, że połączenie jest lepsze.
- **ECE** (expected calibration error) mówi, czy prawdopodobieństwa „mówią prawdę”. Pacjentów
  dzielimy na grupy według pewności modelu (np. 70–77%) i porównujemy tę pewność z odsetkiem
  trafień w grupie. ECE to średnia różnica: 0 = idealnie, 0,03 = model myli się średnio o 3 punkty
  procentowe w ocenie własnej pewności. ECE z grup jest **zawyżone przez szum**: nawet idealnie
  skalibrowany model ma ECE > 0, tym większe, im mniej pacjentów. Dlatego raport podaje też
  „ECE idealnie skalibrowanego modelu” przy tej samej liczności i p-wartość testu kalibracji
  (małe p = kalibracja gorsza, niż wynikałoby z samego szumu). Z tego samego powodu przedział
  bootstrap ECE bywa przesunięty w górę, a wartość punktowa może leżeć tuż pod nim. ECE modeli
  porównuj więc parami (Δ), a nie przez nakładanie przedziałów. *ECE per poziom* robi to samo osobno
  dla każdego poziomu ESI. **MCE** to najgorsza grupa.
- **Temperatura T** to jeden parametr dopasowany na VAL: prawdopodobieństwa podnosi się do potęgi
  1/T i normalizuje. T > 1 „uspokaja” zbyt pewny model, T < 1 go wyostrza. Najbardziej
  prawdopodobny poziom (argmax) się nie zmienia, zmieniają się prawdopodobieństwa, a więc decyzje
  reguły kosztowej. Uczciwa ocena na VAL to wartości „cross-fit”: T dopasowana na jednej połowie val
  i oceniana na drugiej.
- **Podgrupa vc** to pacjenci z kompletem HR, SBP i RR, czyli populacja jak w TOPSOR (punkt 11).
  Raportuj ją zawsze obok całej populacji.

### Opcje `08_final_report.py`

| opcja | znaczenie |
|---|---|
| `--runs RUN …` albo `--feature-set FS --tag TAG [--models a,b]` | jawne runy albo najnowszy ukończony run każdego modelu (filtry jak w 07: `--include-samples`, `--include-untuned`, `--include-weighted`) |
| `--weights JSON\|plik.json\|optimize\|equal` | wagi połączenia (domyślnie `optimize` = minimalny log loss na VAL; `equal` = równe wagi — w testach wyraźnie gorsze); model spoza wag ma wagę 0, ale w tabelach jest oceniany osobno; z pliku w formacie backendu przejmowane są też `method`, `rule` i (nie)obecność temperatury |
| `--method mean\|geometric` | średnia arytmetyczna albo geometryczna prawdopodobieństw (domyślnie z pliku wag, inaczej `mean`) |
| `--temperature fit\|none` | temperatura połączenia dopasowana na VAL (domyślnie z pliku wag, inaczej `fit`) |
| `--rule argmax\|cost\|both` | reguły w raporcie (domyślnie obie); `--backend-rule` — reguła w pliku dla backendu (domyślnie `cost` przy `both`) |
| `--bootstrap N` | liczba losowań (domyślnie 2000); `--alpha 0.05`, `--seed 42` |
| `--decimal-comma` | przecinek dziesiętny w `report.md` (typografia polska) |
| `--out DIR`, `--no-plots`, `--n-bins 15` | katalog raportu, bez wykresów, biny ECE |

**Odmowa (kod wyjścia 2)** następuje, gdy runy pochodzą z różnych zestawów cech, wersji albo
przygotowań danych, gdy ten sam model występuje dwa razy, gdy predykcje val/test mają różne wiersze
lub etykiety (np. inne `--eval-sample`), gdy w predykcjach są prawdopodobieństwa NaN / nieskończone /
ujemne albo wiersz o zerowej sumie (komunikat podaje run i liczbę złych wierszy), gdy wagi są błędne
(model spoza runów, waga ujemna, same zera) albo gdy jawna opcja (`--method`, `--temperature`,
`--rule`, `--backend-rule`) jest sprzeczna z plikiem wag w formacie backendu.

**Czas** (1 rdzeń CPU): bootstrap 1000 losowań × 56 tys. wierszy × wszystkie metryki, obie reguły
to ~7 s na model i populację. Raport z 6 modelami i `--bootstrap 2000` trwa kilka minut.
`04_evaluate.py` liczy ECE zawsze, a przedziały ufności z `--bootstrap N` (domyślnie 1000,
`0` = bez nich; ~10 s na pełnym val). Wyniki zapisuje w `metrics_<split>.json` pod kluczami
`calibration` i `ci`, a wykres w `reliability_<split>.png`.

Funkcje są też dostępne bezpośrednio: `src.evaluation.statistics` (`bootstrap_ci`,
`paired_bootstrap_diff`, `ece_top_label`, `ece_classwise`, `reliability_table`,
`calibration_summary`, `ece_consistency`, `calibrate_on_val`, `crossfit_temperature`).
