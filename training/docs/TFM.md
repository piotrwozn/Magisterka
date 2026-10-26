<!-- Stan: 2026-10-26. Kod przetestowano bez GPU (CPU, losowe wagi testowe). Wszystkie liczby VRAM i czasu
     w tym dokumencie to SZACUNKI z modelu pamięci i FLOP (CPU jako przybliżenie GPU). Zastąp je pomiarem
     `python scripts/13_tfm_setup.py` (runs/analyses/tfm_setup/benchmark.json). Wyników eksperymentów A i B
     jeszcze nie ma — wymagają prawdziwych wag i karty graficznej. -->

# Modele fundamentalne (TFM): TabPFN-3.5 i TabICLv2 obok LightGBM

Ten dokument opisuje, jak w domu (Windows 11, RTX 4080 Super 16 GB) sprawdzić, czy **wstępnie wytrenowane
tabularne modele fundamentalne** pomagają LightGBM w klasyfikacji ESI. Zawiera instalację, kolejność
poleceń, szacunki czasu, opis plików wynikowych, tabelę decyzji, gotowe akapity metod do pracy,
ograniczenia i rozwiązywanie problemów.

## W skrócie

```powershell
# 0. DOPIERO PO zakończeniu treningu klasycznego (python scripts/run_all.py --profile thorough --status)
cd C:\dev\Magisterka\training                                            # aktywny venv (README, punkt 2)
pip install torch --index-url https://download.pytorch.org/whl/cu130     # sterownik < 580: .../whl/cu128
pip install -r requirements-tfm.txt -c requirements-lock.txt
python scripts/13_tfm_setup.py                                           # licencja TabPFN, wagi, benchmark GPU

# 1. Trening TFM (wybór konfiguracji na holdoucie z TRAIN + predykcja VAL/TEST; wznawialny)
python scripts/run_all.py --profile thorough --models tfm

# 2. Analizy: SD seedów LightGBM (do bramki B), eksperyment A, eksperyment B
python scripts/09_seed_refits.py --feature-set triage --tag thorough --models lightgbm   # pomiń, jeśli już jest
python scripts/14_small_data_curve.py --feature-set triage --tag thorough
python scripts/15_tfm_blend.py --feature-set triage --tag thorough
#    (to samo jednym poleceniem razem z analizami klasycznymi: python scripts/run_all.py --profile thorough --stage analyses)

# 3. RAZ: raport końcowy na TEST — polecenie `command` z pliku
#    runs/analyses/tfm_blend__triage__thorough/recommendation.json
```

Linux: `make tfm-setup`, `make tfm-train PROFILE=thorough`, `make small-data TAG=thorough`,
`make tfm-blend TAG=thorough` (zob. [`Makefile`](../Makefile)).

---

## 1. Co i dlaczego

**Model fundamentalny (TFM)** to sieć wstępnie wytrenowana na milionach syntetycznych zbiorów
tabelarycznych. Nie dopasowuje wag do naszych danych. Dostaje oznaczone wiersze (**kontekst**) i nowe
wiersze, a prawdopodobieństwa klas liczy w jednym przebiegu sieci (*in-context learning*). W tym repo
„trening” TFM oznacza więc zapis kontekstu, czyli wierszy TRAIN, w artefakcie modelu. Cała praca
obliczeniowa dzieje się przy predykcji, na GPU.

| model | biblioteka | wagi | uwagi |
|---|---|---|---|
| **TabPFN-3.5** (Prior Labs) | `tabpfn==9.2.0` | `tabpfn-v3.5-20260909.safetensors` (ok. 0,9 GB) | braki (NaN) obsługuje natywnie; na CPU biblioteka dopuszcza najwyżej 5000 wierszy kontekstu |
| **TabICLv2** (Inria) | `tabicl==2.2.0` | `tabicl-classifier-v2-20260212.ckpt` (ok. 110 MB, Hugging Face `jingang/TabICL`) | braki imputuje średnią (flagi `*_missing` zostają w cechach); pretrening objął 300–48 tys. wierszy i ≤ 100 cech (u nas 271) |

**Po co.** Dwa pytania do pracy:

- **A. Mało lokalnych danych** (scenariusz polskiego szpitala z setkami–tysiącami oznaczonych wizyt).
  Literatura zwykle, choć nie zawsze, pokazuje tu przewagę TFM nad boostingiem
  ([LITERATURA.md](LITERATURA.md#tabularne-modele-fundamentalne-tfm)).
  Sprawdzamy, czy TFM „z pudełka” bije LightGBM strojony na tej samej małej próbie i do jakiego n.
- **B. Pełne dane** (446 421 wierszy TRAIN). Czy dołączenie TFM do LightGBM poprawia prawdopodobieństwa?
  Literatura sugeruje efekt mały albo żaden, więc **wynik negatywny jest realny** i sam w sobie wartościowy.

**Zasady (ustalone przed liczeniem):**

1. **LightGBM jest rdzeniem.** System końcowy to LightGBM + **najwyżej 2** TFM, i to tylko te, które
   przejdą bramkę ustaloną z góry (punkt 7).
2. Decyzje zapadają na VAL. TEST ocenia się **raz**, w `08_final_report.py`.
3. `--models all` / `--model all` nadal oznacza **6 modeli klasycznych**. TFM trzeba zażądać jawnie:
   `tfm` (oba), `tabpfn`, `tabicl`, aliasy `pfn`, `icl`. Nazwy runów, tagi, odciski danych, wznawianie
   i wyniki modeli klasycznych się nie zmieniają.
4. Dane zostają na komputerze: biblioteki liczą lokalnie, bez API producentów.

**Jak TFM wpina się w pipeline** (szczegóły: [`src/models/tfm_base.py`](../src/models/tfm_base.py)):

- **Worki kontekstu.** Kontekst to `n_ctx` wierszy TRAIN losowanych warstwowo po ESI × oddział. Kilka
  worków (`n_bags`, tryb `partition` = rozłączne części TRAIN) daje kilka predykcji, które są uśredniane.
  Każdy worek ma co najmniej 2 wiersze każdej klasy (rzadkie ESI 1 w razie potrzeby powtórzone, z notą
  w `tfm.events`). Strategia `balanced` nadreprezentuje ESI 1 i ESI 5, a potem stosuje dokładną korektę
  priorów.
- **„Tuning” = wybór konfiguracji** ([`src/models/tfm_tuning.py`](../src/models/tfm_tuning.py)), a nie
  Optuna. Kandydaci profilu (`n_ctx`, `n_estimators`, strategia, `max_features`) są oceniani na holdoucie
  z TRAIN (20 000 wierszy w `standard` i `thorough`). Wygrywa najniższy log loss; przy różnicy ≤ 0,002
  wygrywa tańszy kandydat. W profilach `standard` i `thorough` potem etap worków: 1 worek kontra `auto`
  (do 4). VAL i TEST nie biorą w tym udziału. Wybór jest wznawialny (wyniki kandydatów
  w `logs/optuna/`).
- **Artefakt** `model.joblib` jest samowystarczalny: kontekst zapisany w modelu (kilkanaście MB na
  dysku, ok. 0,5 GB w RAM po wczytaniu). Dopasowane estymatory torch nigdy nie trafiają do pliku.
- **Pamięć GPU.** Gdy zabraknie pamięci (OOM), wrapper sam schodzi po drabinie awaryjnej (punkt 10).
  Duży kontekst bez GPU kończy się błędem **na starcie**, a nie wielogodzinnym liczeniem na CPU.
- **Reszta pipeline'u:**
  - `04_evaluate.py` ocenia run TFM z zapisanych predykcji, bez ponownej inferencji;
  - `05_explain.py` dla TFM kończy się komunikatem (kod 2): SHAP nie jest liczony, a wyjaśnienia
    w pracy opierają się na LightGBM;
  - `06_cross_validate.py` odmawia TFM;
  - refity seedów (`09`) obejmują TFM tylko na jawne żądanie (`--models tfm`).

## 2. Licencje

| składnik | licencja | praca magisterska | wdrożenie w szpitalu |
|---|---|---|---|
| wagi **TabPFN-3.5** | **niekomercyjna** licencja Prior Labs; jednorazowa akceptacja na https://ux.priorlabs.ai | **dozwolone** | wymaga licencji komercyjnej Prior Labs (sales@priorlabs.ai) |
| kod `tabpfn` | Apache-2.0 | tak | tak |
| kod `tabicl` (TabICLv2) | BSD-3 | tak | tak |
| checkpoint TabICLv2 (`jingang/TabICL`) | **do potwierdzenia**: licencji wag nie dało się odczytać offline. `13_tfm_setup.py` wypisuje ją z karty modelu na Hugging Face (linia „TabICL: licencja wag”). Zapisz ją w pracy | sprawdź | sprawdź |

- README biblioteki TabPFN podaje, że wagi TabPFN-2.5/2.6/3/3.5 są wydane na licencjach
  niekomercyjnych. Akceptacja licencji odblokowuje pobranie wag. Dane nie są wysyłane do Prior Labs,
  a po pobraniu wag biblioteka działa offline.
- Licencja trafia do `run_info.json` każdego runu (`tfm.licence`).
- Do wdrożenia lepszy jest model z licencją permisywną. Jeśli TabICLv2 jest w granicach MID od TabPFN-3.5
  (punkt 7), do systemu w szpitalu wybierz TabICLv2.

## 3. Instalacja (Windows 11, PowerShell), krok po kroku

**Najpierw poczekaj, aż trening klasyczny się skończy.** `pip` podmienia pliki w venv, z którego korzysta
działający `run_all.py`. Na Windows kończy się to błędami zablokowanych plików i DLL. Sprawdź:
`python scripts/run_all.py --profile thorough --status` (wszystkie modele `gotowy`, proces nie działa).

1. **Sterownik karty.** `nvidia-smi` musi pokazać „Driver Version” i „CUDA Version”. Sterownik ≥ 580
   (CUDA 13) jest i tak wymagany przez xgboost (README, punkt 1).
2. **torch z CUDA, z indeksu PyTorch, nie z PyPI.** Koło torch z PyPI na Windows liczy tylko na CPU.

   ```powershell
   python -c "import torch; print(torch.__version__)"      # jeśli jest i kończy się na +cpu:
   pip uninstall -y torch
   pip install torch --index-url https://download.pytorch.org/whl/cu130   # sterownik >= 580
   # starszy sterownik: pip install torch --index-url https://download.pytorch.org/whl/cu128
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
   ```

   Ma się wypisać `True` i nazwa karty.
   **Pamięć współdzielona (Windows, ważne):** sterowniki NVIDIA (od 536.40; przełącznik od 546.01) przy
   braku VRAM domyślnie przenoszą nadmiar do współdzielonej pamięci RAM zamiast zgłosić brak pamięci (OOM).
   Wtedy drabina awaryjna wrappera nie zadziała, a model liczy się wielokrotnie wolniej (wybór konfiguracji
   może przekroczyć limit o godziny). Ustaw: Panel sterowania NVIDIA → Zarządzaj ustawieniami 3D →
   Ustawienia programu → dodaj `python.exe` z venv (`.venv\Scripts\python.exe`) → „CUDA – Sysmem Fallback
   Policy” = **„Prefer No Sysmem Fallback”** → Zastosuj. Objaw braku tego ustawienia: w Menedżerze zadań
   rośnie „Współdzielona pamięć GPU”, a `13_tfm_setup.py` / wybór konfiguracji ostrzegają o podejrzanie
   wolnym pomiarze. W repo testowano `2.14.1+cu130` (bez karty, na CPU).
   **Kolejność ma znaczenie:** torch instaluj **przed** `requirements-tfm.txt`. Odwrotnie pip dociągnie
   koło CPU z PyPI jako zależność `tabpfn`.
3. **Biblioteki TFM**, z ograniczeniami locka, żeby żaden pakiet rdzenia nie zmienił wersji:

   ```powershell
   pip install -r requirements-tfm.txt -c requirements-lock.txt
   ```

   [`requirements-tfm.txt`](../requirements-tfm.txt) przypina `tabpfn==9.2.0`, `tabicl==2.2.0` i nowe
   zależności przechodnie (huggingface-hub 2.3.0, pydantic 2.14.1, pydantic-settings 2.16.1,
   safetensors 0.8.0, einops 0.8.2, skrub 0.10.1).
   **Linux:** pip podmienia przy tym `nvidia-nccl-cu13` (2.32.3 → 2.30.7), z którego korzysta xgboost na
   GPU. Po instalacji uruchom `python scripts/00_check_env.py` (próba GPU xgboost).
4. **Licencja TabPFN-3.5** (jednorazowo):
   1. https://ux.priorlabs.ai → zaloguj się albo załóż konto → zakładka **Licenses** → zaakceptuj
      licencję;
   2. skopiuj klucz API ze strony https://ux.priorlabs.ai/account;
   3. `$env:TABPFN_TOKEN = "<klucz>"` (tylko to okno) albo `setx TABPFN_TOKEN "<klucz>"` i **nowe okno**.

   Bez `TABPFN_TOKEN` biblioteka sama otworzy przeglądarkę, o ile skrypt działa w zwykłej konsoli, i
   zapamięta klucz w `~/.cache/tabpfn/auth_token`.
5. **Pobranie wag, kontrola i benchmark GPU:**

   ```powershell
   python scripts/13_tfm_setup.py                  # pełna kontrola + pobranie + SDPA + benchmark (kilka–kilkanaście min)
   python scripts/13_tfm_setup.py --offline-check  # potem: kontrola BEZ sieci, dokładnie jak przed treningiem
   python scripts/00_check_env.py --models tfm --profile thorough   # preflight z kontrolą TFM
   ```

**Co robi `13_tfm_setup.py`** ([skrypt](../scripts/13_tfm_setup.py)):

1. torch i CUDA: wersja, koło CUDA czy tylko-CPU (z dokładnym poleceniem naprawy), karta, VRAM całkowita
   i **wolna**.
2. Wersje bibliotek oraz plik `.env` z kluczami `TABPFN_*` w bieżącym katalogu (punkt 10).
3. Wagi TabPFN-3.5 po akceptacji licencji, do pamięci podręcznej TabPFN (`TABPFN_MODEL_CACHE_DIR`,
   domyślnie `%APPDATA%\tabpfn`). Błąd jest rozpoznawany jako: brak sieci, brak klucza lub licencji albo
   odmowa Hugging Face.
4. Checkpoint TabICLv2 do pamięci podręcznej Hugging Face (`HF_HOME` / `HF_HUB_CACHE`, domyślnie
   `C:\Users\<użytkownik>\.cache\huggingface\hub`) i licencja z karty modelu.
5. Kontrola offline: pliki w pamięci podręcznej, sha256, konfiguracje prawdziwych checkpointów
   porównane z planem pamięci.
6. Jądro attention (SDPA) na GPU, bez wolnego wariantu MATH: zwykłe MHA i kształt TabPFN z GQA. Bez
   tych jąder pamięć rosłaby kwadratowo z kontekstem.
7. Benchmark na prawdziwych danych (konteksty 5k / 25k / 50k × 2000 wierszy testowych, oba modele przez
   wrappery projektu). Wynik trafia do `runs/analyses/tfm_setup/benchmark.json` (punkt 6.2), razem
   z szacunkami czasu każdego profilu i eksperymentu A (jednostki TFM + LightGBM). Szczyt pamięci ≥ 90%
   VRAM albo czas na wiersz rosnący ponad 10× szybciej niż kontekst → ostrzeżenie „Pamięć GPU” (zob.
   „Pamięć współdzielona” wyżej).
   Pomiar na CPU (`--device cpu` albo brak CUDA), z atrapą lub z przekierowanymi wagami to **test mechaniki**:
   trafia do `benchmark_mechanics.json` i **nie nadpisuje** `benchmark.json` (z niego `run_all.py` i
   `14_small_data_curve.py` biorą szacunki czasu — pomiary nie-GPU są tam ignorowane).

| opcja | znaczenie |
|---|---|
| `--offline-check` | tylko kontrola, bez sieci (torch/CUDA, wersje, wagi w pamięci podręcznej, SDPA) |
| `--skip-download` / `--skip-benchmark` | bez pobierania wag / bez benchmarku |
| `--models tfm\|tabpfn\|tabicl` | które modele (aliasy `pfn`, `icl`); domyślnie oba |
| `--no-browser` | TabPFN bez logowania w przeglądarce (wymaga `TABPFN_TOKEN`) |
| `--tabpfn-version` | wersja wag TabPFN (domyślnie `v3.5`; dozwolone też `v3.5-fast`, `v3`) |
| `--device auto\|cuda\|cpu` | `cpu` = tylko test mechaniki (konteksty > 5000 pominięte; wynik w `benchmark_mechanics.json`) |
| `--contexts 5000,25000,50000` / `--test-rows 2000` / `--n-estimators tabpfn=4,tabicl=8` | ustawienia benchmarku |
| `--n-jobs`, `--seed` | wątki torch na CPU, seed |

Kody wyjścia: 0 = gotowe do treningu TFM, 1 = coś do poprawy (lista `[BŁĄD ]`) albo brak CUDA,
2 = złe argumenty, 130 = Ctrl+C. Bez CUDA brak karty jest błędem, a benchmark GPU jest pomijany; z
`--device cpu` test mechaniki się wykona, ale wynik to „tylko test mechaniki — TFM NIE są gotowe do
treningu” z kodem 1 (na CPU działa tylko profil smoke).

## 4. Kolejność poleceń

Wszystkie kroki są **wznawialne**: po przerwaniu uruchom to samo polecenie. Profil `thorough` i tag
`thorough` zakładają, że trening klasyczny szedł z `--profile thorough`. Przy profilu `standard` zamień
oba na `standard`.

### Krok 1. Trening TFM

```powershell
python scripts/run_all.py --profile thorough --models tfm --dry-run   # plan, ETA, polecenia
python scripts/run_all.py --profile thorough --models tfm
```

- **Preflight** (`00_check_env.py --models tfm`) sprawdza:
  - torch z CUDA (bez CUDA odmowa, chyba że profil `smoke` z kontekstem ≤ 5000);
  - `tabpfn ≥ 9.2.0` i `tabicl ≥ 2.2.0`;
  - wagi w lokalnej pamięci podręcznej, bez sieci (brak → „uruchom `python scripts/13_tfm_setup.py`”);
  - wolną VRAM (ostrzeżenie poniżej 12 GB);
  - miejsce na dysku.
- Dla każdego TFM: `03_train.py --model <tfm> --tune --profile thorough` (wybór konfiguracji, kontekst
  z całego TRAIN, predykcja VAL + TEST w jednym przebiegu każdego worka), potem `04_evaluate.py`
  (z zapisanych predykcji) i porównanie `07`.
- TFM idą **torem GPU** (po xgboost i catboost) i nigdy nie dzielą karty z innym modelem. Błąd GPU TFM
  **nie** jest powtarzany na CPU: log i podsumowanie podają wskazówkę (wagi, CUDA, brak pamięci).
- `--status` pokazuje postęp wyboru konfiguracji (kandydaci ukończeni / pominięci / błędy). W podsumowaniu
  kolumna *CV* dla TFM to log loss holdoutu z TRAIN (`holdout_logloss`), a *próby* to kandydaci.

### Krok 2. SD seedów LightGBM (warunek bramki B)

```powershell
python scripts/09_seed_refits.py --feature-set triage --tag thorough --models lightgbm
```

Kryterium 3 bramki B porównuje |Δ log loss| z 2 × SD log loss LightGBM między seedami **na VAL**. Jeśli
analizy klasyczne (`run_all.py --stage analyses`) już szły dla tego tagu, wynik jest gotowy i skrypt
niczego nie przeliczy. Bez niego B daje werdykt „nierozstrzygnięte” i nie rekomenduje TFM.

### Krok 3. Eksperyment A: krzywa uczenia przy małych danych

```powershell
python scripts/14_small_data_curve.py --feature-set triage --tag thorough --plan   # jednostki i czas, bez liczenia
python scripts/14_small_data_curve.py --feature-set triage --tag thorough
```

- **Scenariusze:**
  - `pooled`: podpróby n ze wszystkich oddziałów (warstwowo ESI × oddział), ocena na stałych 20 000
    wierszach VAL;
  - `site_local`: podpróby tylko z oddziału B, ocena na **wszystkich** 16 633 wierszach VAL oddziału B
    (68 ESI 1).
- **Rozmiary i powtórzenia:** n = 500, 1000, 2000, 5000, 10 000, 50 000; R = 10, 10, 10, 5, 5, 3. Każda
  podpróba jest losowana raz i używana przez wszystkie modele (układ sparowany).
- **Modele:**
  - `lightgbm_tuned`: Optuna, 40 prób na podpróbie, wewnętrzne warstwowe K-fold CV;
  - `lightgbm_default`: domyślne ustawienia biblioteki;
  - `tabpfn`, `tabicl`: „z pudełka”, kontekst = cała podpróba;
  - tylko w `site_local`: `tabpfn+foreign` / `tabicl+foreign` (kontekst + 10 000 wierszy innych
    oddziałów; test transferu negatywnego) oraz `lightgbm_transported` (LightGBM z innych oddziałów,
    surowy i z korektą znanego lokalnego rozkładu ESI).
- **Brak TFM nie przerywa liczenia.** Gdy brakuje torch, biblioteki albo wag, albo jest `--device gpu`
  bez CUDA, dany TFM jest pomijany: na wyjściu pojawia się linia `skipped: <model> — <powód>`, a części
  LightGBM liczą się normalnie. Na CPU jednostki TFM z kontekstem > 5000 są pomijane i wypisywane
  (pełny eksperyment wymaga GPU).
- **Opcje:**

  | opcja | domyślnie | znaczenie |
  |---|---|---|
  | `--models` | `tabpfn,tabicl` | TFM do porównania (`tfm`, `none` = tylko LightGBM; `lightgbm` na liście jest ignorowany) |
  | `--scenarios` | `pooled,site_local` | scenariusze |
  | `--sizes` / `--repeats` | jak wyżej | rozmiary n i powtórzenia (jedna wartość = dla wszystkich) |
  | `--eval-rows` | 20000 | `pooled`: wiersze VAL do oceny (0 = cały VAL) |
  | `--local-eval-rows` | 0 | `site_local`: 0 = wszystkie wiersze VAL oddziału |
  | `--local-site` | `B` (`HOLDOUT_SITE`) | oddział scenariusza lokalnego |
  | `--foreign-rows` | 10000 | wiersze innych oddziałów w kontekście `+foreign` (0 = bez) |
  | `--transport-rows` | 100000 | wiersze dla LightGBM przeniesionego (0 = bez) |
  | `--lgbm-trials` | 40 | próby Optuny na jednostkę LightGBM |
  | `--bootstrap` | 1000 | losowania dwupoziomowego bootstrapu (0 = tylko przedziały t) |
  | `--device` | `auto` | urządzenie TFM (`auto`, `cpu`, `gpu`) |
  | `--seed`, `--plan`, `--force`, `--n-jobs` | | seed losowania prób; plan bez liczenia; przeliczenie od nowa; wątki CPU |

- Kody wyjścia: 0 = OK (także gdy TFM pominięty), 1 = któraś jednostka nieudana, 2 = złe argumenty albo
  dane, 130 = Ctrl+C.

### Krok 4. Eksperyment B: blend z LightGBM na pełnych danych (tylko VAL)

```powershell
python scripts/15_tfm_blend.py --feature-set triage --tag thorough --plan   # runy, warianty, czas, bramka
python scripts/15_tfm_blend.py --feature-set triage --tag thorough
```

- **Wejście:** zapisane predykcje VAL najnowszych ukończonych runów `lightgbm`, `tabpfn`, `tabicl`
  z tagiem. Nie potrzebuje GPU ani ponownej predykcji. Bez runu `lightgbm` albo bez żadnego runu TFM:
  linia `skipped: <powód>` i kod 0.
- **Zestawy:** {LightGBM} (odniesienie), {LightGBM, TabPFN}, {LightGBM, TabICL}, {LightGBM, TabPFN,
  TabICL}, a do porównania także same TFM i zestaw 6 modeli klasycznych z tym samym tagiem. Każdy
  zestaw łączony średnią i średnią geometryczną; liczy się metoda z niższym log loss.
- **Procedura:** 5 × dwukrotny cross-fitting VAL. Na połowie VAL dopasowanie wag, temperatury i reguł
  decyzyjnych jak w 08, ocena na drugiej połowie. Δ względem LightGBM (z własną temperaturą i regułami)
  na tych samych pacjentach, z 95% CI ze sparowanego bootstrapu warstwowanego po ESI.
- **Bramka (wypisywana w logu i raporcie PRZED liczbami).** Zestaw z TFM przechodzi, gdy jednocześnie:
  1. Δ log loss ≤ −0,005 (MID);
  2. górna granica 95% CI Δ log loss < 0;
  3. |Δ| > 2 × SD seedów LightGBM (log loss na VAL; brak wyników seedów → „nierozstrzygnięte”);
  4. niedotriażowanie reguły „progi T” nie gorsze o > 1 pp;
  5. ECE ESI 1 nie gorsze o > 0,01.

  **Reguła oszczędności:** para TFM wygrywa tylko wtedy, gdy bije najlepszy przechodzący zestaw z jednym
  TFM o ≥ 0,005. Nigdy więcej niż 2 TFM.
- **Opcje:** `--runs RUN …` (jawne runy: lightgbm + najwyżej 2 TFM), `--models tfm|tabpfn|tabicl`,
  `--repeats 5`, `--bootstrap 1000`, `--methods mean,geometric`, `--target-undertriage`,
  `--fit-population vc|all`, `--safety-net` / `--no-safety-net` (jak w 08), `--no-reference` (bez zestawu
  klasycznego), `--no-plots`, `--plan`, `--force`, `--seed`, `--n-jobs`.
- Kody wyjścia: 0 = OK (także `skipped:`), 1 = któreś powtórzenie nieudane, 2 = złe argumenty albo runy
  z różnymi wierszami, 130 = Ctrl+C.

**Kroki 2–4 jednym poleceniem:** `python scripts/run_all.py --profile thorough --stage analyses`. Uruchamia
po kolei `09` (6 modeli klasycznych; refity TFM tylko z `--models tfm`), `10`, `11`, a potem `14` i `15`.
Kroki TFM ruszają **tylko wtedy, gdy TFM są w `--models`** (np. `--models all,tfm`) **albo istnieje
ukończony run TFM z tym tagiem** — zwykłe `--stage analyses` dla modeli klasycznych nie uruchamia
eksperymentu A (kilka godzin, głównie strojenie LightGBM) tylko dlatego, że biblioteki TFM są zainstalowane.
Dalej muszą być spełnione warunki kroku: biblioteki TFM zainstalowane (A; modele = TFM z `--models`, a bez
nich — TFM z gotowym runem), run `lightgbm` i co najmniej jeden run TFM z tagiem (B). W przeciwnym razie
krok jest „pominięty: <powód>” w planie, w podsumowaniu i w `--status`. Gotowe jednostki są pomijane.
ETA kroku `small_data` = jednostki TFM z benchmarku GPU + część LightGBM (~5 h przy domyślnych
rozmiarach); bez benchmarku GPU ETA nie jest podawana.

### Krok 5. Raport końcowy na TEST (raz)

Skopiuj polecenie z `runs/analyses/tfm_blend__triage__thorough/recommendation.json`, pole `command`.
Przykład:

```powershell
python scripts/08_final_report.py --feature-set triage --runs <run_lightgbm> <run_tabpfn> --weights optimize --method geometric
```

- Pole `command_tag` podaje równoważną postać przez tag: `--tag … --models … --include-pretrained`.
  Bez `--include-pretrained` albo nazwy TFM w `--models` funkcja `latest_runs` pomija runy TFM.
- Jeśli receptura ma TFM, 08 zapisuje dodatkowo `backend_ensemble_weights_no_tfm.json`: tę samą recepturę
  dopasowaną na VAL bez TFM, z tymi samymi ustawieniami reguły. `report.md` o tym informuje.
- Gdy bramka nie przeszła, rekomendacja to sam LightGBM. Jeśli raport 08 dla tej receptury już istnieje,
  **nie uruchamiaj go ponownie** (TEST ocenia się raz; 08 i tak wypisze wcześniejsze raporty z rejestru
  `runs/final_reports.jsonl`).
- Jeśli 08 szedł już dla receptury bez TFM, a teraz uruchamiasz go dla receptury z TFM, opisz to w pracy
  jako **drugie porównanie, zadeklarowane z góry** w tym dokumencie.

### Krok 6 (opcjonalnie). Porównanie LLM z nową recepturą

```powershell
python scripts/12_llm_bakeoff.py run --recipe runs/final_report__triage__<czas>/recipe.json
```

Części `esi` i `tools` liczą decyzje tą samą recepturą co 08 (opis: [LLM_BAKEOFF.md](LLM_BAKEOFF.md)).
Członek TFM jest ładowany z `keep_fitted_bags="all"`: każdy worek dopasowany **raz** na cały przebieg,
nigdy na wiersz. Polityka pamięci przy starcie: suma szacowanych buforów wszystkich worków wszystkich TFM
≤ 80% **wolnej** VRAM → bufory zostają na GPU; inaczej (WARNING z liczbami) bufory idą poza VRAM —
TabPFN: bufor KV w RAM i na karcie jeden estymator naraz, TabICL: mniejszy bufor `repr`. Przypadki części
liczone są **wsadowo** (jedno wywołanie na wszystkie przypadki), a wyniki zapamiętane: what_if bez zmian
daje ten sam wynik, zmieniony what_if = jedno wywołanie 1 wiersza. TFM, który przy każdym wywołaniu
koduje cały kontekst (TabICL `repr` / bez bufora — domyślnie przy kontekście 50k na 16 GB — albo TabPFN
`fit_preprocessors` po OOM), dostaje WARNING z czasem pierwszego wywołania: tyle kosztuje każdy zmieniony
what_if. Domyślnie bake-off używa modeli chmurowych Ollamy, więc karta jest wolna. Przed startem sprawdź
`ollama ps` i zatrzymaj modele lokalne (`ollama stop <model>`). Gdy mimo to zabraknie pamięci GPU albo
what_if są za wolne, uruchom bake-off bez `--recipe`. Skrypt weźmie wtedy najnowsze runy klasyczne (TFM
są pomijane domyślnie) i dopasuje recepturę na VAL. W pracy zaznacz, że przypadki liczyła receptura bez
TFM.

### Krok 7 (opcjonalnie). Backend

1. Skopiuj `backend_ensemble_weights.json` → `backend/models/ensemble_weights.json`, a
   `backend_ensemble_weights_no_tfm.json` → `backend/models/ensemble_weights_no_tfm.json`.
2. **Domyślnie backend NIE serwuje TFM.** Używa wtedy `ensemble_weights_no_tfm.json` i loguje WARNING.
   Bez tego pliku renormalizuje wagi i ostrzega, że temperatura i progi były dopasowane do pełnego
   blendu.
3. **Opt-in:** `SOR_AI_BACKEND_TFM=1` w środowisku backendu. Wymaga torch z CUDA i bibliotek TFM w tym
   środowisku (te same polecenia co w punkcie 3). Przy starcie backend:
   - odmawia TFM, który koduje cały kontekst przy każdym żądaniu (`TFMPredictor.reencodes_context()`:
     TabICL `kv_cache` `repr` / bez bufora — domyślne dla kontekstu 50k na 16 GB — albo TabPFN
     `fit_preprocessors` po OOM w treningu): jedno żądanie 1 wiersza trwałoby sekundy–dziesiątki sekund,
     a ścieżka żądania jest synchroniczna;
   - sumuje szacowane bufory wszystkich worków i porównuje je z **wolną** VRAM, z zapasem
     `SOR_AI_BACKEND_TFM_RESERVE_MB` (domyślnie 2048);
   - rozgrzewa wszystkie worki i mierzy predykcję 1 wiersza: powyżej
     `SOR_AI_BACKEND_TFM_MAX_LATENCY_S` (domyślnie 2 s) TFM nie jest serwowany.

   Każda porażka oznacza recepturę bez TFM z WARNING. Kartę dzieli Ollama, więc duże konfiguracje
   (thorough) praktycznie się nie zmieszczą. Opis w [`backend/app/models/registry.py`](../../backend/app/models/registry.py).
4. Linia `TABPFN_*` w `backend/.env` przekierowuje miejsce, w którym backend szuka wag TabPFN (punkt 10).

## 5. Czas: szacunki do czasu benchmarku

> **To szacunki, nie pomiary.** Pochodzą z modelu FLOP i pamięci zbudowanego na CPU, przy założeniu
> 40 TFLOP/s na RTX 4080 Super; czas zegarowy zwykle wynosi 1,5–3 × czas FLOP. Nie obejmują wczytania
> danych ani przetwarzania wstępnego na CPU. Po `13_tfm_setup.py` patrz `benchmark.json` →
> `estimates.profiles.<profil>.<model>.total_s` i `estimates.experiment_a.total_s` (punkt 6.2).

| krok | szacunek (czas zegarowy) | górna granica |
|---|---|---|
| instalacja + licencja + `13_tfm_setup.py` (pobranie ok. 1 GB, benchmark) | 0,5–1 h pracy | — |
| trening TabPFN-3.5, `thorough` (7 kandydatów na 20 tys. holdoutu ≈ 11 min FLOP; finał np. 100k × 8 × 4 worki ≈ 32 min FLOP) | ok. 1–2,5 h | 4 h wyboru + 4 h finału |
| trening TabICLv2, `thorough` (wybór ≈ 1,5 min FLOP; finał 50k × 8 × 4 worki ≈ 4 min FLOP) | ok. 15–45 min | 4 h + 4 h |
| trening, `standard` (TabPFN finał 50k × 4 × 4 worki ≈ 7 min FLOP; TabICL ≈ 4 min) | ok. 15–40 min na model | 2 h + 2 h |
| trening, `quick` / `smoke` | minuty (`smoke` działa na CPU) | 20 + 20 min / 15 + 30 min |
| `09` seedy LightGBM | ok. 3 × czas treningu finalnego LightGBM | — |
| eksperyment A: części TFM (TabPFN ≈ 33 min FLOP, TabICL ≈ 12 min FLOP) | TabPFN ok. 1–2 h, TabICL ok. 0,5–1 h | — |
| eksperyment A: LightGBM strojony (2 × 43 podpróby × 40 prób) | `--plan`: ok. 2,5 h na scenariusz (wzór zgrubny) | — |
| eksperyment B (pełny VAL, 5 powtórzeń, bootstrap 1000) | ok. 3–15 min, CPU | — |
| `08_final_report.py` | kilka minut | — |

`run_all.py --profile thorough --models tfm` wypisuje górną granicę z limitów profilu: **≤ 8 h na model,
16 h razem**. Realnie (szacunek) cały ciąg kroków 1–5 zajmie 1–2 noce; najdłużej trwa eksperyment A
z częścią LightGBM (ok. 6–9 h).

## 6. Co powstaje i jak to czytać

### 6.1 Run TFM: `runs/<czas>_<tabpfn|tabicl>_triage_<tag>/`

Te same pliki co dla modeli klasycznych (`predictions_val/test.parquet`, `metrics.json`, `report.md`,
`model.joblib`, `DONE.json`) plus:

| gdzie | co |
|---|---|
| `report.md`, sekcja „Tuning — wybór konfiguracji…” | tabela kandydatów: konfiguracja, status (`complete`, `oom`, `failed`, pominięty przez limit czasu), log loss i QWK holdoutu, czas, wiersze/s, szczyt VRAM. Wybrana konfiguracja i log loss holdoutu obok log loss VAL |
| `report.md`, sekcja „Model fundamentalny (TFM)” | biblioteka i wersja, plik wag + sha256, licencja, worki i rozmiary kontekstu, czas predykcji VAL + TEST, zdarzenia |
| `run_info.json` → `pretrained: true`, blok `tfm` | `library_version`, `torch_version`, `weights` (plik, rozmiar, **sha256**, wersja), `licence`, `device`, `config`, `n_bags`, `context_sizes`, `class_counts_per_bag`, `n_features_used`, `throughput` (`rows_per_s`, `peak_vram_mb`), `predict_s`, `final_budget`, `events` |
| `run_info.json` → `tfm.events` | `oom_downgrade` (krok drabiny OOM), `library_oom_warning`, `rare_class_reuse` (rzadka klasa powtórzona w worku), `bags_reduced` (mniej worków, by zmieścić się w `final_budget_s`; liczba trafia też do `params` — refity seedów nie wracają do pełnej liczby), `final_budget_exceeded` (nawet 1 worek ponad budżetem), `cache_on_host` (bufory poza VRAM w bake-offie), `max_features`, `weights_mismatch`, `device_fallback`, `weights_override` |
| `DONE.json` | `pretrained: true` i skrót `tfm` (plik wag, sha256, worki, czas predykcji, VRAM) |
| `logs/optuna/tfm_<model>_<klucz>.json`, `<badanie>.best.json` | wyniki kandydatów (pamięć podręczna wznawiania) i wynik wyboru |

**Na co patrzeć:**

- `weights_override` albo dopisek „PRZEKIEROWANIE WAG” w raporcie oznacza wagi testowe: wyniki są
  bezwartościowe.
- `oom_downgrade` jest w porządku (wolniej, te same predykcje). `bags_reduced` znaczy, że budżet czasu
  obciął worki.
- Log loss VAL powinien być bliski log loss holdoutu. Duża różnica wskazuje błąd albo przesunięcie
  rozkładu.

### 6.2 `runs/analyses/tfm_setup/benchmark.json`

- `results[]`: jeden wpis na pomiar (`model`, `n_ctx`, `n_estimators`, `test_rows`, `status` = `ok`,
  `skipped`, `oom` albo `failed`, `fit_s`, `predict_s`, `rows_per_s`, `peak_vram_mb`).
- `estimates.profiles.<profil>.<model>`:
  - `selection_s` (wybór konfiguracji);
  - `final_s` i `final_bags` (finał VAL + TEST);
  - `total_s`;
  - `eta_upper_s`;
  - `within_budget` (czy mieści się w limitach profilu).
- `estimates.experiment_a`: `tfm_s` (jednostki TFM eksperymentu A), `lightgbm_s` (strojony / domyślny /
  przeniesiony LightGBM — formuła `--plan` 14) i `total_s` (suma). Pomiary TFM czyta `--plan` w 14, a ETA
  orkiestratora = jednostki TFM + LightGBM. `device` ≠ `cuda`, atrapa albo przekierowane wagi → plik jest
  ignorowany w szacunkach (taki pomiar i tak trafia do `benchmark_mechanics.json`).

Szacunki skalują pomiar liniowo z liczbą estymatorów i interpolują log-log po rozmiarze kontekstu, więc
to rząd wielkości. Najważniejsze są `peak_vram_mb` (zapas do 16 GB) i `rows_per_s`.

### 6.3 Eksperyment A: `runs/analyses/small_data__triage__<tag>/`

| plik | zawartość |
|---|---|
| `report.md` | po polsku, do pracy: kryteria ustalone przed analizą; ustawienia (tabela dostępności TFM: urządzenie, wersja, wagi, powód pominięcia); **Decyzja**; dla każdego scenariusza tabele metryk, Δ log loss z werdyktem, metryki drugorzędowe, krzywe a + b·n^(−c) z n*, ESI 1; akapit metod; zastrzeżenia |
| `results.json` | wszystkie liczby; `decision`, `methods_paragraph` (gotowy akapit metod), `skipped_models`, `skipped_units`, `tfm_provenance` (wagi, wersje, urządzenie), `scenarios.<scenariusz>.comparisons` / `power_law` / `crossover` |
| `units.csv` | jeden wiersz na jednostkę (scenariusz × n × powtórzenie × model) z metrykami |
| `figures/curves_<scenariusz>.png` | log loss, QWK i niedotriażowanie w funkcji n z przedziałami |
| `figures/delta_<scenariusz>.png` | Δ log loss i Δ QWK względem strojonego LightGBM w funkcji n |
| `summary.md`, `plan.json`, `units/`, `samples/`, `preds/` | skrót; plan i jednostki do wznawiania; zapisane `row_id` podprób (układ sparowany); predykcje |

**Jak czytać:**

- **Δ = model − LightGBM strojony.** Ujemna Δ log loss oznacza, że TFM jest lepszy.
- **Werdykt:**
  - p werdyktu = **max(p dwupoziomowego bootstrapu, p sparowanego testu t, df = R − 1)** — wygrana wymaga,
    by OBA przedziały wykluczały 0. Przy 2–5 powtórzeniach sam percentylowy bootstrap po powtórzeniach
    zaniża niepewność (gdy wszystkie Δ mają ten sam znak, jego p spada do 1/B niezależnie od rozrzutu),
    a sam test t pomija wspólny szum wierszy oceny. R = 1 → bez werdyktu („R = 1 < 2 — bez testu”);
  - „model lepszy” / „odniesienie lepsze” tylko wtedy, gdy to p po korekcie Holma (po rozmiarach n) < 0,05
    **i** |Δ| > MID 0,005;
  - „istotna, ale |Δ| ≤ MID” albo „brak istotnej różnicy” oznacza remis.

  Tabela podaje p testu t, p bootstrapu, p = max i p Holma.
- **Decyzja** podsumowuje werdykty, np. „TFM wygrywa do n = 2000” albo „TFM nie wygrywa przy żadnym n”.
  Dodaje punkt przecięcia krzywych n* z 95% CI. n* poza zakresem danych jest oznaczone jako ekstrapolacja.
  Przedział n* jest podawany tylko wtedy, gdy ≥ 50% replikacji bootstrapu daje ten sam typ przecięcia;
  gdy któryś rozmiar ma < 5 powtórzeń (domyślnie n = 50 000: R = 3), przedział jest oznaczony jako
  **orientacyjny**.
- **`+foreign` kontra sam TFM:** „SZKODZĄ (transfer negatywny)” znaczy, że dane innych oddziałów
  w kontekście pogarszają wynik lokalny.
- **LightGBM przeniesiony** (surowy i z korektą priorów) odpowiada na pytanie, czy mały lokalny TFM
  jest lepszy od dużego modelu z innego ośrodka.
- **ESI 1** jest raportowane tylko dla n ≥ 5000, z przedziałem Clopper–Pearsona. W `site_local` jest
  tylko 68 wizyt ESI 1, więc raport ostrzega o małej liczności.
- Ostrzeżenia na górze raportu (atrapa estymatora, przekierowanie wag, mieszane wagi) unieważniają
  wyniki TFM.

### 6.4 Eksperyment B: `runs/analyses/tfm_blend__triage__<tag>/`

| plik | zawartość |
|---|---|
| `report.md` | 1. bramka (przed liczbami); 2. dane i procedura; 3. Δ względem LightGBM z 95% CI dla każdego zestawu i metody; 4. szczegóły bramki (✓ / ✗ / ?) i reguła C1; 5. **decyzja i dokładne polecenie 08**; 6. reguły decyzyjne w wartościach bezwzględnych (cała populacja i vc); 7. LightGBM sam a zestaw klasyczny; 8. różnorodność (niezgodność argmax, korelacja błędów); 9. wagi i temperatury w foldach; 10. **opis metody do pracy**; 11. zastrzeżenia |
| `recommendation.json` | `decision`, `members`, `runs`, `method`, **`command`** (dokładne polecenie 08), `command_tag`, `reason`, `gate` (status każdego zestawu), `noise_criterion_checked`, `seed_sd` |
| `results.json` | wszystkie liczby: metryki wariantów (wartości z powtórzeń), `deltas` z CI, `gate`, `parsimony`, wagi foldów, `diversity` |
| `figures/` | `delta_forest.png` (Δ z CI dla zestawów), `diversity.png` (korelacja błędów LightGBM–TFM), `weights_folds.png` (stabilność wag) |
| `summary.md`, `plan.json`, `units/`, `preds/` | skrót; jednostki i predykcje cross-fittingu (wznawianie) |

**Jak czytać:**

- Ujemna Δ log loss, Brier, ECE i niedotriażowania oznacza, że zestaw jest lepszy; dodatnia Δ QWK też.
- „SD Δ (powt.)” to szum samego podziału VAL.
- Status bramki: „PRZESZEDŁ”, „nie przeszedł” albo „nierozstrzygnięte” (brak SD seedów; uruchom `09`,
  potem ponownie to samo polecenie `15`; gotowe powtórzenia zostaną użyte, bramka przeliczona).
- Niska korelacja błędów przy podobnej jakości to warunek, żeby blend w ogóle mógł pomóc.
- Zmienne wagi między foldami oznaczają niestabilną recepturę.
- Wszystko tu jest **wyłącznie z VAL**; liczby do pracy na TEST daje 08.

### 6.5 Raport końcowy 08 z TFM

`runs/final_report__triage__<czas>/` jak dotąd (README, punkt 14) plus
`backend_ensemble_weights_no_tfm.json`, gdy receptura ma TFM. `recipe.json` zawiera `no_tfm` (wagi,
temperatura i reguła wariantu bez TFM).

## 7. Tabela decyzji

| wynik | co trafia do systemu końcowego | co trafia do pracy |
|---|---|---|
| **B: zestaw z TFM przeszedł bramkę** | LightGBM + rekomendowany TFM (para tylko przy przewadze ≥ MID). Uruchom 08 poleceniem z `recommendation.json`. Backend: domyślnie receptura bez TFM; TFM tylko po opt-in z CUDA i zmieszczeniu w VRAM. TabPFN-3.5 we wdrożeniu wymaga licencji komercyjnej. Jeśli TabICLv2 jest w granicach MID od TabPFN-3.5, do wdrożenia wybierz TabICLv2 | Wynik pozytywny: tabela Δ z VAL (B, punkty 3–4), potem TEST z 08 z regułą C1 (sekcja 3 raportu 08). Opisz jako porównanie zadeklarowane z góry |
| **B: żaden zestaw nie przeszedł** | sam LightGBM (bez zmian; nie powtarzaj 08, jeśli raport dla niego już jest) | Uczciwy wynik negatywny z VAL: Δ z CI, który warunek bramki zawiódł, różnorodność. TEST nie był użyty dla TFM. Zdanie: „na pełnych danych TFM nie dodały wartości ponad strojony LightGBM” |
| **B: nierozstrzygnięte** | nic nie zmieniaj | Uzupełnij `09 --models lightgbm` i uruchom ponownie `15` |
| **A: TFM wygrywa poniżej n\*** (Holm, \|Δ\| > MID) | nie zmienia modelu na pełnych danych | **Rekomendacja dla scenariusza polskiego:** szpital z mniej niż n\* lokalnymi wizytami używa TFM (najlepiej z licencją permisywną) albo LightGBM przeniesionego z korektą priorów; wybór według wyniku `site_local` i modelu przeniesionego |
| **A: TFM nie wygrywa przy żadnym n** | — | Wynik negatywny: LightGBM strojony na małej próbie (albo przeniesiony z korektą priorów) wystarcza; podaj krzywe i n\* (jeśli jest) |
| **A: `+foreign` szkodzi** | — | Nie dokładaj danych z innych ośrodków do kontekstu TFM w polskim scenariuszu (transfer negatywny) |
| **A: wygrana tylko przy n < 5000** | — | Zaznacz, że metryki ESI 1 nie są tam raportowane (za mało przypadków); wygrana dotyczy log loss, nie bezpieczeństwa |
| **zestaw klasyczny wyraźnie lepszy od samego LightGBM** (B, punkt 7) | rdzeń zostaje LightGBM (decyzja projektowa) | Podaj to porównanie jako kontekst i uzasadnij wybór rdzenia |

## 8. Do pracy: akapity metod

Raporty generują akapit metod z rzeczywistymi liczbami: A w `report.md` → „Akapit metod (do pracy)”
i `results.json` → `methods_paragraph`; B w `report.md` → „10. Opis metody do pracy”. **Używaj tych
wersji.** Poniższe akapity opisują ustawienia domyślne; w nawiasach `[…]` uzupełnij liczby z raportów.

**Modele fundamentalne (konfiguracja).** Jako potencjalnych partnerów LightGBM rozważono dwa wstępnie
wytrenowane tabularne modele fundamentalne działające na zasadzie uczenia w kontekście (*in-context
learning*): TabPFN-3.5 (Prior Labs; biblioteka tabpfn 9.2.0, wagi tabpfn-v3.5-20260909) oraz TabICLv2
(Inria; biblioteka tabicl 2.2.0, checkpoint tabicl-classifier-v2-20260212). Modele te nie są dopasowywane
do danych; prawdopodobieństwa poziomów ESI wyznaczają na podstawie kontekstu złożonego z oznaczonych
wizyt zbioru uczącego. Kontekst losowano warstwowo względem poziomu ESI i oddziału; przy dużym zbiorze
uczącym stosowano do czterech rozłącznych kontekstów („worków”), których predykcje uśredniano. Konfigurację
(liczbę wierszy kontekstu, liczbę estymatorów w zespole biblioteki, strategię doboru kontekstu i — dla
TabICLv2 — ograniczenie liczby cech) wybrano na wydzielonej części zbioru uczącego (20 000 wizyt) według
straty logarytmicznej, nie korzystając ze zbioru walidacyjnego ani testowego; przy różnicy ≤ 0,002
wybierano konfigurację tańszą obliczeniowo. W wariancie kontekstu zbalansowanego zastosowano dokładną
korektę rozkładu klas. Obliczenia wykonano na karcie RTX 4080 Super (16 GB, automatyczna mieszana
precyzja fp16); wersje bibliotek i sumy kontrolne SHA-256 wag zapisano w metadanych każdego przebiegu.

**Eksperyment A.** Sprawdzono, czy modele fundamentalne użyte z ustawieniami domyślnymi bibliotek
przewyższają LightGBM, gdy szpital dysponuje niewielką liczbą oznaczonych wizyt. W scenariuszu „wszystkie
oddziały” z wizyt zbioru uczącego losowano podpróby o liczności n ∈ {500; 1000; 2000; 5000; 10 000;
50 000} (warstwowo względem ESI i oddziału; 10, 10, 10, 5, 5 i 3 powtórzenia), a ocenę prowadzono na
stałej, warstwowej próbie 20 000 wizyt zbioru walidacyjnego. W scenariuszu lokalnym podpróby pochodziły
wyłącznie z oddziału B, a ocenę prowadzono na wszystkich 16 633 wizytach walidacyjnych tego oddziału.
Każdą podpróbę losowano raz i używano jej dla wszystkich modeli (układ sparowany), gwarantując co najmniej
dwie wizyty z każdej klasy ESI. LightGBM strojono wyłącznie na podpróbie (Optuna, 40 prób, wewnętrzna
warstwowa walidacja krzyżowa, funkcja celu: strata logarytmiczna, liczba drzew z wczesnego zatrzymania);
jako odniesienie podano też LightGBM z ustawieniami domyślnymi. Modele fundamentalne zastosowano bez
strojenia, z całą podpróbą jako kontekstem. W scenariuszu lokalnym sprawdzono ponadto wpływ dołączenia
do kontekstu 10 000 wizyt z innych oddziałów (transfer negatywny) oraz LightGBM wytrenowany na innych
oddziałach, oceniany bez zmian i z korektą znanego lokalnego rozkładu ESI (Saerens i in., 2002).
Pierwszorzędową, ustaloną przed analizą miarą była wieloklasowa strata logarytmiczna; drugorzędowe: wynik
Briera, ECE (ogółem oraz dla ESI 1 i ESI 5), makro-AUC, ważona kwadratowo kappa i odsetek
niedotriażowania. Dla każdego n obliczono sparowane różnice względem strojonego LightGBM, ich średnią
z 95% przedziałem t oraz dwupoziomowy bootstrap (1000 losowań: powtórzenia, następnie wizyty warstwowo
względem ESI). Do wnioskowania przyjęto większą z wartości p bootstrapu i sparowanego testu t (różnicę
uznawano za istotną tylko wtedy, gdy oba przedziały wykluczały zero), ponieważ przy kilku powtórzeniach
percentylowy bootstrap po powtórzeniach zaniża niepewność. Wartości p miary pierwszorzędowej skorygowano
metodą Holma względem rozmiarów n; przewagę ogłaszano wyłącznie, gdy skorygowane p < 0,05 i |Δ| > 0,005.
Przebieg straty w funkcji n opisano krzywą a + b·n^(−c), a punkt przecięcia krzywych n\* wyznaczono
z 95% przedziałem bootstrap (orientacyjnym przy mniej niż 5 powtórzeniach). Czułość dla ESI 1
raportowano tylko dla n ≥ 5000, z dokładnym przedziałem Clopper–Pearsona.

**Eksperyment B.** Wpływ dołączenia modeli fundamentalnych do LightGBM na pełnych danych oceniono
wyłącznie na zbiorze walidacyjnym (n = 55 803) metodą pięciokrotnie powtarzanego dwukrotnego
cross-fittingu: zbiór dzielono na dwie połowy warstwowane względem ESI i oddziału; na jednej dobierano
wagi połączenia (minimalizacja straty logarytmicznej), temperaturę i regułę decyzyjną z docelowym
niedotriażowaniem 15% (progi oczekiwanego poziomu ESI z siatką bezpieczeństwa, dopasowane w podgrupie
z kompletem parametrów życiowych), a oceniano je na drugiej; następnie role połówek zamieniano. LightGBM
stanowiący punkt odniesienia przechodził identyczną procedurę. Rozważano zestawy LightGBM z jednym albo
dwoma modelami fundamentalnymi, łączone średnią arytmetyczną lub geometryczną prawdopodobieństw. Różnice
liczono na tych samych pacjentach i uśredniano po powtórzeniach; 95% przedziały ufności wyznaczono
sparowanym bootstrapem warstwowanym względem ESI (1000 losowań). Kryteria włączenia modelu
fundamentalnego ustalono przed analizą: spadek straty logarytmicznej o co najmniej 0,005 z górną granicą
przedziału ufności poniżej zera, większy niż dwukrotne odchylenie standardowe straty LightGBM między
seedami, bez pogorszenia niedotriażowania o ponad 1 punkt procentowy i kalibracji ESI 1 (ECE) o ponad
0,01; zestaw z dwoma modelami fundamentalnymi wybierano tylko, gdy poprawiał najlepszy zestaw z jednym
modelem o co najmniej 0,005. Zbiór testowy wykorzystano jednokrotnie, do oceny wybranej receptury.

**Zdanie wyniku (szablony).**

- B pozytywny: „Zestaw LightGBM + [TabPFN-3.5] spełnił wszystkie kryteria ustalone przed analizą (Δ straty
  logarytmicznej na zbiorze walidacyjnym [−0,0xx; 95% CI −0,0xx do −0,0xx]); na zbiorze testowym [wynik
  z sekcji 3 raportu końcowego].”
- B negatywny: „Żaden zestaw z modelem fundamentalnym nie spełnił kryteriów (najlepszy: [zestaw], Δ =
  [−0,00x; 95% CI …], niespełnione kryterium: [nr]); na pełnych danych modele fundamentalne nie wniosły
  wartości ponad strojony LightGBM, a zbioru testowego dla nich nie użyto.”
- A: „W scenariuszu lokalnym [TabPFN-3.5] był istotnie lepszy od strojonego LightGBM dla n ≤ [2000]
  (Δ = …; p po korekcie Holma < 0,05), a krzywe przecinały się przy n\* ≈ [..] (95% CI …).”

## 9. Ograniczenia (do rozdziału „Dyskusja”)

1. **Deklaracje producentów [P].** Przewaga TabPFN-3/3.5 na dużych zbiorach i wynik TabICLv2 jako
   najlepszy w swojej klasie pochodzą z benchmarków autorów modeli. Nasze eksperymenty są ich niezależnym
   sprawdzeniem, ale tylko na tych danych i tych wersjach wag.
2. **Źródła znane ze streszczeń [S].** Część wyników literatury (BeyondArena, TabArena Elo, anemia,
   destylacja) znamy tylko ze streszczeń lub fragmentów wyszukiwarki. Przed cytowaniem sprawdź pełny
   tekst ([LITERATURA.md](LITERATURA.md#tabularne-modele-fundamentalne-tfm)).
3. **Rzadkie klasy.** ESI 1 to 0,94% TRAIN (4216 wizyt). W kontekście 500 wierszy to ok. 5 wizyt;
   gwarancja ≥ 2 na worek lekko je nadreprezentuje. Literatura opisuje przesuwanie TabPFN w stronę klasy
   większościowej i gorszą kalibrację przy wynikach poniżej 20%. Stąd kryterium ECE ESI 1 w bramce i
   brak metryk ESI 1 dla n < 5000.
4. **Oddział B ma w VAL tylko 68 wizyt ESI 1** (16 633 wizyt). Metryki ESI 1 w scenariuszu lokalnym są
   bardzo niepewne: przedziały Clopper–Pearsona, ostrzeżenie w raporcie.
5. **Dane z USA.** Yale to 3 SOR-y jednego amerykańskiego systemu. „Polski szpital” symulujemy oddziałem B,
   co jest słabym przybliżeniem przejścia USA → Polska. Przewaga TFM przy małych danych nie musi się
   przenieść; potrzebna jest walidacja na danych polskich.
6. **Podpróbkowany kontekst.** TFM widzą część TRAIN: np. 4 worki × 100 tys. = 400 tys. z 446 tys.
   wierszy, a 4 × 50 tys. = 200 tys. TabICLv2 pracuje poza zakresem pretreningu (konteksty ≥ 25 tys.
   wierszy i 271 cech, przy pretreningu do ok. 48 tys. wierszy i 100 cech).
7. **Asymetria A jest zamierzona:** TFM „z pudełka” kontra LightGBM strojony. W B konfiguracje TFM wybrano
   na holdoucie z TRAIN, a nie na VAL.
8. **Szum i moc.** W A przy n = 50 000 są tylko 3 powtórzenia. Sam percentylowy bootstrap po tak małej
   liczbie powtórzeń jest antykonserwatywny (fałszywe „wygrane”), dlatego werdykt używa max(p bootstrapu,
   p testu t) — kosztem mocy; przedział n\* jest wtedy tylko orientacyjny. W B kryterium szumu używa SD seedów LightGBM; zmienność losowania worków
   TFM nie jest w nim ujęta (refity seedów TFM tylko na żądanie: `09 --models tfm`).
9. **Tylko VAL w B.** Wybór metody łączenia (2 warianty) na tych samych oszacowaniach daje niewielki
   optymizm; bramka wymaga przewagi ≥ MID i przedziału bez zera.
10. **Sprzęt i powtarzalność.** Liczby pamięci i czasu w tym dokumencie to modele (CPU jako przybliżenie
    GPU). Predykcje na GPU (fp16 autocast) mogą nieznacznie różnić się między kartami i wersjami
    bibliotek. Kod przetestowano bez GPU, na losowych wagach.
11. **Licencje.** Wagi TabPFN-3.5 są niekomercyjne; licencja checkpointu TabICLv2 jest do potwierdzenia.
12. **Interpretowalność.** Dla TFM nie liczymy SHAP (wolne, niezwalidowane klinicznie). Wyjaśnienia
    w pracy i w aplikacji pochodzą z LightGBM.

## 10. Rozwiązywanie problemów

| objaw | przyczyna i co zrobić |
|---|---|
| `torch.cuda.is_available()` = `False`, wersja kończy się na `+cpu`; preflight: „koło tylko-CPU” | torch z PyPI. `pip uninstall -y torch`, potem `pip install torch --index-url https://download.pytorch.org/whl/cu130` (sterownik < 580: `cu128`). Sprawdź poleceniem z punktu 3 i `python scripts/13_tfm_setup.py --offline-check` |
| `TFMDeviceError: … brak GPU … kontekst worka … > TFM_CPU_MAX_CONTEXT=5,000` | trening bez GPU z dużym kontekstem jest blokowany na starcie (na CPU trwałby wiele godzin). Napraw torch/CUDA; do testu mechaniki: `--profile smoke`. Wymuszenie urządzenia: `$env:SOR_AI_DEVICE_TABPFN = "gpu"` (błąd zamiast cichego CPU) |
| w logu `BRAK PAMIĘCI GPU … krok awaryjny: …` | to jeszcze nie błąd. Wrapper sam schodzi po drabinie. **TabPFN:** połowa `predict_batch` (do 1024) → bufor KV w RAM i na karcie tylko jeden estymator naraz → `fit_preprocessors` z porcją 16 384 (dalej znów połowa, do 1024). **TabICL:** `batch_size=1` → bufor `kv` → `repr` → bez bufora → połowa `predict_batch` (do 1024). Kroki są w `tfm.events` (`oom_downgrade`); zmienia się czas, a predykcje praktycznie nie |
| `TFMOutOfMemoryError: … Zmniejsz: …` | drabina się skończyła. Zamknij programy na karcie: `nvidia-smi`, `ollama ps` → `ollama stop <model>`, gry, przeglądarka z akceleracją. Sprawdź wolną VRAM (`13_tfm_setup.py`). Uruchom to samo polecenie: z pamięci podręcznej brani są tylko kandydaci z wynikiem (`complete`); kandydaci `oom` i nieudani są liczeni ponownie (OOM zależy od stanu karty). Gdy błąd zostaje, zejdź do profilu `standard`. Kandydat z OOM w trakcie wyboru nie przerywa wyboru, jest tylko zapisany jako `oom` |
| TFM liczy się wielokrotnie wolniej niż szacunek, w Menedżerze zadań rośnie „Współdzielona pamięć GPU”; WARNING „czas na jednostkę kosztu …× większy” | sterownik NVIDIA przenosi nadmiar VRAM do RAM zamiast zgłosić OOM (Sysmem Fallback) — drabina awaryjna nie działa, limit czasu wyboru jest sprawdzany tylko między kandydatami. Ustaw „Prefer No Sysmem Fallback” dla `python.exe` z venv (punkt 3, krok 2), zamknij inne programy na karcie i uruchom ponownie |
| WARNING „BUDŻET FINALNEJ PREDYKCJI PRZEKROCZONY” (zdarzenie `final_budget_exceeded`) | nawet 1 worek przekracza `final_budget_s` — liczbą worków nie da się zejść niżej; trening trwa dalej. Zmniejsz `n_ctx` / `n_estimators` (mniejszy profil) albo sprawdź, czy GPU nie jest zajęte / nie spada do pamięci współdzielonej |
| `TFMWeightsError: Brak wag TabPFN v3.5: …` / „Brak checkpointu TabICL … (LocalEntryNotFoundError)”; preflight: brak wag | trening działa **offline** i wag nie pobiera. Raz, z siecią: `python scripts/13_tfm_setup.py`; potem `--offline-check`. Gdy pliki są, a błąd zostaje: inny katalog pamięci podręcznej w tym procesie (`TABPFN_MODEL_CACHE_DIR`, `HF_HOME`, `HF_HUB_CACHE`, plik `.env`; wiersz niżej) |
| `TabPFNHuggingFaceGatedRepoError: … This model is gated …` | komunikat **mylący**: TabPFN zgłasza go także przy braku sieci. `13_tfm_setup.py` rozpoznaje przyczynę i wypisuje instrukcję. **Brak sieci:** sprawdź internet, proxy i zaporę. **Licencja / klucz** (`TabPFNLicenseError`): akceptacja na https://ux.priorlabs.ai i `TABPFN_TOKEN` (punkt 3). **Odmowa Hugging Face:** zaakceptuj warunki na https://huggingface.co/Prior-Labs/tabpfn_3_5, potem `hf auth login` albo `$env:HF_TOKEN`. Czy ten krok jest potrzebny po akceptacji u Prior Labs, nie sprawdziliśmy offline |
| wagi „znikają” w innym katalogu albo w backendzie | TabPFN przy imporcie czyta plik **`.env` z bieżącego katalogu** (klucze `TABPFN_*`, np. `TABPFN_MODEL_CACHE_DIR`). Linia w `training/.env` albo `backend/.env` przekierowuje miejsce szukania wag. `13_tfm_setup.py` ostrzega o takim pliku. Inne klucze w `.env` (np. `OLLAMA_URL`) są nieszkodliwe. Chcesz wagi poza dyskiem C:? Ustaw `setx TABPFN_MODEL_CACHE_DIR D:\tabpfn` (nowe okno) **przed** `13_tfm_setup.py` |
| Windows: `Windows fatal exception: code 0xc000001d` | README TabPFN: prawdopodobnie błąd torch/oneDNN. Ustaw `$env:ONEDNN_MAX_CPU_ISA = "AVX512_CORE_FP16"` (na stałe: `setx`) i uruchom ponownie |
| ostrzeżenie Hugging Face o dowiązaniach (symlinks) na Windows | bez trybu dewelopera Windows nie tworzy dowiązań; checkpoint jest kopiowany (więcej miejsca na dysku). Działa. Wyciszenie: `$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"` |
| `pip install` zgłasza zablokowane pliki / DLL | trwa `run_all.py` w tym venv. Poczekaj do końca treningu (punkt 3) |
| po instalacji (Linux) xgboost nie widzi GPU | podmiana `nvidia-nccl-cu13`. `python scripts/00_check_env.py` (próba GPU xgboost); w razie błędu: `pip install -r requirements-lock.txt` |
| `05_explain.py` kończy się kodem 2 dla runu TFM | zamierzone: SHAP nie jest liczony dla TFM. Wyjaśnienia: `05_explain.py --model-name lightgbm --global` |
| `06_cross_validate.py` odmawia TFM | zamierzone: k-fold CV TFM na pełnych danych trwałby wiele godzin |
| `15_tfm_blend.py`: „nierozstrzygnięte” | brak SD seedów LightGBM na VAL. `09_seed_refits.py --tag <tag> --models lightgbm`, potem ponownie `15` |
| `14_small_data_curve.py`: linie `skipped: tabpfn — …` | brak torch, biblioteki lub wag, albo `--device gpu` bez CUDA. LightGBM liczy się dalej. Po naprawie uruchom to samo polecenie: dopisze brakujące jednostki TFM |
| system „staje” przy TabICL na dużym kontekście (RAM) | TabICL przypina do 32 GB RAM na wyjście osadzeń kolumn. Zamknij inne programy; jeśli to nie pomaga, wybierz profil z mniejszym kontekstem (`standard`). Limit przypinanej pamięci (`max_pinned_memory_mb`) nie jest wystawiony w wrapperze |
| backend nie serwuje TFM mimo `SOR_AI_BACKEND_TFM=1` | WARNING przy starcie podaje powód: brak torch z CUDA, biblioteki, kontekst kodowany przy każdym żądaniu (`context re-encoded on every request` — TabICL `repr`, np. kontekst 50k), za mało wolnej VRAM (suma buforów + `SOR_AI_BACKEND_TFM_RESERVE_MB`), błąd rozgrzewania, predykcja 1 wiersza > `SOR_AI_BACKEND_TFM_MAX_LATENCY_S`. Backend używa wtedy `ensemble_weights_no_tfm.json`. Zwolnij kartę (Ollama) albo zostań przy recepturze bez TFM |
| `make demo` / Streamlit | demo domyślnie otwiera najnowszy run **klasyczny**; runy TFM nie są na liście (tylko jawnie: `streamlit run app/streamlit_app.py -- --model runs/<run>/model.joblib`; worki dopasowane raz, SHAP dla TFM wyłączony) |

## 11. Szczegóły techniczne

### 11.1 Siatki profili (`TUNING_PROFILES` w [`src/utils/config.py`](../src/utils/config.py))

| profil | TabPFN-3.5: kandydaci n_ctx × n_estimators | TabICLv2: kandydaci | holdout | worki | limit wyboru + finału |
|---|---|---|---:|---|---|
| `smoke` | 1000 × 1 (działa na CPU) | 1000 × 1 | 500 | 1 | 15 min + 30 min |
| `quick` | {5k, 10k} × 2 | {5k, 10k} × 2 | 5000 | 1 | 20 min + 20 min |
| `standard` | {25k, 50k} × 4 | {25k, 50k} × 8 | 20 000 | etap worków, do 4 | 2 h + 2 h |
| `thorough` | {25k, 50k, 100k} × {4, 8} + `balanced` 50k × 4 | {25k, 50k} × 8 × {wszystkie cechy, `max_features=100`} + `balanced` 50k × 8 | 20 000 | etap worków, do 4 | 4 h + 4 h |

Przed finałem czas predykcji jest szacowany z przepustowości zmierzonej przy wyborze. Gdy przekracza
limit finału, liczba worków jest zmniejszana (WARNING i zdarzenie `bags_reduced`). Run bez `--tune` używa
parametrów domyślnych (`n_ctx` 50 000), więc bez GPU kończy się od razu błędem.

### 11.2 Ustawienia wrapperów (wybrane)

- **TabPFN-3.5:**
  - `fit_mode="fit_with_cache"`: kontekst przetwarzany raz, bufor KV int8 na karcie;
  - `predict_batch` 32 768;
  - autocast fp16;
  - `TABPFN_MODEL_CACHE_SIZE=1` na czas pętli worków, przywracane po pętli;
  - `softmax_temperature` biblioteki domyślne (temperaturę dobieramy na VAL w łączeniu).
- **TabICLv2:**
  - bufor `kv` tylko wtedy, gdy modelowany rozmiar ≤ 35% VRAM karty, inaczej `repr`;
  - `batch_size` 1 przy `n_ctx` ≥ 10 000, inaczej 4;
  - `predict_batch` z budżetu RAM hosta (`TFM_HOST_RAM_BUDGET_GB` = 24);
  - `max_features=100` usuwa tylko najrzadsze flagi skarg `cc_*`; parametry życiowe, wiek, przybycie,
    flagi braków, agregaty skarg i flagi krytyczne są chronione;
  - redukcja nie używa etykiet.
- **Limit CPU:** `TFM_CPU_MAX_CONTEXT` = 5000 (limit CPU samej biblioteki TabPFN v3/v3.5).

### 11.3 Skąd liczby (plan pamięci, model na CPU)

| wielkość | wartość | uwagi |
|---|---|---|
| bufor KV TabPFN-3.5 (int8) | 3840 B na wiersz kontekstu i estymator + ok. 0,29 MiB na kolumnę modelu | 100k × 8 estymatorów ≈ 3,6 GiB |
| szczyt VRAM TabPFN-3.5 (bufor na karcie) | 100k × 8 ≈ 11 GiB (ponowny pomiar współczynnika: ok. 8,8 GiB); 200k × 8 ≈ 13,9 GiB, „na styk”, dlatego pominięte | dominuje porcja etapów 0–2 (ok. 5,3–7,6 GiB), nie bufor |
| bufor KV TabICLv2 (`kv`, fp16) | 24 576 B na wiersz i estymator; 50k × 8 ≈ 9,6 GiB | > 35% VRAM → `repr` (2048 B na wiersz) |
| artefakt TFM | ok. 0,8 MB na 20 tys. wierszy kontekstu na dysku; w RAM tylko wiersze worków × 271 cech × 4 B (400 tys. wierszy ≈ 0,43 GB) | |
| przepustowość | brak pomiaru GPU | benchmark `13_tfm_setup.py` |

Pomiary dotyczyły architektur z losowymi wagami pełnego rozmiaru na CPU (bf16 jako przybliżenie fp16).
Prawdziwe checkpointy mogą mieć inną konfigurację wnioskowania (liczba kolumn widzianych przez TabPFN,
wartość `n_estimators="auto"`).

### 11.4 Zmienne środowiskowe

| zmienna | znaczenie |
|---|---|
| `SOR_AI_DEVICE_TABPFN`, `SOR_AI_DEVICE_TABICL` (oraz wspólna `SOR_AI_DEVICE`) | `gpu` / `cpu` / `auto`: wymuszenie urządzenia TFM |
| `TABPFN_TOKEN` | klucz API Prior Labs (pobranie wag) |
| `TABPFN_MODEL_CACHE_DIR` | katalog wag TabPFN (domyślnie `%APPDATA%\tabpfn`) |
| `HF_HOME`, `HF_HUB_CACHE`, `HF_TOKEN` | pamięć podręczna i token Hugging Face (TabICL) |
| `SOR_AI_BACKEND_TFM`, `SOR_AI_BACKEND_TFM_RESERVE_MB`, `SOR_AI_BACKEND_TFM_MAX_LATENCY_S` | backend: opt-in serwowania TFM, zapas VRAM (MB), limit czasu predykcji 1 wiersza po rozgrzaniu (s, domyślnie 2) |
| `SOR_AI_TFM_FAKE`, `SOR_AI_TABPFN_MODEL_PATH`, `SOR_AI_TABICL_MODEL_PATH` | **tylko testy**: atrapa estymatora albo plik wag (np. losowe wagi). Aktywne = WARNING, wpis `tfm.weights_override` w `run_info.json` i ostrzeżenie w raportach A/B. Nigdy w zwykłym użyciu |

## 12. Literatura

Źródła z oznaczeniami [P] (deklaracja producenta) i [S] (znane tylko ze streszczenia): sekcja
„Tabularne modele fundamentalne (TFM)” w [LITERATURA.md](LITERATURA.md#tabularne-modele-fundamentalne-tfm).
Pozycje TRIPOD+AI dotyczące TFM: [TRIPOD_AI.md](TRIPOD_AI.md#uzupełnienie-modele-fundamentalne-tfm).
Pozycja w menu usprawnień: [ULEPSZENIA.md](ULEPSZENIA.md#grupa-e-modele-fundamentalne-tfm-eksperymenty-a-i-b).
