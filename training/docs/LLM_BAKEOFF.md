# Porównanie modeli językowych do wyjaśnień triażu ESI (bake-off)

Skrypt `scripts/12_llm_bakeoff.py` wybiera model językowy (LLM), który będzie pisał polskie
wyjaśnienia decyzji systemu triażu ESI (a później działał jako agent z narzędziami). Ten dokument
opisuje, co jest mierzone, jak uruchomić pomiar w domu, jak czytać `report.md` i jak opisać wynik
w pracy.

## 1. Kryterium wyboru (uzgodnione)

Wybieramy **model z najlepszym wynikiem w NASZYM pomiarze**:

1. **Kryterium główne:** trafność na polskich pytaniach egzaminacyjnych (LEK + PES) w protokole
   **abstaining_substitution** zbioru NASK (protokół „prawdziwej kompetencji”, zob. punkt 2).
2. **Remis** (95% przedział sparowanej różnicy z najlepszym modelem zawiera 0) rozstrzyga
   trafność w protokole **standardowym** (ta sama reguła remisu), a dalej wynik części **esi**
   (średnia: poprawna struktura odpowiedzi, poziom ESI zgodny z decyzją, pokrycie cech SHAP,
   brak zmyślonych pomiarów, odpowiedź po polsku).
3. **Możliwość wdrożenia w szpitalu nie jest kryterium** — raport podaje ją jako notatkę przy
   każdym modelu (licencja, rozmiar, czy da się uruchomić lokalnie).

Kandydaci (domyślnie, tagi Ollama Cloud): `glm-5.3:cloud`, `gpt-oss:120b-cloud`,
`deepseek-v4.1-flash:cloud`, `kimi-k3:cloud`. Listę zmienia opcja `--models`.

## 2. Co jest mierzone

| Część | Dane | Zadanie modelu | Główne metryki |
|---|---|---|---|
| `exams` | NASK-PIB/Reassess-Polish-Medical-Exams: 300 pytań LEK (losowanie warstwowe wg edycji) + do 200 pytań PES z medycyny ratunkowej i chorób wewnętrznych (gdy nie da się ich rozpoznać — losowe PES); seed 42, te same pytania dla każdego modelu | odpowiedź A–E, prompt NASK „prev_method” po polsku („Odpowiedz w sposób: Prawidłowa odpowiedź to B.”) | trafność z dokładnym 95% przedziałem Clopper–Pearsona (osobno LEK, PES, razem), odsetek odpowiedzi bez litery, porównanie z najlepszym modelem (dokładny test McNemara + sparowany bootstrap Δ), luka standard − abstaining (wskaźnik „bias” NASK), opóźnienie, tokeny, koszt |
| `esi` | 40 przypadków z VAL (tylko wiersze z kompletem HR/SBP/RR), warstwy wg ESI z triażu 8/12/10/6/4 (ESI 1–2 nadreprezentowane), seed 42 | wyjaśnienie deterministycznej decyzji systemu dla pielęgniarki triażowej przez narzędzie `submit_explanation`; wolno tylko doradczo zasugerować WYŻSZY priorytet | odsetek poprawnych odpowiedzi, poziom w odpowiedzi = decyzja, pokrycie 5 najważniejszych cech SHAP, zmyślone liczby (halucynacje), trafne / fałszywe sugestie eskalacji (wobec ESI z triażu), polszczyzna, cytowanie źródeł, opóźnienie, tokeny, koszt + **ślepa ocena ekspercka** |
| `tools` | te same przypadki | agent (maks. 6 kroków) z narzędziami `explain_model` (SHAP), `what_if` (ponowne przeliczenie receptury po zmianie parametrów), `esi_criteria` (własna parafraza ESI v4), `search_knowledge` (15 krótkich notatek) i `submit_explanation` | ukończenie, poprawne wywołania narzędzi (znana nazwa + argumenty zgodne ze schematem), zmyślone narzędzia, średnia liczba kroków, poprawność `what_if`, metryki odpowiedzi końcowej jak w `esi`, opóźnienie, tokeny, koszt |

**Dwa protokoły egzaminu (te same pytania).** *Standardowy* — pytanie bez zmian.
*abstaining_substitution* — tekst POPRAWNEJ opcji zastąpiony zdaniem „żadna z odpowiedzi nie jest
poprawna” (litera poprawnej odpowiedzi bez zmian), dokładnie jak w skrypcie NASK
`data_creator/create_abstaining_substitution.py`, z jego wykluczeniami (pytania z „żadna”,
„najczęstszy”, „z wyjątkiem”… i pytania, które NASK przenosi do części otwartej). Model, który
rozpoznaje wzorzec odpowiedzi zamiast wiedzy, traci tu najwięcej — w pracy NASK najlepsze modele
traciły ok. 30 pp. Losujemy tylko pytania kwalifikujące się do obu protokołów, więc luka
standard − abstaining jest sparowana na identycznych pytaniach.

**Przypadki ESI.** Każdy przypadek to polski opis pacjenta (wiek, płeć, sposób przybycia,
parametry z jednostkami, skargi główne, tlenoterapia) [S1], decyzja systemu z
prawdopodobieństwami [S2], 5 najważniejszych cech SHAP (jeden model drzewiasty receptury) [S3]
i flagi reguł bezpieczeństwa ESI [S4]. Decyzję liczy **ta sama receptura co w raporcie końcowym**
(`--recipe runs/final_report__*/recipe.json`: modele, wagi, temperatura, reguła) — na żywo z
zapisanych modeli, tą samą ścieżką co `what_if`. ESI z triażu (pielęgniarka) nie trafia do
promptu; służy tylko do oceny sugestii eskalacji.

## 3. Instalacja w domu

```powershell
cd C:\dev\Magisterka\training                      # w aktywnym venv (README, punkt 2)
pip install -r requirements-app.txt                # datasets + huggingface_hub (pobranie pytań NASK)
```

**Ollama (modele chmurowe).** Zainstaluj aplikację Ollama (https://ollama.com/download) i
zaloguj demon do chmury — plan Pro pozwala na 3 równoległe zapytania (domyślne `--concurrency 3`):

```powershell
ollama signin                                      # jednorazowo; demon na http://localhost:11434
ollama run gpt-oss:120b-cloud "Napisz OK"          # szybki test jednego modelu chmurowego
```

Alternatywa bez demona: zapytania wprost do `https://ollama.com` z kluczem API
(https://ollama.com/settings/keys). Skrypt sam obcina przyrostek `:cloud` / `-cloud` z nazw
modeli:

```powershell
$env:OLLAMA_API_KEY = "<klucz>"                    # Linux: export OLLAMA_API_KEY=<klucz>
python scripts/12_llm_bakeoff.py run --base-url https://ollama.com
```

**Hugging Face (pytania NASK).** Zbiór `NASK-PIB/Reassess-Polish-Medical-Exams` wymaga konta i
akceptacji warunków na stronie https://huggingface.co/datasets/NASK-PIB/Reassess-Polish-Medical-Exams
(przycisk na stronie zbioru), a potem tokenu:

```powershell
hf auth login                                      # albo: $env:HF_TOKEN = "<token z https://huggingface.co/settings/tokens>"
```

Bez dostępu do HF można podać pytania z pliku: `--exam-file pytania.csv` (CSV UTF-8 albo
parquet z kolumnami `question, answer, exam_name, id` (+ `uuid`, `split`) jak w zbiorze NASK;
kilka plików — opcja kilka razy, prefiks `r_lek=` / `r_pes_latest=` wskazuje egzamin).

## 4. Kolejność poleceń

```powershell
# 0. Plan i szacunek kosztu (bez zapytań do modeli): liczba zapytań, tokeny, $ wg cennika, czas
python scripts/12_llm_bakeoff.py run --dry-run

# 1. TERAZ (bez wytrenowanych modeli): tylko egzaminy — 500 pytań × 2 protokoły × 4 modele = 4000 zapytań
python scripts/12_llm_bakeoff.py run --parts exams

# 2. PO TRENINGU i raporcie końcowym (README, punkt 14–15): wszystkie części jednym poleceniem
python scripts/12_llm_bakeoff.py run --recipe runs/final_report__triage__<czas>/recipe.json

# 3. Ślepa ocena ekspercka (punkt 6), po zebraniu arkuszy:
python scripts/12_llm_bakeoff.py score-ratings --csv oceny_AB.csv --csv oceny_CD.csv --key runs/llm_bakeoff/<czas>/key.json
```

- Krok 2 uruchamia **wszystkie trzy części**, ale odpowiedzi egzaminacyjne z kroku 1 bierze z
  pamięci podręcznej (zero nowych zapytań, o ile nie zmienisz `--n-exam`, `--n-pes`, `--seed`,
  `--think`, `--temperature` ani wersji zbioru). Dzięki temu jeden `report.md` zawiera egzaminy,
  części esi i tools oraz pełny ranking z rozstrzyganiem remisów.
- `--recipe` wskazuje recepturę z `08_final_report.py` (plik `recipe.json` albo cały katalog
  raportu). Bez niej skrypt bierze najnowsze runy po tuningu i dopasowuje wagi, temperaturę i
  regułę `target` na VAL (zapisze to w raporcie). Bez żadnych runów części esi/tools są
  pomijane z komunikatem, a egzaminy liczą się normalnie.
- Linux: `make llm-bakeoff` (wszystkie części), `make llm-bakeoff PARTS=exams`,
  `make llm-bakeoff RECIPE=runs/final_report__triage__<czas>/recipe.json`,
  `make llm-bakeoff LLM_ARGS="--dry-run"`.

**Wznawianie.** Każda udana odpowiedź od razu trafia do `runs/llm_bakeoff/cache/responses.jsonl`
(klucz: część, model, pozycja, skrót promptu). Po przerwaniu (Ctrl+C, restart, utrata sieci)
uruchom **to samo polecenie** — pójdą tylko brakujące albo nieudane zapytania. `--fresh` ignoruje
pamięć podręczną. Błędy 429 / 5xx / przekroczony czas są ponawiane z wykładniczym odstępem (do
`--max-retries 5`); model, który zawiódł 20 razy z rzędu, jest w tym przebiegu pomijany (jego
pozycje liczą się jako błędne, raport ostrzega na górze — uruchom polecenie ponownie). Na początku
skrypt wysyła do każdego modelu krótkie zapytanie testowe i wyłącza modele, które nie odpowiadają
(`--no-preflight` wyłącza ten test).

**Kody wyjścia:** 0 — OK; 1 — któraś część się nie udała; 2 — złe argumenty, brak klucza API
albo pytań; 130 — Ctrl+C.

### Najważniejsze opcje `run`

| opcja | domyślnie | znaczenie |
|---|---|---|
| `--parts` | `exams,esi,tools` | które części liczyć |
| `--models` | 4 modele chmurowe | tagi Ollamy po przecinku |
| `--think` | `low` | rozumowanie: `low` / `medium` / `high` / `max` / `true` / `false` / `default`; model bez żądanego poziomu dostaje najbliższy obsługiwany wg `/api/show` (albo `true`, gdy ma tylko włącz/wyłącz), a model, który odrzuci `think`, dostaje zapytanie bez niego — wartość faktycznie wysyłana jest w raporcie (kolumna „think (wysyłane)”) i w logu |
| `--n-exam`, `--n-pes` | 300, 200 | liczba pytań LEK i maks. liczba pytań PES |
| `--n-cases` | 40 | liczba przypadków ESI (części esi i tools) |
| `--recipe` | — | `recipe.json` z `runs/final_report__*/` |
| `--exam-file` | — | pytania z pliku zamiast z HF (można kilka) |
| `--concurrency` | 3 | równoległe zapytania (plan Pro: 3) |
| `--base-url`, `--api-key-env` | `http://localhost:11434`, `OLLAMA_API_KEY` | adres Ollamy; `https://ollama.com` = bezpośrednio z kluczem |
| `--out`, `--cache`, `--fresh` | `runs/llm_bakeoff/<czas>/`, `runs/llm_bakeoff/cache/responses.jsonl` | katalog wyniku, pamięć podręczna, ignorowanie jej |
| `--dry-run` | — | tylko plan i szacunek kosztu |
| `--prices` | cennik z 2026-10-14 | JSON `{model: {"input": $, "output": $}}` (USD / 1M tokenów) |
| `--temperature`, `--seed`, `--n-boot` | 0, 42, 2000 | temperatura egzaminów, ziarno losowania, liczba losowań bootstrapu |
| `--timeout`, `--max-retries` | 180 s, 5 | limit jednego zapytania i liczba ponowień |
| `--decimal-comma` | — | przecinek dziesiętny w tabelach `report.md` |

## 5. Wyniki i jak je czytać

Katalog `runs/llm_bakeoff/<czas>/`:

| plik | zawartość |
|---|---|
| `report.md` | raport po polsku, tabele gotowe do pracy |
| `run_info.json`, `bakeoff.log` | warunki pomiaru: polecenie, data, wersja serwera Ollamy, `/api/show` każdego modelu (rodzina, parametry, kwantyzacja, model zdalny, digest), modele wyłączone po teście łączności |
| `results.json` | wszystkie liczby (do własnych tabel i wykresów) |
| `raw_responses.jsonl` | surowe odpowiedzi modeli (także z pamięci podręcznej) |
| `exam_items.csv`, `exam_responses.csv`, `exam_metrics.csv` | wylosowane pytania, odpowiedź każdego modelu, metryki model × protokół × egzamin |
| `esi_cases.json`, `esi_items.csv`, `esi_summary.csv` | przypadki (z recepturą i SHAP), wynik każdego wyjaśnienia, metryki na model |
| `tools_items.csv`, `tools_summary.csv`, `tools_trajectories.jsonl` | część narzędziowa i kroki agenta |
| `ratings/ratings.html`, `ratings/ratings_template.csv` | ślepy arkusz ocen (dla oceniających) |
| `key.json` | klucz liter → modeli (**nie dawać oceniającym**) |

Sekcje `report.md`:

1. **Wynik w skrócie** i ewentualne ostrzeżenie o błędach API (wtedy uruchom polecenie ponownie).
2. **Warunki pomiaru** — data, serwer, tagi i informacje `/api/show` (do sekcji „metody”).
3. **Polskie egzaminy lekarskie** — trafność [95% CI] (poprawne/wszystkie) dla LEK, PES i razem
   w obu protokołach; porównanie z najlepszym modelem (Δ w pp z 95% CI z bootstrapu sparowanego,
   pary niezgodne i p z dokładnego testu McNemara; „remis” = przedział zawiera 0); luka
   standard − abstaining (duża luka = model zgaduje wzorzec odpowiedzi); opóźnienia i tokeny.
4. **Część ESI** i **część narzędziowa** — tabele metryk (definicje pod tabelami).
5. **Koszty i zużycie tokenów** — pozycje, tokeny, mediana czasu na pozycję i koszt wg cennika
   API (przy abonamencie — koszt orientacyjny).
6. **Wybór modelu** — kryterium, wybrany model z uzasadnieniem (kryterium główne / remis →
   standard / remis → ESI / remis statystyczny → wartość punktowa), ranking i notatki o
   wdrożeniu.
7. **Ograniczenia** i **pliki przebiegu**.

Jeśli raport pisze „remis statystyczny → wartość punktowa”, różnice między czołowymi modelami
mieszczą się w niepewności pomiaru — zwiększ próbę (`--n-exam 600 --n-pes 400`) albo opisz w
pracy, że modele są równoważne w granicach błędu.

## 6. Ślepa ocena ekspercka (części esi)

1. Wyślij oceniającym (pielęgniarki / lekarze SOR) **tylko** plik `ratings/ratings.html` (działa
   offline w przeglądarce). W każdym przypadku wyjaśnienia modeli są oznaczone literami A, B, C, D
   w losowej kolejności (inna w każdym przypadku); nazw modeli i ESI z triażu nie widać.
2. Oceniający wpisuje inicjały, ocenia każde wyjaśnienie w skalach 1–5 — **zgodność kliniczna**,
   **przydatność** dla triażysty, **jasność** — i zaznacza **halucynację** (informacje sprzeczne z
   danymi albo zmyślone). Oceny zapisują się w przeglądarce; na końcu „Pobierz CSV”
   (`oceny_<inicjały>.csv`). Zamiast HTML można wypełnić `ratings_template.csv` w arkuszu
   (w polskim Excelu otwórz go przez Dane → Z tekstu/CSV; zapis ze średnikami i w kodowaniu
   Windows-1250 też jest wczytywany).
3. Zbierz pliki i policz wyniki (klucz `key.json` z tego samego przebiegu):

```powershell
python scripts/12_llm_bakeoff.py score-ratings --csv oceny_AB.csv --csv oceny_CD.csv --key runs/llm_bakeoff/<czas>/key.json
```

Wynik: średnia każdej skali na model z 95% CI (bootstrap po przypadkach — oceny jednego przypadku
nie są niezależne), odsetek halucynacji z przedziałem Clopper–Pearsona, liczba ocen i
oceniających; zapis w `ratings_scores.md` i `ratings_scores.json` (domyślnie w katalogu
pierwszego CSV, `--out` zmienia katalog). Ślepa ocena **potwierdza** wybór z kryterium głównego —
jeśli przeczy mu wyraźnie (np. wybrany model ma najwięcej halucynacji), opisz to w pracy jako
ograniczenie albo rozważ zmianę kryterium z promotorem.

## 7. Jak opisać i cytować w pracy

**Metoda (propozycja akapitu).** „Model językowy do generowania wyjaśnień wybrano na podstawie
własnego pomiaru. Kryterium głównym była trafność na N pytaniach polskich egzaminów lekarskich
(LEK: n₁, PES z medycyny ratunkowej i chorób wewnętrznych: n₂) ze zbioru
Reassess-Polish-Medical-Exams [Lasik i in., 2026] w protokole abstaining_substitution, w którym
tekst poprawnej odpowiedzi zastąpiono stwierdzeniem »żadna z odpowiedzi nie jest poprawna«.
Trafność podano z dokładnym 95% przedziałem Clopper–Pearsona; modele porównano parami (dokładny
test McNemara, sparowany bootstrap różnicy, 2000 losowań). Wynik potwierdzono na 40 polskich
opisach przypadków triażowych (zgodność z decyzją systemu, wierność liczb, pokrycie czynników
SHAP, ślepa ocena ekspercka) i w teście użycia narzędzi. Modele uruchomiono przez Ollama Cloud
(tagi, wersja serwera i data pomiaru: tabela X), z poziomem rozumowania »low« i temperaturą 0 w części
egzaminacyjnej.”

Liczby (n₁, n₂, trafności, przedziały, p) przepisz z `report.md`; datę pomiaru, wersję serwera
i tagi modeli — z sekcji „Warunki pomiaru” (także w `run_info.json`). Podaj też źródło pytań:
raport zapisuje rewizję zbioru HF (albo sumę SHA-256 pliku).

**Bibliografia:**

```bibtex
@article{lasik2026reassessinghighperformingllmspolish,
  title         = {Reassessing High-Performing LLMs on Polish Medical Exams: True Competence or Bias-Driven Performance?},
  author        = {Antoni Lasik and Jakub Pokrywka and Łukasz Grzybowski and Jeremi Ignacy Kaczmarek and Gabriela Korzańska and Janusz Świeczkowski-Feiz and Oskar Pastuszek and Paulina Hoffman and Jakub Tomasz Dąbrowski and Wojciech Kusa},
  year          = {2026},
  eprint        = {2606.12250},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.12250}
}
```

(cytowanie podane przez autorów w repozytorium https://github.com/NASK-NLP/Reassess-Polish-Medical-Exams;
zbiór: https://huggingface.co/datasets/NASK-PIB/Reassess-Polish-Medical-Exams — przed oddaniem pracy
sprawdź, czy jest już wersja konferencyjna). Pierwotny zbiór polskich egzaminów: Grzybowski i in.,
„Polish Medical Exams: A new dataset for cross-lingual medical knowledge transfer assessment”,
Findings of EMNLP 2025 (https://aclanthology.org/2025.findings-emnlp.480/). Przy modelach podawaj
pełny tag Ollamy i datę pomiaru (np. „gpt-oss:120b-cloud, Ollama Cloud, dostęp 2026-11-02”).

## 8. Ograniczenia (do rozdziału „ograniczenia”)

- **Tagi chmurowe się zmieniają.** Pod tym samym tagiem dostawca może podmienić model. Dlatego
  `run_info.json` i `report.md` zapisują datę, wersję serwera Ollamy i informacje `/api/show`
  (digest, gdy serwer go podaje). Przy powtórzeniu pomiaru porównaj je przed zestawieniem wyników.
- **Structured outputs nie działają w chmurze.** Modele chmurowe Ollamy ignorują parametr
  `format`, więc strukturę wymusza jedyne udostępnione narzędzie `submit_explanation`; gdy model
  go nie wywoła — tolerancyjne parsowanie JSON z treści i jedno ponowienie z opisem błędu.
  „Poprawne wyjście” mierzy więc także posłuszeństwo wobec instrukcji (w tabeli osobno „ścisły
  schemat” — bez żadnych poprawek).
- **Rozumowanie nie jest wyrównane.** „low” u jednego dostawcy to nie to samo co „low” u innego, a
  część modeli ma tylko włącz/wyłącz — porównujemy modele w ustawieniu „oszczędnym”, nie przy tym
  samym budżecie tokenów (wysyłana wartość `think` jest w raporcie; tokeny wyjścia — w tabelach).
- **Kontaminacja.** Pytania egzaminacyjne mogły trafić do danych treningowych modeli (część LEK
  pochodzi z publicznej bazy); protokół abstaining_substitution częściowo to kontroluje.
- **Determinizm.** Egzaminy: temperatura 0 i stały seed (wyjaśnienia: domyślna temperatura
  modelu), ale modele chmurowe nie gwarantują identycznych odpowiedzi; przedziały ufności
  opisują niepewność z próby pytań, nie zmienność modelu. Pamięć podręczna zapewnia, że raport
  da się odtworzyć z tych samych odpowiedzi.
- **Przypadki ESI** to dane z USA (Yale) opisane po polsku, mała próba (40) z celową
  nadreprezentacją ESI 1–2 — odsetki nie są reprezentatywne dla populacji SOR.
- **Metryki tekstowe są heurystyczne** (słowa kluczowe, rdzenie, tolerancje liczb); rozstrzyga
  ślepa ocena ekspercka.
- **`search_knowledge`** to 15 własnych krótkich notatek (zastępstwo, zanim powstanie właściwy
  RAG), a `esi_criteria` — własna parafraza punktów decyzyjnych ESI v4, nie tekst podręcznika.
- **Prywatność.** W pomiarze nie ma danych polskich pacjentów; przy wdrożeniu z prawdziwymi danymi
  model chmurowy wymaga umowy powierzenia przetwarzania (RODO) albo wdrożenia lokalnego
  (notatki o wdrożeniu w raporcie).
- **Koszt** liczony z cennika API (katalog Ollama Cloud z 2026-10-14, `--prices` nadpisuje) —
  przy abonamencie to wartość orientacyjna.

## 9. Test bez Ollamy (atrapa serwera)

Całą ścieżkę można sprawdzić bez internetu i bez modeli — atrapa odpowiada na egzaminy (z kluczem
odpowiedzi), wyjaśnienia ESI i kroki agenta; kolejne modele są coraz słabsze:

```powershell
python -m tests.fake_ollama --models glm-5.3:cloud,gpt-oss:120b-cloud --exam-csv pytania.csv --port 11500
# w drugim oknie:
python scripts/12_llm_bakeoff.py run --base-url http://127.0.0.1:11500 --models glm-5.3:cloud,gpt-oss:120b-cloud --exam-file pytania.csv --out runs/llm_bakeoff/test_atrapy --cache runs/llm_bakeoff/test_atrapy/cache.jsonl
```

Wyniki atrapy nic nie mówią o modelach — to tylko test instalacji. Osobna `--cache` nie miesza
odpowiedzi atrapy z prawdziwymi. Testy automatyczne: `python -m pytest tests -q -k llm_eval`.
