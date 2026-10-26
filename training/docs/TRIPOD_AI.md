<!-- Lista kontrolna przygotowana na podstawie analizy „rigor lens” (2026-10-24). Treść pozycji TRIPOD+AI jest
     PARAFRAZĄ — oficjalnego tekstu nie dało się pobrać. Przed oddaniem pracy porównaj każdą pozycję z PDF:
     Collins i in., BMJ 2024;385:e078378 (lista dla publikacji i osobna dla streszczeń). -->

# TRIPOD+AI: lista kontrolna dla pracy (SOR-AI, triaż ESI)

**TRIPOD+AI** (Collins i in., *BMJ* 2024;385:e078378) to wytyczne raportowania badań, które
budują albo oceniają model predykcyjny, także oparty na uczeniu maszynowym. Ta tabela przypisuje
każdą pozycję do miejsca w repozytorium albo w raporcie końcowym `scripts/08_final_report.py`
(`runs/final_report__…/report.md`, numeracja sekcji jak w raporcie).

**Jak czytać**

- Status:
  - ✓ — repozytorium lub dokumentacja już to pokrywa;
  - ◐ — pokrywa częściowo, brak opisano w kolumnie „co zrobić”;
  - ✗ — brakuje;
  - ✍ — wystarczy tekst w pracy, kod jest niepotrzebny.
- **Treść pozycji jest sparafrazowana.** Numeracja i brzmienie mogą się różnić od oficjalnego
  dokumentu, więc przed oddaniem pracy sprawdź każdą pozycję z PDF-em.
- Charakter badania: **budowa modelu z walidacją wewnętrzną** (losowy podział train/val/test)
  i **zewnętrzną między szpitalami** (leave-site-out, sekcja 13.2 raportu). Tak to nazwij w tytule
  i streszczeniu.

## Tytuł, streszczenie, wstęp

| pozycja | temat (parafraza) | status | gdzie / co zrobić |
|---|---|---|---|
| 1 | Tytuł: model predykcyjny, populacja, wynik, budowa i/lub walidacja | ✍ | np. „Budowa i walidacja (wewnętrzna i między szpitalami) modeli uczenia maszynowego przypisujących poziom ESI…” |
| 2 | Streszczenie wg TRIPOD+AI for Abstracts | ✍ | Osobna lista kontrolna dla streszczeń |
| 3a | Kontekst kliniczny i uzasadnienie | ◐ | `README.md` (TOPSOR, ESI 1–5, kolory). W pracy dopisz literaturę (`docs/LITERATURA.md`) |
| 3b | Zamierzone użycie, użytkownicy, miejsce w ścieżce pacjenta | ◐ | Model **wspiera** pielęgniarkę triażową, a nie ją zastępuje. Odsyłanie niepewnych przypadków: sekcja 12.1 raportu |
| 3c | Znane nierówności zdrowotne w tej dziedzinie | ◐ | Wyniki podgrup (płeć, wiek, sposób przybycia, oddział): sekcja 8 raportu. W tekście przytocz Sax i in. 2023 (gorszy triaż u części grup) i opisz wzorzec z sekcji 8 |
| 4 | Cele: budowa i/lub walidacja | ✍ | — |

## Metody

| pozycja | temat (parafraza) | status | gdzie / co zrobić |
|---|---|---|---|
| 5a | Źródło danych (rejestr, EHR…) i jego reprezentatywność | ◐ | Yale EMMLC (Hong i in. 2018), 3 SOR-y w USA, `DATA_AUDIT.md`. Dopisz różnice USA → Polska (`README.md` §11) |
| 5b | Daty zbierania danych | ◐ | Wizyty z lat 2014–2017. W danych **nie ma roku wizyty**, więc walidacja czasowa jest niemożliwa (`README.md` §10). Napisz to wprost |
| 6a | Miejsce i ośrodki | ✓ | `DATA_AUDIT.md`: 3 oddziały (A, B, C) |
| 6b | Kryteria włączenia i wyłączenia | ✓ | `DATA_AUDIT.md`: dorośli; 560 486 → 558 027 wizyt |
| 6c | Leczenie istotne dla wyniku | ✍ | Etykieta to ocena w chwili triażu, więc leczenie jej nie dotyczy |
| 7 | Przygotowanie i jakość danych, także w podgrupach | ◐ | `DATA_AUDIT.md`, `FEATURE_ENGINEERING.md`, `README.md` §10. Odsetek kompletu parametrów życiowych (vc) per oddział i podgrupa jest w liczebnościach sekcji 1 i 8 raportu; tabelę braków per płeć i wiek dopisz w pracy |
| 8a | Definicja wyniku | ✓ | Natywne ESI 1–5 z Yale, najnowsze w wizycie (po ewentualnym ponownym triażu); `README.md` §11 |
| 8b | Kto ocenia wynik, jego kwalifikacje | ✍ | ESI nadaje pielęgniarka triażowa |
| 8c | Zaślepienie oceny wyniku | ✍ | Brak zaślepienia: pielęgniarka widzi wszystkie predyktory, bo ocena wyniku jest częścią procesu (incorporation). Dlatego raport ma miarę niezależną od etykiety, czyli przyjęcia do szpitala (sekcje 7 i 10.2) |
| 9a | Wybór predyktorów | ✓ | Zestaw `triage` zawiera tylko dane z chwili triażu; wykluczenia z uzasadnieniem w `README.md` §10 |
| 9b | Definicje predyktorów i sposób pomiaru | ✓ | `FEATURE_ENGINEERING.md` |
| 9c | Moment pomiaru predyktorów | ✓ | Dane z chwili triażu; przecieki z przyszłości wizyty usunięte (dyspozycja, leki) |
| 10 | Uzasadnienie liczebności | ◐ | Krzywa uczenia (sekcja 13.3 raportu, `scripts/11_learning_curve.py`) i liczba zdarzeń na klasę (ESI 1 w TRAIN, `DATA_AUDIT.md`) |
| 11 | Braki danych | ✓ | Modele drzewiaste: NaN + flagi braków; logreg: mediana + wskaźnik braku. Mechanizm MNAR opisany w `README.md` §11. Podgrupa vc we wszystkich wynikach |
| 12a | Użycie i podział danych | ✓ | 80/10/10, stratyfikacja ESI × oddział. VAL służy do wyboru modeli, wag, temperatury i progów, TEST do jednej oceny (`README.md` §10, §14) |
| 12b | Obróbka predyktorów | ✓ | `FEATURE_ENGINEERING.md`, `src/features/engineering.py` |
| 12c | Typ modelu, budowa, strojenie hiperparametrów, walidacja wewnętrzna | ✓ | 6 modeli; Optuna z k-krotną CV na TRAIN, log loss, early stopping na wycinku TRAIN (`README.md` §5) |
| 12d | Niejednorodność między ośrodkami | ◐→✓ | Wyniki per oddział (sekcja 6), kalibracja per oddział (sekcja 11), leave-site-out z korektą rozkładu klas (sekcja 13.2, `scripts/10_leave_site_out.py`) |
| 12e | Miary jakości, w tym użyteczność kliniczna | ✓ | QWK, niedotriażowanie, krytyczne pominięcie, AUC, log loss, Brier, ECE z 95% CI (sekcje 1–4), krzywe decyzyjne (sekcja 10), kompromis niedotriażowanie–QWK (punkt pracy na VAL) |
| 12f | Aktualizacja i rekalibracja modelu | ◐ | Temperatura dopasowana na VAL (sekcja 4) i korekta rozkładu klas przy przeniesieniu na inny szpital (sekcja 13.2). W pracy opisz rekalibrację progów na polski rozkład ESI |
| 12g | Sposób liczenia predykcji | ✓ | `backend_ensemble_weights.json` (wagi, metoda, temperatura, reguła i jej parametry), `model.joblib`, `src/evaluation/decision.py` (`decide_rule`) |
| 13 | Niezrównoważenie klas | ✓ | Brak wag klas, więc prawdopodobieństwa są skalibrowane. Asymetria kosztów jest w regule decyzyjnej: progi z celem niedotriażowania T, siatka bezpieczeństwa, decyzja Bayesa (`src/evaluation/decision.py`) |
| 14 | Podejście do równości (fairness) | ◐→✓ | Wykluczone cechy chronione i socjoekonomiczne (`README.md` §10). Ocena w podgrupach: sekcja 8 (Δ z przedziałem Newcombe'a; undertriage „zakotwiczone w przyjęciu”, model vs pielęgniarka) |
| 15 | Wyjście modelu i uzasadnienie progów | ✓ | Reguła główna „progi T” z celem T = 15% dopasowana na VAL vc; krzywa kompromisu, krzywe decyzyjne (p_t ≈ 1/(1 + k)); wrażliwość na mnożnik k (tabela punktu pracy) |
| 16 | Różnice między danymi budowy i oceny | ◐ | Podgrupa vc przybliża populację TOPSOR. Różnice USA → Polska opisz w tekście (`README.md` §11) |
| 17 | Etyka | ✍ | Publiczny, zanonimizowany zbiór (licencja CC-BY, `README.md`). Zgoda komisji bioetycznej niepotrzebna — sprawdź wymagania uczelni |
| 18a–f | Finansowanie, konflikt interesów, protokół, rejestracja, udostępnienie danych i kodu | ◐ | Kod w repozytorium, dane publiczne. Protokołu ani rejestracji nie było — napisz to |
| 19 | Udział pacjentów i społeczeństwa | ✍ | „Brak” z uzasadnieniem (praca magisterska na danych wtórnych) |

## Wyniki

| pozycja | temat (parafraza) | status | gdzie / co zrobić |
|---|---|---|---|
| 20a | Przepływ uczestników | ◐ | Liczby w `DATA_AUDIT.md` (560 486 → 558 027 → podział). Narysuj diagram przepływu |
| 20b | Charakterystyka uczestników (ogółem, per ośrodek i zbiór) | ◐ | Rozkład ESI per zbiór i oddział jest w `DATA_AUDIT.md`. **„Tabeli 1”** (wiek, płeć, przybycie, parametry życiowe, braki) jeszcze nie ma — zrób ją w pracy ze splitów |
| 20c | Porównanie danych budowy i oceny | ◐ | Rozkład ESI w train/val/test (`DATA_AUDIT.md`), zbiór testowy (raport, sekcja „Zbiór testowy”) |
| 21 | Liczebność i liczba zdarzeń w każdej analizie | ◐ | Liczebności `k/n` przy każdym odsetku w raporcie, `run_info.json` (n_fit, n_val). Zbierz w jednej tabeli: tuning, early stopping, VAL i TEST z liczbą ESI 1 |
| 22 | Pełna specyfikacja modelu | ✓ | Artefakty `model.joblib`, `run_info.json` (parametry), `backend_ensemble_weights.json` |
| 23a | Jakość z przedziałami ufności, także w podgrupach | ✓ | Sekcje 1–3 (bootstrap, Clopper–Pearson, sparowane Δ), 8 (podgrupy), 6 (oddziały) |
| 23b | Wyniki niejednorodności | ✓ | Sekcje 6, 11 i 13.2 |
| 24 | Wyniki aktualizacji modelu | ◐ | Temperatura (sekcja 4), korekta rozkładu klas przy leave-site-out (sekcja 13.2) |

## Dyskusja i inne

| pozycja | temat (parafraza) | status | gdzie / co zrobić |
|---|---|---|---|
| 25 | Interpretacja, także pod kątem równości | ✍ | Wyniki sekcji 8, wyjaśnienia globalne (`05_explain.py --global`, C11). Główne zależności: wiek i sposób przybycia jako zastępcze wskaźniki pilności |
| 26 | Ograniczenia | ✍ | Etykieta od pielęgniarki (szum, ok. 0,79 zgodności między pielęgniarkami — Mirhaghi 2015). Brak danych o OIOM i zgonach. Brak roku wizyty i identyfikatora pacjenta. ESI 1 w podgrupie vc rzadkie. Przesunięcie rozkładu klas między szpitalami. Dane z USA |
| 27a–c | Użyteczność: złe lub brakujące dane wejściowe, interakcja z użytkownikiem, dalsze kroki | ✍ | Brakujące parametry życiowe (NaN + flagi), odsyłanie niepewnych (sekcja 12.1), rekalibracja na danych polskich, walidacja prospektywna (DECIDE-AI) |

## PROBAST+AI: samoocena ryzyka błędu

Do pracy dołącz też samoocenę wg **PROBAST+AI** (Moons i in., *BMJ* 2025;388:e082505). Ma część
dla budowy modelu (16 pytań sygnałowych) i część dla jego oceny (18 pytań). Cytowanie i liczbę
pytań sprawdź w oryginale (źródło: analiza „rigor lens”, niezweryfikowane).

Przegląd Miles i in. 2020 pokazuje, że prace o ML w triażu zwykle są słabo raportowane, więc
jawna tabela PROBAST+AI wyróżnia pracę.

Najważniejsze domeny i odpowiedzi z tego repozytorium:

| domena | ryzyko błędu | uzasadnienie |
|---|---|---|
| uczestnicy | niskie–umiarkowane | Kolejne wizyty z 3 SOR-ów, kryteria wyłączenia jawne. Pacjent może się powtarzać między zbiorami, bo brak identyfikatora |
| predyktory | niskie | Tylko dane z chwili triażu; przecieki usunięte; definicje w `FEATURE_ENGINEERING.md` |
| wynik | **wysokie** (dla „prawdziwej pilności”) / niskie (dla „ESI pielęgniarki”) | Etykieta to ocena człowieka, który zna predyktory. Dlatego dodano analizę względem przyjęć |
| analiza | niskie | VAL i TEST rozdzielone; reguły dopasowane tylko na VAL; CI dla wszystkich metryk; zmienność seedów (sekcja 13.1); kalibracja oceniona |

## Uzupełnienie: modele fundamentalne (TFM)

Dotyczy eksperymentów z TabPFN-3.5 i TabICLv2 ([`TFM.md`](TFM.md)). Status ◐ oznacza: kod i raporty są
gotowe, wyników jeszcze nie ma (wymagają GPU i prawdziwych wag).

| pozycja | temat (parafraza) | status | gdzie / co zrobić |
|---|---|---|---|
| 9a, 12b | Predyktory i ich obróbka | ✓ | Te same 271 cech `triage` co modele klasyczne. TabICLv2 ma kandydata `max_features=100`, który usuwa najrzadsze flagi skarg bez użycia etykiet (`TFM.md` §11.2) |
| 10 | Uzasadnienie liczebności | ◐ | Eksperyment A: krzywa małych danych (n = 500–50 000) i próg n\* (`runs/analyses/small_data__…/report.md`) |
| 11 | Braki danych | ✓ | TabPFN obsługuje NaN natywnie; TabICL imputuje średnią, więc flagi `*_missing` zostają w cechach |
| 12c | Typ modelu, budowa, dobór hiperparametrów | ✓ | Modele wstępnie wytrenowane bez dopasowania wag; kontekst z TRAIN (worki warstwowane ESI × oddział); konfiguracja wybrana na holdoucie z TRAIN, bez VAL i TEST (sekcja „Tuning — wybór konfiguracji” w `report.md` runu) |
| 12e, 23a | Ocena połączenia z TFM | ◐ | Eksperyment B: powtarzany cross-fitting na VAL, sparowany bootstrap, bramka; TEST raz w 08 (`runs/analyses/tfm_blend__…/report.md`, sekcja 3 raportu 08) |
| 12g, 22 | Pełna specyfikacja modelu | ✓ | `run_info.json` → `tfm`: wersje bibliotek i torch, plik wag + SHA-256, konfiguracja, worki, zdarzenia OOM. Kontekst jest zapisany w `model.joblib` |
| 13 | Niezrównoważenie klas | ✓ | Każdy worek ma ≥ 2 wiersze każdej klasy; kontekst `balanced` z dokładną korektą priorów; ECE ESI 1 jest kryterium bramki B |
| 18c–d | Protokół i rejestracja | ◐ | Kryteria A i B ustalono przed analizą: stałe w kodzie, wypisywane w raportach przed liczbami. Publicznej rejestracji nie było — napisz to |
| 18e | Dostępność modelu | ✍ | Wagi TabPFN-3.5 mają licencję niekomercyjną; kod TabICL ma licencję BSD-3, a licencję jego wag trzeba potwierdzić na karcie modelu (`TFM.md` §2) |
| 26 | Ograniczenia | ✍ | Deklaracje producentów, rzadkie ESI 1, oddział B z 68 wizytami ESI 1 w VAL, podpróbkowany kontekst, dane z USA (`TFM.md` §9) |
| 27a–c | Wdrożenie | ✍ | Backend domyślnie nie używa TFM (receptura bez TFM). Serwowanie wymaga opt-in, GPU i licencji komercyjnej TabPFN |

PROBAST+AI, domena „analiza”: ryzyko optymizmu przy wyborze połączenia z TFM ograniczają cross-fitting na
VAL, bramka ustalona z góry (MID, przedział bez zera, 2 × SD seedów) i jednorazowa ocena na TEST.
