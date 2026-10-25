<!-- Zestawienie przygotowane na podstawie analizy „rigor lens” (2026-10-24, wyszukiwanie w sieci). Liczby z literatury
     oznaczone jako „źródło wtórne” albo „do weryfikacji” trzeba sprawdzić w oryginale przed cytowaniem. „Nasz wynik
     orientacyjny” pochodzi z LightGBM bez tuningu na 60 tys. wierszy (VAL), nie z modeli końcowych — w pracy podaj
     liczby z raportu końcowego (08_final_report.py). -->

# Porównanie z literaturą (do rozdziału „Dyskusja”)

## Najpierw: co wolno porównywać

- **Inna etykieta, inna liczba.** Część prac ocenia model względem **ESI pielęgniarki** (tak jak
  my), a część względem **wyniku klinicznego**: przyjęcia, OIOM, zgonu albo eksperckiego „złotego
  standardu”.
  - „Niedotriażowanie” w tym repozytorium oznacza ESI 1–2 według pielęgniarki przewidziane jako
    ESI 3–5.
  - U Sax i in. 2023 mistriage definiuje się przez wynik kliniczny.
  - **Odsetków niedotriażowania nie porównuj między pracami bez zgodnej definicji.**
- **Porównywalne są:**
  - AUC dla ESI 1–2 względem reszty;
  - AUC przewidywania przyjęcia: model vs ESI pielęgniarki, na tych samych pacjentach (sekcja 7
    raportu);
  - QWK model–pielęgniarka, zestawione ze zgodnością między pielęgniarkami.
- **Ten sam zbiór danych co u nas** ma Hong i in. 2018 (Yale). Mają jednak inny wynik (przyjęcie)
  i inną listę zmiennych, a jej w tym repozytorium nie sprawdzono.

## Tabela porównawcza

Kolumna „pewność” mówi, skąd pochodzi liczba z literatury:

- **abstrakt** — z abstraktu albo strony PubMed/PMC, którą widziała analiza „rigor lens”;
- **źródło wtórne** — z omówienia w innej pracy;
- **do weryfikacji** — niepotwierdzona; przed cytowaniem sprawdź w pełnym tekście.

| praca | dane | zadanie / punkt odniesienia | wynik w pracy | pewność | nasz odpowiednik (gdzie w raporcie) | nasz wynik orientacyjny (LightGBM 60k, VAL) |
|---|---|---|---|---|---|---|
| Hong, Haimovich, Taylor 2018, *PLoS One* | **Yale, 560 486 wizyt (te same dane)**, 2014–2017 | przyjęcie do szpitala w chwili triażu | AUC 0,87 dla danych z triażu (LR, XGBoost, DNN); 0,92 z historią pacjenta | abstrakt; lista zmiennych „triage-only” **do weryfikacji** (może zawierać ESI i dane socjodemograficzne, które my wykluczamy) | AUC przyjęcia dla −E[ESI] (sekcja 7) | −E[ESI] 0,779; ESI pielęgniarki 0,769; LightGBM uczony wprost na przyjęciu 0,846 (górny pułap) |
| Levin i in. 2018, *Ann Emerg Med* | 172 726 wizyt, 2 SOR-y | e-triage (las losowy): opieka krytyczna, zabieg, hospitalizacja | AUC 0,73–0,92. Ok. 10% pacjentów ESI 3 przesuniętych wyżej; mieli ≥ 5× częściej wynik krytyczny i 2× częściej przyjęcie | abstrakt | niezgodność model–pielęgniarka a przyjęcia (sekcja 7.2) | argmax podnosi 15% pacjentów ESI 3; są przyjmowani w 48% vs 28% pozostałych (1,7×) |
| Raita i in. 2019, *Crit Care* | NHAMCS, 135 470 wizyt | opieka krytyczna; hospitalizacja | opieka krytyczna: AUC 0,86 (DNN) vs 0,74 (ESI). Hospitalizacja: ok. 0,82 vs 0,69. Mniej niedotriażowanych w ESI 3–5 | opieka krytyczna — abstrakt; hospitalizacja — **źródło wtórne** | AUC przyjęcia, model vs pielęgniarka (sekcja 7) | ESI pielęgniarki w Yale: AUC przyjęcia 0,769 |
| Ivanov i in. 2021, *J Emerg Nurs* (KATE; arXiv 2020) | 166 175 wizyt | ESI względem eksperckiego złotego standardu | trafność 75,9% vs pielęgniarki 59,8%; granica ESI 2/3: 80% vs 41% | abstrakt (arXiv) | — (inna etykieta) | trafność względem pielęgniarki 66,4% — **nieporównywalne** (inny standard odniesienia) |
| Sax i in. 2023, *JAMA Netw Open* | 5,3 mln wizyt, 21 SOR-ów KPNC | mistriage ESI zdefiniowany wynikiem klinicznym | mistriage ok. 32% (overtriage 28,9%, undertriage 3,3%). Czułość dla krytycznie chorych ok. 66%. Częstsze niedotriażowanie u części grup rasowych | ogólne wnioski — abstrakt; liczby szczegółowe — **źródło wtórne** | podgrupy i równość (sekcja 8) | — (inna definicja niedotriażowania; nie porównuj odsetków) |
| Mirhaghi i in. 2015 (metaanaliza) | 19 badań | zgodność ESI między oceniającymi | łączna zgodność ok. 0,79 | **do weryfikacji** (czasopismo i miara zgodności — sprawdź w oryginale) | QWK model–pielęgniarka (sekcja 1) | QWK argmax 0,684 (prognoza z krzywej uczenia dla pełnych danych ok. 0,70). Zgodność między pielęgniarkami ogranicza osiągalną zgodność modelu z etykietą |
| Miles i in. 2020, *Diagn Progn Res* (przegląd) | 25 badań, 92 modele | ML w ocenie pilności | mediana C dla opieki krytycznej: sieci neuronowe 0,89, drzewa 0,85, LR 0,83; raportowanie zwykle słabe | abstrakt | AUC ESI 1–2 vs reszta, AUC ESI 1 (tabele CSV raportu) | AUC ESI 1–2 0,887 (vc 0,896); ESI 1 jeden-przeciw-reszcie 0,953 |
| Fernandes i in. 2020, *Artif Intell Med* (przegląd) | 62 badania | systemy wspomagania decyzji w triażu | niewiele systemów wdrożonych albo zwalidowanych prospektywnie | abstrakt | — | uzasadnia ramę TRIPOD+AI i DECIDE-AI (`docs/TRIPOD_AI.md`) |
| Kwon i in. 2018, *PLoS One* | koreański NEDIS, 151 SOR-ów | DTAS vs KTAS | AUROC śmiertelności 0,935 vs 0,785; walidacja czasowa i zewnętrzna | abstrakt | leave-site-out (sekcja 13.2) | wzorzec walidacji zewnętrznej; walidacji czasowej na Yale nie da się zrobić (brak roku wizyty) |
| Zachariasse i in. 2019, *BMJ Open* (przegląd) | 66 badań | trafność systemów triażu | „rozsądna trafność, ale duża zmienność” | abstrakt | — | tło dla ograniczeń etykiety ESI |
| Hinson i in. 2018, *Int J Emerg Med* | Brazylia, 96 071 wizyt | trafność ESI względem prawdy opartej na wyniku | ok. 17% triażu niedokładne; predyktory niedotriażowania: wiek, bradykardia lub tachykardia, hipoksja | abstrakt | analiza błędów (sekcja 9) | nasze pomyłki skupiają się u młodych pacjentów przychodzących samodzielnie i przy dolegliwościach brzusznych |
| Agius i in. 2025, *BMC Med Inform Decis Mak* | Malta, 653 546 wizyt | XGBoost, poziom ESI | trafność 0,75 (etap triażu) | **do weryfikacji** | — | trafność argmax 0,664 — inna populacja i rozkład etykiet |

## Metody, na które powołuje się raport

| metoda | gdzie w repozytorium | źródło |
|---|---|---|
| Bootstrap percentylowy stratyfikowany po ESI, przedziały Clopper–Pearsona dla odsetków | `src/evaluation/statistics.py`, sekcje 1–3 | Efron i Tibshirani 1993; Clopper i Pearson, *Biometrika* 1934 |
| Różnica dwóch niezależnych odsetków, przedział Newcombe'a (hybryda Wilsona) | `report_sections.newcombe_diff`, sekcja 8 | Newcombe, *Stat Med* 1998;17:873–890 |
| Krzywa decyzyjna (net benefit) | `report_sections.dca_*`, sekcja 10 | Vickers i Elkin, *Med Decis Making* 2006;26:565–574 |
| Hierarchia kalibracji: kalibracja „in the large”, nachylenie | `report_sections.calibration_groups`, sekcja 11 | Van Calster i in., *BMC Med* 2019;17:230 |
| Korekta rozkładu klas (znany rozkład i EM bez etykiet) | `src/analysis/priors.py`, sekcja 13.2 | Saerens, Latinne i Decaestecker, *Neural Comput* 2002;14:21–41 |
| Conformal prediction (LAC, wersja warunkowa po klasie) | `report_sections.conformal_section`, sekcja 12.2 | Vovk i in. 2005; Angelopoulos i Bates 2021 (przegląd) |
| Modele „glass-box” (EBM) zamiast wyjaśniania czarnej skrzynki | `05_explain.py --global` (krzywe kształtu EBM) | Rudin, *Nat Mach Intell* 2019;1:206–215 |
| Wytyczne raportowania | `docs/TRIPOD_AI.md` | Collins i in., *BMJ* 2024;385:e078378 (TRIPOD+AI); Moons i in., *BMJ* 2025 (PROBAST+AI — do weryfikacji) |

## Wnioski do dyskusji

1. **Zgodność z etykietą ma sufit.** Pielęgniarki zgadzają się między sobą na poziomie ok. 0,79
   (Mirhaghi 2015, do weryfikacji). Model, który ma QWK z pielęgniarką ok. 0,70, nie jest więc
   „o 30% gorszy od ideału”. Część różnicy to szum samej etykiety. Wspiera to analiza błędów:
   niedotriażowani pacjenci ESI 2 są przyjmowani do szpitala rzadziej (sekcja 9).
2. **Ciągła ocena modelu przewiduje przyjęcie co najmniej tak dobrze jak ESI pielęgniarki.**
   Jego twarda decyzja 5-poziomowa — już nie. To zgodne z kierunkiem Levin 2018 i Raita 2019.
   Uczciwe porównanie z pielęgniarką to AUC przyjęcia na tych samych pacjentach z przedziałem
   sparowanym (sekcja 7).
3. **Przeniesienie między szpitalami psuje głównie rozkład klas, a nie ranking.** Widać to
   w sekcji 13.2 i podobnie u Kwon 2018 przy walidacji zewnętrznej. W Polsce progi warstwy
   decyzyjnej trzeba dobrać na nowo na lokalnym rozkładzie ESI (TOPSOR).
4. **Odsetków niedotriażowania nie zestawiaj z Sax 2023 ani z Hinson 2018**, bo definicje są
   inne. Porównuj kierunki: kto jest niedotriażowany (sekcje 8 i 9).
