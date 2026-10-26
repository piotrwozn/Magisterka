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
| Korekta Holma dla wielu porównań (eksperyment A, TFM) | `src/analysis/small_data.py` | Holm, *Scand J Stat* 1979;6:65–70 |
| Model fundamentalny uczący się w kontekście (TabPFN) | `docs/TFM.md` | Hollmann i in., *Nature* 2025, doi:10.1038/s41586-024-08328-6 (TabPFN v2; wersja 3.5 — raport producenta, sekcja TFM niżej) |

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

## Tabularne modele fundamentalne (TFM)

Do rozdziału o eksperymentach A i B ([`TFM.md`](TFM.md)). Źródła zebrano 2026-10-25. Repozytoria GitHub
(README, kod, metadane) czytano bezpośrednio. Strony arXiv, medRxiv, HF i tabarena.ai nie dały się pobrać,
więc wyniki prac pochodzą z abstraktów, stron wyszukiwarki i streszczeń.

Oznaczenia:

- **[P]** — twierdzenie producenta modelu;
- **[S]** — znane tylko ze streszczenia lub fragmentu w wyszukiwarce; przed cytowaniem sprawdź pełny tekst;
- **[?]** — źródło niezweryfikowane albo źródła się różnią.

### Benchmarki i raporty producentów

| praca / źródło | dane | wynik | ozn. | znaczenie dla nas |
|---|---|---|---|---|
| TabArena (arXiv 2506.16791; metadane w repozytorium autogluon/tabarena) | 51 zbiorów; tylko 2 mają ≥ 100 tys. wierszy (największy 150 tys., mediana ok. 6,5 tys.); 8 wieloklasowych | TFM w czołówce, np. Mitra‑v2 1774,6 Elo, strojony i ensemblowany LightGBM 1404,0 [S]. Post-hoc ensemble różnych rodzin poprawia wynik, ale wagi dobierane na walidacji się przeuczają [S] | [S] | ranking dotyczy głównie małych danych, nie naszej skali (446 tys. wierszy); stąd cross-fitting w eksperymencie B |
| BeyondArena (arXiv 2606.30410) | 142 zbiory, do 1,3 mln wierszy | TFM wygrywają na małych i średnich danych IID; GBDT i strojone MLP prowadzą na danych dużych, czasowych, grupowanych i wysokowymiarowych. Rozmiar zbioru to jeden z najsilniejszych predyktorów przewagi GBDT; braki danych działają w tę samą stronę, słabiej | [S] | hipoteza: w B mały efekt albo żaden, w A możliwa przewaga TFM |
| Raport TabPFN‑3 (Prior Labs, PDF) | klasyfikacja, 100 tys.–1 mln wierszy | przewaga nad XGBoost i CatBoost strojonymi 8 h nieistotna statystycznie; istotna tylko nad strojonym LightGBM | [P] | deklaracja, którą B sprawdza na danych SOR |
| Raport TabPFN‑3.5 (arXiv 2609.17895) | — | strojone MLP nadal najlepsze na danych dużych, czasowych i grupowanych; TabPFN‑3.5 tylko „zmniejsza dystans” | [S][P] | ograniczenie na pełnych danych |
| TabICLv2 (arXiv 2602.11139; README soda-inria/tabicl) | TALENT; pretrening 300–48 tys. wierszy, 2–100 kolumn | autorzy deklarują SOTA na TALENT [S][P]; README: model „może się pogarszać” znacznie powyżej 100 cech | [S][P] | u nas 271 cech i kontekst 25–50 tys., poza zakresem pretreningu; stąd kandydat `max_features=100` |
| AutoGluon, wydania 1.4–1.6 (GitHub) | portfele modeli | portfel TFM tylko dla zbiorów ≤ 30 tys. (1.4); zyski w 1.5 „głównie do 100 tys. wierszy” | — | brak kontrolowanego dowodu dla blendu TFM ze strojonym GBDT na ≥ 100 tys. wierszy |
| Ensemble i kalibracja TFM (arXiv 2605.18696) | TabArena | stacker LR nad TFM ma najgorszą rangę log loss; TabICLv2 najniższe ECE | — | w B wagi minimalizują log loss (bez stackera LR) |
| Niezbalansowanie w TabPFN (arXiv 2605.21742) | — | przy rosnącym niezbalansowaniu predykcje przesuwają się w stronę klasy większościowej; pomagają progi i podpróbkowanie kontekstu | — | ESI 1 = 0,94%: kontekst `balanced` z korektą priorów, reguły decyzyjne zamiast oversamplingu |
| Destylacja TFM do GBDT (arXiv 2605.18654) | — | działa głównie dla tabel z < 21 cechami; przy wielu cechach gorzej | [S] | przy 271 cechach destylacja nie jest drogą do szybkiego wdrożenia |
| Przedziały ufności rankingu TabArena (substack „The Tabular Guy”) | — | przedziały pierwszej piątki się nakładają | [S] | rankingu nie traktuj jako rozstrzygającego |

### Dowody kliniczne i małe dane

| praca | dane | wynik | ozn. | znaczenie dla nas |
|---|---|---|---|---|
| Brima i Atemkeng, *Sci Rep* 2026 | MIMIC-IV-ED i eICU, 7 niezbalansowanych zadań, Macro-F1 | MIMIC-IV-ED: TabPFN v2.6 i TabICL z najlepszą średnią rangą, XGBoost tuż za nimi; eICU: najlepszy XGBoost | — | najbliższy analog SOR; nikt nie stosował TFM do triażu ESI/KTAS/MTS ani do danych Yale (luka = wkład pracy) |
| Benchmark kliniczny TabPFN, *BMC Med Inform Decis Mak* 2026 | 12 zadań binarnych, 788–139 528 pacjentów | TabPFN lepszy od najlepszego modelu ML tylko w 16,7% zadań; różnice AUROC zwykle ±0,02; obliczenia ok. 5,5× dłuższe | — | w klinice raczej parytet niż przewaga |
| Anemia u dzieci, 16 krajów (arXiv 2605.26589) | — | poniżej ok. 200 próbek na kraj TabPFN v2.6 lepiej rozróżniał klasy niż LR, XGBoost i LightGBM; na pełnych danych różnice ≤ 0,05 AUC | [S] | scenariusz A |
| MCI → AD (arXiv 2604.27195) | n = 530 | przy N = 50 TabPFN najlepszy (AUC 0,810); na pełnych danych zbliżony do strojonego XGBoost (0,879 vs 0,887) | — | scenariusz A |
| Ryzyko chirurgii ratunkowej, NSQIP (*Surgery* 2026; PubMed 41666517) | n = 7281 | TFM z najwyższym AUROC (powikłania 0,82, zgon 0,89); autorzy proponują TFM jako „portable standard” dla modeli lokalnych | — | scenariusz A (lokalny model) |
| Omdlenia na SOR (PMC12331830) | n = 257 | gradient boosting AUC 0,78; TabPFN istotnie gorszy (p < 0,01) | — | kontrprzykład: mało danych nie gwarantuje wygranej TFM |
| TL-ANDI (arXiv 2607.04809) | — | dane innego ośrodka w kontekście TFM mogą dać transfer negatywny przy przesunięciu rozkładu | — | warianty `+foreign` w eksperymencie A |
| TFM pod przesunięciem rozkładu (arXiv 2607.26000) | TableShift | wszystkie 9 badanych TFM traciło (luka 0,003–0,060) | — | scenariusz `site_local` |
| Kalibracja w zadaniach okołooperacyjnych (SAGE, doi:10.1177/20552076261443585) | — | kalibracja TabPFN wyraźnie spadała przy częstości wyniku < 20% | — | ESI 1 (0,9%) i ESI 5 (5%): kryterium ECE ESI 1 w bramce B |

### Licencje i narzędzia

| źródło | treść | ozn. |
|---|---|---|
| README PriorLabs/TabPFN (odczytane z repozytorium) | wagi TabPFN‑2.5/2.6/3/3.5 na licencjach niekomercyjnych, kod Apache-2.0; po pobraniu wag działa offline, dane nie są wysyłane do Prior Labs; wdrożenie komercyjne wymaga licencji Prior Labs | — |
| soda-inria/tabicl, plik LICENSE | kod BSD-3; licencji wag (checkpoint `jingang/TabICL` na HF) nie potwierdzono | [?] |
| Status przejęcia Prior Labs przez SAP | tech.eu: transakcja zamknięta; IT Brief: dopiero planowana | [?] |

### Wnioski do dyskusji (TFM)

1. **Na pełnych danych spodziewamy się małego efektu.** Rankingi, w których TFM wygrywają, mierzą głównie
   małe zbiory (TabArena), a na dużych prowadzą GBDT (BeyondArena [S]). Jedyny realny mechanizm zysku
   w B to dywersyfikacja błędów w blendzie (nasz wniosek). Wynik negatywny eksperymentu B jest zgodny
   z literaturą i warto go raportować.
2. **Przy małych danych przewaga TFM jest prawdopodobna, ale niepewna.** Część prac ją pokazuje
   (anemia [S], MCI → AD, NSQIP), część nie (omdlenia na SOR). Eksperyment A odpowiada na to pytanie dla
   danych SOR i podaje próg n\*.
3. **Dane obcego ośrodka w kontekście mogą szkodzić** (TL-ANDI, TableShift). Wynik `+foreign`
   w eksperymencie A sprawdza to dla danych Yale.
4. **Deklaracje producentów [P] cytuj jako deklaracje**, a wyniki [S] dopiero po sprawdzeniu pełnego tekstu.
