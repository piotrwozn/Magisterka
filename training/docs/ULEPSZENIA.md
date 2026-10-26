<!-- Wyniki eksperymentów z 2026-10-24: LightGBM na próbce 60 tys. (część na 150 tys.) wierszy TRAIN, ocena na połowach VAL; TEST nieużywany. To przybliżenie — po pełnym treningu zmierz ponownie. -->

# Menu usprawnień: triaż ESI (synteza czterech analiz)

## Skąd są liczby i jak je czytać

- **Punkt odniesienia.** Wszystkie liczby pochodzą z LightGBM z domyślnymi parametrami, trenowanego na 60 tys. wierszy. Część pomiarów powtórzono na 150 tys. Ocena była tylko na `val` (55 803 wiersze), a `test` nie był używany.
  - vc to `vitals_complete==1`, czyli 38 487 wierszy i populacja najbliższa TOPSOR.
  - Parametry dobierane na `val` (progi, wagi, kalibracja) dopasowano na połowie A, a wynik podano na połowie B.
- **Baseline:**

  | reguła | QWK | undertriage | log loss |
  |---|---|---|---|
  | argmax | 0,684 | 33,5% | 0,788 |
  | reguła kosztowa k=3 | 0,614 | 20,3% | 0,788 |

- **To przybliżenie, nie wynik końcowy.**
  - Zyski z cech i z łączenia modeli maleją z ilością danych: mniej więcej o połowę na każde podwojenie.
  - Zyski z reguły decyzyjnej były stabilne między 60 a 150 tys.
  - Po treningu domowym każdą wielkość trzeba zmierzyć ponownie na zapisanych predykcjach.
- **Szum (SD między seedami):** log loss 0,0003–0,001, QWK argmax ok. 0,0006, QWK reguły kosztowej 0,002–0,004. Reguła kosztowa wzmacnia szum. Różnicę mniejszą niż ok. 2× SD traktuj jako brak efektu.
- **ESI1 jest bardzo rzadkie:** 528 przypadków w `val`, 63 w vc, 28 w połowie B vc. Critical miss i recall ESI1 w vc to w dużej mierze szum, bo jeden pacjent przesuwa wynik o 1,6–3,6 pp.
- **Definicja undertriage w repo:** etykieta ESI1–2 przewidziana jako ESI3–5, mierzona względem oceny pielęgniarki, a nie względem wyniku klinicznego.

## W skrócie

1. **Przed treningiem domowym praktycznie nic nie zmieniaj.**
   - Największą dźwignią są pełne dane, a to już jest w planie.
     - Przejście z 60 na 150 tys. wierszy: log loss −0,02.
     - Ekstrapolacja do pełnych danych: log loss ok. −0,03 i QWK argmax ok. +0,015 względem 60 tys.
   - Żaden pomysł na nowe cechy, wagi klas ani porządkowy cel treningu nie przetrwał tuningu ani przejścia na 150 tys.
2. **Największy mierzalny zysk jest po treningu i nie wymaga ponownego trenowania.** To zmiana reguły decyzyjnej (B1). Przy tym samym undertriage daje QWK od +0,09 do +0,16 względem obecnej reguły kosztowej.
3. **Najważniejsze analizy do pracy:**
   - zmienność między seedami (różnice między modelami są tej samej wielkości co szum);
   - podgrupy i fairness;
   - walidacja z wyłączeniem szpitala (leave-site-out) z korektą rozkładu klas;
   - porównanie z pielęgniarką na przyjęciach do szpitala;
   - TRIPOD+AI.
4. **Sprawdź to od razu:** `08_final_report.py` ma domyślnie `--weights equal`. Równe wagi są wyraźnie gorsze od `--weights optimize` (B3).

---

## Grupa A: zrób przed długim treningiem w domu

| # | Co to jest | Zmierzony efekt (60k/150k) | Nakład | Ryzyko | Rekomendacja |
|---|---|---|---|---|---|
| A1 | Nie zmieniaj cech, celu treningu ani wag klas. Trenuj na pełnych danych z tuningiem, jak w planie | Z 60 na 150 tys.: log loss −0,020, QWK argmax +0,010. Krótki tuning (16 prób) na 60 tys.: log loss −0,012. To więcej niż którykolwiek pomysł z tej listy wymagający treningu | 0 h | brak | **Zrób.** Kontrola sensu: dostrojony GBM na pełnych danych z log loss na `val` > ok. 0,765 albo QWK argmax < ok. 0,695 wskazuje na błąd |
| A2 | Sprawdź, czy zapiszą się `predictions_val/test.parquet` (z `row_id`), `run_info.json` z najlepszymi parametrami i bazy Optuny | Brak efektu na metryki. To warunek dla grup B i C: wszystko liczy się z predykcji albo z refitów z najlepszymi parametrami, bez nowego tuningu | 15 min (repo robi to domyślnie) | Bez tego wiele analiz wymaga powtórki tuningu | **Zrób** |
| A3 | Pomiń random forest: `--models lr,lgbm,xgb,cb,ebm` | Waga RF w optymalnym blendzie to 0,00 w obu kierunkach. Usunięcie RF zmienia log loss o 0,0000. To najsłabszy model: log loss 0,849 wobec 0,780 dla XGBoost | 0 h; oszczędza ok. 1,5 h tuningu na torze CPU | Sprawdzone tylko bez tuningu na 60 tys. | **Warto**, chyba że chcesz RF w pracy jako punkt odniesienia. Wtedy nie dawaj mu wagi w blendzie |
| A4 | Refit na 100% TRAIN po early stopping, z liczbą iteracji × n_all/n_fit | Log loss −0,0026/−0,0022, QWK reguły kosztowej +0,0055 (CI > 0), czyli tyle co +11% danych. Na pełnych danych spodziewany efekt to tylko −0,001…−0,0015 | 2–3 h w `src/models/train.py` + 1 dodatkowy fit na model | Błąd w nazwie parametru iteracji (XGBoost: `n_estimators`, CatBoost: `iterations`, wyłączone early stopping) | **Marginalne.** Tylko jeśli masz czas przed startem |
| A5 | Opcjonalny drugi model do porównania w pracy: LightGBM na `triage` + historia wizyt na SOR + obrazowanie + choroby przewlekłe (536 cech, **nie** `full`) | Log loss −0,018, QWK argmax +0,015, QWK reguły kosztowej +0,020, undertriage (reguła kosztowa) −1,3 pp. Na 150 tys. log loss −0,020. Samo 8 cech historii wizyt daje ok. 65% tego zysku. **Zysk nie przenosi się między szpitalami:** po treningu na A+B i teście na C log loss się nie poprawia (−0,005…0), QWK argmax −0,004, expected cost +0,009…+0,014 | Ok. 0,5 dnia (nowy zestaw w `prepare.py`/`dataset.py`) + preprocessing + 1 trening | W Polsce historia jest dostępna tylko częściowo (zwykle z tego samego szpitala). `full` (903 cechy) jest gorszy o 0,005–0,006 log loss i trenuje 5–6× wolniej | **Tylko jako analiza do pracy**, jeśli zostanie czas maszyny. Nie do modelu w aplikacji TOPSOR |

---

## Grupa B: po treningu, bez ponownego trenowania

| # | Co to jest | Zmierzony efekt (połowa B, średnio z 12 powtórzeń) | Nakład | Ryzyko | Rekomendacja |
|---|---|---|---|---|---|
| B1 | **Reguła „undertriage ≤ T”.** Oczekiwany poziom EL = Σk·p_k dzielony 4 progami. Próg ESI2/ESI3 wynika z celu T, pozostałe 3 progi maksymalizują QWK. Wszystko dobierane na `val` **w podgrupie vc** | Szczegóły w tabeli pod spodem. Przy T=20% B1 wygrywa z argmax w obu miarach: QWK 0,704 wobec 0,686 i undertriage 20% wobec 33%. Bez celu T (progi tylko pod QWK) daje QWK +0,024 ± 0,001 względem argmax | 4–6 h minimum; 8–12 h z testami, wykresem i backendem | Patrz lista ryzyk pod tabelą | **Główna rekomendacja** |
| B2 | **Siatka bezpieczeństwa:** jeśli P(ESI1)+P(ESI2) ≥ q (q ≈ 0,12), pacjent dostaje najwyżej ESI3 | Critical miss spada z 2,6% do ok. 0,4% (90 → 11 zdarzeń w 12 powtórzeniach). QWK −0,007 (vc −0,003). Undertriage bez zmian. Zmienia się 2,6% decyzji | +1 h | q dobrane na ok. 264 pacjentach ESI1 (w vc tylko 28) | **Zrób razem z B1** |
| B3 | **Blend z optymalnymi wagami:** `08_final_report.py --weights optimize` zamiast domyślnego `equal`. Regresja logistyczna zostaje w składzie | Względem najlepszego pojedynczego modelu (XGBoost): log loss −0,0072, QWK reguły kosztowej +0,007. Na 150 tys. (lgb + xgb + logreg): log loss −0,0033. Równe wagi względem optymalnych: log loss +0,006, QWK reguły kosztowej −0,019, czyli gorzej niż sam LightGBM. Temperatura (już w repo) daje niewiele: log loss −0,0005 | 0 h | Wagi dobierane na `val`. Waga logreg spadnie przy pełnych danych (0,31 → ok. 0,2) | **Zrób** |
| B4 | **Tańszy wariant B1 (decyzja Bayesa):** macierz kosztów (p−t)² + λ·1[ESI1–2 → ESI3–5], λ dobrane do celu T, przez istniejące `decide(proba, "cost", costs=…)` | QWK 0,644 / 0,671 / 0,683 przy T = 10 / 15 / 20%. To o 0,02 mniej niż B1, ale o 0,07–0,14 więcej niż obecna reguła. Critical miss 1,0%. Expected cost 0,73 (przy B1 0,92) | ok. 1 h | Prawie nie przewiduje skrajnych klas: 0,2% ESI1 i 0,9% ESI5 | **Alternatywa dla B1.** Łatwiej ją obronić jako decyzję optymalną przy jawnie zapisanej stracie klinicznej. Wybierz B1 albo B4, nie oba |
| B5 | **Matrix scaling blendu:** wieloklasowa rekalibracja, regresja logistyczna na 5 log-prawdopodobieństwach (30 parametrów) dopasowana na `val` | Względem blendu: log loss −0,0022, QWK reguły kosztowej +0,011, undertriage (reguła kosztowa) −0,8 pp. Na 150 tys.: undertriage −0,9 pp, QWK reguły kosztowej +0,0075 | 4–6 h (`src/ensemble`, `08`, backend, testy) | **Mierzone tylko z regułą kosztową.** Z B1, gdzie undertriage jest ustalony z góry, efekt nie jest znany. Wyniki na `val` wymagają cross-fittingu. Psuje gwarancję monotoniczności (C12) | **Przetestuj po B1** tym samym protokołem A/B. Wdrażaj tylko wtedy, gdy coś dodaje |
| B6 | *(nietestowane)* Gwarancja zamiast średniej: próg B1 wybrany tak, żeby górna granica Clopper-Pearsona dla undertriage na `val` była ≤ T | Szacunkowy koszt: 1–1,5 pp undertriage. Daje w pracy zdanie „undertriage ≤ T z prawdopodobieństwem 95%” | ok. 1 h | Nie zmierzone | Dobre uzupełnienie B1 |

**B1: QWK przy zadanym celu undertriage (połowa B)**

| cel T | B1 | obecna reguła kosztowa | różnica |
|---|---|---|---|
| 10% | 0,662 | 0,506 | +0,156 |
| 15% | 0,693 | 0,570 | +0,123 |
| 20% | 0,704 | 0,613 | +0,091 |

- Dla T=15% sparowany 95% CI różnicy to [+0,125; +0,134].
- W podgrupie vc, przy doborze progów na A∩vc: QWK 0,645 / 0,683 / 0,700, a reguła kosztowa 0,447 / 0,514 / 0,571.

**B1: ryzyka**

- **Accuracy spada:** z 0,666 do 0,595–0,622.
- **Expected cost liczony macierzą z repo rośnie:** 0,92 wobec 0,67. W pracy trzeba jawnie przyjąć „QWK przy zadanym undertriage” jako metrykę główną.
- **Critical miss się nie poprawia** (2,6%), stąd B2.
- **Cel jest trafiany średnio, z rozrzutem ±0,7–1 pp.** Ustaw T ok. 1,5 pp niżej albo użyj B6.
- **Dobór progów na całym `val` przestrzeliwuje w vc o 2–4 pp.** Dobieraj na vc.
- **Progi zależą od rozkładu ESI.** W Polsce trzeba je dobrać na nowo.

**Dlaczego obecna reguła kosztowa traci.** Macierz (p−t)²×3 karze każde niedotriażowanie, także ESI3 → ESI5. Reguła prawie nigdy nie przewiduje ESI5 (0,0% przy 5% prawdziwych), więc zbija przewidywania do ESI2–3. Na wykresie kompromisu krzywe B1 leżą o 0,05–0,20 QWK wyżej przy każdym poziomie undertriage.

---

## Grupa C: do rozdziału pracy

Pozycje C1–C5 to minimum. Pozostałe są uporządkowane od najbardziej przydatnych.

| # | Co to jest | Zmierzony efekt (60k, `val`) | Nakład | Ryzyko | Rekomendacja |
|---|---|---|---|---|---|
| C1 | **Zmienność między seedami i uczciwe porównywanie modeli.** Refit każdego finalnego modelu na 3–5 seedach z najlepszymi parametrami; wynik jako średnia ± SD | Dwa runy różniące się tylko seedem wychodzą „istotnie różne”: QWK reguły kosztowej Δ 0,0048, CI [0,0029; 0,0068], p = 0,001; McNemar p = 0,013; DeLong (vc) p = 0,029. SD QWK reguły kosztowej (0,0033) jest 10× większe niż przy argmax | 3–5 h kodu (`09_seed_refits.py`) + 5× czas finalnego fitu na model (GBM minuty, EBM ok. 1 h) | Niskie | **Konieczne.** Wyższość ogłaszaj tylko wtedy, gdy CI wyklucza 0 **i** |Δ| przekracza ustaloną z góry minimalną istotną różnicę (np. QWK 0,01, undertriage 1 pp, AUC 0,005) **i** 2× SD seedów. Cel to pomiar szumu; uśrednianie seedów dla dokładności nie jest warte zachodu (D) |
| C2 | **Podgrupy i fairness:** płeć, wiek, sposób przybycia, szpital, vc. Do tego undertriage „zakotwiczone w przyjęciu”: odsetek przyjętych do szpitala, którzy dostali ESI3–5, liczony też dla pielęgniarki | Undertriage (reguła kosztowa): kobiety 24,2%, mężczyźni 16,7%; 18–39 lat 32,3%, 80+ 10,0%; przyjście samodzielne 38,5%, karetka 8,4%. Spośród przyjętych w wieku 18–39 lat model daje ESI3–5 w 58% (reguła kosztowa) / 68% (argmax), pielęgniarka w 54%. U pacjentów 80+ duży overtriage: 21,6% pacjentów ESI4–5 trafia do ESI1–2 | 3–4 h (join po `row_id`), bez treningu | Część różnic wynika z innego profilu pacjentów. Zawsze pokazuj obok kolumnę przyjęć i kalibrację. Policz to dla reguły końcowej (B1+B2), nie tylko dla k=3 | **Konieczne** |
| C3 | **Walidacja leave-site-out i korekta rozkładu klas.** Trening na wierszach TRAIN z dwóch szpitali, ocena na wszystkich wierszach (train + val) trzeciego szpitala. Nie wymaga ponownego preprocessingu, w przeciwieństwie do `--split site` | Wyłączony szpital A: QWK argmax 0,695 → 0,576, undertriage argmax 32% → 59%, log loss +0,28. AUC przyjęć bez zmian (0,751 → 0,749): ranking się przenosi, rozkład klas nie. Znany lokalny rozkład klas zamyka ok. 85–90% luki w undertriage i ok. połowę luki w QWK. Estymacja tego rozkładu metodą EM bez etykiet zamyka ok. 40% luki w undertriage. Ocena na train + val daje 608 przypadków ESI1 zamiast 68, a szerokość CI dla recall ESI1 spada z 0,25 do 0,08 | 4–6 h + 1 refit na model dla każdego wyłączonego szpitala (A i B; C ma tylko 17 przypadków ESI1) | Szpitale jednego amerykańskiego systemu to słabe przybliżenie przejścia USA → Polska. Trzeba to napisać wprost | **Konieczne.** Jest też argumentem, że warstwę decyzyjną trzeba skalibrować na polskim rozkładzie |
| C4 | **Model a pielęgniarka na przyjęciach do szpitala** (miara niezależna od etykiety ESI). Repo ma to już w `_admission_validity` i w sekcji `08`; brakuje podgrup i tabeli niezgodności | AUC przyjęć dla −E[ESI]: 0,779 wobec 0,769 dla ESI pielęgniarki, Δ +0,010 [0,007; 0,014]; w vc +0,010; bez parametrów życiowych −0,002 (n.s.). Twarde poziomy ESI z modelu są gorsze od pielęgniarki (−0,03). Pacjenci ESI3 podniesieni przez model do ESI ≤ 2 są przyjmowani w 45,6% przypadków; ci, których model zostawia w ESI3, w 23,0% | 1–2 h | Przyjęcie częściowo wynika z triażu, co faworyzuje pielęgniarkę. Przyjęcie to nie to samo co pilność. Brak danych o OIOM i zgonach | **Konieczne** (w większości gotowe) |
| C5 | **Checklista TRIPOD+AI i samoocena PROBAST+AI** | 9 pozycji spełnionych, 15 częściowo, 5 brakuje: m.in. nierówności zdrowotne (3c), zaślepienie oceny wyniku (8b/c), uzasadnienie liczebności (10) | ok. 4 h pisania | Treść pozycji była parafrazowana. Sprawdź z oficjalnym PDF (Collins i in., BMJ 2024) | **Konieczne** |
| C6 | **Wykres kompromisu undertriage–QWK:** przegląd k reguły kosztowej i krzywa B1 na jednym wykresie | k = 1 / 2 / 3 / 5 / 10: undertriage 39,5 / 26,5 / 20,3 / 13,9 / 7,6%, QWK 0,678 / 0,650 / 0,614 / 0,555 / 0,464. Każde zmniejszenie undertriage o połowę kosztuje 0,06–0,09 QWK. Krzywe B1 leżą o 0,05–0,20 wyżej | ok. 1 h (wykres jest gotowy w analizie decyzyjnej) | — | **Warto.** Uzasadnia wybór k albo T |
| C7 | **Decision curve analysis** | Wynik ESI1–2, pt = 0,25: net benefit 0,198 wobec 0,070 dla „wszyscy pilni”. Przyjęcia, model wobec reguły „ESI ≤ 2” pielęgniarki: +0,018 [0,016; 0,021] przy pt = 0,25 (vc +0,007), ale −0,007 przy pt = 0,40 | 2–3 h | Analiza na etykiecie ESI jest częściowo okrężna. Jako główną pokaż wersję na przyjęciach | **Warto** |
| C8 | **Kalibracja per klasa, szpital i podgrupa** | ECE 0,010 (małe, ale powyżej szumu, p = 0,01). ESI1 jest niedoszacowane: O/E 1,22, w vc 1,71. Wyraz wolny kalibracji: szpital A +0,43, B −0,34, C −1,80 | ok. 2 h | Temperatura tego nie naprawi; potrzebna jest korekta rozkładu z C3 | **Warto** |
| C9 | **Krzywa uczenia** (uzasadnienie liczebności, TRIPOD pkt 10) | Od 15 do 150 tys. wierszy: QWK argmax 0,640 → 0,694, undertriage (reguła kosztowa) 23,3% → 19,0%, log loss 0,856 → 0,769. Ekstrapolacja do pełnych danych: log loss ok. 0,754–0,76, QWK argmax ok. 0,70 (bez tuningu; dwie niezależne ekstrapolacje są zgodne) | 2–3 h + ≤ 1 h obliczeń w domu | Ekstrapolacja ok. 3× poza zakres danych | **Warto** |
| C10 | **Analiza błędów** | 97% przypadków undertriage przy regule kosztowej to ESI2 → ESI3. Dolegliwości brzuszne to 22% pomyłek, a pomyłki dotyczą 52% takich pacjentów ESI1–2. Ból pleców: 78%, wypadek komunikacyjny: 63%, ból w klatce piersiowej: tylko 7%. Niedotriażowani pacjenci ESI2 są przyjmowani rzadziej niż pozostali ESI2 (43,7% wobec 58,3%), więc część „pomyłek” to szum etykiet | 2–3 h | — | **Warto.** Daje konkretne, klinicznie czytelne przykłady |
| C11 | **Wyjaśnienia globalne:** zależności SHAP (SHAP jest już w `05_explain`) i krzywe kształtu EBM | Większość zależności jest wiarygodna: tętno i ciśnienie skurczowe w kształcie U, SpO₂ monotoniczne, wiek. Do omówienia: gorączka obniża pilność (zgodne z ESI v4), spadek przy RR = 20, RR 12–16 podnosi ESI1, krzywe EBM są postrzępione (17–19 zmian znaku; trzeba je wygładzić albo pokazać ±SD). Sposób przybycia i wiek to cechy nr 1 i 2, co tłumaczy wzorzec z C2. EBM przy tych samych danych nie jest gorszy od LightGBM: QWK reguły kosztowej +0,008 | 2–3 h | — | **Warto** |
| C12 | **„Cena wiarygodności”:** porządkowy LightGBM z ograniczeniami monotoniczności (45 cech, np. niższe SpO₂, wyższy wiek, wyższy NEWS2 nie może obniżyć pilności) | Log loss +0,003…0,004 (gorzej), QWK argmax +0,003, undertriage bez zmian (±0,003). Model bez ograniczeń u ok. 1 na 20 pacjentów (4,6–4,7%) obniża pilność, gdy SpO₂ spada; model z ograniczeniami u 0% | 2–3 h + 1 trening na pełnym TRAIN (skrypt `exp4_monotone.py` gotowy; ok. 3× wolniej niż zwykły fit) | Zmieszanie w blendzie z modelami bez ograniczeń albo B5 unieważnia gwarancję. Monotoniczność wieku trzeba uzasadnić | **Warto jako analiza do pracy.** Jako 7. model w aplikacji: 1–1,5 dnia, nie teraz |
| C13 | **Conformal warunkowy po ESI i odsyłanie niepewnych do pielęgniarki** (jako analiza) | Zwykły LAC 90%: pokrycie 0,900, ale ESI1 tylko 0,42 (vc 0,17). Warunkowy po ESI: ESI1 0,90 (vc 0,70) przy średnim zbiorze 2,04 poziomu. B1 (T = 15%) z odesłaniem ok. 21% pacjentów w vc: undertriage wśród pacjentów, o których decyduje model, spada z 14,3% do 8,5%; niedotriażowanych i nieodesłanych jest 19 zamiast 36 na 1000 (argmax: 93,5) | 2–3 h każda część | Accuracy na pacjentach, o których decyduje model, nie jest accuracy całego systemu. Nie da się zasymulować błędów pielęgniarki, bo etykiety pochodzą od niej | **Warto jako analiza**; w aplikacji opcjonalnie |
| C14 | **Tabela porównania z literaturą** | QWK model–pielęgniarka 0,684 (prognoza ok. 0,70) wobec zgodności między pielęgniarkami ok. 0,79 (Mirhaghi 2015). AUC przyjęć dla modelu trenowanego na przyjęciach: 0,846 wobec 0,87 u Hong 2018 (te same dane Yale; lista zmiennych Hong nie została sprawdzona) | ok. 2 h pisania | Nie porównuj odsetka undertriage między pracami bez zgodnej definicji (Sax 2023 definiuje go przez wynik kliniczny) | **Warto** |

---

## Grupa D: sprawdzone, nie warto

| Pomysł | Wynik, który to pokazuje |
|---|---|
| Dodatkowe cechy kliniczne (shock index × wiek, MEWS, odchylenia parametrów od norm dla wieku, ból, liczba nieprawidłowych parametrów) | Razem: log loss −0,0008, QWK argmax −0,0006 (3 seedy), czyli szum. Przy zmianie szpitala wersja bezstanowa jest gorsza (+0,002) |
| Target encoding dolegliwości | 60 tys., bez tuningu: log loss −0,0065. Po tuningu −0,0019, na 150 tys. −0,0020. Głównie zastępuje strojenie `min_child_samples`. Przy zmianie szpitala pogarsza vc. Koszt 1–1,5 dnia i ryzyko wycieku etykiety. Najwyżej po treningu |
| Klastry dolegliwości (NMF/SVD) | Log loss −0,0018 (seedy: +0,0005 / −0,0022 / −0,0036), czyli szum. Jedyny dodatni sygnał to zmiana szpitala bez tuningu. Marginalne |
| Prawdopodobieństwo przyjęcia jako cecha | Na 150 tys. log loss +0,0004 |
| Trening tylko na vc | Dla całej populacji recall ESI1 (reguła kosztowa) spada z 0,31 do ok. 0,005, log loss +0,10. Zysk w vc (−0,005) bierze się tylko z większej liczby wierszy vc w próbce. Przy tej samej liczbie wierszy vc usunięcie pozostałych pogarsza vc (+0,0096 na 60 tys., +0,0043 na 150 tys.) |
| Większa waga wierszy vc (×2, ×3) | Log loss w vc +0,0014/+0,0018, w całej populacji +0,002…+0,003 (gorzej) |
| Usunięcie flag braków `*_missing` | ±0,0005 na każdej metryce. Zostaw flagi, bo logreg ich potrzebuje |
| Przycinanie cech | Dolegliwości z < 500 przypadkami: neutralne na 60 tys., szkodliwe na 150 tys. (+0,003). Próg < 1000: +0,014. Tylko 100 najlepszych cech: +0,015 |
| Zestaw `full` (903 cechy) | Gorszy od przyciętej historii z A5 o 0,005–0,006 log loss i 5–6× wolniejszy |
| Wagi klas w treningu (balanced/sqrt) | Log loss +0,052/+0,015, co psuje kalibrację, blend i temperaturę. Przy regule kosztowej najniższy expected cost ma model bez wag (0,668 wobec 0,705). Zysk QWK przy stałym undertriage daje też korekta po treningu |
| Korekta rozkładu klas (α) jako reguła decyzyjna | Działa względem reguły kosztowej (QWK przy undertriage 20%: 0,670 wobec 0,618), ale B1 na tej samej połowie B osiąga 0,704 (porównanie niesparowane). Nie rób obu. Przydatna natomiast przy przenoszeniu do innego szpitala (C3) |
| Uśrednianie seedów albo CV-bagging dla dokładności | Pojedynczy LightGBM: log loss −0,0023. W blendzie tylko −0,0005/−0,0001, przy kosztach ×3–5 obliczeń i ok. 1 dnia kodu |
| Pełny stacking (LR na 30 prawdopodobieństwach) ponad B5 | Log loss −0,001 [−0,0023; +0,0003] (n.s.), undertriage n.s. Koszt 6–8 h plus backend |
| Porządkowe cele treningu (regresja, 4 modele P(y > k), CORAL) | Regresja z progami: QWK +0,0225, czyli tyle co B1 bez treningu (+0,0238). Modele kumulatywne: +0,0015 ponad B1 (szum), log loss gorszy (0,793 wobec 0,786), 1,7% wierszy łamie monotoniczność |
| Wagi klas w argmax, dobrane pod QWK | QWK +0,004; B1 jest lepsze |
| Temperatura dla pojedynczego LightGBM | T = 1,02, log loss bez zmian; model jest już skalibrowany. Na blendzie temperaturę zostaw, jest już w repo |
| Conformal jako reguła decyzyjna („najpilniejszy poziom w zbiorze 90%”) | Przy undertriage ok. 10%: QWK 0,550 wobec 0,662 dla B1 |
| Siatka bezpieczeństwa oparta tylko na P(ESI1) | Pominięci pacjenci ESI1 mają P(ESI1) < 0,05, więc ta wersja nic nie łapie. Użyj P(ESI1)+P(ESI2), jak w B2 |
| Ograniczenia monotoniczności per klasa w trybie wieloklasowym | Naruszenia spadają tylko o połowę do dwóch trzecich. To nie jest gwarancja; nie nazywaj tego „monotonicznym” |
| Mocniej regularyzowany LightGBM | Log loss +0,004 na 60 tys., bez różnicy na 150 tys. Zostaw to Optunie |
| DeLong i McNemar obok bootstrapu | Zgodne z bootstrapem, nic nie dodają i tak samo „wykrywają” różnicę między seedami. Wystarczy jedno zdanie w metodach |

---

## Grupa E: modele fundamentalne (TFM), eksperymenty A i B

Wstępnie wytrenowane modele tabelaryczne TabPFN-3.5 i TabICLv2 obok LightGBM. Robi się to **po** treningu
klasycznym. Instalacja, polecenia, tabela decyzji i ograniczenia: [`TFM.md`](TFM.md). Liczb z naszych
danych jeszcze nie ma: wymagają prawdziwych wag i GPU, a czasy poniżej to szacunki.

| # | Co to jest | Zmierzony efekt | Nakład | Ryzyko | Rekomendacja |
|---|---|---|---|---|---|
| E1 | **Krzywa małych danych (eksperyment A)**, `14_small_data_curve.py`. TFM „z pudełka” kontra LightGBM strojony na tej samej podpróbie, n = 500–50 000, wszystkie oddziały i sam oddział B. Do tego TFM z danymi innych oddziałów w kontekście oraz LightGBM przeniesiony z korektą priorów | Nie zmierzony. Literatura: przewaga TFM zwykle przy setkach–tysiącach wierszy, ale są kontrprzykłady (np. omdlenia na SOR, n = 257) | Kod gotowy. Ok. 1 noc: części TFM ok. 1,5–3 h GPU, LightGBM ok. 5 h CPU | Mała moc przy dużym n (3 powtórzenia). Oddział B ma w VAL 68 wizyt ESI 1. Dane z USA | **Warto do pracy.** Odpowiada na pytanie o polski szpital z małą liczbą etykiet |
| E2 | **Blend z LightGBM na pełnych danych (eksperyment B)**, `15_tfm_blend.py`. Cross-fitting na VAL i bramka ustalona z góry: Δ log loss ≤ −0,005 z CI < 0, \|Δ\| > 2 × SD seedów, niedotriażowanie ≤ +1 pp, ECE ESI 1 ≤ +0,01 | Nie zmierzony. Oczekiwany efekt (literatura, nasz wniosek): rzędu MID albo mniejszy | Kod gotowy. Trening TFM ok. 1,5–3 h GPU (`thorough`, szacunek), analiza kilka minut | Wynik negatywny jest realny. Licencja TabPFN-3.5 jest niekomercyjna | **Warto do pracy.** Do systemu końcowego tylko po przejściu bramki (najwyżej 2 TFM) |
| E3 | **Serwowanie TFM w backendzie** (`SOR_AI_BACKEND_TFM=1`) | — | Konfiguracja; wymaga CUDA i wolnej VRAM | Karta 16 GB dzielona z Ollamą; duże konfiguracje się nie zmieszczą. Domyślnie backend używa receptury bez TFM | **Nie domyślnie.** Tylko gdy E2 przeszło i starcza VRAM |

---

## Zależności między pozycjami

- **B1, B4, korekta α i wagi klas rozwiązują ten sam problem:** reguła kosztowa ściska przewidywania do ESI2–3. Wybierz jedno rozwiązanie: B1 albo B4.
  - **B1**, jeśli metryką główną ma być QWK przy zadanym undertriage.
  - **B4**, jeśli chcesz bronić decyzji jako optymalnej przy jawnie zapisanej stracie.
- **B5 mierzono tylko z regułą kosztową.** Progi B1 dopasuj po kalibracji, na tych samych prawdopodobieństwach, które trafią do backendu.
- **Gwarancja monotoniczności z C12** znika po zmieszaniu z modelami bez ograniczeń i po B5.
- **Wszystkie parametry warstwy decyzyjnej zależą od rozkładu ESI:** progi B1, q z B2 i rozkład klas z C3. W Polsce trzeba je dobrać na nowo.
- **`val` pełni kilka ról:** dobór wag, temperatury i progów. `test` liczysz raz, na końcu, z tymi samymi parametrami.
- **C2 i C10 policz dla reguły końcowej (B1+B2),** nie tylko dla k=3.

## Proponowana kolejność

1. **Przed treningiem:** A1–A3 (≤ 0,5 h). Opcjonalnie A4 i A5.
2. **Po treningu, ok. 1,5–2 dni:** B3 → B1 + B2 (albo B4) → wykres C6 → test B5.
3. **Do pracy, ok. 2–3 dni pisania i kodu:** najpierw C1–C5, potem C7–C11, na końcu C12–C14.
   - Dodatkowe obliczenia w domu to refity bez tuningu: seedy, wyłączone szpitale, krzywa uczenia, model monotoniczny.
4. **Opcjonalnie, po treningu klasycznym:** E1 i E2 według [`TFM.md`](TFM.md). Raport końcowy 08 dla
   receptury z TFM uruchamiasz tylko wtedy, gdy E2 przeszło bramkę.
