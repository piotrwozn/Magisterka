# Audyt danych Yale EMMLC → ESI

Wygenerowano automatycznie przez `scripts/02_preprocess.py` (wersja danych `esi-v2`, podział `random`). Nie edytuj ręcznie.

## Źródło i czyszczenie

- Surowe dane: 560,486 wizyt × 972 kolumn.
- Usunięto 2,457 wizyt bez ESI i 2 duplikaty.
- Po czyszczeniu: **558,027 wizyt**, wyłącznie dorośli (wiek 18–108).
- Brak roku wizyty i identyfikatora pacjenta → podział chronologiczny i podział po pacjentach nie są możliwe; ten sam pacjent może wystąpić w train i test.
- Etykieta = **najnowsze ESI wizyty**: Yale (`20_cleanmerged.R`) zastępuje ESI z triażu wartością z ponownego triażu (`second_esi`), jeśli był. Część etykiet może więc uwzględniać informacje zebrane po triażu (szum etykiety); `second_esi` nie jest udostępnione, więc nie da się tego cofnąć.

## Etykieta: rozkład ESI

| ESI | kolor TOPSOR | liczba | udział | odsetek przyjęć |
|---|---|---:|---:|---:|
| 1 | czerwony | 5,271 | 0.9% | 85.6% |
| 2 | pomarańczowy | 163,532 | 29.3% | 55.0% |
| 3 | żółty | 236,229 | 42.3% | 29.1% |
| 4 | zielony | 125,003 | 22.4% | 2.2% |
| 5 | niebieski | 27,992 | 5.0% | 0.4% |

ESI wg oddziału (udział w wierszu):

| oddział | n | ESI 1 | ESI 2 | ESI 3 | ESI 4 | ESI 5 |
|---|---:|---:|---:|---:|---:|---:|
| A | 320,031 | 1.4% | 37.9% | 38.7% | 16.4% | 5.6% |
| B | 166,335 | 0.4% | 23.1% | 45.7% | 26.8% | 4.0% |
| C | 71,661 | 0.0% | 5.2% | 50.9% | 38.9% | 5.0% |

## Parametry życiowe z triażu

- Temperatura w Yale jest w **°F** (mediana 98.0 °F) → przeliczona na °C (także temperatury z historii).
- Yale usunął wartości odstające jeszcze przed udostępnieniem danych (`9_edvitalsclean.R`: x < min lub x ≥ max → brak):

| parametr | zachowane przez Yale |
|---|---|
| HR /min | 30 ≤ x < 300 |
| SBP mmHg | 30 ≤ x < 400 |
| DBP mmHg | 25 ≤ x < 300 |
| SpO2 % | 60 ≤ x < 100 |
| RR /min | 8 ≤ x < 70 |
| temperatura °F | 90 ≤ x < 110 |

- **SpO₂ = 100% przywrócone.** Filtr Yale (x ≥ 100 → brak) skasował każdy odczyt 100%. Przywrócono SpO₂ = 100 w **104,990** wizytach, w których zmierzono HR (ten sam pulsoksymetr) i SBP, a SpO₂ brakowało. Historia `spo2_*` pozostaje ucięta na 99% (NaN, gdy wszystkie wcześniejsze odczyty wynosiły 100%) — tego nie da się odtworzyć.
- Nasze zakresy wiarygodności (poza zakresem → NaN, nie mediana). Filtr Yale jest zwykle ostrzejszy, więc usuwają niewiele:

| parametr | zakres | wyzerowane |
|---|---|---:|
| `triage_vital_hr` | 20–300 | 0 |
| `triage_vital_sbp` | 40–300 | 1 |
| `triage_vital_dbp` | 15–200 | 1 |
| `triage_vital_rr` | 4–80 | 0 |
| `triage_vital_o2` | 50–100 | 0 |
| `triage_vital_temp` | 30–44 | 0 |
| DBP ≥ SBP | — | 3 |

| parametr | braki | mediana | 1. percentyl | 99. percentyl |
|---|---:|---:|---:|---:|
| `triage_vital_hr` | 29.4% | 84.0 | 53.0 | 131.0 |
| `triage_vital_sbp` | 29.8% | 131.0 | 92.0 | 196.0 |
| `triage_vital_dbp` | 29.8% | 80.0 | 49.0 | 117.0 |
| `triage_vital_rr` | 30.2% | 18.0 | 14.0 | 24.0 |
| `triage_vital_o2` | 29.5% | 98.0 | 92.0 | 100.0 |
| `triage_vital_temp` | 32.5% | 36.7 | 35.7 | 38.2 |
| `triage_vital_o2_device` | 40.3% | 0.0 | 0.0 | 1.0 |

**Braki parametrów odzwierciedlają organizację pracy Yale, nie stan pacjenta.** Yale uznaje za pomiar z triażu tylko odczyty sprzed przyjęcia na salę (`RECORDED_TIME <= roomtime`, `9b_gettriagevitals.R`). Pacjent wprowadzony na salę przed pomiarem — typowo ESI 1, od razu na salę resuscytacyjną — nie ma żadnych parametrów. Brak jest więc skutkiem decyzji triażowej i zależy od oddziału. Braki zostają jako NaN plus flagi `*_missing` (nie mediana). Odsetek wizyt bez ŻADNEGO parametru życiowego:

| ESI | bez vitals |
|---|---:|
| 1 | 85.2% |
| 2 | 39.5% |
| 3 | 26.2% |
| 4 | 17.1% |
| 5 | 37.7% |

W TOPSOR pomiary są obowiązkowe, więc populacją docelową są wizyty z kompletem HR, SBP i RR (`vitals_complete = 1`, kolumna meta — NIE cecha). Metryki raportujemy osobno dla tej podgrupy; wyniki na całym zbiorze zawyżają wykrywanie ESI 1, które w Yale opiera się głównie na wzorcu braków.

| `vitals_complete` | n | ESI 1 | ESI 2 | ESI 3 | ESI 4 | ESI 5 | ESI 1 (n) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 385,922 | 0.2% | 24.8% | 44.3% | 26.4% | 4.4% | 598 |
| 0 | 172,105 | 2.7% | 39.5% | 38.0% | 13.4% | 6.4% | 4,673 |

## Przybycie i czas

| sposób przybycia (Yale) | liczba | kategoria kanoniczna |
|---|---:|---|
| Car | 222,638 | self |
| ambulance | 189,679 | ambulance |
| Walk-in | 113,564 | self |
| <brak> | 20,799 | self |
| Other | 6,853 | other |
| Public Transportation | 2,820 | self |
| Wheelchair | 1,575 | other |
| Police | 99 | other |

- Brak sposobu przybycia (20,799 wizyt) → `self`, tak jak w backendzie. W Yale to prawie wyłącznie oddział A: ścieżka typu fast-track dla studentów i pracowników, wyznaczona ubezpieczeniem i zatrudnieniem w USA (64% wizyt ESI 5 tego oddziału). Osobna kategoria `arrival_unknown` uczyłaby model tej ścieżki, a w TOPSOR nigdy nie występuje.
- Policja → `other` (kilkadziesiąt wizyt — osobna kolumna byłaby praktycznie stała).
- Miesiąc, dzień tygodnia i przedział godzin zakodowane jawnie (1–12, 1–7 od poniedziałku, 0–5 od 23:00).

## Skargi główne

- Flagi 0/1: Yale zlicza powtórzone wpisy skargi (wartości 2–3) — przycięto 920 wartości do 1 (backend wysyła 0/1).
- `cc_fulltrauma`, `cc_modifiedtrauma` → `cc_trauma_activation`: pełna i zmodyfikowana aktywacja zespołu urazowego to dwustopniowy protokół oddziału A Yale, nieodtwarzalny z polskich danych. `cc_strokealert` zostaje.
- Wizyty bez żadnej skargi głównej: 2,576 (skargi uzupełnione zerami).

## Wykluczone kolumny

- Specyficzne dla USA lub wrażliwe (niedostępne w polskim SOR, ryzyko uprzedzeń): `ethnicity`, `race`, `lang`, `religion`, `maritalstatus`, `employstatus`, `insurance_status`.
- `dep_name` (oddział) — tylko do stratyfikacji i podziału `--split site`, nie jako cecha.
- `disposition` (przyjęcie/wypis) — wynik wizyty, zapisany jako `admitted` tylko do analiz; nigdy nie jest cechą.
- eGFR liczony z rasą: `egfr_last`, `egfr(nonafricanamerican)_last`, `egfr(aframer)_last`, `egfr_min`, `egfr(nonafricanamerican)_min`, `egfr(aframer)_min`, `egfr_max`, `egfr(nonafricanamerican)_max`, `egfr(aframer)_max`, `egfr_median`, `egfr(nonafricanamerican)_median`, `egfr(aframer)_median`.
- Leki `meds_*` (liczba kolumn: 48): lista leków powstaje głównie przy przyjęciu do szpitala (rekoncyliacja lekowa), czyli po triażu — przeciek wyniku bieżącej wizyty.
- PMH socjoekonomiczne: `socialadmin` (kody V dotyczące mieszkania, sytuacji materialnej i zatrudnienia — wskaźnik ubezpieczenia).
- PMH administracyjne: `exameval`, `otherscreen`, `immunizscrn`, `otaftercare` (powód wizyty, nie choroba).
- PMH o małym wsparciu (28, stała lista): `ecodesmachinery`, `ecodesfireburn`, `septicemiaexceptinlabor`, `ecodesdrowningsubmersion`, `ecodesplaceofoccurrence`, `umbilcord`, `obrelatedperintrauma`, `birthasphyx` … (+20) — < 50 pozytywnych w train (kody okołoporodowe u dorosłych, kody E).
- Zepsute badania (24): `acetonebld_last`, `benzodiazepinesscreen,urine,noconf._last`, `epithelialcells_last`, `phencyclidine(pcp)screen,urine,noconf._last`, `pocph_last`, `acetonebld_min`, `benzodiazepinesscreen,urine,noconf._min`, `epithelialcells_min` … (+16) — puste, stałe, śmieciowe wartości lub mieszane skale.
- D-dimer (4): `d-dimer_last`, `d-dimer_min`, `d-dimer_max`, `d-dimer_median` — mieszane jednostki (µg/mL FEU obok ng/mL).
- Duplikaty cech inżynieryjnych: `esi_danger_hr`, `esi_danger_rr` (identyczne z `tachycardia` / `tachypnea`; liczone tylko wewnętrznie dla `esi_danger_count`).

## Historia pacjenta (zestaw `full`)

- `previousdispo` (dyspozycja POPRZEDNIEJ wizyty) → 5 kategorii: `prev_dispo_admitted`, `prev_dispo_discharged`, `prev_dispo_transferred`, `prev_dispo_left_early`, `prev_dispo_none`. Observation (status Medicare) i Send to L&D (ścieżka położnicza USA) → przyjęcie; AMA, ucieczka i LWBS → `left_early`.
- Badania jakościowe (`_last`, `_npos`): Yale wpisał 0 także, gdy badania nie wykonano; przy `_count == 0` → NaN (nie badano ≠ wynik ujemny). Wiersze: `bloodua` 392,687, `glucoseua` 392,595, `ketonesua` 392,609, `leukocytesua` 392,637, `nitriteua` 392,610, `proteinua` 392,682, `urineculture,routine` 464,356.
- Ujednolicone jednostki (komórki/µL → 10³/µL, wartość > próg / 1000): `basosabs_last` 989, `monosabs_last` 1,185, `basosabs_min` 1,261, `monosabs_min` 1,522, `basosabs_max` 1,447, `monosabs_max` 1,728, `basosabs_median` 1,372, `monosabs_median` 1,651.
- Badania są w jednostkach konwencjonalnych USA — przed użyciem modelu `full` z polskimi wynikami trzeba je przeliczyć:

| badanie | jednostka w Yale (przeliczenie) |
|---|---|
| glucose | mg/dL (mmol/L = mg/dL / 18,0) |
| creatinine | mg/dL (µmol/L = mg/dL × 88,4) |
| bun | mg/dL azotu mocznikowego (mocznik mg/dL = BUN × 2,14; mmol/L = BUN × 0,357) |
| calcium | mg/dL (mmol/L = mg/dL × 0,25) |
| magnesium | mg/dL (mmol/L = mg/dL × 0,411) |
| phosphorus | mg/dL (mmol/L = mg/dL × 0,323) |
| hemoglobin | g/dL (mmol/L = g/dL × 0,621) |
| troponin | ng/mL, POC TnI i TnT — nieporównywalne z polskim hs-cTn (ng/L); nie mapować |
| monosabs, basosabs | 10³/µL (po ujednoliceniu jednostek) |

## Zestawy cech

| grupa | liczba cech | w zestawie |
|---|---:|---|
| demographics | 2 | triage + full |
| arrival | 3 | triage + full |
| time | 3 | triage + full |
| triage_vitals | 7 | triage + full |
| chief_complaints | 199 | triage + full |
| past_medical | 248 | full |
| ed_usage | 8 | full |
| imaging_history | 9 | full |
| history_vitals | 28 | full |
| history_labs | 339 | full |
| engineered | 57 | triage + full |
| **triage** | **271** | dostępne w chwili triażu |
| **full** | **903** | triaż + historia pacjenta |

## Podział `random`

| zbiór | n | ESI 1 | ESI 2 | ESI 3 | ESI 4 | ESI 5 | oddziały |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 446,421 | 0.9% | 29.3% | 42.3% | 22.4% | 5.0% | A: 256,024, B: 133,068, C: 57,329 |
| val | 55,803 | 0.9% | 29.3% | 42.3% | 22.4% | 5.0% | A: 32,004, B: 16,633, C: 7,166 |
| test | 55,803 | 0.9% | 29.3% | 42.3% | 22.4% | 5.0% | A: 32,003, B: 16,634, C: 7,166 |

Oddziały w każdym zbiorze (liczba wizyt wg ESI; ESI 1 decyduje o tym, czy metryki ESI 1 da się w ogóle policzyć):

| zbiór | oddział | n | ESI 1 | ESI 2 | ESI 3 | ESI 4 | ESI 5 | `vitals_complete` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | A | 256,024 | 3,661 | 97,035 | 99,040 | 42,052 | 14,236 | 73.8% |
| train | B | 133,068 | 540 | 30,804 | 60,760 | 35,676 | 5,288 | 58.1% |
| train | C | 57,329 | 15 | 2,987 | 29,183 | 22,275 | 2,869 | 73.9% |
| val | A | 32,004 | 458 | 12,130 | 12,380 | 5,256 | 1,780 | 73.7% |
| val | B | 16,633 | 68 | 3,850 | 7,595 | 4,459 | 661 | 57.9% |
| val | C | 7,166 | 2 | 373 | 3,648 | 2,784 | 359 | 73.6% |
| test | A | 32,003 | 458 | 12,129 | 12,380 | 5,257 | 1,779 | 74.1% |
| test | B | 16,634 | 67 | 3,851 | 7,595 | 4,460 | 661 | 58.4% |
| test | C | 7,166 | 2 | 373 | 3,648 | 2,784 | 359 | 73.6% |

- `train` — trening modeli (10% z niego wydzielane w locie na early stopping).
- `val` — porównanie modeli i ręczne łączenie (wagi ensemble, progi decyzyjne).
- `test` — tylko ostateczna ocena; nie dostrajać na nim niczego.

