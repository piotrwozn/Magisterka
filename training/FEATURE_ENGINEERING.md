# Cechy kliniczne — SOR-AI (ESI)

Implementacja: [`src/features/engineering.py`](src/features/engineering.py) (`engineer_features`,
`ENGINEERED_FEATURES` — 57 cech). Ta sama funkcja działa w treningu (`src/data/prepare.py`)
i w backendzie (pojedynczy pacjent). Etykieta to natywne ESI 1–5; kolory TOPSOR służą
wyłącznie do prezentacji.

## Zasady

- **Jednostki SI, temperatura w °C.** Yale podaje °F; przeliczenie dzieje się w `prepare`, zanim
  powstanie jakakolwiek cecha.
- **Brak pomiaru ≠ norma.** Flaga liczona z brakującego parametru jest NaN, a o braku mówią
  osobne cechy `*_missing`. Modele drzewiaste i EBM obsługują NaN natywnie.
- **Tylko cechy z uzasadnieniem klinicznym.** Przekształcenia monotoniczne (x², log x)
  i przypadkowe iloczyny parametrów nic nie wnoszą modelom drzewiastym, więc zostały usunięte.
- **Bez duplikatów.** `esi_danger_hr` i `esi_danger_rr` były identyczne z `tachycardia`
  i `tachypnea` (dzieliły ważność SHAP/EBM między dwie nazwy), dlatego liczą się tylko
  wewnętrznie, do `esi_danger_count`.
- **Dorośli.** Yale zawiera wyłącznie pacjentów ≥ 18 lat, dlatego nie ma cech pediatrycznych.

## Wejście (schemat kanoniczny)

| kolumna | opis |
|---|---|
| `triage_vital_hr`, `_sbp`, `_dbp`, `_rr`, `_o2`, `_temp` | HR /min, ciśnienie mmHg, RR /min, SpO₂ %, temperatura °C |
| `triage_vital_o2_device` | tlenoterapia w chwili pomiaru (0/1) |
| `age`, `is_male` | wiek w latach, płeć |
| `arrival_ambulance`, `_self`, `_other` | sposób przybycia (one-hot); brak → `self`, policja → `other` |
| `arrival_month`, `arrival_weekday`, `arrival_hour_bin` | 1–12, 1–7 (pon.=1), 0–5 (0 = 23:00–02:59) |
| `cc_*` | 199 skarg głównych z Yale/Epic jako flagi 0/1 (nazwy znormalizowane) |

Uwagi do wejścia:

- **SpO₂ = 100%.** Filtr Yale usunął każdy odczyt 100% (`x ≥ 100 → brak`). `prepare` przywraca
  100, gdy w triażu zmierzono HR i SBP, więc wartość 100 z formularza jest w rozkładzie
  treningowym i backend przekazuje ją bez zmian.
- **Aktywacja zespołu urazowego.** Yale ma dwa kody protokołu oddziału A (`cc_fulltrauma`,
  `cc_modifiedtrauma`); scalono je w `cc_trauma_activation`. Backend mapuje stare nazwy
  na nową kolumnę.
- **Skargi jako flagi.** Yale zlicza powtórzone wpisy skargi (wartości 2–3); `prepare` przycina
  je do 1, tak jak wysyła backend.

## Cechy

| grupa | cechy | definicja |
|---|---|---|
| Hemodynamika | `pulse_pressure`, `map`, `shock_index`, `modified_shock_index` | SBP−DBP; DBP + PP/3; HR/SBP; HR/MAP |
| | `tachycardia`, `severe_tachycardia`, `bradycardia` | HR > 100, > 130, < 50 |
| | `hypotension`, `severe_hypotension` | SBP < 100, < 90 |
| | `severe_hypertension`, `hypertensive_crisis` | SBP ≥ 180; SBP ≥ 200 lub DBP ≥ 120 |
| | `shock_index_critical` | SI ≥ 1,0 |
| Oddech | `tachypnea`, `severe_tachypnea`, `bradypnea` | RR > 20, ≥ 30, < 10 |
| | `hypoxia`, `severe_hypoxia` | SpO₂ < 94, < 90 |
| Temperatura | `fever`, `high_fever`, `hypothermia`, `temp_deviation` | ≥ 38,0 °C; ≥ 39,0 °C; < 35,0 °C (hipotermia kliniczna); \|t − 37\| |
| Strefa zagrożenia ESI | `esi_danger_spo2`, `esi_danger_count`, `esi_danger_zone` | punkt decyzyjny D (dorośli): HR > 100, RR > 20, SpO₂ < 92; liczba spełnionych kryteriów (HR i RR = `tachycardia`, `tachypnea`); ≥ 1 |
| Skale | `sirs_score`, `sirs_positive` | temp > 38 lub < 36, HR > 90, RR > 20 (bez leukocytów); ≥ 2 |
| | `qsofa_score`, `qsofa_positive` | RR ≥ 22, SBP ≤ 100, zaburzenia świadomości (proxy ze skarg); ≥ 2. NaN, gdy brak RR i SBP |
| | `news2_score`, `news2_high` | NEWS2 bez oceny świadomości (skala SpO₂ 1); ≥ 7 |
| Wiek | `age_65plus`, `age_80plus` | |
| Braki | `hr_missing` … `temp_missing`, `n_vitals_missing`, `all_vitals_missing` | brak pomiaru w triażu |
| Czas | `night_arrival`, `weekend_arrival` | przedział 23:00–06:59; sobota/niedziela |
| Skargi | `cc_count`, `cc_critical` | liczba skarg; NZK, alert udarowy, niewydolność oddechowa, brak reakcji, aktywacja zespołu urazowego |
| | `cc_group_cardiac`, `_neuro`, `_respiratory`, `_trauma`, `_psych_tox`, `_abdominal`, `_low_acuity` | liczba skarg w grupie |
| Interakcje | `chestpain_tachycardia`, `resp_complaint_hypoxia`, `trauma_shock`, `ams_fever`, `elderly_hypotension`, `ambulance_danger_zone` | skarga × parametr / wiek × parametr |

Punkty NEWS2:

| parametr | 3 | 2 | 1 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| RR | ≤ 8 | | 9–11 | 12–20 | | 21–24 | ≥ 25 |
| SpO₂ (skala 1) | ≤ 91 | 92–93 | 94–95 | ≥ 96 | | | |
| tlen | | tak | | nie | | | |
| SBP | ≤ 90 | 91–100 | 101–110 | 111–219 | | | ≥ 220 |
| HR | ≤ 40 | | 41–50 | 51–90 | 91–110 | 111–130 | ≥ 131 |
| temperatura | ≤ 35,0 | | 35,1–36,0 | 36,1–38,0 | 38,1–39,0 | ≥ 39,1 | |

## Braki parametrów życiowych

| ESI | bez żadnego parametru życiowego |
|---|---:|
| 1 | 85% |
| 2 | 40% |
| 3 | 26% |
| 4 | 17% |
| 5 | 38% |

Braki odzwierciedlają organizację pracy Yale, a nie stan pacjenta. Yale uznaje za pomiar
z triażu tylko odczyty sprzed przyjęcia na salę (`9b_gettriagevitals.R`), a pacjent ESI 1
zwykle trafia na salę resuscytacyjną przed pomiarem. Brak jest więc skutkiem decyzji
triażowej. Imputacja medianą (jak w poprzedniej wersji) robiła z tych pacjentów „pacjentów
z prawidłowymi parametrami”, dlatego braki zostają NaN z flagami `*_missing`.

W TOPSOR pomiary są obowiązkowe. Populację docelową opisuje kolumna meta `vitals_complete`
(HR, SBP i RR zmierzone; nie jest cechą), a metryki raportuje się osobno dla tej podgrupy.

## Zestaw `full`: jednostki badań

Historia badań w Yale jest w jednostkach konwencjonalnych USA (`LAB_UNITS_US` w
[`src/data/yale_schema.py`](src/data/yale_schema.py)). Zestaw `triage` ich nie używa; przed
podaniem polskich wyników modelowi `full` trzeba je przeliczyć do jednostek Yale:

| badanie | jednostka w Yale | przeliczenie |
|---|---|---|
| glukoza | mg/dL | mmol/L = mg/dL / 18,0 |
| kreatynina | mg/dL | µmol/L = mg/dL × 88,4 |
| BUN | mg/dL azotu mocznikowego | mocznik mg/dL = BUN × 2,14; mmol/L = BUN × 0,357 |
| wapń | mg/dL | mmol/L = mg/dL × 0,25 |
| magnez | mg/dL | mmol/L = mg/dL × 0,411 |
| fosfor | mg/dL | mmol/L = mg/dL × 0,323 |
| hemoglobina | g/dL | mmol/L = g/dL × 0,621 |
| troponina | ng/mL (POC TnI i TnT) | nieporównywalna z polskim hs-cTn (ng/L) — nie mapować |
| monocyty, bazofile (bezwzględne) | 10³/µL | `prepare` dzieli przez 1000 wartości podane w komórkach/µL |

## Kontekst kliniczny dla LLM

Te same wartości (SI, MAP, NEWS2, SIRS, qSOFA, strefa zagrożenia ESI) można podać lokalnemu
modelowi językowemu jako ustrukturyzowany kontekst. Wystarczy wywołać `engineer_features` na
wierszu pacjenta i sformatować wybrane kolumny, bez osobnej implementacji wzorów.
