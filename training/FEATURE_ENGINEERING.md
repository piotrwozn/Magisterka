# Cechy kliniczne — SOR-AI (ESI)

Implementacja: [`src/features/engineering.py`](src/features/engineering.py) (`engineer_features`,
`ENGINEERED_FEATURES` — 59 cech). Ta sama funkcja działa w treningu (`src/data/prepare.py`)
i w backendzie (pojedynczy pacjent).

## Zasady

- **Jednostki SI, temperatura w °C.** Yale podaje °F; przeliczenie dzieje się w `prepare`, zanim
  powstanie jakakolwiek cecha.
- **Brak pomiaru ≠ norma.** Flaga liczona z brakującego parametru jest NaN, a o braku mówią
  osobne cechy `*_missing`. Modele drzewiaste i EBM obsługują NaN natywnie.
- **Tylko cechy z uzasadnieniem klinicznym.** Przekształcenia monotoniczne (x², log x)
  i przypadkowe iloczyny parametrów nic nie wnoszą modelom drzewiastym, więc zostały usunięte.
- **Dorośli.** Yale zawiera wyłącznie pacjentów ≥ 18 lat, dlatego nie ma cech pediatrycznych.

## Wejście (schemat kanoniczny)

| kolumna | opis |
|---|---|
| `triage_vital_hr`, `_sbp`, `_dbp`, `_rr`, `_o2`, `_temp` | HR /min, ciśnienie mmHg, RR /min, SpO₂ %, temperatura °C |
| `triage_vital_o2_device` | tlenoterapia w chwili pomiaru (0/1) |
| `age`, `is_male` | wiek w latach, płeć |
| `arrival_ambulance`, `_self`, `_police`, `_other`, `_unknown` | sposób przybycia (one-hot) |
| `arrival_month`, `arrival_weekday`, `arrival_hour_bin` | 1–12, 1–7 (pon.=1), 0–5 (0 = 23:00–02:59) |
| `cc_*` | 200 skarg głównych z Yale/Epic (nazwy znormalizowane) |

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
| Temperatura | `fever`, `high_fever`, `hypothermia`, `temp_deviation` | ≥ 38,0 °C; ≥ 39,0 °C; < 36,0 °C; \|t − 37\| |
| Strefa zagrożenia ESI | `esi_danger_hr`, `esi_danger_rr`, `esi_danger_spo2`, `esi_danger_count`, `esi_danger_zone` | punkt decyzyjny D (dorośli): HR > 100, RR > 20, SpO₂ < 92 |
| Skale | `sirs_score`, `sirs_positive` | temp > 38 lub < 36, HR > 90, RR > 20 (bez leukocytów); ≥ 2 |
| | `qsofa_score`, `qsofa_positive` | RR ≥ 22, SBP ≤ 100, zaburzenia świadomości (proxy ze skarg); ≥ 2 |
| | `news2_score`, `news2_high` | NEWS2 bez oceny świadomości (skala SpO₂ 1); ≥ 7 |
| Wiek | `age_65plus`, `age_80plus` | |
| Braki | `hr_missing` … `temp_missing`, `n_vitals_missing`, `all_vitals_missing` | brak pomiaru w triażu |
| Czas | `night_arrival`, `weekend_arrival` | przedział 23:00–06:59; sobota/niedziela |
| Skargi | `cc_count`, `cc_critical` | liczba skarg; NZK, alert udarowy, niewydolność oddechowa, brak reakcji, pełny uraz |
| | `cc_group_cardiac`, `_neuro`, `_respiratory`, `_trauma`, `_psych_tox`, `_abdominal`, `_low_acuity` | sumy skarg w grupach |
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

## Dlaczego braki są tak ważne

| ESI | bez żadnego parametru życiowego |
|---|---:|
| 1 | 85% |
| 2 | 40% |
| 3 | 26% |
| 4 | 17% |
| 5 | 38% |

Pacjenci ESI 1 trafiają od razu na salę resuscytacyjną, bez pomiaru w triażu. Imputacja medianą
(jak w poprzedniej wersji) robiła z nich „pacjentów z prawidłowymi parametrami”. W TOPSOR pomiary
są obowiązkowe, więc przed wdrożeniem trzeba sprawdzić, czy ten wzorzec występuje w polskich danych.

## Kontekst kliniczny dla LLM

Te same wartości (SI, MAP, NEWS2, SIRS, qSOFA, strefa zagrożenia ESI) można podać lokalnemu
modelowi językowemu jako ustrukturyzowany kontekst. Wystarczy wywołać `engineer_features` na
wierszu pacjenta i sformatować wybrane kolumny, bez osobnej implementacji wzorów.
