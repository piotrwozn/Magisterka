# Audyt danych Yale EMMLC → ESI

Wygenerowano automatycznie przez `scripts/02_preprocess.py` (wersja danych `esi-v1`, podział `random`). Nie edytuj ręcznie.

## Źródło i czyszczenie

- Surowe dane: 560,486 wizyt × 972 kolumn.
- Usunięto 2,457 wizyt bez ESI i 2 duplikaty.
- Po czyszczeniu: **558,027 wizyt**, wyłącznie dorośli (wiek 18–108).
- Brak roku wizyty i identyfikatora pacjenta → podział chronologiczny i podział po pacjentach nie są możliwe; ten sam pacjent może wystąpić w train i test.

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
- Wartości poza zakresem fizjologicznym zamienione na brak (NaN), nie na medianę:

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
| `triage_vital_o2` | 48.3% | 98.0 | 91.0 | 99.0 |
| `triage_vital_temp` | 32.5% | 36.7 | 35.7 | 38.2 |
| `triage_vital_o2_device` | 40.3% | 0.0 | 0.0 | 1.0 |

**Braki są informatywne.** Odsetek wizyt bez ŻADNEGO parametru życiowego:

| ESI | bez vitals |
|---|---:|
| 1 | 85.2% |
| 2 | 39.5% |
| 3 | 26.2% |
| 4 | 17.1% |
| 5 | 37.7% |

Pacjenci ESI 1 zwykle trafiają od razu na salę resuscytacyjną, bez pomiaru w triażu. Dlatego braki zostają jako NaN plus flagi `*_missing`, a nie jako mediana. Uwaga przy wdrożeniu: w TOPSOR pomiary są obowiązkowe, więc ten wzorzec może się różnić.

## Przybycie i czas

| sposób przybycia (Yale) | liczba | kategoria kanoniczna |
|---|---:|---|
| Car | 222,638 | self |
| ambulance | 189,679 | ambulance |
| Walk-in | 113,564 | self |
| <brak> | 20,799 | unknown |
| Other | 6,853 | other |
| Public Transportation | 2,820 | self |
| Wheelchair | 1,575 | other |
| Police | 99 | police |

Miesiąc, dzień tygodnia i przedział godzin zakodowane jawnie (1–12, 1–7 od poniedziałku, 0–5 od 23:00). Wcześniej LabelEncoder kodował je alfabetycznie, przez co `ambulance_flag`, `weekend_flag` i pory dnia liczyły się na złych kategoriach.

## Wykluczone kolumny

- Specyficzne dla USA: `ethnicity`, `race`, `lang`, `religion`, `maritalstatus`, `employstatus`, `insurance_status`.
- `dep_name` (oddział) — tylko do stratyfikacji i podziału `--split site`, nie jako cecha.
- `disposition` (przyjęcie/wypis) — wynik wizyty, zapisany jako `admitted` tylko do analiz; nigdy nie jest cechą.
- eGFR liczony z rasą: `egfr_last`, `egfr(nonafricanamerican)_last`, `egfr(aframer)_last`, `egfr_min`, `egfr(nonafricanamerican)_min`, `egfr(aframer)_min`, `egfr_max`, `egfr(nonafricanamerican)_max`, `egfr(aframer)_max`, `egfr_median`, `egfr(nonafricanamerican)_median`, `egfr(aframer)_median`.
- Wizyty bez żadnej skargi głównej: 2,576 (skargi uzupełnione zerami).

## Zestawy cech

| grupa | liczba cech | w zestawie |
|---|---:|---|
| demographics | 2 | triage + full |
| arrival | 5 | triage + full |
| time | 3 | triage + full |
| triage_vitals | 7 | triage + full |
| chief_complaints | 200 | triage + full |
| engineered | 59 | triage + full |
| past_medical | 281 | full |
| medications | 48 | full |
| ed_usage | 13 | full |
| imaging_history | 9 | full |
| history_vitals | 28 | full |
| history_labs | 367 | full |
| **triage** | **276** | dostępne w chwili triażu |
| **full** | **1022** | triaż + historia pacjenta |

## Podział `random`

| zbiór | n | ESI 1 | ESI 2 | ESI 3 | ESI 4 | ESI 5 | oddziały |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 446,421 | 0.9% | 29.3% | 42.3% | 22.4% | 5.0% | A: 256,024, B: 133,068, C: 57,329 |
| val | 55,803 | 0.9% | 29.3% | 42.3% | 22.4% | 5.0% | A: 32,004, B: 16,633, C: 7,166 |
| test | 55,803 | 0.9% | 29.3% | 42.3% | 22.4% | 5.0% | A: 32,003, B: 16,634, C: 7,166 |

- `train` — trening modeli (10% z niego wydzielane w locie na early stopping).
- `val` — porównanie modeli i ręczne łączenie (wagi ensemble, progi decyzyjne).
- `test` — tylko ostateczna ocena; nie dostrajać na nim niczego.

