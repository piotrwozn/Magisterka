"""
Polskie nazwy cech modelu ESI — do opisów pacjentów i wyjaśnień w porównaniu LLM.

    CC_LABELS_PL       — WSZYSTKIE 199 skarg głównych Yale (kolumny cc_* po `sanitize_name`,
                         z `cc_trauma_activation` = scalone aktywacje urazowe)
    FEATURE_LABELS_PL  — cechy demograficzne, przybycie, parametry życiowe i cechy inżynieryjne
                         (src.features.engineering.ENGINEERED_FEATURES)
    feature_label(nazwa)     — polska nazwa dowolnej cechy (nieznana → nazwa techniczna)
    feature_synonyms(nazwa)  — rdzenie słów (małe litery, bez polskich znaków), których obecność
                               w tekście LLM oznacza, że cecha została wspomniana (pokrycie SHAP)
    fold_text(tekst)         — małe litery bez polskich znaków diakrytycznych (do dopasowań)

Tłumaczenia są własne (opisowe), nie pochodzą z żadnej licencjonowanej terminologii.
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────
# Skargi główne (199 kolumn cc_* z Yale)
# ─────────────────────────────────────────
CC_LABELS_PL: dict[str, str] = {
    "cc_abdominalcramping": "Skurcze brzucha",
    "cc_abdominaldistention": "Wzdęcie brzucha",
    "cc_abdominalpain": "Ból brzucha",
    "cc_abdominalpainpregnant": "Ból brzucha w ciąży",
    "cc_abnormallab": "Nieprawidłowy wynik badań laboratoryjnych",
    "cc_abscess": "Ropień",
    "cc_addictionproblem": "Problem z uzależnieniem",
    "cc_agitation": "Pobudzenie",
    "cc_alcoholintoxication": "Upojenie alkoholowe",
    "cc_alcoholproblem": "Problem alkoholowy",
    "cc_allergicreaction": "Reakcja alergiczna",
    "cc_alteredmentalstatus": "Zaburzenia świadomości",
    "cc_animalbite": "Pogryzienie przez zwierzę",
    "cc_ankleinjury": "Uraz stawu skokowego",
    "cc_anklepain": "Ból stawu skokowego",
    "cc_anxiety": "Lęk",
    "cc_arminjury": "Uraz kończyny górnej",
    "cc_armpain": "Ból kończyny górnej",
    "cc_armswelling": "Obrzęk kończyny górnej",
    "cc_assaultvictim": "Ofiara napaści",
    "cc_asthma": "Astma",
    "cc_backpain": "Ból pleców",
    "cc_bleeding_bruising": "Krwawienie / siniaki",
    "cc_blurredvision": "Niewyraźne widzenie",
    "cc_bodyfluidexposure": "Ekspozycja na płyny ustrojowe",
    "cc_breastpain": "Ból piersi",
    "cc_breathingdifficulty": "Trudności w oddychaniu",
    "cc_breathingproblem": "Problem z oddychaniem",
    "cc_burn": "Oparzenie",
    "cc_cardiacarrest": "Zatrzymanie krążenia",
    "cc_cellulitis": "Zapalenie tkanki podskórnej (cellulitis)",
    "cc_chestpain": "Ból w klatce piersiowej",
    "cc_chesttightness": "Ucisk w klatce piersiowej",
    "cc_chills": "Dreszcze",
    "cc_coldlikesymptoms": "Objawy przeziębienia",
    "cc_confusion": "Splątanie",
    "cc_conjunctivitis": "Zapalenie spojówek",
    "cc_constipation": "Zaparcie",
    "cc_cough": "Kaszel",
    "cc_cyst": "Torbiel",
    "cc_decreasedbloodsugar_symptomatic": "Objawowa hipoglikemia (niski cukier we krwi)",
    "cc_dehydration": "Odwodnienie",
    "cc_dentalpain": "Ból zęba",
    "cc_depression": "Depresja",
    "cc_detoxevaluation": "Kwalifikacja do detoksykacji",
    "cc_diarrhea": "Biegunka",
    "cc_dizziness": "Zawroty głowy",
    "cc_drug_alcoholassessment": "Ocena w kierunku narkotyków / alkoholu",
    "cc_drugproblem": "Problem z narkotykami",
    "cc_dyspnea": "Duszność",
    "cc_dysuria": "Bolesne oddawanie moczu (dyzuria)",
    "cc_earpain": "Ból ucha",
    "cc_earproblem": "Problem z uchem",
    "cc_edema": "Obrzęki",
    "cc_elbowpain": "Ból łokcia",
    "cc_elevatedbloodsugar_nosymptoms": "Podwyższony cukier we krwi bez objawów",
    "cc_elevatedbloodsugar_symptomatic": "Podwyższony cukier we krwi z objawami",
    "cc_emesis": "Wymioty",
    "cc_epigastricpain": "Ból w nadbrzuszu",
    "cc_epistaxis": "Krwawienie z nosa",
    "cc_exposuretostd": "Narażenie na chorobę przenoszoną drogą płciową",
    "cc_extremitylaceration": "Rana kończyny",
    "cc_extremityweakness": "Osłabienie kończyny",
    "cc_eyeinjury": "Uraz oka",
    "cc_eyepain": "Ból oka",
    "cc_eyeproblem": "Problem z okiem",
    "cc_eyeredness": "Zaczerwienienie oka",
    "cc_facialinjury": "Uraz twarzy",
    "cc_faciallaceration": "Rana twarzy",
    "cc_facialpain": "Ból twarzy",
    "cc_facialswelling": "Obrzęk twarzy",
    "cc_fall": "Upadek",
    "cc_fall_65": "Upadek u osoby po 65. roku życia",
    "cc_fatigue": "Zmęczenie",
    "cc_femaleguproblem": "Problem ginekologiczny lub urologiczny (kobieta)",
    "cc_fever": "Gorączka",
    "cc_fever_75yearsorolder": "Gorączka u osoby w wieku 75 lat lub starszej",
    "cc_fever_9weeksto74years": "Gorączka (wiek od 9 tygodni do 74 lat)",
    "cc_feverimmunocompromised": "Gorączka u pacjenta z obniżoną odpornością",
    "cc_fingerinjury": "Uraz palca ręki",
    "cc_fingerpain": "Ból palca ręki",
    "cc_fingerswelling": "Obrzęk palca ręki",
    "cc_flankpain": "Ból w okolicy lędźwiowej (boku)",
    "cc_follow_upcellulitis": "Kontrola zapalenia tkanki podskórnej",
    "cc_footinjury": "Uraz stopy",
    "cc_footpain": "Ból stopy",
    "cc_footswelling": "Obrzęk stopy",
    "cc_foreignbodyineye": "Ciało obce w oku",
    "cc_generalizedbodyaches": "Uogólnione bóle ciała",
    "cc_gibleeding": "Krwawienie z przewodu pokarmowego",
    "cc_giproblem": "Dolegliwości żołądkowo-jelitowe",
    "cc_groinpain": "Ból pachwiny",
    "cc_hallucinations": "Omamy",
    "cc_handinjury": "Uraz dłoni",
    "cc_handpain": "Ból dłoni",
    "cc_headache": "Ból głowy",
    "cc_headache_newonsetornewsymptoms": "Ból głowy nowy lub z nowymi objawami",
    "cc_headache_recurrentorknowndxmigraines": "Nawracający ból głowy lub znana migrena",
    "cc_headachere_evaluation": "Ból głowy — ponowna ocena",
    "cc_headinjury": "Uraz głowy",
    "cc_headlaceration": "Rana głowy",
    "cc_hematuria": "Krwiomocz",
    "cc_hemoptysis": "Krwioplucie",
    "cc_hippain": "Ból biodra",
    "cc_homicidal": "Zamiary zabójcze (zagrożenie dla innych)",
    "cc_hyperglycemia": "Hiperglikemia",
    "cc_hypertension": "Nadciśnienie tętnicze",
    "cc_hypotension": "Niedociśnienie",
    "cc_influenza": "Grypa",
    "cc_ingestion": "Połknięcie substancji",
    "cc_insectbite": "Ukąszenie owada",
    "cc_irregularheartbeat": "Nieregularne bicie serca",
    "cc_jawpain": "Ból żuchwy",
    "cc_jointswelling": "Obrzęk stawu",
    "cc_kneeinjury": "Uraz kolana",
    "cc_kneepain": "Ból kolana",
    "cc_laceration": "Rana cięta lub szarpana",
    "cc_leginjury": "Uraz kończyny dolnej",
    "cc_legpain": "Ból kończyny dolnej",
    "cc_legswelling": "Obrzęk kończyny dolnej",
    "cc_lethargy": "Senność i spowolnienie (letarg)",
    "cc_lossofconsciousness": "Utrata przytomności",
    "cc_maleguproblem": "Problem urologiczny (mężczyzna)",
    "cc_mass": "Guz lub zgrubienie",
    "cc_medicalproblem": "Problem medyczny (nieokreślony)",
    "cc_medicalscreening": "Ocena medyczna (badanie przesiewowe)",
    "cc_medicationproblem": "Problem z lekami",
    "cc_medicationrefill": "Przepisanie leków",
    "cc_migraine": "Migrena",
    "cc_motorcyclecrash": "Wypadek motocyklowy",
    "cc_motorvehiclecrash": "Wypadek samochodowy",
    "cc_multiplefalls": "Wielokrotne upadki",
    "cc_nasalcongestion": "Niedrożność nosa",
    "cc_nausea": "Nudności",
    "cc_nearsyncope": "Stan przedomdleniowy",
    "cc_neckpain": "Ból szyi",
    "cc_neurologicproblem": "Problem neurologiczny",
    "cc_numbness": "Drętwienie",
    "cc_oralswelling": "Obrzęk jamy ustnej",
    "cc_otalgia": "Ból ucha (otalgia)",
    "cc_other": "Inna skarga",
    "cc_overdose_accidental": "Przypadkowe przedawkowanie",
    "cc_overdose_intentional": "Celowe przedawkowanie",
    "cc_pain": "Ból (nieokreślony)",
    "cc_palpitations": "Kołatanie serca",
    "cc_panicattack": "Napad paniki",
    "cc_pelvicpain": "Ból miednicy",
    "cc_poisoning": "Zatrucie",
    "cc_post_opproblem": "Problem pooperacyjny",
    "cc_psychiatricevaluation": "Ocena psychiatryczna",
    "cc_psychoticsymptoms": "Objawy psychotyczne",
    "cc_rapidheartrate": "Szybkie bicie serca",
    "cc_rash": "Wysypka",
    "cc_rectalbleeding": "Krwawienie z odbytu",
    "cc_rectalpain": "Ból odbytu",
    "cc_respiratorydistress": "Niewydolność oddechowa",
    "cc_ribinjury": "Uraz żeber",
    "cc_ribpain": "Ból żeber",
    "cc_seizure_newonset": "Pierwszy w życiu napad drgawek",
    "cc_seizure_priorhxof": "Napad drgawek u osoby z padaczką w wywiadzie",
    "cc_seizures": "Drgawki",
    "cc_shortnessofbreath": "Duszność (brak tchu)",
    "cc_shoulderinjury": "Uraz barku",
    "cc_shoulderpain": "Ból barku",
    "cc_sicklecellpain": "Ból w przebiegu niedokrwistości sierpowatokrwinkowej",
    "cc_sinusproblem": "Dolegliwości zatok",
    "cc_skinirritation": "Podrażnienie skóry",
    "cc_skinproblem": "Problem skórny",
    "cc_sorethroat": "Ból gardła",
    "cc_stdcheck": "Kontrola w kierunku chorób przenoszonych drogą płciową",
    "cc_strokealert": "Alert udarowy",
    "cc_suicidal": "Myśli samobójcze",
    "cc_suture_stapleremoval": "Usunięcie szwów lub zszywek",
    "cc_swallowedforeignbody": "Połknięte ciało obce",
    "cc_syncope": "Omdlenie",
    "cc_tachycardia": "Kołatanie / przyspieszone tętno (zgłaszane)",
    "cc_testiclepain": "Ból jądra",
    "cc_thumbinjury": "Uraz kciuka",
    "cc_tickremoval": "Usunięcie kleszcza",
    "cc_toeinjury": "Uraz palca stopy",
    "cc_toepain": "Ból palca stopy",
    "cc_trauma": "Uraz",
    "cc_trauma_activation": "Aktywacja zespołu urazowego",
    "cc_unresponsive": "Brak reakcji na bodźce",
    "cc_uri": "Infekcja górnych dróg oddechowych",
    "cc_urinaryfrequency": "Częste oddawanie moczu",
    "cc_urinaryretention": "Zatrzymanie moczu",
    "cc_urinarytractinfection": "Infekcja dróg moczowych",
    "cc_vaginalbleeding": "Krwawienie z dróg rodnych",
    "cc_vaginaldischarge": "Upławy",
    "cc_vaginalpain": "Ból pochwy",
    "cc_weakness": "Osłabienie",
    "cc_wheezing": "Świszczący oddech",
    "cc_withdrawal_alcohol": "Alkoholowy zespół abstynencyjny",
    "cc_woundcheck": "Kontrola rany",
    "cc_woundinfection": "Zakażenie rany",
    "cc_woundre_evaluation": "Ponowna ocena rany",
    "cc_wristinjury": "Uraz nadgarstka",
    "cc_wristpain": "Ból nadgarstka",
}

# ─────────────────────────────────────────
# Pozostałe cechy (demografia, przybycie, parametry, cechy inżynieryjne)
# ─────────────────────────────────────────
FEATURE_LABELS_PL: dict[str, str] = {
    # Demografia i przybycie
    "age": "Wiek",
    "is_male": "Płeć (mężczyzna)",
    "arrival_ambulance": "Przybycie z zespołem ratownictwa medycznego",
    "arrival_self": "Przybycie samodzielne",
    "arrival_other": "Inny sposób przybycia (np. policja)",
    "arrival_month": "Miesiąc przybycia",
    "arrival_weekday": "Dzień tygodnia przybycia",
    "arrival_hour_bin": "Pora przybycia (przedział godzin)",
    # Parametry życiowe z triażu
    "triage_vital_hr": "Tętno (HR)",
    "triage_vital_sbp": "Ciśnienie skurczowe (SBP)",
    "triage_vital_dbp": "Ciśnienie rozkurczowe (DBP)",
    "triage_vital_rr": "Częstość oddechów",
    "triage_vital_o2": "Saturacja (SpO2)",
    "triage_vital_temp": "Temperatura ciała",
    "triage_vital_o2_device": "Tlenoterapia przy triażu",
    # Hemodynamika
    "pulse_pressure": "Ciśnienie tętna (SBP − DBP)",
    "map": "Średnie ciśnienie tętnicze (MAP)",
    "shock_index": "Wskaźnik wstrząsowy (HR / SBP)",
    "modified_shock_index": "Zmodyfikowany wskaźnik wstrząsowy (HR / MAP)",
    "tachycardia": "Tachykardia (HR > 100/min)",
    "severe_tachycardia": "Ciężka tachykardia (HR > 130/min)",
    "bradycardia": "Bradykardia (HR < 50/min)",
    "hypotension": "Hipotensja (SBP < 100 mmHg)",
    "severe_hypotension": "Ciężka hipotensja (SBP < 90 mmHg)",
    "severe_hypertension": "Ciężkie nadciśnienie (SBP ≥ 180 mmHg)",
    "hypertensive_crisis": "Przełom nadciśnieniowy (SBP ≥ 200 lub DBP ≥ 120 mmHg)",
    "shock_index_critical": "Krytyczny wskaźnik wstrząsowy (≥ 1,0)",
    # Oddech
    "tachypnea": "Tachypnoe (oddechy > 20/min)",
    "severe_tachypnea": "Ciężkie tachypnoe (oddechy ≥ 30/min)",
    "bradypnea": "Bradypnoe (oddechy < 10/min)",
    "hypoxia": "Hipoksemia (SpO2 < 94%)",
    "severe_hypoxia": "Ciężka hipoksemia (SpO2 < 90%)",
    # Temperatura
    "fever": "Gorączka (T ≥ 38 °C)",
    "high_fever": "Wysoka gorączka (T ≥ 39 °C)",
    "hypothermia": "Hipotermia (T < 35 °C)",
    "temp_deviation": "Odchylenie temperatury od 37 °C",
    # Strefa zagrożenia ESI i skale
    "esi_danger_spo2": "Strefa zagrożenia ESI: SpO2 < 92%",
    "esi_danger_count": "Liczba parametrów w strefie zagrożenia ESI",
    "esi_danger_zone": "Strefa zagrożenia ESI (HR > 100, oddechy > 20 lub SpO2 < 92%)",
    "sirs_score": "Punktacja SIRS (bez leukocytów)",
    "sirs_positive": "SIRS dodatni (≥ 2 kryteria)",
    "qsofa_score": "Punktacja qSOFA",
    "qsofa_positive": "qSOFA dodatni (≥ 2 pkt)",
    "news2_score": "Punktacja NEWS2",
    "news2_high": "Wysoki NEWS2 (≥ 7 pkt)",
    # Wiek
    "age_65plus": "Wiek ≥ 65 lat",
    "age_80plus": "Wiek ≥ 80 lat",
    # Braki pomiarów
    "hr_missing": "Brak pomiaru tętna",
    "sbp_missing": "Brak pomiaru ciśnienia skurczowego",
    "dbp_missing": "Brak pomiaru ciśnienia rozkurczowego",
    "rr_missing": "Brak pomiaru częstości oddechów",
    "o2_missing": "Brak pomiaru SpO2",
    "temp_missing": "Brak pomiaru temperatury",
    "n_vitals_missing": "Liczba brakujących parametrów życiowych",
    "all_vitals_missing": "Brak wszystkich parametrów życiowych",
    # Czas
    "night_arrival": "Przybycie w nocy (23:00–06:59)",
    "weekend_arrival": "Przybycie w weekend",
    # Skargi główne — cechy zbiorcze
    "cc_count": "Liczba skarg głównych",
    "cc_critical": "Skargi krytyczne (np. zatrzymanie krążenia, alert udarowy)",
    "cc_group_cardiac": "Skargi kardiologiczne",
    "cc_group_neuro": "Skargi neurologiczne",
    "cc_group_respiratory": "Skargi oddechowe",
    "cc_group_trauma": "Skargi urazowe",
    "cc_group_psych_tox": "Skargi psychiatryczne lub toksykologiczne",
    "cc_group_abdominal": "Skargi brzuszne",
    "cc_group_low_acuity": "Skargi typowe dla niskiej pilności",
    # Interakcje
    "chestpain_tachycardia": "Ból w klatce piersiowej z tachykardią",
    "resp_complaint_hypoxia": "Skarga oddechowa z hipoksemią",
    "trauma_shock": "Uraz z wskaźnikiem wstrząsowym ≥ 1",
    "ams_fever": "Zaburzenia świadomości z gorączką",
    "elderly_hypotension": "Wiek ≥ 65 lat z hipotensją",
    "ambulance_danger_zone": "Przybycie z ZRM i strefa zagrożenia ESI",
}

# Opisy wartości do opisu pacjenta
ARRIVAL_MODES_PL: dict[str, str] = {
    "ambulance": "zespół ratownictwa medycznego (karetka)",
    "self": "samodzielnie",
    "other": "inny sposób (np. policja)",
}
HOUR_BIN_LABELS_PL: dict[int, str] = {
    0: "23:00–02:59", 1: "03:00–06:59", 2: "07:00–10:59",
    3: "11:00–14:59", 4: "15:00–18:59", 5: "19:00–22:59",
}
WEEKDAYS_PL: dict[int, str] = {
    1: "poniedziałek", 2: "wtorek", 3: "środa", 4: "czwartek", 5: "piątek", 6: "sobota", 7: "niedziela",
}
MONTHS_PL: dict[int, str] = {
    1: "styczeń", 2: "luty", 3: "marzec", 4: "kwiecień", 5: "maj", 6: "czerwiec",
    7: "lipiec", 8: "sierpień", 9: "wrzesień", 10: "październik", 11: "listopad", 12: "grudzień",
}

# ─────────────────────────────────────────
# Synonimy do sprawdzania pokrycia czynników SHAP w tekście LLM
# (rdzenie po `fold_text`; dopasowanie od początku słowa)
# ─────────────────────────────────────────
_HR = ["tetn", "hr", "akcja serca", "akcji serca", "czestosc serca", "czestosci serca", "puls",
       "tachykard", "bradykard"]
_SBP = ["cisnien", "sbp", "skurczow", "rr", "hipotens", "nadcisn", "niedocisn", "mmhg"]
_DBP = ["rozkurcz", "dbp", "cisnien"]
_RESP = ["oddech", "oddychan", "tachypno", "bradypno", "rr"]
_SPO2 = ["spo2", "saturac", "utlenow", "hipoksem", "hipoksj", "desaturac"]
_TEMP = ["temperatur", "goraczk", "goraczkuj", "hipoterm", "podgoraczk", "wychlodz"]
_ARRIVAL = ["przybyc", "przybyl", "przywiez", "karetk", "zrm", "ratownic", "zespol ratown", "samodziel",
            "policj", "transport"]
_DANGER = ["strefa zagrozenia", "strefie zagrozenia", "strefy zagrozenia", "danger zone"]

FEATURE_SYNONYMS: dict[str, list[str]] = {
    "age": ["wiek", "lat", "letni", "latek", "latk", "starsz", "senior"],
    "is_male": ["plec", "mezczyzn", "kobiet"],
    "arrival_ambulance": _ARRIVAL,
    "arrival_self": _ARRIVAL,
    "arrival_other": _ARRIVAL,
    "arrival_month": ["miesiac", "pora roku", "sezon"],
    "arrival_weekday": ["dzien tygodnia", "dnia tygodnia", "weekend", "poniedzial", "wtor", "srod", "czwart",
                        "piat", "sobot", "niedziel"],
    "arrival_hour_bin": ["pora", "godzin", "noc", "nocn", "rano", "wieczor", "popoludni"],
    "triage_vital_hr": _HR,
    "triage_vital_sbp": _SBP,
    "triage_vital_dbp": _DBP,
    "triage_vital_rr": _RESP,
    "triage_vital_o2": _SPO2,
    "triage_vital_temp": _TEMP,
    "triage_vital_o2_device": ["tlenoterap", "tlen", "maska", "wasy tlen", "kaniul"],
    "pulse_pressure": ["cisnienie tetna", "cisnienia tetna", "amplitud", *_SBP],
    "map": ["srednie cisnien", "sredniego cisnien", "map", *_SBP],
    "shock_index": ["wskaznik wstrzas", "wskaznika wstrzas", "shock index", "wstrzas"],
    "modified_shock_index": ["wskaznik wstrzas", "wskaznika wstrzas", "shock index", "wstrzas"],
    "tachycardia": _HR,
    "severe_tachycardia": _HR,
    "bradycardia": _HR,
    "hypotension": _SBP,
    "severe_hypotension": _SBP,
    "severe_hypertension": _SBP,
    "hypertensive_crisis": ["przelom nadcisn", *_SBP],
    "shock_index_critical": ["wskaznik wstrzas", "wskaznika wstrzas", "shock index", "wstrzas"],
    "tachypnea": _RESP,
    "severe_tachypnea": _RESP,
    "bradypnea": _RESP,
    "hypoxia": _SPO2,
    "severe_hypoxia": _SPO2,
    "fever": _TEMP,
    "high_fever": _TEMP,
    "hypothermia": _TEMP,
    "temp_deviation": _TEMP,
    "esi_danger_spo2": [*_DANGER, *_SPO2],
    "esi_danger_count": [*_DANGER, *_HR, *_RESP, *_SPO2],
    "esi_danger_zone": [*_DANGER, *_HR, *_RESP, *_SPO2],
    "sirs_score": ["sirs", "uogolnion", "sepsa", "seps"],
    "sirs_positive": ["sirs", "uogolnion", "sepsa", "seps"],
    "qsofa_score": ["qsofa", "sepsa", "seps"],
    "qsofa_positive": ["qsofa", "sepsa", "seps"],
    "news2_score": ["news2", "news 2", "news", "wczesnego ostrzegania"],
    "news2_high": ["news2", "news 2", "news", "wczesnego ostrzegania"],
    "age_65plus": ["wiek", "lat", "letni", "latek", "starsz", "senior", "podeszl"],
    "age_80plus": ["wiek", "lat", "letni", "latek", "starsz", "senior", "podeszl"],
    "hr_missing": ["brak pomiaru", "nie zmierzon", "brak tetn", *_HR],
    "sbp_missing": ["brak pomiaru", "nie zmierzon", *_SBP],
    "dbp_missing": ["brak pomiaru", "nie zmierzon", *_DBP],
    "rr_missing": ["brak pomiaru", "nie zmierzon", *_RESP],
    "o2_missing": ["brak pomiaru", "nie zmierzon", *_SPO2],
    "temp_missing": ["brak pomiaru", "nie zmierzon", *_TEMP],
    "n_vitals_missing": ["brak pomiar", "brakuj", "nie zmierzon", "niekompletn"],
    "all_vitals_missing": ["brak pomiar", "brakuj", "nie zmierzon", "niekompletn"],
    "night_arrival": ["noc", "nocn", "pora", "godzin"],
    "weekend_arrival": ["weekend", "sobot", "niedziel"],
    "cc_count": ["liczba skarg", "liczby skarg", "kilka skarg", "wiele skarg", "wielu skarg", "skarg",
                 "dolegliwosc", "objaw"],
    "cc_critical": ["krytyczn", "zatrzyman", "udar", "zespol urazow", "brak reakcji", "niewydolnosc oddech"],
    "cc_group_cardiac": ["kardiolog", "sercow", "serca", "klatk", "omdlen", "kolatan"],
    "cc_group_neuro": ["neurolog", "swiadomos", "udar", "drgaw", "splatan", "oslabien", "dretwien"],
    "cc_group_respiratory": ["oddech", "oddychan", "dusznos", "kaszel", "kaszl", "swiszcz", "astm"],
    "cc_group_trauma": ["uraz", "upad", "wypad", "napas"],
    "cc_group_psych_tox": ["psychiatr", "samoboj", "przedawkow", "zatruc", "alkohol", "narkot", "lek"],
    "cc_group_abdominal": ["brzuch", "brzuszn", "nudnos", "wymiot", "nadbrzusz", "krwawien"],
    "cc_group_low_acuity": ["niskiej pilnosci", "niska pilnosc", "drobn", "kontrol", "rutynow"],
    "chestpain_tachycardia": ["klatk", "piersiow", *_HR],
    "resp_complaint_hypoxia": ["oddech", "dusznos", *_SPO2],
    "trauma_shock": ["uraz", "wstrzas"],
    "ams_fever": ["swiadomos", "splatan", *_TEMP],
    "elderly_hypotension": ["wiek", "starsz", "lat", *_SBP],
    "ambulance_danger_zone": [*_ARRIVAL, *_DANGER],
}

# Jawne synonimy części skarg (pozostałe — automatycznie z polskiej nazwy)
CC_SYNONYMS: dict[str, list[str]] = {
    "cc_chestpain": ["klatk", "piersiow", "bol w klatce"],
    "cc_chesttightness": ["klatk", "piersiow", "ucisk"],
    "cc_shortnessofbreath": ["dusznos", "dusznosc", "brak tchu", "brakiem tchu", "dusz"],
    "cc_dyspnea": ["dusznos", "dusz", "brak tchu"],
    "cc_breathingdifficulty": ["oddych", "oddech", "dusznos"],
    "cc_breathingproblem": ["oddych", "oddech", "dusznos"],
    "cc_respiratorydistress": ["niewydolnos", "oddech", "oddych"],
    "cc_strokealert": ["udar", "alert udar"],
    "cc_trauma_activation": ["zespol urazow", "zespolu urazow", "aktywac", "uraz"],
    "cc_trauma": ["uraz", "urazow"],
    "cc_alteredmentalstatus": ["swiadomos", "splatan"],
    "cc_unresponsive": ["brak reakcji", "nieprzytom", "reakcj"],
    "cc_cardiacarrest": ["zatrzyman", "nzk", "krazeni"],
    "cc_emesis": ["wymiot"],
    "cc_syncope": ["omdlen", "omdlal"],
    "cc_nearsyncope": ["przedomdlen", "omdlen", "zaslabn"],
    "cc_fall_65": ["upad"],
    "cc_tachycardia": ["kolatan", "tachykard", "przyspieszon", *_HR],
    "cc_rapidheartrate": ["szybkie bicie", "kolatan", "tachykard", *_HR],
    "cc_hypotension": ["niedocisn", "hipotens", "cisnien"],
    "cc_hypertension": ["nadcisn", "cisnien"],
    "cc_uri": ["infekc", "gorne drogi oddech", "gornych drog oddech", "przeziebien"],
    "cc_fever_75yearsorolder": ["goraczk"],
    "cc_fever_9weeksto74years": ["goraczk"],
    "cc_feverimmunocompromised": ["goraczk", "odpornos", "immunosupres"],
    "cc_suicidal": ["samoboj"],
    "cc_overdose_intentional": ["przedawkow"],
    "cc_overdose_accidental": ["przedawkow"],
    "cc_lossofconsciousness": ["utrat", "przytomn"],
}

# Słowa zbyt ogólne, by same świadczyć o wspomnieniu konkretnej skargi
_GENERIC_WORDS: frozenset[str] = frozenset({
    "bol", "w", "z", "na", "do", "i", "lub", "oraz", "u", "po", "od", "we", "bez", "przez", "osoby", "osob",
    "problem", "uraz", "inny", "inna", "ocena", "objawy", "objawami", "kontrola", "ponowna", "pacjenta",
    "roku", "zycia", "wieku", "wiek", "lat", "lub", "nieokreslony", "kierunku", "badanie",
})

_FOLD = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


# ─────────────────────────────────────────
# Funkcje
# ─────────────────────────────────────────
def fold_text(text: str) -> str:
    """Małe litery, bez polskich znaków diakrytycznych, pojedyncze spacje."""
    return re.sub(r"\s+", " ", str(text).translate(_FOLD).lower()).strip()


def feature_label(name: str) -> str:
    """Polska nazwa cechy (skarga, parametr, cecha inżynieryjna); nieznana → nazwa techniczna."""
    if name in CC_LABELS_PL:
        return CC_LABELS_PL[name]
    if name in FEATURE_LABELS_PL:
        return FEATURE_LABELS_PL[name]
    # Termy interakcji EBM ("a & b")
    if " & " in name:
        return " × ".join(feature_label(part.strip()) for part in name.split(" & "))
    return name


def _stem(word: str) -> str:
    """Prosty rdzeń: słowo skrócone o końcówkę fleksyjną (min. 4, maks. 6 znaków)."""
    if len(word) <= 4:
        return word
    return word[:max(4, min(6, len(word) - 2))]


def _auto_synonyms(label: str) -> list[str]:
    """Rdzenie znaczących słów polskiej nazwy (bez słów ogólnych; same ogólne → wszystkie)."""
    words = [w for w in re.split(r"[^a-z0-9]+", fold_text(label)) if w]
    specific = [w for w in words if w not in _GENERIC_WORDS and len(w) >= 3]
    return list(dict.fromkeys(_stem(w) for w in (specific or words)))


def feature_synonyms(name: str) -> list[str]:
    """Rdzenie (po `fold_text`) świadczące o wspomnieniu cechy w tekście."""
    if " & " in name:
        out: list[str] = []
        for part in name.split(" & "):
            out += feature_synonyms(part.strip())
        return list(dict.fromkeys(out))
    if name in FEATURE_SYNONYMS:
        return list(FEATURE_SYNONYMS[name])
    if name in CC_SYNONYMS:
        return list(dict.fromkeys(CC_SYNONYMS[name] + _auto_synonyms(CC_LABELS_PL.get(name, ""))))
    if name in CC_LABELS_PL:
        return _auto_synonyms(CC_LABELS_PL[name])
    return _auto_synonyms(feature_label(name)) or [fold_text(name)]


def mentions(text: str, synonyms: list[str]) -> bool:
    """Czy tekst (dowolny) zawiera któryś z rdzeni — dopasowanie od początku słowa."""
    folded = fold_text(text)
    return any(re.search(r"(?<![a-z0-9])" + re.escape(fold_text(s)), folded) for s in synonyms if s)


def is_complaint(name: str) -> bool:
    """Kolumna skargi głównej z danych (nie zbiorcza cecha cc_count / cc_critical / cc_group_*)."""
    return name in CC_LABELS_PL


__all__ = [
    "ARRIVAL_MODES_PL",
    "CC_LABELS_PL",
    "FEATURE_LABELS_PL",
    "HOUR_BIN_LABELS_PL",
    "MONTHS_PL",
    "WEEKDAYS_PL",
    "feature_label",
    "feature_synonyms",
    "fold_text",
    "is_complaint",
    "mentions",
]
