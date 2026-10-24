# SOR-AI — Technical Analysis
## Multi-Agent Triage Decision Support System

**Autor:** Piotr  
**Data:** 2026-10-19 (pierwsza wersja: 2026-05-17)  
**Wersja:** 2.0  
**Sprzęt docelowy:** stacja robocza (i9-14900K, RTX 4080 Super 16 GB) lub lokalny serwer

> **Historia:** wcześniejsza wersja projektu mapowała ESI→MTS; mapowanie usunięto — etykietą jest
> natywny ESI. Szczegóły uruchamiania treningu: `training/README.md`; raport z danych: `training/DATA_AUDIT.md`.

---

## 1. Cel systemu

System wspomagania decyzji triażowych oparty na architekturze multi-agent, łączący modele tabelaryczne ML z modelami językowymi (LLM) w celu klasyfikacji pacjentów do poziomów **Emergency Severity Index (ESI 1–5)**. Tego samego algorytmu polskie SOR-y używają w systemie TOPSOR (Dz.U. 2021 poz. 1182, § 2 pkt 9); kolory TOPSOR służą wyłącznie do prezentacji poziomu ESI:

| ESI | Kolor TOPSOR | Maks. czas do kontaktu z lekarzem |
|---|---|---|
| 1 | czerwony | natychmiast |
| 2 | pomarańczowy | do 10 min |
| 3 | żółty | do 60 min |
| 4 | zielony | do 120 min |
| 5 | niebieski | do 240 min |

Klasy modelu: indeks 0–4 = ESI 1–5. System jest zaprojektowany jako **narzędzie wspomagające** — ostateczna decyzja zawsze należy do pielęgniarki triażowej.

---

## 2. Architektura systemu

### 2.1 Diagram przepływu

```
┌─────────────────────────────────────────────────────────┐
│                     RAW INPUT                           │
│  "pacjent 67l, blady, spocony, ból w klatce,            │
│   temp 38.2, HR 118, SBP 95"                            │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              WARSTWA 0 — PARSER                         │
│                  Llama 3.2 3B                           │
│            Structured Output (Ollama)                   │
│         Temperatura = 0, Format = JSON                  │
└──────────────────┬──────────────────────────────────────┘
                   ↓
         JSON Payload (Pydantic)
                   ↓
        ┌──────────┴──────────┐
        ↓                     ↓
┌───────────────┐    ┌─────────────────────┐
│  WARSTWA 1A   │    │     WARSTWA 1B       │
│   Tabular     │    │   Clinical NLP       │
│   Pipeline    │    │   MedGemma 27B       │
│               │    │                     │
│ XGBoost       │    │ Analiza notatki      │
│ LightGBM      │    │ klinicznej           │
│ CatBoost      │    │ pielęgniarki         │
│ RF, EBM       │    │                     │
│ LogReg        │    │ "blady + spocony +  │
│    ↓          │    │  ból w klatce =     │
│ ważona średnia│    │  ryzyko OZW"        │
│ (wagi z val)  │    │                     │
│    ↓          │    │                     │
│ SHAP + reguły │    │                     │
└───────┬───────┘    └──────────┬──────────┘
        ↓                       ↓
        └───────────┬───────────┘
                    ↓
┌───────────────────────────────────────────────────────┐
│               WARSTWA 2 — SYNTEZA                     │
│                   Qwen3 32B                           │
│               Temperatura = 0                         │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │           CONFLICT RESOLUTION                   │ │
│  │                                                 │ │
│  │  Ensemble < MedGemma → Eskaluj + Flaguj         │ │
│  │  Ensemble = MedGemma → Wysoki confidence        │ │
│  │  Ensemble > MedGemma → Zostań + Wyjaśnij        │ │
│  │  Różnica ≥ 2 stopnie → Alert lekarski           │ │
│  └─────────────────────────────────────────────────┘ │
└───────────────────────────┬───────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────┐
│                    OUTPUT                             │
│                                                       │
│  Sugestia: ESI 2 · pomarańczowy (71%)                 │
│  Powód: temp 38.2 + HR 118 + obserwacja kliniczna    │
│  ⚠️ KONFLIKT: ensemble=ESI 3, kliniczne=ESI 2         │
│  Top SHAP: triage_vital_hr, triage_vital_sbp          │
│                                                       │
│           [ZATWIERDŹ]  [ODRZUĆ]                       │
└───────────────────────────────────────────────────────┘
                            ↓
                  Pielęgniarka zatwierdza
                            ↓
                    Decyzja logowana
```

---

## 3. Warstwa 0 — Parser (Llama 3.2 3B)

### Zadanie
Konwersja surowego tekstu wejściowego (mix danych vitals + obserwacja kliniczna) na ustrukturyzowany JSON payload wysyłany równolegle do Warstwy 1A i 1B.

### Implementacja

```python
import ollama
from pydantic import BaseModel
from typing import Optional

class StructuredObservations(BaseModel):
    pallor: Optional[bool] = None
    diaphoresis: Optional[bool] = None
    pain_location: Optional[str] = None
    consciousness: Optional[str] = None
    agitation: Optional[bool] = None
    skin_color: Optional[str] = None

class TabularFeatures(BaseModel):
    age: Optional[int] = None
    triage_vital_temp: Optional[float] = None
    triage_vital_hr: Optional[float] = None
    triage_vital_sbp: Optional[float] = None
    triage_vital_dbp: Optional[float] = None
    triage_vital_rr: Optional[float] = None
    triage_vital_o2: Optional[float] = None
    cc_chestpain: Optional[int] = None
    cc_respiratorydistress: Optional[int] = None
    # ... pozostałe cechy (razem 271 w zestawie triage, 903 w full — training/DATA_AUDIT.md)

class TriagePayload(BaseModel):
    tabular: TabularFeatures
    clinical_note: str
    structured_observations: StructuredObservations

SYSTEM_PROMPT = """
Jesteś parserem JSON dla systemu triażowego SOR.
Zwracasz WYŁĄCZNIE poprawny JSON bez żadnego tekstu przed ani po.
Nigdy nie interpretujesz danych medycznych.
Nigdy nie sugerujesz diagnozy ani kategorii triażowej.
Jeśli pole jest nieznane lub niepodane — zwróć null.
"""

def parse_input(raw_text: str) -> TriagePayload:
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_text}
        ],
        format=TriagePayload.model_json_schema()
    )
    return TriagePayload.model_validate_json(
        response['message']['content']
    )
```

### Dlaczego Llama 3.2 3B
- Szybki (< 1s na RTX lokalnie)
- Structured output przez Ollama — fizycznie nie może zwrócić niepoprawnego JSON
- Parsowanie tekstu to zadanie gdzie małe modele nie ustępują dużym
- Tani w inference — nie marnuje zasobów GPU na proste formatowanie

---

## 4. Warstwa 1A — Tabular ML Pipeline

### 4.1 Dane i etykieta

- **Źródło:** Yale EMMLC (3 oddziały ratunkowe w USA, 2014–2017), ok. 558 tys. wizyt dorosłych
  po odrzuceniu wizyt bez ESI i duplikatów.
- **Etykieta:** natywny poziom ESI 1–5 nadany przez pielęgniarkę triażową (`target` 0–4).
  Rozkład: ESI 1 ≈ 0,9%, ESI 2 ≈ 29%, ESI 3 ≈ 42%, ESI 4 ≈ 22%, ESI 5 ≈ 5%.
- **Podział:** train 80% / val 10% / test 10%, losowy ze stratyfikacją ESI × oddział
  (brak roku wizyty i identyfikatora pacjenta). Wariant zewnętrzny `--split site`: oddział B
  jako test (oddział C ma tylko 19 wizyt ESI 1). Z train wydzielane jest 10% na early stopping;
  `val` służy do porównania i łączenia modeli, `test` wyłącznie do raportu końcowego.
- **Zestawy cech:** `triage` (to, co pielęgniarka wie w chwili triażu: wiek, płeć, sposób
  przybycia, czas, parametry życiowe w °C, skargi główne, cechy kliniczne jak NEWS2, qSOFA,
  wskaźnik wstrząsowy, strefa zagrożenia ESI) i `full` (triaż + historia z WCZEŚNIEJSZYCH wizyt:
  choroby przewlekłe, korzystanie z SOR, obrazowanie, parametry i badania z historii).
- **Wykluczone, bo specyficzne dla USA lub będące proxy społeczno-ekonomicznym:** ubezpieczenie,
  rasa, etniczność, język, wyznanie, stan cywilny, zatrudnienie, eGFR liczony z rasą, `socialadmin`,
  administracyjne kody V, rzadkie kategorie chorób (małe wsparcie).
- **Wykluczone jako przeciek lub artefakt organizacyjny:** `disposition` (wynik wizyty), oddział
  Yale (tylko do stratyfikacji i ewaluacji), leki `meds_*` (dokumentowane głównie przy przyjęciu —
  przeciek bieżącej wizyty), uszkodzone kolumny badań i D-dimer (mieszane jednostki).
- **Ujednolicenia przenośne na polski SOR:** sposób przybycia → `ambulance / self / other`
  (brak danych → `self`, policja → `other`); pełna i zmodyfikowana aktywacja zespołu urazowego →
  jedna flaga `cc_trauma_activation`; dyspozycja poprzedniej wizyty → 5 kategorii
  (`admitted / discharged / transferred / left_early / none`); skargi główne jako flagi 0/1.
- **Braki parametrów życiowych zostają NaN.** W Yale brak pomiaru to głównie pacjent wprowadzony
  od razu na salę (decyzja po ocenie pilności, ≈ 85% wizyt ESI 1). W TOPSOR pomiar jest obowiązkowy,
  dlatego każdą metrykę raportujemy także w podgrupie `vitals_complete == 1` (HR, SBP i RR zmierzone).

### 4.2 Modele (6, celowo zróżnicowane)

| Model | Rola | Kluczowa zaleta |
|---|---|---|
| Regresja logistyczna | liniowy punkt odniesienia | przejrzystość; imputacja medianą + flagi braków w pipeline |
| LightGBM | boosting, drzewa liść-po-liściu | najszybszy, dobra jakość prawdopodobieństw |
| XGBoost | boosting, drzewa poziomowe (GPU) | TreeSHAP, stabilność |
| CatBoost | boosting, drzewa symetryczne (GPU) | silna regularyzacja, inna struktura błędów |
| Random Forest | bagging | błędy słabo skorelowane z boostingiem |
| EBM | interpretowalny GAM (bez interakcji) | krzywe wkładu cech czytelne dla klinicysty |

Domyślnie modele trenują się **bez wag klas** (skalibrowane prawdopodobieństwa). Asymetrię kosztów
(niedotriażowanie 3× droższe niż przetriażowanie, koszt rośnie z kwadratem odległości poziomów)
realizuje reguła decyzyjna na prawdopodobieństwach (`cost`), obok zwykłego `argmax`.

### 4.3 Tuning — Optuna

- Cel domyślny: **log loss** w k-fold CV na TRAIN (`--tune-metric qwk` optymalizuje QWK i dobiera
  wagi klas `cw_esi1`, `cw_esi2`, `cw_esi4`, `cw_esi5`). `val` i `test` nie biorą udziału w tuningu.
- Early stopping wewnątrz CV na wycinku części treningowej foldu (nigdy na foldzie ocenianym).
- Sampler TPE (multivariate, group); profile budżetu `smoke / quick / standard / thorough`
  (liczba prób, wiersze do tuningu, foldy, limit czasu na model); `standard` stroi modele GBDT
  i regresję logistyczną na całym zbiorze treningowym.
- Badania zapisywane w SQLite (`logs/optuna/`) z kluczem z pełnej konfiguracji eksperymentu
  — przerwany tuning wznawia się i wykonuje tylko brakujące próby.
- XGBoost i CatBoost na GPU, gdy próbny trening na karcie się powiedzie; w przeciwnym razie CPU
  z ostrzeżeniem i mniejszym budżetem.

```python
# Wagi klas jako hiperparametry (tylko cel 'qwk'); ESI 3 = klasa referencyjna
class_weights = {
    0: trial.suggest_float("cw_esi1", 1.0, 30.0, log=True),
    1: trial.suggest_float("cw_esi2", 0.5, 5.0, log=True),
    2: 1.0,
    3: trial.suggest_float("cw_esi4", 0.5, 5.0, log=True),
    4: trial.suggest_float("cw_esi5", 0.5, 10.0, log=True),
}
```

### 4.4 Metryki i raportowanie wyników

| Metryka | Znaczenie |
|---|---|
| `quadratic_weighted_kappa` | główna metryka porządkowa (zgodność z pielęgniarką) |
| `undertriage_rate` | ESI 1–2 przewidziane jako ESI 3–5 — metryka bezpieczeństwa |
| `critical_miss_rate` | ESI 1 przewidziane jako ESI 4–5 |
| `overtriage_rate` | ESI 4–5 przewidziane jako ESI 1–2 |
| `auc_macro`, `log_loss`, `brier` | jakość prawdopodobieństw |
| `expected_cost` | średni koszt decyzji wg macierzy kosztów |

Każdy trening zapisuje komplet w `runs/<run>/` (model, metryki val/test dla obu reguł, predykcje
do łączenia modeli). Ewaluacja (`scripts/04_evaluate.py`) raportuje dodatkowo podgrupę
`vitals_complete == 1` (populacja zbliżona do TOPSOR) i każdy oddział osobno. Wyniki referencyjne
uzupełniamy po pełnym treningu na danych w bieżącej wersji (`esi-v2`); liczby z wcześniejszych
wersji pipeline'u są nieważne.

### 4.5 Łączenie modeli (ręczne, na `val`)

Zamiast automatycznego meta-learnera modele łączone są ręcznie na zapisanych predykcjach
(`src.ensemble`): ważona średnia prawdopodobieństw (`blend`, podpowiedź wag `optimize_weights`
— minimum log loss na val), opcjonalna kalibracja temperaturą (`fit_temperature`) i analiza
różnorodności błędów (`diversity`). Wagi, kalibrację i wybór modeli dobieramy wyłącznie na `val`
(najlepiej w podgrupie `vitals_complete == 1`), a na `test` raportujemy dokładnie tę samą recepturę.

```python
from src.ensemble import latest_runs, aligned_probas, blend, evaluate, optimize_weights

runs = latest_runs("triage")
y_val, P_val = aligned_probas(runs, "val")
weights = optimize_weights(P_val, y_val)
print(evaluate(y_val, blend(P_val, weights), rule="cost")["quadratic_weighted_kappa"])
```

**Dlaczego bez stackingu:** wagi średniej to kilka liczb czytelnych dla audytora, a ważona średnia
skalibrowanych prawdopodobieństw nie wymaga drugiego poziomu uczenia na predykcjach OOF.

### 4.6 Regułowa weryfikacja ESI (sanity check)

`src/explain/esi_rules.py` sprawdza punkty decyzyjne ESI możliwe do oceny z danych triażu:
A (zagrożenie życia → ESI 1: np. SpO₂ < 85%, SBP < 80, HR < 40 lub > 150), B (wysokie ryzyko →
ESI 2: SBP < 90, alert udarowy, aktywacja zespołu urazowego, myśli samobójcze …) i D (strefa
zagrożenia u dorosłych → rozważ ESI 2: HR > 100, RR > 20, SpO₂ < 92%). Model mniej pilny niż
reguły = sygnał potencjalnego niedotriażowania. Punktu C (liczba zasobów, ESI 3–5) z samych
parametrów nie da się ustalić.

### 4.7 Conflict detection między modelami

```python
def detect_model_conflict(oof_probs: dict) -> dict:
    predictions = {
        name: probs.argmax() 
        for name, probs in oof_probs.items()
    }
    
    unique_preds = set(predictions.values())
    max_diff = max(predictions.values()) - min(predictions.values())
    
    return {
        "conflict": len(unique_preds) > 1,
        "severity": "high" if max_diff >= 2 else "low",
        "predictions": predictions,
        "alert_doctor": max_diff >= 2
    }
```

---

## 5. Warstwa 1B — Clinical NLP (MedGemma 27B)

### Zadanie
Analiza tekstu obserwacji klinicznej pielęgniarki — wychwytywanie sygnałów których dane tabelaryczne nie zawierają.

### Dlaczego MedGemma 27B
- Trenowana na PubMed, notatkach klinicznych, terminologii medycznej
- Rozumie niuanse kliniczne: "wygląda gorzej niż mówi", "marmurkowatość skóry", "nie może znaleźć pozycji"
- Najlepszy dostępny model dla tekstu medycznego
- 27B daje znacząco lepsze rozumowanie kliniczne niż 4B

### Przykłady obserwacji których modele tabelaryczne nie widzą

| Obserwacja pielęgniarki | Znaczenie kliniczne |
|---|---|
| "blady, spocony, niespokojny" | Wstrząs we wczesnej fazie |
| "dziwny zapach z ust" | Kwasica ketonowa (DKA) |
| "marmurkowatość skóry" | Wstrząs septyczny |
| "nie może znaleźć pozycji" | Kolka nerkowa / ostre brzucho |
| "mówi że boli 3/10 ale się trzęsie" | Niedoszacowanie bólu |
| "wygląda gorzej niż mówi" | Ogólna ocena kliniczna |

### Implementacja

```python
import ollama

MEDGEMMA_SYSTEM = """
Jesteś systemem analizy klinicznej dla triażu SOR.
Analizujesz obserwację pielęgniarki i zwracasz ocenę ryzyka.
Odpowiadasz TYLKO w formacie JSON.
Nie sugerujesz diagnozy — tylko poziom ESI (1–5) i uzasadnienie.
"""

class ClinicalAssessment(BaseModel):
    esi_level: int                # 1-5 (ESI)
    confidence: float             # 0-1
    key_findings: list[str]       # max 3 kluczowe obserwacje
    risk_flags: list[str]         # czerwone flagi kliniczne
    reasoning: str                # uzasadnienie dla Qwen3

def analyze_clinical_note(note: str) -> ClinicalAssessment:
    response = ollama.chat(
        model="medgemma:27b",
        messages=[
            {"role": "system", "content": MEDGEMMA_SYSTEM},
            {"role": "user", "content": note}
        ],
        format=ClinicalAssessment.model_json_schema()
    )
    return ClinicalAssessment.model_validate_json(
        response['message']['content']
    )
```

---

## 6. Warstwa 2 — Synteza (Qwen3 32B)

### Zadanie
Orkiestrator — łączy wyniki Warstwy 1A i 1B, wykrywa konflikty, generuje finalną sugestię z uzasadnieniem czytelnym dla pielęgniarki.

### Conflict Resolution Policy

```python
def resolve_conflict(
    ensemble_category: int,      # ESI 1-5
    medgemma_category: int,      # ESI 1-5
    ensemble_confidence: float,
    medgemma_confidence: float
) -> dict:
    
    diff = abs(ensemble_category - medgemma_category)
    
    # Konflikt krytyczny — różnica 2+ stopni
    if diff >= 2:
        return {
            "final_category": min(ensemble_category, medgemma_category),
            "flag": "ALERT_DOCTOR",
            "message": "Znaczna rozbieżność — wymagana ocena lekarska",
            "override": True
        }
    
    # MedGemma bardziej pilna — zawsze eskaluj
    if medgemma_category < ensemble_category:
        return {
            "final_category": medgemma_category,
            "flag": "CLINICAL_OVERRIDE",
            "message": "Obserwacja kliniczna nadpisała model tabelaryczny",
            "override": True
        }
    
    # Ensemble bardziej pilny lub zgodność
    return {
        "final_category": ensemble_category,
        "flag": "MODEL_DOMINANT" if diff > 0 else "CONSENSUS",
        "message": "Dane vitalne dominują nad obserwacją" if diff > 0 else "Pełna zgodność",
        "override": False
    }
```

### Zasada nadrzędna
**Przy konflikcie zawsze eskaluj w górę — nigdy w dół.**  
Lepiej zawyżyć poziom pilności niż go zaniżyć. Zgodne z metryką bezpieczeństwa: undertriage jest niedopuszczalny, overtriage jest akceptowalny.

### Prompt dla Qwen3 32B

```python
QWEN_SYSTEM = """
Jesteś systemem syntezującym decyzję triażową.
Otrzymujesz:
1. Wynik modeli ML (poziom ESI + pewność + SHAP top 5 cech + weryfikacja regułowa ESI)
2. Ocenę kliniczną MedGemma (poziom ESI + uzasadnienie)

Twoje zadanie:
- Wykryj konflikt między źródłami
- Zastosuj politykę conflict resolution
- Wygeneruj uzasadnienie czytelne dla pielęgniarki (max 2 zdania)
- Zwróć TYLKO JSON

Temperatura = 0. Nie halucynuj. Nie dodawaj wiedzy medycznej spoza kontekstu.
"""
```

---

## 7. Polityka bezpieczeństwa systemu

### Metryki bezpieczeństwa (cel)

| Metryka | Cel | Gdzie mierzymy |
|---|---|---|
| Critical Miss Rate | < 0.1% | val/test, całość i podgrupa `vitals_complete == 1` |
| Undertriage Rate (ESI 1–2 → 3–5) | < 2% (reguła kosztowa) | jw. + każdy oddział osobno |
| Overtriage Rate (ESI 4–5 → 1–2) | < 30% | jw. |

Cele dotyczą reguły kosztowej; reguła `argmax` maksymalizuje QWK kosztem wyższego niedotriażowania.

### Reguły nadrzędne (hardcoded, poza LLM)

```python
SAFETY_RULES = [
    # Reguła 1 — nigdy nie obniżaj poniżej minimum z modeli
    lambda ensemble, medgemma: min(ensemble, medgemma),
    
    # Reguła 2 — przy różnicy ≥ 2 zawsze alert lekarski
    lambda diff: "ALERT" if diff >= 2 else "OK",
    
    # Reguła 3 — confidence < 0.6 → zawsze flaguj
    lambda conf: "LOW_CONFIDENCE" if conf < 0.6 else "OK",
]
```

**Reguły bezpieczeństwa są zakodowane w logice aplikacji, nie w prompcie LLM.** LLM nie może ich nadpisać.

---

## 8. Structured Outputs — implementacja lokalna

Wszystkie modele LLM używają Structured Outputs przez Ollama — model fizycznie nie może zwrócić niepoprawnego JSON.

```python
# Instalacja
# ollama >= 0.5.0 wspiera natywny structured output

import ollama
from pydantic import BaseModel

def llm_call(
    model: str,
    messages: list,
    response_schema: type[BaseModel],
    temperature: float = 0.0
) -> BaseModel:
    response = ollama.chat(
        model=model,
        messages=messages,
        format=response_schema.model_json_schema(),
        options={"temperature": temperature}
    )
    return response_schema.model_validate_json(
        response['message']['content']
    )
```

---

## 9. Stack technologiczny

| Komponent | Technologia | Wersja |
|---|---|---|
| Tabular ML | XGBoost, LightGBM, CatBoost, scikit-learn, InterpretML (EBM) | najnowsze |
| Hyperparameter tuning | Optuna + MultivariateTPE (SQLite, wznawialne) | najnowsze |
| Łączenie modeli | ważona średnia + temperature scaling (`src.ensemble`) | — |
| Explainability | SHAP, EBM, reguły ESI | najnowsze |
| LLM runtime | Ollama | >= 0.5.0 |
| Structured outputs | Pydantic v2 | >= 2.0 |
| Parser | Llama 3.2 3B | lokalnie |
| Clinical NLP | MedGemma 27B | lokalnie |
| Orchestrator | Qwen3 32B | lokalnie |
| API | FastAPI | najnowsze |
| Logging | Python logging + SQLite | — |

---

## 10. Zgodność z regulacjami (MDR 2017/745)

System jest zaprojektowany z myślą o ścieżce certyfikacji jako wyrób medyczny klasy IIa/IIb:

| Wymaganie MDR | Jak spełnione |
|---|---|
| Interpretowalność decyzji | EBM + SHAP + reguły ESI + jawne wagi łączenia modeli |
| Audytowalność | Każda decyzja logowana z pełnym kontekstem |
| Deterministyczność | Temperatura=0 dla wszystkich LLM |
| Człowiek w pętli | Pielęgniarka zawsze zatwierdza |
| Zarządzanie ryzykiem (ISO 14971) | Reguły bezpieczeństwa poza LLM |
| Walidacja kliniczna | Pilotaż obserwacyjny jako pierwszy krok |

### Rekomendowana ścieżka wdrożenia

1. **Pilotaż obserwacyjny** — system działa równolegle, nie wpływa na decyzje, zbiera dane o zgodności z pielęgniarką
2. **Walidacja kliniczna** — analiza zgodności na danych pilotażowych
3. **Certyfikacja MDR** — jednostka notyfikowana, Design History File
4. **Wdrożenie produkcyjne** — z pełnym systemem audytu

---

## 11. Kierunki rozwoju

- **NLP na notatkach triażowych** — główny potencjał wzrostu QWK powyżej sufitu modeli tabelarycznych
- **Walidacja na polskich danych TOPSOR** — Yale to populacja i organizacja SOR z USA (inne skargi
  główne, inny wzorzec braków pomiarów); przed wdrożeniem konieczna ocena na danych z polskiego SOR
- **MedGemma dla obrazów** — RTG, EKG, zdjęcia ran gdy dane staną się dostępne
- **Federated learning** — trening na danych z wielu SOR-ów bez centralizacji danych
- **Continuous learning** — aktualizacja modeli na nowych danych z zachowaniem audytu

---

*Dokument zaktualizowany: 2026-10-19*  
*Status: Proof of Concept — nie wdrożony klinicznie*
