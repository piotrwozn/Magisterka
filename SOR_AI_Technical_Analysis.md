# SOR-AI — Technical Analysis
## Multi-Agent Triage Decision Support System

**Autor:** Piotr  
**Data:** 2026-05-17  
**Wersja:** 1.0  
**Sprzęt docelowy:** Lokalny serwer (bez ograniczeń VRAM)

---

## 1. Cel systemu

System wspomagania decyzji triażowych oparty na architekturze multi-agent, łączący modele tabelaryczne ML z modelami językowymi (LLM) w celu klasyfikacji pacjentów według skali Manchester Triage System (MTS):

| Kategoria | Kolor | Czas do oceny |
|---|---|---|
| 1 | Czerwony (Red) | Natychmiastowy |
| 2 | Pomarańczowy (Orange) | 10 minut |
| 3 | Żółty (Yellow) | 30 minut |
| 4 | Zielony (Green) | 60 minut |
| 5 | Niebieski (Blue) | 120 minut |

System jest zaprojektowany jako **narzędzie wspomagające** — ostateczna decyzja zawsze należy do pielęgniarki triażowej.

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
│ RF            │    │                     │
│ EBM           │    │ "blady + spocony +  │
│    ↓          │    │  ból w klatce =     │
│ Stacking      │    │  ryzyko OZW"        │
│ LogReg        │    │                     │
│    ↓          │    │                     │
│ SHAP top 5    │    │                     │
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
│  Sugestia: CZERWONY (94%)                             │
│  Powód: temp 38.2 + HR 118 + obserwacja kliniczna    │
│  ⚠️ KONFLIKT: ensemble=Yellow, kliniczne=Red          │
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
    # ... pozostałe 220 cech

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

### 4.1 Ensemble 5 modeli

| Model | Rola w ensemblu | Kluczowa zaleta |
|---|---|---|
| XGBoost | Bezpieczeństwo Red/Blue | Critical miss 0.02%, tunowane wagi klas |
| LightGBM | Yellow/środkowe klasy | Leaf-wise splitting, lepszy recall Yellow |
| CatBoost | Diversyfikacja boostingu | Ordered boosting, natywne QWK jako eval metric |
| Random Forest | Jedyny bagging | Nieskorelowane błędy z boostingami, stabilność |
| EBM | Interpretowalność MDR | Natywne wyjaśnienia krzywych cech, certyfikowalny |

### 4.2 Tuning — Optuna

```python
# Strategia tuningu (wspólna dla wszystkich modeli)
sampler = optuna.samplers.TPESampler(
    seed=42,
    multivariate=True,    # modeluje korelacje między hiperparametrami
    group=True,           # conditional params (DART, GOSS)
    n_startup_trials=50,
    n_ei_candidates=48,
)

# 5-fold StratifiedKFold — stały seed, fair comparison
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cost-sensitive wagi klas jako hiperparametry Optuny
# Red i Blue dostają wyższe wagi (bezpieczeństwo)
class_weights = {
    0: trial.suggest_float("cw_red", 2.0, 25.0),
    1: trial.suggest_float("cw_orange", 1.0, 15.0),
    2: 1.0,  # Yellow — referencyjna
    3: trial.suggest_float("cw_green", 0.5, 5.0),
    4: trial.suggest_float("cw_blue", 0.5, 10.0),
}
```

### 4.3 Aktualne wyniki XGBoost (najlepszy run)

| Metryka | Wartość |
|---|---|
| QWK (test) | 0.8516 |
| CV QWK | 0.8766 |
| AUC-ROC Macro | 0.9711 |
| Undertriage Rate | 1.01% |
| Critical Miss Rate | 0.02% |
| Red F1 | 0.974 |
| Yellow F1 | 0.428 ← obszar do poprawy |

### 4.4 Stacking Meta-Learner

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# OOF predictions — 5 modeli × 5 klas = 25 features bazowych
# + cechy dodatkowe: pewność i niepewność każdego modelu
def build_meta_features(oof_probs: dict) -> np.ndarray:
    features = []
    for model_name, probs in oof_probs.items():
        features.extend(probs)                    # 5 prawdopodobieństw
        features.append(probs.max())              # pewność modelu
        features.append(probs.std())              # niepewność modelu
        features.append(probs.max() - sorted(probs)[-2])  # margin
    return np.array(features)  # 25 + 15 = 40 cech dla meta-learnera

# Meta-learner: LogReg — pełna interpretowalność, audytowalność MDR
meta_learner = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
```

**Dlaczego Logistic Regression jako meta-learner:**
- Pełna interpretowalność — wagi to liczby które audytor MDR może przeczytać
- Deterministyczny — ten sam input zawsze daje ten sam output
- Zero ryzyka overfittingu na OOF predictions
- Wymagany przez ISO 14971 (zarządzanie ryzykiem) — każda decyzja musi być wytłumaczalna

### 4.5 Conflict detection między modelami

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
Nie sugerujesz diagnozy — tylko kategorię ryzyka MTS i uzasadnienie.
"""

class ClinicalAssessment(BaseModel):
    mts_suggestion: int           # 1-5
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
    ensemble_category: int,      # 1-5
    medgemma_category: int,      # 1-5
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
Lepiej zawyżyć kategorię niż ją zaniżyć. Zgodne z metryką bezpieczeństwa: undertriage jest niedopuszczalny, overtriage jest akceptowalny.

### Prompt dla Qwen3 32B

```python
QWEN_SYSTEM = """
Jesteś systemem syntezującym decyzję triażową.
Otrzymujesz:
1. Wynik modeli ML (kategoria + pewność + SHAP top 5 cech)
2. Ocenę kliniczną MedGemma (kategoria + uzasadnienie)

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

| Metryka | Cel | XGBoost aktualnie |
|---|---|---|
| Critical Miss Rate | < 0.1% | 0.02% ✅ |
| Undertriage Rate | < 2% | 1.01% ✅ |
| Overtriage Rate | < 30% | 23.17% ✅ |

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
| Tabular ML | XGBoost, LightGBM, CatBoost, scikit-learn | najnowsze |
| Hyperparameter tuning | Optuna + MultivariateTPE | najnowsze |
| Meta-learner | scikit-learn LogisticRegression | najnowsze |
| Explainability | SHAP, InterpretML (EBM) | najnowsze |
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
| Interpretowalność decyzji | EBM + SHAP + LogReg meta-learner |
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

- **NLP na notatkach triażowych** — główny potencjał wzrostu QWK powyżej sufitu tabularycznego (~0.88)
- **MedGemma dla obrazów** — RTG, EKG, zdjęcia ran gdy dane staną się dostępne
- **Federated learning** — trening na danych z wielu SOR-ów bez centralizacji danych
- **Continuous learning** — aktualizacja modeli na nowych danych z zachowaniem audytu

---

*Dokument wygenerowany: 2026-05-17*  
*Status: Proof of Concept — nie wdrożony klinicznie*
