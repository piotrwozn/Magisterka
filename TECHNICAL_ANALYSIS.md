# SOR-AI: System Klasyfikacji Triażu MTS oparty na ML
## Pełna Analiza Techniczna Projektu Magisterskiego

---

## 1. OVERVIEW PROJEKTU

### 1.1 Cel
Zbudowanie systemu wspomagania decyzji triażowych na SOR, który:
- Przyjmuje dane pacjenta (objawy, wiek, vital signs, chief complaint itd.)
- Klasyfikuje pacjenta do jednej z 5 kategorii MTS (Red/Orange/Yellow/Green/Blue)
- Wyjaśnia decyzję na dwóch poziomach:
  - **Model-level**: dlaczego model ML wybrał tę kategorię (SHAP/LIME)
  - **Medical-level**: dlaczego medycznie to ma sens (LLM via Ollama)

### 1.2 Architektura High-Level
```
┌─────────────────────────────────────────────────────────┐
│                    INPUT PACJENTA                        │
│  (wiek, płeć, vital signs, chief complaint, historia)   │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
  ┌───────────────┐       ┌─────────────────┐
  │  STRUCTURED   │       │   TEXT BRANCH    │
  │   BRANCH      │       │   (HerBERT /    │
  │  (XGBoost /   │       │   BioClinBERT   │
  │   LightGBM)   │       │   fine-tuned)   │
  └───────┬───────┘       └────────┬────────┘
          │                        │
          └──────────┬─────────────┘
                     ▼
           ┌─────────────────┐
           │   LATE FUSION   │
           │   (MLP / Stack) │
           └────────┬────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────┐      ┌─────────────────┐
│  PREDYKCJA   │      │  EXPLAINABILITY  │
│  MTS 1-5     │      │                 │
│  (softmax +  │      │  ┌───────────┐  │
│  calibrated  │      │  │   SHAP    │  │
│  probs)      │      │  │  (model)  │  │
└──────────────┘      │  └───────────┘  │
                      │  ┌───────────┐  │
                      │  │  OLLAMA   │  │
                      │  │  (medical)│  │
                      │  └───────────┘  │
                      └─────────────────┘
```

### 1.3 Stack Technologiczny
| Komponent | Technologia |
|-----------|-------------|
| Język | Python 3.11+ |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| DL Framework | PyTorch + HuggingFace Transformers |
| XAI | shap, lime, dice-ml, alibi |
| LLM (medical reasoning) | Ollama (lokalne LLM - Mistral/Llama3/Medllama) |
| Data processing | pandas, polars, numpy |
| NLP (polski) | HerBERT (allegro/herbert-base-cased) |
| NLP (angielski/MIMIC) | Bio_ClinicalBERT |
| Walidacja | scikit-learn (StratifiedKFold), imbalanced-learn |
| Wizualizacja | matplotlib, seaborn, plotly |
| API/UI (opcjonalnie) | FastAPI + Streamlit/Gradio |
| Experiment tracking | MLflow / Weights & Biases |

---

## 2. DATASET: Yale EMMLC (admissionprediction)

### 2.1 Podstawowe informacje
| Parametr | Wartość |
|----------|---------|
| **Źródło** | Yale New Haven Health System (3 ED) |
| **Okres** | Marzec 2014 – Lipiec 2017 |
| **Rozmiar** | 560,486 wizyt × 972 zmienne |
| **Format** | R dataframe (.RData), eksportowalny do CSV (~1.7 GB) |
| **Licencja** | Open Access (CC-BY, cytowanie wymagane) |
| **GitHub** | https://github.com/yaleemmlc/admissionprediction |
| **Zenodo DOI** | 10.5281/zenodo.1308993 |
| **Kaggle mirror** | https://www.kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data |
| **Paper** | Hong WS, Haimovich AD, Taylor RA (2018) PLoS ONE 13(7):e0201016 |
| **Triage system** | ESI (Emergency Severity Index) 1-5 |
| **Unikalni pacjenci** | 202,953 |
| **Mediana wizyt/pacjent** | 1 (mean 2.76) |
| **Szpitale** | 1 Level I trauma center (~85k/rok), 1 community ED (~75k/rok), 1 suburban ED (~30k/rok) |

### 2.2 Plik danych
```
Results/5v_cleandf.RData
```
- R dataframe: 560,486 wierszy × 972 kolumn
- Eksport do CSV: `readr::write_csv(df, "yale_emmlc.csv")` → ~1.7 GB
- Alternatywnie: Kaggle mirror ma wersję CSV gotową do pobrania

### 2.3 Rozkład ESI (= nasz target po mapowaniu na MTS)
| ESI Level | MTS Odpowiednik | Kolor | % wizyt | Admission rate |
|-----------|-----------------|-------|---------|----------------|
| ESI-1 | Immediate | 🔴 Red | ~1-2% | 85.6% |
| ESI-2 | Very Urgent | 🟠 Orange | ~10-15% | 55.0% |
| ESI-3 | Urgent | 🟡 Yellow | ~45-50% | 29.1% |
| ESI-4 | Standard | 🟢 Green | ~25-30% | 2.2% |
| ESI-5 | Non-Urgent | 🔵 Blue | ~5-8% | 0.4% |

**UWAGA**: Dataset zawiera TYLKO wizyty zakończone admission lub discharge.
Wykluczone: transfer, AMA (against medical advice), eloped.
Overall admission rate: 29.7%.

### 2.4 Kategorie zmiennych (972 total)

#### A. TRIAGE VARIABLES (dostępne w momencie triażu)
Te zmienne są kluczowe — to one symulują dane wejściowe z SOR:

| Kategoria | Opis | Przykładowe zmienne | Ile zmiennych |
|-----------|------|---------------------|---------------|
| **ESI Level** | Poziom triażu 1-5 | `esi` | 1 |
| **Demographics** | Wiek, płeć, rasa, ubezpieczenie, zatrudnienie | `age`, `sex`, `race_*`, `insurance_*`, `employment_*` | ~15-20 |
| **Arrival info** | Tryb przyjazdu, pora dnia, dzień tygodnia, miesiąc, ED lokalizacja | `arrivalmode`, `arrivalmonth`, `arrivalday`, `arrivalhour_bin` | ~8 |
| **Triage vitals** | Vital signs przy triażu | `triage_sbp`, `triage_dbp`, `triage_pulse`, `triage_resp`, `triage_o2sat`, `triage_temp`, `triage_o2device`, `triage_pain` | 8 |
| **Chief complaint** | Top 200 najczęstszych + "other" bin | `cc_chestpain`, `cc_abdominalpain`, `cc_shortnessofbreath`, ... (binarne flagi) | ~201 |

#### B. PATIENT HISTORY VARIABLES (z EHR, przed wizytą)
| Kategoria | Opis | Ile zmiennych |
|-----------|------|---------------|
| **Past Medical History (PMH)** | ICD-9 → AHRQ CCS categories (binarne) | ~280 |
| **Outpatient Medications** | Liczba leków w 48 kategoriach terapeutycznych | 48 |
| **Prior surgeries/procedures** | Liczba procedur z EHR | ~10 |
| **Historical vitals** | Last/min/max/median vital signs z poprzednich wizyt (rok) | ~28 |
| **Historical labs** | Last/min/max/median 150 najczęstszych lab testów | ~300+ |
| **Previous ED usage** | Liczba wizyt, wynik poprzedniej wizyty, czas od ostatniej | ~20 |
| **Previous imaging** | Liczba zamówień RTG, CT, MRI, EKG, USG w ciągu roku | ~36 |

#### C. OUTCOME / TARGET
| Zmienna | Opis |
|---------|------|
| `disposition` | Binary: admission (1) vs discharge (0) — oryginalny target |
| `esi` | ESI level 1-5 — **NASZ TARGET po mapowaniu na MTS** |

### 2.5 Jak załadować dane (Python)

```python
# === OPCJA 1: Z pliku .RData (wymaga pyreadr) ===
import pyreadr

result = pyreadr.read_r('5v_cleandf.RData')
df = result[list(result.keys())[0]]  # wyciągnij dataframe
print(f"Shape: {df.shape}")  # (560486, 972)
print(f"Columns: {df.columns.tolist()[:20]}")
print(f"ESI distribution:\n{df['esi'].value_counts().sort_index()}")

# === OPCJA 2: Z CSV (Kaggle lub eksport z R) ===
import pandas as pd

df = pd.read_csv('yale_emmlc.csv', low_memory=False)

# === OPCJA 3: Z Kaggle API ===
# pip install kaggle
# kaggle datasets download -d maalona/hospital-triage-and-patient-history-data
```

### 2.6 Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── 1. LOAD & BASIC CLEAN ───
df = pd.read_csv('yale_emmlc.csv', low_memory=False)

# ─── 2. CREATE MTS TARGET (mapowanie ESI → MTS) ───
esi_to_mts = {
    1: 'Red',      # Immediate
    2: 'Orange',   # Very Urgent
    3: 'Yellow',   # Urgent
    4: 'Green',    # Standard
    5: 'Blue'      # Non-Urgent
}
esi_to_mts_numeric = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

df['mts_color'] = df['esi'].map(esi_to_mts)
df['mts_numeric'] = df['esi'].map(esi_to_mts_numeric)

# ─── 3. DEFINE FEATURE GROUPS ───
TRIAGE_VITALS = [
    'triage_sbp', 'triage_dbp', 'triage_pulse', 
    'triage_resp', 'triage_o2sat', 'triage_temp', 'triage_pain'
]

DEMOGRAPHICS = [col for col in df.columns if col.startswith(('age', 'sex', 'race_', 'ethnic_', 'lang_', 'insurance_', 'employment_'))]

ARRIVAL = [col for col in df.columns if col.startswith(('arrivalmode', 'arrivalmonth', 'arrivalday', 'arrivalhour'))]

CHIEF_COMPLAINTS = [col for col in df.columns if col.startswith('cc_')]

PAST_MEDICAL = [col for col in df.columns if col.startswith('pmh_')]  # AHRQ CCS categories

MEDICATIONS = [col for col in df.columns if col.startswith('med_')]

HISTORICAL_VITALS = [col for col in df.columns if col.startswith(('prev_sbp', 'prev_dbp', 'prev_pulse', 'prev_resp', 'prev_o2', 'prev_temp'))]

HISTORICAL_LABS = [col for col in df.columns if col.startswith(('lab_', 'prev_lab'))]

ED_USAGE = [col for col in df.columns if col.startswith(('n_ed', 'n_admit', 'prev_dispo'))]

# ─── 4. FEATURE SETS FOR EXPERIMENTS ───
# Set A: Only Triage (co ma pielęgniarka na triażu)
FEATURES_TRIAGE_ONLY = TRIAGE_VITALS + DEMOGRAPHICS + ARRIVAL + CHIEF_COMPLAINTS

# Set B: Triage + Patient History (pełny obraz)
FEATURES_FULL = (FEATURES_TRIAGE_ONLY + PAST_MEDICAL + MEDICATIONS + 
                 HISTORICAL_VITALS + HISTORICAL_LABS + ED_USAGE)

# Set C: Top variables (z analizy information gain)
# ESI, demographics, hospital usage, outpatient medications
FEATURES_TOP = ['esi'] + DEMOGRAPHICS + ED_USAGE + MEDICATIONS

print(f"Triage-only features: {len(FEATURES_TRIAGE_ONLY)}")
print(f"Full features: {len(FEATURES_FULL)}")

# ─── 5. HANDLE MISSING VALUES ───
# Vital signs: median imputation
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
df[TRIAGE_VITALS] = num_imputer.fit_transform(df[TRIAGE_VITALS])

# Binary/count features: fill with 0 (absence = no)
binary_cols = CHIEF_COMPLAINTS + PAST_MEDICAL + MEDICATIONS
df[binary_cols] = df[binary_cols].fillna(0)

# ─── 6. TRAIN/VAL/TEST SPLIT (TEMPORAL SPLIT - Podejście Doktoranckie) ───
# Zamiast prostego podziału losowego (który sztucznie zawyża wyniki i ignoruje 
# sezonowość), stosujemy podział CZASOWY (np. train: 2014-2016, test: 2017).
# Udowadnia to, że system działa w realnych warunkach zmienności sezonowej 
# (np. epidemie grypy w zimie) i potrafi generalizować w czasie. Jest to poziom doktorancki.

# Zakładając posortowanie po czasie przybycia lub wyciągnięcie roku/miesiąca:
# train_df = df[df['arrival_year'] <= 2016]
# test_df = df[df['arrival_year'] == 2017]

# Dla celów demonstracyjnych podział chronologiczny bez tasowania:
X = df[FEATURES_TRIAGE_ONLY]
y = df['mts_numeric']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=False  # UWAGA: shuffle=False dla podziału czasowego!
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.111, shuffle=False
)
# Wynik: Model trenowany na przeszłości, weryfikowany na przyszłości.

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
print(f"Class distribution (train):\n{y_train.value_counts().sort_index()}")
```

---

## 3. MODELE DO WYTRENOWANIA

### 3.1 Model 1: XGBoost (structured branch — BASELINE + PRIMARY)

```python
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

# ─── Oblicz wagi klas (asymetryczne — Red/Orange ważniejsze) ───
# Custom cost matrix: undertriage koszuje więcej niż overtriage
COST_MATRIX = {
    # (true, predicted) → cost
    # Undertriage (np. Red classified as Green) = VERY HIGH COST
    # Overtriage (np. Green classified as Orange) = low cost
}

# Prostsze podejście: class weights
sample_weights = compute_sample_weight('balanced', y_train)

# ─── XGBoost z optymalnymi hiperparametrami ───
params = {
    'objective': 'multi:softprob',
    'num_class': 5,
    'eval_metric': ['mlogloss', 'merror'],
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'tree_method': 'gpu_hist',  # GPU acceleration
    'random_state': 42,
    'early_stopping_rounds': 50,
}

model_xgb = xgb.XGBClassifier(**params)
model_xgb.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=100
)
```

**Hyperparameter tuning** (Optuna):
```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    
    model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=5,
        tree_method='gpu_hist', random_state=42,
        early_stopping_rounds=50, **params
    )
    model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_val, y_val)], verbose=0)
    
    y_pred = model.predict(X_val)
    return cohen_kappa_score(y_val, y_pred, weights='quadratic')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 3.2 Model 2: LightGBM (structured branch — COMPARISON)

```python
import lightgbm as lgb

model_lgbm = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=5,
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model_lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

### 3.3 Model 3: Random Forest (interpretable baseline)

```python
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model_rf.fit(X_train, y_train)
```

### 3.4 Model 4: Explainable Boosting Machine (inherently interpretable)

```python
from interpret.glassbox import ExplainableBoostingClassifier

model_ebm = ExplainableBoostingClassifier(
    max_bins=256,
    interactions=10,
    learning_rate=0.01,
    min_samples_leaf=5,
    random_state=42
)

model_ebm.fit(X_train, y_train)

# EBM jest inherently interpretable ("szklana skrzynka") — każdy feature ma wizualizowalną funkcję.
# UWAGA BADAWCZA: Porównanie EBM z modelem "czarnej skrzynki" (XGBoost + SHAP) 
# to świetny materiał na cały rozdział badawczy. EBM jest niemal tak dokładny jak XGBoost, 
# a z natury interpretowalny, co w medycynie jest bezcenne.
from interpret import show
ebm_global = model_ebm.explain_global()
show(ebm_global)
```

### 3.5 Model 5: Fine-tuned BERT na Chief Complaints (text branch)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch

# Dla datasetu Yale EMMLC: chief complaints są już zbinaryzowane
# (201 kolumn cc_*), więc BERT nie jest bezpośrednio potrzebny.
#
# ALE: jeśli chcesz przetwarzać RAW TEXT chief complaints
# (np. z polskiego SOR), użyj HerBERT:

MODEL_NAME = "allegro/herbert-base-cased"  # Polish BERT
# Alternatywnie dla angielskiego tekstu z MIMIC:
# MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_bert = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=5
)

# Fine-tuning config
training_args = TrainingArguments(
    output_dir='./results_bert',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    fp16=True,
    logging_steps=100,
)

# UWAGA: Dla Yale EMMLC z binaryzowanymi chief complaints,
# lepiej jest użyć cc_* kolumn jako features w XGBoost.
# BERT ma sens gdy masz surowy tekst chief complaint.
```

### 3.6 Model 6: Late Fusion (hybrid)

```python
import torch
import torch.nn as nn

class TriageFusionModel(nn.Module):
    """
    Late fusion: łączy output z XGBoost leaf features
    z BERT [CLS] embeddings (jeśli masz raw text).
    
    Dla Yale EMMLC bez raw text:
    łączy XGBoost predictions + structured features.
    """
    def __init__(self, xgb_dim=5, struct_dim=50, hidden_dim=128, n_classes=5):
        super().__init__()
        input_dim = xgb_dim + struct_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, xgb_probs, struct_features):
        combined = torch.cat([xgb_probs, struct_features], dim=1)
        return self.classifier(combined)

# Stacking approach (prostsze, polecane na start):
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stacking_model = StackingClassifier(
    estimators=[
        ('xgb', model_xgb),
        ('lgbm', model_lgbm),
        ('rf', model_rf),
    ],
    final_estimator=LogisticRegression(
        multi_class='multinomial', max_iter=1000, C=1.0
    ),
    cv=5,
    n_jobs=-1
)

stacking_model.fit(X_train, y_train)
```

---

## 4. CLASS IMBALANCE — STRATEGIA

### 4.1 Problem
```
Red (ESI-1):    ~1-2%   → KRYTYCZNY do wykrycia
Orange (ESI-2): ~10-15% → Ważny
Yellow (ESI-3): ~45-50% → Dominujący
Green (ESI-4):  ~25-30% → Częsty
Blue (ESI-5):   ~5-8%   → Rzadki
```

**UWAGA BADAWCZA**: Kategoria "Red" to tylko 1-2% danych. Oznacza to, że naiwny model może mieć aż 98% ogólnego "Accuracy" (dokładności), a jednocześnie kompletnie ignorować pacjentów umierających. Dlatego w tym projekcie "Accuracy" jako metryka ewaluacji jest absolutnie bezużyteczna.

### 4.2 Podejście wielopoziomowe

```python
# ─── LEVEL 1: Class weights ───
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Custom asymmetric weights (undertriage > overtriage)
CUSTOM_WEIGHTS = {
    0: 10.0,   # Red    — highest priority, must not miss
    1: 5.0,    # Orange — high priority
    2: 1.0,    # Yellow — baseline
    3: 2.0,    # Green  — moderate (small class)
    4: 3.0     # Blue   — moderate (small class)
}

# ─── LEVEL 2: Focal Loss (dla PyTorch modeli) ───
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights tensor
    
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ─── LEVEL 3: SMOTE-NC (jeśli potrzebne) ───
from imblearn.over_sampling import SMOTENC

# Identyfikuj kolumny kategoryczne
categorical_mask = [col in (CHIEF_COMPLAINTS + PAST_MEDICAL) for col in FEATURES_TRIAGE_ONLY]

smote = SMOTENC(
    categorical_features=categorical_mask,
    sampling_strategy={0: 20000, 4: 20000},  # oversample Red i Blue
    random_state=42,
    k_neighbors=5
)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ─── LEVEL 4: Ordinal cost-sensitive loss ───
# Undertriage penalizacja: dystans ordinalny
def ordinal_cost(y_true, y_pred_class):
    """
    Asymmetric cost: undertriage (predicting lower urgency) 
    costs more than overtriage.
    """
    diff = y_true - y_pred_class  # positive = undertriage
    cost = np.where(diff > 0, diff * 3.0, abs(diff) * 1.0)  # 3x penalty for undertriage
    return cost.mean()
```

### 4.3 Metryki ewaluacji

```python
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    cohen_kappa_score, roc_auc_score
)

def full_evaluation(y_true, y_pred, y_proba, class_names=['Red','Orange','Yellow','Green','Blue']):
    """Pełna ewaluacja modelu triażowego."""
    
    results = {}
    
    # 1. Classification report
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 2. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # 3. Quadratic Weighted Kappa (najważniejsza metryka dla ordinalnej klasyfikacji)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    results['quadratic_weighted_kappa'] = qwk
    print(f"\nQuadratic Weighted Kappa: {qwk:.4f}")
    
    # 4. Per-class AUC-ROC (One-vs-Rest)
    auc_ovr = roc_auc_score(y_true, y_proba, multi_class='ovr', average=None)
    for i, name in enumerate(class_names):
        results[f'auc_{name}'] = auc_ovr[i]
        print(f"AUC {name}: {auc_ovr[i]:.4f}")
    
    # 5. Macro AUC
    auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    results['auc_macro'] = auc_macro
    print(f"\nMacro AUC: {auc_macro:.4f}")
    
    # 6. UNDERTRIAGE RATE (CRITICAL SAFETY METRIC!)
    # Należy to mocno wyeksponować w pracy. To najważniejsza metryka bezpieczeństwa w medycynie ratunkowej:
    # Zawsze lepiej wysłać zdrowego pacjenta na niepotrzebne badania (overtriage), 
    # niż przeoczyć zawał czy udar (undertriage).
    # Obliczamy pacjentów Red/Orange błędnie zaklasyfikowanych jako Yellow/Green/Blue:
    high_acuity_mask = (y_true <= 1)  # Red (0) or Orange (1)
    if high_acuity_mask.sum() > 0:
        undertriage_rate = (y_pred[high_acuity_mask] > 1).mean()
        results['undertriage_rate'] = undertriage_rate
        print(f"\nUNDERTRIAGE RATE (Krytyczna metryka bezpieczeństwa!): {undertriage_rate:.4f}")
    
    # 7. Overtriage rate
    low_acuity_mask = (y_true >= 3)  # Green (3) or Blue (4)
    if low_acuity_mask.sum() > 0:
        overtriage_rate = (y_pred[low_acuity_mask] <= 1).mean()
        results['overtriage_rate'] = overtriage_rate
        print(f"OVERTRIAGE RATE (Green/Blue → Red/Orange): {overtriage_rate:.4f}")
    
    return results
```

---

## 5. EXPLAINABILITY (XAI) — DWIE WARSTWY

### 5.1 Warstwa 1: Model Explainability (SHAP)

```python
import shap

# ─── TreeSHAP dla XGBoost (szybki, dokładny) ───
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)
# shap_values: list of 5 arrays (one per class), each shape (n_test, n_features)

# ─── Per-patient explanation ───
def explain_patient(patient_idx, X_data, shap_vals, feature_names, class_names):
    """Generuj wyjaśnienie dla pojedynczego pacjenta."""
    
    predicted_class = model_xgb.predict(X_data.iloc[[patient_idx]])[0]
    probs = model_xgb.predict_proba(X_data.iloc[[patient_idx]])[0]
    
    explanation = {
        'predicted_class': class_names[predicted_class],
        'probabilities': dict(zip(class_names, probs.tolist())),
        'top_features_for': {},
        'top_features_against': {},
    }
    
    # Top features pushing TOWARD predicted class
    sv = shap_vals[predicted_class][patient_idx]
    feature_importance = list(zip(feature_names, sv, X_data.iloc[patient_idx].values))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    explanation['top_features_for'] = [
        {'feature': f, 'shap_value': float(s), 'patient_value': float(v)}
        for f, s, v in feature_importance[:10] if s > 0
    ]
    
    explanation['top_features_against'] = [
        {'feature': f, 'shap_value': float(s), 'patient_value': float(v)}
        for f, s, v in feature_importance[:10] if s < 0
    ]
    
    return explanation

# ─── SHAP visualizations ───
# Global: feature importance across all classes
shap.summary_plot(shap_values, X_test, class_names=['Red','Orange','Yellow','Green','Blue'])

# Per-patient: force plot
shap.force_plot(explainer.expected_value[predicted_class], 
                shap_values[predicted_class][idx], X_test.iloc[idx])

# ─── DiCE counterfactuals ───
import dice_ml

dice_data = dice_ml.Data(
    dataframe=pd.concat([X_train, y_train], axis=1),
    continuous_features=TRIAGE_VITALS,
    outcome_name='mts_numeric'
)

dice_model = dice_ml.Model(model=model_xgb, backend='sklearn')
dice_exp = dice_ml.Dice(dice_data, dice_model, method='random')

# "Co by się musiało zmienić, żeby pacjent był Orange zamiast Yellow?"
counterfactuals = dice_exp.generate_counterfactuals(
    query_instances=X_test.iloc[[patient_idx]],
    total_CFs=3,
    desired_class=1  # Orange
)
counterfactuals.visualize_as_dataframe()
```

### 5.2 Warstwa 2: Medical Reasoning (Ollama LLM)

```python
import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"

def get_medical_explanation(
    patient_data: dict,
    predicted_class: str,
    shap_explanation: dict,
    model_name: str = "mistral"  # lub "medllama2", "llama3", "gemma2"
):
    """
    Generuj medyczne wyjaśnienie triażu za pomocą lokalnego LLM (Ollama).
    
    Łączy dane pacjenta + wyjaśnienie SHAP → medyczny opis.
    """
    
    prompt = f"""Jesteś doświadczonym lekarzem medycyny ratunkowej pracującym na polskim SOR.
Na podstawie poniższych danych pacjenta i wyników modelu ML, wyjaśnij w języku polskim 
dlaczego pacjent został zaklasyfikowany do kategorii triażu MTS.

## Dane pacjenta:
- Wiek: {patient_data.get('age', 'N/A')}
- Płeć: {patient_data.get('sex', 'N/A')}
- Ciśnienie skurczowe: {patient_data.get('triage_sbp', 'N/A')} mmHg
- Ciśnienie rozkurczowe: {patient_data.get('triage_dbp', 'N/A')} mmHg
- Tętno: {patient_data.get('triage_pulse', 'N/A')} bpm
- Saturacja O2: {patient_data.get('triage_o2sat', 'N/A')}%
- Temperatura: {patient_data.get('triage_temp', 'N/A')}°C
- Częstość oddechów: {patient_data.get('triage_resp', 'N/A')}/min
- Ból (skala 1-10): {patient_data.get('triage_pain', 'N/A')}
- Główna skarga: {patient_data.get('chief_complaint', 'N/A')}

## Predykcja modelu ML:
- Przydzielona kategoria: **{predicted_class}**
- Top cechy (SHAP):
{json.dumps(shap_explanation['top_features_for'][:5], indent=2, ensure_ascii=False)}

## Twoje zadanie:
1. Wyjaśnij MEDYCZNE powody tej klasyfikacji (np. "podwyższone tętno w połączeniu z bólem w klatce piersiowej może wskazywać na ostry zespół wieńcowy")
2. Wskaż potencjalne zagrożenia i na co zwrócić uwagę
3. Oceń czy klasyfikacja ML wydaje się zasadna z punktu widzenia klinicznego
4. Zasugeruj potrzebne badania diagnostyczne

Odpowiadaj zwięźle (max 300 słów), po polsku, profesjonalnym językiem medycznym.
"""
    
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # niska temperatura dla spójności medycznej
                "top_p": 0.9,
                "num_predict": 512,
            }
        }
    )
    
    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error: {response.status_code} - {response.text}"


# ─── UŻYCIE ───
patient = X_test.iloc[0].to_dict()
predicted = model_xgb.predict(X_test.iloc[[0]])[0]
class_names = ['Red', 'Orange', 'Yellow', 'Green', 'Blue']

shap_exp = explain_patient(0, X_test, shap_values, X_test.columns.tolist(), class_names)
medical_explanation = get_medical_explanation(patient, class_names[predicted], shap_exp)
print(medical_explanation)
```

### 5.3 MTS Flowchart Rule Engine (dodatkowa warstwa kliniczna)

```python
# Enkodowanie reguł MTS jako decision tree / rule engine
# MTS ma 52 presentational flowcharts z klinicznymi dyskryminatorami

MTS_VITAL_THRESHOLDS = {
    'Red': {
        'description': 'Natychmiastowy',
        'max_wait_minutes': 0,
        'discriminators': {
            'airway_compromise': True,
            'breathing_inadequate': True,
            'shock': True,  # SBP < 90
            'unresponsive': True,  # GCS ≤ 8
            'seizure_active': True,
            'severe_pain': 10,
        },
        'vital_rules': {
            'triage_sbp': lambda x: x < 90,
            'triage_o2sat': lambda x: x < 85,
            'triage_pulse': lambda x: x > 150 or x < 40,
            'triage_resp': lambda x: x > 35 or x < 8,
            'triage_temp': lambda x: x > 41.0 or x < 32.0,
        }
    },
    'Orange': {
        'description': 'Bardzo pilny',
        'max_wait_minutes': 10,
        'vital_rules': {
            'triage_sbp': lambda x: x < 100,
            'triage_o2sat': lambda x: x < 92,
            'triage_pulse': lambda x: x > 130 or x < 50,
            'triage_resp': lambda x: x > 30,
            'triage_temp': lambda x: x > 40.0 or x < 34.0,
            'triage_pain': lambda x: x >= 8,
        }
    },
    'Yellow': {
        'description': 'Pilny',
        'max_wait_minutes': 60,
        'vital_rules': {
            'triage_sbp': lambda x: 100 <= x < 110,
            'triage_o2sat': lambda x: 92 <= x < 95,
            'triage_pulse': lambda x: 100 < x <= 130,
            'triage_temp': lambda x: 38.5 <= x < 40.0,
            'triage_pain': lambda x: 5 <= x < 8,
        }
    },
    'Green': {
        'description': 'Standardowy',
        'max_wait_minutes': 120,
        'vital_rules': {
            # Vital signs w normie, ale wymaga oceny
            'triage_pain': lambda x: 2 <= x < 5,
        }
    },
    'Blue': {
        'description': 'Niepilny',
        'max_wait_minutes': 240,
        'vital_rules': {
            # Vital signs w normie, minor complaint
            'triage_pain': lambda x: x < 2,
        }
    }
}

def rule_based_triage(patient_vitals: dict) -> dict:
    """
    Reguły MTS na vital signs — uzupełnienie modelu ML.
    Zwraca: suggested_category + triggered_rules (do wyjaśnień).
    """
    triggered = []
    
    for category in ['Red', 'Orange', 'Yellow', 'Green', 'Blue']:
        rules = MTS_VITAL_THRESHOLDS[category]['vital_rules']
        matches = []
        
        for vital_name, rule_fn in rules.items():
            if vital_name in patient_vitals and patient_vitals[vital_name] is not None:
                try:
                    if rule_fn(patient_vitals[vital_name]):
                        matches.append({
                            'vital': vital_name,
                            'value': patient_vitals[vital_name],
                            'category': category,
                            'description': MTS_VITAL_THRESHOLDS[category]['description']
                        })
                except:
                    pass
        
        if matches:
            triggered.extend(matches)
            return {
                'suggested_category': category,
                'triggered_rules': matches,
                'max_wait': MTS_VITAL_THRESHOLDS[category]['max_wait_minutes']
            }
    
    return {
        'suggested_category': 'Green',
        'triggered_rules': [],
        'max_wait': 120
    }
```

---

## 6. MAPOWANIE ESI → MTS — STRATEGIA I LIMITACJE

### 6.1 Problem
ESI i MTS to RÓŻNE systemy:
- **ESI**: bazuje na oczekiwanym zużyciu zasobów (resource-based)
- **MTS**: bazuje na objawach klinicznych i 52 flowchartach (symptom-based)
- **Zgodność**: Cohen's κ ≈ 0.51, Spearman's ρ ≈ 0.49

### 6.2 Strategie naprawcze

```python
# STRATEGIA 1: Proste mapowanie ESI → MTS (baseline)
# ESI 1 → Red, ESI 2 → Orange, ESI 3 → Yellow, ESI 4 → Green, ESI 5 → Blue
# Problem: ESI 3 to "worek" (~50% pacjentów), MTS byłby bardziej precyzyjny

# STRATEGIA 2: ESI + vital signs discriminators (REKOMENDOWANA STRATEGIA GŁÓWNA)
# Używamy ESI jako startowego labela, ale WERYFIKUJEMY za pomocą MTS vital thresholds.
# To kluczowy punkt obrony metodologicznej! Ponieważ zgodność ESI i MTS to tylko ~0.51,
# recenzent może zapytać: "Czy Twój model uczy się MTS, czy po prostu ESI pod inną nazwą?".
# Strategia 2 (dostosowanie na bazie dyskryminatorów parametrów życiowych) czyni mapowanie
# autentycznie "MTS-owym" i chroni przed zarzutem pójścia na łatwiznę.
def enhanced_mts_label(row):
    """
    Wzbogacone mapowanie: ESI + MTS vital sign discriminators.
    Np. ESI-3 z SpO2 < 92% → Orange (nie Yellow).
    """
    esi = row['esi']
    
    # Start z prostym mapowaniem
    mts = esi  # 1=Red, 2=Orange, 3=Yellow, 4=Green, 5=Blue
    
    # Apply MTS discriminators to UPGRADE (never downgrade)
    if esi >= 3:  # Yellow/Green/Blue
        # Check for Red discriminators
        if (row.get('triage_o2sat', 100) < 85 or 
            row.get('triage_sbp', 120) < 90 or
            row.get('triage_pulse', 80) > 150):
            mts = 1  # Upgrade to Red
        
        # Check for Orange discriminators  
        elif (row.get('triage_o2sat', 100) < 92 or
              row.get('triage_sbp', 120) < 100 or
              row.get('triage_pulse', 80) > 130 or
              row.get('triage_temp', 37) > 40.0 or
              row.get('triage_pain', 0) >= 8):
            mts = min(mts, 2)  # Upgrade to at least Orange
    
    return mts

# STRATEGIA 3: Multi-task learning (najlepsza, ale trudniejsza)
# Target 1: ESI level (supervision)
# Target 2: Clinical outcome (ICU admission, mortality) — proxy for true urgency
# Target 3: MTS vital-sign rules (weak supervision)
# → Model uczy się jednocześnie ESI + outcome + rules
```

### 6.3 Walidacja mapowania

```python
# Kluczowe: porównaj ESI-based prediction z clinical outcomes
# Jeśli model poprawnie predykuje ICU admission / mortality,
# to niezależnie od labela (ESI vs MTS) — jest klinicznie użyteczny

OUTCOME_PROXIES = {
    'hospital_admission': 'disposition',  # dostępny w Yale EMMLC
    # Poniższe wymagają MIMIC-IV linkage:
    # 'icu_admission': ...,
    # '72h_return': ...,
    # 'mortality_30d': ...,
}
```

---

## 7. WALIDACJA I RAPORTOWANIE

### 7.1 Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_results = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_fold_train, y_fold_train)
    
    y_pred = model.predict(X_fold_val)
    y_proba = model.predict_proba(X_fold_val)
    
    fold_results = full_evaluation(y_fold_val, y_pred, y_proba)
    fold_results['fold'] = fold
    cv_results.append(fold_results)

# Report: mean ± std across folds
cv_df = pd.DataFrame(cv_results)
print(cv_df.describe())
```

### 7.2 TRIPOD+AI Checklist
Dołącz do pracy jako appendix — 27-punktowa checklista raportowania modeli predykcyjnych w medycynie (BMJ, April 2024). Obejmuje: populację, definicję outcome, obsługę missing data, walidację, performance measures.

### 7.3 Key Metrics to Report
1. **Quadratic Weighted Kappa** (QWK) — główna metryka
2. **Macro AUC-ROC** — overall performance
3. **Per-class AUC-ROC** — szczególnie Red i Orange
4. **Undertriage rate** — BEZPIECZEŃSTWO
5. **Overtriage rate** — efektywność
6. **Sensitivity/Specificity per class**
7. **5×5 Confusion Matrix**
8. **Calibration plot** (reliability diagram)

---

## 8. STRUKTURA PROJEKTU

```
sor-ai-triage/
├── README.md
├── TECHNICAL_ANALYSIS.md          ← ten plik
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── raw/
│   │   └── 5v_cleandf.RData      ← Yale EMMLC dataset
│   ├── processed/
│   │   ├── yale_emmlc.csv         ← CSV export
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── external/
│       └── mts_flowcharts.json    ← MTS rules encoded
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py           ← RData/CSV loading
│   │   ├── preprocessing.py       ← Feature engineering
│   │   ├── esi_mts_mapping.py     ← ESI→MTS conversion
│   │   └── splits.py              ← Train/val/test splits
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py       ← XGBoost training & tuning
│   │   ├── lightgbm_model.py      ← LightGBM training
│   │   ├── random_forest.py       ← RF baseline
│   │   ├── ebm_model.py           ← Explainable Boosting Machine
│   │   ├── bert_model.py          ← HerBERT/BioClinBERT (jeśli raw text)
│   │   ├── fusion_model.py        ← Late fusion / stacking
│   │   └── train.py               ← Unified training pipeline
│   │
│   ├── explain/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py      ← SHAP explanations
│   │   ├── lime_explainer.py      ← LIME local explanations
│   │   ├── dice_counterfactual.py ← "What-if" scenarios
│   │   ├── mts_rules.py           ← MTS flowchart rule engine
│   │   └── ollama_medical.py      ← LLM medical reasoning
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             ← Full evaluation suite
│   │   ├── cross_validation.py    ← Stratified k-fold CV
│   │   └── visualizations.py      ← Confusion matrices, ROC curves
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py              ← Hyperparameters, paths
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_explainability.ipynb
│   └── 05_results_analysis.ipynb
│
├── app/
│   ├── main.py                    ← FastAPI backend
│   ├── streamlit_app.py           ← Streamlit UI demo
│   └── templates/
│
├── models/                        ← Saved model artifacts
│   ├── xgb_best.json
│   ├── lgbm_best.txt
│   └── scaler.pkl
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── reports/
│
└── tests/
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_explainability.py
```

---

## 9. REQUIREMENTS

```
# requirements.txt
# Core ML
numpy>=1.24
pandas>=2.0
polars>=0.20
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
imbalanced-learn>=0.11
optuna>=3.5

# Deep Learning (opcjonalnie, dla BERT branch)
torch>=2.1
transformers>=4.35
tokenizers>=0.15

# Explainability
shap>=0.44
lime>=0.2
dice-ml>=0.11
interpret>=0.5
alibi>=0.9

# Data loading
pyreadr>=0.5          # read .RData files
pyarrow>=14.0         # parquet support

# Visualization
matplotlib>=3.8
seaborn>=0.13
plotly>=5.18

# API / UI
fastapi>=0.104
uvicorn>=0.24
streamlit>=1.28
requests>=2.31        # Ollama API calls

# Experiment tracking
mlflow>=2.9
# wandb>=0.16        # alternatywa

# Utils
tqdm>=4.66
pyyaml>=6.0
python-dotenv>=1.0
```

---

## 10. OLLAMA SETUP

```bash
# Instalacja Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pobierz modele (wybierz jeden lub kilka):
ollama pull mistral          # 7B, dobry ogólnie, szybki
ollama pull llama3           # 8B, Meta, dobry po polsku
ollama pull medllama2        # 7B, medical fine-tune
ollama pull gemma2:9b        # 9B, Google, dobry po polsku

# Sprawdź czy działa:
ollama run mistral "Wyjaśnij po polsku co to jest Manchester Triage System"

# API jest domyślnie na http://localhost:11434
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "What is MTS triage?",
  "stream": false
}'
```

---

## 11. CYTOWANIE

```bibtex
@article{hong2018predicting,
  title={Predicting hospital admission at emergency department triage using machine learning},
  author={Hong, Woo Suk and Haimovich, Adrian Daniel and Taylor, R Andrew},
  journal={PLoS ONE},
  volume={13},
  number={7},
  pages={e0201016},
  year={2018},
  doi={10.1371/journal.pone.0201016}
}
```

---

## 12. TODO / ROADMAP

- [ ] Pobrać dataset Yale EMMLC (GitHub/Kaggle)
- [ ] Eksport .RData → CSV/Parquet
- [ ] EDA (Exploratory Data Analysis) — rozkłady, missing values, korelacje
- [ ] Feature engineering + ESI→MTS mapowanie
- [ ] Train baseline XGBoost (triage-only features)
- [ ] Train XGBoost (full features)
- [ ] Train LightGBM + RF + EBM (porównanie)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Implement class imbalance strategies
- [ ] 10-fold stratified CV
- [ ] SHAP explainability pipeline
- [ ] DiCE counterfactuals
- [ ] MTS rule engine
- [ ] Ollama LLM integration
- [ ] Streamlit demo app
- [ ] Wyniki + wizualizacje
- [ ] Napisać magisterkę
