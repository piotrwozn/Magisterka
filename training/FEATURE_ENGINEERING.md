# Feature Engineering — SOR-AI Triage
## Kompletna lista cech do obliczenia

---

## 1. Indeksy Hemodynamiczne

### Shock Index (SI)
```python
df['shock_index'] = df['triage_vital_hr'] / df['triage_vital_sbp']
```
- **Norma:** < 0.7
- **Niepokojący:** 0.7–1.0
- **Krytyczny:** > 1.0 (wstrząs we wczesnej fazie)
- **Zastosowanie:** Wstrząs hipowolemiczny, krwotok, sepsa

### Modified Shock Index (MSI)
```python
df['modified_shock_index'] = df['triage_vital_hr'] / df['triage_vital_map']
# wymaga MAP — patrz niżej
```
- **Krytyczny:** > 1.4
- Lepszy predyktor niż SI dla pacjentów w podeszłym wieku

### Mean Arterial Pressure (MAP)
```python
df['pulse_pressure'] = df['triage_vital_sbp'] - df['triage_vital_dbp']
df['map'] = df['triage_vital_dbp'] + (df['pulse_pressure'] / 3)
```
- **Norma:** 70–100 mmHg
- **Krytyczny:** < 65 mmHg (perfuzja narządów zagrożona)
- **Zastosowanie:** Sepsa, wstrząs, nadciśnienie przełomowe

### Pulse Pressure (PP)
```python
df['pulse_pressure'] = df['triage_vital_sbp'] - df['triage_vital_dbp']
```
- **Norma:** 40–60 mmHg
- **Wąskie PP < 25 mmHg:** tamponada, wstrząs kardiogenny
- **Szerokie PP > 80 mmHg:** niedomykalność aortalna, sepsa

### Pulse Pressure Ratio (PPR)
```python
df['pulse_pressure_ratio'] = df['pulse_pressure'] / df['triage_vital_sbp']
```
- **Krytyczny:** < 0.25 (wąskie PP względem SBP)

---

## 2. Indeksy Oddechowe i Saturacji

### SpO2/FiO2 proxy (bez FiO2 — uproszczony)
```python
df['o2_hr_ratio'] = df['triage_vital_o2'] / df['triage_vital_hr']
```
- Niski SpO2 przy wysokim HR = hipoksja kompensowana

### Respiratory Rate × SpO2 interaction
```python
df['rr_o2_product'] = df['triage_vital_rr'] * df['triage_vital_o2']
```
- Niski produkt = respiratory failure

### Hipoksja flag
```python
df['hypoxia_flag'] = (df['triage_vital_o2'] < 94).astype(int)
df['severe_hypoxia_flag'] = (df['triage_vital_o2'] < 90).astype(int)
```

### Tachypnea flag
```python
df['tachypnea_flag'] = (df['triage_vital_rr'] > 20).astype(int)
df['severe_tachypnea_flag'] = (df['triage_vital_rr'] > 30).astype(int)
```

---

## 3. Indeksy Sepsy i SIRS

### SIRS Score (Systemic Inflammatory Response Syndrome)
```python
df['sirs_temp'] = (
    (df['triage_vital_temp'] > 38.0) | 
    (df['triage_vital_temp'] < 36.0)
).astype(int)

df['sirs_hr'] = (df['triage_vital_hr'] > 90).astype(int)
df['sirs_rr'] = (df['triage_vital_rr'] > 20).astype(int)

df['sirs_score'] = df['sirs_temp'] + df['sirs_hr'] + df['sirs_rr']
# SIRS ≥ 2 = podejrzenie sepsy
df['sirs_positive'] = (df['sirs_score'] >= 2).astype(int)
```
- **SIRS ≥ 2:** podejrzenie sepsy
- **SIRS ≥ 3:** wysokie ryzyko sepsy ciężkiej

### qSOFA Score (quick Sequential Organ Failure Assessment)
```python
df['qsofa_sbp'] = (df['triage_vital_sbp'] <= 100).astype(int)
df['qsofa_rr'] = (df['triage_vital_rr'] >= 22).astype(int)
# GCS nie mamy — używamy proxy z cc_alteredmentalstatus
df['qsofa_mental'] = df.get('cc_alteredmentalstatus', 0)

df['qsofa_score'] = df['qsofa_sbp'] + df['qsofa_rr'] + df['qsofa_mental']
# qSOFA ≥ 2 = wysokie ryzyko sepsy z dysfunkcją narządową
df['qsofa_positive'] = (df['qsofa_score'] >= 2).astype(int)
```

### NEWS2 proxy (National Early Warning Score)
```python
# Temperatura
df['news_temp'] = pd.cut(
    df['triage_vital_temp'],
    bins=[-np.inf, 35.0, 36.0, 38.0, 39.0, np.inf],
    labels=[3, 1, 0, 1, 2]
).astype(int)

# HR
df['news_hr'] = pd.cut(
    df['triage_vital_hr'],
    bins=[-np.inf, 40, 50, 90, 110, 130, np.inf],
    labels=[3, 1, 0, 1, 2, 3]
).astype(int)

# SBP
df['news_sbp'] = pd.cut(
    df['triage_vital_sbp'],
    bins=[-np.inf, 90, 100, 110, 219, np.inf],
    labels=[3, 2, 1, 0, 3]
).astype(int)

# RR
df['news_rr'] = pd.cut(
    df['triage_vital_rr'],
    bins=[-np.inf, 8, 11, 20, 24, np.inf],
    labels=[3, 1, 0, 2, 3]
).astype(int)

# SpO2
df['news_o2'] = pd.cut(
    df['triage_vital_o2'],
    bins=[-np.inf, 91, 93, 95, np.inf],
    labels=[3, 2, 1, 0]
).astype(int)

df['news2_score'] = (
    df['news_temp'] + df['news_hr'] + 
    df['news_sbp'] + df['news_rr'] + df['news_o2']
)
df['news2_high_risk'] = (df['news2_score'] >= 7).astype(int)
```

---

## 4. Interakcje Kliniczne

### Temperatura × HR (gorączka z tachykardią)
```python
df['temp_hr_interaction'] = df['triage_vital_temp'] * df['triage_vital_hr']
```
- Wysoka wartość = sepsa lub ciężka infekcja

### HR × SBP (kompensacja hemodynamiczna)
```python
df['hr_sbp_product'] = df['triage_vital_hr'] * df['triage_vital_sbp']
```
- Niski produkt = dekompensacja hemodynamiczna

### Temperatura odchylenie od normy
```python
df['temp_deviation'] = abs(df['triage_vital_temp'] - 37.0)
df['hypothermia_flag'] = (df['triage_vital_temp'] < 36.0).astype(int)
df['high_fever_flag'] = (df['triage_vital_temp'] >= 39.0).astype(int)
df['extreme_fever_flag'] = (df['triage_vital_temp'] >= 40.0).astype(int)
```

### Hipotensja flags
```python
df['hypotension_flag'] = (df['triage_vital_sbp'] < 100).astype(int)
df['severe_hypotension_flag'] = (df['triage_vital_sbp'] < 90).astype(int)
df['hypertension_flag'] = (df['triage_vital_sbp'] > 180).astype(int)
df['hypertensive_crisis_flag'] = (df['triage_vital_sbp'] > 200).astype(int)
```

### Tachykardia flags
```python
df['tachycardia_flag'] = (df['triage_vital_hr'] > 100).astype(int)
df['severe_tachycardia_flag'] = (df['triage_vital_hr'] > 130).astype(int)
df['bradycardia_flag'] = (df['triage_vital_hr'] < 50).astype(int)
```

---

## 5. Composite Risk Scores

### Vitals Instability Score (własny)
```python
df['vitals_instability'] = (
    df['shock_index_critical'] +      # SI > 1.0
    df['hypotension_flag'] +          # SBP < 100
    df['severe_tachycardia_flag'] +   # HR > 130
    df['severe_tachypnea_flag'] +     # RR > 30
    df['severe_hypoxia_flag'] +       # SpO2 < 90
    df['high_fever_flag']             # temp > 39
)
# 0 = stabilny, 6 = wszystkie parametry krytyczne
```

### Shock Index kategoryczny
```python
df['shock_index_normal'] = (df['shock_index'] < 0.7).astype(int)
df['shock_index_elevated'] = (
    (df['shock_index'] >= 0.7) & (df['shock_index'] < 1.0)
).astype(int)
df['shock_index_critical'] = (df['shock_index'] >= 1.0).astype(int)
```

### Krytyczne kombinacje
```python
# Klasyczna triada wstrząsu
df['shock_triad'] = (
    (df['triage_vital_sbp'] < 90) & 
    (df['triage_vital_hr'] > 110) & 
    (df['triage_vital_o2'] < 94)
).astype(int)

# Sepsa proxy
df['sepsis_proxy'] = (
    (df['sirs_positive'] == 1) & 
    (df['qsofa_sbp'] == 1)
).astype(int)

# Respiratory failure proxy  
df['resp_failure_proxy'] = (
    (df['triage_vital_o2'] < 92) & 
    (df['triage_vital_rr'] > 25)
).astype(int)
```

---

## 6. Features Czasowe i Demograficzne

### Wiek × ryzyko
```python
df['elderly_flag'] = (df['age'] >= 75).astype(int)
df['pediatric_flag'] = (df['age'] <= 14).astype(int)
df['age_risk_group'] = pd.cut(
    df['age'],
    bins=[0, 14, 30, 60, 75, np.inf],
    labels=[2, 0, 1, 2, 3]  # dzieci i starsi = wyższe ryzyko
).astype(int)
```

### Godzina przyjęcia (dobowy rytm SOR)
```python
df['night_arrival'] = (
    (df['arrivalhour_bin'] >= 22) | 
    (df['arrivalhour_bin'] <= 6)
).astype(int)
df['peak_hours'] = (
    (df['arrivalhour_bin'] >= 10) & 
    (df['arrivalhour_bin'] <= 14)
).astype(int)
```

---

## 7. Implementacja — pipeline

```python
import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # === HEMODYNAMICZNE ===
    df['pulse_pressure'] = df['triage_vital_sbp'] - df['triage_vital_dbp']
    df['map'] = df['triage_vital_dbp'] + (df['pulse_pressure'] / 3)
    df['shock_index'] = df['triage_vital_hr'] / df['triage_vital_sbp'].clip(1)
    df['modified_shock_index'] = df['triage_vital_hr'] / df['map'].clip(1)
    df['pulse_pressure_ratio'] = df['pulse_pressure'] / df['triage_vital_sbp'].clip(1)
    df['hr_sbp_product'] = df['triage_vital_hr'] * df['triage_vital_sbp']
    
    # === FLAGI HEMODYNAMICZNE ===
    df['shock_index_critical'] = (df['shock_index'] >= 1.0).astype(int)
    df['shock_index_elevated'] = ((df['shock_index'] >= 0.7) & (df['shock_index'] < 1.0)).astype(int)
    df['hypotension_flag'] = (df['triage_vital_sbp'] < 100).astype(int)
    df['severe_hypotension_flag'] = (df['triage_vital_sbp'] < 90).astype(int)
    df['hypertension_flag'] = (df['triage_vital_sbp'] > 180).astype(int)
    df['hypertensive_crisis_flag'] = (df['triage_vital_sbp'] > 200).astype(int)
    df['tachycardia_flag'] = (df['triage_vital_hr'] > 100).astype(int)
    df['severe_tachycardia_flag'] = (df['triage_vital_hr'] > 130).astype(int)
    df['bradycardia_flag'] = (df['triage_vital_hr'] < 50).astype(int)
    
    # === TEMPERATURA ===
    df['temp_deviation'] = abs(df['triage_vital_temp'] - 37.0)
    df['hypothermia_flag'] = (df['triage_vital_temp'] < 36.0).astype(int)
    df['high_fever_flag'] = (df['triage_vital_temp'] >= 39.0).astype(int)
    df['extreme_fever_flag'] = (df['triage_vital_temp'] >= 40.0).astype(int)
    
    # === ODDECHOWE ===
    df['tachypnea_flag'] = (df['triage_vital_rr'] > 20).astype(int)
    df['severe_tachypnea_flag'] = (df['triage_vital_rr'] > 30).astype(int)
    df['hypoxia_flag'] = (df['triage_vital_o2'] < 94).astype(int)
    df['severe_hypoxia_flag'] = (df['triage_vital_o2'] < 90).astype(int)
    df['rr_o2_product'] = df['triage_vital_rr'] * df['triage_vital_o2']
    df['o2_hr_ratio'] = df['triage_vital_o2'] / df['triage_vital_hr'].clip(1)
    
    # === SIRS ===
    df['sirs_temp'] = ((df['triage_vital_temp'] > 38.0) | (df['triage_vital_temp'] < 36.0)).astype(int)
    df['sirs_hr'] = (df['triage_vital_hr'] > 90).astype(int)
    df['sirs_rr'] = (df['triage_vital_rr'] > 20).astype(int)
    df['sirs_score'] = df['sirs_temp'] + df['sirs_hr'] + df['sirs_rr']
    df['sirs_positive'] = (df['sirs_score'] >= 2).astype(int)
    
    # === qSOFA ===
    df['qsofa_sbp'] = (df['triage_vital_sbp'] <= 100).astype(int)
    df['qsofa_rr'] = (df['triage_vital_rr'] >= 22).astype(int)
    df['qsofa_mental'] = df.get('cc_alteredmentalstatus', pd.Series(0, index=df.index))
    df['qsofa_score'] = df['qsofa_sbp'] + df['qsofa_rr'] + df['qsofa_mental']
    df['qsofa_positive'] = (df['qsofa_score'] >= 2).astype(int)
    
    # === INTERAKCJE ===
    df['temp_hr_interaction'] = df['triage_vital_temp'] * df['triage_vital_hr']
    
    # === COMPOSITE SCORES ===
    df['vitals_instability'] = (
        df['shock_index_critical'] +
        df['hypotension_flag'] +
        df['severe_tachycardia_flag'] +
        df['severe_tachypnea_flag'] +
        df['severe_hypoxia_flag'] +
        df['high_fever_flag']
    )
    df['shock_triad'] = (
        (df['triage_vital_sbp'] < 90) & 
        (df['triage_vital_hr'] > 110) & 
        (df['triage_vital_o2'] < 94)
    ).astype(int)
    df['sepsis_proxy'] = (
        (df['sirs_positive'] == 1) & 
        (df['qsofa_sbp'] == 1)
    ).astype(int)
    df['resp_failure_proxy'] = (
        (df['triage_vital_o2'] < 92) & 
        (df['triage_vital_rr'] > 25)
    ).astype(int)
    
    # === DEMOGRAFICZNE ===
    df['elderly_flag'] = (df['age'] >= 75).astype(int)
    df['pediatric_flag'] = (df['age'] <= 14).astype(int)
    
    return df

# Nowa liczba cech: 220 + ~50 = ~270
```

---

## 8. Podsumowanie — liczba nowych cech

| Kategoria | Liczba cech |
|---|---|
| Hemodynamiczne (indeksy) | 6 |
| Flagi hemodynamiczne | 10 |
| Temperatura | 4 |
| Oddechowe | 6 |
| SIRS | 5 |
| qSOFA | 5 |
| Interakcje | 1 |
| Composite scores | 4 |
| Demograficzne | 2 |
| **Łącznie nowych** | **~43** |
| **Łącznie z oryginalnymi** | **~263** |

---

## 9. Użycie w prompcie MedGemma (Opcja B — bez retreningu)

```python
def build_clinical_context(row) -> str:
    si = row['triage_vital_hr'] / max(row['triage_vital_sbp'], 1)
    pp = row['triage_vital_sbp'] - row['triage_vital_dbp']
    map_val = row['triage_vital_dbp'] + (pp / 3)
    
    sirs = int(
        (row['triage_vital_temp'] > 38 or row['triage_vital_temp'] < 36) +
        (row['triage_vital_hr'] > 90) +
        (row['triage_vital_rr'] > 20)
    )
    qsofa = int(
        (row['triage_vital_sbp'] <= 100) +
        (row['triage_vital_rr'] >= 22) +
        row.get('cc_alteredmentalstatus', 0)
    )
    
    return f"""
PARAMETRY KLINICZNE:
  Shock Index:    {si:.2f}  {'🔴 KRYTYCZNY (>1.0)' if si > 1.0 else '🟡 PODWYŻSZONY' if si > 0.7 else '🟢 OK'}
  MAP:            {map_val:.1f} mmHg  {'🔴 NISKI (<65)' if map_val < 65 else '🟢 OK'}
  Pulse Pressure: {pp:.1f} mmHg  {'🔴 WĄSKIE (<25)' if pp < 25 else '🟢 OK'}
  SIRS Score:     {sirs}/3  {'🔴 SEPSA' if sirs >= 2 else '🟢 OK'}
  qSOFA Score:    {qsofa}/3  {'🔴 DYSFUNKCJA NARZĄDOWA' if qsofa >= 2 else '🟢 OK'}
"""
```

---

*Wygenerowano: 2026-05-17 | SOR-AI Feature Engineering Reference*
