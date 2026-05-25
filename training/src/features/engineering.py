import pandas as pd
import numpy as np
from itertools import combinations


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    c = {}

    # === HEMODYNAMICZNE ===
    c['pulse_pressure'] = df['triage_vital_sbp'] - df['triage_vital_dbp']
    c['map'] = df['triage_vital_dbp'] + (c['pulse_pressure'] / 3)
    c['shock_index'] = df['triage_vital_hr'] / df['triage_vital_sbp'].clip(1)
    c['modified_shock_index'] = df['triage_vital_hr'] / c['map'].clip(1)
    c['pulse_pressure_ratio'] = c['pulse_pressure'] / df['triage_vital_sbp'].clip(1)
    c['hr_sbp_product'] = df['triage_vital_hr'] * df['triage_vital_sbp']

    # === FLAGI HEMODYNAMICZNE ===
    c['shock_index_critical'] = (c['shock_index'] >= 1.0).astype(int)
    c['shock_index_elevated'] = ((c['shock_index'] >= 0.7) & (c['shock_index'] < 1.0)).astype(int)
    c['hypotension_flag'] = (df['triage_vital_sbp'] < 100).astype(int)
    c['severe_hypotension_flag'] = (df['triage_vital_sbp'] < 90).astype(int)
    c['hypertension_flag'] = (df['triage_vital_sbp'] > 180).astype(int)
    c['hypertensive_crisis_flag'] = (df['triage_vital_sbp'] > 200).astype(int)
    c['tachycardia_flag'] = (df['triage_vital_hr'] > 100).astype(int)
    c['severe_tachycardia_flag'] = (df['triage_vital_hr'] > 130).astype(int)
    c['bradycardia_flag'] = (df['triage_vital_hr'] < 50).astype(int)

    # === FLAGI DBP ===
    c['low_dbp_flag'] = (df['triage_vital_dbp'] < 40).astype(int)
    c['dbp_risk'] = pd.cut(
        df['triage_vital_dbp'],
        bins=[-np.inf, 40, 60, 90, 110, np.inf],
        labels=[3, 1, 0, 1, 2],
        ordered=False,
    ).astype(int)

    # === TEMPERATURA ===
    c['temp_deviation'] = abs(df['triage_vital_temp'] - 37.0)
    c['hypothermia_flag'] = (df['triage_vital_temp'] < 36.0).astype(int)
    c['high_fever_flag'] = (df['triage_vital_temp'] >= 39.0).astype(int)
    c['extreme_fever_flag'] = (df['triage_vital_temp'] >= 40.0).astype(int)

    # === ODDECHOWE ===
    c['tachypnea_flag'] = (df['triage_vital_rr'] > 20).astype(int)
    c['severe_tachypnea_flag'] = (df['triage_vital_rr'] > 30).astype(int)
    c['hypoxia_flag'] = (df['triage_vital_o2'] < 94).astype(int)
    c['severe_hypoxia_flag'] = (df['triage_vital_o2'] < 90).astype(int)
    c['rr_o2_product'] = df['triage_vital_rr'] * df['triage_vital_o2']
    c['o2_hr_ratio'] = df['triage_vital_o2'] / df['triage_vital_hr'].clip(1)

    # === KOMPENSACJA ===
    c['compensation_ratio'] = df['triage_vital_hr'] / df['triage_vital_o2'].clip(1)
    c['over_compensation'] = (c['compensation_ratio'] > 1.2).astype(int)

    # === SIRS ===
    c['sirs_temp'] = ((df['triage_vital_temp'] > 38.0) | (df['triage_vital_temp'] < 36.0)).astype(int)
    c['sirs_hr'] = (df['triage_vital_hr'] > 90).astype(int)
    c['sirs_rr'] = (df['triage_vital_rr'] > 20).astype(int)
    c['sirs_score'] = c['sirs_temp'] + c['sirs_hr'] + c['sirs_rr']
    c['sirs_positive'] = (c['sirs_score'] >= 2).astype(int)

    # === qSOFA ===
    c['qsofa_sbp'] = (df['triage_vital_sbp'] <= 100).astype(int)
    c['qsofa_rr'] = (df['triage_vital_rr'] >= 22).astype(int)
    if 'cc_alteredmentalstatus' in df.columns:
        c['qsofa_mental'] = df['cc_alteredmentalstatus']
    else:
        c['qsofa_mental'] = 0
    c['qsofa_score'] = c['qsofa_sbp'] + c['qsofa_rr'] + c['qsofa_mental']
    c['qsofa_positive'] = (c['qsofa_score'] >= 2).astype(int)

    # === RATIOS MIĘDZY VITALS ===
    c['temp_hr_ratio'] = df['triage_vital_temp'] / df['triage_vital_hr'].clip(1)
    c['sbp_dbp_ratio'] = df['triage_vital_sbp'] / df['triage_vital_dbp'].clip(1)
    c['rr_hr_ratio'] = df['triage_vital_rr'] / df['triage_vital_hr'].clip(1)
    c['o2_rr_ratio'] = df['triage_vital_o2'] / df['triage_vital_rr'].clip(1)
    c['hr_rr_ratio'] = df['triage_vital_hr'] / df['triage_vital_rr'].clip(1)
    c['map_hr_ratio'] = c['map'] / df['triage_vital_hr'].clip(1)

    # === NONLINEAR TRANSFORMS ===
    c['hr_squared'] = df['triage_vital_hr'] ** 2
    c['temp_squared'] = df['triage_vital_temp'] ** 2
    c['shock_index_squared'] = c['shock_index'] ** 2
    c['log_hr'] = np.log1p(df['triage_vital_hr'])
    c['log_sbp'] = np.log1p(df['triage_vital_sbp'])

    # === POLYNOMIAL INTERACTIONS BETWEEN VITALS (15 pairs) ===
    key_vitals = [
        'triage_vital_temp', 'triage_vital_hr',
        'triage_vital_sbp', 'triage_vital_dbp',
        'triage_vital_rr', 'triage_vital_o2'
    ]
    for a, b in combinations(key_vitals, 2):
        aname = a.replace('triage_vital_', '')
        bname = b.replace('triage_vital_', '')
        c[f'{aname}_x_{bname}'] = df[a] * df[b]

    # === INTERAKCJE ===
    c['temp_hr_interaction'] = df['triage_vital_temp'] * df['triage_vital_hr']

    # === CC × VITALS INTERAKCJE ===
    if 'cc_chestpain' in df.columns:
        c['chestpain_tachycardia'] = df['cc_chestpain'] * c['tachycardia_flag']
    else:
        c['chestpain_tachycardia'] = 0
    if 'cc_respiratorydistress' in df.columns:
        c['resdistress_hypoxia'] = df['cc_respiratorydistress'] * c['hypoxia_flag']
    else:
        c['resdistress_hypoxia'] = 0
    if 'cc_fulltrauma' in df.columns:
        c['trauma_shock'] = df['cc_fulltrauma'] * c['shock_index_critical']
    else:
        c['trauma_shock'] = 0
    if 'cc_strokealert' in df.columns:
        c['stroke_mental'] = df['cc_strokealert'] * c['qsofa_mental']
    else:
        c['stroke_mental'] = 0
    c['cardiac_high_risk'] = (
        (df.get('cc_cardiacarrest', 0).astype(int)) |
        (c['chestpain_tachycardia'].astype(int) & c['hypotension_flag'])
    ).astype(int)

    # === WIEK × RYZYKO KLINICZNE ===
    c['elderly_flag'] = (df['age'] >= 75).astype(int)
    c['pediatric_flag'] = (df['age'] <= 14).astype(int)
    c['elderly_hypotension'] = c['elderly_flag'] * c['hypotension_flag']
    c['pediatric_fever'] = c['pediatric_flag'] * c['high_fever_flag']
    c['elderly_tachycardia'] = c['elderly_flag'] * c['tachycardia_flag']

    # === ARRIVAL MODE × SEVERITY ===
    c['ambulance_flag'] = (df['arrivalmode'] == 1).astype(int)
    c['ambulance_shock'] = c['ambulance_flag'] * c['shock_index_critical']
    c['ambulance_hypotension'] = c['ambulance_flag'] * c['hypotension_flag']

    # === BRAKUJĄCE WARTOŚCI ===
    c['temp_missing'] = df['triage_vital_temp'].isna().astype(int)
    c['sbp_missing'] = df['triage_vital_sbp'].isna().astype(int)
    c['o2_missing'] = df['triage_vital_o2'].isna().astype(int)
    c['rr_missing'] = df['triage_vital_rr'].isna().astype(int)
    c['missing_vitals_count'] = (
        c['temp_missing'] + c['sbp_missing'] +
        c['o2_missing'] + c['rr_missing']
    )

    # === MTS BINNING ===
    c['temp_mts_bin'] = pd.cut(
        df['triage_vital_temp'],
        bins=[-np.inf, 35.0, 36.0, 37.5, 38.5, 39.5, np.inf],
        labels=[0, 1, 2, 3, 4, 5],
        ordered=False,
    ).astype(int)
    c['hr_mts_bin'] = pd.cut(
        df['triage_vital_hr'],
        bins=[-np.inf, 40, 50, 100, 110, 130, np.inf],
        labels=[5, 3, 0, 2, 3, 4],
        ordered=False,
    ).astype(int)
    c['sbp_mts_bin'] = pd.cut(
        df['triage_vital_sbp'],
        bins=[-np.inf, 80, 100, 120, 160, 200, np.inf],
        labels=[5, 3, 0, 1, 2, 4],
        ordered=False,
    ).astype(int)
    c['rr_mts_bin'] = pd.cut(
        df['triage_vital_rr'],
        bins=[-np.inf, 8, 12, 20, 25, 30, np.inf],
        labels=[4, 2, 0, 2, 3, 4],
        ordered=False,
    ).astype(int)
    c['o2_mts_bin'] = pd.cut(
        df['triage_vital_o2'],
        bins=[-np.inf, 85, 90, 94, 96, np.inf],
        labels=[5, 4, 2, 1, 0],
        ordered=False,
    ).astype(int)
    c['mts_vitals_score'] = (
        c['temp_mts_bin'] + c['hr_mts_bin'] +
        c['sbp_mts_bin'] + c['rr_mts_bin'] +
        c['o2_mts_bin']
    )

    # === WIELOOBJAWOWOŚĆ ===
    cc_cols = [col for col in df.columns if col.startswith('cc_')]
    if cc_cols:
        c['cc_count'] = df[cc_cols].sum(axis=1).clip(upper=20)
        c['multi_complaint'] = (c['cc_count'] >= 3).astype(int)
        critical_cc = [col for col in cc_cols if col in [
            'cc_cardiacarrest', 'cc_strokealert', 'cc_respiratorydistress',
            'cc_unresponsive', 'cc_fulltrauma'
        ]]
        if critical_cc:
            c['critical_cc_count'] = df[critical_cc].sum(axis=1).clip(upper=5)
        else:
            c['critical_cc_count'] = 0
    else:
        c['cc_count'] = 0
        c['multi_complaint'] = 0
        c['critical_cc_count'] = 0

    # === AGGREGATED CC GROUPS ===
    def _cc(cols, df):
        return sum(df.get(col, pd.Series(0, index=df.index)) for col in cols)

    c['cc_cardiac_group'] = _cc(['cc_cardiacarrest', 'cc_chestpain', 'cc_palpitations', 'cc_syncope'], df)
    c['cc_neuro_group'] = _cc(['cc_strokealert', 'cc_unresponsive', 'cc_neurologicproblem', 'cc_headache', 'cc_alteredmentalstatus'], df)
    c['cc_respiratory_group'] = _cc(['cc_respiratorydistress', 'cc_uri', 'cc_asthma', 'cc_cough'], df)
    c['cc_trauma_group'] = _cc(['cc_fulltrauma', 'cc_trauma', 'cc_motorvehiclecrash', 'cc_fall'], df)
    c['cc_psych_group'] = _cc(['cc_suicidal', 'cc_psychiatricevaluation', 'cc_alcoholintoxication'], df)
    c['cc_abdominal_group'] = _cc(['cc_abdominalpain', 'cc_nausea', 'cc_vomiting'], df)
    system_cols = ['cc_cardiac_group', 'cc_neuro_group', 'cc_respiratory_group', 'cc_trauma_group']
    if all(col in c for col in system_cols):
        c['dominant_system'] = pd.DataFrame({k: c[k] for k in system_cols}).idxmax(axis=1).astype('category').cat.codes
    else:
        c['dominant_system'] = 0

    # === COMPOSITE SCORES ===
    c['vitals_instability'] = (
        c['shock_index_critical'] +
        c['hypotension_flag'] +
        c['severe_tachycardia_flag'] +
        c['severe_tachypnea_flag'] +
        c['severe_hypoxia_flag'] +
        c['high_fever_flag']
    )
    c['shock_triad'] = (
        (df['triage_vital_sbp'] < 90) &
        (df['triage_vital_hr'] > 110) &
        (df['triage_vital_o2'] < 94)
    ).astype(int)
    c['sepsis_proxy'] = (
        (c['sirs_positive'] == 1) &
        (c['qsofa_sbp'] == 1)
    ).astype(int)
    c['resp_failure_proxy'] = (
        (df['triage_vital_o2'] < 92) &
        (df['triage_vital_rr'] > 25)
    ).astype(int)

    # === PSYCHIATRYCZNE I BEHAWIORALNE (zależą od vitals_instability) ===
    if 'cc_psychiatricevaluation' in df.columns:
        c['psych_with_vitals_issue'] = df['cc_psychiatricevaluation'] * c['vitals_instability']
    else:
        c['psych_with_vitals_issue'] = 0
    if 'cc_suicidal' in df.columns:
        c['suicidal_high_risk'] = df['cc_suicidal'] * (c['vitals_instability'] >= 2).astype(int)
    else:
        c['suicidal_high_risk'] = 0
    if 'cc_alcoholintoxication' in df.columns:
        c['alcohol_hypotension'] = df['cc_alcoholintoxication'] * c['hypotension_flag']
    else:
        c['alcohol_hypotension'] = 0
    if 'cc_alteredmentalstatus' in df.columns:
        c['ams_fever'] = df['cc_alteredmentalstatus'] * c['high_fever_flag']
    else:
        c['ams_fever'] = 0

    # === PAIN SCORE PROXY ===
    c['pain_tachycardia'] = df.get('cc_chestpain', 0) * c['severe_tachycardia_flag']
    c['abdominal_shock'] = df.get('cc_abdominalpain', 0) * c['shock_index_critical']
    c['headache_hypertension'] = df.get('cc_headache', 0) * c['hypertension_flag']

    # === TEMPORAL PATTERNS ===
    c['night_arrival'] = ((df['arrivalhour_bin'] >= 22) | (df['arrivalhour_bin'] <= 6)).astype(int)
    c['night_critical'] = c['night_arrival'] * (c['vitals_instability'] >= 2).astype(int)
    c['weekend_flag'] = df['arrivalday'].isin([6, 7]).astype(int)
    c['weekend_critical'] = c['weekend_flag'] * c['shock_index_critical']
    c['winter_respiratory'] = (
        df['arrivalmonth'].isin([11, 12, 1, 2]).astype(int) *
        df.get('cc_respiratorydistress', 0)
    )

    return pd.concat([df, pd.DataFrame(c, index=df.index)], axis=1)


ENGINEERED_FEATURES = [
    # Hemodynamiczne
    'pulse_pressure', 'map', 'shock_index', 'modified_shock_index',
    'pulse_pressure_ratio', 'hr_sbp_product',
    # Flagi hemodynamiczne
    'shock_index_critical', 'shock_index_elevated',
    'hypotension_flag', 'severe_hypotension_flag',
    'hypertension_flag', 'hypertensive_crisis_flag',
    'tachycardia_flag', 'severe_tachycardia_flag', 'bradycardia_flag',
    # DBP
    'low_dbp_flag', 'dbp_risk',
    # Temperatura
    'temp_deviation', 'hypothermia_flag', 'high_fever_flag', 'extreme_fever_flag',
    # Oddechowe
    'tachypnea_flag', 'severe_tachypnea_flag',
    'hypoxia_flag', 'severe_hypoxia_flag',
    'rr_o2_product', 'o2_hr_ratio',
    # Kompensacja
    'compensation_ratio', 'over_compensation',
    # SIRS
    'sirs_temp', 'sirs_hr', 'sirs_rr', 'sirs_score', 'sirs_positive',
    # qSOFA
    'qsofa_sbp', 'qsofa_rr', 'qsofa_mental', 'qsofa_score', 'qsofa_positive',
    # Ratios
    'temp_hr_ratio', 'sbp_dbp_ratio', 'rr_hr_ratio',
    'o2_rr_ratio', 'hr_rr_ratio', 'map_hr_ratio',
    # Nonlinear
    'hr_squared', 'temp_squared', 'shock_index_squared', 'log_hr', 'log_sbp',
    # Polynomial interactions (15)
    'temp_x_hr', 'temp_x_sbp', 'temp_x_dbp', 'temp_x_rr', 'temp_x_o2',
    'hr_x_sbp', 'hr_x_dbp', 'hr_x_rr', 'hr_x_o2',
    'sbp_x_dbp', 'sbp_x_rr', 'sbp_x_o2',
    'dbp_x_rr', 'dbp_x_o2', 'rr_x_o2',
    # Interakcje
    'temp_hr_interaction',
    # CC × Vitals
    'chestpain_tachycardia', 'resdistress_hypoxia',
    'trauma_shock', 'stroke_mental', 'cardiac_high_risk',
    # Psych/behavioral
    'psych_with_vitals_issue', 'suicidal_high_risk',
    'alcohol_hypotension', 'ams_fever',
    # Pain proxy
    'pain_tachycardia', 'abdominal_shock', 'headache_hypertension',
    # Wiek × ryzyko
    'elderly_flag', 'pediatric_flag',
    'elderly_hypotension', 'pediatric_fever', 'elderly_tachycardia',
    # Arrival mode
    'ambulance_flag', 'ambulance_shock', 'ambulance_hypotension',
    # Temporal
    'night_arrival', 'night_critical', 'weekend_flag',
    'weekend_critical', 'winter_respiratory',
    # Missing
    'temp_missing', 'sbp_missing', 'o2_missing', 'rr_missing',
    'missing_vitals_count',
    # MTS bins
    'temp_mts_bin', 'hr_mts_bin', 'sbp_mts_bin', 'rr_mts_bin',
    'o2_mts_bin', 'mts_vitals_score',
    # CC aggregated groups
    'cc_cardiac_group', 'cc_neuro_group', 'cc_respiratory_group',
    'cc_trauma_group', 'cc_psych_group', 'cc_abdominal_group',
    'dominant_system',
    # Wieloobjawowość
    'cc_count', 'multi_complaint', 'critical_cc_count',
    # Composite
    'vitals_instability', 'shock_triad', 'sepsis_proxy', 'resp_failure_proxy',
]
