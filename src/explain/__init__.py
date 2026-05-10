"""
Warstwy wyjaśnień (XAI) dla modeli triażowych:
    1. SHAP / LIME      — model-level (które cechy popchnęły decyzję?)
    2. DiCE             — counterfactuals ("co musiałoby się zmienić, by Yellow→Orange?")
    3. MTS rule engine  — sprawdzenie zgodności z protokołem klinicznym
    4. Ollama LLM       — medyczne wyjaśnienie w języku naturalnym
"""

from src.explain.dice_counterfactual import DiCEExplainer
from src.explain.lime_explainer import LIMETriageExplainer
from src.explain.mts_rules import (
    MTS_VITAL_THRESHOLDS,
    rule_based_triage,
)
from src.explain.ollama_medical import OllamaMedicalExplainer
from src.explain.shap_explainer import (
    SHAPTriageExplainer,
    shap_summary_plot,
    shap_waterfall_plot,
)

__all__ = [
    "DiCEExplainer",
    "LIMETriageExplainer",
    "MTS_VITAL_THRESHOLDS",
    "OllamaMedicalExplainer",
    "SHAPTriageExplainer",
    "rule_based_triage",
    "shap_summary_plot",
    "shap_waterfall_plot",
]
