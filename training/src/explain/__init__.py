"""
Warstwy wyjaśnień (XAI) dla modeli triażowych ESI (klasy 0–4 = ESI 1–5):
    1. SHAP / LIME        — model-level (które cechy popchnęły decyzję?)
    2. DiCE               — counterfactuals ("co musiałoby się zmienić, by ESI 3 → ESI 2?")
    3. Reguły ESI         — weryfikacja parametrów życiowych (punkty A / B / D,
                            moduł `esi_rules`)
    4. Ollama LLM         — medyczne wyjaśnienie w języku naturalnym

Kolory TOPSOR (czerwony … niebieski) służą wyłącznie do prezentacji poziomów ESI.
"""

from src.explain.dice_counterfactual import DiCEExplainer
from src.explain.esi_rules import (
    ESI_VITAL_RULES,
    check_consistency,
    explain_rule_decision,
    rule_based_triage,
)
from src.explain.lime_explainer import LIMETriageExplainer
from src.explain.ollama_medical import OllamaMedicalExplainer
from src.explain.shap_explainer import (
    ExplainerUnavailable,
    SHAPTriageExplainer,
    load_background,
    shap_summary_plot,
    shap_waterfall_plot,
)

__all__ = [
    "DiCEExplainer",
    "ESI_VITAL_RULES",
    "ExplainerUnavailable",
    "LIMETriageExplainer",
    "OllamaMedicalExplainer",
    "SHAPTriageExplainer",
    "check_consistency",
    "explain_rule_decision",
    "load_background",
    "rule_based_triage",
    "shap_summary_plot",
    "shap_waterfall_plot",
]
