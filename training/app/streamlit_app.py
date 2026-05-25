"""
Streamlit demo SOR-AI.

Interaktywne UI:
    1. Wprowadzanie danych pacjenta (vital signs, chief complaint, demografia).
    2. Wybór modelu z dropdown (lista zapisanych modeli).
    3. Predykcja MTS z probabilities.
    4. Trzy warstwy wyjaśnień:
        - SHAP (top features for/against)
        - MTS rule check
        - Ollama medical reasoning (jeśli dostępne)

Uruchomienie:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explain.mts_rules import (  # noqa: E402
    MTS_VITAL_THRESHOLDS,
    check_consistency,
    explain_rule_decision,
    rule_based_triage,
)
from src.explain.ollama_medical import OllamaMedicalExplainer  # noqa: E402
from src.explain.shap_explainer import SHAPTriageExplainer  # noqa: E402
from src.models.base import BaseTriageModel  # noqa: E402
from src.utils.config import (  # noqa: E402
    CLASS_NAMES,
    CLASS_NAMES_PL,
    MODELS_DIR,
    OLLAMA_DEFAULT_MODEL,
    TRAIN_PARQUET,
)


# ─────────────────────────────────────────
# Konfiguracja strony
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SOR-AI: System Triażu MTS",
    page_icon="[+]",
    layout="wide",
    initial_sidebar_state="expanded",
)

MTS_COLORS_HEX = {
    "Red": "#D32F2F",
    "Orange": "#F57C00",
    "Yellow": "#FBC02D",
    "Green": "#388E3C",
    "Blue": "#1976D2",
}


# ─────────────────────────────────────────
# Ładowanie modelu (cache)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Ładowanie modelu…")
def load_model_cached(model_path: str) -> BaseTriageModel:
    return BaseTriageModel.load(model_path)


@st.cache_data(show_spinner="Ładowanie próbki tła…")
def load_background_sample(n: int = 200) -> pd.DataFrame | None:
    if not TRAIN_PARQUET.exists():
        return None
    df = pd.read_parquet(TRAIN_PARQUET)
    return df.sample(min(n, len(df)), random_state=42)


@st.cache_resource(show_spinner="Inicjalizacja SHAP…")
def init_shap_explainer(_model: BaseTriageModel, _background: pd.DataFrame) -> SHAPTriageExplainer:
    explainer = SHAPTriageExplainer(model=_model, background_data=_background)
    explainer.fit()
    return explainer


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def list_available_models() -> list[Path]:
    return sorted(MODELS_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)


def render_class_badge(class_name: str, label_pl: str | None = None) -> str:
    color = MTS_COLORS_HEX.get(class_name, "#666")
    label = label_pl or class_name
    return (
        f'<div style="display:inline-block; padding:0.6rem 1.2rem; '
        f'background-color:{color}; color:white; font-weight:bold; '
        f'border-radius:8px; font-size:1.4rem; margin:0.5rem 0;">'
        f'{label}</div>'
    )


def render_probability_bars(probabilities: dict[str, float]) -> None:
    for class_name in CLASS_NAMES:
        prob = probabilities.get(class_name, 0.0)
        color = MTS_COLORS_HEX[class_name]
        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                    <span><b>{class_name}</b></span>
                    <span>{prob:.1%}</span>
                </div>
                <div style="background-color:#eee; border-radius:4px; height:14px;">
                    <div style="background-color:{color}; width:{prob*100}%;
                                height:14px; border-radius:4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_patient_input() -> dict[str, Any]:
    """Formularz pacjenta."""
    st.sidebar.header("Dane pacjenta")

    # Demografia
    with st.sidebar.expander("Demografia", expanded=True):
        age = st.number_input("Wiek", 0, 120, 55, step=1)
        sex = st.selectbox("Płeć", ["Mężczyzna", "Kobieta"], index=0)

    # Vital signs
    with st.sidebar.expander("Vital signs (TRIAGE)", expanded=True):
        sbp = st.number_input("Ciśnienie skurczowe (SBP, mmHg)", 30, 250, 120, step=1)
        dbp = st.number_input("Ciśnienie rozkurczowe (DBP, mmHg)", 20, 150, 80, step=1)
        pulse = st.number_input("Tętno (bpm)", 20, 220, 75, step=1)
        resp = st.number_input("Częstość oddechów (/min)", 4, 60, 16, step=1)
        o2sat = st.number_input("Saturacja O₂ (%)", 50, 100, 98, step=1)
        temp = st.number_input("Temperatura (°C)", 28.0, 45.0, 36.6, step=0.1)
        pain = st.slider("Skala bólu (0-10)", 0, 10, 0)

    # Chief complaint
    with st.sidebar.expander("Główna skarga", expanded=False):
        cc_options = [
            "Ból w klatce piersiowej",
            "Duszność",
            "Ból brzucha",
            "Ból głowy",
            "Wymioty / nudności",
            "Uraz / wypadek",
            "Krwawienie",
            "Utrata przytomności",
            "Gorączka",
            "Inne (wpisz poniżej)",
        ]
        cc = st.selectbox("Wybierz", cc_options, index=0)
        cc_other = st.text_input("Inna skarga (opcjonalnie)", "") if cc == "Inne (wpisz poniżej)" else ""
        chief_complaint = cc_other if cc_other else cc

    return {
        "age": age,
        "sex": sex,
        "triage_sbp": sbp,
        "triage_dbp": dbp,
        "triage_pulse": pulse,
        "triage_resp": resp,
        "triage_o2sat": o2sat,
        "triage_temp": temp,
        "triage_pain": pain,
        "chief_complaint": chief_complaint,
    }


def patient_dict_to_features(
    patient: dict[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Buduje wiersz DataFrame z danymi pacjenta wyrównany do feature_names modelu.
    Brakujące cechy (np. PMH, MED, lab tests) są wypełniane zerami / medianą.
    """
    row: dict[str, Any] = {}

    # Mapowanie demograficznych
    sex_male = 1 if patient["sex"] == "Mężczyzna" else 0
    row["sex"] = sex_male

    # Wpisz vital signs (wszystkie aliases)
    vitals_map = {
        "triage_sbp": patient["triage_sbp"],
        "triage_dbp": patient["triage_dbp"],
        "triage_pulse": patient["triage_pulse"],
        "triage_resp": patient["triage_resp"],
        "triage_o2sat": patient["triage_o2sat"],
        "triage_temp": patient["triage_temp"],
        "triage_pain": patient["triage_pain"],
        "age": patient["age"],
    }
    row.update(vitals_map)

    # Cecha CC: aktywuj odpowiednią flagę 'cc_*'
    cc_keyword_map = {
        "Ból w klatce piersiowej": "cc_chestpain",
        "Duszność": "cc_shortnessofbreath",
        "Ból brzucha": "cc_abdominalpain",
        "Ból głowy": "cc_headache",
        "Wymioty / nudności": "cc_vomiting",
        "Uraz / wypadek": "cc_trauma",
        "Krwawienie": "cc_bleeding",
        "Utrata przytomności": "cc_syncope",
        "Gorączka": "cc_fever",
    }
    cc_flag = cc_keyword_map.get(patient["chief_complaint"])
    if cc_flag:
        row[cc_flag] = 1

    # Buduj DataFrame z wszystkimi cechami modelu
    feature_dict = {}
    for col in feature_names:
        feature_dict[col] = row.get(col, 0)

    return pd.DataFrame([feature_dict])


# ─────────────────────────────────────────
# UI — main
# ─────────────────────────────────────────
def main() -> None:
    st.title("SOR-AI: System Klasyfikacji Triażu MTS")
    st.markdown(
        "*Asystent decyzyjny dla pielęgniarek triażowych.* "
        "Klasyfikuje pacjenta w 5 kategoriach Manchester Triage System "
        "i wyjaśnia decyzję na trzech poziomach: SHAP, reguły MTS, "
        "lokalne LLM (Ollama)."
    )

    # ─── Sidebar — model + Ollama ───
    st.sidebar.header("Konfiguracja")
    available = list_available_models()
    if not available:
        st.error(
            f"Nie znaleziono żadnych modeli w `{MODELS_DIR}`.\n\n"
            "Wytrenuj model: `python scripts/03_train.py --model xgboost`"
        )
        return

    model_path = st.sidebar.selectbox(
        "Wybierz model",
        options=available,
        format_func=lambda p: f"{p.name} ({p.stat().st_size / 1e6:.1f} MB)",
    )

    use_ollama = st.sidebar.checkbox("Użyj Ollama (medical reasoning)", value=False)
    ollama_model_name = OLLAMA_DEFAULT_MODEL
    if use_ollama:
        ollama_model_name = st.sidebar.text_input("Nazwa modelu Ollama", OLLAMA_DEFAULT_MODEL)

    # ─── Dane pacjenta ───
    patient = build_patient_input()

    # ─── Główny przycisk ───
    if not st.sidebar.button("Klasyfikuj pacjenta", type="primary", use_container_width=True):
        st.info("Wprowadź dane pacjenta w panelu po lewej, następnie kliknij **Klasyfikuj**.")
        return

    # ─── Predykcja ───
    model = load_model_cached(str(model_path))

    if not model.feature_names:
        st.error("Model nie ma zapisanych nazw cech (`feature_names`). Wytrenuj go ponownie.")
        return

    X_patient = patient_dict_to_features(patient, model.feature_names)

    proba = model.predict_proba(X_patient)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = CLASS_NAMES[pred_idx]
    pred_class_pl = CLASS_NAMES_PL[pred_idx]
    probabilities = dict(zip(CLASS_NAMES, proba.tolist()))

    # ─── Layout: 2 kolumny ───
    col_left, col_right = st.columns([1, 1])

    # ─── LEWA: predykcja + probabilities + reguły ───
    with col_left:
        st.subheader("Predykcja modelu")
        st.markdown(render_class_badge(pred_class, pred_class_pl), unsafe_allow_html=True)
        st.markdown(f"**Pewność**: {max(proba):.1%}")
        st.markdown("**Rozkład prawdopodobieństw:**")
        render_probability_bars(probabilities)

        # MTS rule check
        st.subheader("Weryfikacja regułowa MTS")
        rule_check = rule_based_triage(patient)

        rule_class = rule_check["suggested_category"]
        st.markdown(render_class_badge(rule_class), unsafe_allow_html=True)
        st.markdown(f"**Maks. czas oczekiwania**: {rule_check['max_wait_minutes']} min")

        if rule_check["triggered_rules"]:
            st.markdown("**Zadziałane reguły:**")
            for rule in rule_check["triggered_rules"]:
                st.markdown(f"- {rule['description']}")
        else:
            st.success("Vital signs w normie — brak alarmujących reguł.")

        # Spójność ML vs reguły
        st.subheader("Zgodność ML ↔ MTS")
        consistency = check_consistency(pred_idx, rule_check["suggested_class_idx"])

        if consistency["agree"]:
            st.success(consistency["verdict"])
        elif consistency["safer_side"] == "ml":
            st.info(consistency["verdict"])
        else:
            st.error(consistency["verdict"])

    # ─── PRAWA: SHAP + Ollama ───
    with col_right:
        st.subheader("Wyjaśnienia SHAP (top cechy)")

        background = load_background_sample()
        if background is None:
            st.warning("Brak danych train.parquet — SHAP niedostępny.")
            shap_exp = None
        else:
            X_train_aligned = background[model.feature_names].fillna(0)
            shap_explainer = init_shap_explainer(model, X_train_aligned)
            try:
                shap_exp = shap_explainer.explain_patient(X_patient, top_n=8)
            except Exception as e:
                st.error(f"SHAP błąd: {e}")
                shap_exp = None

        if shap_exp:
            st.markdown(f"**Cechy wspierające klasyfikację jako {pred_class}:**")
            for f in shap_exp["top_features_for"][:5]:
                st.markdown(
                    f"- `{f['feature']}` (wartość {f['patient_value']}): "
                    f"SHAP **+{f['shap_value']:.3f}**"
                )

            st.markdown(f"**Cechy działające PRZECIW {pred_class}:**")
            if shap_exp["top_features_against"]:
                for f in shap_exp["top_features_against"][:3]:
                    st.markdown(
                        f"- `{f['feature']}` (wartość {f['patient_value']}): "
                        f"SHAP **{f['shap_value']:.3f}**"
                    )
            else:
                st.markdown("(brak)")

        # Ollama
        if use_ollama and shap_exp:
            st.subheader("Medical reasoning (Ollama LLM)")
            ollama = OllamaMedicalExplainer(model_name=ollama_model_name)
            if not ollama.is_available():
                st.error(
                    f"Ollama niedostępne lub model '{ollama_model_name}' nie zainstalowany.\n"
                    f"Dostępne modele: {ollama.list_available_models() or 'brak'}\n\n"
                    f"Aby aktywować:\n```\nollama serve\nollama pull {ollama_model_name}\n```"
                )
            else:
                with st.spinner("Generowanie wyjaśnienia medycznego…"):
                    try:
                        text = ollama.explain(
                            patient_data=patient,
                            predicted_class=pred_class,
                            predicted_class_pl=pred_class_pl,
                            probabilities=probabilities,
                            shap_explanation=shap_exp,
                            rule_check=rule_check,
                        )
                        st.markdown(text)
                    except Exception as e:
                        st.error(f"Błąd Ollama: {e}")

    # ─── Stopka — surowe dane pacjenta ───
    with st.expander("Surowe dane pacjenta + cechy modelu", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Dane wejściowe:**")
            st.json(patient)
        with c2:
            st.markdown("**Wektor cech modelu (top 20):**")
            st.dataframe(X_patient.T.head(20))


if __name__ == "__main__":
    main()
