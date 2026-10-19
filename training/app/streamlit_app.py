"""
Streamlit demo SOR-AI — triaż ESI 1–5 (kolory TOPSOR).

Interaktywne UI:
    1. Formularz pacjenta w schemacie kanonicznym: wiek, płeć, sposób przybycia
       (ZRM / samodzielnie / inny), parametry życiowe (temperatura w °C, tlenoterapia),
       skargi główne z listy Yale (cc_*), czas przybycia. Wiersz przechodzi przez
       `src.features.engineering.engineer_features` i jest wyrównywany do cech modelu
       (brakujące → NaN, tak jak w treningu).
    2. Wybór runa z runs/ (domyślnie najnowszy z zapisanym modelem) albo
       ścieżka do modelu: `-- --model runs/<run>/model.joblib`.
    3. Predykcja ESI z kolorem TOPSOR i maks. czasem oczekiwania dla obu reguł
       decyzyjnych (argmax i kosztowa — niedotriażowanie droższe), prawdopodobieństwa.
    4. Wyjaśnienia: weryfikacja regułowa parametrów (ESI, punkty A / B / D),
       SHAP, opcjonalnie lokalny LLM (Ollama).

Uruchomienie:
    streamlit run app/streamlit_app.py
    streamlit run app/streamlit_app.py -- --model runs/<run>/model.joblib

Funkcje pomocnicze (budowa wiersza pacjenta, wybór runa) można importować bez
zainstalowanego streamlit — np. w testach lub w notatnikach.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Dodaj root projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:  # helpery działają bez streamlit; main() go wymaga
    st = None

from src.data.yale_schema import ARRIVAL_CATEGORIES, TRAUMA_ACTIVATION_COL, hour_to_bin  # noqa: E402
from src.evaluation.metrics import decide  # noqa: E402
from src.explain.esi_rules import (  # noqa: E402
    check_consistency,
    explain_rule_decision,
    rule_based_triage,
)
from src.explain.ollama_medical import OllamaMedicalExplainer  # noqa: E402
from src.explain.shap_explainer import (  # noqa: E402
    ExplainerUnavailable,
    SHAPTriageExplainer,
    load_background,
    shap_waterfall_plot,
)
from src.features.engineering import ENGINEERED_FEATURES, VITALS, engineer_features  # noqa: E402
from src.models import load_model  # noqa: E402
from src.models.base import BaseTriageModel  # noqa: E402
from src.utils.config import (  # noqa: E402
    CLASS_NAMES,
    CLASS_NAMES_PL,
    DATA_VERSION,
    OLLAMA_DEFAULT_MODEL,
    RUNS_DIR,
    TOPSOR_COLORS,
    TOPSOR_COLORS_PL,
    TOPSOR_HEX,
    TOPSOR_MAX_WAIT_MINUTES,
)

# ─────────────────────────────────────────
# Stałe formularza
# ─────────────────────────────────────────
# Klucze = src.data.yale_schema.ARRIVAL_CATEGORIES
ARRIVAL_LABELS_PL: dict[str, str] = {
    "ambulance": "Zespół ratownictwa medycznego (ZRM)",
    "self": "Samodzielnie (pieszo, samochodem, transportem publicznym)",
    "other": "Inny (np. policja, wózek)",
}

# (kolumna kanoniczna, etykieta, min, max, domyślnie, krok)
VITAL_FIELDS: list[tuple[str, str, float, float, float, float]] = [
    (VITALS["hr"], "Tętno (/min)", 20.0, 250.0, 80.0, 1.0),
    (VITALS["sbp"], "Ciśnienie skurczowe SBP (mmHg)", 40.0, 280.0, 130.0, 1.0),
    (VITALS["dbp"], "Ciśnienie rozkurczowe DBP (mmHg)", 20.0, 180.0, 80.0, 1.0),
    (VITALS["rr"], "Częstość oddechów (/min)", 4.0, 70.0, 16.0, 1.0),
    (VITALS["o2"], "Saturacja SpO₂ (%)", 50.0, 100.0, 98.0, 1.0),
    (VITALS["temp"], "Temperatura (°C)", 30.0, 43.0, 36.6, 0.1),
]
O2_DEVICE_COL = "triage_vital_o2_device"

# Najczęstsze / najważniejsze skargi główne Yale z polskimi etykietami
# (pozostałe dostępne w osobnej liście pod oryginalnymi nazwami)
COMPLAINT_LABELS_PL: dict[str, str] = {
    "cc_chestpain": "Ból w klatce piersiowej",
    "cc_shortnessofbreath": "Duszność",
    "cc_abdominalpain": "Ból brzucha",
    "cc_headache": "Ból głowy",
    "cc_emesis": "Wymioty",
    "cc_nausea": "Nudności",
    "cc_fever": "Gorączka",
    "cc_dizziness": "Zawroty głowy",
    "cc_syncope": "Omdlenie",
    "cc_weakness": "Osłabienie",
    "cc_palpitations": "Kołatanie serca",
    "cc_alteredmentalstatus": "Zaburzenia świadomości",
    "cc_seizures": "Drgawki",
    "cc_strokealert": "Alert udarowy",
    TRAUMA_ACTIVATION_COL: "Aktywacja zespołu urazowego",
    "cc_respiratorydistress": "Niewydolność oddechowa",
    "cc_cardiacarrest": "Zatrzymanie krążenia",
    "cc_unresponsive": "Brak reakcji",
    "cc_fall": "Upadek",
    "cc_fall_65": "Upadek (≥ 65 lat)",
    "cc_headinjury": "Uraz głowy",
    "cc_motorvehiclecrash": "Wypadek komunikacyjny",
    "cc_laceration": "Rana cięta",
    "cc_backpain": "Ból pleców",
    "cc_gibleeding": "Krwawienie z przewodu pokarmowego",
    "cc_allergicreaction": "Reakcja alergiczna",
    "cc_suicidal": "Myśli samobójcze",
    "cc_overdose_intentional": "Celowe przedawkowanie",
    "cc_alcoholintoxication": "Upojenie alkoholowe",
    "cc_urinarytractinfection": "Infekcja dróg moczowych",
    "cc_rash": "Wysypka",
    "cc_sorethroat": "Ból gardła",
    "cc_medicationrefill": "Przepisanie leków",
}


# ─────────────────────────────────────────
# Helpery — runy i model (bez streamlit)
# ─────────────────────────────────────────
def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Argumenty po `--` w `streamlit run app/streamlit_app.py -- --model …`."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=None, help="Ścieżka do model.joblib (zamiast wyboru runa)")
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return args


def list_model_runs(runs_dir: Path = RUNS_DIR) -> list[dict[str, Any]]:
    """Runy z zapisanym model.joblib, najnowsze pierwsze (nazwa zaczyna się od znacznika czasu)."""
    from src.ensemble import list_runs

    if not Path(runs_dir).exists():
        return []
    table = list_runs(runs_dir)
    out = []
    for row in table.sort_values("run", ascending=False).to_dict("records"):
        path = Path(runs_dir) / row["run"] / "model.joblib"
        if path.exists():
            is_sample = not (pd.isna(row.get("train_sample")) and pd.isna(row.get("eval_sample")))
            out.append({**row, "path": path, "is_sample": is_sample})
    return out


def default_run_index(runs: list[dict[str, Any]]) -> int:
    """Domyślnie najnowszy run na pełnych danych; gdy brak — najnowszy w ogóle."""
    for i, run in enumerate(runs):
        if not run["is_sample"]:
            return i
    return 0


def complaint_columns(model_features: list[str] | None = None) -> list[str]:
    """Pełna lista skarg głównych cc_* (z feature_sets.json; awaryjnie z cech modelu)."""
    try:
        from src.data.dataset import feature_groups

        cols = feature_groups().get("chief_complaints", [])
        if cols:
            return list(cols)
    except Exception:
        pass
    engineered = set(ENGINEERED_FEATURES)
    return [c for c in (model_features or []) if c.startswith("cc_") and c not in engineered]


# ─────────────────────────────────────────
# Helpery — pacjent → cechy modelu (bez streamlit)
# ─────────────────────────────────────────
def build_patient_row(
    age: float,
    is_male: bool,
    arrival_mode: str,
    vitals: dict[str, float | None],
    o2_device: bool,
    complaints: list[str],
    arrival_time: datetime,
    all_complaints: list[str] | None = None,
) -> pd.DataFrame:
    """
    Jeden wiersz w schemacie KANONICZNYM (wejście `engineer_features`).

    vitals: {kolumna triage_vital_*: wartość lub None (= brak pomiaru → NaN)}, temperatura w °C.
    complaints: wybrane kolumny cc_*; pozostałe z `all_complaints` = 0 (jak w danych treningowych).
    """
    if arrival_mode not in ARRIVAL_CATEGORIES:
        raise ValueError(f"Nieznany sposób przybycia '{arrival_mode}'. Dostępne: {ARRIVAL_CATEGORIES}")
    row: dict[str, float] = {
        "age": float(age),
        "is_male": 1.0 if is_male else 0.0,
        "arrival_month": float(arrival_time.month),
        "arrival_weekday": float(arrival_time.isoweekday()),      # pon = 1 … nd = 7
        "arrival_hour_bin": float(hour_to_bin(arrival_time.hour)),
    }
    for category in ARRIVAL_CATEGORIES:
        row[f"arrival_{category}"] = 1.0 if category == arrival_mode else 0.0
    for col in VITALS.values():
        value = vitals.get(col)
        row[col] = np.nan if value is None else float(value)
    row[O2_DEVICE_COL] = 1.0 if o2_device else 0.0
    for col in all_complaints or []:
        row[col] = 0.0
    for col in complaints:
        row[col] = 1.0
    return pd.DataFrame([row])


def prepare_model_input(patient_row: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Cechy kliniczne (`engineer_features`) + wyrównanie do cech modelu (brakujące → NaN)."""
    engineered = engineer_features(patient_row)
    return engineered.reindex(columns=list(feature_names)).astype(np.float32)


def predict_levels(model: Any, X: pd.DataFrame) -> dict[str, Any]:
    """Prawdopodobieństwa ESI 1–5 i decyzje obu reguł (indeksy 0–4)."""
    proba = np.asarray(model.predict_proba(X), dtype=float)
    return {
        "proba": proba[0],
        "argmax": int(decide(proba, "argmax")[0]),
        "cost": int(decide(proba, "cost")[0]),
    }


def level_info(class_idx: int) -> dict[str, Any]:
    color = TOPSOR_COLORS[class_idx]
    return {
        "esi": class_idx + 1,
        "label_pl": CLASS_NAMES_PL[class_idx],
        "color": color,
        "color_pl": TOPSOR_COLORS_PL[class_idx],
        "hex": TOPSOR_HEX[CLASS_NAMES[class_idx]],
        "max_wait_minutes": TOPSOR_MAX_WAIT_MINUTES[color],
    }


def format_value(value: Any) -> str:
    """Wartość cechy do wyświetlenia (4 cyfry znaczące, NaN → 'brak')."""
    if isinstance(value, str):
        return "brak" if value == "NaN" else value
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "brak" if f != f else f"{f:.4g}"


def level_badge_html(class_idx: int, caption: str | None = None) -> str:
    info = level_info(class_idx)
    text_color = "#212121" if info["color"] == TOPSOR_COLORS[2] else "white"   # ciemny tekst na żółtym
    caption_html = f'<div style="font-size:0.8rem; opacity:0.9;">{caption}</div>' if caption else ""
    return (
        f'<div style="display:inline-block; padding:0.6rem 1.2rem; background-color:{info["hex"]}; '
        f'color:{text_color}; font-weight:bold; border-radius:8px; font-size:1.3rem; margin:0.4rem 0;">'
        f'{caption_html}{info["label_pl"]}</div>'
    )


# ─────────────────────────────────────────
# UI (wymaga streamlit)
# ─────────────────────────────────────────
def _cache_resource(func):
    return st.cache_resource(show_spinner=False)(func) if st is not None else func


def _cache_data(func):
    return st.cache_data(show_spinner=False)(func) if st is not None else func


@_cache_resource
def load_model_cached(model_path: str) -> BaseTriageModel:
    return load_model(model_path)


@_cache_data
def load_background_cached(feature_names: tuple[str, ...], n: int = 200) -> pd.DataFrame | None:
    try:
        return load_background(list(feature_names), n=n)
    except Exception:
        return None


@_cache_resource
def init_shap_explainer(model_path: str, _model: BaseTriageModel) -> SHAPTriageExplainer:
    background = None
    if SHAPTriageExplainer.method_for(_model) in ("linear", "permutation"):
        background = load_background_cached(tuple(_model.feature_names))
    return SHAPTriageExplainer(_model, background_data=background).fit()


def render_probability_bars(proba: np.ndarray) -> None:
    for idx, prob in enumerate(proba):
        info = level_info(idx)
        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                    <span><b>{info['label_pl']}</b></span>
                    <span>{prob:.1%}</span>
                </div>
                <div style="background-color:#eee; border-radius:4px; height:14px;">
                    <div style="background-color:{info['hex']}; width:{prob * 100:.1f}%;
                                height:14px; border-radius:4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def patient_form(all_complaints: list[str]) -> dict[str, Any]:
    """Formularz pacjenta w panelu bocznym → argumenty `build_patient_row`."""
    st.sidebar.header("Dane pacjenta")

    with st.sidebar.expander("Demografia i przybycie", expanded=True):
        age = st.number_input("Wiek (lata)", 0, 120, 55, step=1)
        sex = st.selectbox("Płeć", ["Mężczyzna", "Kobieta"], index=0)
        arrival_mode = st.selectbox(
            "Sposób przybycia", ARRIVAL_CATEGORIES, index=ARRIVAL_CATEGORIES.index("self"),
            format_func=lambda k: ARRIVAL_LABELS_PL.get(k, k),
        )
        now = datetime.now()
        arrival_date = st.date_input("Data przybycia", now.date())
        arrival_clock = st.time_input("Godzina przybycia", now.time().replace(second=0, microsecond=0))

    with st.sidebar.expander("Parametry życiowe (triaż)", expanded=True):
        labels = {col: label for col, label, *_ in VITAL_FIELDS}
        not_measured = st.multiselect(
            "Niezmierzone (brak pomiaru)", [col for col, *_ in VITAL_FIELDS],
            format_func=lambda c: labels[c],
            help=(
                "W TOPSOR pomiar jest obowiązkowy. Brak pomiaru model odczytuje jak w Yale "
                "(pacjent od razu na sali resuscytacyjnej) — wynik bez parametrów "
                "jest mniej wiarygodny."
            ),
        )
        vitals: dict[str, float | None] = {}
        for col, label, lo, hi, default, step in VITAL_FIELDS:
            if col in not_measured:
                vitals[col] = None
                continue
            vitals[col] = st.number_input(label, lo, hi, default, step=step)
        o2_device = st.checkbox("Tlenoterapia (tlen przez urządzenie)", value=False)

    with st.sidebar.expander("Skargi główne", expanded=True):
        curated = [c for c in COMPLAINT_LABELS_PL if c in all_complaints]
        complaints = st.multiselect(
            "Najczęstsze skargi", curated, format_func=lambda c: COMPLAINT_LABELS_PL[c],
        )
        others = [c for c in all_complaints if c not in COMPLAINT_LABELS_PL]
        complaints += st.multiselect(
            "Inne skargi (lista Yale)", others, format_func=lambda c: c.removeprefix("cc_"),
        )

    return {
        "age": age,
        "is_male": sex == "Mężczyzna",
        "arrival_mode": arrival_mode,
        "vitals": vitals,
        "o2_device": o2_device,
        "complaints": complaints,
        "arrival_time": datetime.combine(arrival_date, arrival_clock),
    }


def main() -> None:
    if st is None:
        raise SystemExit("Brak pakietu streamlit. Zainstaluj: pip install streamlit")

    st.set_page_config(
        page_title="SOR-AI: triaż ESI",
        page_icon="[+]",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("SOR-AI: asystent triażu ESI (TOPSOR)")
    st.markdown(
        "*Asystent decyzyjny dla pielęgniarek triażowych.* Model przewiduje poziom "
        "**Emergency Severity Index (ESI 1–5)** — ten sam algorytm, który polskie SOR-y "
        "stosują w systemie TOPSOR (kolory czerwony → niebieski) — i wyjaśnia decyzję: "
        "reguły ESI, SHAP, lokalny LLM (Ollama)."
    )
    st.caption(
        "Modele trenowane na danych Yale (USA) bez cech specyficznych dla USA (ubezpieczenie, "
        "rasa, język …). Wynik to sugestia — decyzję podejmuje pielęgniarka triażowa."
    )

    # ─── Sidebar — model ───
    st.sidebar.header("Model")
    cli = parse_cli_args()
    if cli.model:
        model_path = Path(cli.model)
        st.sidebar.markdown(f"Model z argumentu: `{model_path}`")
        if not model_path.exists():
            st.error(f"Nie znaleziono pliku modelu: `{model_path}`")
            return
    else:
        runs = list_model_runs()
        if not runs:
            st.error(
                f"Brak runów z zapisanym modelem w `{RUNS_DIR}`.\n\n"
                "Wytrenuj model: `python scripts/03_train.py --model xgboost`"
            )
            return
        choice = st.sidebar.selectbox(
            "Run",
            options=range(len(runs)),
            index=default_run_index(runs),
            format_func=lambda i: f"{runs[i]['run']}" + (" (próbka)" if runs[i]["is_sample"] else ""),
        )
        model_path = runs[choice]["path"]

    model = load_model_cached(str(model_path))
    if not model.feature_names:
        st.error("Model nie ma zapisanych nazw cech (`feature_names`). Wytrenuj go ponownie.")
        return
    st.sidebar.caption(f"{model.name} · zestaw cech: {model.feature_set} · {len(model.feature_names)} cech")
    if model.data_version and model.data_version != DATA_VERSION:
        st.sidebar.warning(
            f"Model trenowano na danych '{model.data_version}', bieżąca wersja danych to "
            f"'{DATA_VERSION}' — cechy mogą się różnić. Przetrenuj model."
        )

    use_ollama = st.sidebar.checkbox("Wyjaśnienie medyczne LLM (Ollama)", value=False)
    ollama_model_name = OLLAMA_DEFAULT_MODEL
    if use_ollama:
        ollama_model_name = st.sidebar.text_input("Model Ollama", OLLAMA_DEFAULT_MODEL)

    # ─── Dane pacjenta ───
    all_complaints = complaint_columns(model.feature_names)
    form = patient_form(all_complaints)

    if not st.sidebar.button("Klasyfikuj pacjenta", type="primary", use_container_width=True):
        st.info("Wprowadź dane pacjenta w panelu po lewej, następnie kliknij **Klasyfikuj pacjenta**.")
        return

    if form["age"] < 18:
        st.warning("Model trenowano na dorosłych (Yale: wiek ≥ 18) — wynik dla dziecka jest niewiarygodny.")
    if model.feature_set == "full":
        st.warning("Model używa zestawu 'full' — historia pacjenta (choroby, leki, badania) jest tu nieznana (NaN).")

    patient_row = build_patient_row(all_complaints=all_complaints, **form)
    X_patient = prepare_model_input(patient_row, model.feature_names)
    pred = predict_levels(model, X_patient)
    proba = pred["proba"]
    patient_dict = patient_row.iloc[0].to_dict()

    col_left, col_right = st.columns([1, 1])

    # ─── LEWA: predykcja + reguły ───
    with col_left:
        st.subheader("Predykcja modelu")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(level_badge_html(pred["argmax"], "reguła argmax"), unsafe_allow_html=True)
        with c2:
            st.markdown(level_badge_html(pred["cost"], "reguła kosztowa"), unsafe_allow_html=True)
        st.caption(
            "Argmax = najbardziej prawdopodobny poziom. Reguła kosztowa minimalizuje oczekiwany koszt "
            "błędu (niedotriażowanie 3× droższe niż przetriażowanie). "
            f"Maks. czas do kontaktu z lekarzem (TOPSOR): {level_info(pred['cost'])['max_wait_minutes']} min "
            "wg reguły kosztowej."
        )
        st.markdown("**Rozkład prawdopodobieństw:**")
        render_probability_bars(proba)

        st.subheader("Weryfikacja regułowa (ESI)")
        rule_check = rule_based_triage(patient_dict)
        if rule_check["suggested_class_idx"] is None:
            st.success(rule_check["description"])
        else:
            prefix = "rozważ " if rule_check["advisory"] else ""
            st.markdown(
                level_badge_html(rule_check["suggested_class_idx"], f"{prefix}punkt {rule_check['decision_point']}"),
                unsafe_allow_html=True,
            )
            for rule in rule_check["all_violations"]:
                value = f" ({rule['value']})" if rule.get("type") == "vital" else ""
                st.markdown(f"- [ESI {rule['esi']}, punkt {rule['decision_point']}] {rule['description']}{value}")
        for note in rule_check["notes"]:
            st.caption(note)

        st.subheader("Zgodność model ↔ reguły")
        for rule_name, label in (("argmax", "argmax"), ("cost", "kosztowa")):
            consistency = check_consistency(pred[rule_name], rule_check["suggested_class_idx"])
            text = f"**{label}:** {consistency['verdict']}"
            if consistency["undertriage"]:
                st.error(text)
            elif consistency["agree"]:
                st.success(text)
            else:
                st.info(text)

    # ─── PRAWA: SHAP + Ollama ───
    with col_right:
        st.subheader("Wyjaśnienie SHAP")
        explain_rule = st.radio(
            "Wyjaśnij poziom wskazany przez regułę", ["cost", "argmax"], horizontal=True,
            format_func=lambda r: "kosztową" if r == "cost" else "argmax",
        )
        explained_idx = pred[explain_rule]
        shap_exp = None
        try:
            explainer = init_shap_explainer(str(model_path), model)
            shap_exp = explainer.explain_patient(X_patient, top_n=10, class_idx=explained_idx)
        except ExplainerUnavailable as e:
            st.warning(f"SHAP niedostępny dla tego modelu: {e}")
        except Exception as e:
            st.error(f"Błąd SHAP: {e}")

        if shap_exp:
            st.caption(f"Metoda: {shap_exp['method']} · jednostki: {shap_exp['output_space']}")
            name = CLASS_NAMES_PL[explained_idx]
            st.markdown(f"**Cechy podnoszące wynik poziomu {name}:**")
            for f in shap_exp["top_features_for"][:6]:
                st.markdown(f"- `{f['feature']}` = {format_value(f['patient_value'])} → **{f['shap_value']:+.3f}**")
            st.markdown(f"**Cechy obniżające wynik poziomu {name}:**")
            if shap_exp["top_features_against"]:
                for f in shap_exp["top_features_against"][:4]:
                    st.markdown(f"- `{f['feature']}` = {format_value(f['patient_value'])} → **{f['shap_value']:+.3f}**")
            else:
                st.markdown("(brak)")
            with tempfile.TemporaryDirectory() as tmp:
                png = shap_waterfall_plot(explainer, X_patient, class_idx=explained_idx,
                                          save_path=Path(tmp) / "waterfall.png", max_display=12)
                if png is not None:
                    st.image(str(png))

        if use_ollama:
            st.subheader("Wyjaśnienie medyczne (Ollama)")
            ollama = OllamaMedicalExplainer(model_name=ollama_model_name)
            if not ollama.is_available():
                st.error(
                    f"Ollama niedostępne lub model '{ollama_model_name}' nie zainstalowany.\n"
                    f"Dostępne modele: {ollama.list_available_models() or 'brak'}\n\n"
                    f"Aby aktywować:\n```\nollama serve\nollama pull {ollama_model_name}\n```"
                )
            else:
                with st.spinner("Generowanie wyjaśnienia medycznego…"):
                    text = ollama.explain(
                        patient_data={**patient_dict, "arrival_mode": ARRIVAL_LABELS_PL[form["arrival_mode"]]},
                        predicted_class=CLASS_NAMES[explained_idx],
                        predicted_class_pl=CLASS_NAMES_PL[explained_idx],
                        probabilities=dict(zip(CLASS_NAMES, proba.tolist())),
                        shap_explanation=shap_exp,
                        rule_check=rule_check,
                    )
                    st.markdown(text)

    # ─── Stopka — dane pacjenta i wektor cech ───
    with st.expander("Wiersz pacjenta i cechy modelu", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Wiersz kanoniczny (bez zerowych skarg):**")
            shown = {k: v for k, v in patient_dict.items() if not (k.startswith("cc_") and v == 0)}
            st.json({k: (None if pd.isna(v) else v) for k, v in shown.items()})
            st.markdown("**Weryfikacja regułowa (tekst):**")
            st.text(explain_rule_decision(rule_check))
        with c2:
            st.markdown("**Niezerowe cechy modelu:**")
            row = X_patient.iloc[0]
            st.dataframe(row[(row != 0) & row.notna()].rename("wartość"))
            st.caption(f"Brakujące (NaN): {int(row.isna().sum())} z {len(row)} cech")


if __name__ == "__main__":
    main()
