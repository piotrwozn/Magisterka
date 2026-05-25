"""
Ollama medical reasoning — lokalny LLM generujący wyjaśnienia kliniczne.

Łączy:
    - Dane pacjenta (vital signs, chief complaint, demografia)
    - Predykcję ML modelu
    - SHAP explanation (top features)
    - MTS rule check
    →
    Wyjaśnienie po polsku w stylu lekarza medycyny ratunkowej:
        "Pacjent zaklasyfikowany jako Red ze względu na...
         Sugerowane badania diagnostyczne:..."
"""

from __future__ import annotations

import json
from typing import Any

import requests

from src.utils.config import (
    OLLAMA_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_MAX_TOKENS,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_S,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────
SYSTEM_PROMPT_PL = """Jesteś doświadczonym lekarzem medycyny ratunkowej pracującym na polskim Szpitalnym Oddziale Ratunkowym (SOR). \
Specjalizujesz się w protokole Manchester Triage System (MTS). Twoje wyjaśnienia są zwięzłe, profesjonalne, \
oparte na evidence-based medicine i zawsze ostrożne (bezpieczeństwo pacjenta > efektywność)."""


PATIENT_PROMPT_TEMPLATE = """## DANE PACJENTA NA TRIAŻU
- Wiek: {age}
- Płeć: {sex}
- Ciśnienie tętnicze: {sbp}/{dbp} mmHg
- Tętno: {pulse} bpm
- Częstość oddechów: {resp}/min
- Saturacja O2: {o2sat}%
- Temperatura: {temp}°C
- Skala bólu (0-10): {pain}
- Główna skarga: {chief_complaint}

## KLASYFIKACJA SYSTEMU AI
- **Przydzielona kategoria MTS**: {predicted_class} ({predicted_class_pl})
- **Pewność predykcji**: {confidence:.1%}
- **Probabilities (wszystkie klasy)**: {probabilities}

## TOP CECHY POPYCHAJĄCE DECYZJĘ (SHAP)
{shap_top_for}

## CECHY DZIAŁAJĄCE PRZECIW (SHAP)
{shap_top_against}

## REGUŁOWA WERYFIKACJA MTS
{rule_check}

## TWOJE ZADANIE
Wyjaśnij zwięźle (max 250 słów, w języku POLSKIM):
1. **MEDYCZNE UZASADNIENIE** klasyfikacji — odnieś się do konkretnych objawów i parametrów.
2. **POTENCJALNE ZAGROŻENIA** — na co pielęgniarka triażowa powinna zwrócić uwagę?
3. **OCENA ZGODNOŚCI** — czy predykcja AI wydaje się klinicznie zasadna? Jeśli nie — dlaczego?
4. **SUGEROWANE DALSZE POSTĘPOWANIE** — pilne badania diagnostyczne, monitoring.

Format: krótkie sekcje z nagłówkami **pogrubionymi**. Język medyczny, ale zrozumiały.
"""


# ─────────────────────────────────────────
# Klasa explainer
# ─────────────────────────────────────────
class OllamaMedicalExplainer:
    """
    Lokalny LLM (Ollama) jako warstwa medical reasoning.

    Wymaga uruchomionego Ollama:
        ollama serve
        ollama pull mistral  # lub llama3, gemma2:9b, medllama2
    """

    def __init__(
        self,
        model_name: str = OLLAMA_DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = OLLAMA_TEMPERATURE,
        max_tokens: int = OLLAMA_MAX_TOKENS,
        timeout: int = OLLAMA_TIMEOUT_S,
        system_prompt: str | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt or SYSTEM_PROMPT_PL

    # ──────── Health check ────────
    def is_available(self) -> bool:
        """Sprawdza czy Ollama jest uruchomione i model dostępny."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if response.status_code != 200:
                return False
            models = [m["name"] for m in response.json().get("models", [])]
            # Akceptuj 'mistral' i 'mistral:latest' jako to samo
            base = self.model_name.split(":")[0]
            return any(m.split(":")[0] == base for m in models)
        except (requests.RequestException, ValueError):
            return False

    def list_available_models(self) -> list[str]:
        """Lista modeli zainstalowanych w Ollamie."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except (requests.RequestException, ValueError):
            return []

    # ──────── Główna metoda ────────
    def explain(
        self,
        patient_data: dict[str, Any],
        predicted_class: str,
        predicted_class_pl: str,
        probabilities: dict[str, float],
        shap_explanation: dict | None = None,
        rule_check: dict | None = None,
    ) -> str:
        """
        Generuje medyczne wyjaśnienie triażu.

        Parameters
        ----------
        patient_data : dict
            Dane pacjenta (vital signs, chief complaint, demografia).
        predicted_class : str
            np. 'Orange'
        predicted_class_pl : str
            np. 'Pomarańczowy (Bardzo pilny)'
        probabilities : dict
            {class_name: probability}
        shap_explanation : dict, optional
            Z SHAPTriageExplainer.explain_patient().
        rule_check : dict, optional
            Z mts_rules.rule_based_triage().

        Returns
        -------
        str — sformatowane wyjaśnienie po polsku.
        """
        prompt = self._build_prompt(
            patient_data=patient_data,
            predicted_class=predicted_class,
            predicted_class_pl=predicted_class_pl,
            probabilities=probabilities,
            shap_explanation=shap_explanation,
            rule_check=rule_check,
        )

        return self._call_ollama(prompt)

    # ──────── Wewnętrzne ────────
    def _build_prompt(
        self,
        patient_data: dict,
        predicted_class: str,
        predicted_class_pl: str,
        probabilities: dict,
        shap_explanation: dict | None,
        rule_check: dict | None,
    ) -> str:
        """Składa prompt dla LLM."""
        # SHAP top features
        if shap_explanation:
            shap_for = self._format_shap_features(shap_explanation.get("top_features_for", []), max_n=5)
            shap_against = self._format_shap_features(shap_explanation.get("top_features_against", []), max_n=3)
        else:
            shap_for = "(brak danych SHAP)"
            shap_against = "(brak danych SHAP)"

        # Rule check
        if rule_check:
            from src.explain.mts_rules import explain_rule_decision
            rule_text = explain_rule_decision(rule_check)
        else:
            rule_text = "(brak weryfikacji regułowej)"

        confidence = max(probabilities.values()) if probabilities else 0.0
        proba_str = ", ".join(f"{k}: {v:.1%}" for k, v in probabilities.items())

        return PATIENT_PROMPT_TEMPLATE.format(
            age=patient_data.get("age", "n/a"),
            sex=patient_data.get("sex", "n/a"),
            sbp=patient_data.get("triage_sbp", patient_data.get("sbp", "n/a")),
            dbp=patient_data.get("triage_dbp", patient_data.get("dbp", "n/a")),
            pulse=patient_data.get("triage_pulse", patient_data.get("pulse", "n/a")),
            resp=patient_data.get("triage_resp", patient_data.get("resp", "n/a")),
            o2sat=patient_data.get("triage_o2sat", patient_data.get("o2sat", "n/a")),
            temp=patient_data.get("triage_temp", patient_data.get("temp", "n/a")),
            pain=patient_data.get("triage_pain", patient_data.get("pain", "n/a")),
            chief_complaint=patient_data.get("chief_complaint", "n/a"),
            predicted_class=predicted_class,
            predicted_class_pl=predicted_class_pl,
            confidence=confidence,
            probabilities=proba_str,
            shap_top_for=shap_for,
            shap_top_against=shap_against,
            rule_check=rule_text,
        )

    @staticmethod
    def _format_shap_features(features: list[dict], max_n: int = 5) -> str:
        """Formatuje listę cech SHAP do tekstu."""
        if not features:
            return "(brak)"
        lines = []
        for f in features[:max_n]:
            val = f.get("patient_value", "n/a")
            shap_val = f.get("shap_value", 0.0)
            lines.append(f"  - {f['feature']}: wartość={val}, wpływ SHAP={shap_val:+.3f}")
        return "\n".join(lines)

    def _call_ollama(self, prompt: str) -> str:
        """Wywołuje API Ollama (POST /api/generate)."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": self.system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": 0.9,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=self.timeout,
            )

            if response.status_code != 200:
                log.error(f"Ollama błąd HTTP {response.status_code}: {response.text[:200]}")
                return self._fallback_explanation(
                    error=f"HTTP {response.status_code}",
                )

            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError:
            log.error(f"Nie udało się połączyć z Ollama na {self.base_url}. Czy serwis jest uruchomiony?")
            return self._fallback_explanation(error="Ollama nieuruchomione")
        except requests.exceptions.Timeout:
            log.error(f"Timeout Ollama (>{self.timeout}s)")
            return self._fallback_explanation(error="Timeout")
        except (requests.RequestException, ValueError, KeyError) as e:
            log.error(f"Błąd Ollama: {e}")
            return self._fallback_explanation(error=str(e))

    @staticmethod
    def _fallback_explanation(error: str = "") -> str:
        """Generuje fallback gdy LLM nie działa."""
        return (
            f"⚠️ Wyjaśnienie medyczne LLM niedostępne (błąd: {error}).\n\n"
            "Zalecenie: Skorzystaj z wyjaśnienia regułowego (sekcja MTS) i SHAP. "
            "Aby aktywować wyjaśnienia LLM, uruchom Ollama:\n"
            "    ollama serve\n"
            "    ollama pull mistral"
        )

    # ──────── Streaming (opcjonalnie) ────────
    def explain_stream(
        self,
        patient_data: dict,
        predicted_class: str,
        predicted_class_pl: str,
        probabilities: dict,
        shap_explanation: dict | None = None,
        rule_check: dict | None = None,
    ):
        """
        Generator zwracający kawałki odpowiedzi (dla streamingu w UI).
        """
        prompt = self._build_prompt(
            patient_data, predicted_class, predicted_class_pl,
            probabilities, shap_explanation, rule_check,
        )

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": self.system_prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                stream=True,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        if "response" in chunk:
                            yield chunk["response"]
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except requests.RequestException as e:
            yield self._fallback_explanation(error=str(e))
