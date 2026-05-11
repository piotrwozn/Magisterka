# Ostateczny Stack Architektoniczny Multi-Agent SOR-AI (Lokalny RTX 4080 SUPER)

Zatwierdzony i zoptymalizowany podział potężnych, lokalnych modeli na konkretne warstwy.

## Warstwa 1: Ingestia i Percepcja
**Model 1 (Agent Recepcjonista): `llama3` (Wersja 8B)**
*   **Zadanie:** Parser surowego tekstu do formatu JSON. Szybki, bezlitosny, bezbłędny w formatowaniu. Przygotowuje "dane wejściowe" dla kolejnych warstw.

## Warstwa 2: Równolegli Eksperci Logiczni
**Model 2 (Agent Klasyfikator - Obsługa Drzew Decyzyjnych): `qwen3-coder:30b`**
*   **Zadanie:** Analiza wyników modeli tabularycznych "Tu i Teraz" (XGBoost, LightGBM, Random Forest).
*   **Dlaczego on:** Ponieważ to model typu "Coder" (30 miliardów parametrów). Rozumie ułamki dziesiętne, wagi SHAP i logikę decyzyjną lepiej niż modele medyczne.

**Model 3 (Agent Prognosta - Obsługa Modeli Predykcyjnych/Wizualnych): `medgemma1.5:4b` lub `medgemma:27b` (v1)**
*   **Zadanie:** Ocena ryzyka długoterminowego ("Co będzie z pacjentem za 2 godziny?") na podstawie objawów ukrytych, np. z obrazów VLM (Vision).
*   **Dlaczego on:** MedGemma to model multimodalny (VLM) dostrojony stricte na wiedzy medycznej (PubMed, obrazy RTG, zdjęcia zmian skórnych). Doskonale zrekompensuje braki typowych LLM-ów w rozumieniu patofizjologii. Posiada gigantyczne okno kontekstowe (128K), więc "połknie" w ułamek sekundy całą historyczną kartotekę z bazy EHR pacjenta bez zająknięcia.

## Warstwa 3: Synteza Ostateczna (Wyrok)
**Model 4 (Agent Konsylium / Orkiestrator): `qwen3.6` (Wersja 32B)**
*   **Zadanie:** Główny Ordynator. Odbiera raport z `qwen3-coder` (matematyka na teraz) i z Modelu 3 (prognoza). Ignoruje własną wiedzę medyczną, aby uniknąć halucynacji. Podejmuje logiczny werdykt "MTS = Pomarańczowy" w oparciu o "SHAP Grounding".
