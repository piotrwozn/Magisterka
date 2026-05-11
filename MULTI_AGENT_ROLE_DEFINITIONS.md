# Specyfikacja Ról Multi-Agentowych (SOR-AI)

Niniejszy dokument szczegółowo rozpisuje obowiązki, formaty We/Wy oraz modele przypisane do każdego węzła w docelowej, 3-warstwowej architekturze LLM dla systemu triażu.

---

## WARSTWA 1: Zrozumienie (Percepcja)

### 🤖 AGENT RECEPCJONISTA (Triage Parser)
*   **Silnik:** Dowolny mały, ultra-szybki model "instruct" (np. `llama3.2:8b` lub lekki Qwen).
*   **Zadanie:** Translacja surowych danych i chaosu językowego na format maszynowy, zrozumiały dla modeli numerycznych.
*   **Wejście (Co dostaje):**
    *   Tekst wklepany przez pielęgniarkę (często pełen skrótów i błędów, np. *"pt. z OZW, duszno, BP wysokie, od 2h gorzej"*).
*   **Logika działania (Prompt Systemowy):**
    *   Przeanalizuj tekst i zidentyfikuj "Czerwone Flagi" wg słownika (np. "OZW" -> zawał).
    *   Usuń emocje i szum.
*   **Wyjście (Co oddaje):**
    *   Tylko i wyłącznie ustrukturyzowany słownik JSON, mapujący zmienne do treningu XGBoost, np. `{"is_chest_pain": 1, "is_dyspnea": 1, "time_of_onset_hours": 2.0}`.

---

## WARSTWA 2: Równoległa Dedukcja (Eksperci)

### 🏥 AGENT PROGNOSTA / WIZJONER (Medical Simulator)
*   **Silnik:** Model dotrenowany na danych medycznych i obrazowych (np. **`medgemma1.5:4b`**).
*   **Zadanie:** Ocena pacjenta pod kątem długoterminowej trajektorii klinicznej i nietypowych objawów wzrokowych. Odpowiada na pytanie: "Co mu będzie za 2 godziny?".
*   **Wejście:**
    *   Zdjęcie pacjenta / rany / wydruku EKG (jako obraz do VLM).
    *   Surowy tekst z opisem wywiadu.
*   **Logika działania (Prompt Systemowy):**
    *   Ignoruj procedury i regulaminy szpitalne. Jesteś lekarzem i patologiem.
    *   Określ prawdopodobieństwo wystąpienia ciężkich stanów (sepsa, wstrząs, nagłe zatrzymanie krążenia) w ciągu 2h, jeśli pacjent nie zostanie obsłużony.
*   **Wyjście:**
    *   Krótki raport medyczny z oszacowanym % ryzyka załamania. Przykład: *"Rozpoznaję na zdjęciu sinicę. Ryzyko krytycznej hipoksji w ciągu godziny: 85%."*

### 🧮 AGENT KLASYFIKATOR (Tabular Analyzer)
*   **Silnik:** Model doskonale czytający logi programistyczne (np. **`qwen2.5-coder:7b`**).
*   **Zadanie:** Odpalenie modeli Machine Learning (XGBoost, LightGBM, Random Forest) na danych JSON z Recepcji i przetłumaczenie suchej matematyki (SHAP) na jednoznaczny werdykt "tu i teraz".
*   **Wejście:**
    *   JSON z wynikami i wartościami SHAP z XGBoost, LightGBM i Random Forest.
*   **Logika działania (Prompt Systemowy):**
    *   Jesteś chłodnym algorytmem matematycznym. 
    *   Jeśli LightGBM znalazł SHAP > |2.0| dla zagrożenia, ufaj mu ponad XGBoostem.
    *   Odrzuć RF w przypadkach skrajnych, słuchaj RF w przypadkach granicznych (MTS 3 vs 4).
*   **Wyjście:**
    *   Syntetyczne, ustrukturyzowane oświadczenie. Przykład: *"Modele optymalizacyjne ustalają obecny status MTS na Żółty. Decyzja poparta: Tętno 130 (SHAP XGBoost +1.2)."*

---

## WARSTWA 3: Wyrok (Rozwiązanie Konfliktu)

### 👨‍⚖️ AGENT KONSYLIUM (Meta-Orchestrator)
*   **Silnik:** Największy, najbardziej zaawansowany model dedukcyjny na serwerze (np. **`qwen3.6`** 32B/72B).
*   **Zadanie:** Wcielenie się w rolę Głównego Ordynatora. Rozstrzyga o losie pacjenta rozwiązując spór pomiędzy Prognostą i Klasyfikatorem. Gwarantuje bezpieczeństwo wdrożenia poprzez "Explanation-Augmented Prompting" (zapobiega halucynacjom).
*   **Wejście:**
    *   Raport od Agenta Prognosty (`medgemma1.5`).
    *   Raport od Agenta Klasyfikatora (Wraz z wartościami SHAP).
*   **Logika działania (Prompt Systemowy):**
    *   Otrzymujesz diagnozy swoich asystentów. Twoim jedynym celem jest nadanie kodu Triażu (0-4).
    *   MUSISZ opierać się wyłącznie na dowodach w logach.
    *   Zasada nadrzędna: Oczekiwany stan pacjenta za 2 godziny (Prognosta) zawsze podnosi priorytet, jeśli obecny stan (Klasyfikator) jest stabilny, ale ryzyko degradacji przekracza 50%.
*   **Wyjście:**
    *   Ścisły, poprawny programistycznie plik JSON, gotowy do wyświetlenia na ekranie systemu SOR-AI, zawierający decyzję i audyt decyzyjny (Dlaczego tak uznano na bazie SHAP).
