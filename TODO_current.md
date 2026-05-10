# TODO_current.md: Realistyczny Plan Implementacji "Tu i Teraz"

Ten plik zawiera listę modułów z głównego `TODO.md`, które jesteś w stanie zaimplementować i włączyć do obecnego projektu magisterskiego **samodzielnie, bez dostępu do hardware'u szpitalnego, zewnętrznych baz danych (e-Zdrowie), czy fizycznych symulacji**.

## Co MOŻESZ zaimplementować natychmiast (Soft-Features & Architecture):

### 1. Podstawowa Architektura Orkiestratora (Agent Konsylium - Wersja Light)
Nie potrzebujesz szpitala, by udowodnić koncepcję SHAP-Groundingu.
*   **Jak to zrobić:** Napisać skrypt w Pythonie (korzystając z LangChain lub API Ollamy z modelem Llama-3 uruchamianym lokalnie), który przyjmuje na wejście:
    1.  Parametry pacjenta (z Twojego pliku Parquet).
    2.  Decyzję XGBoosta (którego właśnie trenujesz).
    3.  Wartości SHAP (które już potrafisz generować).
*   **Cel:** Zbudować prompt systemowy: *"Otrzymałeś decyzję klasyfikatora i dowody SHAP. Napisz po polsku uzasadnienie decyzji dla ratownika, nie używając cech o SHAP < 0.5"*.

### 2. Symulacja Modułu Behawioralnego ("Frequent Flyers")
Nie masz dostępu do bazy e-Zdrowie, ale możesz *wygenerować* sztuczne dane (Synthetic Data) do eksperymentu w pracy magisterskiej.
*   **Jak to zrobić:** 
    1.  Dodać do Twojego pipeline'u danych sztucznie wygenerowaną kolumnę `historical_visits_last_30_days`.
    2.  Dla części pacjentów z grupy "Zielonych/Niebieskich" z błahymi objawami, ustawić tę wartość na wysoką (np. >5).
    3.  Wytrenować prosty model np. Isolation Forest lub dodać tę kolumnę do XGBoosta, by udowodnić, że system potrafi używać historii pacjenta do "zbijania" priorytetów u symulantów.
*   **Cel:** Dowód koncepcji (Proof-of-Concept), że moduł behawioralny radykalnie zmienia politykę Overtriage'u.

### 3. Agent Ekstraktor NLP (Na danych syntetycznych)
Nie masz historycznych, polskich notatek pielęgniarskich ze szpitala.
*   **Jak to zrobić:**
    1.  Użyć modelu LLM (np. Llama-3/ChatGPT), by wygenerować 500 realistycznych notatek ratowniczych po polsku w stylu *"pt. przywieziony przez zrm po updaku, skarży się na ból klatki, bp stabilne"*.
    2.  Podpiąć te teksty do pacjentów z Twojego obecnego zbioru.
    3.  Użyć modelu (np. BioClinicalBERT lub lokalnego LLM), by ekstrahował ustrukturyzowane flagi (np. `has_chest_pain=1`).
    4.  Dorzucić te flagi do wejścia XGBoosta.
*   **Cel:** Udowodnienie w pracy magisterskiej, że fuzja NLP+Tabele (choćby na danych syntetycznych) faktycznie poprawia QWK.

### 4. Active Learning ("Human-in-the-loop" - Wersja Streamlit)
To możesz zaimplementować bezpośrednio w Twojej aplikacji Streamlit w `app/streamlit_app.py`.
*   **Jak to zrobić:**
    1.  Jeśli XGBoost na podstawie wprowadzonych w UI parametrów wypluwa np. pewność 49% dla Żółtego i 51% dla Pomarańczowego (niskie prawdopodobieństwa klas), aplikacja *blokuje* podanie ostatecznego wyniku.
    2.  Wyświetla się wielki czerwony komunikat: *"System niepewny. Proszę zmierzyć X i uzupełnić dane"*.
*   **Cel:** Zaprezentowanie na obronie, jak system odmawia diagnozowania "w ciemno" chroniąc pacjenta i zrzucając odpowiedzialność z AI na pomiary.

## Czego NIE MOŻESZ zaimplementować (Wymaga sprzętu/zgód):
*   **Kamery Termowizyjne / rPPG:** Wymaga fizycznego montażu na wejściu i zgód bioetycznych na badanie pacjentów.
*   **Prawdziwe Bazy EHR / e-Zdrowie P1:** Niedostępne bez umów z rządem/szpitalem.
*   **Survival DL (Time-to-Deterioration):** Twój aktualny zbiór z Yale to klasyfikacja "w Minucie Zero". Nie posiada krzywych czasowych, więc nie wytrenujesz modeli przeżycia (chyba że zasymulujesz całą fizjologię rozwoju choroby, co jest zbyt skomplikowane na tym etapie).

## Rekomendowany "Next Step" na teraz:
Najlepszym kierunkiem badawczym, który "sprzeda się" na obronie, a nie wymaga infrastruktury, jest **Punkt 1 z tej listy (Orkiestrator z SHAP-Groundingiem)** zintegrowany z Twoim Streamlitem. 
