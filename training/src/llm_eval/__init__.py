"""
Porównanie modeli językowych (bake-off) do polskich wyjaśnień decyzji triażu ESI.

Kryterium wyboru (uzgodnione): model z najlepszym wynikiem w NASZYM pomiarze — przede wszystkim
trafność na polskich egzaminach lekarskich w protokole „abstaining_substitution” (LEK + PES),
potwierdzona testem na polskich przypadkach triażu (część „esi”) i testem użycia narzędzi
(część „tools”). Uruchomienie: `python scripts/12_llm_bakeoff.py run …` (opis: docs/LLM_BAKEOFF.md).

Moduły
    client      — OllamaChat: /api/chat z ponowieniami, limitem równoległości, trybem chmury
    cache       — pamięć podręczna odpowiedzi (JSONL) → wznawianie po przerwaniu
    pricing     — cennik $/1M tokenów i szacunek kosztu
    exams       — część „exams” (NASK-PIB/Reassess-Polish-Medical-Exams, dwa protokoły)
    report      — report.md (tabele do pracy) i wybór modelu
    cc_labels_pl — polskie nazwy wszystkich skarg głównych i cech (+ synonimy do pokrycia SHAP)
    esi_cases   — przypadki triażowe z VAL: receptura decyzji, SHAP, reguły bezpieczeństwa
    explain_task — część „esi” (wyjaśnienie decyzji przez submit_explanation, metryki)
    ratings     — ślepy arkusz ocen ekspertów (HTML/CSV + key.json) i ich zliczanie (score-ratings)
    knowledge_notes — parafraza punktów decyzyjnych ESI v4 i 15 notatek (narzędzia agenta)
    tools_task  — część „tools” (agent: explain_model, what_if, esi_criteria, search_knowledge)

Pakiet nie importuje modułów przy starcie (część zależy od wytrenowanych runów i dodatków).
"""
