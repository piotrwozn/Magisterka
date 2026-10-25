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
    esi_cases, cc_labels_pl, explain_task, tools_task — części „esi” i „tools”

Pakiet nie importuje modułów przy starcie (część zależy od wytrenowanych runów i dodatków).
"""
