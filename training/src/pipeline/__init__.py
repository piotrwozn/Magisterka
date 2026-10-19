"""
Nocny przebieg jednym poleceniem (`scripts/run_all.py`) i kontrola środowiska (`scripts/00_check_env.py`).

    orchestrator — kroki w osobnych procesach, wznawianie po znacznikach DONE.json, tory GPU/CPU,
    preflight    — Python, pakiety, RAM, dysk, GPU (próbny trening), dane, ścieżki, kodowanie,
    keepawake    — blokada uśpienia systemu na czas przebiegu.

Moduły importuj bezpośrednio (`from src.pipeline.orchestrator import main`) — pakiet nic nie
importuje przy starcie, żeby `00_check_env.py` działał nawet przy brakujących zależnościach.
"""
