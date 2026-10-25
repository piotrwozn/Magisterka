"""
Pamięć podręczna odpowiedzi modeli (JSONL) — wznowienie po przerwaniu i powtórny raport bez kosztów.

    from src.llm_eval.cache import ResponseCache, prompt_hash

    cache = ResponseCache(path)                 # runs/llm_bakeoff/cache/responses.jsonl
    h = prompt_hash(messages, tools=tools, think="low", options=opts)
    hit = cache.get("exams", "glm-5.3:cloud", "standard:<uuid>", h)   # dict ChatResult albo None
    cache.put("exams", "glm-5.3:cloud", "standard:<uuid>", h, result)  # ChatResult albo dict

Klucz = (część, model, id pozycji, skrót promptu). Skrót obejmuje wiadomości, narzędzia, `think`
i opcje — zmiana promptu albo poziomu rozumowania to automatycznie nowa pozycja, a nie stara
odpowiedź. Zapisywane są TYLKO udane odpowiedzi (error = None): błędy sieci / limitów są
ponawiane przy następnym uruchomieniu.

Plik jest dopisywany linia po linii (flush po każdej), więc przerwanie (Ctrl+C, restart) gubi
co najwyżej ostatnią, uciętą linię — przy wczytaniu jest pomijana (ostrzeżenie w logu).
Kolejne zapisy tego samego klucza: wygrywa ostatni. `fresh=True` (CLI --fresh) — odczyt
ignoruje istniejące wpisy, nowe odpowiedzi są dopisywane (nadpisują stare przy następnym odczycie).
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

CacheKey = tuple[str, str, str, str]


def prompt_hash(messages: Any, *, tools: Any = None, think: Any = None, options: Any = None,
                extra: Any = None) -> str:
    """Skrót SHA-256 (16 znaków hex) kanonicznego JSON-a zapytania."""
    payload = {"messages": messages, "tools": tools, "think": think, "options": options, "extra": extra}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ResponseCache:
    """Pamięć podręczna JSONL bezpieczna wątkowo (zob. docstring modułu)."""

    def __init__(self, path: Path | str, fresh: bool = False) -> None:
        self.path = Path(path)
        self.fresh = bool(fresh)
        self._lock = threading.Lock()
        self._data: dict[CacheKey, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if not self.fresh:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        bad = 0
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    key = (str(rec["part"]), str(rec["model"]), str(rec["item_id"]), str(rec["prompt_hash"]))
                    result = rec["result"]
                except (ValueError, KeyError, TypeError):
                    bad += 1
                    continue
                if isinstance(result, dict) and not result.get("error"):
                    self._data[key] = result
        if bad:
            log.warning(f"Pamięć podręczna {self.path}: pominięto {bad} uszkodzonych linii (np. przerwany zapis)")

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: CacheKey) -> bool:
        return key in self._data

    def get(self, part: str, model: str, item_id: str, phash: str) -> dict[str, Any] | None:
        """Zapisany wynik (dict pól ChatResult) albo None."""
        with self._lock:
            hit = self._data.get((str(part), str(model), str(item_id), str(phash)))
            if hit is None:
                self.misses += 1
            else:
                self.hits += 1
            return hit

    def put(self, part: str, model: str, item_id: str, phash: str, result: Any,
            meta: dict[str, Any] | None = None) -> bool:
        """Dopisuje udaną odpowiedź (ChatResult albo dict); błędy nie są zapisywane → False."""
        data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        if data.get("error"):
            return False
        key = (str(part), str(model), str(item_id), str(phash))
        rec = {"part": key[0], "model": key[1], "item_id": key[2], "prompt_hash": key[3],
               "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "result": data}
        if meta:
            rec["meta"] = meta
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
            self._data[key] = data
            self.writes += 1
        return True

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._data), "hits": self.hits, "misses": self.misses, "writes": self.writes}


class JsonlWriter:
    """Dopisywanie rekordów JSON (UTF-8, linia na rekord, flush po każdym) — np. surowe odpowiedzi przebiegu."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
            self.count += 1


__all__ = ["JsonlWriter", "ResponseCache", "prompt_hash"]
