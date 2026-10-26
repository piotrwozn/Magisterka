"""
Cennik modeli (USD za 1 mln tokenów wejścia / wyjścia) i szacunek kosztu porównania.

Domyślne ceny: katalog Ollama Cloud z 2026-10-14. Przy planie abonamentowym (Pro) zapytania
nie są fakturowane per token — koszt w raporcie to ORIENTACYJNY koszt tego samego pomiaru
według cennika API (do porównania modeli i do rozdziału „koszty” w pracy).

    prices = load_prices("ceny.json")          # nadpisania: {"glm-5.3": {"input": 1.4, "output": 4.4}}
    cost_usd("glm-5.3:cloud", 12_000, 3_000, prices)
    estimate_cost("kimi-k3:cloud", n_calls=500, prompt_tokens=400, completion_tokens=300, prices=prices)

Nazwy modeli są porównywane bez przyrostka chmurowego (`glm-5.3:cloud` = `glm-5.3`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.llm_eval.client import canonical_model

PRICING_SOURCE: str = "katalog Ollama Cloud, 2026-10-14"

# USD / 1M tokenów: (wejście, wyjście)
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "gpt-oss:120b": {"input": 0.15, "output": 0.60},
    "glm-5.3": {"input": 1.40, "output": 4.40},
    "deepseek-v4.1-flash": {"input": 0.30, "output": 1.20},
    "kimi-k3": {"input": 3.00, "output": 15.00},
}

# Założenia do szacunku przed uruchomieniem (--dry-run): tokeny na jedno wywołanie / przypadek.
# Egzaminy: prompt liczony z długości tekstu (≈ 3,5 znaku na token dla polskiego), wyjście = krótkie
# rozumowanie „low” + odpowiedź. ESI: opis przypadku + kontekst decyzji; narzędzia: do 6 kroków agenta.
CHARS_PER_TOKEN: float = 3.5
TOKEN_ASSUMPTIONS: dict[str, dict[str, float]] = {
    "exams": {"prompt_tokens": 0.0, "completion_tokens": 350.0, "calls_per_item": 1.0},
    "esi": {"prompt_tokens": 2500.0, "completion_tokens": 900.0, "calls_per_item": 1.2},
    "tools": {"prompt_tokens": 3500.0, "completion_tokens": 400.0, "calls_per_item": 5.0},
}


def load_prices(path: Path | str | None = None) -> dict[str, dict[str, float]]:
    """Domyślny cennik + nadpisania z pliku JSON {model: {"input": $, "output": $}} (UTF-8)."""
    prices = {k: dict(v) for k, v in DEFAULT_PRICES.items()}
    if path is None:
        return prices
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: oczekiwano obiektu JSON {{model: {{input, output}}}}")
    for model, entry in data.items():
        if not isinstance(entry, dict) or not {"input", "output"} <= set(entry):
            raise ValueError(f"{path}: wpis '{model}' musi mieć pola 'input' i 'output' (USD / 1M tokenów)")
        prices[canonical_model(model)] = {"input": float(entry["input"]), "output": float(entry["output"])}
    return prices


def price_for(model: str, prices: dict[str, dict[str, float]] | None = None) -> dict[str, float] | None:
    """Cena modelu (dopasowanie po nazwie bez przyrostka chmurowego) albo None."""
    table = DEFAULT_PRICES if prices is None else prices
    name = canonical_model(model)
    if name in table:
        return table[name]
    base = name.split(":", 1)[0]
    return table.get(base)


def cost_usd(model: str, prompt_tokens: float | None, completion_tokens: float | None,
             prices: dict[str, dict[str, float]] | None = None) -> float | None:
    """Koszt w USD; None, gdy model nie ma ceny."""
    price = price_for(model, prices)
    if price is None:
        return None
    return (float(prompt_tokens or 0) * price["input"] + float(completion_tokens or 0) * price["output"]) / 1e6


def estimate_tokens(text: str) -> int:
    """Przybliżona liczba tokenów tekstu (polski ≈ 3,5 znaku na token)."""
    return int(round(len(text or "") / CHARS_PER_TOKEN))


def estimate_cost(model: str, n_calls: float, prompt_tokens: float, completion_tokens: float,
                  prices: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """Szacunek: n_calls wywołań po prompt_tokens / completion_tokens (średnio)."""
    p_total = float(n_calls) * float(prompt_tokens)
    c_total = float(n_calls) * float(completion_tokens)
    return {"model": model, "calls": float(n_calls), "prompt_tokens": p_total, "completion_tokens": c_total,
            "cost_usd": cost_usd(model, p_total, c_total, prices)}


def estimate_part(part: str, model: str, n_items: int, prices: dict[str, dict[str, float]] | None = None,
                  prompt_tokens: float | None = None) -> dict[str, Any]:
    """Szacunek dla części (exams / esi / tools) wg TOKEN_ASSUMPTIONS (prompt_tokens nadpisuje założenie)."""
    a = TOKEN_ASSUMPTIONS.get(part, TOKEN_ASSUMPTIONS["esi"])
    calls = n_items * a["calls_per_item"]
    out = estimate_cost(model, calls, a["prompt_tokens"] if prompt_tokens is None else prompt_tokens,
                        a["completion_tokens"], prices)
    out["part"] = part
    return out


def format_usd(value: float | None) -> str:
    """'$1.23' / '$0.004' / '—' (brak ceny)."""
    if value is None:
        return "—"
    return f"${value:.2f}" if value >= 0.1 else f"${value:.4f}"


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_PRICES",
    "PRICING_SOURCE",
    "TOKEN_ASSUMPTIONS",
    "cost_usd",
    "estimate_cost",
    "estimate_part",
    "estimate_tokens",
    "format_usd",
    "load_prices",
    "price_for",
]
