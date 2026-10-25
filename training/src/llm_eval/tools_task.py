"""
Część "tools" porównania LLM: tryb agenta na tych samych przypadkach co część "esi".

Model dostaje dane pacjenta [S1], decyzję systemu [S2] i reguły bezpieczeństwa [S4] (BEZ SHAP)
oraz narzędzia:
    explain_model(top_n?)       — top cechy SHAP decyzji (z przypadku, polskie nazwy),
    what_if(changes)            — ponowne przeliczenie RECEPTURY na zmienionym wierszu kanonicznym
                                  (cechy inżynieryjne przez src.features.engineering, połączenie,
                                  temperatura, reguła); wolno zmieniać tylko parametry życiowe, wiek,
                                  sposób przybycia, tlenoterapię i skargi główne,
    esi_criteria(level)         — własna parafraza punktów decyzyjnych ESI v4 (knowledge_notes.py),
    search_knowledge(query)     — wyszukiwanie w ~15 notatkach (zastępstwo RAG),
    submit_explanation(...)     — odpowiedź końcowa (schemat z części "esi").
Maks. 6 kroków (wywołań modelu). Każdy krok trafia do pamięci podręcznej osobno (skrót całej
historii), więc przerwany przebieg wznawia się od ostatniego kroku.

Metryki: ukończenie (poprawne submit_explanation w limicie kroków), odsetek poprawnych wywołań
narzędzi (znana nazwa + argumenty zgodne ze schematem), zmyślone nazwy narzędzi, średnia liczba
kroków, poprawność użycia what_if (wywołanie wykonane: dozwolone cechy i wiarygodne wartości),
metryki odpowiedzi końcowej jak w części "esi", opóźnienie, tokeny, koszt.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.engineering import O2_DEVICE_COL, VITALS
from src.llm_eval.cache import JsonlWriter, ResponseCache
from src.llm_eval.cc_labels_pl import CC_LABELS_PL, feature_label, fold_text
from src.llm_eval.client import ChatResult, parse_think
from src.llm_eval.esi_cases import (
    ARRIVAL_COLS,
    EsiCase,
    EsiContext,
    _rule_flags,
    fmt_num,
    format_feature_value,
    plural_pl,
    try_prepare_context,
)
from src.llm_eval.explain_task import (
    EXPLANATION_SCHEMA,
    SUBMIT_NAME,
    SUBMIT_TOOL,
    api_error,
    assistant_message,
    cached_chat,
    coerce_explanation,
    decimal_point,
    extract_json_object,
    int_text,
    items_frame,
    pct,
    run_items,
    score_explanation,
    skipped_result,
    summarize_explanations,
    user_message,
)
from src.llm_eval.knowledge_notes import esi_criteria, search_knowledge
from src.utils.config import CLASS_NAMES
from src.utils.logger import get_logger

log = get_logger(__name__)

PART = "tools"
RAW_FILE = "raw_tools.jsonl"
MAX_STEPS = 6

# ─────────────────────────────────────────
# what_if — dozwolone zmiany
# ─────────────────────────────────────────
# Zakresy wiarygodnych wartości (dorośli; temperatura w °C)
VITAL_RANGES: dict[str, tuple[float, float]] = {
    VITALS["hr"]: (20, 250), VITALS["sbp"]: (40, 300), VITALS["dbp"]: (20, 200), VITALS["rr"]: (4, 80),
    VITALS["o2"]: (50, 100), VITALS["temp"]: (25, 45), "age": (18, 110),
}
FEATURE_ALIASES: dict[str, str] = {
    **{v: v for v in VITALS.values()},
    "hr": VITALS["hr"], "tetno": VITALS["hr"], "heart_rate": VITALS["hr"], "puls": VITALS["hr"],
    "sbp": VITALS["sbp"], "cisnienie_skurczowe": VITALS["sbp"], "systolic": VITALS["sbp"],
    "dbp": VITALS["dbp"], "cisnienie_rozkurczowe": VITALS["dbp"], "diastolic": VITALS["dbp"],
    "rr": VITALS["rr"], "oddechy": VITALS["rr"], "czestosc_oddechow": VITALS["rr"], "resp": VITALS["rr"],
    "respiratory_rate": VITALS["rr"],
    "o2": VITALS["o2"], "spo2": VITALS["o2"], "saturacja": VITALS["o2"],
    "temp": VITALS["temp"], "temperatura": VITALS["temp"], "temperature": VITALS["temp"],
    O2_DEVICE_COL: O2_DEVICE_COL, "o2_device": O2_DEVICE_COL, "tlen": O2_DEVICE_COL,
    "tlenoterapia": O2_DEVICE_COL,
    "age": "age", "wiek": "age",
    "arrival": "arrival", "arrival_mode": "arrival", "przybycie": "arrival", "sposob_przybycia": "arrival",
    **{col: col for col in ARRIVAL_COLS.values()},
}
ARRIVAL_VALUES: dict[str, str] = {
    "ambulance": "ambulance", "karetka": "ambulance", "zrm": "ambulance", "ems": "ambulance",
    "zespol ratownictwa medycznego": "ambulance", "ratownictwo": "ambulance",
    "self": "self", "samodzielnie": "self", "walk-in": "self", "walk_in": "self", "car": "self",
    "other": "other", "inny": "other", "inne": "other", "policja": "other", "police": "other",
}
_LABEL_TO_CC: dict[str, str] = {fold_text(label): col for col, label in CC_LABELS_PL.items()}

# ─────────────────────────────────────────
# Narzędzia
# ─────────────────────────────────────────
TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "explain_model",
        "description": "Zwraca najważniejsze czynniki decyzji systemu wg analizy SHAP (polskie nazwy cech, "
                       "wartości pacjenta, kierunek wpływu na wyznaczony poziom ESI).",
        "parameters": {"type": "object", "properties": {
            "top_n": {"type": "integer", "minimum": 1, "maximum": 10,
                      "description": "Ile czynników zwrócić (domyślnie 5)."}}},
    }},
    {"type": "function", "function": {
        "name": "what_if",
        "description": (
            "Symulacja „co by było, gdyby”: przelicza decyzję systemu (te same modele i reguła) po zmianie "
            "danych pacjenta. Dozwolone klucze: triage_vital_hr (tętno /min), triage_vital_sbp i "
            "triage_vital_dbp (ciśnienie mmHg), triage_vital_rr (oddechy /min), triage_vital_o2 (SpO2 %), "
            "triage_vital_temp (°C), triage_vital_o2_device (tlenoterapia 0/1), age (lata, 18–110), "
            "arrival ('ambulance' | 'self' | 'other') oraz skargi główne cc_<nazwa> (1 = obecna, 0 = brak). "
            "Wartość null przy parametrze życiowym = brak pomiaru."),
        "parameters": {"type": "object", "required": ["changes"], "properties": {
            "changes": {"type": "object", "description": "Słownik {cecha: nowa wartość}, np. "
                                                         "{\"triage_vital_o2\": 88, \"cc_chestpain\": 1}."}}},
    }},
    {"type": "function", "function": {
        "name": "esi_criteria",
        "description": "Kryteria poziomu ESI 1–5 i właściwe punkty decyzyjne algorytmu ESI (A–D), skrótowo po polsku.",
        "parameters": {"type": "object", "required": ["level"], "properties": {
            "level": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Poziom ESI 1–5."}}},
    }},
    {"type": "function", "function": {
        "name": "search_knowledge",
        "description": "Wyszukuje krótkie notatki kliniczne (strefa zagrożenia, NEWS2, qSOFA, wskaźnik wstrząsowy, "
                       "ból w klatce, udar, zasoby ESI, TOPSOR, ograniczenia systemu…). Zwraca do 3 notatek z id.",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "Zapytanie po polsku."}}},
    }},
    SUBMIT_TOOL,
]
TOOL_NAMES: tuple[str, ...] = tuple(t["function"]["name"] for t in TOOLS)
_SCHEMAS: dict[str, dict[str, Any]] = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

SYSTEM_PROMPT_TOOLS_PL = """Jesteś asystentem, który wyjaśnia decyzje systemu wspomagania triażu na szpitalnym oddziale ratunkowym (SOR).
System uczenia maszynowego wyznaczył poziom pilności ESI (1 = najpilniejszy, 5 = najmniej pilny) według stałej, deterministycznej receptury. Wyjaśnij tę decyzję pielęgniarce lub ratownikowi triażowemu — po polsku, rzeczowo i zwięźle (3–6 zdań).

Masz narzędzia:
- explain_model — czynniki, które najbardziej wpłynęły na decyzję (SHAP),
- what_if — symulacja decyzji po zmianie danych pacjenta (np. innej saturacji),
- esi_criteria — kryteria poziomów i punktów decyzyjnych ESI,
- search_knowledge — krótkie notatki kliniczne,
- submit_explanation — odpowiedź końcowa.
Korzystaj z narzędzi, gdy pomagają uzasadnić decyzję; masz najwyżej 6 kroków. Zakończ wywołaniem submit_explanation.

Zasady:
1. Nie zmieniasz decyzji systemu: w polu poziom_esi podaj DOKŁADNIE poziom wyznaczony przez system.
2. Możesz wyłącznie DORADCZO zasugerować wyższy priorytet (niższy numer ESI); nigdy niższy.
3. Nie wymyślaj pomiarów, objawów ani wywiadu — korzystaj tylko z danych i wyników narzędzi. Wyniki what_if to scenariusze hipotetyczne, nie stan pacjenta.
4. Powołuj się na źródła przez identyfikatory w nawiasach kwadratowych: [S1], [S2], [S4] oraz identyfikatory zwrócone przez narzędzia (np. [T1], [W03])."""


def tools_user_message(case: EsiCase, recipe_text: str | None) -> str:
    return user_message(case, recipe_text, with_shap=False,
                        closing="Wyjaśnij decyzję systemu. Użyj narzędzi, jeśli są potrzebne, i zakończ "
                                "wywołaniem submit_explanation.")


# ─────────────────────────────────────────
# Walidacja argumentów (podzbiór JSON Schema użyty w narzędziach)
# ─────────────────────────────────────────
def _type_ok(value: Any, expected: Any) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    for t in types:
        if t == "null" and value is None:
            return True
        if t == "integer" and isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "string" and isinstance(value, str):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "object" and isinstance(value, dict):
            return True
        if t == "array" and isinstance(value, list):
            return True
    return False


def validate_args(schema: dict[str, Any], value: Any, path: str = "argumenty") -> list[str]:
    """Błędy zgodności ze schematem (type, required, properties, items, minimum, maximum)."""
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None and not _type_ok(value, expected):
        return [f"{path}: oczekiwano typu {expected}, jest {type(value).__name__}"]
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: brak wymaganego pola '{key}'")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                errors += validate_args(sub, value[key], f"{path}.{key}")
        if props and path == "argumenty":         # nieznane pola tylko na najwyższym poziomie
            extra = sorted(set(value) - set(props))
            if extra:
                errors.append(f"{path}: nieznane pola {extra}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errors += validate_args(schema["items"], v, f"{path}[{i}]")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: wartość {value} < {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: wartość {value} > {schema['maximum']}")
    return errors


# ─────────────────────────────────────────
# what_if
# ─────────────────────────────────────────
class WhatIfError(ValueError):
    """Niedozwolona zmiana w what_if (komunikat dla modelu)."""


def _as_number(key: str, value: Any) -> float:
    if isinstance(value, bool):
        raise WhatIfError(f"'{key}': oczekiwano liczby, podano wartość logiczną")
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise WhatIfError(f"'{key}': oczekiwano liczby, podano {value!r}") from None
    if not math.isfinite(out):
        raise WhatIfError(f"'{key}': wartość nieskończona")
    return out


def _as_flag(key: str, value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in ("tak", "true", "yes", "nie", "false", "no"):
        return float(value.strip().lower() in ("tak", "true", "yes"))
    v = _as_number(key, value)
    if v not in (0.0, 1.0):
        raise WhatIfError(f"'{key}': dozwolone 0 albo 1, podano {value!r}")
    return v


def resolve_feature(key: str) -> str:
    """Klucz z what_if → kolumna kanoniczna (aliasy, polskie nazwy skarg). Nieznany → WhatIfError."""
    raw = str(key).strip()
    if raw in CC_LABELS_PL:
        return raw
    norm = fold_text(raw).replace(" ", "_").replace("-", "_")
    if norm in FEATURE_ALIASES:
        return FEATURE_ALIASES[norm]
    if fold_text(raw) in _LABEL_TO_CC:
        return _LABEL_TO_CC[fold_text(raw)]
    if norm.startswith("cc_"):
        raise WhatIfError(f"Nieznana skarga '{key}' — użyj nazwy cc_<nazwa> z danych (np. cc_chestpain)")
    raise WhatIfError(f"Cecha '{key}' nie może być zmieniana (dozwolone: parametry życiowe, wiek, sposób "
                      f"przybycia, tlenoterapia, skargi główne cc_*)")


def apply_changes(base_row: dict[str, float | None], changes: dict[str, Any]) -> tuple[dict[str, float | None],
                                                                                       list[dict[str, Any]]]:
    """
    Zmiany what_if na kopii wiersza kanonicznego. Zwraca (nowy wiersz, lista zmian {cecha, przed, po}).
    Błędne klucze / wartości → WhatIfError (komunikat dla modelu).
    """
    if not isinstance(changes, dict):
        raise WhatIfError("'changes' musi być obiektem {cecha: wartość}")
    row = dict(base_row)
    applied: list[dict[str, Any]] = []
    for key, value in changes.items():
        col = resolve_feature(key)
        if col == "arrival" or col in ARRIVAL_COLS.values():
            if col == "arrival":
                mode = ARRIVAL_VALUES.get(fold_text(str(value)))
                if mode is None:
                    raise WhatIfError(f"'arrival': dozwolone 'ambulance', 'self', 'other', podano {value!r}")
            else:
                if _as_flag(key, value) != 1.0:
                    raise WhatIfError(f"'{key}': sposób przybycia ustawiaj wartością 1 (albo kluczem 'arrival')")
                mode = next(m for m, c in ARRIVAL_COLS.items() if c == col)
            before = next((m for m, c in ARRIVAL_COLS.items() if (row.get(c) or 0) > 0), None)
            for m, c in ARRIVAL_COLS.items():
                row[c] = 1.0 if m == mode else 0.0
            applied.append({"cecha": "Sposób przybycia", "kolumna": "arrival", "przed": before, "po": mode})
            continue
        if col in CC_LABELS_PL or col == O2_DEVICE_COL:
            if col not in row:
                raise WhatIfError(f"Skarga '{key}' nie występuje w danych modelu")
            new = _as_flag(key, value)
        elif value is None and col in VITALS.values():
            new = None
        else:
            new = _as_number(key, value)
            lo, hi = VITAL_RANGES[col]
            if col == VITALS["temp"] and new > 50:
                raise WhatIfError("'triage_vital_temp': podaj temperaturę w °C")
            if not lo <= new <= hi:
                raise WhatIfError(f"'{key}': wartość {fmt_num(new, 1)} poza zakresem {fmt_num(lo)}–{fmt_num(hi)}")
        applied.append({"cecha": feature_label(col), "kolumna": col, "przed": row.get(col), "po": new})
        row[col] = new
    sbp, dbp = row.get(VITALS["sbp"]), row.get(VITALS["dbp"])
    if sbp is not None and dbp is not None and dbp >= sbp:
        raise WhatIfError("Ciśnienie rozkurczowe musi być niższe niż skurczowe")
    return row, applied


def predict_row(ctx: EsiContext, row: dict[str, float | None]) -> tuple[np.ndarray, int]:
    """Receptura na wierszu kanonicznym → (prawdopodobieństwa (5,), poziom ESI 1–5)."""
    proba, idx, _ = ctx.predictor.predict_row(row)
    return proba, idx + 1


def what_if(ctx: EsiContext, case: EsiCase, changes: dict[str, Any]) -> dict[str, Any]:
    """Wynik narzędzia what_if (słownik JSON) — błąd zmiany → {"ok": False, "blad": …}."""
    try:
        row, applied = apply_changes(case.base_row, changes)
    except WhatIfError as exc:
        return {"ok": False, "blad": str(exc)}
    proba, level = predict_row(ctx, row)

    def shown(col: str, v: Any) -> Any:
        if col == "arrival":
            return v
        return format_feature_value(col, v)

    rules = _rule_flags(row, level - 1)
    return {
        "ok": True,
        "zmiany": [{"cecha": a["cecha"], "przed": shown(a["kolumna"], a["przed"]), "po": shown(a["kolumna"], a["po"])}
                   for a in applied] or "brak zmian (decyzja bazowa)",
        "poziom_esi_przed": case.decision_esi,
        "poziom_esi_po": level,
        "prawdopodobienstwa_po": {f"ESI {i + 1}": f"{fmt_num(100 * float(p), 1)}%" for i, p in enumerate(proba)},
        "reguly_bezpieczenstwa_po": {"sugerowany_esi": rules["suggested_esi"],
                                     "spelnione_reguly": rules["violations"]},
        "uwaga": "Scenariusz hipotetyczny — nie opisuje stanu pacjenta.",
    }


def explain_model(case: EsiCase, top_n: int = 5) -> dict[str, Any]:
    top_n = max(1, min(int(top_n), len(case.shap) or 1))
    if not case.shap:
        return {"ok": False, "blad": "Wyjaśnienie SHAP niedostępne dla tego przypadku"}
    return {
        "ok": True,
        "model": case.shap_model,
        "poziom_esi": case.decision_esi,
        "czynniki": [{"cecha": f["label_pl"], "wartosc": f["value_pl"],
                      "kierunek": f"{f['direction']} ESI {case.decision_esi}", "shap": round(f["shap"], 3)}
                     for f in case.shap[:top_n]],
        "objasnienie": "SHAP: wkład cechy w wynik modelu dla wyznaczonego poziomu (dodatni = za, ujemny = przeciw).",
    }


def _what_if_values(changes: Any) -> dict[str, set[float]]:
    """Wartości parametrów użyte w what_if (dozwolone w odpowiedzi końcowej jako hipotetyczne)."""
    out: dict[str, set[float]] = {}
    kinds = {v: k for k, v in VITALS.items()}
    if not isinstance(changes, dict):
        return out
    for key, value in changes.items():
        try:
            col = resolve_feature(key)
            num = _as_number(key, value)
        except (WhatIfError, TypeError):
            continue
        kind = "age" if col == "age" else kinds.get(col)
        if kind:
            out.setdefault(kind, set()).add(num)
    return out


# ─────────────────────────────────────────
# Pętla agenta
# ─────────────────────────────────────────
def execute_tool(ctx: EsiContext, case: EsiCase, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Wykonanie narzędzia (bez submit_explanation) → wynik JSON-owy."""
    if name == "explain_model":
        return explain_model(case, int(args.get("top_n") or 5))
    if name == "what_if":
        return what_if(ctx, case, args.get("changes") or {})
    if name == "esi_criteria":
        return {"ok": True, **esi_criteria(int(args["level"]))}
    if name == "search_knowledge":
        hits = search_knowledge(str(args["query"]))
        return {"ok": True, "wyniki": hits} if hits else {"ok": True, "wyniki": [],
                                                           "uwaga": "Brak notatek pasujących do zapytania"}
    raise KeyError(name)


def run_tools_item(client: Any, model: str, case: EsiCase, ctx: EsiContext, *, think: Any = "low",
                   options: dict[str, Any] | None = None, cache: ResponseCache | None = None,
                   raw: JsonlWriter | None = None, max_steps: int = MAX_STEPS) -> dict[str, Any]:
    """Trajektoria agenta dla jednego przypadku × modelu + metryki."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT_TOOLS_PL},
                                      {"role": "user", "content": tools_user_message(case, ctx.recipe_text)}]
    calls: list[ChatResult] = []
    log_calls: list[dict[str, Any]] = []
    valid_ids = {"S1", "S2", "S4"}
    extra_allowed: dict[str, set[float]] = {}
    final: dict[str, Any] | None = None
    via: str | None = None
    strict = False
    n_tool = 0
    errors: list[str] = []
    for step in range(1, max_steps + 1):
        if step == max_steps and final is None:
            messages.append({"role": "user", "content": "To ostatni krok — wywołaj teraz submit_explanation."})
        res, _ = cached_chat(client, cache, PART, model, f"{case.case_id}#s{step}", messages, TOOLS, think, options,
                             raw, {"case_id": case.case_id, "step": step})
        calls.append(res)
        if not res.ok:
            errors.append(res.error or "błąd zapytania")
            break
        messages.append(assistant_message(res))
        if not res.tool_calls:
            obj = extract_json_object(res.content)
            if obj is not None:
                expl, errs, _ = coerce_explanation(obj)
                if expl is not None:
                    final, via = expl, "content"
                    break
                errors += errs
            messages.append({"role": "user", "content": "Nie wywołano żadnego narzędzia. Użyj narzędzi albo zakończ "
                                                        "wywołaniem submit_explanation."})
            continue
        for call in res.tool_calls:
            name = call.get("name")
            args = call.get("arguments") or {}
            raw_args = call.get("arguments_raw")
            n_tool += 1
            entry: dict[str, Any] = {"step": step, "name": name, "known": name in TOOL_NAMES}
            if raw_args and not args:
                parsed = extract_json_object(raw_args)
                args = parsed if parsed is not None else {}
                entry["args_text"] = True
            schema_errors = (validate_args(_SCHEMAS[name], args) if name in _SCHEMAS
                             else [f"nieznane narzędzie '{name}'"])
            if raw_args and not entry.get("args_text"):
                schema_errors.append("argumenty nie są poprawnym JSON-em")
            entry["schema_ok"] = not schema_errors
            entry["args"] = args
            if name == SUBMIT_NAME:
                expl, errs, st = coerce_explanation(args)
                if expl is not None:
                    final, via, strict = expl, "tool", st and not schema_errors
                    entry["result_ok"] = True
                    log_calls.append(entry)
                    break
                result = {"ok": False, "blad": "; ".join(errs)}
                errors += errs
            elif name not in TOOL_NAMES:
                result = {"ok": False, "blad": f"Nieznane narzędzie '{name}'. Dostępne: {', '.join(TOOL_NAMES)}"}
            elif schema_errors:
                result = {"ok": False, "blad": "; ".join(schema_errors)}
            else:
                try:
                    result = execute_tool(ctx, case, name, args)
                except Exception as exc:          # narzędzie nie może przerwać przebiegu
                    log.warning(f"{model} / {case.case_id}: narzędzie {name} — wyjątek {exc!r}")
                    result = {"ok": False, "blad": f"błąd narzędzia: {exc}"}
            entry["result_ok"] = bool(result.get("ok"))
            if result.get("ok"):
                sid = f"T{sum(1 for c in log_calls if c.get('result_ok')) + 1}"
                result = {"id_zrodla": sid, **result}
                valid_ids.add(sid)
                valid_ids.update(h["id"] for h in result.get("wyniki", []) if isinstance(h, dict))
                if name == "what_if":
                    for kind, vals in _what_if_values(args.get("changes")).items():
                        extra_allowed.setdefault(kind, set()).update(vals)
            log_calls.append(entry)
            messages.append({"role": "tool", "tool_name": str(name),
                             "content": json.dumps(result, ensure_ascii=False)})
        if final is not None:
            break

    what_ifs = [c for c in log_calls if c["name"] == "what_if"]
    item: dict[str, Any] = {
        "model": model, "case_id": case.case_id, "row_id": case.row_id, "nurse_esi": case.nurse_esi,
        "decision_esi": case.decision_esi, "valid": int(final is not None), "completed": int(final is not None),
        "strict": int(bool(strict)), "via": via, "retried": 0, "api_error": api_error(calls, final),
        "errors": errors, "explanation": final, "steps": len(calls),
        "tool_calls": n_tool,
        "tool_calls_valid": sum(1 for c in log_calls if c["known"] and c["schema_ok"]),
        "hallucinated_tools": sum(1 for c in log_calls if not c["known"]),
        "what_if_calls": len(what_ifs),
        "what_if_ok": sum(1 for c in what_ifs if c["schema_ok"] and c.get("result_ok")),
        "used_explain_model": int(any(c["name"] == "explain_model" and c.get("result_ok") for c in log_calls)),
        "used_knowledge": int(any(c["name"] in ("search_knowledge", "esi_criteria") for c in log_calls)),
        "trajectory": [dict(c) for c in log_calls],
        "n_calls": len(calls),
        "latency_s": float(sum(c.latency_s or 0.0 for c in calls)),
        "prompt_tokens": int(sum(c.prompt_tokens or 0 for c in calls)),
        "completion_tokens": int(sum(c.completion_tokens or 0 for c in calls)),
    }
    if final is not None:
        item.update(score_explanation(case, final, valid_ids, extra_allowed))
    return item


def summarize_tools(items: list[dict[str, Any]], prices: dict[str, Any] | None = None) -> dict[str, Any]:
    """Metryki części tools dla jednego modelu (+ metryki odpowiedzi końcowej jak w części esi)."""
    base = summarize_explanations(items, prices)
    n = len(items)
    tool_calls = sum(it.get("tool_calls", 0) for it in items)
    wi = sum(it.get("what_if_calls", 0) for it in items)
    base.update({
        "completion_rate": base["valid_rate"],
        "tool_calls": int(tool_calls),
        "valid_tool_call_rate": (sum(it.get("tool_calls_valid", 0) for it in items) / tool_calls) if tool_calls else None,
        "hallucinated_tools": int(sum(it.get("hallucinated_tools", 0) for it in items)),
        "avg_steps": float(np.mean([it.get("steps", 0) for it in items])) if n else None,
        "what_if_calls": int(wi),
        "what_if_ok_rate": (sum(it.get("what_if_ok", 0) for it in items) / wi) if wi else None,
        "what_if_use_rate": (sum(1 for it in items if it.get("what_if_calls")) / n) if n else None,
        "explain_model_rate": (sum(it.get("used_explain_model", 0) for it in items) / n) if n else None,
        "knowledge_rate": (sum(it.get("used_knowledge", 0) for it in items) / n) if n else None,
    })
    return base


def render_markdown(summary: dict[str, dict[str, Any]], ctx: EsiContext, think: Any, max_steps: int) -> str:
    from src.llm_eval.pricing import format_usd

    lines = [
        "## Część narzędziowa — agent z narzędziami",
        "",
        f"- Te same przypadki co w części ESI ({plural_pl(len(ctx.cases), 'przypadek', 'przypadki', 'przypadków')}); "
        f"w prompcie dane [S1], decyzja [S2] i reguły [S4] "
        f"(bez SHAP — dostępny przez `explain_model`). Maks. {max_steps} kroków, rozumowanie `think={think!r}`.",
        "- Narzędzia: `explain_model`, `what_if` (ponowne przeliczenie receptury: cechy inżynieryjne, połączenie, "
        "temperatura, reguła), `esi_criteria` (własna parafraza ESI v4), `search_knowledge` (15 notatek), "
        "`submit_explanation`.",
        "",
        "| Model | Ukończone | Poprawne wywołania narzędzi | Zmyślone narzędzia | Śr. kroków | what_if: użycie / "
        "poprawne | explain_model | Poziom = decyzja | Pokrycie SHAP | Halucynacje liczb | Po polsku | "
        "Eskalacje: trafne / fałszywe / zły kier. | Czas [s] (med.) | Tokeny we / wy | Koszt |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, m in summary.items():
        lines.append(
            f"| `{model}` | {pct(m['completion_rate'])} | {pct(m['valid_tool_call_rate'])} ({m['tool_calls']}) | "
            f"{m['hallucinated_tools']} | {'—' if m['avg_steps'] is None else fmt_num(m['avg_steps'], 1)} | "
            f"{pct(m['what_if_use_rate'])} / {pct(m['what_if_ok_rate'])} | {pct(m['explain_model_rate'])} | "
            f"{pct(m['level_match_rate'])} | {pct(m['shap_coverage'])} | {pct(m['hallucination_rate'])} | "
            f"{pct(m['polish_rate'])} | {m['esc_useful']} / {m['esc_false_alarm']} / {m['esc_wrong_direction']} | "
            f"{'—' if m['latency_s_median'] is None else fmt_num(m['latency_s_median'], 1)} | "
            f"{int_text(m['prompt_tokens'])} / {int_text(m['completion_tokens'])} | {format_usd(m['cost_usd'])} |")
    lines += [
        "",
        "Definicje: *ukończone* — poprawne `submit_explanation` (albo JSON w treści) w limicie kroków; "
        "*poprawne wywołania* — znana nazwa narzędzia i argumenty zgodne ze schematem (w nawiasie liczba "
        "wywołań); *what_if poprawne* — wywołanie wykonane (dozwolone cechy, wiarygodne wartości); "
        "liczby z własnych scenariuszy what_if nie są liczone jako halucynacje.",
        "",
    ]
    return "\n".join(lines)


def run_tools_part(
    client: Any,
    models: Sequence[str],
    *,
    out_dir: Path | str,
    context: EsiContext | None = None,
    recipe: Path | str | None = None,
    n_cases: int = 40,
    seed: int = 42,
    think: Any = "low",
    options: dict[str, Any] | None = None,
    cache: ResponseCache | None = None,
    prices: dict[str, Any] | None = None,
    runs_dir: Path | str | None = None,
    raw_log: JsonlWriter | None = None,
    max_steps: int = MAX_STEPS,
    decimal_comma: bool = False,
) -> dict[str, Any]:
    """
    Część "tools": agent × przypadek. Zwraca {"part", "skipped", "summary", "items", "files",
    "markdown", "context"}; pliki: tools_items.csv, tools_summary.csv, tools_trajectories.jsonl,
    raw_tools.jsonl (+ esi_cases.json, jeśli kontekst budowany tutaj). decimal_comma — jak w run_esi_part.
    """
    from src.utils.fileio import write_text_atomic

    title = "Część narzędziowa — agent z narzędziami"
    out = Path(out_dir)
    if context is None:
        context, reason = try_prepare_context(recipe_path=recipe, n_cases=n_cases, seed=seed, runs_dir=runs_dir)
        if context is None:
            return skipped_result(PART, title, reason or "brak przypadków")
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    if not (out / "esi_cases.json").exists():
        files["cases"] = str(context.save(out / "esi_cases.json"))
    think = parse_think(think)
    raw = raw_log if raw_log is not None else JsonlWriter(out / RAW_FILE)
    concurrency = int(getattr(client, "concurrency", 3) or 3)
    work = [(m, c) for m in models for c in context.cases]
    log.info(f"Część narzędziowa: {len(models)} model(e) × {len(context.cases)} przypadków (≤ {max_steps} kroków)")

    def worker(model: str, case: EsiCase) -> dict[str, Any]:
        return run_tools_item(client, model, case, context, think=think, options=options, cache=cache, raw=raw,
                              max_steps=max_steps)

    items = run_items(work, worker, concurrency)
    summary = {m: summarize_tools([it for it in items if it["model"] == m], prices) for m in models}
    extra = ["completed", "steps", "tool_calls", "tool_calls_valid", "hallucinated_tools", "what_if_calls",
             "what_if_ok", "used_explain_model", "used_knowledge"]
    items_frame(items, extra).to_csv(out / "tools_items.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"model": m, **{k: v for k, v in s.items() if not isinstance(v, list)}}
                  for m, s in summary.items()]).to_csv(out / "tools_summary.csv", index=False, encoding="utf-8")
    traj_path = out / "tools_trajectories.jsonl"
    traj_path.unlink(missing_ok=True)              # trajektorie TEGO przebiegu (ten sam --out → bez dopisywania)
    traj = JsonlWriter(traj_path)
    for it in items:
        traj.write({"model": it["model"], "case_id": it["case_id"], "trajectory": it.get("trajectory"),
                    "explanation": it.get("explanation")})
    markdown = render_markdown(summary, context, think, max_steps)
    if not decimal_comma:
        markdown = decimal_point(markdown)
    write_text_atomic(out / "tools_section.md", markdown)
    files.update({"items_csv": str(out / "tools_items.csv"), "summary_csv": str(out / "tools_summary.csv"),
                  "trajectories": str(out / "tools_trajectories.jsonl"), "raw": str(raw.path),
                  "section_md": str(out / "tools_section.md")})
    return {"part": PART, "skipped": None, "summary": summary, "items": items, "files": files,
            "markdown": markdown, "context": context.summary()}


__all__ = [
    "EXPLANATION_SCHEMA",
    "MAX_STEPS",
    "TOOLS",
    "TOOL_NAMES",
    "WhatIfError",
    "apply_changes",
    "explain_model",
    "predict_row",
    "run_tools_item",
    "run_tools_part",
    "summarize_tools",
    "validate_args",
    "what_if",
]
