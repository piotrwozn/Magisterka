"""
Część "esi" porównania LLM: wyjaśnienie deterministycznej decyzji ESI dla pielęgniarki triażowej.

Model dostaje opis pacjenta [S1], decyzję systemu z prawdopodobieństwami [S2], top-5 cech SHAP [S3]
i flagi reguł bezpieczeństwa ESI [S4] (src/llm_eval/esi_cases.py) i ma zwrócić wyjaśnienie
WYŁĄCZNIE przez narzędzie `submit_explanation`:

    {poziom_esi: int, uzasadnienie: str, kluczowe_czynniki: list[str],
     sugestia_eskalacji: {czy: bool, poziom: int | null, powod: str | null}}

Parametr `format` (structured outputs) jest ignorowany przez modele chmurowe Ollamy, więc nie
polegamy na nim: brak wywołania narzędzia → JSON wyłuskany z treści; nadal źle → jedno ponowienie
z opisem błędu.

Metryki automatyczne (na model): odsetek poprawnych odpowiedzi; poziom w odpowiedzi = decyzja;
pokrycie top-5 SHAP (polskie synonimy cech); wierność liczb (liczby przy słowach kluczowych
parametrów porównane z przypadkiem → odsetek odpowiedzi z halucynacją); sugestie eskalacji vs ESI
pielęgniarki (trafna = prawdziwe ESI pilniejsze niż decyzja, inaczej fałszywy alarm; sugestia
niższego priorytetu = zły kierunek); polszczyzna (udział polskich słów funkcyjnych); cytowanie
źródeł; opóźnienie, tokeny, koszt. Do tego ślepy arkusz ocen (src/llm_eval/ratings.py).

Metryki tekstowe są heurystyczne (słowa kluczowe, rdzenie) — w pracy podawać je jako przybliżone
i weryfikować ślepą oceną ekspercką.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.llm_eval.cache import JsonlWriter, ResponseCache, prompt_hash
from src.llm_eval.cc_labels_pl import feature_synonyms, fold_text, mentions
from src.llm_eval.client import ChatResult, parse_think
from src.llm_eval.esi_cases import EsiCase, EsiContext, fmt_num, plural_pl, try_prepare_context
from src.utils.logger import get_logger

log = get_logger(__name__)

PART = "esi"
SUBMIT_NAME = "submit_explanation"
RAW_FILE = "raw_esi.jsonl"

# ─────────────────────────────────────────
# Prompt i narzędzie
# ─────────────────────────────────────────
SYSTEM_PROMPT_PL = """Jesteś asystentem, który wyjaśnia decyzje systemu wspomagania triażu na szpitalnym oddziale ratunkowym (SOR).
System uczenia maszynowego wyznaczył poziom pilności ESI (Emergency Severity Index: 1 = najpilniejszy, 5 = najmniej pilny) według stałej, deterministycznej receptury. Twoim zadaniem jest wyjaśnić tę decyzję pielęgniarce lub ratownikowi triażowemu — po polsku, rzeczowo i zwięźle (3–6 zdań).

Zasady:
1. Nie zmieniasz decyzji systemu: w polu poziom_esi podaj DOKŁADNIE poziom wyznaczony przez system.
2. Możesz wyłącznie DORADCZO zasugerować wyższy priorytet (niższy numer ESI), jeśli dane pacjenta lub reguły bezpieczeństwa to uzasadniają. Nigdy nie sugeruj niższego priorytetu.
3. Nie wymyślaj pomiarów, objawów, wywiadu ani wyników badań — korzystaj tylko z podanych danych. Jeśli czegoś nie zmierzono, napisz, że brak pomiaru.
4. Powołuj się na źródła przez ich identyfikatory w nawiasach kwadratowych, np. [S1], [S3].
5. Wskaż czynniki, które najbardziej wpłynęły na decyzję (analiza SHAP), i odnieś się do reguł bezpieczeństwa ESI.
6. Odpowiedź przekaż WYŁĄCZNIE przez wywołanie narzędzia submit_explanation. Jeśli nie możesz wywołać narzędzia, odpowiedz samym obiektem JSON z polami: poziom_esi, uzasadnienie, kluczowe_czynniki, sugestia_eskalacji."""

EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["poziom_esi", "uzasadnienie", "kluczowe_czynniki", "sugestia_eskalacji"],
    "properties": {
        "poziom_esi": {"type": "integer", "minimum": 1, "maximum": 5,
                       "description": "Poziom ESI wyznaczony przez system (bez zmian)."},
        "uzasadnienie": {"type": "string",
                         "description": "Wyjaśnienie decyzji po polsku (3–6 zdań) z odwołaniami do źródeł, np. [S1]."},
        "kluczowe_czynniki": {"type": "array", "items": {"type": "string"},
                              "description": "Najważniejsze czynniki decyzji (krótkie hasła po polsku)."},
        "sugestia_eskalacji": {
            "type": "object",
            "required": ["czy"],
            "properties": {
                "czy": {"type": "boolean", "description": "Czy doradczo sugerujesz WYŻSZY priorytet (niższy numer ESI)."},
                "poziom": {"type": ["integer", "null"], "minimum": 1, "maximum": 5,
                           "description": "Sugerowany poziom ESI (tylko pilniejszy niż decyzja) albo null."},
                "powod": {"type": ["string", "null"], "description": "Krótki powód sugestii albo null."},
            },
        },
    },
}

SUBMIT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBMIT_NAME,
        "description": "Przekaż końcowe wyjaśnienie decyzji ESI dla pielęgniarki triażowej.",
        "parameters": EXPLANATION_SCHEMA,
    },
}


def user_message(case: EsiCase, recipe_text: str | None = None, *, with_shap: bool = True,
                 with_rules: bool = True, closing: str | None = None) -> str:
    """Treść wiadomości użytkownika: źródła [S1]…[S4] + polecenie."""
    titles = {"S1": "Dane pacjenta", "S2": "Decyzja systemu", "S3": "Czynniki decyzji (SHAP)",
              "S4": "Reguły bezpieczeństwa ESI"}
    parts = [f"Przypadek {case.case_id}."]
    for sid, text in case.sources(recipe_text, with_shap=with_shap, with_rules=with_rules).items():
        parts.append(f"[{sid}] {titles[sid]}:\n{text}")
    parts.append(closing or "Wyjaśnij decyzję systemu i wywołaj narzędzie submit_explanation.")
    return "\n\n".join(parts)


def build_messages(case: EsiCase, recipe_text: str | None = None) -> list[dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM_PROMPT_PL},
            {"role": "user", "content": user_message(case, recipe_text)}]


def retry_hint(errors: Sequence[str]) -> str:
    """Wiadomość ponowienia z opisem błędu."""
    return ("Twoja odpowiedź nie była poprawna: " + "; ".join(errors)[:600] + ". Wywołaj narzędzie "
            "submit_explanation z polami: poziom_esi (liczba całkowita 1–5, taka jak decyzja systemu), "
            "uzasadnienie (tekst), kluczowe_czynniki (lista tekstów), sugestia_eskalacji "
            "({\"czy\": true/false, \"poziom\": liczba albo null, \"powod\": tekst albo null}).")


def assistant_message(res: ChatResult) -> dict[str, Any]:
    """Odpowiedź modelu do historii rozmowy (z rozumowaniem i wywołaniami narzędzi, jak w dokumentacji Ollamy)."""
    msg: dict[str, Any] = {"role": "assistant", "content": res.content or ""}
    if res.thinking:
        msg["thinking"] = res.thinking
    if res.tool_calls:
        msg["tool_calls"] = [{"type": "function", "function": {"index": i, "name": c.get("name"),
                                                                "arguments": c.get("arguments") or {}}}
                             for i, c in enumerate(res.tool_calls)]
    return msg


# ─────────────────────────────────────────
# Parsowanie odpowiedzi
# ─────────────────────────────────────────
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _balanced_objects(text: str) -> list[str]:
    """Wszystkie zrównoważone fragmenty {…} (uwzględnia napisy JSON), od najdłuższych."""
    out = []
    for start in [i for i, ch in enumerate(text) if ch == "{"]:
        depth, in_str, esc = 0, False, False
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start:j + 1])
                    break
    return sorted(out, key=len, reverse=True)


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    """Pierwszy poprawny obiekt JSON w tekście (bloki ```json```, potem zrównoważone {…})."""
    if not text:
        return None
    candidates = [m.strip() for m in _FENCE.findall(text)] + [text.strip()] + _balanced_objects(text)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _unwrap(obj: dict[str, Any]) -> dict[str, Any]:
    """{"name": "submit_explanation", "arguments": {...}} / {"parameters": {...}} → argumenty."""
    for key in ("arguments", "parameters", "input", "args"):
        inner = obj.get(key)
        if isinstance(inner, str):
            inner = extract_json_object(inner)
        if isinstance(inner, dict) and "poziom_esi" not in obj:
            return inner
    return obj


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(?:ESI\s*)?([1-5])\s*", value, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "tak", "false", "nie"):
        return value.strip().lower() in ("true", "tak")
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    return None


def coerce_explanation(obj: Any) -> tuple[dict[str, Any] | None, list[str], bool]:
    """
    Sprawdzenie i lekka normalizacja wyjaśnienia.

    Zwraca (wyjaśnienie | None, błędy, strict). `strict` = typy dokładnie jak w schemacie (bez
    poprawek); dopuszczalne poprawki: liczba jako tekst ("2", "ESI 2"), wartość logiczna jako tekst,
    pojedynczy tekst zamiast listy czynników, brak sugestii eskalacji (= brak sugestii).
    """
    if not isinstance(obj, dict):
        return None, ["odpowiedź nie jest obiektem JSON"], False
    obj = _unwrap(obj)
    errors: list[str] = []
    strict = True
    level_raw = obj.get("poziom_esi")
    level = _as_int(level_raw)
    if level is None or not 1 <= level <= 5:
        errors.append("poziom_esi musi być liczbą całkowitą 1–5")
    elif not isinstance(level_raw, int) or isinstance(level_raw, bool):
        strict = False
    text = obj.get("uzasadnienie")
    if not isinstance(text, str) or not text.strip():
        errors.append("brak pola uzasadnienie (niepusty tekst)")
    factors = obj.get("kluczowe_czynniki")
    if isinstance(factors, str) and factors.strip():
        factors, strict = [factors.strip()], False
    if not isinstance(factors, list) or not all(isinstance(f, (str, int, float)) for f in factors):
        errors.append("kluczowe_czynniki musi być listą tekstów")
        factors = []
    elif not all(isinstance(f, str) for f in factors):
        strict = False
    esc = obj.get("sugestia_eskalacji")
    if esc is None:
        esc, strict = {"czy": False, "poziom": None, "powod": None}, False
    if not isinstance(esc, dict):
        errors.append("sugestia_eskalacji musi być obiektem {czy, poziom, powod}")
        esc = {}
    czy = _as_bool(esc.get("czy"))
    if czy is None:
        errors.append("sugestia_eskalacji.czy musi być wartością logiczną")
    elif not isinstance(esc.get("czy"), bool):
        strict = False
    esc_level = esc.get("poziom")
    esc_level_int = None if esc_level is None else _as_int(esc_level)
    if esc_level is not None and esc_level_int is None:
        errors.append("sugestia_eskalacji.poziom musi być liczbą 1–5 albo null")
    reason = esc.get("powod")
    if reason is not None and not isinstance(reason, str):
        reason, strict = str(reason), False
    if errors:
        return None, errors, False
    return {
        "poziom_esi": level,
        "uzasadnienie": text.strip(),
        "kluczowe_czynniki": [str(f).strip() for f in factors if str(f).strip()],
        "sugestia_eskalacji": {"czy": bool(czy), "poziom": esc_level_int, "powod": reason},
    }, [], strict


def parse_submission(res: ChatResult) -> tuple[dict[str, Any] | None, list[str], str | None, bool]:
    """
    Wyjaśnienie z odpowiedzi: wywołanie submit_explanation, inaczej JSON z treści.
    Zwraca (wyjaśnienie | None, błędy, źródło 'tool' | 'content' | None, strict).
    """
    errors: list[str] = []
    for call in res.tool_calls or []:
        if call.get("name") != SUBMIT_NAME:
            errors.append(f"wywołano nieznane narzędzie {call.get('name')!r}")
            continue
        args = call.get("arguments") or {}
        if not args and call.get("arguments_raw"):
            args = extract_json_object(call["arguments_raw"]) or {}
            if not args:
                errors.append("argumenty submit_explanation nie są poprawnym JSON-em")
                continue
        expl, errs, strict = coerce_explanation(args)
        if expl is not None:
            return expl, [], "tool", strict and not call.get("arguments_raw")
        errors += errs
    obj = extract_json_object(res.content)
    if obj is not None:
        expl, errs, strict = coerce_explanation(obj)
        if expl is not None:
            return expl, [], "content", False
        errors += errs
    elif not res.tool_calls:
        errors.append("brak wywołania narzędzia submit_explanation i brak obiektu JSON w treści")
    return None, errors or ["niepoprawna odpowiedź"], None, False


# ─────────────────────────────────────────
# Metryki tekstu
# ─────────────────────────────────────────
def explanation_text(expl: dict[str, Any]) -> str:
    esc = expl.get("sugestia_eskalacji") or {}
    return " ".join([expl.get("uzasadnienie") or "", " ; ".join(expl.get("kluczowe_czynniki") or []),
                     str(esc.get("powod") or "")])


def shap_coverage(text: str, case: EsiCase) -> tuple[float | None, list[str]]:
    """Odsetek top-5 cech SHAP wspomnianych w tekście (synonimy) i lista nazw wspomnianych."""
    top = case.shap_top
    if not top:
        return None, []
    hit = [f["feature"] for f in top if mentions(text, feature_synonyms(f["feature"]))]
    return len(hit) / len(top), hit


# Słowa kluczowe parametrów (wyrażenia regularne na tekście po fold_text)
_KEYWORDS: dict[str, list[str]] = {
    "hr": [r"tetn(?!icz)", r"hr\b", r"akcj\w* serca", r"czestosc\w* (?:akcji )?serca", r"puls(?!oks)",
           r"rytm\w* serca", r"uderzen"],
    "bp": [r"cisnieni\w*(?: tetnicz\w*)?", r"tetnicz\w*", r"sbp\b", r"dbp\b", r"skurczow\w*", r"rozkurczow\w*"],
    "rr_amb": [r"rr\b"],
    "rr": [r"oddech\w*", r"oddychan\w*", r"tachypno\w*", r"bradypno\w*"],
    "o2": [r"spo2", r"saturac\w*", r"sat\b", r"utlenow\w*", r"pulsoksymetr\w*"],
    "temp": [r"temperatur\w*", r"goraczk\w*", r"cieplot\w*"],
    "age": [r"wiek\w*", r"w wieku"],
}
# Słowa, przy których liczby nie są pomiarami pacjenta (prawdopodobieństwa, skale, poziomy, źródła)
_BLOCKERS: list[str] = [r"esi", r"prawdopodob\w*", r"shap", r"poziom\w*", r"punkt\w*", r"pkt", r"news\w*",
                        r"qsofa", r"sirs", r"wskaznik\w*", r"srednie\w*", r"map\b", r"cisnieni\w* tetna",
                        r"amplitud\w*", r"krok\w*", r"skal\w*", r"wynik\w*", r"pewnos\w*", r"szans\w*"]
_COMPARATORS = re.compile(r"(<|>|≥|≤|=<|>=|powyzej|ponizej|ponad|przekracz\w*|prog\w*|nizej|wyzej|mniej niz|"
                          r"wiecej niz|\bdo\b|\bod\b|norm\w*|granic\w*|min\.|maks\w*|zakres\w*|miedzy)\s*$")
_RULE_CONTEXT = re.compile(r"strefa|strefie|strefy|zagrozen|kryteri|regul|prog")
_NUMBER = re.compile(r"(?<![\w.,/])(\d{1,3}(?:[.,]\d{1,2})?)(?:\s*/\s*(\d{1,3}))?(?![\w])")
_SUFFIX_AGE = re.compile(r"^\s*(?:-\s*)?(?:lat\b|lata\b|lat[a-z]*|rok\w*|r\.\s*z|letn\w*|latk\w*|latek)")
_SUFFIX_TEMP = re.compile(r"^\s*(?:°\s*c|st\.?\s*c\b|stopni)")
_SUFFIX_BP = re.compile(r"^\s*mm\s*hg")
TOLERANCE: dict[str, float] = {"hr": 2.0, "sbp": 2.0, "dbp": 2.0, "rr": 1.0, "o2": 1.0, "temp": 0.25, "age": 1.0}
THRESHOLDS: dict[str, set[float]] = {
    "hr": {40, 50, 90, 100, 110, 130, 150}, "sbp": {80, 90, 100, 110, 140, 180, 200, 219, 220},
    "dbp": {90, 110, 120}, "rr": {8, 10, 12, 20, 22, 24, 25, 30, 35}, "o2": {85, 88, 90, 91, 92, 93, 94, 95, 96},
    "temp": {32, 35, 36, 37, 38, 39, 41}, "age": {18, 65, 75, 80},
}


def _find_all(patterns: list[str], text: str) -> list[tuple[int, int]]:
    out = []
    for pat in patterns:
        out += [(m.start(), m.end()) for m in re.finditer(r"(?<![a-z0-9])" + pat, text)]
    return out


def _num(text: str) -> float:
    return float(text.replace(",", "."))


def _matches(kind: str, claimed: float, vitals: dict[str, float | None], extra: dict[str, set[float]]) -> bool:
    actual = vitals.get(kind)
    tol = TOLERANCE[kind]
    if actual is not None and abs(claimed - actual) <= tol + 1e-9:
        return True
    return any(abs(claimed - v) <= tol + 1e-9 for v in extra.get(kind, set()))


def numeric_faithfulness(text: str, case: EsiCase,
                         extra_allowed: dict[str, set[float]] | None = None) -> dict[str, Any]:
    """
    Liczby podane jako pomiary pacjenta (przy słowach kluczowych parametrów albo z jednostką
    °C / mmHg / lat) porównane z przypadkiem w tolerancji (`TOLERANCE`). Progi kliniczne
    (po „>”, „powyżej”, „próg”…) są pomijane. Parametr niezmierzony z podaną wartością = błąd.

    extra_allowed : wartości z narzędzia what_if (scenariusze hipotetyczne) — {rodzaj: {liczby}}.
    Zwraca {"checked", "wrong", "errors": [{"kind", "claimed", "actual"}]}.
    """
    t = fold_text(text)
    extra = extra_allowed or {}
    vitals = dict(case.vitals)
    numbers = list(_NUMBER.finditer(t))
    spans = [(m.start(), m.end()) for m in numbers]
    keyword_hits = [(s, e, kind) for kind, pats in _KEYWORDS.items() for s, e in _find_all(pats, t)]
    blockers = _find_all(_BLOCKERS, t)
    checked, errors = 0, []
    for idx, m in enumerate(numbers):
        start, end = m.span()
        before, after = t[max(0, start - 40):start], t[end:end + 14]
        a_text, b_text = m.group(1), m.group(2)
        kind: str | None = None
        kw_start = start
        if _SUFFIX_AGE.match(after) and b_text is None:
            kind = "age"
        elif _SUFFIX_TEMP.match(after) and b_text is None:
            kind = "temp"
        elif _SUFFIX_BP.match(after) or b_text is not None:
            kind = "bp"
        # Najbliższe słowo kluczowe przed liczbą (≤ 35 znaków, bez innej liczby pomiędzy)
        prev_end = spans[idx - 1][1] if idx > 0 else -1
        candidates = [(e, -s, k) for s, e, k in keyword_hits if prev_end <= s and e <= start and start - e <= 35]
        if candidates:
            e, neg_s, k = max(candidates)          # najbliższe; przy tym samym końcu — najdłuższe dopasowanie
            s = -neg_s
            kw_start = s
            if kind is None or (kind == "bp" and k in ("bp", "rr_amb")):
                kind = k
        if kind is None:
            continue
        if kind == "bp" and b_text is None and not candidates and not _SUFFIX_BP.match(after):
            continue
        # Blokery (prawdopodobieństwa, skale, poziomy) między słowem kluczowym a liczbą albo tuż przed
        # słowem kluczowym (np. „średnie ciśnienie”)
        if any((kw_start <= s and e <= start)
               or (e <= kw_start and kw_start - e <= 10 and not re.search(r"[,;.:!?()\d]", t[e:kw_start]))
               for s, e in blockers):
            continue
        window = t[max(0, kw_start - 15):start]
        if _COMPARATORS.search(t[max(0, start - 18):start].rstrip()):
            continue
        claims: list[tuple[str, float]] = []
        a = _num(a_text)
        if kind == "bp" or (kind == "rr_amb" and b_text is not None):
            if b_text is not None:
                claims = [("sbp", a), ("dbp", float(b_text))]
            else:
                where = t[max(0, kw_start - 2):start]
                claims = [("dbp", a)] if "rozkurcz" in where else [("sbp", a)] if "skurcz" in where else [("bp?", a)]
        elif kind == "rr_amb":
            claims = [("rr|bp", a)]
        else:
            claims = [(kind, a)]
        for ckind, value in claims:
            options = {"bp?": ["sbp", "dbp"], "rr|bp": ["rr", "sbp", "dbp"]}.get(ckind, [ckind])
            if any(value in THRESHOLDS.get(o, set()) for o in options) and _RULE_CONTEXT.search(window):
                continue
            checked += 1
            if not any(_matches(o, value, vitals, extra) for o in options):
                actual = {o: vitals.get(o) for o in options}
                errors.append({"kind": "/".join(options), "claimed": value, "actual": actual})
    return {"checked": checked, "wrong": len(errors), "errors": errors}


_PL_STOPWORDS = frozenset(
    "i w z na do nie sie jest to ze o jak dla oraz lub po przez od przy ze sa ma by byc moze co tak ale jego jej "
    "ten ta te tego tej tym takze rowniez czy gdy jesli poniewaz wiec juz tylko bardzo pacjent pacjenta pacjentka "
    "u a oraz bez nad pod wskazuje wynika decyzja poziom".split())
_EN_STOPWORDS = frozenset(
    "the and of to is in with for are be this that on as by an or it was not patient which has have from at "
    "these their there due level decision indicates".split())


def polish_check(text: str) -> dict[str, Any]:
    """Prosty test języka: udział polskich i angielskich słów funkcyjnych (+ obecność polskich znaków)."""
    words = re.findall(r"[a-ząćęłńóśźż]+", str(text).lower())
    folded = [fold_text(w) for w in words]
    n = len(folded)
    pl = sum(w in _PL_STOPWORDS for w in folded) / n if n else 0.0
    en = sum(w in _EN_STOPWORDS for w in words) / n if n else 0.0
    diacritics = bool(re.search(r"[ąćęłńóśźż]", str(text).lower()))
    return {"pl_ratio": pl, "en_ratio": en, "is_polish": bool(n >= 5 and pl >= 0.08 and pl > en and
                                                               (diacritics or pl >= 0.15))}


_BRACKETS = re.compile(r"\[([^\[\]]{1,40})\]")
_SOURCE_ID = re.compile(r"\b([A-Z]\d{1,2})\b")


def citations(text: str, valid_ids: set[str]) -> dict[str, Any]:
    """Identyfikatory źródeł w nawiasach kwadratowych: poprawne i nieistniejące."""
    found = [sid for group in _BRACKETS.findall(text or "") for sid in _SOURCE_ID.findall(group.upper())]
    valid = sorted({s for s in found if s in valid_ids})
    invalid = sorted({s for s in found if s not in valid_ids})
    return {"valid": valid, "invalid": invalid}


def escalation_outcome(expl: dict[str, Any], case: EsiCase) -> str:
    """
    'none' (brak sugestii), 'useful' (sugestia pilniejsza i ESI pielęgniarki pilniejsze niż decyzja),
    'false_alarm' (sugestia pilniejsza, a ESI pielęgniarki nie), 'wrong_direction' (sugestia tego
    samego / mniej pilnego poziomu albo bez poziomu).
    """
    esc = expl.get("sugestia_eskalacji") or {}
    if not esc.get("czy"):
        return "none"
    level = esc.get("poziom")
    if level is None or not 1 <= int(level) < case.decision_esi:
        return "wrong_direction"
    return "useful" if case.nurse_esi < case.decision_esi else "false_alarm"


def score_explanation(case: EsiCase, expl: dict[str, Any], valid_ids: set[str],
                      extra_allowed: dict[str, set[float]] | None = None) -> dict[str, Any]:
    """Wszystkie metryki automatyczne jednego poprawnego wyjaśnienia."""
    text = explanation_text(expl)
    coverage, covered = shap_coverage(text, case)
    nums = numeric_faithfulness(text, case, extra_allowed)
    lang = polish_check(expl.get("uzasadnienie") or "")
    cites = citations(text, valid_ids)
    return {
        "level_match": int(expl["poziom_esi"] == case.decision_esi),
        "shap_coverage": coverage,
        "shap_covered": covered,
        "numbers_checked": nums["checked"],
        "numbers_wrong": nums["wrong"],
        "number_errors": nums["errors"],
        "hallucination": int(nums["wrong"] > 0),
        "escalation": escalation_outcome(expl, case),
        "polish": int(lang["is_polish"]),
        "pl_ratio": lang["pl_ratio"],
        "cites": int(bool(cites["valid"])),
        "citations_valid": cites["valid"],
        "citations_invalid": cites["invalid"],
    }


# ─────────────────────────────────────────
# Wywołania (pamięć podręczna + surowe odpowiedzi)
# ─────────────────────────────────────────
def cached_chat(client: Any, cache: ResponseCache | None, part: str, model: str, item_id: str,
                messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, think: Any,
                options: dict[str, Any] | None, raw: JsonlWriter | None,
                meta: dict[str, Any] | None = None) -> tuple[ChatResult, bool]:
    """client.chat z pamięcią podręczną (część, model, pozycja, skrót promptu) i zapisem surowej odpowiedzi."""
    phash = prompt_hash(messages, tools=tools, think=think, options=options)
    hit = cache.get(part, model, item_id, phash) if cache is not None else None
    if hit is not None:
        res, cached = ChatResult.from_dict(hit), True
    else:
        res, cached = client.chat(model, messages, tools=tools, think=think, options=options), False
        if cache is not None and res.ok:
            cache.put(part, model, item_id, phash, res, meta=meta)
    if raw is not None:
        rec = res.to_dict()
        rec.pop("raw", None)
        raw.write({"part": part, "model": model, "item_id": item_id, "prompt_hash": phash, "cached": cached,
                   **(meta or {}), "result": rec})
    return res, cached


def api_error(calls: list[ChatResult], result: Any) -> str | None:
    """Błąd zapytania (sieć / limit / serwer), gdy przez niego nie ma wyniku; inaczej None."""
    failed = [c.error for c in calls if not c.ok]
    return failed[-1] if failed and result is None else None


def _usage(calls: list[ChatResult]) -> dict[str, Any]:
    return {
        "n_calls": len(calls),
        "latency_s": float(sum(c.latency_s or 0.0 for c in calls)),
        "prompt_tokens": int(sum(c.prompt_tokens or 0 for c in calls)),
        "completion_tokens": int(sum(c.completion_tokens or 0 for c in calls)),
    }


def run_esi_item(client: Any, model: str, case: EsiCase, ctx: EsiContext, *, think: Any = "low",
                 options: dict[str, Any] | None = None, cache: ResponseCache | None = None,
                 raw: JsonlWriter | None = None) -> dict[str, Any]:
    """Jeden przypadek × model: zapytanie, parsowanie, ewentualne ponowienie, metryki."""
    messages = build_messages(case, ctx.recipe_text)
    tools = [SUBMIT_TOOL]
    meta = {"case_id": case.case_id}
    res, _ = cached_chat(client, cache, PART, model, f"{case.case_id}#1", messages, tools, think, options, raw, meta)
    calls = [res]
    retried = False
    if not res.ok:
        expl, errors, via, strict = None, [res.error or "błąd zapytania"], None, False
    else:
        expl, errors, via, strict = parse_submission(res)
        if expl is None:
            retried = True
            retry_msgs = messages + [assistant_message(res), {"role": "user", "content": retry_hint(errors)}]
            res2, _ = cached_chat(client, cache, PART, model, f"{case.case_id}#2", retry_msgs, tools, think,
                                  options, raw, {**meta, "retry": True})
            calls.append(res2)
            if res2.ok:
                expl, errors, via, strict = parse_submission(res2)
            else:
                errors = [res2.error or "błąd zapytania"]
    item: dict[str, Any] = {
        "model": model, "case_id": case.case_id, "row_id": case.row_id, "nurse_esi": case.nurse_esi,
        "decision_esi": case.decision_esi, "valid": int(expl is not None), "strict": int(bool(strict)),
        "via": via, "retried": int(retried), "api_error": api_error(calls, expl),
        "errors": errors, "explanation": expl, **_usage(calls),
    }
    if expl is not None:
        item.update(score_explanation(case, expl, set(case.sources(ctx.recipe_text))))
    return item


def run_items(items: list[tuple[str, EsiCase]], worker: Any, concurrency: int) -> list[dict[str, Any]]:
    """Pozycje (model, przypadek) w puli wątków; wynik w kolejności wejścia. Ctrl+C → anuluje resztę."""
    results: list[dict[str, Any] | None] = [None] * len(items)
    pool = ThreadPoolExecutor(max_workers=max(1, int(concurrency)), thread_name_prefix="llm-esi")
    futures: dict[Future, int] = {pool.submit(worker, model, case): i for i, (model, case) in enumerate(items)}
    try:
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:          # błąd programistyczny pozycji — nie przerywa przebiegu
                model, case = items[i]
                log.error(f"{model} / {case.case_id}: wyjątek {exc!r}")
                results[i] = {"model": model, "case_id": case.case_id, "row_id": case.row_id,
                              "nurse_esi": case.nurse_esi, "decision_esi": case.decision_esi, "valid": 0,
                              "strict": 0, "api_error": f"wyjątek: {exc!r}", "errors": [repr(exc)],
                              "explanation": None, "n_calls": 0, "latency_s": 0.0, "prompt_tokens": 0,
                              "completion_tokens": 0}
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [r for r in results if r is not None]


# ─────────────────────────────────────────
# Podsumowanie
# ─────────────────────────────────────────
def _rate(k: float, n: float) -> float | None:
    return float(k) / float(n) if n else None


def _ci(k: int, n: int) -> list[float | None]:
    from src.evaluation.statistics import clopper_pearson

    if not n:
        return [None, None]
    lo, hi = clopper_pearson(int(k), int(n))
    return [lo, hi]


def _mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else None


def summarize_explanations(items: list[dict[str, Any]], prices: dict[str, Any] | None = None) -> dict[str, Any]:
    """Metryki wspólne części esi i tools (na model): poprawność, wierność, eskalacje, język, zużycie."""
    from src.llm_eval.pricing import cost_usd

    n = len(items)
    valid = [it for it in items if it.get("valid")]
    k = len(valid)
    esc = [it.get("escalation") for it in valid]
    opportunities = sum(1 for it in items if it["nurse_esi"] < it["decision_esi"])
    useful = esc.count("useful")
    false_alarm = esc.count("false_alarm")
    model = items[0]["model"] if items else ""
    prompt_tokens = int(sum(it.get("prompt_tokens") or 0 for it in items))
    completion_tokens = int(sum(it.get("completion_tokens") or 0 for it in items))
    latencies = [float(it.get("latency_s") or 0.0) for it in items if it.get("n_calls")]
    return {
        "n_items": n,
        "api_errors": sum(1 for it in items if it.get("api_error")),
        "valid": k,
        "valid_rate": _rate(k, n),
        "valid_ci": _ci(k, n),
        "strict_rate": _rate(sum(it.get("strict", 0) for it in items), n),
        "retry_rate": _rate(sum(it.get("retried", 0) for it in items), n),
        "level_match_rate": _rate(sum(it["level_match"] for it in valid), k),
        "shap_coverage": _mean([it.get("shap_coverage") for it in valid]),
        "hallucination_rate": _rate(sum(it["hallucination"] for it in valid), k),
        "hallucination_ci": _ci(sum(it["hallucination"] for it in valid), k),
        "numbers_checked": int(sum(it["numbers_checked"] for it in valid)),
        "numbers_wrong": int(sum(it["numbers_wrong"] for it in valid)),
        "esc_suggestions": useful + false_alarm + esc.count("wrong_direction"),
        "esc_useful": useful,
        "esc_false_alarm": false_alarm,
        "esc_wrong_direction": esc.count("wrong_direction"),
        "esc_opportunities": opportunities,
        "esc_missed": sum(1 for it in valid if it["nurse_esi"] < it["decision_esi"] and it["escalation"] == "none"),
        "esc_precision": _rate(useful, useful + false_alarm),
        "esc_recall": _rate(useful, opportunities),
        "polish_rate": _rate(sum(it["polish"] for it in valid), k),
        "citation_rate": _rate(sum(it["cites"] for it in valid), k),
        "invalid_citations": int(sum(len(it.get("citations_invalid") or []) for it in valid)),
        "latency_s_mean": float(np.mean(latencies)) if latencies else None,
        "latency_s_median": float(np.median(latencies)) if latencies else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd(model, prompt_tokens, completion_tokens, prices) if items else None,
    }


# ─────────────────────────────────────────
# Zapis i raport
# ─────────────────────────────────────────
ITEM_COLUMNS = ["model", "case_id", "row_id", "nurse_esi", "decision_esi", "valid", "strict", "via", "retried",
                "api_error", "level_match", "shap_coverage", "numbers_checked", "numbers_wrong", "hallucination",
                "escalation", "polish", "pl_ratio", "cites", "n_calls", "latency_s", "prompt_tokens",
                "completion_tokens"]


def items_frame(items: list[dict[str, Any]], extra_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Pozycje → tabela CSV (z tekstem wyjaśnienia i sugestią eskalacji)."""
    rows = []
    for it in items:
        expl = it.get("explanation") or {}
        esc = expl.get("sugestia_eskalacji") or {}
        row = {c: it.get(c) for c in [*ITEM_COLUMNS, *extra_columns]}
        row.update({
            "poziom_esi": expl.get("poziom_esi"),
            "esc_czy": esc.get("czy"), "esc_poziom": esc.get("poziom"), "esc_powod": esc.get("powod"),
            "kluczowe_czynniki": " | ".join(expl.get("kluczowe_czynniki") or []),
            "uzasadnienie": expl.get("uzasadnienie"),
            "shap_covered": ", ".join(it.get("shap_covered") or []),
            "citations_invalid": ", ".join(it.get("citations_invalid") or []),
            "errors": "; ".join(it.get("errors") or []),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def pct(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{fmt_num(100 * value, digits)}%"


def _ci_text(ci: list[float | None]) -> str:
    if not ci or ci[0] is None:
        return ""
    return f" ({fmt_num(100 * ci[0], 0)}–{fmt_num(100 * ci[1], 0)}%)"


def int_text(value: int | None) -> str:
    """Liczba całkowita z odstępem tysięcy (12 345)."""
    return "—" if value is None else f"{int(value):,}".replace(",", " ")


def _num_text(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else fmt_num(value, digits)


def _usd(value: float | None) -> str:
    from src.llm_eval.pricing import format_usd

    return format_usd(value)


def render_markdown(summary: dict[str, dict[str, Any]], ctx: EsiContext, files: dict[str, str],
                    think: Any) -> str:
    """Sekcja raportu (po polsku, tabele gotowe do pracy); ścieżki plików względem katalogu przebiegu."""
    s = ctx.summary()

    def rel(key: str) -> str:
        path = Path(files.get(key, ""))
        base = Path(files["section_dir"]) if files.get("section_dir") else path.parent
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            return path.as_posix()

    lines = [
        "## Część ESI — wyjaśnienia decyzji triażowej",
        "",
        f"- Przypadki: {plural_pl(s['n_cases'], 'wiersz', 'wiersze', 'wierszy')} VAL z kompletem HR/SBP/RR, "
        f"warstwy wg ESI z triażu "
        f"(pielęgniarka): " + ", ".join(f"ESI {k}: {v}" for k, v in s["nurse_esi_counts"].items())
        + "; decyzje systemu: " + ", ".join(f"ESI {k}: {v}" for k, v in s["decision_esi_counts"].items()) + ".",
        f"- Receptura decyzji ({ctx.recipe.source}): {ctx.recipe_text}.",
        f"- SHAP: model `{s['shap_model']}` (metoda {s['shap_method']}), top-5 cech dla poziomu decyzji.",
        f"- Wymuszone narzędzie `{SUBMIT_NAME}` (bez `format` — chmura go ignoruje); brak wywołania → JSON z treści; "
        f"nadal źle → 1 ponowienie z opisem błędu. Rozumowanie: `think={think!r}`.",
    ]
    if s["live_vs_stored_mismatches"]:
        lines.append(f"- UWAGA: decyzja liczona z modeli różni się od zapisanych predykcji dla "
                     f"{len(s['live_vs_stored_mismatches'])} przypadków ({', '.join(s['live_vs_stored_mismatches'])}).")
    for note in ctx.recipe.notes:
        lines.append(f"- Uwaga: {note}.")
    lines += ["", "| Model | Poprawne wyjście (95% CI) | Ścisły schemat | Poziom = decyzja | Pokrycie top-5 SHAP | "
                  "Halucynacje liczb | Po polsku | Cytuje źródła | Eskalacje: trafne / fałszywe / zły kier. | "
                  "Pominięte okazje | Czas [s] (med.) | Tokeny we / wy | Koszt |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, m in summary.items():
        lines.append(
            f"| `{model}` | {pct(m['valid_rate'])}{_ci_text(m['valid_ci'])} | {pct(m['strict_rate'])} | "
            f"{pct(m['level_match_rate'])} | {pct(m['shap_coverage'])} | {pct(m['hallucination_rate'])} "
            f"({m['numbers_wrong']}/{m['numbers_checked']} liczb) | {pct(m['polish_rate'])} | {pct(m['citation_rate'])} | "
            f"{m['esc_useful']} / {m['esc_false_alarm']} / {m['esc_wrong_direction']} | "
            f"{m['esc_missed']} z {m['esc_opportunities']} | {_num_text(m['latency_s_median'])} | "
            f"{int_text(m['prompt_tokens'])} / {int_text(m['completion_tokens'])} | {_usd(m['cost_usd'])} |")
    lines += [
        "",
        "Definicje: *poprawne wyjście* — obiekt zgodny ze schematem po lekkiej normalizacji (np. „2” → 2), "
        "w tym po ponowieniu; *ścisły schemat* — typy dokładnie jak w schemacie, przez narzędzie; "
        "*halucynacje liczb* — odsetek poprawnych odpowiedzi z co najmniej jedną wartością parametru "
        "niezgodną z przypadkiem (tolerancja: tętno/ciśnienie ±2, oddechy/SpO2/wiek ±1, temperatura "
        "±0,25 °C; progi kliniczne pomijane); *eskalacja trafna* — sugestia pilniejszego poziomu, gdy ESI "
        "z triażu było pilniejsze niż decyzja systemu; *pominięte okazje* — takie przypadki bez sugestii. "
        "Metryki tekstowe są heurystyczne — rozstrzyga ślepa ocena ekspercka.",
        "",
        f"Ślepy arkusz ocen: `{rel('ratings_html')}` (+ `{rel('ratings_template')}`); "
        f"klucz modeli osobno: `{rel('ratings_key')}` (nie przekazywać oceniającym). Po wypełnieniu: "
        f"`python scripts/12_llm_bakeoff.py score-ratings --csv <plik.csv>`.",
        "",
    ]
    return "\n".join(lines)


_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")


def decimal_point(markdown: str) -> str:
    """Przecinek dziesiętny → kropka (sekcja zgodna z resztą report.md bez --decimal-comma).

    Liczby w sekcjach części esi/tools są formatowane po polsku (`fmt_num`); separator tysięcy
    to spacja, więc zamiana „cyfra,cyfra” dotyczy wyłącznie ułamków.
    """
    return _DECIMAL_COMMA.sub(".", markdown)


def skipped_result(part: str, title: str, reason: str) -> dict[str, Any]:
    return {"part": part, "skipped": reason, "summary": {}, "items": [], "files": {},
            "markdown": f"## {title}\n\nPominięto: {reason}\n"}


def run_esi_part(
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
    write_ratings: bool = True,
    decimal_comma: bool = False,
) -> dict[str, Any]:
    """
    Część "esi": każdy model × każdy przypadek. Zwraca {"part", "skipped", "summary" (model →
    metryki), "items", "files", "markdown", "context"}; pliki w out_dir: esi_cases.json,
    esi_items.csv, esi_summary.csv, esi_section.md, ratings/ratings.html, ratings/ratings_template.csv,
    key.json i surowe odpowiedzi (`raw_log` przebiegu albo raw_esi.jsonl). Brak runów / receptury →
    {"skipped": powód} (część exams działa dalej). Kontekst (przypadki) jest współdzielony z częścią
    tools tego samego procesu (`try_prepare_context`). decimal_comma=False → liczby w sekcji
    raportu z kropką dziesiętną (jak domyślny report.md; CLI --decimal-comma).
    """
    from src.llm_eval import ratings as rt
    from src.utils.fileio import write_text_atomic

    title = "Część ESI — wyjaśnienia decyzji triażowej"
    if context is None:
        context, reason = try_prepare_context(recipe_path=recipe, n_cases=n_cases, seed=seed, runs_dir=runs_dir)
        if context is None:
            return skipped_result(PART, title, reason or "brak przypadków")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    think = parse_think(think)
    files = {"cases": str(context.save(out / "esi_cases.json"))}
    raw = raw_log if raw_log is not None else JsonlWriter(out / RAW_FILE)
    concurrency = int(getattr(client, "concurrency", 3) or 3)
    # Przeplot przypadek → model (jak egzaminy): limit planu / awaria chmury pod koniec przebiegu
    # rozkłada się na wszystkie modele, a nie tylko na ostatnie z listy.
    work = [(m, c) for c in context.cases for m in models]
    log.info(f"Część ESI: {len(models)} model(e) × {len(context.cases)} przypadków = {len(work)} pozycji "
             f"(równolegle {concurrency})")

    def worker(model: str, case: EsiCase) -> dict[str, Any]:
        return run_esi_item(client, model, case, context, think=think, options=options, cache=cache, raw=raw)

    items = run_items(work, worker, concurrency)
    summary = {m: summarize_explanations([it for it in items if it["model"] == m], prices) for m in models}

    frame = items_frame(items)
    frame.to_csv(out / "esi_items.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"model": m, **{k: v for k, v in s.items() if not isinstance(v, list)}}
                  for m, s in summary.items()]).to_csv(out / "esi_summary.csv", index=False, encoding="utf-8")
    files.update({"items_csv": str(out / "esi_items.csv"), "summary_csv": str(out / "esi_summary.csv"),
                  "raw": str(raw.path)})
    if write_ratings:
        files.update(rt.write_rating_sheet(out, context, items, models, seed=seed))
    markdown = render_markdown(summary, context, {**files, "section_dir": str(out)}, think)
    if not decimal_comma:
        markdown = decimal_point(markdown)
    write_text_atomic(out / "esi_section.md", markdown)
    files["section_md"] = str(out / "esi_section.md")
    return {"part": PART, "skipped": None, "summary": summary, "items": items, "files": files,
            "markdown": markdown, "context": context.summary()}


__all__ = [
    "EXPLANATION_SCHEMA",
    "SUBMIT_TOOL",
    "SYSTEM_PROMPT_PL",
    "assistant_message",
    "build_messages",
    "cached_chat",
    "citations",
    "coerce_explanation",
    "decimal_point",
    "escalation_outcome",
    "extract_json_object",
    "numeric_faithfulness",
    "parse_submission",
    "polish_check",
    "run_esi_item",
    "run_esi_part",
    "score_explanation",
    "shap_coverage",
    "summarize_explanations",
]
