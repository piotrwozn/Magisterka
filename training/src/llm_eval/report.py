"""
Raport porównania modeli (report.md, po polsku, tabele gotowe do pracy) i wybór modelu.

    sel = select_model(exams_results, esi_results)          # ranking wg uzgodnionego kryterium
    text = render_report(meta, parts, part_markdown, sel, costs)

Kryterium wyboru (uzgodnione z promotorem / autorem pracy)
    1. GŁÓWNE: trafność w protokole abstaining_substitution na LEK + PES (egzaminy NASK).
       Modele, których sparowana różnica z najlepszym ma 95% przedział zawierający 0
       (bootstrap; obok dokładny test McNemara), są z nim w REMISIE.
    2. Remis rozstrzyga protokół standardowy (ta sama reguła remisu, sparowanie na tych samych
       pytaniach), a dalej metryki części „esi”: wynik `ranking_score` ({model: liczba}, im
       więcej, tym lepiej) z wyników `run_esi_part` — gdy go brak, decyduje wartość punktowa
       (raport zaznacza wtedy remis statystyczny).
    Możliwość wdrożenia w szpitalu NIE jest kryterium — tylko notatką przy każdym modelu.

Sekcje części „esi” i „tools” dostarczają ich moduły (markdown zaczynający się od nagłówków ###);
raport tylko je wstawia.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from src.evaluation.report_sections import Fmt, fnum, md_table
from src.llm_eval.client import canonical_model
from src.llm_eval.exams import PROTOCOL_LABELS, PROTOCOLS, SPLIT_LABELS, SPLITS, paired_bootstrap_delta
from src.llm_eval.pricing import PRICING_SOURCE, format_usd

PRIMARY_PROTOCOL = "abstaining_substitution"
SECONDARY_PROTOCOL = "standard"

SELECTION_CRITERION: str = (
    "Kryterium główne: trafność w protokole abstaining_substitution (NASK) na pytaniach LEK + PES. "
    "Modele, których sparowana różnica z najlepszym ma 95% przedział zawierający 0, są z nim w remisie; "
    "remis rozstrzyga trafność w protokole standardowym (ta sama reguła), a następnie wynik części ESI."
)

# Notatki o możliwości wdrożenia lokalnego (NIE kryterium). Źródła: notatki badawcze 2026-10-25
# (karty modeli, llama.cpp, raporty społeczności) — przed cytowaniem w pracy sprawdzić aktualność.
DEPLOYABILITY_NOTES: dict[str, str] = {
    "gpt-oss:120b": (
        "Otwarte wagi (Apache 2.0 + krótka polityka użycia). Lokalnie ok. 65 GB (Ollama) / 59 GiB (GGUF): na "
        "stacji z RTX 4080 Super 16 GB i ≥ 64 GB RAM przez llama.cpp z odciążeniem ekspertów MoE (--n-cpu-moe, "
        "raporty ~25–30 tok/s), w szpitalu na jednym GPU 80 GB. Natywne wywołania narzędzi. Najłatwiejszy do "
        "wdrożenia lokalnego z czterech."
    ),
    "glm-5.3": (
        "Otwarte wagi rodziny GLM-5.x (GLM-5.1: MIT; licencję 5.3 potwierdzić na karcie modelu). Duży model "
        "MoE — lokalnie tylko na serwerze wielo-GPU; na stacji 16 GB niewykonalny."
    ),
    "deepseek-v4.1-flash": (
        "Otwarte wagi (MIT). Bardzo duży MoE (rozbieżne dane: 552–755 mld parametrów; V4-Flash 284 mld / 13 mld "
        "aktywnych) — ~110 GB pamięci nawet w kwantyzacji 3-bitowej; wdrożenie szpitalne wymaga serwera GPU."
    ),
    "kimi-k3": (
        "Licencja niestandardowa z warunkami komercyjnymi (sprawdzić przed użyciem). ~2,8 bln parametrów "
        "(104 mld aktywnych) — w praktyce tylko przez API chmurowe."
    ),
}
CLOUD_NOTE: str = (
    "Pomiar wykonano przez Ollama Cloud: treść zapytań opuszcza szpital. Przy rzeczywistych danych pacjentów "
    "potrzebna jest umowa powierzenia przetwarzania (RODO) z dostawcą albo wdrożenie lokalne."
)


def deployability_note(model: str) -> str:
    name = canonical_model(model)
    return DEPLOYABILITY_NOTES.get(name) or DEPLOYABILITY_NOTES.get(name.split(":", 1)[0], "brak notatki")


# ─────────────────────────────────────────
# Pomocnicze
# ─────────────────────────────────────────
def _metric(exams: dict[str, Any], model: str, protocol: str, split: str = "all") -> dict[str, Any] | None:
    for row in exams.get("metrics", []):
        if row["model"] == model and row["protocol"] == protocol and row["split"] == split:
            return row
    return None


def _acc(exams: dict[str, Any], model: str, protocol: str) -> float:
    row = _metric(exams, model, protocol)
    v = fnum(row.get("accuracy")) if row else None
    return -1.0 if v is None else v


def esi_scores(esi: dict[str, Any] | None) -> dict[str, float]:
    """{model: wynik} z wyników części ESI (`ranking_score` albo `score`); brak → {}."""
    if not isinstance(esi, dict):
        return {}
    raw = esi.get("ranking_score") or esi.get("score")
    if not isinstance(raw, dict):
        return {}
    return {str(m): float(v) for m, v in raw.items() if fnum(v) is not None}


# ─────────────────────────────────────────
# Wybór modelu
# ─────────────────────────────────────────
def _tied(corr: dict[str, np.ndarray] | None, a: str, b: str, n_boot: int, seed: int) -> bool | None:
    if not corr or a not in corr or b not in corr:
        return None
    boot = paired_bootstrap_delta(corr[a], corr[b], n_boot=n_boot, seed=seed)
    return boot["ci_low"] is not None and boot["ci_low"] <= 0 <= boot["ci_high"]


def select_model(exams: dict[str, Any] | None, esi: dict[str, Any] | None = None,
                 n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    """
    Ranking wg kryterium (docstring modułu). Zwraca {criterion, winner, decided_by, primary_tie,
    secondary_tie, ranking: [{rank, model, acc_abstaining, acc_standard, tied_primary, esi_score, note}],
    explanation}.
    """
    scores = esi_scores(esi)
    out: dict[str, Any] = {"criterion": SELECTION_CRITERION, "winner": None, "decided_by": None,
                           "primary_tie": [], "secondary_tie": [], "ranking": [], "explanation": ""}
    if not exams or not exams.get("models"):
        if scores:
            order = sorted(scores, key=lambda m: -scores[m])
            out.update(winner=order[0], decided_by="esi",
                       explanation="Brak części egzaminacyjnej — ranking tylko wg wyniku części ESI (kryterium "
                                   "główne niedostępne).")
            out["ranking"] = [{"rank": i + 1, "model": m, "esi_score": scores[m],
                               "note": deployability_note(m)} for i, m in enumerate(order)]
        else:
            out["explanation"] = "Brak wyników egzaminów i części ESI — nie można wybrać modelu."
        return out

    models = list(exams["models"])
    corr = exams.get("_correctness") or {}
    paired = exams.get("paired_vs_best", {}).get(PRIMARY_PROTOCOL, {})
    best = paired.get("best") or max(models, key=lambda m: (_acc(exams, m, PRIMARY_PROTOCOL),
                                                             _acc(exams, m, SECONDARY_PROTOCOL)))
    g1 = [best] + [r["model"] for r in paired.get("rows", []) if r.get("tied_with_best")]
    g2: list[str] = []
    if len(g1) == 1:
        winner, decided = best, "primary"
        why = (f"{best} ma najwyższą trafność w protokole abstaining_substitution, a różnica z każdym innym "
               f"modelem jest istotna (95% przedział sparowanej różnicy nie zawiera 0).")
    else:
        best2 = max(g1, key=lambda m: (_acc(exams, m, SECONDARY_PROTOCOL), _acc(exams, m, PRIMARY_PROTOCOL)))
        g2 = [best2] + [m for m in g1 if m != best2
                        and _tied(corr.get(SECONDARY_PROTOCOL), m, best2, n_boot, seed) is not False]
        tie_txt = ", ".join(g1)
        if len(g2) == 1:
            winner, decided = best2, "standard"
            why = (f"Remis w kryterium głównym ({tie_txt}); rozstrzyga protokół standardowy — {best2} jest w nim "
                   f"istotnie lepszy od pozostałych modeli z remisu.")
        elif scores and all(m in scores for m in g2):
            winner, decided = max(g2, key=lambda m: scores[m]), "esi"
            why = (f"Remis w kryterium głównym ({tie_txt}) i w protokole standardowym ({', '.join(g2)}); "
                   f"rozstrzyga wynik części ESI — najwyższy ma {max(g2, key=lambda m: scores[m])}.")
        else:
            winner = max(g2, key=lambda m: (_acc(exams, m, PRIMARY_PROTOCOL), _acc(exams, m, SECONDARY_PROTOCOL)))
            decided = "point_estimate"
            why = (f"Remis statystyczny w kryterium głównym ({tie_txt}) i w protokole standardowym "
                   f"({', '.join(g2)}); brak wyniku części ESI do rozstrzygnięcia — wskazano model z najwyższą "
                   f"wartością punktową ({winner}). Wybór między modelami z remisu nie jest rozstrzygnięty "
                   f"statystycznie — warto zwiększyć próbę (--n-exam / --n-pes) albo uruchomić część ESI.")

    def key(m: str) -> tuple:
        return (m != winner, m not in g2, m not in g1, -_acc(exams, m, PRIMARY_PROTOCOL),
                -_acc(exams, m, SECONDARY_PROTOCOL), m)

    ranking = []
    for i, m in enumerate(sorted(models, key=key)):
        ranking.append({
            "rank": i + 1, "model": m,
            "acc_abstaining": _metric(exams, m, PRIMARY_PROTOCOL),
            "acc_standard": _metric(exams, m, SECONDARY_PROTOCOL),
            "tied_primary": m in g1, "tied_secondary": m in g2,
            "esi_score": scores.get(m), "note": deployability_note(m),
        })
    out.update(winner=winner, decided_by=decided, primary_tie=g1 if len(g1) > 1 else [],
               secondary_tie=g2 if len(g2) > 1 else [], ranking=ranking, explanation=why)
    return out


# ─────────────────────────────────────────
# Sekcje markdown
# ─────────────────────────────────────────
def _rate(fmt: Fmt, row: dict[str, Any] | None) -> str:
    if not row or not row.get("n"):
        return "—"
    return fmt.rate(row["correct"], row["n"])


def exams_markdown(exams: dict[str, Any], fmt: Fmt | None = None) -> list[str]:
    """Sekcja egzaminów: próba, trafność z CI per protokół, porównania parami, luka, opóźnienia."""
    fmt = fmt or Fmt()
    models = list(exams.get("models", []))
    s = exams.get("sample", {})
    n = s.get("n", {})
    pes_txt = ("medycyna ratunkowa / choroby wewnętrzne (rozpoznane w nazwie egzaminu)"
               if s.get("pes_mode") == "specialty" else "losowe pytania PES (specjalizacji nie rozpoznano)")
    lines = [
        "## 1. Polskie egzaminy lekarskie (NASK-PIB/Reassess-Polish-Medical-Exams)",
        "",
        f"Próba: **{n.get('lek', 0)} pytań LEK** (losowanie warstwowe wg edycji egzaminu, {s.get('lek_editions', '?')} "
        f"edycji) + **{n.get('pes', 0)} pytań PES** ({pes_txt}); seed {s.get('seed')}, te same pytania dla każdego "
        f"modelu i obu protokołów. Pula: pytania jednokrotnego wyboru A–E kwalifikujące się do protokołu "
        f"abstaining_substitution (wykluczenia jak w skrypcie NASK).",
        "",
        "Protokoły: **standardowy** (pytanie bez zmian) i **abstaining_substitution** (tekst poprawnej opcji "
        "zastąpiony zdaniem „żadna z odpowiedzi nie jest poprawna”; litera poprawnej odpowiedzi bez zmian). "
        f"Prompt NASK „prev_method” (po polsku), rozumowanie modelu: `think={exams.get('think')}`. "
        "Odpowiedź bez rozpoznawalnej litery i błąd API po wszystkich ponowieniach liczone są jako błędne.",
        "",
    ]
    for protocol in (PRIMARY_PROTOCOL, SECONDARY_PROTOCOL):
        title = "kryterium główne" if protocol == PRIMARY_PROTOCOL else "kontrola i rozstrzyganie remisów"
        lines += [f"### Trafność — protokół {PROTOCOL_LABELS[protocol]} ({title})", ""]
        rows = []
        for m in models:
            cells = [f"`{m}`"] + [_rate(fmt, _metric(exams, m, protocol, sp)) for sp in (*SPLITS, "all")]
            tot = _metric(exams, m, protocol) or {}
            cells += [fmt.pct(tot.get("parse_failure_rate")) + f" ({tot.get('parse_failures', 0)})",
                      str(tot.get("api_errors", 0))]
            rows.append(cells)
        lines += [md_table(["Model", SPLIT_LABELS[SPLITS[0]], SPLIT_LABELS[SPLITS[1]], SPLIT_LABELS["all"],
                            "Brak litery", "Błędy API"], rows),
                  "", "Trafność [95% przedział Clopper–Pearsona] (poprawne/wszystkie).", ""]

    lines += ["### Porównanie z najlepszym modelem (LEK+PES, sparowane na tych samych pytaniach)", ""]
    for protocol in (PRIMARY_PROTOCOL, SECONDARY_PROTOCOL):
        block = exams.get("paired_vs_best", {}).get(protocol, {})
        best = block.get("best")
        if not best:
            continue
        lines += [f"Protokół {PROTOCOL_LABELS[protocol]} — najlepszy: `{best}`.", ""]
        rows = []
        for r in block.get("rows", []):
            verdict = "remis (różnica nieistotna)" if r.get("tied_with_best") else "istotnie gorszy"
            rows.append([f"`{r['model']}`", fmt.pp(r.get("delta"), r.get("ci_low"), r.get("ci_high")),
                         f"{r.get('model_only', 0)} / {r.get('best_only', 0)}", fmt.p(r.get("p_mcnemar")), verdict])
        if rows:
            lines += [md_table(["Model", "Δ trafności vs najlepszy [95% CI]", "tylko model / tylko najlepszy",
                                "p (McNemar, dokładny)", "Wniosek"], rows), ""]
    lines += ["Δ w punktach procentowych (model − najlepszy), przedział z bootstrapu sparowanego "
              f"({exams.get('n_boot', 2000)} losowań pytań). Kolumna „tylko model / tylko najlepszy” — liczba pytań, "
              "na które poprawnie odpowiedział tylko jeden z dwóch modeli (pary niezgodne testu McNemara).", ""]

    lines += ["### Luka standard − abstaining (wskaźnik „bias” NASK)", ""]
    rows = [[f"`{g['model']}`", fmt.pct(g.get("acc_standard")), fmt.pct(g.get("acc_abstaining")),
             fmt.pp(g.get("delta"), g.get("ci_low"), g.get("ci_high")), fmt.p(g.get("p_mcnemar"))]
            for g in exams.get("gap", [])]
    lines += [md_table(["Model", "Standard", "Abstaining", "Δ [95% CI]", "p (McNemar)"], rows), "",
              "Duża luka = model częściej „zgaduje” wzorzec odpowiedzi niż rozumie pytanie (NASK: spadek "
              "o ~30 pp u najlepszych modeli otwartych).", ""]

    lines += ["### Opóźnienie, tokeny i koszt (egzaminy, oba protokoły)", ""]
    rows = []
    for m in models:
        rs = [_metric(exams, m, p) or {} for p in PROTOCOLS]
        lat = [fnum(r.get("latency_median_s")) for r in rs if fnum(r.get("latency_median_s")) is not None]
        p90 = [fnum(r.get("latency_p90_s")) for r in rs if fnum(r.get("latency_p90_s")) is not None]
        pin = [fnum(r.get("prompt_tokens_mean")) for r in rs if fnum(r.get("prompt_tokens_mean")) is not None]
        pout = [fnum(r.get("completion_tokens_mean")) for r in rs if fnum(r.get("completion_tokens_mean")) is not None]
        costs = [fnum(r.get("cost_usd")) for r in rs]
        cost = None if any(c is None for c in costs) else sum(costs)
        rows.append([f"`{m}`", fmt.num(np.mean(lat) if lat else None, 1), fmt.num(np.mean(p90) if p90 else None, 1),
                     fmt.num(np.mean(pin) if pin else None, 0), fmt.num(np.mean(pout) if pout else None, 0),
                     format_usd(cost)])
    lines += [md_table(["Model", "Mediana opóźnienia [s]", "p90 [s]", "Tokeny wej. (śr.)", "Tokeny wyj. (śr.)",
                        "Koszt (cennik API)"], rows), ""]
    return lines


def costs_markdown(costs: list[dict[str, Any]], fmt: Fmt | None = None) -> list[str]:
    """Tabela kosztów per część × model."""
    fmt = fmt or Fmt()
    if not costs:
        return ["Brak danych o zużyciu tokenów.", ""]
    rows = [[c.get("part", "?"), f"`{c.get('model')}`", str(int(c.get("calls") or 0)),
             f"{int(c.get('prompt_tokens') or 0):,}".replace(",", " "),
             f"{int(c.get('completion_tokens') or 0):,}".replace(",", " "), format_usd(c.get("cost_usd"))]
            for c in costs]
    total = [c.get("cost_usd") for c in costs]
    lines = [md_table(["Część", "Model", "Wywołania", "Tokeny wej.", "Tokeny wyj.", "Koszt"], rows), ""]
    if all(t is not None for t in total):
        lines += [f"Razem: **{format_usd(sum(total))}** (ceny: {PRICING_SOURCE}; przy abonamencie Ollama "
                  f"koszt orientacyjny — zapytania nie są rozliczane per token).", ""]
    else:
        lines += [f"Ceny: {PRICING_SOURCE}; „—” = model bez ceny w cenniku (--prices).", ""]
    return lines


def selection_markdown(sel: dict[str, Any], fmt: Fmt | None = None) -> list[str]:
    fmt = fmt or Fmt()
    lines = ["## Wybór modelu", "", sel.get("criterion", SELECTION_CRITERION), ""]
    if sel.get("winner"):
        lines += [f"**Wybrany model: `{sel['winner']}`.** {sel.get('explanation', '')}", ""]
    else:
        lines += [sel.get("explanation", "Brak rozstrzygnięcia."), ""]
    rows = []
    for r in sel.get("ranking", []):
        abst, std = r.get("acc_abstaining") or {}, r.get("acc_standard") or {}
        rows.append([str(r["rank"]), f"`{r['model']}`",
                     fmt.pct_ci(abst.get("accuracy"), abst.get("ci_low"), abst.get("ci_high")) if abst else "—",
                     fmt.pct_ci(std.get("accuracy"), std.get("ci_low"), std.get("ci_high")) if std else "—",
                     "tak" if r.get("tied_primary") else "nie",
                     fmt.num(r.get("esi_score"), 3) if r.get("esi_score") is not None else "—"])
    if rows:
        lines += [md_table(["#", "Model", "Abstaining (LEK+PES)", "Standard (LEK+PES)", "Remis z najlepszym",
                            "Wynik ESI"], rows), ""]
    lines += ["### Możliwość wdrożenia (notatka, nie kryterium)", ""]
    for r in sel.get("ranking", []):
        lines.append(f"- `{r['model']}` — {r.get('note') or deployability_note(r['model'])}")
    lines += ["", CLOUD_NOTE, ""]
    return lines


def meta_markdown(meta: dict[str, Any]) -> list[str]:
    server = meta.get("server", {}) or {}
    lines = [
        f"- Data uruchomienia: {meta.get('started_at', '?')} (zakończenie: {meta.get('finished_at', '?')})",
        f"- Serwer Ollama: `{server.get('base_url', '?')}` (tryb: {'chmura (bezpośrednio)' if server.get('mode') == 'cloud' else 'lokalny demon'}), "
        f"wersja: {server.get('version') or 'nieznana'}" + (f" — błąd: {server['error']}" if server.get("error") else ""),
        f"- Rozumowanie (`think`): {meta.get('think')}; równoległość: {meta.get('concurrency')}; seed: {meta.get('seed')}",
        f"- Polecenie: `{meta.get('command', '')}`",
        "",
    ]
    rows = []
    for m, info in (meta.get("model_info") or {}).items():
        d = info.get("details", {}) or {}
        rows.append([f"`{m}`", info.get("api_model", m), (info.get("digest") or "—")[:12],
                     d.get("family") or "—", d.get("parameter_size") or "—", d.get("quantization_level") or "—",
                     info.get("remote_model") or "—", info.get("error") or ""])
    if rows:
        lines += [md_table(["Model (tag)", "Nazwa w API", "Digest", "Rodzina", "Parametry", "Kwantyzacja",
                            "Model zdalny", "Uwagi"], rows), ""]
    excluded = meta.get("excluded_models") or {}
    if excluded:
        lines += ["Modele wyłączone po teście łączności:", ""]
        lines += [f"- `{m}` — {why}" for m, why in excluded.items()] + [""]
    return lines


def render_report(meta: dict[str, Any], parts: dict[str, Any], part_markdown: dict[str, str],
                  selection: dict[str, Any], costs: list[dict[str, Any]], decimal_comma: bool = False) -> str:
    """Cały report.md."""
    fmt = Fmt(decimal_comma=decimal_comma)
    date = meta.get("started_at") or datetime.now().isoformat(timespec="seconds")
    lines = [f"# Porównanie modeli językowych do wyjaśnień triażu ESI — {date[:10]}", ""]
    if selection.get("winner"):
        lines += [f"**Wynik w skrócie:** wybrany model `{selection['winner']}` "
                  f"(rozstrzygnięcie: {_decided_label(selection.get('decided_by'))}). Szczegóły w sekcji „Wybór modelu”.", ""]
    lines += ["## Warunki pomiaru", ""] + meta_markdown(meta)

    exams = parts.get("exams")
    if isinstance(exams, dict) and exams.get("metrics"):
        lines += exams_markdown(exams, fmt)
    else:
        lines += ["## 1. Polskie egzaminy lekarskie", "", _skip_text(parts.get("exams"), "exams"), ""]
    for num, (part, title) in enumerate((("esi", "Wyjaśnienia decyzji ESI (polskie przypadki triażu)"),
                                         ("tools", "Agent z narzędziami")), start=2):
        lines += [f"## {num}. {title}", ""]
        md = part_markdown.get(part)
        lines += [md.strip(), ""] if md else [_skip_text(parts.get(part), part), ""]
    lines += ["## 4. Koszty i zużycie tokenów", ""] + costs_markdown(costs, fmt)
    lines += selection_markdown(selection, fmt)
    lines += [
        "## Ograniczenia",
        "",
        "- Pytania egzaminacyjne mogły trafić do danych treningowych modeli (kontaminacja; część pytań LEK "
        "pochodzi z publicznej bazy) — protokół abstaining_substitution częściowo to kontroluje.",
        "- Przedziały ufności opisują niepewność z powodu skończonej próby pytań, nie zmienność samego modelu "
        "(temperatura 0, ale modele chmurowe nie gwarantują determinizmu).",
        "- Modele chmurowe mogą być po cichu aktualizowane — wersja serwera, digest i data pomiaru są podane wyżej.",
        "",
    ]
    return "\n".join(lines)


def _decided_label(decided: str | None) -> str:
    return {"primary": "kryterium główne (abstaining)", "standard": "remis → protokół standardowy",
            "esi": "remis → część ESI", "point_estimate": "remis statystyczny → wartość punktowa"}.get(decided or "", "—")


def _skip_text(part: Any, name: str) -> str:
    if isinstance(part, dict) and part.get("skipped"):
        return f"Część pominięta: {part['skipped']}"
    if isinstance(part, dict) and part.get("failed"):
        return f"Część nieudana: {part['failed']}"
    return f"Części „{name}” nie uruchomiono (--parts)."


__all__ = [
    "CLOUD_NOTE",
    "DEPLOYABILITY_NOTES",
    "SELECTION_CRITERION",
    "costs_markdown",
    "deployability_note",
    "esi_scores",
    "exams_markdown",
    "meta_markdown",
    "render_report",
    "select_model",
    "selection_markdown",
]
