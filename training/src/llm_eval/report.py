"""
Raport porównania modeli (report.md, po polsku, tabele gotowe do pracy) i wybór modelu.

    sel = select_model(exams_results, esi_results)          # ranking wg uzgodnionego kryterium
    text = render_report(meta, parts, part_markdown, sel, costs)

Kryterium wyboru (uzgodnione z promotorem / autorem pracy)
    1. GŁÓWNE: trafność w protokole abstaining_substitution na LEK + PES (egzaminy NASK).
       Model jest z najlepszym w REMISIE, chyba że różnica jest istotna w OBU testach: 95%
       przedział sparowanego bootstrapu Δ nie zawiera 0 ORAZ dokładny test McNemara p < 0,05
       (`exams.is_tie` — ta sama reguła w tabeli porównań i w wyborze).
    2. Remis rozstrzyga protokół standardowy (ta sama reguła remisu, sparowanie na tych samych
       pytaniach), a dalej metryki części „esi”: wynik `ranking_score` ({model: liczba}, im
       więcej, tym lepiej) z wyników `run_esi_part` albo średnia składników ESI_SCORE_COMPONENTS
       liczonych na WSZYSTKICH przypadkach (`esi_scores`) — gdy go brak, decyduje wartość
       punktowa (raport zaznacza wtedy remis statystyczny).
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
from src.llm_eval.exams import (
    PROTOCOL_LABELS,
    PROTOCOLS,
    SPLIT_LABELS,
    SPLITS,
    is_tie,
    mcnemar_exact,
    paired_bootstrap_delta,
)
from src.llm_eval.pricing import PRICING_SOURCE, format_usd

PRIMARY_PROTOCOL = "abstaining_substitution"
SECONDARY_PROTOCOL = "standard"

SELECTION_CRITERION: str = (
    "Kryterium główne: trafność w protokole abstaining_substitution (NASK) na pytaniach LEK + PES. "
    "Model jest w remisie z najlepszym, chyba że różnica jest istotna w obu testach: 95% przedział "
    "sparowanego bootstrapu różnicy nie zawiera 0 i dokładny test McNemara daje p < 0,05. "
    "Remis rozstrzyga trafność w protokole standardowym (ta sama reguła), a następnie wynik części ESI "
    "(średnia na wszystkich przypadkach: poprawna struktura, zgodność poziomu ESI, pokrycie cech SHAP, brak "
    "halucynacji, język polski; niepoprawna odpowiedź liczy się jako porażka w każdym składniku)."
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


# Wynik części ESI do rozstrzygania remisów, gdy część nie podała własnego `ranking_score`:
# średnia (równe wagi) z poniższych odsetków w `summary[model]` (brakujące pomijane). Składniki
# po `valid_rate` są w podsumowaniu liczone tylko na POPRAWNYCH odpowiedziach — tu przechodzą na
# wszystkie przypadki (× valid_rate; niepoprawna odpowiedź = poziom niezgodny, pokrycie 0,
# halucynacja, nie po polsku), inaczej model z 2/40 poprawnymi, ale „idealnymi” odpowiedziami
# wygrywałby z modelem rzetelnym.
ESI_SCORE_COMPONENTS: tuple[tuple[str, bool], ...] = (
    ("valid_rate", False),            # poprawna struktura odpowiedzi
    ("level_match_rate", False),      # przytoczony poziom ESI = decyzja systemu
    ("shap_coverage", False),         # pokrycie 5 najważniejszych cech SHAP
    ("hallucination_rate", True),     # zmyślone pomiary (odwrócone: 1 − odsetek)
    ("polish_rate", False),           # odpowiedź po polsku
)
ESI_SCORE_FORMULA: str = (
    "wynik ESI = średnia z: odsetka poprawnych odpowiedzi oraz odsetków WSZYSTKICH przypadków z poprawną "
    "odpowiedzią, która ma poziom zgodny z decyzją / pokrycie cech SHAP / brak zmyślonych liczb / język polski "
    "(niepoprawna odpowiedź = porażka w każdym składniku)"
)


def esi_scores(esi: dict[str, Any] | None) -> dict[str, float]:
    """
    {model: wynik ∈ [0, 1]} z wyników części ESI: `ranking_score` / `score` podany przez część,
    a gdy go brak — średnia ESI_SCORE_COMPONENTS z `summary[model]` (ESI_SCORE_FORMULA: odsetki
    warunkowe na poprawnych odpowiedziach × valid_rate, czyli na wszystkich przypadkach); brak danych → {}.
    """
    if not isinstance(esi, dict) or esi.get("skipped"):
        return {}
    raw = esi.get("ranking_score") or esi.get("score")
    if isinstance(raw, dict):
        return {str(m): float(v) for m, v in raw.items() if fnum(v) is not None}
    summary = esi.get("summary")
    if not isinstance(summary, dict):
        return {}
    out = {}
    for model, s in summary.items():
        if not isinstance(s, dict):
            continue
        valid = fnum(s.get("valid_rate"))
        vals = []
        for key, invert in ESI_SCORE_COMPONENTS:
            v = fnum(s.get(key))
            if key == "valid_rate" or valid is None:
                if v is not None:                       # bez valid_rate: jak dotąd (odsetki wprost)
                    vals.append((1.0 - v) if invert else v)
                continue
            if v is None:
                if valid == 0.0 and key in s:           # brak poprawnych odpowiedzi → składnik = 0
                    vals.append(0.0)
                continue
            vals.append(valid * ((1.0 - v) if invert else v))
        if vals:
            out[str(model)] = float(np.mean(vals))
    return out


# ─────────────────────────────────────────
# Wybór modelu
# ─────────────────────────────────────────
def _tied(corr: dict[str, np.ndarray] | None, a: str, b: str, n_boot: int, seed: int) -> bool | None:
    """Remis wg reguły `exams.is_tie` (bootstrap Δ i dokładny McNemar muszą oba wskazać różnicę)."""
    if not corr or a not in corr or b not in corr:
        return None
    boot = paired_bootstrap_delta(corr[a], corr[b], n_boot=n_boot, seed=seed)
    return is_tie(boot, mcnemar_exact(corr[a], corr[b])["p_mcnemar"])


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
                                   "główne niedostępne). Do wyboru według uzgodnionego kryterium uruchom wszystkie "
                                   "części jednym poleceniem (`--parts exams,esi,tools`) — odpowiedzi egzaminacyjne "
                                   "z wcześniejszego przebiegu zostaną wzięte z pamięci podręcznej.")
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
               f"modelem jest istotna (95% przedział sparowanej różnicy nie zawiera 0 i dokładny test McNemara "
               f"p < 0,05).")
    else:
        best2 = max(g1, key=lambda m: (_acc(exams, m, SECONDARY_PROTOCOL), _acc(exams, m, PRIMARY_PROTOCOL)))
        g2 = [best2] + [m for m in g1 if m != best2
                        and _tied(corr.get(SECONDARY_PROTOCOL), m, best2, n_boot, seed) is not False]
        tie_txt = ", ".join(g1)
        if len(g2) == 1:
            winner, decided = best2, "standard"
            why = (f"Remis w kryterium głównym ({tie_txt}); rozstrzyga protokół standardowy — {best2} jest w nim "
                   f"istotnie lepszy od pozostałych modeli z remisu.")
        elif scores and all(m in scores for m in g2) and \
                sum(abs(scores[m] - max(scores[x] for x in g2)) < 1e-9 for m in g2) == 1:
            winner, decided = max(g2, key=lambda m: scores[m]), "esi"
            why = (f"Remis w kryterium głównym ({tie_txt}) i w protokole standardowym ({', '.join(g2)}); "
                   f"rozstrzyga wynik części ESI — najwyższy ma {max(g2, key=lambda m: scores[m])}.")
        else:
            winner = max(g2, key=lambda m: (_acc(exams, m, PRIMARY_PROTOCOL), _acc(exams, m, SECONDARY_PROTOCOL)))
            decided = "point_estimate"
            why = (f"Remis statystyczny w kryterium głównym ({tie_txt}) i w protokole standardowym "
                   f"({', '.join(g2)}); brak (albo remis) wyniku części ESI — wskazano model z najwyższą "
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
    out.update(winner=winner, decided_by=decided, best_primary=best, primary_tie=g1 if len(g1) > 1 else [],
               secondary_tie=g2 if len(g2) > 1 else [], ranking=ranking, explanation=why)
    return out


# ─────────────────────────────────────────
# Sekcje markdown
# ─────────────────────────────────────────
def _rate(fmt: Fmt, row: dict[str, Any] | None) -> str:
    if not row or not row.get("n"):
        return "—"
    return fmt.rate(row["correct"], row["n"])


def _pes_names(names: list[str], limit: int = 12) -> str:
    """Egzaminy PES w próbie: nazwy (do sprawdzenia filtra specjalizacji) albo sama liczba, gdy jest ich dużo."""
    if not names:
        return "brak egzaminów"
    if len(names) > limit:
        return f"liczba egzaminów: {len(names)}"
    return "egzaminy: " + ", ".join(names)


def _source_text(source: dict[str, Any] | None) -> str:
    """Źródło pytań do raportu: zbiór HF z rewizją albo pliki z sumą SHA-256."""
    if not source:
        return ""
    if source.get("type") == "hf":
        rev = source.get("revision")
        return (f"Źródło: Hugging Face `{source.get('repo')}`"
                + (f", rewizja `{str(rev)[:12]}`." if rev else " (rewizji nie udało się ustalić)."))
    if source.get("type") == "file":
        files = ", ".join(f"`{f['file']}`" + (f" (SHA-256 {f['sha256']})" if f.get("sha256") else "")
                          for f in source.get("files", []))
        return f"Źródło: plik(i) {files}."
    return ""


def exams_markdown(exams: dict[str, Any], fmt: Fmt | None = None) -> list[str]:
    """Sekcja egzaminów: próba, trafność z CI per protokół, porównania parami, luka, opóźnienia."""
    fmt = fmt or Fmt()
    models = list(exams.get("models", []))
    s = exams.get("sample", {})
    n = s.get("n", {})
    pes_txt = ("specjalizacje medycyna ratunkowa i choroby wewnętrzne, rozpoznane w nazwie egzaminu"
               if s.get("pes_mode") == "specialty" else "losowe pytania PES — specjalizacji nie rozpoznano")
    lines = [
        "## Polskie egzaminy lekarskie (NASK-PIB/Reassess-Polish-Medical-Exams)",
        "",
        f"Próba (seed {s.get('seed')}, te same pytania dla każdego modelu i obu protokołów): liczba pytań LEK "
        f"**{n.get('lek', 0)}** (losowanie warstwowe wg edycji egzaminu; liczba edycji: {s.get('lek_editions', '?')}), "
        f"liczba pytań PES **{n.get('pes', 0)}** ({pes_txt}; {_pes_names(s.get('pes_exam_names') or [])}). "
        f"Pula: pytania jednokrotnego wyboru A–E kwalifikujące się do protokołu abstaining_substitution "
        f"(wykluczenia jak w skrypcie NASK). {_source_text(s.get('source'))}",
        "",
        "Protokoły: **standardowy** (pytanie bez zmian) i **abstaining_substitution** (tekst poprawnej opcji "
        "zastąpiony zdaniem „żadna z odpowiedzi nie jest poprawna”; litera poprawnej odpowiedzi bez zmian). "
        f"Prompt NASK „prev_method” (po polsku), rozumowanie modelu: `think={exams.get('think')}` (wartość "
        "faktycznie wysyłana do każdego modelu — kolumna „think (wysyłane)” w „Warunkach pomiaru”). "
        "Odpowiedź bez rozpoznawalnej litery i błąd API po wszystkich ponowieniach liczone są jako błędne.",
        "",
    ]
    for protocol in (PRIMARY_PROTOCOL, SECONDARY_PROTOCOL):
        title = "kryterium główne" if protocol == PRIMARY_PROTOCOL else "kontrola i rozstrzyganie remisów"
        lines += [f"### Trafność: protokół {protocol} — {title}", "", f"Protokół {PROTOCOL_LABELS[protocol]}.", ""]
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
        lines += [f"Protokół {protocol} — najlepszy: `{best}`.", ""]
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
              "na które poprawnie odpowiedział tylko jeden z dwóch modeli (pary niezgodne testu McNemara). "
              "„Istotnie gorszy” tylko wtedy, gdy oba testy wskazują różnicę (przedział nie zawiera 0 i p < 0,05); "
              "w każdym innym przypadku remis.", ""]

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
             f"{int(c.get('completion_tokens') or 0):,}".replace(",", " "),
             fmt.num(fnum(c.get("latency_median_s")), 1), format_usd(c.get("cost_usd"))]
            for c in costs]
    total = [c.get("cost_usd") for c in costs]
    lines = [md_table(["Część", "Model", "Pozycje", "Tokeny wej.", "Tokeny wyj.", "Czas na pozycję [s] (mediana)",
                       "Koszt"], rows), "",
             "Pozycja: w części exams jedno pytanie w jednym protokole (jedno zapytanie), w części esi jeden "
             "przypadek (z ewentualnym ponowieniem), w części tools jeden przypadek (wszystkie kroki agenta). "
             "Tokeny i koszt obejmują też odpowiedzi z pamięci podręcznej (koszt całego pomiaru).", ""]
    if all(t is not None for t in total):
        lines += [f"Razem: **{format_usd(sum(total))}** (ceny: {PRICING_SOURCE}; przy abonamencie Ollama "
                  f"koszt orientacyjny — zapytania nie są rozliczane per token).", ""]
    else:
        lines += [f"Ceny: {PRICING_SOURCE}; „—” = model bez ceny w cenniku (--prices).", ""]
    return lines


def uses_cloud(meta: dict[str, Any]) -> bool:
    """Czy pomiar szedł przez chmurę Ollamy (tryb bezpośredni, tag ':cloud'/'-cloud' albo model zdalny)."""
    server = meta.get("server") or {}
    models = [str(m) for m in meta.get("models") or []]
    remote = any((info or {}).get("remote_host") for info in (meta.get("model_info") or {}).values())
    return server.get("mode") == "cloud" or remote or any(canonical_model(m) != m for m in models)


def excluded_banner(excluded: dict[str, Any] | None) -> list[str]:
    """Ostrzeżenie: ranking bez modeli wyłączonych po teście łączności (wybór niepełny)."""
    if not excluded:
        return []
    names = ", ".join(f"`{m}`" for m in excluded)
    return [f"> **Uwaga:** wybór bez modelu(-i) {names} — wyłączone po teście łączności (powód w „Warunkach "
            "pomiaru”), więc nie ma ich w żadnej części ani w rankingu. Uruchom to samo polecenie ponownie, gdy "
            "model odpowiada (gotowe odpowiedzi pozostałych modeli zostaną wzięte z pamięci podręcznej), zanim "
            "użyjesz wyniku w pracy.", ""]


def selection_markdown(sel: dict[str, Any], fmt: Fmt | None = None, cloud: bool = True,
                       excluded: dict[str, Any] | None = None) -> list[str]:
    fmt = fmt or Fmt()
    lines = ["## Wybór modelu", "", sel.get("criterion", SELECTION_CRITERION), ""]
    lines += excluded_banner(excluded)
    if sel.get("winner"):
        lines += [f"**Wybrany model: `{sel['winner']}`.** {sel.get('explanation', '')}", ""]
    else:
        lines += [sel.get("explanation", "Brak rozstrzygnięcia."), ""]
    rows = []
    best_primary = sel.get("best_primary")
    for r in sel.get("ranking", []):
        abst, std = r.get("acc_abstaining") or {}, r.get("acc_standard") or {}
        rows.append([str(r["rank"]), f"`{r['model']}`",
                     fmt.pct_ci(abst.get("accuracy"), abst.get("ci_low"), abst.get("ci_high")) if abst else "—",
                     fmt.pct_ci(std.get("accuracy"), std.get("ci_low"), std.get("ci_high")) if std else "—",
                     "—" if not abst else
                     "najlepszy" if r["model"] == best_primary else ("tak" if r.get("tied_primary") else "nie"),
                     fmt.num(r.get("esi_score"), 3) if r.get("esi_score") is not None else "—"])
    if rows:
        lines += [md_table(["#", "Model", "Abstaining (LEK+PES)", "Standard (LEK+PES)", "Remis z najlepszym",
                            "Wynik ESI"], rows), ""]
        if any(r.get("esi_score") is not None for r in sel.get("ranking", [])):
            lines += [f"Wynik ESI: {ESI_SCORE_FORMULA} (gdy część nie podała własnego wyniku).", ""]
    lines += ["### Możliwość wdrożenia (notatka, nie kryterium)", ""]
    for r in sel.get("ranking", []):
        lines.append(f"- `{r['model']}` — {r.get('note') or deployability_note(r['model'])}")
    lines += ["", CLOUD_NOTE if cloud else "Pomiar wykonano na lokalnym serwerze Ollamy (bez chmury).", ""]
    return lines


def _think_text(sent: Any, requested: Any) -> str:
    """Wartość `think` wysyłana do modelu (None przy żądanej wartości = serwer odrzucił pole)."""
    if sent is not None:
        return str(sent)
    return "domyślne modelu" if requested is None else "pominięte (serwer odrzucił think) — domyślne modelu"


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
    effective = meta.get("think_effective") or {}
    for m, info in (meta.get("model_info") or {}).items():
        d = info.get("details", {}) or {}
        sent = effective.get(m, meta.get("think"))
        rows.append([f"`{m}`", info.get("api_model", m), (info.get("digest") or "—")[:12],
                     d.get("family") or "—", d.get("parameter_size") or "—", d.get("quantization_level") or "—",
                     info.get("remote_model") or "—", _think_text(sent, meta.get("think")),
                     info.get("error") or ""])
    if rows:
        lines += [md_table(["Model (tag)", "Nazwa w API", "Digest", "Rodzina", "Parametry", "Kwantyzacja",
                            "Model zdalny", "think (wysyłane)", "Uwagi"], rows), ""]
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
    excluded = meta.get("excluded_models") or {}
    if selection.get("winner"):
        without = (" — **bez** " + ", ".join(f"`{m}`" for m in excluded)) if excluded else ""
        lines += [f"**Wynik w skrócie:** wybrany model `{selection['winner']}` "
                  f"(rozstrzygnięcie: {_decided_label(selection.get('decided_by'))}){without}. "
                  "Szczegóły w sekcji „Wybór modelu”.", ""]
    lines += excluded_banner(excluded)
    lines += api_error_banner(parts)
    lines += ["## Warunki pomiaru", ""] + meta_markdown(meta)

    exams = parts.get("exams")
    if isinstance(exams, dict) and exams.get("metrics"):
        lines += exams_markdown(exams, fmt)
    else:
        lines += ["## Polskie egzaminy lekarskie", "", _skip_text(parts.get("exams"), "exams"), ""]
    for part, title in (("esi", "Wyjaśnienia decyzji ESI (polskie przypadki triażu)"),
                        ("tools", "Agent z narzędziami")):
        md = (part_markdown.get(part) or "").strip()
        if not md.startswith("## "):          # sekcja części bez własnego nagłówka najwyższego poziomu
            lines += [f"## {title}", ""]
        lines += [md, ""] if md else [_skip_text(parts.get(part), part), ""]
    lines += ["## Koszty i zużycie tokenów", ""] + costs_markdown(costs, fmt)
    lines += selection_markdown(selection, fmt, cloud=uses_cloud(meta), excluded=excluded)
    lines += limitations_markdown(parts)
    lines += files_markdown(parts)
    return "\n".join(lines)


def _ran(part: Any) -> bool:
    return isinstance(part, dict) and not part.get("skipped") and not part.get("failed")


def limitations_markdown(parts: dict[str, Any]) -> list[str]:
    """Ograniczenia pomiaru (do rozdziału „ograniczenia” pracy) — tylko dla części, które policzono."""
    lines = ["## Ograniczenia", ""]
    if _ran(parts.get("exams")):
        lines += [
            "- Pytania egzaminacyjne mogły trafić do danych treningowych modeli (kontaminacja; część pytań LEK "
            "pochodzi z publicznej bazy) — protokół abstaining_substitution częściowo to kontroluje.",
            "- Przedziały ufności opisują niepewność z powodu skończonej próby pytań, nie zmienność samego modelu "
            "(temperatura 0, ale modele chmurowe nie gwarantują determinizmu).",
        ]
    if _ran(parts.get("esi")) or _ran(parts.get("tools")):
        lines += [
            "- Przypadki ESI pochodzą z danych amerykańskich (Yale EMMLC) opisanych po polsku; mała liczba "
            "przypadków (szerokie przedziały) i celowa nadreprezentacja ESI 1–2 — odsetki nie są reprezentatywne "
            "dla populacji SOR.",
            "- Metryki tekstowe (pokrycie SHAP, wierność liczb, język, eskalacje) są heurystyczne (słowa kluczowe "
            "i rdzenie); rozstrzyga ślepa ocena ekspercka (arkusz ratings.html → `score-ratings`).",
            "- Modele chmurowe Ollamy ignorują parametr `format` (structured outputs), dlatego strukturę wymusza "
            "jedyne narzędzie `submit_explanation` + tolerancyjne parsowanie JSON z treści i jedno ponowienie; "
            "odsetek poprawnych odpowiedzi mierzy więc także posłuszeństwo wobec instrukcji.",
        ]
    lines += [
        "- Modele chmurowe mogą być po cichu aktualizowane pod tym samym tagiem — wersja serwera, digest "
        "(jeśli serwer go podaje) i data pomiaru są zapisane wyżej i w run_info.json; przy powtórzeniu pomiaru "
        "porównaj je przed zestawieniem wyników.",
        "- Ceny pochodzą z cennika API; przy abonamencie Ollama koszt jest orientacyjny.",
        "",
    ]
    return lines


def files_markdown(parts: dict[str, Any]) -> list[str]:
    """Spis plików przebiegu (ścieżki względem katalogu przebiegu)."""
    lines = ["## Pliki przebiegu", "",
             "- `run_info.json` — warunki pomiaru (polecenie, tagi modeli, /api/show, wersja serwera, daty); "
             "`bakeoff.log` — log; `results.json` — wszystkie liczby; `raw_responses.jsonl` — surowe odpowiedzi "
             "tego przebiegu (także z pamięci podręcznej)."]
    if _ran(parts.get("exams")):
        lines.append("- `exam_items.csv` (wylosowane pytania), `exam_responses.csv` (odpowiedź każdego modelu), "
                     "`exam_metrics.csv` (metryki model × protokół × egzamin).")
    if _ran(parts.get("esi")):
        lines.append("- `esi_cases.json` (przypadki, receptura, SHAP), `esi_items.csv`, `esi_summary.csv`, "
                     "`ratings/ratings.html` + `ratings/ratings_template.csv` (dla oceniających), `key.json` "
                     "(klucz liter → modeli — NIE dawać oceniającym).")
    if _ran(parts.get("tools")):
        lines.append("- `tools_items.csv`, `tools_summary.csv`, `tools_trajectories.jsonl` (kroki agenta).")
    lines.append("")
    return lines


def api_error_banner(parts: dict[str, Any]) -> list[str]:
    """Ostrzeżenie na początku raportu, gdy część odpowiedzi to błędy API (liczone jako błędne)."""
    counts: dict[str, int] = {}
    exams = parts.get("exams") if isinstance(parts.get("exams"), dict) else {}
    for row in exams.get("metrics", []) or []:
        if row.get("split") == "all" and row.get("api_errors"):
            counts[row["model"]] = counts.get(row["model"], 0) + int(row["api_errors"])
    for part in ("esi", "tools"):
        summary = (parts.get(part) or {}).get("summary") if isinstance(parts.get(part), dict) else None
        for model, s in (summary or {}).items():
            if isinstance(s, dict) and s.get("api_errors"):
                counts[model] = counts.get(model, 0) + int(s["api_errors"])
    if not counts:
        return []
    detail = ", ".join(f"`{m}`: {n}" for m, n in sorted(counts.items()))
    return [f"> **Uwaga:** część zapytań zakończyła się błędem API po wszystkich ponowieniach ({detail}); takie "
            "odpowiedzi liczone są jako błędne. Uruchom to samo polecenie ponownie — pobierze tylko brakujące "
            "odpowiedzi (pamięć podręczna) — zanim użyjesz wyników w pracy.", ""]


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
    "api_error_banner",
    "DEPLOYABILITY_NOTES",
    "ESI_SCORE_FORMULA",
    "SELECTION_CRITERION",
    "costs_markdown",
    "deployability_note",
    "esi_scores",
    "excluded_banner",
    "exams_markdown",
    "files_markdown",
    "limitations_markdown",
    "meta_markdown",
    "render_report",
    "select_model",
    "selection_markdown",
    "uses_cloud",
]
