"""
Krok 12: porównanie modeli językowych (bake-off) do polskich wyjaśnień decyzji triażu ESI.

Użycie (z katalogu training/; opis: docs/LLM_BAKEOFF.md):
    python scripts/12_llm_bakeoff.py run --dry-run                      # plan + szacunek kosztu
    python scripts/12_llm_bakeoff.py run                                # egzaminy + ESI + narzędzia, 4 modele
    python scripts/12_llm_bakeoff.py run --parts exams --models glm-5.3:cloud,gpt-oss:120b-cloud
    python scripts/12_llm_bakeoff.py run --base-url https://ollama.com  # bezpośrednio (OLLAMA_API_KEY)
    python scripts/12_llm_bakeoff.py run --exam-file pytania.csv        # pytania z pliku zamiast HF
    python scripts/12_llm_bakeoff.py score-ratings --csv ratings_filled.csv   # ślepe oceny ekspertów

Domyślnie: lokalny demon Ollamy (http://localhost:11434, po `ollama signin`) i modele chmurowe
glm-5.3:cloud, gpt-oss:120b-cloud, deepseek-v4.1-flash:cloud, kimi-k3:cloud; think=low;
3 równoległe zapytania (plan Pro). Wynik: runs/llm_bakeoff/<czas>/ — report.md, results.json,
run_info.json, exam_*.csv, raw_responses.jsonl, bakeoff.log (+ pliki części ESI i narzędzi).

Wznawianie: odpowiedzi trafiają na bieżąco do runs/llm_bakeoff/cache/responses.jsonl; ponowne
uruchomienie tego samego polecenia pobiera tylko brakujące (i nieudane) odpowiedzi. --fresh
ignoruje pamięć podręczną.

Kody wyjścia: 0 = OK, 1 = któraś część nieudana, 2 = złe argumenty / brak klucza / brak danych,
130 = Ctrl+C.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import os
import shlex
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger  # noqa: E402

log = get_logger("12_llm_bakeoff")

DEFAULT_MODELS: tuple[str, ...] = ("glm-5.3:cloud", "gpt-oss:120b-cloud", "deepseek-v4.1-flash:cloud", "kimi-k3:cloud")
ALL_PARTS: tuple[str, ...] = ("exams", "esi", "tools")
# Części dostarczane przez osobne moduły (import leniwy; brak modułu = część pominięta)
PART_ENTRY: dict[str, tuple[str, str]] = {
    "esi": ("src.llm_eval.explain_task", "run_esi_part"),
    "tools": ("src.llm_eval.tools_task", "run_tools_part"),
}
RATINGS_ENTRY: tuple[tuple[str, str], ...] = (("src.llm_eval.ratings", "score_ratings"),
                                              ("src.llm_eval.explain_task", "score_ratings"))
SECONDS_PER_CALL: dict[str, float] = {"exams": 6.0, "esi": 25.0, "tools": 10.0}   # tylko do --dry-run
PING_MESSAGES = [{"role": "user", "content": "Test łączności. Odpowiedz jednym słowem: OK"}]


# ─────────────────────────────────────────
# Argumenty
# ─────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Porównanie modeli LLM (egzaminy NASK, wyjaśnienia ESI, narzędzia)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Uruchom porównanie i zapisz report.md")
    run.add_argument("--parts", default=",".join(ALL_PARTS), help="Części po przecinku: exams,esi,tools")
    run.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Tagi modeli Ollamy po przecinku")
    run.add_argument("--think", default="low", help="Rozumowanie: low | medium | high | max | true | false | default")
    run.add_argument("--n-exam", type=int, default=300, help="Liczba pytań LEK (domyślnie 300)")
    run.add_argument("--n-pes", type=int, default=200, help="Maks. liczba pytań PES (domyślnie 200)")
    run.add_argument("--n-cases", type=int, default=40, help="Liczba przypadków ESI (części esi i tools)")
    run.add_argument("--exam-file", action="append", default=None,
                     help="Plik pytań CSV/parquet w formacie NASK (można kilka; prefiks r_lek= / r_pes_latest=)")
    run.add_argument("--concurrency", type=int, default=3, help="Równoległe zapytania (plan Pro: 3)")
    run.add_argument("--base-url", default=None,
                     help="Adres Ollamy (domyślnie http://localhost:11434; https://ollama.com = bezpośrednio)")
    run.add_argument("--api-key-env", default="OLLAMA_API_KEY", help="Zmienna środowiskowa z kluczem API")
    run.add_argument("--recipe", default=None, help="recipe.json z runs/final_report__*/ (części esi/tools)")
    run.add_argument("--out", default=None, help="Katalog wyników (domyślnie runs/llm_bakeoff/<czas>)")
    run.add_argument("--cache", default=None, help="Plik pamięci podręcznej (domyślnie runs/llm_bakeoff/cache/responses.jsonl)")
    run.add_argument("--fresh", action="store_true", help="Ignoruj pamięć podręczną (wszystko od nowa)")
    run.add_argument("--dry-run", action="store_true", help="Tylko plan i szacunek kosztu (bez zapytań do modeli)")
    run.add_argument("--prices", default=None, help="JSON z cenami {model: {input, output}} (USD / 1M tokenów)")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--timeout", type=float, default=180.0, help="Limit czasu jednego zapytania [s]")
    run.add_argument("--max-retries", type=int, default=5, help="Ponowienia przy 429/5xx/timeout")
    run.add_argument("--temperature", type=float, default=0.0, help="Temperatura w części egzaminacyjnej")
    run.add_argument("--n-boot", type=int, default=2000, help="Losowania bootstrapu (porównania parami)")
    run.add_argument("--no-preflight", action="store_true", help="Bez testu łączności z każdym modelem")
    run.add_argument("--decimal-comma", action="store_true", help="Przecinek dziesiętny w report.md")

    score = sub.add_parser("score-ratings", help="Wczytaj wypełnione ślepe oceny (ratings) i policz średnie")
    score.add_argument("--csv", required=True, action="append",
                       help="Wypełniony arkusz ocen (CSV); kilku oceniających — podaj --csv kilka razy")
    score.add_argument("--key", default=None, help="key.json z przebiegu (domyślnie szukany obok arkuszy)")
    score.add_argument("--out", default=None, help="Katalog wyniku (domyślnie katalog CSV)")
    score.add_argument("--seed", type=int, default=42)
    return parser


def _split_list(text: str) -> list[str]:
    seen: list[str] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if item and item not in seen:
            seen.append(item)
    return seen


# ─────────────────────────────────────────
# Log przebiegu (wszystkie loggery pakietu → bakeoff.log)
# ─────────────────────────────────────────
def _run_loggers() -> list[logging.Logger]:
    names = [n for n in logging.Logger.manager.loggerDict
             if n.startswith("src.llm_eval") or n == "12_llm_bakeoff"]
    return [logging.getLogger(n) for n in names]


def _attach_log(path: Path) -> None:
    for lg in _run_loggers():
        get_logger(lg.name, log_file=path)


def _detach_log(path: Path) -> None:
    target = str(path.resolve())
    for lg in _run_loggers():
        for h in list(lg.handlers):
            if isinstance(h, logging.FileHandler) and str(Path(h.baseFilename).resolve()) == target:
                lg.removeHandler(h)
                h.close()


# ─────────────────────────────────────────
# Części zewnętrzne (esi, tools) — import leniwy, wywołanie z pasującymi argumentami
# ─────────────────────────────────────────
def load_entry(module: str, attr: str) -> tuple[Any, str | None]:
    """(funkcja, None) albo (None, powód) — brak modułu nie jest błędem (część pominięta)."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == module:
            return None, f"moduł {module} nie istnieje"
        raise
    fn = getattr(mod, attr, None)
    if fn is None:
        return None, f"moduł {module} nie ma funkcji {attr}"
    return fn, None


def call_with_context(fn: Any, ctx: dict[str, Any]) -> Any:
    """Wywołuje fn z tymi argumentami z ctx, które przyjmuje (albo wszystkimi, gdy ma **kwargs)."""
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**ctx)
    return fn(**{k: v for k, v in ctx.items() if k in params})


def unpack_part(ret: Any) -> tuple[dict[str, Any], str]:
    """Wynik części → (słownik wyników, markdown). Obsługuje (dict, str) i dict z kluczem markdown/sections."""
    md: Any = ""
    if isinstance(ret, tuple) and len(ret) == 2:
        res, md = ret
    elif isinstance(ret, dict):
        res = ret
        md = ret.get("markdown") or ret.get("sections") or ""
    elif ret is None:
        res = {"skipped": "część nie zwróciła wyników"}
    else:
        res = {"result": ret}
    if isinstance(md, dict):
        md = "\n\n".join(f"### {k}\n\n{v}" if not str(v).lstrip().startswith("#") else str(v) for k, v in md.items())
    elif isinstance(md, (list, tuple)):
        md = "\n\n".join(str(x) for x in md)
    res = res if isinstance(res, dict) else {"result": res}
    return res, str(md or "")


def _usage_diff(before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for model, u in after.items():
        b = before.get(model, {})
        out[model] = {k: u.get(k, 0) - b.get(k, 0) for k in u}
    return out


def part_costs(part: str, res: dict[str, Any], usage: dict[str, dict[str, float]],
               prices: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Zużycie części: `usage` z wyników części (gdy podała) albo licznik klienta z tego przebiegu."""
    from src.llm_eval.pricing import cost_usd

    if isinstance(res.get("usage"), dict):
        source = res["usage"]
    elif isinstance(res.get("summary"), dict) and any(isinstance(v, dict) and "prompt_tokens" in v
                                                      for v in res["summary"].values()):
        source = res["summary"]          # część ESI/narzędzi: tokeny wszystkich pozycji (także z cache)
    else:
        source = usage
    rows = []
    for model, u in (source or {}).items():
        if not isinstance(u, dict):
            continue
        calls = u.get("calls", u.get("n_items", 0)) or 0
        pt, ct = float(u.get("prompt_tokens") or 0), float(u.get("completion_tokens") or 0)
        if not calls and not pt:
            continue
        row = {"part": part, "model": model, "calls": calls, "prompt_tokens": pt,
               "completion_tokens": ct, "cost_usd": cost_usd(model, pt, ct, prices)}
        if u.get("latency_s_median") is not None:     # mediana czasu na pozycję (przypadek) z wyników części
            row["latency_median_s"] = float(u["latency_s_median"])
        rows.append(row)
    return rows


def exam_costs(exams: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in exams.get("models", []):
        ms = [m for m in exams.get("metrics", []) if m["model"] == model and m["split"] == "all"]
        costs = [m.get("cost_usd") for m in ms]
        lat = [m["latency_median_s"] for m in ms if m.get("latency_median_s") is not None]
        rows.append({"part": "exams", "model": model, "calls": sum(m["n"] for m in ms),
                     "prompt_tokens": sum(m["prompt_tokens_total"] for m in ms),
                     "completion_tokens": sum(m["completion_tokens_total"] for m in ms),
                     "cost_usd": None if any(c is None for c in costs) else sum(costs),
                     "latency_median_s": sum(lat) / len(lat) if lat else None})
    return rows


def _public(obj: Any) -> Any:
    """Bez kluczy prywatnych ('_…', np. wektory trafień) — do results.json."""
    if isinstance(obj, dict):
        return {k: _public(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_public(v) for v in obj]
    return obj


# ─────────────────────────────────────────
# run
# ─────────────────────────────────────────
def _dry_run(args: argparse.Namespace, parts: list[str], models: list[str], client: Any, cache: Any,
             prices: dict[str, dict[str, float]], think: Any) -> int:
    from src.llm_eval.exams import DEFAULT_OPTIONS, estimate_exams, load_exam_sources, sample_items
    from src.llm_eval.pricing import PRICING_SOURCE, estimate_part, format_usd

    print("PLAN (--dry-run, bez zapytań do modeli)")
    info = client.server_info()
    print(f"  Serwer: {info['base_url']} ({'chmura' if info['mode'] == 'cloud' else 'lokalny demon'}), "
          f"wersja: {info['version'] or 'brak odpowiedzi'}" + (f" — {info['error']}" if info.get("error") else ""))
    print(f"  Modele: {', '.join(models)}; części: {', '.join(parts)}; think={think}; "
          f"równoległość {args.concurrency}; pamięć podręczna: {cache.path} ({len(cache)} wpisów)")
    rows: list[dict[str, Any]] = []
    if "exams" in parts:
        items = None
        try:
            items, sinfo = sample_items(load_exam_sources(args.exam_file), n_lek=args.n_exam, n_pes=args.n_pes,
                                        seed=args.seed)
            pes_txt = "specjalizacje SOR / interna" if sinfo["pes_mode"] == "specialty" else "losowe"
            print(f"  Egzaminy: {sinfo['n']['lek']} LEK + {sinfo['n']['pes']} PES (PES: {pes_txt}) × 2 protokoły")
        except Exception as exc:     # noqa: BLE001 — w planie tylko ostrzeżenie
            print(f"  Egzaminy: nie udało się wczytać pytań ({exc}) — szacunek dla N={args.n_exam + args.n_pes}")
        opts = {**DEFAULT_OPTIONS, "temperature": args.temperature, "seed": args.seed}
        rows += estimate_exams(items, models, cache, think, prices, args.n_exam + args.n_pes, options=opts)
    cases_ok, cases_detail = True, ""
    if "esi" in parts or "tools" in parts:
        from src.llm_eval.esi_cases import cases_status

        cases_ok, cases_detail = cases_status(args.recipe)
    for part in ("esi", "tools"):
        if part in parts:
            fn, why = load_entry(*PART_ENTRY[part])
            if fn is None or not cases_ok:
                print(f"  Część {part}: zostanie POMINIĘTA — {why or cases_detail}")
                continue
            print(f"  Część {part}: {args.n_cases} przypadków ({cases_detail}); szacunek wg założeń, "
                  f"bez pamięci podręcznej")
            for model in models:
                est = estimate_part(part, model, args.n_cases, prices)
                rows.append({**est, "uncached_calls": est["calls"], "uncached_cost_usd": est["cost_usd"]})
    print("\n  Część   Model                          Wywołania  (bez cache)  Tokeny wej.  Tokeny wyj.  Koszt   (do zapłaty)")
    total = total_un = 0.0
    for r in rows:
        total += r["cost_usd"] or 0.0
        total_un += r["uncached_cost_usd"] or 0.0
        print(f"  {r['part']:<7} {r['model']:<30} {int(r['calls']):>9}  {int(r['uncached_calls']):>11}  "
              f"{int(r['prompt_tokens']):>11}  {int(r['completion_tokens']):>11}  {format_usd(r['cost_usd']):>7}  "
              f"{format_usd(r['uncached_cost_usd']):>7}")
    print(f"\n  Razem: {format_usd(total)} (bez pozycji z pamięci podręcznej: {format_usd(total_un)}); "
          f"ceny: {PRICING_SOURCE}. Przy abonamencie Ollama to koszt orientacyjny (limit planu, nie faktura).")
    seconds = sum(r["uncached_calls"] * SECONDS_PER_CALL.get(r["part"], 10.0) for r in rows) / max(1, args.concurrency)
    print(f"  Szacowany czas: ≈ {seconds / 3600:.1f} h (założenie: "
          + ", ".join(f"{p} ~{s:.0f} s/zapytanie" for p, s in SECONDS_PER_CALL.items())
          + f", {args.concurrency} naraz; rzeczywisty czas zależy od obciążenia chmury)")
    return 0


def _preflight(client: Any, models: list[str], think: Any) -> dict[str, str]:
    """Jedno krótkie zapytanie do każdego modelu; zwraca {model: powód} modeli niedziałających."""
    failed = {}
    for model in models:
        res = client.chat(model, PING_MESSAGES, think=think)
        if res.error:
            failed[model] = res.error
            log.error(f"Test łączności: {model} — {res.error}")
        else:
            log.info(f"Test łączności: {model} OK ({res.latency_s:.1f} s)")
    client.reset_breaker()
    return failed


def sent_think(client: Any, models: list[str], think: Any) -> dict[str, Any]:
    """{model: think faktycznie wysyłane} — po odrzuceniu `think` przez serwer (400) model idzie bez niego (None)."""
    return {m: client.effective_think(m, think) for m in models}


def start_fresh(path: Path) -> None:
    """Plik TEGO przebiegu od zera (ten sam --out → bez duplikatów); zablokowany (Windows) → tylko ostrzeżenie."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning(f"Nie udało się wyczyścić {path.name} ({exc}) — nowe rekordy zostaną dopisane na końcu")


def cmd_run(args: argparse.Namespace) -> int:
    from src.llm_eval.cache import ResponseCache
    from src.llm_eval.client import DEFAULT_BASE_URL, OllamaChat, is_cloud_url, parse_think
    from src.llm_eval.pricing import load_prices
    from src.utils import config

    parts = _split_list(args.parts)
    unknown = [p for p in parts if p not in ALL_PARTS]
    models = _split_list(args.models)
    if unknown or not parts:
        log.error(f"Nieznane części {unknown or parts} (dostępne: {', '.join(ALL_PARTS)})")
        return 2
    if not models:
        log.error("Pusta lista modeli (--models)")
        return 2
    try:
        think = parse_think(args.think)
        prices = load_prices(args.prices)
    except (ValueError, OSError) as exc:
        log.error(str(exc))
        return 2

    base_url = args.base_url or DEFAULT_BASE_URL
    api_key = os.environ.get(args.api_key_env) or None
    if is_cloud_url(base_url) and not api_key:
        log.error(f"Tryb bezpośredni {base_url} wymaga klucza API: ustaw zmienną {args.api_key_env} "
                  f"(https://ollama.com/settings/keys) albo użyj lokalnego demona (`ollama signin`, bez --base-url).")
        return 2

    root = Path(config.RUNS_DIR) / "llm_bakeoff"
    cache = ResponseCache(Path(args.cache) if args.cache else root / "cache" / "responses.jsonl", fresh=args.fresh)
    client = OllamaChat(base_url=base_url, api_key=api_key, timeout_s=args.timeout, max_retries=args.max_retries,
                        concurrency=args.concurrency)
    if args.dry_run:
        return _dry_run(args, parts, models, client, cache, prices, think)

    started = datetime.now()
    out_dir = Path(args.out) if args.out else root / started.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "bakeoff.log"
    _attach_log(log_path)
    try:
        return _run(args, parts, models, think, prices, client, cache, out_dir, started, log_path)
    except KeyboardInterrupt:
        log.warning("Przerwano (Ctrl+C) — uruchom to samo polecenie ponownie: gotowe odpowiedzi są w pamięci "
                    f"podręcznej {cache.path}")
        return 130
    finally:
        _detach_log(log_path)


def _run(args: argparse.Namespace, parts: list[str], models: list[str], think: Any, prices: dict[str, Any],
         client: Any, cache: Any, out_dir: Path, started: datetime, log_path: Path) -> int:
    from src.evaluation.report_sections import jsonable
    from src.llm_eval.cache import JsonlWriter
    from src.llm_eval.exams import DEFAULT_OPTIONS, run_exams_part
    from src.llm_eval.report import render_report, select_model
    from src.utils.fileio import write_json_atomic, write_text_atomic

    _attach_log(log_path)          # loggery modułów zaimportowanych przed chwilą też do bakeoff.log
    command = "python scripts/12_llm_bakeoff.py " + " ".join(shlex.quote(a) for a in sys.argv[1:]) \
        if Path(sys.argv[0]).name == "12_llm_bakeoff.py" else "run (wywołanie z kodu)"
    server = client.server_info()
    log.info(f"Ollama: {server['base_url']} ({server['mode']}), wersja {server['version'] or '?'}"
             + (f" — {server['error']}" if server.get("error") else ""))
    model_info = {}
    for model in models:
        info = client.model_info(model)
        model_info[model] = info
        d = info.get("details", {}) or {}
        log.info(f"Model {model} (API: {info['api_model']}): digest={(info.get('digest') or '—')[:12]}, "
                 f"rodzina={d.get('family', '—')}, parametry={d.get('parameter_size', '—')}, "
                 f"kwantyzacja={d.get('quantization_level', '—')}, zdalny={info.get('remote_model', '—')}"
                 + (f" — {info['error']}" if info.get("error") else ""))
    think_effective = {}
    for model, info in model_info.items():
        thinking = info.get("thinking") if isinstance(info.get("thinking"), dict) else {}
        client.set_think_values(model, thinking.get("values"))
        think_effective[model] = client.effective_think(model, think)
        if think_effective[model] != think:
            log.warning(f"Model {model}: think={think!r} nieobsługiwane (wartości modelu: {thinking.get('values')}) "
                        f"— wysyłane think={think_effective[model]!r}")
    meta: dict[str, Any] = {
        "started_at": started.isoformat(timespec="seconds"), "command": command, "server": server,
        "model_info": model_info, "models": models, "parts": parts, "think": think,
        "think_effective": think_effective, "seed": args.seed,
        "concurrency": args.concurrency, "n_exam": args.n_exam, "n_pes": args.n_pes, "n_cases": args.n_cases,
        "cache": str(cache.path), "fresh": bool(args.fresh), "out_dir": str(out_dir),
    }
    write_json_atomic(out_dir / "run_info.json", jsonable(meta))

    excluded: dict[str, str] = {}
    if not args.no_preflight:
        excluded = _preflight(client, models, think)
        models = [m for m in models if m not in excluded]
        meta["excluded_models"] = excluded
        meta["think_effective"] = sent_think(client, meta["models"], think)   # test mógł trafić na 400 „think”
        write_json_atomic(out_dir / "run_info.json", jsonable(meta))
        if not models:
            log.error("Żaden model nie odpowiada — sprawdź demon Ollamy (`ollama serve`, `ollama signin`) "
                      "albo klucz API; nic nie zostało policzone")
            return 1

    start_fresh(out_dir / "raw_responses.jsonl")      # odpowiedzi z pamięci podręcznej i tak są zapisywane ponownie
    raw_log = JsonlWriter(out_dir / "raw_responses.jsonl")
    results: dict[str, Any] = {}
    part_md: dict[str, str] = {}
    costs: list[dict[str, Any]] = []
    failed = False

    if "exams" in parts:
        try:
            opts = {**DEFAULT_OPTIONS, "temperature": args.temperature, "seed": args.seed}
            exams = run_exams_part(client, models, out_dir=out_dir, cache=cache, think=think, n_lek=args.n_exam,
                                   n_pes=args.n_pes, seed=args.seed, exam_files=args.exam_file, prices=prices,
                                   raw_log=raw_log, options=opts, n_boot=args.n_boot)
            results["exams"] = exams
            costs += exam_costs(exams)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            log.error(f"Część exams: {exc}")
            results["exams"] = {"failed": str(exc)}
            failed = True
        except Exception as exc:     # noqa: BLE001 — pozostałe części mają szansę się wykonać
            log.error(f"Część exams nieudana: {exc!r}\n{traceback.format_exc()}")
            results["exams"] = {"failed": repr(exc)}
            failed = True

    ctx = {"client": client, "models": models, "out_dir": out_dir, "cache": cache, "think": think,
           "n_cases": args.n_cases, "recipe": args.recipe, "seed": args.seed, "prices": prices, "raw_log": raw_log,
           "log": log, "fresh": bool(args.fresh), "concurrency": args.concurrency, "dry_run": False,
           "decimal_comma": bool(args.decimal_comma)}
    for part in ("esi", "tools"):
        if part not in parts:
            continue
        try:
            fn, why = load_entry(*PART_ENTRY[part])
            if fn is None:
                log.warning(f"Część {part} pominięta: {why}")
                results[part] = {"skipped": why}
                continue
            _attach_log(log_path)
            before = {m: dict(u) for m, u in client.usage.items()}
            res, md = unpack_part(call_with_context(fn, ctx))
            results[part] = res
            if md:
                part_md[part] = md
            if res.get("skipped"):
                log.warning(f"Część {part} pominięta: {res['skipped']}")
            else:
                costs += part_costs(part, res, _usage_diff(before, client.usage), prices)
        except Exception as exc:     # noqa: BLE001
            log.error(f"Część {part} nieudana: {exc!r}\n{traceback.format_exc()}")
            results[part] = {"failed": repr(exc)}
            failed = True

    exams_res = results.get("exams") if isinstance(results.get("exams"), dict) and results["exams"].get("metrics") else None
    selection = select_model(exams_res, results.get("esi"), n_boot=args.n_boot, seed=args.seed)
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["think_effective"] = sent_think(client, meta["models"], think)   # odrzucenia `think` w trakcie przebiegu
    meta["cache_stats"] = cache.stats()
    meta["usage_this_run"] = client.usage
    report = render_report(meta, results, part_md, selection, costs, decimal_comma=args.decimal_comma)
    write_text_atomic(out_dir / "report.md", report)
    write_json_atomic(out_dir / "results.json", jsonable(_public({"meta": meta, "parts": results,
                                                                  "selection": selection, "costs": costs})))
    write_json_atomic(out_dir / "run_info.json", jsonable(meta))
    if selection.get("winner"):
        log.info(f"Wybrany model: {selection['winner']} ({selection.get('decided_by')})")
    log.info(f"Raport: {out_dir / 'report.md'}  (pamięć podręczna: {cache.stats()})")
    return 1 if failed else 0


# ─────────────────────────────────────────
# score-ratings
# ─────────────────────────────────────────
def cmd_score_ratings(args: argparse.Namespace) -> int:
    csv_paths = [Path(p) for p in args.csv]
    missing = [str(p) for p in csv_paths if not p.exists()]
    if missing:
        log.error(f"Brak pliku ocen: {', '.join(missing)}")
        return 2
    key = Path(args.key) if args.key else None
    out = Path(args.out) if args.out else csv_paths[0].parent
    fn, why, module = None, None, None
    for module, attr in RATINGS_ENTRY:
        fn, why = load_entry(module, attr)
        if fn is not None:
            break
    if fn is None:
        log.error(f"Ocena arkuszy niedostępna: {why}")
        return 2
    ctx = {"csv_paths": csv_paths, "csv": csv_paths[0], "csv_path": csv_paths[0], "key": key, "key_path": key,
           "out": out, "out_dir": out, "seed": args.seed, "log": log}
    try:
        ret = call_with_context(fn, ctx)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        log.error(f"Nie udało się wczytać ocen: {exc}")
        return 2
    res, md = unpack_part(ret)
    writer = getattr(importlib.import_module(module), "write_scores", None)
    if writer is not None and isinstance(ret, dict):
        files = call_with_context(writer, {"result": ret, "out_dir": out})
        log.info(f"Wyniki ocen zapisane: {files}")
    print(md or (ret if isinstance(ret, str) else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        return cmd_score_ratings(args)
    except KeyboardInterrupt:
        print("Przerwano (Ctrl+C) — uruchom to samo polecenie, gotowe odpowiedzi zostaną pominięte.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
