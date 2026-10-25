"""
Ślepa ocena wyjaśnień ESI przez ekspertów (pielęgniarki / lekarzy SOR).

    write_rating_sheet(out_dir, ctx, items, models)  → ratings/ratings.html, ratings/ratings_template.csv,
                                                       key.json (klucz OSOBNO — nie dawać oceniającym)
    score_ratings(csv_paths, key_path)               → średnie ocen na model z bootstrapowymi 95% CI

W każdym przypadku modele są losowo przetasowane i oznaczone literami A, B, C, D… (inna kolejność
w każdym przypadku, ziarno zapisane w key.json). Oceniający widzi te same źródła [S1]–[S4], co
modele, ale NIE widzi ESI z triażu ani nazw modeli.

Skale 1–5: zgodność kliniczna, przydatność dla triażysty, jasność; pole „halucynacja” (0/1) — czy
wyjaśnienie zawiera informacje sprzeczne z danymi albo zmyślone. Arkusz HTML działa bez internetu:
oceny zapisują się w przeglądarce (localStorage) i eksportują do CSV w formacie szablonu.
"""

from __future__ import annotations

import html
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.llm_eval.esi_cases import EsiContext, fmt_num
from src.utils.fileio import write_json_atomic, write_text_atomic
from src.utils.logger import get_logger

log = get_logger(__name__)

SCALES: dict[str, str] = {
    "zgodnosc_kliniczna": "Zgodność kliniczna",
    "przydatnosc": "Przydatność dla triażysty",
    "jasnosc": "Jasność",
}
HALLUCINATION_COL = "halucynacja"
TEMPLATE_COLUMNS: list[str] = ["oceniajacy", "case_id", "litera", *SCALES, HALLUCINATION_COL, "komentarz"]
RATINGS_DIR = "ratings"
KEY_FILE = "key.json"
_TRUE = {"1", "tak", "true", "t", "x", "yes", "y"}
_FALSE = {"0", "nie", "false", "f", "no", "n", ""}


def letters(n: int) -> list[str]:
    """A, B, C, … (n ≤ 26)."""
    if not 1 <= n <= 26:
        raise ValueError(f"Arkusz obsługuje 1–26 modeli, podano {n}")
    return [chr(ord("A") + i) for i in range(n)]


def assign_letters(case_ids: Sequence[str], models: Sequence[str], seed: int = 42) -> dict[str, dict[str, str]]:
    """{case_id: {litera: model}} — osobna losowa permutacja modeli w każdym przypadku."""
    rng = np.random.default_rng(seed)
    out = {}
    for cid in case_ids:
        order = [models[i] for i in rng.permutation(len(models))]
        out[cid] = dict(zip(letters(len(models)), order))
    return out


# ─────────────────────────────────────────
# Arkusz
# ─────────────────────────────────────────
_CSS = """
:root { --bg: #f7f7f5; --card: #ffffff; --text: #1d1d1b; --muted: #5f5f5a; --line: #d9d9d4; --accent: #1f5fa8;
        --src: #f0f2f5; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161615; --card: #20201f; --text: #ececea; --muted: #a5a5a0; --line: #3a3a37; --accent: #7fb0ea;
          --src: #2a2a28; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, -apple-system, "Segoe UI",
       Roboto, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.4rem; margin: 8px 0 4px; } h2 { font-size: 1.15rem; margin: 0 0 8px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 16px 0; }
.src { background: var(--src); border-radius: 8px; padding: 10px 12px; white-space: pre-wrap; margin: 6px 0;
       font-size: .92rem; }
.src b { color: var(--accent); }
.expl { border-top: 1px solid var(--line); padding-top: 12px; margin-top: 12px; }
.expl h3 { margin: 0 0 6px; font-size: 1.05rem; }
.text { white-space: pre-wrap; }
.muted { color: var(--muted); font-size: .9rem; }
.row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 8px; }
label { font-size: .9rem; } select, input[type=text], textarea { font: inherit; color: var(--text);
  background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 4px 6px; }
textarea { width: 100%; min-height: 2.4em; }
.bar { position: sticky; top: 0; background: var(--bg); padding: 8px 0; border-bottom: 1px solid var(--line);
       z-index: 1; }
button { font: inherit; padding: 6px 14px; border-radius: 6px; border: 1px solid var(--accent);
         background: var(--accent); color: var(--card); cursor: pointer; }
"""

_JS = """
(function () {
  var KEY = 'sorai-ratings-' + document.body.dataset.sheet;
  var COLS = %COLS%;
  function fields() { return Array.prototype.slice.call(document.querySelectorAll('[data-f]')); }
  function save() {
    var data = {rater: document.getElementById('rater').value};
    fields().forEach(function (el) { data[el.id] = el.type === 'checkbox' ? el.checked : el.value; });
    try { localStorage.setItem(KEY, JSON.stringify(data)); } catch (e) {}
  }
  function load() {
    var data = null;
    try { data = JSON.parse(localStorage.getItem(KEY) || 'null'); } catch (e) { data = null; }
    if (!data) return;
    document.getElementById('rater').value = data.rater || '';
    fields().forEach(function (el) {
      if (!(el.id in data)) return;
      if (el.type === 'checkbox') el.checked = !!data[el.id]; else el.value = data[el.id];
    });
  }
  function q(v) { return '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"'; }
  function exportCsv() {
    var rater = document.getElementById('rater').value;
    var lines = [COLS.join(',')];
    document.querySelectorAll('[data-item]').forEach(function (box) {
      var id = box.dataset.item;
      var get = function (s) { var el = document.getElementById(id + '-' + s); return el ? el : null; };
      var row = [rater, box.dataset.case, box.dataset.letter,
                 get('zgodnosc_kliniczna').value, get('przydatnosc').value, get('jasnosc').value,
                 get('halucynacja').checked ? '1' : '0', get('komentarz').value];
      lines.push(row.map(q).join(','));
    });
    var blob = new Blob(['\\ufeff' + lines.join('\\r\\n') + '\\r\\n'], {type: 'text/csv;charset=utf-8'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'oceny_' + (rater || 'anonim').replace(/[^a-zA-Z0-9_-]+/g, '_') + '.csv';
    document.body.appendChild(a); a.click(); a.remove();
  }
  document.addEventListener('change', save);
  document.addEventListener('input', save);
  document.getElementById('export').addEventListener('click', exportCsv);
  load();
})();
"""


def _esc(text: Any) -> str:
    return html.escape(str(text or ""), quote=True)


def _scale_select(item_id: str, scale: str, label: str) -> str:
    options = "".join(f'<option value="{v}">{v}</option>' for v in range(1, 6))
    return (f'<label>{_esc(label)} <select id="{item_id}-{scale}" data-f>'
            f'<option value="">–</option>{options}</select></label>')


def _explanation_html(expl: dict[str, Any] | None) -> str:
    if not expl:
        return '<p class="muted">Model nie zwrócił poprawnego wyjaśnienia.</p>'
    esc = expl.get("sugestia_eskalacji") or {}
    factors = "".join(f"<li>{_esc(f)}</li>" for f in expl.get("kluczowe_czynniki") or [])
    if esc.get("czy"):
        esc_text = (f"TAK — sugerowany poziom: ESI {_esc(esc.get('poziom'))}"
                    + (f"; powód: {_esc(esc.get('powod'))}" if esc.get("powod") else ""))
    else:
        esc_text = "nie"
    return (f'<p><b>Poziom ESI w odpowiedzi:</b> {_esc(expl.get("poziom_esi"))}</p>'
            f'<div class="text">{_esc(expl.get("uzasadnienie"))}</div>'
            + (f"<p><b>Kluczowe czynniki:</b></p><ul>{factors}</ul>" if factors else "")
            + f"<p><b>Sugestia wyższego priorytetu:</b> {esc_text}</p>")


def render_sheet(ctx: EsiContext, outputs: dict[str, dict[str, dict[str, Any] | None]],
                 key: dict[str, dict[str, str]], sheet_id: str, title: str = "Ocena wyjaśnień triażu ESI") -> str:
    """HTML arkusza: przypadki ze źródłami [S1]–[S4] i zanonimizowane wyjaśnienia (litery)."""
    titles = {"S1": "Dane pacjenta", "S2": "Decyzja systemu", "S3": "Czynniki decyzji (SHAP)",
              "S4": "Reguły bezpieczeństwa ESI"}
    blocks = []
    for case in ctx.cases:
        sources = "".join(f'<div class="src"><b>[{sid}] {titles[sid]}</b>\n{_esc(text)}</div>'
                          for sid, text in case.sources(ctx.recipe_text).items())
        expls = []
        for letter, model in key[case.case_id].items():
            item_id = f"{case.case_id}-{letter}"
            expl = outputs.get(model, {}).get(case.case_id)
            selects = "".join(_scale_select(item_id, s, lbl) for s, lbl in SCALES.items())
            expls.append(
                f'<div class="expl" data-item="{item_id}" data-case="{case.case_id}" data-letter="{letter}">'
                f"<h3>Wyjaśnienie {letter}</h3>{_explanation_html(expl)}"
                f'<div class="row">{selects}<label><input type="checkbox" id="{item_id}-halucynacja" data-f> '
                f"halucynacja / sprzeczność z danymi</label></div>"
                f'<textarea id="{item_id}-komentarz" data-f placeholder="Komentarz (opcjonalnie)"></textarea></div>')
        blocks.append(f'<section class="card"><h2>Przypadek {case.case_id}</h2>{sources}{"".join(expls)}</section>')
    js = _JS.replace("%COLS%", json.dumps(TEMPLATE_COLUMNS))
    intro = (
        "<p>Każdy przypadek zawiera dane, które otrzymały modele językowe, i kilka wyjaśnień oznaczonych "
        "literami (kolejność losowa w każdym przypadku; nazwy modeli są ukryte). Oceń każde wyjaśnienie "
        "w skali 1–5 (1 = bardzo źle, 5 = bardzo dobrze):</p><ul>"
        "<li><b>Zgodność kliniczna</b> — czy wyjaśnienie jest merytorycznie poprawne i zgodne z danymi;</li>"
        "<li><b>Przydatność</b> — czy pomogłoby triażyście zrozumieć i zweryfikować decyzję;</li>"
        "<li><b>Jasność</b> — czy jest zwięzłe, zrozumiałe i poprawne językowo.</li></ul>"
        "<p>Zaznacz <b>halucynację</b>, jeśli wyjaśnienie podaje informacje sprzeczne z danymi albo zmyślone "
        "(np. nieistniejący pomiar). Oceny zapisują się w tej przeglądarce; na końcu kliknij "
        "„Pobierz CSV” i odeślij plik.</p>")
    return (f'<!doctype html><html lang="pl"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{_esc(title)}</title><style>{_CSS}</style></head>"
            f'<body data-sheet="{_esc(sheet_id)}"><main><h1>{_esc(title)}</h1>{intro}'
            f'<div class="bar row"><label>Oceniający (inicjały): <input type="text" id="rater"></label>'
            f'<button type="button" id="export">Pobierz CSV</button></div>'
            f'{"".join(blocks)}</main><script>{js}</script></body></html>')


def write_rating_sheet(out_dir: Path | str, ctx: EsiContext, items: list[dict[str, Any]],
                       models: Sequence[str], seed: int = 42) -> dict[str, str]:
    """
    Zapisuje ratings/ratings.html, ratings/ratings_template.csv (dla oceniających) i key.json
    (klucz liter → modeli, POZA katalogiem ratings/). Zwraca ścieżki.
    """
    out = Path(out_dir)
    outputs: dict[str, dict[str, dict[str, Any] | None]] = {m: {} for m in models}
    for it in items:
        outputs.setdefault(it["model"], {})[it["case_id"]] = it.get("explanation")
    key = assign_letters([c.case_id for c in ctx.cases], list(models), seed=seed)
    sheet_id = f"{ctx.recipe.data_fingerprint or 'esi'}-{seed}-{len(ctx.cases)}-{len(models)}"
    sheet_dir = out / RATINGS_DIR
    html_path = write_text_atomic(sheet_dir / "ratings.html", render_sheet(ctx, outputs, key, sheet_id))
    rows = [{"oceniajacy": "", "case_id": cid, "litera": letter, **{s: "" for s in SCALES},
             HALLUCINATION_COL: "", "komentarz": ""}
            for cid, mapping in key.items() for letter in mapping]
    template = sheet_dir / "ratings_template.csv"
    write_text_atomic(template, pd.DataFrame(rows, columns=TEMPLATE_COLUMNS).to_csv(index=False))
    key_path = write_json_atomic(out / KEY_FILE, {"seed": seed, "models": list(models), "sheet_id": sheet_id,
                                                  "cases": key})
    return {"ratings_html": str(html_path), "ratings_template": str(template), "ratings_key": str(key_path)}


# ─────────────────────────────────────────
# Wyniki ocen
# ─────────────────────────────────────────
def _find_key(csv_paths: Sequence[Path], key_path: Path | str | None) -> Path:
    if key_path is not None:
        return Path(key_path)
    for p in csv_paths:
        for cand in (p.parent / KEY_FILE, p.parent.parent / KEY_FILE):
            if cand.exists():
                return cand
    raise FileNotFoundError("Nie znaleziono key.json — podaj ścieżkę klucza (--key)")


def _score(value: Any) -> float:
    try:
        v = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return float("nan")
    return v if v in (1.0, 2.0, 3.0, 4.0, 5.0) else float("nan")


def _flag(value: Any) -> float:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip().lower()
    if text in _TRUE:
        return 1.0
    if text in _FALSE:
        return 0.0
    return float("nan")


def read_sheet_csv(path: Path | str) -> pd.DataFrame:
    """
    Arkusz ocen CSV: eksport z ratings.html (UTF-8 z BOM, przecinki) albo szablon zapisany w Excelu
    z polskimi ustawieniami (średniki, często kodowanie Windows-1250). Wszystkie pola jako tekst.
    """
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1250")
        log.warning(f"{Path(path).name}: plik nie jest w UTF-8 — odczyt jako Windows-1250 (Excel)")
    header = text.split("\n", 1)[0]
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(io.StringIO(text), dtype=str, sep=sep, keep_default_na=False)


def load_ratings(csv_paths: Sequence[Path | str], key_path: Path | str | None = None) -> pd.DataFrame:
    """Wypełnione arkusze CSV → ramka (model, case_id, litera, oceniajacy, skale, halucynacja)."""
    paths = [Path(p) for p in csv_paths]
    key = json.loads(_find_key(paths, key_path).read_text(encoding="utf-8"))
    frames = []
    for path in paths:
        df = read_sheet_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = [c for c in ("case_id", "litera", *SCALES) if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: brak kolumn {missing} (oczekiwane: {TEMPLATE_COLUMNS})")
        df["plik"] = path.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["case_id"] = df["case_id"].str.strip()
    df["litera"] = df["litera"].str.strip().str.upper()
    df["model"] = [key["cases"].get(c, {}).get(letter) for c, letter in zip(df["case_id"], df["litera"])]
    unknown = int(df["model"].isna().sum())
    if unknown:
        log.warning(f"Oceny: {unknown} wierszy z nieznanym przypadkiem / literą — pominięte")
    df = df[df["model"].notna()].copy()
    for s in SCALES:
        raw = df[s].astype(str).str.strip()
        df[s] = df[s].map(_score)
        bad = int((df[s].isna() & (raw != "")).sum())
        if bad:
            log.warning(f"Oceny: {bad} wartości '{s}' spoza 1–5 — pominięte")
    df[HALLUCINATION_COL] = df[HALLUCINATION_COL].map(_flag) if HALLUCINATION_COL in df.columns else np.nan
    if "oceniajacy" not in df.columns:
        df["oceniajacy"] = ""
    rated = df[list(SCALES)].notna().any(axis=1)
    # Halucynacja bez żadnej oceny w skalach = wiersz nieoceniony (domyślne „0” z eksportu)
    df.loc[~rated, HALLUCINATION_COL] = np.nan
    return df[rated].reset_index(drop=True)


def _cluster_bootstrap(df: pd.DataFrame, col: str, n_boot: int, seed: int, alpha: float = 0.05) -> list[float | None]:
    """95% CI średniej z bootstrapem po przypadkach (oceny tego samego przypadku nie są niezależne)."""
    sub = df[["case_id", col]].dropna()
    if sub.empty:
        return [None, None]
    groups = sub.groupby("case_id")[col]
    sums, counts = groups.sum().to_numpy(), groups.count().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def score_ratings(csv_paths: Sequence[Path | str], key_path: Path | str | None = None, n_boot: int = 2000,
                  seed: int = 42) -> dict[str, Any]:
    """
    Średnie ocen na model z 95% CI (bootstrap po przypadkach, `n_boot` powtórzeń), odsetek
    halucynacji (Clopper–Pearson), liczba ocen i oceniających. Zwraca {"summary", "n_ratings",
    "raters", "markdown"}.
    """
    from src.evaluation.statistics import clopper_pearson

    df = load_ratings(csv_paths, key_path)
    summary: dict[str, dict[str, Any]] = {}
    for model, sub in df.groupby("model", sort=True):
        entry: dict[str, Any] = {"n_ratings": int(len(sub)), "n_cases": int(sub["case_id"].nunique())}
        for s in SCALES:
            entry[s] = float(sub[s].mean()) if sub[s].notna().any() else None
            entry[f"{s}_ci"] = _cluster_bootstrap(sub, s, n_boot, seed)
        sub = sub.assign(_mean=sub[list(SCALES)].mean(axis=1))
        entry["mean_all"] = float(sub["_mean"].mean())
        entry["mean_all_ci"] = _cluster_bootstrap(sub, "_mean", n_boot, seed)
        flags = sub[HALLUCINATION_COL].dropna()
        k, n = int(flags.sum()), int(len(flags))
        entry["hallucination_rate"] = k / n if n else None
        entry["hallucination_ci"] = list(clopper_pearson(k, n)) if n else [None, None]
        summary[str(model)] = entry
    raters = sorted({str(r) for r in df["oceniajacy"] if str(r).strip()})
    result = {"summary": summary, "n_ratings": int(len(df)), "raters": raters}
    result["markdown"] = render_scores(result, n_boot)
    return result


def _ci_text(value: float | None, ci: list[float | None], digits: int = 2) -> str:
    if value is None:
        return "—"
    if not ci or ci[0] is None:
        return fmt_num(value, digits)
    return f"{fmt_num(value, digits)} ({fmt_num(ci[0], digits)}–{fmt_num(ci[1], digits)})"


def render_scores(result: dict[str, Any], n_boot: int = 2000) -> str:
    summary = result["summary"]
    ranked = sorted(summary, key=lambda m: -(summary[m]["mean_all"] or 0))
    lines = [
        "## Ślepa ocena ekspercka wyjaśnień ESI",
        "",
        f"Ocen: {result['n_ratings']}; oceniający: {len(result['raters']) or 'nie podano'}"
        + (f" ({', '.join(result['raters'])})" if result["raters"] else "") + ". Skale 1–5, średnia (95% CI, "
        f"bootstrap po przypadkach, {n_boot} powtórzeń).",
        "",
        "| Model | Ocen | " + " | ".join(SCALES.values()) + " | Średnia 3 skal | Halucynacje |",
        "|---|---|" + "---|" * len(SCALES) + "---|---|",
    ]
    for model in ranked:
        m = summary[model]
        hall = ("—" if m["hallucination_rate"] is None else
                f"{fmt_num(100 * m['hallucination_rate'], 0)}% ({fmt_num(100 * m['hallucination_ci'][0], 0)}–"
                f"{fmt_num(100 * m['hallucination_ci'][1], 0)}%)")
        lines.append(f"| `{model}` | {m['n_ratings']} | "
                     + " | ".join(_ci_text(m[s], m[f"{s}_ci"]) for s in SCALES)
                     + f" | {_ci_text(m['mean_all'], m['mean_all_ci'])} | {hall} |")
    return "\n".join(lines) + "\n"


def write_scores(result: dict[str, Any], out_dir: Path | str) -> dict[str, str]:
    """ratings_scores.json i ratings_scores.md w out_dir."""
    out = Path(out_dir)
    payload = {k: v for k, v in result.items() if k != "markdown"}
    return {"json": str(write_json_atomic(out / "ratings_scores.json", payload)),
            "md": str(write_text_atomic(out / "ratings_scores.md", result["markdown"]))}


__all__ = [
    "SCALES",
    "TEMPLATE_COLUMNS",
    "assign_letters",
    "load_ratings",
    "render_sheet",
    "score_ratings",
    "write_rating_sheet",
    "write_scores",
]
