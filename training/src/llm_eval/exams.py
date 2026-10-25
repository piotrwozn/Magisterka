"""
Część „exams” porównania modeli: polskie egzaminy lekarskie (NASK-PIB/Reassess-Polish-Medical-Exams).

Źródło pytań
    - Hugging Face (w domu): `NASK-PIB/Reassess-Polish-Medical-Exams`, podzbiory `r_lek` (LEK) i
      `r_pes_latest` (PES); wymaga `pip install -r requirements-app.txt` (datasets) i tokenu HF
      (`hf auth login` albo zmienna HF_TOKEN) oraz akceptacji warunków zbioru na stronie HF;
    - albo lokalny plik CSV / parquet (`--exam-file`, można podać kilka razy; prefiks
      `r_lek=ścieżka` / `r_pes_latest=ścieżka` wskazuje egzamin, inaczej kolumna `split`, nazwa
      pliku albo `exam_name`). Kolumny jak w zbiorze NASK: question, answer, exam_name, id (+ uuid).
      `question` = treść w 1. linii, opcje „A. …” … „E. …” w liniach 2–6; `answer` = litera.

Dwa protokoły (te same pytania, ten sam prompt)
    standard                 — pytanie jednokrotnego wyboru A–E bez zmian;
    abstaining_substitution  — tekst POPRAWNEJ opcji zastąpiony zdaniem „żadna z odpowiedzi nie
                               jest poprawna” (litera poprawnej odpowiedzi bez zmian) — dokładnie
                               jak `data_creator/create_abstaining_substitution.py` NASK, z jego
                               wykluczeniami: wzorce w treści (żadna, najczęstszy, z wyjątkiem, …)
                               i pytania, które NASK przeznacza do części otwartej (free form).
                               Model, który rozpoznaje wzorzec odpowiedzi zamiast wiedzy, traci
                               tu najwięcej — to protokół „prawdziwej kompetencji”.
    Pytania losujemy z puli kwalifikującej się do OBU protokołów, więc różnica standard −
    abstaining (wskaźnik „bias” NASK) jest sparowana na identycznych pytaniach.

Próba (seed 42, identyczna dla każdego modelu)
    N = 300 pytań LEK warstwowo wg edycji egzaminu (`exam_name`, przydział proporcjonalny) +
    do 200 pytań PES z medycyny ratunkowej / chorób wewnętrznych, gdy specjalizację da się
    rozpoznać w `exam_name` (inaczej losowe PES, też warstwowo).

Prompt (polski, metoda „prev_method” NASK): systemowy „Twoje zadanie to udzielenie odpowiedzi na
test medyczny dla lekarzy. … Odpowiedz w sposób:\\nPrawidłowa odpowiedź to B.”, w wiadomości
użytkownika pytanie. Litera wyciągana z OSTATNIEGO „odpowiedź to X” / „Answer: X”, samotnej litery
albo linii „X. …”; rozumowanie (thinking) jest ignorowane. Brak litery = odpowiedź błędna,
liczona osobno (odsetek błędów parsowania); błąd API po wszystkich ponowieniach — też błędna
i liczona osobno (ponowne uruchomienie ponawia tylko te pozycje).

Metryki (model × protokół × egzamin): trafność z dokładnym przedziałem Clopper–Pearsona,
odsetek błędów parsowania i API, opóźnienie, tokeny, koszt; porównanie parami z najlepszym
modelem (dokładny test McNemara + sparowany bootstrap Δ) i luka standard − abstaining.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.statistics import clopper_pearson
from src.llm_eval.cache import JsonlWriter, ResponseCache, prompt_hash
from src.llm_eval.client import ChatJob, ChatResult, OllamaChat
from src.llm_eval.pricing import TOKEN_ASSUMPTIONS, cost_usd, estimate_cost, estimate_tokens
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# Stałe
# ─────────────────────────────────────────
PART = "exams"
HF_REPO: str = "NASK-PIB/Reassess-Polish-Medical-Exams"
SPLIT_LEK, SPLIT_PES = "r_lek", "r_pes_latest"
SPLITS: tuple[str, ...] = (SPLIT_LEK, SPLIT_PES)
SPLIT_LABELS: dict[str, str] = {SPLIT_LEK: "LEK", SPLIT_PES: "PES", "all": "LEK+PES"}
PROTOCOLS: tuple[str, ...] = ("standard", "abstaining_substitution")
PROTOCOL_LABELS: dict[str, str] = {"standard": "standardowy (A–E)",
                                   "abstaining_substitution": "z podmianą na „żadna z odpowiedzi” (NASK)"}
LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E")
ABSTAIN_TEXT: str = "żadna z odpowiedzi nie jest poprawna"
REQUIRED_COLUMNS: tuple[str, ...] = ("question", "answer", "exam_name", "id")

# Prompt „prev_method” NASK (scripts/run_scripts/evaluation_all_splits.py)
SYSTEM_PROMPT_PL: str = (
    "Twoje zadanie to udzielenie odpowiedzi na test medyczny dla lekarzy. Spośród wszystkich odpowiedzi "
    "A,B,C,D,E wybierz tylko jedną. Jeżeli nie jesteś pewien, wybierz najbardziej prawdopodobną. "
    "Odpowiedz w sposób:\nPrawidłowa odpowiedź to B."
)

# Wykluczenia z create_abstaining_substitution.py (identify_not_already_abstain) — dosłownie
ABSTAIN_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r"żadn(?:a|e) ", r"żaden", r"prawidłowa odpowiedź", r"prawdziwe są odpowiedzi", r"powyższe",
    r"prawdopodobna", r"nie należy", r"nie jest", r"wyjątkiem", r"nie zalicza", r"wszystkie wymienione",
    r"z wyjątkiem", r"wszystki", r"przedstawiony", r"wymienionych", r"naj\w*",
)
# Pytania części otwartej NASK (create_free_form.py: identify + create_key) — wykluczane z abstaining
FREE_FORM_NEGATIVE_PATTERNS: tuple[str, ...] = (
    r"poniżej", r"który", r"stwierdzeń", r"Prawidłowa odpowiedź", r"które", r"żadna", r"prawdziwe są",
    r"wymienionych", r"prawdopodobne", r"poniższych", r"powyższych", r"podanych", r"zaproponowanych",
    r"powyższe", r"wszystkie", r"przedstawionych", r"niżej", r"przedstawiony", r"z wyjątkiem",
)
# Specjalizacje PES bliskie SOR (dopasowanie w exam_name, bez rozróżniania wielkości liter)
PES_SPECIALTY_PATTERNS: tuple[str, ...] = (r"ratunk", r"wewnętrzn", r"emergency", r"internal")

DEFAULT_N_LEK, DEFAULT_N_PES, DEFAULT_SEED = 300, 200, 42
DEFAULT_OPTIONS: dict[str, Any] = {"temperature": 0.0, "seed": 42, "num_predict": 4096}


# ─────────────────────────────────────────
# Wczytanie pytań
# ─────────────────────────────────────────
def _hf_hint() -> str:
    return (f"Zaloguj się do Hugging Face (`hf auth login` albo zmienna HF_TOKEN), zaakceptuj warunki "
            f"zbioru https://huggingface.co/datasets/{HF_REPO} i uruchom ponownie — albo podaj lokalny plik "
            f"--exam-file (CSV/parquet z kolumnami question, answer, exam_name, id).")


def load_hf_split(split: str) -> pd.DataFrame:
    """Podzbiór NASK z Hugging Face (te same warianty nazw co data_creator/utils.py NASK)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Brak pakietu `datasets` — zainstaluj dodatki: pip install -r requirements-app.txt. "
                           + _hf_hint()) from exc
    candidates = [f"{split}_pandas", split]
    last: Exception | None = None
    attempts: list[Callable[[str], Any]] = [
        lambda c: load_dataset(HF_REPO, c, split="all"),
        lambda c: load_dataset("parquet", data_files={"train": f"hf://datasets/{HF_REPO}/{c}/*.parquet"},
                               split="train"),
        lambda c: load_dataset(HF_REPO, split=c),
    ]
    for attempt in attempts:
        for cand in candidates:
            try:
                return attempt(cand).to_pandas()
            except Exception as exc:      # noqa: BLE001 — próbujemy kolejnych wariantów nazwy
                last = exc
    raise RuntimeError(f"Nie udało się pobrać podzbioru '{split}' z {HF_REPO}: {last}. " + _hf_hint()) from last


def _infer_split(name: str) -> str | None:
    text = str(name).lower()
    if "pes" in text:
        return SPLIT_PES
    if "lek" in text and "ldek" not in text:
        return SPLIT_LEK
    return None


def load_exam_file(spec: str | Path) -> pd.DataFrame:
    """
    Lokalny plik pytań (CSV UTF-8 albo parquet). spec może mieć prefiks 'r_lek=' / 'r_pes_latest='.
    Egzamin: prefiks > kolumna `split` > nazwa pliku > exam_name (wiersze nierozpoznane → błąd).
    """
    text = str(spec)
    forced = None
    for split in SPLITS:
        if text.startswith(split + "="):
            forced, text = split, text[len(split) + 1:]
    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku z pytaniami: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path, encoding="utf-8")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: brak kolumn {missing} (wymagane: {list(REQUIRED_COLUMNS)})")
    df = df.copy()
    if forced:
        df["split"] = forced
    elif "split" not in df.columns or df["split"].isna().all():
        from_name = _infer_split(path.stem)
        df["split"] = from_name if from_name else [_infer_split(e) for e in df["exam_name"]]
    df["split"] = df["split"].astype("string").str.replace("_pandas", "", regex=False)
    unknown = ~df["split"].isin(SPLITS)
    if unknown.any():
        raise ValueError(f"{path}: {int(unknown.sum())} wierszy bez rozpoznanego egzaminu (LEK/PES) — dodaj kolumnę "
                         f"'split' (r_lek / r_pes_latest) albo prefiks 'r_lek=' przed ścieżką")
    return df


def load_exam_sources(exam_files: Sequence[str | Path] | None = None,
                      splits: Sequence[str] = SPLITS) -> pd.DataFrame:
    """Pytania LEK i PES (z plików albo z Hugging Face) z kolumną `split`."""
    if exam_files:
        frames = [load_exam_file(f) for f in exam_files]
        df = pd.concat(frames, ignore_index=True)
    else:
        frames = []
        for split in splits:
            part = load_hf_split(split)
            part = part.copy()
            part["split"] = split
            frames.append(part)
        df = pd.concat(frames, ignore_index=True)
    return normalize_frame(df)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Typy kolumn, litera odpowiedzi, uuid (gdy brak — zastępczy z egzaminu, edycji i id), item_id."""
    out = df.copy()
    out["question"] = out["question"].astype(str).str.replace("\r\n", "\n", regex=False)   # CSV z Windows
    out["answer"] = out["answer"].astype(str).str.strip().str.upper()
    out["exam_name"] = out["exam_name"].astype(str)
    out["id"] = out["id"].astype(str)
    if "uuid" not in out.columns:
        out["uuid"] = None
    surrogate = out["split"].astype(str) + "|" + out["exam_name"] + "|" + out["id"]
    out["uuid"] = out["uuid"].where(out["uuid"].notna() & (out["uuid"].astype(str) != ""), surrogate).astype(str)
    out["item_id"] = out["uuid"]
    return out.reset_index(drop=True)


# ─────────────────────────────────────────
# Protokoły (NASK)
# ─────────────────────────────────────────
def split_question(question: str) -> tuple[str, list[str]]:
    """(treść, opcje) — jak NASK: linia 1 = treść, kolejne linie = opcje."""
    lines = str(question).split("\n")
    return lines[0], lines[1:]


def free_form_mask(df: pd.DataFrame) -> pd.Series:
    """Pytania, które NASK przenosi do części otwartej (create_free_form.py) — wykluczane z abstaining."""
    q = df["question"].astype(str)
    has_q = q.str.contains(r"\?", case=False, regex=True)
    negative = q.str.contains("|".join(FREE_FORM_NEGATIVE_PATTERNS), case=False, regex=True)

    def five_after_qmark(text: str) -> bool:
        rest = re.split(r"[\?]", text, maxsplit=1)[-1]
        return len(rest.split("\n")[1:]) == 5

    shape = q.map(five_after_qmark)
    return has_q & ~negative & shape & df["answer"].isin(LETTERS)


def abstaining_mask(df: pd.DataFrame) -> pd.Series:
    """Pytania kwalifikujące się do abstaining_substitution (i do standardu: 5 opcji, litera A–E)."""
    q = df["question"].astype(str)
    excluded = q.str.contains("|".join(ABSTAIN_EXCLUDE_PATTERNS), flags=re.IGNORECASE, regex=True, na=False)
    five = q.map(lambda t: len(split_question(t)[1]) == 5)
    return ~excluded & ~free_form_mask(df) & five & df["answer"].isin(LETTERS)


def substitute_abstain(question: str, answer: str) -> str:
    """Tekst poprawnej opcji → „żadna z odpowiedzi nie jest poprawna” (create_key_abstaining NASK)."""
    stem, options = split_question(question)
    if len(options) != 5 or answer not in LETTERS:
        raise ValueError("Pytanie musi mieć dokładnie 5 opcji i odpowiedź A–E")
    options = list(options)
    options[LETTERS.index(answer)] = f"{answer}. {ABSTAIN_TEXT}"
    return stem + "\n" + "\n".join(options)


def question_text(item: pd.Series | dict[str, Any], protocol: str) -> str:
    if protocol == "standard":
        return str(item["question"])
    if protocol == "abstaining_substitution":
        return substitute_abstain(str(item["question"]), str(item["answer"]))
    raise ValueError(f"Nieznany protokół '{protocol}' (dostępne: {PROTOCOLS})")


def build_messages(text: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT_PL}, {"role": "user", "content": text}]


# ─────────────────────────────────────────
# Losowanie próby
# ─────────────────────────────────────────
def stratified_sample(df: pd.DataFrame, n: int, strata_col: str, seed: int) -> pd.DataFrame:
    """
    n wierszy z przydziałem proporcjonalnym do liczności warstw (metoda największych reszt),
    w warstwie losowo bez zwracania. Deterministyczne: wiersze i warstwy sortowane przed losowaniem.
    """
    if n <= 0 or df.empty:
        return df.iloc[0:0]
    df = df.sort_values(["item_id"]).reset_index(drop=True)
    if n >= len(df):
        return df
    sizes = df.groupby(strata_col, sort=True).size()
    exact = sizes / sizes.sum() * n
    quota = np.floor(exact).astype(int)
    remainder = n - int(quota.sum())
    if remainder > 0:
        order = sorted(sizes.index, key=lambda s: (-(exact[s] - quota[s]), str(s)))
        for s in order[:remainder]:
            quota[s] += 1
    rng = np.random.default_rng(seed)
    picked = []
    for stratum in sizes.index:
        rows = df[df[strata_col] == stratum]
        k = int(min(quota[stratum], len(rows)))
        if k > 0:
            idx = np.sort(rng.choice(len(rows), size=k, replace=False))
            picked.append(rows.iloc[idx])
    return pd.concat(picked).sort_values([strata_col, "item_id"]).reset_index(drop=True)


SPECIALTY_COLUMNS: tuple[str, ...] = ("specialty", "specialization", "specjalizacja", "category", "subject")


def _specialty_text(df: pd.DataFrame) -> pd.Series:
    """exam_name + ewentualne kolumny ze specjalizacją (jeśli zbiór je ma) — tekst do dopasowania wzorców PES."""
    text = df["exam_name"].astype(str)
    for col in SPECIALTY_COLUMNS:
        if col in df.columns:
            text = text + " " + df[col].fillna("").astype(str)
    return text


def sample_items(df: pd.DataFrame, n_lek: int = DEFAULT_N_LEK, n_pes: int = DEFAULT_N_PES,
                 seed: int = DEFAULT_SEED,
                 pes_patterns: Sequence[str] = PES_SPECIALTY_PATTERNS) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Próba pytań (LEK warstwowo; PES — specjalizacje SOR, gdy rozpoznawalne) + opis losowania."""
    df = normalize_frame(df) if "item_id" not in df.columns else df
    df = df.drop_duplicates("item_id").reset_index(drop=True)
    eligible = abstaining_mask(df)
    pool = df[eligible]
    lek_pool = pool[pool["split"] == SPLIT_LEK]
    pes_pool = pool[pool["split"] == SPLIT_PES]
    pattern = "|".join(pes_patterns)
    spec = pes_pool[_specialty_text(pes_pool).str.contains(pattern, case=False, regex=True, na=False)] \
        if pattern else pes_pool.iloc[0:0]
    pes_mode = "specialty" if len(spec) else "random"
    lek = stratified_sample(lek_pool, n_lek, "exam_name", seed)
    pes = stratified_sample(spec if len(spec) else pes_pool, n_pes, "exam_name", seed)
    items = pd.concat([lek, pes], ignore_index=True)
    info = {
        "seed": seed,
        "n_requested": {"lek": int(n_lek), "pes": int(n_pes)},
        "n": {"lek": int(len(lek)), "pes": int(len(pes)), "all": int(len(items))},
        "source_rows": {s: int((df["split"] == s).sum()) for s in SPLITS},
        "eligible_rows": {s: int((pool["split"] == s).sum()) for s in SPLITS},
        "pes_mode": pes_mode,
        "pes_specialty_rows": int(len(spec)),
        "pes_patterns": list(pes_patterns),
        "pes_exam_names": sorted(pes["exam_name"].unique().tolist()),
        "lek_editions": int(lek["exam_name"].nunique()),
    }
    if n_pes > 0 and pes_mode == "random":
        log.warning("PES: nie rozpoznano specjalizacji SOR / chorób wewnętrznych w exam_name — losowe pytania PES")
    for name, got, want in (("LEK", len(lek), n_lek), ("PES", len(pes), n_pes)):
        if got < want:
            log.warning(f"{name}: dostępnych tylko {got} pytań kwalifikujących się (żądano {want})")
    cols = ["item_id", "split", "exam_name", "id", "uuid", "answer", "question"]
    return items[cols], info


# ─────────────────────────────────────────
# Parsowanie odpowiedzi
# ─────────────────────────────────────────
_LETTER_TAIL = r"[\s:：\-–—=]*[\*_\"'„”«»(\[]*\s*([A-E])(?!\w)"
_KW = r"(?:odpowied(?:ź|zią|z(?!i))|answer)"
_ADJ = r"(?:prawidłow\w*|poprawn\w*|właściw\w*|correct|final|ostateczn\w*|końcow\w*)"
_LINK = r"(?:\s+(?:to|jest|brzmi|is))"
# Silne wzorce: jawna deklaracja odpowiedzi („Prawidłowa odpowiedź to B”, „Odpowiedź prawidłowa: B”,
# „odpowiedź to B”, „Answer: B”). Słabe: samo „odpowiedź X” (np. w uzasadnieniu „odpowiedź A jest błędna”).
_STRONG_RES = tuple(re.compile(p + _LETTER_TAIL) for p in (
    rf"(?i:{_ADJ}\s+{_KW}{_LINK}*)",
    rf"(?i:{_KW}\s+{_ADJ}{_LINK}*)",
    rf"(?i:{_KW}{_LINK}+)",
    rf"(?i:{_KW})\s*[:：]",
))
_WEAK_RE = re.compile(rf"(?i:{_KW}){_LINK}*" + _LETTER_TAIL, )
_LONE_RE = re.compile(r"^[\s\*_\"'„”«»(\[]*([A-E])[\s\.\)\]\*_\"'”»:]*$")
_LINE_RE = re.compile(r"^[\s\*_\"'„«(\[]*([A-E])(?:[\.\)\]:]|\*\*|\s*$)")


def parse_answer(content: str | None) -> str | None:
    """
    Litera A–E z odpowiedzi modelu albo None. Kolejność: OSTATNIA jawna deklaracja („Prawidłowa
    odpowiedź to X”, „Odpowiedź prawidłowa: X”, „odpowiedź to X”, „Answer: X”), samotna litera
    („B”, „**B.**”), ostatnia linia zaczynająca się od „X.” / „X)” / „X”, ostatnie słabe
    „odpowiedź X”, pierwsza linia „X. …”. Rozumowanie (thinking) nie jest tu przekazywane.
    """
    text = str(content or "").strip()
    if not text:
        return None
    strong = [(m.end(), m.group(1)) for rx in _STRONG_RES for m in rx.finditer(text)]
    if strong:
        return max(strong)[1]
    lone = _LONE_RE.match(text)
    if lone:
        return lone.group(1)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    last = _LINE_RE.match(lines[-1])
    if last:
        return last.group(1)
    weak = _WEAK_RE.findall(text)
    if weak:
        return weak[-1]
    first = _LINE_RE.match(lines[0])
    return first.group(1) if first else None


# ─────────────────────────────────────────
# Uruchomienie (zapytania + pamięć podręczna)
# ─────────────────────────────────────────
def plan_jobs(items: pd.DataFrame, models: Sequence[str], think: Any = "low",
              options: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Zadania w kolejności pytanie → protokół → model (przerwany przebieg pokrywa wszystkie modele)."""
    opts = DEFAULT_OPTIONS if options is None else options
    jobs = []
    for item in items.to_dict(orient="records"):
        for protocol in PROTOCOLS:
            messages = build_messages(question_text(item, protocol))
            phash = prompt_hash(messages, think=think, options=opts)
            for model in models:
                jobs.append({"model": model, "protocol": protocol, "item": item, "messages": messages,
                             "cache_id": f"{protocol}:{item['item_id']}", "prompt_hash": phash})
    return jobs


def _row(job: dict[str, Any], res: ChatResult, cached: bool) -> dict[str, Any]:
    item = job["item"]
    predicted = parse_answer(res.content) if res.ok else None
    return {
        "model": job["model"], "protocol": job["protocol"], "item_id": item["item_id"], "split": item["split"],
        "exam_name": item["exam_name"], "answer": item["answer"], "predicted": predicted,
        "correct": bool(predicted is not None and predicted == item["answer"]),
        "parse_ok": predicted is not None, "error": res.error,
        "latency_s": res.latency_s if res.ok else None,
        "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
        "thinking_chars": len(res.thinking or ""), "cached": cached,
        "content": (res.content or "")[:300],
    }


def run_exam_jobs(client: OllamaChat, jobs: list[dict[str, Any]], *, think: Any, cache: ResponseCache | None,
                  options: dict[str, Any] | None = None, raw_log: JsonlWriter | None = None,
                  progress_every: int | None = None) -> pd.DataFrame:
    """Odpowiedzi na wszystkie zadania (z pamięci podręcznej albo z API) → ramka wyników pozycji."""
    opts = DEFAULT_OPTIONS if options is None else options
    rows: list[dict[str, Any] | None] = [None] * len(jobs)
    todo: list[int] = []
    for i, job in enumerate(jobs):
        hit = cache.get(PART, job["model"], job["cache_id"], job["prompt_hash"]) if cache is not None else None
        if hit is not None:
            res = ChatResult.from_dict(hit)
            rows[i] = _row(job, res, cached=True)
            if raw_log is not None:
                raw_log.write({"part": PART, "model": job["model"], "item_id": job["cache_id"], "cached": True,
                               "result": hit})
        else:
            todo.append(i)
    n_cached = len(jobs) - len(todo)
    log.info(f"Egzaminy: {len(jobs)} zapytań, z pamięci podręcznej {n_cached}, do wysłania {len(todo)}")
    if not todo:
        return pd.DataFrame(rows)

    every = progress_every or max(10, len(todo) // 20)
    started = time.perf_counter()
    done = {"n": 0, "errors": 0}
    chat_jobs = [ChatJob(model=jobs[i]["model"], messages=jobs[i]["messages"], think=think, options=opts, key=i)
                 for i in todo]

    def on_result(_: int, cj: ChatJob, res: ChatResult) -> None:
        i = int(cj.key)
        job = jobs[i]
        rows[i] = _row(job, res, cached=False)
        if cache is not None:
            cache.put(PART, job["model"], job["cache_id"], job["prompt_hash"], res,
                      meta={"protocol": job["protocol"], "think": think})
        if raw_log is not None:
            raw_log.write({"part": PART, "model": job["model"], "item_id": job["cache_id"], "cached": False,
                           "result": res.to_dict()})
        done["n"] += 1
        done["errors"] += 0 if res.ok else 1
        if done["n"] % every == 0 or done["n"] == len(todo):
            elapsed = time.perf_counter() - started
            eta = elapsed / done["n"] * (len(todo) - done["n"])
            log.info(f"Egzaminy: {done['n']}/{len(todo)} ({100 * done['n'] / len(todo):.0f}%), błędy API "
                     f"{done['errors']}, pozostało ≈ {eta / 60:.1f} min")

    client.chat_many(chat_jobs, on_result=on_result)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Statystyka
# ─────────────────────────────────────────
def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Dokładny test McNemara (dwumianowy na parach niezgodnych): b = A dobrze/B źle, c = A źle/B dobrze."""
    from scipy.stats import binomtest

    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    p = 1.0 if n01 + n10 == 0 else float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
    return {"a_only": n01, "b_only": n10, "p_mcnemar": min(1.0, p)}


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = DEFAULT_SEED,
                           alpha: float = 0.05) -> dict[str, Any]:
    """Δ = trafność(A) − trafność(B) na tych samych pytaniach; 95% przedział z percentyli bootstrapu."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    if n == 0:
        return {"delta": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    chunk = max(1, 2_000_000 // max(n, 1))
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        boots[start:start + size] = d[rng.integers(0, n, size=(size, n))].mean(axis=1)
    low, high = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"delta": float(d.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(n)}


def correctness_matrix(responses: pd.DataFrame, items: pd.DataFrame, protocol: str,
                       models: Sequence[str], split: str = "all") -> dict[str, np.ndarray]:
    """{model: wektor trafień} wyrównany do kolejności pytań (brak wiersza = błędna)."""
    ids = items["item_id"] if split == "all" else items.loc[items["split"] == split, "item_id"]
    ids = ids.astype(str).tolist()
    sub = responses[responses["protocol"] == protocol]
    out = {}
    for model in models:
        m = sub[sub["model"] == model].set_index("item_id")["correct"]
        m = m[~m.index.duplicated(keep="last")]
        out[model] = m.reindex(ids).fillna(False).astype(bool).to_numpy()
    return out


def _summary(model: str, protocol: str, split: str, sub: pd.DataFrame,
             prices: dict[str, dict[str, float]] | None) -> dict[str, Any]:
    n = int(len(sub))
    k = int(sub["correct"].sum())
    low, high = clopper_pearson(k, n) if n else (None, None)
    ok = sub[sub["error"].isna()]
    lat = pd.to_numeric(ok["latency_s"], errors="coerce").dropna()
    pt = pd.to_numeric(ok["prompt_tokens"], errors="coerce")
    ct = pd.to_numeric(ok["completion_tokens"], errors="coerce")
    errors = int(sub["error"].notna().sum())
    parse_fail = int(((~sub["parse_ok"]) & sub["error"].isna()).sum())
    return {
        "model": model, "protocol": protocol, "split": split, "n": n, "correct": k,
        "accuracy": k / n if n else None, "ci_low": low, "ci_high": high,
        "parse_failures": parse_fail, "parse_failure_rate": parse_fail / n if n else None,
        "api_errors": errors, "api_error_rate": errors / n if n else None,
        "latency_median_s": float(lat.median()) if len(lat) else None,
        "latency_mean_s": float(lat.mean()) if len(lat) else None,
        "latency_p90_s": float(lat.quantile(0.9)) if len(lat) else None,
        "prompt_tokens_mean": float(pt.mean()) if pt.notna().any() else None,
        "completion_tokens_mean": float(ct.mean()) if ct.notna().any() else None,
        "prompt_tokens_total": int(pt.fillna(0).sum()), "completion_tokens_total": int(ct.fillna(0).sum()),
        "cost_usd": cost_usd(model, pt.fillna(0).sum(), ct.fillna(0).sum(), prices),
        "cached": int(sub["cached"].sum()) if "cached" in sub else 0,
    }


def evaluate(responses: pd.DataFrame, items: pd.DataFrame, models: Sequence[str],
             prices: dict[str, dict[str, float]] | None = None, n_boot: int = 2000,
             seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Metryki, porównania parami z najlepszym modelem (na protokół) i luka standard − abstaining."""
    models = list(models)
    metrics = []
    for model in models:
        for protocol in PROTOCOLS:
            base = responses[(responses["model"] == model) & (responses["protocol"] == protocol)]
            for split in (*SPLITS, "all"):
                sub = base if split == "all" else base[base["split"] == split]
                if split != "all" and not len(sub):
                    continue
                metrics.append(_summary(model, protocol, split, sub, prices))
    table = pd.DataFrame(metrics)

    def acc(model: str, protocol: str, split: str = "all") -> float:
        row = table[(table["model"] == model) & (table["protocol"] == protocol) & (table["split"] == split)]
        return float(row["accuracy"].iloc[0]) if len(row) and row["accuracy"].iloc[0] is not None else -1.0

    paired: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        other = "standard" if protocol != "standard" else "abstaining_substitution"
        best = max(models, key=lambda m: (acc(m, protocol), acc(m, other), -models.index(m))) if models else None
        rows = []
        if best is not None:
            corr = correctness_matrix(responses, items, protocol, models)
            for model in models:
                if model == best:
                    continue
                boot = paired_bootstrap_delta(corr[model], corr[best], n_boot=n_boot, seed=seed)
                mc = mcnemar_exact(corr[model], corr[best])
                tied = boot["ci_low"] is not None and boot["ci_low"] <= 0 <= boot["ci_high"]
                rows.append({"model": model, "vs": best, **boot, "model_only": mc["a_only"],
                             "best_only": mc["b_only"], "p_mcnemar": mc["p_mcnemar"],
                             "tied_with_best": bool(tied)})
        paired[protocol] = {"best": best, "rows": rows}

    gap = []
    for model in models:
        std = correctness_matrix(responses, items, "standard", [model])[model]
        abst = correctness_matrix(responses, items, "abstaining_substitution", [model])[model]
        boot = paired_bootstrap_delta(std, abst, n_boot=n_boot, seed=seed)
        mc = mcnemar_exact(std, abst)
        gap.append({"model": model, "acc_standard": float(std.mean()) if len(std) else None,
                    "acc_abstaining": float(abst.mean()) if len(abst) else None, **boot,
                    "p_mcnemar": mc["p_mcnemar"]})
    return {"metrics": metrics, "paired_vs_best": paired, "gap": gap, "n_boot": n_boot}


def pairwise(responses: pd.DataFrame, items: pd.DataFrame, protocol: str, a: str, b: str,
             n_boot: int = 2000, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Sparowane porównanie dwóch modeli (Δ = A − B, CI bootstrap, McNemar) w danym protokole."""
    corr = correctness_matrix(responses, items, protocol, [a, b])
    boot = paired_bootstrap_delta(corr[a], corr[b], n_boot=n_boot, seed=seed)
    mc = mcnemar_exact(corr[a], corr[b])
    return {**boot, "p_mcnemar": mc["p_mcnemar"],
            "tied": bool(boot["ci_low"] is not None and boot["ci_low"] <= 0 <= boot["ci_high"])}


# ─────────────────────────────────────────
# Część „exams” dla CLI
# ─────────────────────────────────────────
def estimate_exams(items: pd.DataFrame | None, models: Sequence[str], cache: ResponseCache | None, think: Any,
                   prices: dict[str, dict[str, float]] | None, n_items_nominal: int,
                   options: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Plan i szacunek kosztu (wszystkie zapytania i tylko te spoza pamięci podręcznej) per model."""
    out = []
    completion = TOKEN_ASSUMPTIONS[PART]["completion_tokens"]
    if items is None or not len(items):
        n_calls = n_items_nominal * len(PROTOCOLS)
        for model in models:
            est = estimate_cost(model, n_calls, 250 + estimate_tokens(SYSTEM_PROMPT_PL), completion, prices)
            out.append({**est, "part": PART, "uncached_calls": n_calls, "uncached_cost_usd": est["cost_usd"]})
        return out
    jobs = plan_jobs(items, models, think=think, options=options)
    for model in models:
        mj = [j for j in jobs if j["model"] == model]
        prompt = float(np.mean([sum(estimate_tokens(m["content"]) for m in j["messages"]) for j in mj])) if mj else 0.0
        uncached = [j for j in mj if cache is None
                    or cache.get(PART, model, j["cache_id"], j["prompt_hash"]) is None]
        est = estimate_cost(model, len(mj), prompt, completion, prices)
        est_un = estimate_cost(model, len(uncached), prompt, completion, prices)
        out.append({**est, "part": PART, "uncached_calls": len(uncached), "uncached_cost_usd": est_un["cost_usd"]})
    return out


def run_exams_part(client: OllamaChat, models: Sequence[str], *, out_dir: Path, cache: ResponseCache | None,
                   think: Any = "low", n_lek: int = DEFAULT_N_LEK, n_pes: int = DEFAULT_N_PES,
                   seed: int = DEFAULT_SEED, exam_files: Sequence[str] | None = None,
                   prices: dict[str, dict[str, float]] | None = None, raw_log: JsonlWriter | None = None,
                   options: dict[str, Any] | None = None, n_boot: int = 2000,
                   source: pd.DataFrame | None = None) -> dict[str, Any]:
    """
    Cała część egzaminacyjna: próba → zapytania (z pamięcią podręczną) → metryki. Zapisuje
    exam_items.csv, exam_responses.csv, exam_metrics.csv w out_dir. Zwraca słownik wyników
    (klucze: sample, protocols, think, options, metrics, paired_vs_best, gap, n_boot).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = source if source is not None else load_exam_sources(exam_files)
    items, info = sample_items(df, n_lek=n_lek, n_pes=n_pes, seed=seed)
    if items.empty:
        raise RuntimeError("Brak pytań kwalifikujących się do porównania (sprawdź plik / podzbiory LEK i PES)")
    log.info(f"Egzaminy: próba {info['n']['lek']} LEK + {info['n']['pes']} PES "
             f"(PES: {'specjalizacje SOR/interna' if info['pes_mode'] == 'specialty' else 'losowe'}), seed {seed}")
    items.to_csv(out_dir / "exam_items.csv", index=False, encoding="utf-8")
    opts = DEFAULT_OPTIONS if options is None else options
    jobs = plan_jobs(items, models, think=think, options=opts)
    responses = run_exam_jobs(client, jobs, think=think, cache=cache, options=opts, raw_log=raw_log)
    responses.to_csv(out_dir / "exam_responses.csv", index=False, encoding="utf-8")
    results = evaluate(responses, items, models, prices=prices, n_boot=n_boot, seed=seed)
    pd.DataFrame(results["metrics"]).to_csv(out_dir / "exam_metrics.csv", index=False, encoding="utf-8")
    results.update({"sample": info, "protocols": list(PROTOCOLS), "think": think, "options": opts,
                    "models": list(models), "prompt": SYSTEM_PROMPT_PL})
    # Wektory trafień (do rozstrzygania remisów w wyborze modelu) — tylko w pamięci, nie w JSON
    results["_correctness"] = {p: correctness_matrix(responses, items, p, models) for p in PROTOCOLS}
    return results


__all__ = [
    "ABSTAIN_EXCLUDE_PATTERNS",
    "ABSTAIN_TEXT",
    "DEFAULT_OPTIONS",
    "FREE_FORM_NEGATIVE_PATTERNS",
    "HF_REPO",
    "PES_SPECIALTY_PATTERNS",
    "PROTOCOLS",
    "SYSTEM_PROMPT_PL",
    "abstaining_mask",
    "build_messages",
    "correctness_matrix",
    "estimate_exams",
    "evaluate",
    "free_form_mask",
    "load_exam_file",
    "load_exam_sources",
    "load_hf_split",
    "mcnemar_exact",
    "paired_bootstrap_delta",
    "pairwise",
    "parse_answer",
    "plan_jobs",
    "run_exam_jobs",
    "run_exams_part",
    "sample_items",
    "stratified_sample",
    "substitute_abstain",
]
