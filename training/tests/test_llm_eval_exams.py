"""
Testy części „exams” porównania modeli (src/llm_eval/exams.py) i CLI scripts/12_llm_bakeoff.py
na syntetycznych pytaniach w formacie NASK i atrapie serwera Ollamy (bez sieci, bez HF).
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.llm_eval.cache import ResponseCache
from src.llm_eval.client import OllamaChat
from src.llm_eval.exams import (
    ABSTAIN_TEXT,
    SPLIT_LEK,
    SPLIT_PES,
    abstaining_mask,
    free_form_mask,
    load_exam_file,
    load_hf_split,
    mcnemar_exact,
    normalize_frame,
    paired_bootstrap_delta,
    parse_answer,
    run_exams_part,
    sample_items,
    stratified_sample,
    substitute_abstain,
)
from tests.fake_ollama import FakeOllama, Reply, answer_key_from_frame, exam_oracle

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def srv():
    """Uruchomiona atrapa Ollamy (tests/fake_ollama.py) — bez modeli, dodawane w teście."""
    with FakeOllama() as server:
        yield server


# ─────────────────────────────────────────
# Syntetyczne pytania w formacie NASK
# ─────────────────────────────────────────
EDITIONS = {
    SPLIT_LEK: ["LEK jesień 2022", "LEK wiosna 2023", "LEK jesień 2023"],
    SPLIT_PES: ["PES Medycyna ratunkowa wiosna 2023", "PES Choroby wewnętrzne jesień 2023",
                "PES Kardiologia jesień 2022", "PES Pediatria wiosna 2024"],
}


def make_nask_frame(n_per_edition: int = 40, seed: int = 0, pes_editions: list[str] | None = None) -> pd.DataFrame:
    """
    Pytania jak w NASK (treść w 1. linii, opcje „A. …” w liniach 2–6, litera w `answer`). Co 10. pytanie
    ma „naj…” (wykluczenie abstaining), co 10. jest pytaniem otwartym NASK („…?” bez słów wykluczających),
    co 10. ma tylko 4 opcje — pozostałe kwalifikują się do obu protokołów.
    """
    rng = random.Random(seed)
    rows = []
    editions = dict(EDITIONS)
    if pes_editions is not None:
        editions[SPLIT_PES] = pes_editions
    for split, names in editions.items():
        for e, edition in enumerate(names):
            for q in range(n_per_edition):
                kind = q % 10
                tag = f"{split[2:5]}{e}-{q}"
                if kind == 0:
                    stem = f"Najczęstszą przyczyną objawu {tag} jest:"
                elif kind == 1:
                    stem = f"Jaki lek podać w stanie {tag}?"
                elif kind == 2:
                    stem = f"Wskaż, który z leków {tag} stosuje się w tym zespole?"
                else:
                    stem = f"Lekiem pierwszego wyboru w zespole {tag} jest:"
                letters = "ABCD" if kind == 3 else "ABCDE"
                options = [f"{L}. preparat {L.lower()}-{tag}" for L in letters]
                rows.append({"question": stem + "\n" + "\n".join(options), "answer": rng.choice(letters),
                             "exam_name": edition, "id": q + 1, "uuid": f"u-{tag}", "split": split})
    return pd.DataFrame(rows)


@pytest.fixture
def nask_csv(tmp_path) -> Path:
    path = tmp_path / "nask_synthetic.csv"
    make_nask_frame().to_csv(path, index=False, encoding="utf-8")
    return path


# ─────────────────────────────────────────
# Protokoły NASK
# ─────────────────────────────────────────
def test_substitute_abstain_replaces_only_correct_option():
    q = "Lekiem z wyboru jest:\nA. aspiryna\nB. heparyna\nC. tlen\nD. morfina\nE. nitrogliceryna"
    out = substitute_abstain(q, "C")
    lines = out.split("\n")
    assert lines[0] == "Lekiem z wyboru jest:" and lines[3] == f"C. {ABSTAIN_TEXT}"
    assert lines[1] == "A. aspiryna" and lines[5] == "E. nitrogliceryna"
    with pytest.raises(ValueError):
        substitute_abstain("Pytanie:\nA. x\nB. y", "A")


def test_abstaining_mask_applies_nask_exclusions():
    df = normalize_frame(make_nask_frame(n_per_edition=10))
    mask = abstaining_mask(df)
    kinds = df["id"].astype(int).sub(1) % 10
    assert not mask[kinds == 0].any()            # „naj…”
    assert not mask[kinds == 1].any()            # pytanie otwarte NASK
    assert not mask[kinds == 3].any()            # 4 opcje
    assert mask[kinds == 2].all()                # „…który…?” nie jest pytaniem otwartym
    assert mask[kinds.isin([4, 5, 6, 7, 8, 9])].all()
    assert free_form_mask(df)[kinds == 1].all() and not free_form_mask(df)[kinds == 2].any()
    extra = normalize_frame(pd.DataFrame([
        {"question": "Z wyjątkiem jednego, objawem jest:\nA. a\nB. b\nC. c\nD. d\nE. e", "answer": "A",
         "exam_name": "LEK", "id": 1, "split": SPLIT_LEK},
        {"question": "ŻADEN z tych leków nie działa:\nA. a\nB. b\nC. c\nD. d\nE. e", "answer": "A",
         "exam_name": "LEK", "id": 2, "split": SPLIT_LEK},
        {"question": "Objawem jest:\nA. a\nB. b\nC. c\nD. d\nE. e", "answer": "F",
         "exam_name": "LEK", "id": 3, "split": SPLIT_LEK},
    ]))
    assert not abstaining_mask(extra).any()


@pytest.mark.parametrize("text,expected", [
    ("Prawidłowa odpowiedź to B.", "B"),
    ("prawidłowa odpowiedź to **D**", "D"),
    ("Odpowiedź: C", "C"),
    ("Answer: E", "E"),
    ("The correct answer is A.", "A"),
    ("Odpowiedź A jest błędna. Prawidłowa odpowiedź to C.", "C"),
    ("B", "B"),
    ("**E.**", "E"),
    ("(A)", "A"),
    ("C. Zawał serca", "C"),
    ("Uzasadnienie: objawy typowe.\nD) zatorowość płucna", "D"),
    ("A to zależy od sytuacji", None),
    ("Trudno powiedzieć.", None),
    ("", None),
    (None, None),
    ("Prawidłowa odpowiedź to Bóle głowy", None),
    ("Odpowiedź prawidłowa: B", "B"),
    ("Prawidłowa odpowiedź to B. Odpowiedź A jest błędna, bo dotyczy innej choroby.", "B"),
    ("Final answer: C", "C"),
    ("Odpowiedź końcowa – **E**", "E"),
    ("Rozważam odpowiedź A, ale wybieram B.\nB", "B"),
    ("Moim zdaniem odpowiedź D, bo objawy są typowe", "D"),
    ("C. Zawał serca\nUzasadnienie: typowy obraz.", "C"),
])
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


# ─────────────────────────────────────────
# Losowanie próby
# ─────────────────────────────────────────
def test_stratified_sample_is_proportional_and_deterministic():
    df = pd.DataFrame({"item_id": [f"i{i:03d}" for i in range(100)],
                       "exam_name": ["a"] * 50 + ["b"] * 30 + ["c"] * 20})
    s1 = stratified_sample(df, 10, "exam_name", seed=42)
    s2 = stratified_sample(df.sample(frac=1, random_state=3), 10, "exam_name", seed=42)
    assert s1["item_id"].tolist() == s2["item_id"].tolist()          # niezależne od kolejności wierszy
    assert s1["exam_name"].value_counts().to_dict() == {"a": 5, "b": 3, "c": 2}
    assert len(stratified_sample(df, 500, "exam_name", 42)) == 100
    assert stratified_sample(df, 10, "exam_name", seed=1)["item_id"].tolist() != s1["item_id"].tolist()


def test_sample_items_prefers_ed_and_internal_pes():
    items, info = sample_items(make_nask_frame(), n_lek=30, n_pes=20, seed=42)
    assert info["n"] == {"lek": 30, "pes": 20, "all": 50} and info["pes_mode"] == "specialty"
    pes = items[items["split"] == SPLIT_PES]
    assert set(pes["exam_name"]) <= {"PES Medycyna ratunkowa wiosna 2023", "PES Choroby wewnętrzne jesień 2023"}
    assert abstaining_mask(items.assign(answer=items["answer"])).all()
    assert items["item_id"].is_unique


def test_pes_specialty_matching_ignores_diacritics_and_case():
    df = make_nask_frame(pes_editions=["PES_CHOROBY_WEWNETRZNE_2023", "pes kardiologia 2023"])
    items, info = sample_items(df, n_lek=5, n_pes=10, seed=42)
    assert info["pes_mode"] == "specialty" and info["pes_exam_names"] == ["PES_CHOROBY_WEWNETRZNE_2023"]


def test_sample_items_random_pes_when_specialty_unknown():
    df = make_nask_frame(pes_editions=["PES jesień 2022", "PES wiosna 2023"])
    items, info = sample_items(df, n_lek=10, n_pes=15, seed=42)
    assert info["pes_mode"] == "random" and info["n"]["pes"] == 15


def test_sample_items_caps_at_available_questions():
    items, info = sample_items(make_nask_frame(n_per_edition=10), n_lek=300, n_pes=200, seed=42)
    assert info["n"]["lek"] == info["eligible_rows"][SPLIT_LEK] == 21
    assert info["n"]["pes"] == 14                                    # 2 edycje specjalistyczne × 7


# ─────────────────────────────────────────
# Wczytywanie plików / HF
# ─────────────────────────────────────────
def test_load_exam_file_split_resolution(tmp_path):
    df = make_nask_frame(n_per_edition=10)
    lek = df[df["split"] == SPLIT_LEK].drop(columns="split")
    lek.to_parquet(tmp_path / "pytania.parquet")
    loaded = load_exam_file(f"r_lek={tmp_path / 'pytania.parquet'}")
    assert set(loaded["split"]) == {SPLIT_LEK}
    pes = df[df["split"] == SPLIT_PES].drop(columns="split")
    pes.to_csv(tmp_path / "pes_2023.csv", index=False, encoding="utf-8")
    assert set(load_exam_file(tmp_path / "pes_2023.csv")["split"]) == {SPLIT_PES}
    mixed = df.drop(columns="split")
    mixed.to_csv(tmp_path / "mix.csv", index=False, encoding="utf-8")
    assert set(load_exam_file(tmp_path / "mix.csv")["split"]) == {SPLIT_LEK, SPLIT_PES}   # z exam_name
    df.drop(columns="answer").to_csv(tmp_path / "zly.csv", index=False, encoding="utf-8")
    with pytest.raises(ValueError, match="answer"):
        load_exam_file(tmp_path / "zly.csv")
    with pytest.raises(FileNotFoundError):
        load_exam_file(tmp_path / "brak.csv")


def test_describe_source_files_and_hf(tmp_path, monkeypatch):
    import hashlib

    from src.llm_eval.exams import HF_REPO, describe_source
    from src.llm_eval.report import _source_text

    path = tmp_path / "pytania.csv"
    make_nask_frame(n_per_edition=2).to_csv(path, index=False, encoding="utf-8")
    src = describe_source([f"r_lek={path}"])
    assert src == {"type": "file", "files": [{"file": "pytania.csv",
                                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}]}
    assert "SHA-256" in _source_text(src) and "pytania.csv" in _source_text(src)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)        # brak pakietu → rewizja nieznana
    hf = describe_source(None)
    assert hf == {"type": "hf", "repo": HF_REPO, "revision": None}
    assert "nie udało się ustalić" in _source_text(hf)


def test_hf_loader_without_datasets_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)       # import datasets → ImportError
    with pytest.raises(RuntimeError, match="requirements-app.txt"):
        load_hf_split("r_lek")


# ─────────────────────────────────────────
# Statystyka
# ─────────────────────────────────────────
def test_mcnemar_and_paired_bootstrap():
    a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0] * 10, dtype=bool)
    b = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0] * 10, dtype=bool)
    mc = mcnemar_exact(a, b)
    assert mc["a_only"] == 60 and mc["b_only"] == 0 and mc["p_mcnemar"] < 1e-10
    boot = paired_bootstrap_delta(a, b, n_boot=500, seed=1)
    assert boot["delta"] == pytest.approx(0.6) and 0.45 < boot["ci_low"] < 0.6 < boot["ci_high"] < 0.75
    same = mcnemar_exact(a, a)
    assert same["p_mcnemar"] == 1.0
    assert paired_bootstrap_delta(a, a, n_boot=100)["ci_low"] == 0.0


# ─────────────────────────────────────────
# Cała część „exams” na atrapie serwera
# ─────────────────────────────────────────
def _setup_fake_models(srv: FakeOllama, df: pd.DataFrame) -> None:
    key = answer_key_from_frame(df)
    srv.add_model("fake-a", exam_oracle(key, p_correct=0.95, p_correct_abstain=0.9, seed=1, thinking=True))
    srv.add_model("fake-b", exam_oracle(key, p_correct=0.6, p_correct_abstain=0.25, seed=2, p_unparseable=0.1,
                                        style="answer"))


def test_run_exams_part_metrics_and_cache(srv, tmp_path):
    df = make_nask_frame()
    _setup_fake_models(srv, df)
    srv.fail_next(503, times=2)                    # chwilowe przeciążenie — ponowienia
    client = OllamaChat(srv.base_url, backoff_base_s=0, backoff_max_s=0, timeout_s=10)
    cache = ResponseCache(tmp_path / "cache.jsonl")
    res = run_exams_part(client, ["fake-a", "fake-b"], out_dir=tmp_path / "out", cache=cache, n_lek=30, n_pes=20,
                         source=df, n_boot=300)
    rows = {(m["model"], m["protocol"], m["split"]): m for m in res["metrics"]}
    a = rows[("fake-a", "abstaining_substitution", "all")]
    b = rows[("fake-b", "abstaining_substitution", "all")]
    assert a["n"] == b["n"] == 50 and a["ci_low"] < a["accuracy"] < a["ci_high"]
    assert a["accuracy"] > b["accuracy"]
    assert b["parse_failures"] > 0 and a["parse_failures"] == 0 and a["api_errors"] == 0
    assert rows[("fake-a", "standard", "r_lek")]["n"] == 30
    paired = res["paired_vs_best"]["abstaining_substitution"]
    assert paired["best"] == "fake-a" and paired["rows"][0]["model"] == "fake-b"
    assert paired["rows"][0]["delta"] < 0 and paired["rows"][0]["p_mcnemar"] < 0.05
    assert {g["model"] for g in res["gap"]} == {"fake-a", "fake-b"}
    for name in ("exam_items.csv", "exam_responses.csv", "exam_metrics.csv"):
        assert (tmp_path / "out" / name).exists()
    resp = pd.read_csv(tmp_path / "out" / "exam_responses.csv", encoding="utf-8")
    assert len(resp) == 2 * 2 * 50 and not resp["cached"].any()
    # te same pytania dla obu modeli i protokołów
    assert resp.groupby(["model", "protocol"])["item_id"].apply(lambda s: tuple(sorted(s))).nunique() == 1
    # system prompt NASK + podmiana w abstaining
    bodies = srv.chat_requests("fake-a")
    assert bodies[0]["messages"][0]["content"].endswith("Prawidłowa odpowiedź to B.") and bodies[0]["think"] == "low"
    assert any(ABSTAIN_TEXT in b["messages"][1]["content"] for b in bodies)

    # Drugi przebieg: wszystko z pamięci podręcznej, zero zapytań
    srv.reset_log()
    res2 = run_exams_part(client, ["fake-a", "fake-b"], out_dir=tmp_path / "out2",
                          cache=ResponseCache(tmp_path / "cache.jsonl"), n_lek=30, n_pes=20, source=df, n_boot=300)
    assert srv.chat_count() == 0
    assert res2["metrics"] == res["metrics"] or [m["accuracy"] for m in res2["metrics"]] == \
        [m["accuracy"] for m in res["metrics"]]


def test_api_errors_count_as_wrong_and_are_retried_next_run(srv, tmp_path):
    df = make_nask_frame(n_per_edition=10)
    key = answer_key_from_frame(df)
    srv.add_model("ok", exam_oracle(key, p_correct=1.0))
    srv.add_model("pada", Reply(status=500, error="model crashed"))
    client = OllamaChat(srv.base_url, backoff_base_s=0, backoff_max_s=0, max_retries=0,
                        max_consecutive_failures=0)
    cache = ResponseCache(tmp_path / "c.jsonl")
    res = run_exams_part(client, ["ok", "pada"], out_dir=tmp_path / "o", cache=cache, n_lek=5, n_pes=3,
                         source=df, n_boot=100)
    rows = {(m["model"], m["protocol"], m["split"]): m for m in res["metrics"]}
    assert rows[("ok", "standard", "all")]["accuracy"] == 1.0
    bad = rows[("pada", "standard", "all")]
    assert bad["accuracy"] == 0.0 and bad["api_errors"] == 8 and bad["parse_failures"] == 0
    srv.add_model("pada", exam_oracle(key, p_correct=1.0))
    srv.reset_log()
    run_exams_part(client, ["ok", "pada"], out_dir=tmp_path / "o2", cache=ResponseCache(tmp_path / "c.jsonl"),
                   n_lek=5, n_pes=3, source=df, n_boot=100)
    assert srv.chat_count("ok") == 0 and srv.chat_count("pada") == 16


# ─────────────────────────────────────────
# CLI scripts/12_llm_bakeoff.py
# ─────────────────────────────────────────
def _load_cli():
    spec = importlib.util.spec_from_file_location("llm_bakeoff_script", PROJECT_ROOT / "scripts" / "12_llm_bakeoff.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli(monkeypatch, tmp_path):
    from src.utils import config

    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    return _load_cli()


def test_cli_run_exams_end_to_end_and_rerun_uses_cache(cli, srv, nask_csv, tmp_path):
    _setup_fake_models(srv, pd.read_csv(nask_csv, encoding="utf-8"))
    srv.version = "0.31.0-fake"
    argv = ["run", "--parts", "exams", "--models", "fake-a,fake-b", "--base-url", srv.base_url,
            "--exam-file", str(nask_csv), "--n-exam", "30", "--n-pes", "20", "--n-boot", "300",
            "--out", str(tmp_path / "run1"), "--cache", str(tmp_path / "cache.jsonl")]
    assert cli.main(argv) == 0
    out = tmp_path / "run1"
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Clopper–Pearsona" in report and "McNemar" in report and "## Wybór modelu" in report
    assert "Wybrany model: `fake-a`" in report and "0.31.0-fake" in report
    assert "%" in report and "[" in report                                  # trafność z przedziałem
    assert "Źródło: plik(i) `nask_synthetic.csv` (SHA-256" in report
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["selection"]["winner"] == "fake-a"
    assert "_correctness" not in results["parts"]["exams"]
    assert results["parts"]["exams"]["sample"]["n"] == {"lek": 30, "pes": 20, "all": 50}
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert info["server"]["version"] == "0.31.0-fake" and "fake-a" in info["model_info"]
    raw = (out / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw) == 200
    assert (out / "bakeoff.log").exists()
    first_calls = srv.chat_count()
    assert first_calls == 200 + 2                                            # + test łączności

    srv.reset_log()
    assert cli.main(argv[:-4] + ["--out", str(tmp_path / "run2"), "--cache", str(tmp_path / "cache.jsonl")]) == 0
    assert srv.chat_count() == 2                                     # tylko test łączności
    report2 = (tmp_path / "run2" / "report.md").read_text(encoding="utf-8")
    assert "Wybrany model: `fake-a`" in report2
    rows2 = pd.read_csv(tmp_path / "run2" / "exam_responses.csv", encoding="utf-8")
    assert rows2["cached"].all()


def test_cli_dry_run_sends_no_chat(cli, srv, nask_csv, tmp_path, capsys):
    srv.add_model("*", "ok")
    rc = cli.main(["run", "--dry-run", "--parts", "exams", "--models", "glm-5.3:cloud,kimi-k3:cloud",
                   "--base-url", srv.base_url, "--exam-file", str(nask_csv), "--n-exam", "20",
                   "--n-pes", "10", "--cache", str(tmp_path / "c.jsonl")])
    assert rc == 0 and srv.chat_count() == 0
    text = capsys.readouterr().out
    assert "PLAN" in text and "kimi-k3:cloud" in text and "$" in text and "20 LEK + 10 PES" in text


def test_cli_rejects_cloud_without_key_and_bad_args(cli, tmp_path):
    assert cli.main(["run", "--base-url", "https://ollama.com", "--parts", "exams"]) == 2
    assert cli.main(["run", "--parts", "exams,zle"]) == 2
    assert cli.main(["run", "--think", "bardzo"]) == 2


def test_cli_excludes_dead_model_after_preflight(cli, srv, nask_csv, tmp_path):
    _setup_fake_models(srv, pd.read_csv(nask_csv, encoding="utf-8"))
    rc = cli.main(["run", "--parts", "exams", "--models", "fake-a,nie-ma", "--base-url", srv.base_url,
                   "--exam-file", str(nask_csv), "--n-exam", "10", "--n-pes", "5", "--n-boot", "100",
                   "--max-retries", "0", "--out", str(tmp_path / "r"), "--cache", str(tmp_path / "c.jsonl")])
    assert rc == 0
    assert srv.chat_count("nie-ma") == 1
    report = (tmp_path / "r" / "report.md").read_text(encoding="utf-8")
    assert "wyłączone po teście łączności" in report and "`nie-ma`" in report


def test_cli_part_plumbing_helpers(cli):
    fn, why = cli.load_entry("src.llm_eval.nie_istnieje", "run_x")
    assert fn is None and "nie istnieje" in why
    res, md = cli.unpack_part(({"a": 1}, "### Sekcja\n\ntekst"))
    assert res == {"a": 1} and md.startswith("### Sekcja")
    res, md = cli.unpack_part({"b": 2, "markdown": ["x", "y"]})
    assert res["b"] == 2 and md == "x\n\ny"
    res, md = cli.unpack_part({"sections": {"Tytuł": "treść"}})
    assert md == "### Tytuł\n\ntreść"

    def only_two(client, models):
        return {"client": client, "models": models}

    def with_kwargs(**kw):
        return sorted(kw)

    assert cli.call_with_context(only_two, {"client": 1, "models": [2], "out_dir": 3}) == {"client": 1, "models": [2]}
    assert cli.call_with_context(with_kwargs, {"x": 1, "y": 2}) == ["x", "y"]
    costs = cli.part_costs("esi", {"usage": {"m": {"calls": 2, "prompt_tokens": 1000, "completion_tokens": 10}}},
                           {}, {"m": {"input": 1.0, "output": 1.0}})
    assert costs[0]["cost_usd"] == pytest.approx(0.00101)


def test_cli_skips_missing_part_modules(cli, srv, nask_csv, tmp_path, monkeypatch):
    """Brak modułu części ESI/narzędzi → część pominięta z komunikatem, egzaminy liczą się dalej."""
    monkeypatch.setattr(cli, "PART_ENTRY", {"esi": ("src.llm_eval.brak_esi", "run_esi_part"),
                                            "tools": ("src.llm_eval.brak_tools", "run_tools_part")})
    _setup_fake_models(srv, pd.read_csv(nask_csv, encoding="utf-8"))
    rc = cli.main(["run", "--parts", "exams,esi,tools", "--models", "fake-a,fake-b", "--base-url",
                   srv.base_url, "--exam-file", str(nask_csv), "--n-exam", "10", "--n-pes", "5",
                   "--n-boot", "100", "--out", str(tmp_path / "r"), "--cache", str(tmp_path / "c.jsonl")])
    assert rc == 0
    report = (tmp_path / "r" / "report.md").read_text(encoding="utf-8")
    assert "Część pominięta: moduł src.llm_eval.brak_esi nie istnieje" in report


def test_cli_assembles_external_part_results(cli, srv, nask_csv, tmp_path, monkeypatch):
    """Część ESI zwraca (wyniki, markdown) — raport ją wstawia, a ranking_score rozstrzyga remis."""
    import types

    fake_mod = types.ModuleType("src.llm_eval._test_esi_part")
    calls = {}

    def run_esi_part(client, models, out_dir, n_cases, think):
        calls.update(models=list(models), n_cases=n_cases, think=think)
        res = client.chat(models[0], [{"role": "user", "content": "x"}])
        return {"ranking_score": {m: float(i) for i, m in enumerate(models)}, "ok": res.ok}, "### Wyniki ESI\n\nTabela."

    fake_mod.run_esi_part = run_esi_part
    monkeypatch.setitem(sys.modules, "src.llm_eval._test_esi_part", fake_mod)
    monkeypatch.setattr(cli, "PART_ENTRY", {"esi": ("src.llm_eval._test_esi_part", "run_esi_part"),
                                            "tools": ("src.llm_eval.brak_tools", "run_tools_part")})
    srv.add_model("*", "ok")
    rc = cli.main(["run", "--parts", "esi", "--models", "m1,m2", "--base-url", srv.base_url,
                   "--n-cases", "7", "--out", str(tmp_path / "r"), "--cache", str(tmp_path / "c.jsonl")])
    assert rc == 0 and calls == {"models": ["m1", "m2"], "n_cases": 7, "think": "low"}
    report = (tmp_path / "r" / "report.md").read_text(encoding="utf-8")
    assert "### Wyniki ESI" in report and "Wybrany model: `m2`" in report
    results = json.loads((tmp_path / "r" / "results.json").read_text(encoding="utf-8"))
    assert results["costs"][0]["part"] == "esi" and results["costs"][0]["calls"] == 1


def test_cli_part_costs_from_part_summary(cli):
    res = {"summary": {"glm-5.3:cloud": {"n_items": 40, "prompt_tokens": 1_000_000, "completion_tokens": 0}}}
    rows = cli.part_costs("esi", res, {"x": {"calls": 5, "prompt_tokens": 1}}, cli_prices())
    assert rows == [{"part": "esi", "model": "glm-5.3:cloud", "calls": 40, "prompt_tokens": 1_000_000.0,
                     "completion_tokens": 0.0, "cost_usd": pytest.approx(1.40)}]
    assert cli.part_costs("esi", {}, {"x": {"calls": 5, "prompt_tokens": 1}}, cli_prices())[0]["model"] == "x"


def cli_prices():
    from src.llm_eval.pricing import load_prices

    return load_prices()


def test_cli_score_ratings_delegates_to_ratings_module(cli, tmp_path, monkeypatch, capsys):
    import types

    mod = types.ModuleType("src.llm_eval._test_ratings")
    seen = {}

    def score_ratings(csv_paths, key_path=None, n_boot=2000, seed=42):
        seen.update(paths=[p.name for p in csv_paths], key=key_path, seed=seed)
        return {"summary": {}, "markdown": "## Ślepa ocena\n\ntabela"}

    def write_scores(result, out_dir):
        seen["out"] = Path(out_dir).name
        return {"md": "x"}

    mod.score_ratings, mod.write_scores = score_ratings, write_scores
    monkeypatch.setitem(sys.modules, "src.llm_eval._test_ratings", mod)
    monkeypatch.setattr(cli, "RATINGS_ENTRY", (("src.llm_eval._test_ratings", "score_ratings"),))
    a, b = tmp_path / "ocena_1.csv", tmp_path / "ocena_2.csv"
    a.write_text("case_id,litera\n", encoding="utf-8")
    b.write_text("case_id,litera\n", encoding="utf-8")
    assert cli.main(["score-ratings", "--csv", str(a), "--csv", str(b)]) == 0
    assert seen == {"paths": ["ocena_1.csv", "ocena_2.csv"], "key": None, "seed": 42, "out": tmp_path.name}
    assert "Ślepa ocena" in capsys.readouterr().out
    assert cli.main(["score-ratings", "--csv", str(tmp_path / "brak.csv")]) == 2
