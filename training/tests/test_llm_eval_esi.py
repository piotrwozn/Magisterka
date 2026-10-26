"""
Testy części "esi" porównania LLM (src/llm_eval: cc_labels_pl, esi_cases, explain_task, ratings,
knowledge_notes) na syntetycznym Yale i atrapie Ollamy (tests/fake_ollama.py).

Środowisko (raz na sesję, `llm_esi_env`): syntetyczne splity → dwa prawdziwe runy (logreg +
lightgbm) → recipe.json w formacie 08 (reguła 'target' z siatką bezpieczeństwa i temperaturą) →
przypadki. Sprawdzamy m.in., że decyzja liczona na żywo z modeli (what_if bez zmian) jest
IDENTYCZNA z decyzją receptury na zapisanych predykcjach.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.llm_eval.cc_labels_pl import (
    CC_LABELS_PL,
    FEATURE_LABELS_PL,
    feature_label,
    feature_synonyms,
    fold_text,
    mentions,
)
from src.llm_eval.client import ChatResult, OllamaChat
from tests.conftest import _patch_paths
from tests.fake_ollama import FakeOllama, Reply

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_FEATURE_SETS = PROJECT_ROOT / "data" / "processed" / "esi" / "feature_sets.json"
FAST = {"logreg": {"max_iter": 300}, "lightgbm": {"n_estimators": 40, "num_leaves": 7, "min_child_samples": 5}}
N_CASES = 12
TARGET_SPEC = {"rule": "target", "cutpoints": [0.9, 1.7, 2.4, 3.1], "safety_q": 0.35}
_ENV_CACHE: dict[str, Any] = {}


# ─────────────────────────────────────────
# Środowisko: syntetyczne dane + runy + receptura + przypadki (raz na proces)
# ─────────────────────────────────────────
def _patch_logs(mp: pytest.MonkeyPatch, logs: Path) -> None:
    from src.models import base
    from src.utils import config

    mp.setattr(config, "LOGS_DIR", logs)
    mp.setattr(config, "EVAL_LOGS_DIR", logs / "evaluation")
    mp.setattr(config, "MODELS_DIR", logs / "models")
    mp.setattr(base, "MODELS_DIR", logs / "models")


def _build_env(prepared_esi_dir: Path, root: Path) -> SimpleNamespace:
    from src.data import dataset
    from src.ensemble import fit_temperature
    from src.ensemble import aligned_probas, blend, load_run
    from src.llm_eval.esi_cases import prepare_context
    from src.models.train import run_training

    data_dir = root / "data" / "esi"
    shutil.copytree(prepared_esi_dir, data_dir)
    with pytest.MonkeyPatch.context() as mp:
        paths = _patch_paths(mp, data_dir, root)
        _patch_logs(mp, root / "logs")
        names = [run_training(m, custom_params=FAST[m], tag="llm")["run_name"] for m in ("logreg", "lightgbm")]
        runs = [load_run(n, paths.RUNS_DIR) for n in names]
        weights = {"logreg": 0.4, "lightgbm": 0.6}
        y, probas = aligned_probas(runs, "val")
        temperature = fit_temperature(blend(probas, weights), y)
        report = paths.RUNS_DIR / "final_report__triage__llm"
        report.mkdir(parents=True)
        recipe = {
            "feature_set": "triage", "data_fingerprint": runs[0].info.get("data_fingerprint"),
            "runs": [{"name": r.name, "model": r.model, "path": str(r.path)} for r in runs],
            "weights": weights, "method": "mean", "temperature": temperature,
            "decision": {"main_rule": "target", "main_spec": TARGET_SPEC},
        }
        (report / "recipe.json").write_text(json.dumps(recipe, ensure_ascii=False), encoding="utf-8")
        ctx = prepare_context(recipe_path=report, n_cases=N_CASES, seed=42, runs_dir=paths.RUNS_DIR)
    dataset._feature_sets_payload.cache_clear()
    return SimpleNamespace(root=root, data_dir=data_dir, runs_dir=paths.RUNS_DIR, recipe_dir=report,
                           run_names=names, ctx=ctx)


@pytest.fixture(scope="session")
def llm_esi_env(prepared_esi_dir, tmp_path_factory) -> SimpleNamespace:
    """Runy, receptura i przypadki na syntetycznych danych (wspólne dla testów esi i tools)."""
    if "env" not in _ENV_CACHE:
        _ENV_CACHE["env"] = _build_env(prepared_esi_dir, tmp_path_factory.mktemp("llm_esi"))
    return _ENV_CACHE["env"]


@pytest.fixture
def env_paths(llm_esi_env, monkeypatch):
    """Ścieżki danych/runów środowiska aktywne w teście (np. ponowne budowanie receptury)."""
    from src.data import dataset

    paths = _patch_paths(monkeypatch, llm_esi_env.data_dir, llm_esi_env.root)
    yield paths
    dataset._feature_sets_payload.cache_clear()


# ─────────────────────────────────────────
# Atrapy modeli językowych
# ─────────────────────────────────────────
def _last_user(req: dict[str, Any]) -> str:
    return next(m["content"] for m in reversed(req["messages"]) if m.get("role") == "user")


def _first_user(req: dict[str, Any]) -> str:
    return next(m["content"] for m in req["messages"] if m.get("role") == "user")


def decision_from_prompt(text: str) -> int:
    return int(re.search(r"Decyzja systemu: ESI (\d)", text).group(1))


def good_responder(req: dict[str, Any]) -> Reply:
    """Poprawny model: narzędzie, poziom = decyzja, prawdziwe tętno, cytaty, po polsku."""
    prompt = _first_user(req)
    level = decision_from_prompt(prompt)
    hr = re.search(r"tętno (\d+)/min", prompt)
    hr_text = f"tętno {hr.group(1)}/min" if hr else "brak pomiaru tętna"
    shap = re.search(r"\n1\. (.+?) = ", prompt)
    factor = shap.group(1) if shap else "parametry życiowe"
    args = {
        "poziom_esi": level,
        "uzasadnienie": (f"System wyznaczył poziom ESI {level} [S2]. U pacjenta jest {hr_text} [S1], a "
                         f"najważniejszym czynnikiem decyzji jest {factor.lower()} [S3]. Reguły bezpieczeństwa "
                         f"nie wskazują na konieczność zmiany [S4]."),
        "kluczowe_czynniki": [factor],
        "sugestia_eskalacji": {"czy": False, "poziom": None, "powod": None},
    }
    return Reply(tool_calls=[{"name": "submit_explanation", "arguments": args}], prompt_tokens=1000,
                 completion_tokens=200)


SLOPPY_JSON = ('```json\n{"poziom_esi": "ESI 5", "uzasadnienie": "The patient has HR 250/min and the '
               'decision is fine [S9].", "kluczowe_czynniki": "tachycardia", "sugestia_eskalacji": '
               '{"czy": true, "poziom": 5, "powod": "x"}}\n```')


def sloppy_responder(req: dict[str, Any]) -> Reply:
    """Słaby model: najpierw tekst bez narzędzia, po ponowieniu JSON w treści (angielski, zmyślone tętno)."""
    if "Twoja odpowiedź nie była poprawna" in _last_user(req):
        return Reply(content=SLOPPY_JSON, prompt_tokens=1200, completion_tokens=80)
    return Reply(content="Nie wiem, trudno powiedzieć.", prompt_tokens=1000, completion_tokens=10)


# ─────────────────────────────────────────
# Etykiety
# ─────────────────────────────────────────
class TestLabels:
    def test_every_complaint_and_feature_has_polish_label(self, prepared_esi_dir):
        from src.features.engineering import ENGINEERED_FEATURES

        payload = json.loads((prepared_esi_dir / "feature_sets.json").read_text(encoding="utf-8"))
        for name in payload["feature_sets"]["triage"]:
            assert feature_label(name) != name, f"brak polskiej nazwy cechy {name}"
            # Polska nazwa cechy zawsze „wspomina” cechę (spójność etykiet i synonimów pokrycia SHAP)
            assert mentions(feature_label(name), feature_synonyms(name)), name
        assert all(f in FEATURE_LABELS_PL for f in ENGINEERED_FEATURES)
        assert len(CC_LABELS_PL) == 199 and all(v.strip() for v in CC_LABELS_PL.values())

    @pytest.mark.skipif(not REAL_FEATURE_SETS.exists(), reason="brak przygotowanych danych Yale")
    def test_all_real_chief_complaints_have_labels(self):
        payload = json.loads(REAL_FEATURE_SETS.read_text(encoding="utf-8"))
        complaints = set(payload["groups"]["chief_complaints"])
        assert complaints == set(CC_LABELS_PL)
        for fs in payload["feature_sets"].values():
            assert not [c for c in fs if c.startswith("cc_") and feature_label(c) == c]
        assert not [c for c in payload["feature_sets"]["triage"]
                    if not mentions(feature_label(c), feature_synonyms(c))]

    def test_mentions_and_folding(self):
        assert fold_text("Ból w KLATCE piersiowej") == "bol w klatce piersiowej"
        assert mentions("Pacjentka zgłasza ból w klatce piersiowej.", feature_synonyms("cc_chestpain"))
        assert mentions("Saturacja 91% [S1]", feature_synonyms("triage_vital_o2"))
        assert not mentions("chrapanie w nocy", ["hr"])                      # rdzeń od początku słowa
        assert mentions("wysokie ciśnienie tętnicze", feature_synonyms("triage_vital_sbp"))
        assert feature_label("age & triage_vital_hr") == "Wiek × Tętno (HR)"


# ─────────────────────────────────────────
# Przypadki i receptura
# ─────────────────────────────────────────
class TestCases:
    def test_allocate_strata(self):
        from src.llm_eval.esi_cases import allocate_strata

        full = {1: 100, 2: 100, 3: 100, 4: 100, 5: 100}
        assert allocate_strata(40, full) == {1: 8, 2: 12, 3: 10, 4: 6, 5: 4}
        assert sum(allocate_strata(10, full).values()) == 10
        short = allocate_strata(40, {1: 3, 2: 100, 3: 100, 4: 100, 5: 100})
        # Brak ESI 1 → niedobór rozdzielany po jednym wg REFILL_ORDER (najpierw ESI 2)
        assert short == {1: 3, 2: 14, 3: 11, 4: 7, 5: 5}
        assert sum(allocate_strata(40, {1: 2, 2: 3, 3: 1, 4: 0, 5: 0}).values()) == 6

    def test_cases_from_recipe(self, llm_esi_env):
        ctx = llm_esi_env.ctx
        assert len(ctx.cases) == N_CASES
        assert ctx.recipe.spec == {**TARGET_SPEC, "cutpoints": [float(c) for c in TARGET_SPEC["cutpoints"]]}
        assert [c.case_id for c in ctx.cases] == [f"C{i:02d}" for i in range(1, N_CASES + 1)]
        val = pd.read_parquet(llm_esi_env.data_dir / "val.parquet", columns=["row_id", "vitals_complete", "esi"])
        val = val.set_index("row_id")
        for case in ctx.cases:
            assert val.loc[case.row_id, "vitals_complete"] == 1
            assert val.loc[case.row_id, "esi"] == case.nurse_esi
            assert case.decision_check["stored_esi"] == case.decision_check["live_esi"] == case.decision_esi
            assert case.decision_check["max_abs_proba_diff"] < 1e-6            # ≤ ulp float32 (pojedynczy wiersz)
            assert abs(sum(case.probabilities.values()) - 1) < 1e-9
            assert 1 <= len(case.shap_top) <= 5 and all(f["label_pl"] for f in case.shap_top)
            assert case.shap_model == "lightgbm"                      # model drzewiasty receptury
            assert "Parametry życiowe przy triażu" in case.description_pl and "tętno" in case.description_pl
            sources = case.sources(ctx.recipe_text)
            assert set(sources) == {"S1", "S2", "S3", "S4"}
            assert f"Decyzja systemu: ESI {case.decision_esi}" in sources["S2"]
            # ESI z triażu nie trafia do żadnego źródła (ślepa ocena)
            assert "pielęgniar" not in " ".join(sources.values()).lower()
        assert ctx.summary()["live_vs_stored_mismatches"] == []

    def test_stratified_selection_is_deterministic(self, llm_esi_env, env_paths):
        from src.llm_eval.esi_cases import prepare_context

        again = prepare_context(recipe_path=llm_esi_env.recipe_dir, n_cases=N_CASES, seed=42,
                                runs_dir=env_paths.RUNS_DIR)
        assert [c.row_id for c in again.cases] == [c.row_id for c in llm_esi_env.ctx.cases]
        other = prepare_context(recipe_path=llm_esi_env.recipe_dir, n_cases=N_CASES, seed=7,
                                runs_dir=env_paths.RUNS_DIR)
        assert [c.row_id for c in other.cases] != [c.row_id for c in llm_esi_env.ctx.cases]

    def test_live_recipe_equals_stored_decisions_on_whole_val(self, llm_esi_env, env_paths):
        """Odtworzenie receptury z modeli na WSZYSTKICH wierszach VAL: te same decyzje i cechy."""
        from src.data.dataset import load_split
        from src.llm_eval.esi_cases import base_columns, stored_decisions

        ctx = llm_esi_env.ctx
        stored = stored_decisions(ctx.recipe)
        cols = base_columns(ctx.predictor)
        val = load_split("val").set_index("row_id").loc[stored.index]
        X = ctx.predictor.features(val[cols].astype(np.float32))
        # Cechy inżynieryjne przeliczone z wiersza kanonicznego = zapisane w splitach
        pd.testing.assert_frame_equal(X, val[ctx.predictor.feature_names].astype(np.float32))
        proba, pred = ctx.predictor.predict_features(X)
        assert np.array_equal(pred, stored["decision"].to_numpy())
        p_cols = [f"p_esi{k}" for k in range(1, 6)]
        np.testing.assert_allclose(proba, stored[p_cols].to_numpy(), rtol=0, atol=1e-12)

    def test_recipe_from_runs_and_skip_without_runs(self, llm_esi_env, env_paths, tmp_path):
        from src.llm_eval.esi_cases import CasesUnavailable, recipe_from_runs, try_prepare_context

        recipe = recipe_from_runs(runs_dir=env_paths.RUNS_DIR, require_tuned=False)
        assert set(recipe.weights) == {"logreg", "lightgbm"} and abs(sum(recipe.weights.values()) - 1) < 1e-3
        assert recipe.temperature is not None and recipe.spec["rule"] in ("target", "cost")
        with pytest.raises(CasesUnavailable, match="po tuningu"):
            recipe_from_runs(runs_dir=env_paths.RUNS_DIR)            # runy testowe są bez tuningu
        ctx, reason = try_prepare_context(runs_dir=tmp_path / "pusty", n_cases=5)
        assert ctx is None and "Brak ukończonych runów" in reason
        ctx, reason = try_prepare_context(recipe_path=tmp_path / "brak", n_cases=5)
        assert ctx is None and "receptury" in reason


# ─────────────────────────────────────────
# Parsowanie i metryki
# ─────────────────────────────────────────
def _case(**overrides: Any):
    from src.llm_eval.esi_cases import EsiCase

    base = dict(
        case_id="C01", row_id=1, nurse_esi=2, decision_esi=3,
        probabilities={"ESI1": 0.02, "ESI2": 0.3, "ESI3": 0.5, "ESI4": 0.15, "ESI5": 0.03}, expected_esi=2.85,
        shap=[{"feature": "triage_vital_o2", "label_pl": "Saturacja (SpO2)", "value": 91.0, "value_pl": "91%",
               "shap": 0.4, "direction": "za"},
              {"feature": "cc_chestpain", "label_pl": "Ból w klatce piersiowej", "value": 1.0, "value_pl": "tak",
               "shap": 0.3, "direction": "za"},
              {"feature": "age", "label_pl": "Wiek", "value": 74.0, "value_pl": "74 lata", "shap": -0.1,
               "direction": "przeciw"}],
        shap_model="lightgbm", rules={"suggested_esi": 2, "decision_point": "D", "advisory": True,
                                      "violations": ["strefa zagrożenia: SpO2 < 92%"], "verdict": "x"},
        description_pl="Pacjent: kobieta, wiek 74 lata.",
        vitals={"hr": 112.0, "sbp": 155.0, "dbp": 97.0, "rr": 24.0, "o2": 91.0, "temp": None, "age": 74.0},
        complaints=["cc_chestpain"], base_row={},
    )
    base.update(overrides)
    return EsiCase(**base)


class TestParsing:
    def test_tool_call(self):
        from src.llm_eval.explain_task import parse_submission

        args = {"poziom_esi": 3, "uzasadnienie": "Tekst.", "kluczowe_czynniki": ["a"],
                "sugestia_eskalacji": {"czy": False, "poziom": None, "powod": None}}
        expl, errors, via, strict = parse_submission(ChatResult(tool_calls=[{"name": "submit_explanation",
                                                                            "arguments": args}]))
        assert expl["poziom_esi"] == 3 and via == "tool" and strict and errors == []

    def test_arguments_as_text_and_content_json(self):
        from src.llm_eval.explain_task import parse_submission

        raw = '{"poziom_esi": "2", "uzasadnienie": "Ok", "kluczowe_czynniki": "SpO2", "sugestia_eskalacji": {"czy": "tak", "poziom": 1, "powod": "SpO2"}}'
        expl, _, via, strict = parse_submission(ChatResult(tool_calls=[{"name": "submit_explanation", "arguments": {},
                                                                        "arguments_raw": raw}]))
        assert expl["poziom_esi"] == 2 and expl["kluczowe_czynniki"] == ["SpO2"] and not strict
        assert expl["sugestia_eskalacji"] == {"czy": True, "poziom": 1, "powod": "SpO2"}
        expl, _, via, _ = parse_submission(ChatResult(content="Oto odpowiedź:\n" + SLOPPY_JSON))
        assert via == "content" and expl["poziom_esi"] == 5
        wrapped = json.dumps({"name": "submit_explanation", "arguments": json.loads(raw)})
        assert parse_submission(ChatResult(content=wrapped))[0]["poziom_esi"] == 2

    def test_invalid_outputs(self):
        from src.llm_eval.explain_task import parse_submission

        expl, errors, via, _ = parse_submission(ChatResult(content="Nie wiem."))
        assert expl is None and via is None and "brak wywołania" in errors[0]
        expl, errors, _, _ = parse_submission(ChatResult(content='{"poziom_esi": 7, "uzasadnienie": ""}'))
        assert expl is None and any("poziom_esi" in e for e in errors) and any("uzasadnienie" in e for e in errors)
        expl, errors, _, _ = parse_submission(ChatResult(tool_calls=[{"name": "submit", "arguments": {}}]))
        assert expl is None and "nieznane narzędzie" in errors[0]


class TestMetrics:
    def test_numeric_faithfulness(self):
        from src.llm_eval.explain_task import numeric_faithfulness

        case = _case()
        ok = ("Pacjentka 74-letnia, tętno 112/min, ciśnienie tętnicze 155/97 mmHg, oddechy 24/min, SpO2 91% [S1]. "
              "Prawdopodobieństwo ESI 3 wynosi 50,0% [S2]. Strefa zagrożenia: tętno powyżej 100/min, SpO2 < 92%.")
        res = numeric_faithfulness(ok, case)
        assert res["wrong"] == 0 and res["checked"] >= 6, res
        bad = "Tętno 130/min, saturacja 85%, temperatura 38,5 °C, pacjentka w wieku 60 lat."
        res = numeric_faithfulness(bad, case)
        assert res["wrong"] == 4, res                          # temperatury nie mierzono → każda wartość to błąd
        assert numeric_faithfulness("Średnie ciśnienie tętnicze 116 mmHg, NEWS2 7 pkt.", case)["wrong"] == 0
        assert numeric_faithfulness("RR 155/97, RR 24/min.", case)["wrong"] == 0       # RR = ciśnienie albo oddechy
        hypo = "Gdyby SpO2 wynosiło 85%, decyzja byłaby inna."
        assert numeric_faithfulness(hypo, case)["wrong"] == 1
        assert numeric_faithfulness(hypo, case, {"o2": {85.0}})["wrong"] == 0

    def test_coverage_language_citations_escalation(self):
        from src.llm_eval.explain_task import citations, escalation_outcome, polish_check, shap_coverage

        case = _case()
        cov, hit = shap_coverage("Niska saturacja i ból w klatce piersiowej.", case)
        assert hit == ["triage_vital_o2", "cc_chestpain"] and abs(cov - 2 / 3) < 1e-9
        assert polish_check("Pacjent ma niską saturację, więc system wskazuje poziom ESI 3 i to jest zgodne.")["is_polish"]
        assert not polish_check("The patient has low saturation and the decision is consistent with the rules.")["is_polish"]
        assert citations("Zob. [S1], [S2, S3] oraz [S9] i [T1]", {"S1", "S2", "S3"}) == {"valid": ["S1", "S2", "S3"],
                                                                                    "invalid": ["S9", "T1"]}
        esc = lambda czy, poziom: {"sugestia_eskalacji": {"czy": czy, "poziom": poziom, "powod": None}}  # noqa: E731
        assert escalation_outcome(esc(False, None), case) == "none"
        assert escalation_outcome(esc(True, 2), case) == "useful"                    # pielęgniarka: ESI 2 < 3
        assert escalation_outcome(esc(True, 2), _case(nurse_esi=4)) == "false_alarm"
        assert escalation_outcome(esc(True, 4), case) == "wrong_direction"           # niższy priorytet
        assert escalation_outcome(esc(True, None), case) == "wrong_direction"


# ─────────────────────────────────────────
# Część esi z atrapą Ollamy
# ─────────────────────────────────────────
class TestEsiPart:
    def test_end_to_end_with_fake_server(self, llm_esi_env, tmp_path):
        from src.llm_eval.cache import ResponseCache
        from src.llm_eval.explain_task import run_esi_part

        ctx = llm_esi_env.ctx
        with FakeOllama() as srv:
            srv.add_model("good:cloud", good_responder)
            srv.add_model("sloppy:cloud", sloppy_responder)
            srv.fail_next(503, times=1, model="good:cloud")              # chwilowa awaria → ponowienie klienta
            client = OllamaChat(base_url=srv.base_url, max_retries=2, backoff_base_s=0.0, concurrency=2)
            cache = ResponseCache(tmp_path / "cache.jsonl")
            out = tmp_path / "out"
            res = run_esi_part(client, ["good:cloud", "sloppy:cloud"], out_dir=out, context=ctx, think="low",
                               cache=cache, prices={"good": {"input": 1.0, "output": 2.0}})
            n_calls = srv.chat_count()
            body = srv.chat_requests("good:cloud")[-1]
            assert body["think"] == "low" and body["tools"][0]["function"]["name"] == "submit_explanation"
            assert "format" not in body
            # Drugie uruchomienie: wszystko z pamięci podręcznej (wznowienie bez kosztów)
            again = run_esi_part(client, ["good:cloud", "sloppy:cloud"], out_dir=tmp_path / "out2", context=ctx,
                                 think="low", cache=ResponseCache(tmp_path / "cache.jsonl"))
            assert srv.chat_count() == n_calls
        assert n_calls == N_CASES + 1 + 2 * N_CASES                    # good: 1 (+503), sloppy: 2 na przypadek

        good, sloppy = res["summary"]["good:cloud"], res["summary"]["sloppy:cloud"]
        assert good["valid_rate"] == 1.0 and good["strict_rate"] == 1.0 and good["retry_rate"] == 0.0
        assert good["level_match_rate"] == 1.0 and good["hallucination_rate"] == 0.0
        assert good["polish_rate"] == 1.0 and good["citation_rate"] == 1.0 and good["invalid_citations"] == 0
        assert good["shap_coverage"] > 0 and good["esc_suggestions"] == 0
        assert good["esc_missed"] == good["esc_opportunities"]
        assert good["prompt_tokens"] == 1000 * N_CASES and abs(good["cost_usd"] - (1000 * N_CASES + 400 * N_CASES) / 1e6) < 1e-12
        assert sloppy["valid_rate"] == 1.0 and sloppy["retry_rate"] == 1.0 and sloppy["strict_rate"] == 0.0
        n5 = sum(c.decision_esi == 5 for c in ctx.cases)
        assert sloppy["level_match_rate"] == n5 / N_CASES
        assert sloppy["hallucination_rate"] == 1.0 and sloppy["polish_rate"] == 0.0
        assert sloppy["esc_wrong_direction"] == N_CASES and sloppy["invalid_citations"] == N_CASES
        assert sloppy["cost_usd"] is None                               # brak ceny dla modelu
        assert again["summary"]["good:cloud"]["valid_rate"] == 1.0

        files = res["files"]
        for key in ("cases", "items_csv", "summary_csv", "raw", "ratings_html", "ratings_template", "ratings_key",
                    "section_md"):
            assert Path(files[key]).exists(), key
        assert Path(files["ratings_key"]).parent == out and Path(files["ratings_html"]).parent == out / "ratings"
        items = pd.read_csv(files["items_csv"], encoding="utf-8")
        assert len(items) == 2 * N_CASES and set(items["via"]) == {"tool", "content"}
        raw_lines = (out / "raw_esi.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(raw_lines) == 3 * N_CASES and all("raw" not in json.loads(l)["result"] for l in raw_lines)
        md = res["markdown"]
        assert "## Część ESI" in md and "`good:cloud`" in md and "| Model |" in md
        assert json.loads(Path(files["cases"]).read_text(encoding="utf-8"))["summary"]["n_cases"] == N_CASES

    def test_requests_interleave_models_case_by_case(self, llm_esi_env, tmp_path):
        """Kolejność pracy przypadek → model: limit / awaria chmury pod koniec rozkłada się na wszystkie modele."""
        from src.llm_eval.explain_task import run_esi_part

        with FakeOllama() as srv:
            srv.add_model("good:cloud", good_responder)
            srv.add_model("other:cloud", good_responder)
            client = OllamaChat(base_url=srv.base_url, max_retries=0, backoff_base_s=0.0, concurrency=1)
            res = run_esi_part(client, ["good:cloud", "other:cloud"], out_dir=tmp_path, context=llm_esi_env.ctx,
                               write_ratings=False)
            sequence = [b["model"] for b in srv.chat_requests()]
        assert sequence[:4] == ["good:cloud", "other:cloud"] * 2
        assert [it["model"] for it in res["items"][:2]] == ["good:cloud", "other:cloud"]
        assert res["items"][0]["case_id"] == res["items"][1]["case_id"]

    def test_api_errors_are_reported_not_raised(self, llm_esi_env, tmp_path):
        from src.llm_eval.explain_task import run_esi_part

        with FakeOllama() as srv:
            srv.add_model("down", Reply(status=500, error="boom"))
            client = OllamaChat(base_url=srv.base_url, max_retries=0, concurrency=3)
            res = run_esi_part(client, ["down"], out_dir=tmp_path, context=llm_esi_env.ctx, write_ratings=False)
        s = res["summary"]["down"]
        assert s["valid_rate"] == 0.0 and s["api_errors"] == N_CASES and s["level_match_rate"] is None

    def test_skipped_without_runs(self, tmp_path):
        from src.llm_eval.explain_task import run_esi_part

        res = run_esi_part(OllamaChat(), ["x"], out_dir=tmp_path, runs_dir=tmp_path / "brak-runow", n_cases=3)
        assert res["skipped"] and "Pominięto" in res["markdown"] and res["summary"] == {}


# ─────────────────────────────────────────
# Arkusz ocen
# ─────────────────────────────────────────
class TestRatings:
    def test_sheet_and_scores_roundtrip(self, llm_esi_env, tmp_path):
        from src.llm_eval.ratings import TEMPLATE_COLUMNS, assign_letters, score_ratings, write_rating_sheet, write_scores

        ctx = llm_esi_env.ctx
        models = ["m-good", "m-bad", "m-mid"]
        items = [{"model": m, "case_id": c.case_id, "explanation": None if m == "m-bad" else
                  {"poziom_esi": c.decision_esi, "uzasadnienie": f"Wyjaśnienie <b>{m}</b>", "kluczowe_czynniki": ["x"],
                   "sugestia_eskalacji": {"czy": False, "poziom": None, "powod": None}}}
                 for m in models for c in ctx.cases]
        files = write_rating_sheet(tmp_path, ctx, items, models, seed=3)
        html = Path(files["ratings_html"]).read_text(encoding="utf-8")
        assert "&lt;b&gt;m-good&lt;/b&gt;" in html and "<b>m-good</b>" not in html     # tekst modeli escapowany
        assert "m-good" not in html.replace("&lt;b&gt;m-good&lt;/b&gt;", "")             # nazwy modeli ukryte
        key = json.loads(Path(files["ratings_key"]).read_text(encoding="utf-8"))
        assert key["cases"] == assign_letters([c.case_id for c in ctx.cases], models, seed=3)
        assert len({tuple(v.values()) for v in key["cases"].values()}) > 1               # różne permutacje

        template = pd.read_csv(files["ratings_template"], dtype=str, keep_default_na=False)
        assert list(template.columns) == TEMPLATE_COLUMNS and len(template) == len(models) * N_CASES
        score = {"m-good": "5", "m-mid": "3", "m-bad": "1"}
        filled = template.copy()
        filled["oceniajacy"] = "AB"
        model_of = [key["cases"][c][letter] for c, letter in zip(filled["case_id"], filled["litera"])]
        for col in ("zgodnosc_kliniczna", "przydatnosc", "jasnosc"):
            filled[col] = [score[m] for m in model_of]
        filled["halucynacja"] = ["tak" if m == "m-bad" else "0" for m in model_of]
        filled.loc[0, ["zgodnosc_kliniczna", "przydatnosc", "jasnosc", "halucynacja"]] = ["", "", "", "0"]  # nieoceniony
        extra = pd.DataFrame([{**{c: "" for c in TEMPLATE_COLUMNS}, "case_id": "C99", "litera": "A",
                               "zgodnosc_kliniczna": "4"}])
        csv_path = tmp_path / "ratings" / "oceny_AB.csv"
        csv_path.write_text("﻿" + pd.concat([filled, extra]).to_csv(index=False), encoding="utf-8")

        result = score_ratings([csv_path], n_boot=200)                 # klucz znaleziony obok katalogu ratings/
        s = result["summary"]
        assert s["m-good"]["zgodnosc_kliniczna"] == 5.0 and s["m-mid"]["mean_all"] == 3.0
        assert s["m-bad"]["hallucination_rate"] == 1.0 and s["m-good"]["hallucination_rate"] == 0.0
        assert result["n_ratings"] == len(models) * N_CASES - 1 and result["raters"] == ["AB"]
        assert s["m-good"]["zgodnosc_kliniczna_ci"] == [5.0, 5.0]
        assert result["markdown"].index("`m-good`") < result["markdown"].index("`m-bad`")
        written = write_scores(result, tmp_path)
        assert Path(written["md"]).read_text(encoding="utf-8").startswith("## Ślepa ocena")

    def test_sheet_id_depends_on_content_and_template_has_single_line_breaks(self, llm_esi_env, tmp_path,
                                                                            monkeypatch):
        """Pilotaż i przebieg końcowy na tych samych danych → inny klucz localStorage; CSV bez \\r\\r\\n."""
        import os

        from src.llm_eval.ratings import write_rating_sheet

        ctx = llm_esi_env.ctx
        models = ["m1", "m2"]

        def items(text: str) -> list[dict]:
            return [{"model": m, "case_id": c.case_id,
                     "explanation": {"poziom_esi": c.decision_esi, "uzasadnienie": f"{text} {m}",
                                     "kluczowe_czynniki": [], "sugestia_eskalacji": {"czy": False}}}
                    for m in models for c in ctx.cases]

        monkeypatch.setattr(os, "linesep", "\r\n")                   # pandas: domyślny koniec linii jak na Windows
        pilot = write_rating_sheet(tmp_path / "pilot", ctx, items("pilotaż"), models, seed=42)
        final = write_rating_sheet(tmp_path / "final", ctx, items("wersja końcowa"), models, seed=42)
        same = write_rating_sheet(tmp_path / "again", ctx, items("wersja końcowa"), models, seed=42)
        ids = [json.loads(Path(f["ratings_key"]).read_text(encoding="utf-8"))["sheet_id"] for f in (pilot, final, same)]
        assert ids[0] != ids[1] and ids[1] == ids[2]                   # ten sam arkusz ponownie → oceny zostają
        assert f'data-sheet="{ids[1]}"' in Path(final["ratings_html"]).read_text(encoding="utf-8")
        data = Path(final["ratings_template"]).read_bytes()
        assert b"\r\r\n" not in data and data.count(b"\n") == 1 + len(models) * N_CASES
        if os.name != "nt":
            assert b"\r" not in data

    def test_excel_semicolon_cp1250_sheet(self, tmp_path):
        """Szablon zapisany w polskim Excelu (średniki, Windows-1250) też jest wczytywany."""
        from src.llm_eval.ratings import score_ratings

        key = {"cases": {"C01": {"A": "m1", "B": "m2"}, "C02": {"A": "m2", "B": "m1"}}}
        (tmp_path / "key.json").write_text(json.dumps(key), encoding="utf-8")
        text = ("oceniajacy;case_id;litera;zgodnosc_kliniczna;przydatnosc;jasnosc;halucynacja;komentarz\r\n"
                "Łukasz;C01;A;5;4;5;0;świetne\r\nŁukasz;C01;B;2;2;3;tak;zmyślone tętno\r\n"
                "Łukasz;C02;A;1;2;2;1;\r\nŁukasz;C02;B;4;4;4;0;\r\n")
        path = tmp_path / "oceny_excel.csv"
        path.write_bytes(text.encode("cp1250"))
        res = score_ratings([path], n_boot=100)
        assert res["raters"] == ["Łukasz"] and res["n_ratings"] == 4
        assert res["summary"]["m1"]["zgodnosc_kliniczna"] == 4.5 and res["summary"]["m2"]["hallucination_rate"] == 1.0

    def test_bad_csv_columns(self, tmp_path):
        from src.llm_eval.ratings import score_ratings

        (tmp_path / "key.json").write_text(json.dumps({"cases": {}}), encoding="utf-8")
        path = tmp_path / "x.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="brak kolumn"):
            score_ratings([path])


class TestKnowledge:
    def test_search_and_criteria(self):
        from src.llm_eval.knowledge_notes import KNOWLEDGE_NOTES, esi_criteria, search_knowledge

        assert len(KNOWLEDGE_NOTES) == 15 and len({n.id for n in KNOWLEDGE_NOTES}) == 15
        assert search_knowledge("saturacja i tlenoterapia")[0]["id"] == "W06"
        assert search_knowledge("wskaźnik wstrząsowy")[0]["id"] == "W02"
        assert search_knowledge("xyzzy qwerty") == []
        crit = esi_criteria(2)
        assert set(crit["punkty_decyzyjne"]) == {"B", "D"} and "10 min" in crit["opis"]
        with pytest.raises(ValueError):
            esi_criteria(6)


# ─────────────────────────────────────────
# Receptura z modelem fundamentalnym (TFM)
# ─────────────────────────────────────────
class _StubModel:
    """Model receptury jak BaseTriageModel (unwrap_model → `.model`), bez trenowania."""

    def __init__(self, raw: Any, pretrained: bool = False):
        self.model = raw
        self.feature_names = ["age"]
        self.is_fitted = True
        self.pretrained = pretrained
        self.keep_fitted_bags: int | str = 1


def _raw(kind: str) -> Any:
    if kind == "linear":
        return SimpleNamespace(coef_=np.zeros((5, 1)), intercept_=np.zeros(5))
    names = {"tree": "LGBMClassifier", "ebm": "ExplainableBoostingClassifier", "tfm": "TFMPredictor"}
    return type(names[kind], (), {"predict_proba": lambda self, X: None})()


class TestTfmMembers:
    @pytest.fixture
    def explainer(self, monkeypatch):
        from src.explain import shap_explainer

        chosen: list[Any] = []

        class FakeExplainer:
            def __init__(self, model):
                chosen.append(model)
                self.method = "fake"

            def fit(self):
                return self

        monkeypatch.setattr(shap_explainer, "SHAPTriageExplainer", FakeExplainer)
        return chosen

    @staticmethod
    def _predictor(members: dict[str, tuple[str, float]]) -> SimpleNamespace:
        models = {k: _StubModel(_raw(kind), pretrained=kind == "tfm") for k, (kind, _) in members.items()}
        return SimpleNamespace(models=models, weights={k: w for k, (_, w) in members.items()})

    def test_classical_recipe_keeps_tree_ebm_linear_preference(self, explainer):
        from src.llm_eval.esi_cases import ShapHelper

        assert ShapHelper(self._predictor({"logreg": ("linear", 0.6)})).model_key == "logreg"
        helper = ShapHelper(self._predictor({"logreg": ("linear", 0.5), "ebm": ("ebm", 0.5)}))
        assert helper.model_key == "ebm"

    def test_recipe_with_tfm_uses_tree_model_only(self, explainer):
        from src.llm_eval.esi_cases import ShapHelper

        helper = ShapHelper(self._predictor({"logreg": ("linear", 0.4), "lightgbm": ("tree", 0.2),
                                             "tabpfn": ("tfm", 0.4)}))
        assert helper.model_key == "lightgbm" and not getattr(explainer[-1], "pretrained", True)

    def test_recipe_with_tfm_and_no_tree_model_has_no_shap(self, explainer):
        from src.llm_eval.esi_cases import CasesUnavailable, ShapHelper

        for members in ({"tabpfn": ("tfm", 1.0)}, {"ebm": ("ebm", 0.7), "tabicl": ("tfm", 0.3)}):
            with pytest.raises(CasesUnavailable, match="model drzewiasty"):
                ShapHelper(self._predictor(members))
        assert explainer == []

    def test_recipe_predictor_keeps_all_tfm_bags_fitted(self, monkeypatch, tmp_path):
        import src.models as models_pkg
        from src.llm_eval.esi_cases import RecipePredictor

        loaded = {"lightgbm": _StubModel(_raw("tree")), "tabpfn": _StubModel(_raw("tfm"), pretrained=True)}
        runs = []
        for name in loaded:
            run_dir = tmp_path / f"run_{name}"
            run_dir.mkdir()
            (run_dir / "model.joblib").write_bytes(b"x")
            runs.append(SimpleNamespace(name=run_dir.name, path=run_dir))
        monkeypatch.setattr(models_pkg, "load_model", lambda path: loaded[Path(path).parent.name.split("_", 1)[1]])
        recipe = SimpleNamespace(keys=list(loaded), runs=runs, weights={"lightgbm": 0.6, "tabpfn": 0.4})
        predictor = RecipePredictor(recipe)
        assert predictor.models["tabpfn"].keep_fitted_bags == "all"      # bez ponownego dopasowania na wiersz
        assert predictor.models["lightgbm"].keep_fitted_bags == 1
