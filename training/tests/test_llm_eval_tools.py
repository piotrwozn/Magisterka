"""
Testy części "tools" porównania LLM (src/llm_eval/tools_task.py): what_if (dokładne odtworzenie
decyzji receptury dla niezmienionego wiersza, dozwolone zmiany, walidacja), narzędzia pomocnicze,
walidacja argumentów, pętla agenta z atrapą Ollamy (ukończenie, zmyślone narzędzia, limit kroków,
wznowienie z pamięci podręcznej) i metryki części.

Środowisko (runy + receptura + przypadki) współdzielone z tests/test_llm_eval_esi.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.llm_eval.client import OllamaChat
from tests.fake_ollama import FakeOllama, Reply
from tests.test_llm_eval_esi import N_CASES, decision_from_prompt, env_paths, llm_esi_env  # noqa: F401 (fixture)


def _first_user(req: dict[str, Any]) -> str:
    return next(m["content"] for m in req["messages"] if m.get("role") == "user")


def _tool_results(req: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(m["content"]) for m in req["messages"] if m.get("role") == "tool"]


def agent_responder(req: dict[str, Any]) -> Reply:
    """Dobry agent: krok 1 — explain_model + what_if; krok 2 — submit z czynnikiem SHAP i scenariuszem."""
    results = _tool_results(req)
    if not results:
        return Reply(tool_calls=[{"name": "explain_model", "arguments": {}},
                                 {"name": "what_if", "arguments": {"changes": {"triage_vital_o2": 85}}}])
    level = decision_from_prompt(_first_user(req))
    shap = next(r for r in results if "czynniki" in r)
    wi = next(r for r in results if "poziom_esi_po" in r)
    factor = shap["czynniki"][0]["cecha"]
    text = (f"System wyznaczył poziom ESI {level} [S2]. Najważniejszym czynnikiem jest {factor.lower()} "
            f"[{shap['id_zrodla']}]. Gdyby SpO2 wynosiło 85%, poziom wyniósłby ESI {wi['poziom_esi_po']} "
            f"[{wi['id_zrodla']}], ale to tylko scenariusz [S1].")
    return Reply(tool_calls=[{"name": "submit_explanation", "arguments": {
        "poziom_esi": level, "uzasadnienie": text, "kluczowe_czynniki": [factor],
        "sugestia_eskalacji": {"czy": False, "poziom": None, "powod": None}}}])


def lost_responder(req: dict[str, Any]) -> Reply:
    """Słaby agent: zmyślone narzędzie, niedozwolona zmiana what_if, nigdy nie kończy."""
    n = len(_tool_results(req))
    if n % 2 == 0:
        return Reply(tool_calls=[{"name": "get_patient_history", "arguments": {"id": 1}}])
    return Reply(tool_calls=[{"name": "what_if", "arguments": {"changes": {"site": "A"}}}])


def text_responder(req: dict[str, Any]) -> Reply:
    """Agent bez narzędzi: od razu JSON w treści (ukończone przez treść)."""
    level = decision_from_prompt(_first_user(req))
    return Reply(content=json.dumps({"poziom_esi": level, "uzasadnienie": f"Poziom ESI {level} wynika z danych [S2].",
                                     "kluczowe_czynniki": ["dane"],
                                     "sugestia_eskalacji": {"czy": False, "poziom": None, "powod": None}},
                                    ensure_ascii=False))


# ─────────────────────────────────────────
# what_if
# ─────────────────────────────────────────
class TestWhatIf:
    def test_unchanged_row_reproduces_recipe_decision_exactly(self, llm_esi_env):
        """what_if bez zmian (i ze zmianą na tę samą wartość) = decyzja receptury dla tego wiersza."""
        from src.llm_eval.tools_task import predict_row, what_if

        ctx = llm_esi_env.ctx
        for case in ctx.cases:
            proba, level = predict_row(ctx, case.base_row)
            assert level == case.decision_esi == case.decision_check["stored_esi"]
            expected = np.array([case.probabilities[f"ESI{k}"] for k in range(1, 6)])
            np.testing.assert_allclose(proba, expected, rtol=0, atol=1e-12)
            res = what_if(ctx, case, {})
            assert res["ok"] and res["poziom_esi_przed"] == res["poziom_esi_po"] == case.decision_esi
            hr = case.base_row["triage_vital_hr"]
            same = what_if(ctx, case, {"triage_vital_hr": hr})
            assert same["poziom_esi_po"] == case.decision_esi
            assert same["prawdopodobienstwa_po"] == res["prawdopodobienstwa_po"]

    def test_changes_are_applied_through_feature_engineering(self, llm_esi_env):
        from src.llm_eval.tools_task import apply_changes, predict_row, what_if

        ctx = llm_esi_env.ctx
        case = ctx.cases[0]
        row, applied = apply_changes(case.base_row, {"spo2": "83,5", "tetno": 160, "arrival": "karetka",
                                                     "Ból w klatce piersiowej": True, "tlen": "tak"})
        assert row["triage_vital_o2"] == 83.5 and row["triage_vital_hr"] == 160.0
        assert (row["arrival_ambulance"], row["arrival_self"], row["arrival_other"]) == (1.0, 0.0, 0.0)
        assert row["cc_chestpain"] == 1.0 and row["triage_vital_o2_device"] == 1.0
        assert {a["kolumna"] for a in applied} == {"triage_vital_o2", "triage_vital_hr", "arrival", "cc_chestpain",
                                                   "triage_vital_o2_device"}
        X = ctx.predictor.features(pd.DataFrame([row]).astype(np.float32))
        assert X.loc[0, "severe_hypoxia"] == 1.0 and X.loc[0, "esi_danger_zone"] == 1.0
        assert X.loc[0, "chestpain_tachycardia"] == 1.0 and X.loc[0, "cc_count"] >= 1
        res = what_if(ctx, case, {"triage_vital_o2": 80, "triage_vital_hr": 160})
        assert res["ok"] and res["reguly_bezpieczenstwa_po"]["sugerowany_esi"] == 1        # SpO2 < 85, HR > 150
        assert res["poziom_esi_po"] == predict_row(ctx, apply_changes(case.base_row, {"triage_vital_o2": 80,
                                                                                    "triage_vital_hr": 160})[0])[1]
        assert {z["cecha"] for z in res["zmiany"]} == {"Saturacja (SpO2)", "Tętno (HR)"}
        total = sum(float(v.rstrip("%").replace(",", ".")) for v in res["prawdopodobienstwa_po"].values())
        assert abs(total - 100) < 0.2
        missing = what_if(ctx, case, {"triage_vital_temp": None})
        assert missing["ok"] and missing["zmiany"][0]["po"] == "brak pomiaru"

    @pytest.mark.parametrize("changes, fragment", [
        ({"site": "A"}, "nie może być zmieniana"),
        ({"esi": 1}, "nie może być zmieniana"),
        ({"cc_nieistniejaca": 1}, "Nieznana skarga"),
        ({"triage_vital_hr": 500}, "poza zakresem"),
        ({"age": 5}, "poza zakresem"),
        ({"triage_vital_temp": 101}, "°C"),
        ({"triage_vital_dbp": 200, "triage_vital_sbp": 120}, "rozkurczowe"),
        ({"cc_chestpain": 2}, "0 albo 1"),
        ({"triage_vital_hr": "szybko"}, "oczekiwano liczby"),
        ({"arrival": "helikopter"}, "arrival"),
    ])
    def test_invalid_changes(self, llm_esi_env, changes, fragment):
        from src.llm_eval.tools_task import what_if

        res = what_if(llm_esi_env.ctx, llm_esi_env.ctx.cases[0], changes)
        assert res["ok"] is False and fragment in res["blad"]


class TestToolHelpers:
    def test_validate_args(self):
        from src.llm_eval.tools_task import _SCHEMAS, validate_args

        assert validate_args(_SCHEMAS["what_if"], {"changes": {"triage_vital_hr": 120}}) == []
        assert "brak wymaganego pola 'changes'" in validate_args(_SCHEMAS["what_if"], {})[0]
        assert validate_args(_SCHEMAS["what_if"], {"changes": [1]})
        assert validate_args(_SCHEMAS["esi_criteria"], {"level": 6})
        assert validate_args(_SCHEMAS["esi_criteria"], {"level": True})          # bool to nie liczba
        assert validate_args(_SCHEMAS["explain_model"], {"top_n": 3, "x": 1})    # nieznane pole
        good = {"poziom_esi": 2, "uzasadnienie": "x", "kluczowe_czynniki": ["a"],
                "sugestia_eskalacji": {"czy": True, "poziom": 1, "powod": None}}
        assert validate_args(_SCHEMAS["submit_explanation"], good) == []
        assert validate_args(_SCHEMAS["submit_explanation"], {**good, "kluczowe_czynniki": [1]})

    def test_explain_model_and_criteria(self, llm_esi_env):
        from src.llm_eval.tools_task import execute_tool

        ctx = llm_esi_env.ctx
        case = ctx.cases[0]
        res = execute_tool(ctx, case, "explain_model", {"top_n": 2})
        assert res["ok"] and len(res["czynniki"]) == min(2, len(case.shap))
        assert res["czynniki"][0]["cecha"] == case.shap[0]["label_pl"]
        crit = execute_tool(ctx, case, "esi_criteria", {"level": 1})
        assert crit["ok"] and "A" in crit["punkty_decyzyjne"]
        assert execute_tool(ctx, case, "search_knowledge", {"query": "NEWS2"})["wyniki"][0]["id"] == "W03"
        assert execute_tool(ctx, case, "search_knowledge", {"query": "zzz"})["wyniki"] == []


# ─────────────────────────────────────────
# Pętla agenta
# ─────────────────────────────────────────
class TestToolsPart:
    def test_agents_with_fake_server(self, llm_esi_env, tmp_path):
        from src.llm_eval.cache import ResponseCache
        from src.llm_eval.tools_task import MAX_STEPS, TOOL_NAMES, predict_row, run_tools_part

        ctx = llm_esi_env.ctx
        models = ["agent:cloud", "lost:cloud", "text:cloud"]
        with FakeOllama() as srv:
            srv.add_model("agent:cloud", agent_responder)
            srv.add_model("lost:cloud", lost_responder)
            srv.add_model("text:cloud", text_responder)
            client = OllamaChat(base_url=srv.base_url, max_retries=1, backoff_base_s=0.0, concurrency=3)
            cache = ResponseCache(tmp_path / "cache.jsonl")
            res = run_tools_part(client, models, out_dir=tmp_path / "out", context=ctx, think="low", cache=cache)
            n_calls = srv.chat_count()
            agent_bodies = srv.chat_requests("agent:cloud")
            again = run_tools_part(client, models, out_dir=tmp_path / "out2", context=ctx, think="low",
                                   cache=ResponseCache(tmp_path / "cache.jsonl"))
            assert srv.chat_count() == n_calls                       # wznowienie: każdy krok z pamięci
        assert n_calls == N_CASES * (2 + MAX_STEPS + 1)

        first = agent_bodies[0]
        assert [t["function"]["name"] for t in first["tools"]] == list(TOOL_NAMES)
        assert "[S3]" not in first["messages"][1]["content"] and "[S4]" in first["messages"][1]["content"]
        second = next(b for b in agent_bodies if any(m["role"] == "tool" for m in b["messages"]))
        assistant = next(m for m in second["messages"] if m["role"] == "assistant")
        assert [c["function"]["name"] for c in assistant["tool_calls"]] == ["explain_model", "what_if"]
        tool_msgs = [m for m in second["messages"] if m["role"] == "tool"]
        assert [m["tool_name"] for m in tool_msgs] == ["explain_model", "what_if"]
        wi = json.loads(tool_msgs[1]["content"])
        case = ctx.case_by_id(re.search(r"Przypadek (C\d+)", second["messages"][1]["content"]).group(1))
        row = dict(case.base_row, triage_vital_o2=85.0)
        assert wi["id_zrodla"] == "T2" and wi["poziom_esi_po"] == predict_row(ctx, row)[1]

        # przeplot przypadek → model (awaria pod koniec przebiegu nie trafia tylko w ostatni model z listy)
        assert [it["model"] for it in res["items"][:3]] == models
        assert len({it["case_id"] for it in res["items"][:3]}) == 1

        agent, lost, text = (res["summary"][m] for m in models)
        assert agent["completion_rate"] == 1.0 and agent["valid_tool_call_rate"] == 1.0
        assert agent["hallucinated_tools"] == 0 and agent["avg_steps"] == 2.0
        assert agent["what_if_use_rate"] == 1.0 and agent["what_if_ok_rate"] == 1.0
        assert agent["explain_model_rate"] == 1.0 and agent["level_match_rate"] == 1.0
        assert agent["hallucination_rate"] == 0.0              # SpO2 85% z własnego what_if — nie halucynacja
        assert agent["citation_rate"] == 1.0 and agent["invalid_citations"] == 0
        assert agent["strict_rate"] == 1.0 and agent["shap_coverage"] > 0
        assert lost["completion_rate"] == 0.0 and lost["avg_steps"] == MAX_STEPS
        assert lost["hallucinated_tools"] == N_CASES * 3 and lost["what_if_ok_rate"] == 0.0
        assert lost["valid_tool_call_rate"] == 0.5 and lost["level_match_rate"] is None
        assert text["completion_rate"] == 1.0 and text["tool_calls"] == 0 and text["avg_steps"] == 1.0
        assert text["valid_tool_call_rate"] is None and text["strict_rate"] == 0.0
        assert again["summary"]["agent:cloud"]["completion_rate"] == 1.0

        files = res["files"]
        for key in ("items_csv", "summary_csv", "trajectories", "raw", "section_md", "cases"):
            assert Path(files[key]).exists() and Path(files[key]).parent == tmp_path / "out", key
        traj = [json.loads(l) for l in (tmp_path / "out" / "tools_trajectories.jsonl").read_text(encoding="utf-8")
                .splitlines()]
        lost_traj = next(t for t in traj if t["model"] == "lost:cloud")["trajectory"]
        assert lost_traj[0]["name"] == "get_patient_history" and lost_traj[0]["known"] is False
        assert "## Część narzędziowa" in res["markdown"] and "`agent:cloud`" in res["markdown"]

    def test_skipped_without_runs(self, tmp_path):
        from src.llm_eval.tools_task import run_tools_part

        res = run_tools_part(OllamaChat(), ["x"], out_dir=tmp_path, runs_dir=tmp_path / "brak-runow", n_cases=3)
        assert res["skipped"] and "Pominięto" in res["markdown"]


# ─────────────────────────────────────────
# Integracja z CLI (scripts/12_llm_bakeoff.py): części esi + tools w jednym przebiegu
# ─────────────────────────────────────────
def _dispatch_responder(req: dict[str, Any]) -> Reply:
    """Część tools (5 narzędzi) → dobry agent; część esi (1 narzędzie) → poprawne wyjaśnienie."""
    from tests.test_llm_eval_esi import good_responder

    if len(req.get("tools") or []) > 1:
        return agent_responder(req)
    return good_responder(req)


def _load_cli():
    import importlib.util
    import sys

    from tests.test_llm_eval_esi import PROJECT_ROOT

    script = PROJECT_ROOT / "scripts" / "12_llm_bakeoff.py"
    spec = importlib.util.spec_from_file_location("llm_bakeoff_cli_for_esi_tests", script)
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)
    return cli


def test_cli_runs_esi_and_tools_parts(llm_esi_env, env_paths, tmp_path):
    cli = _load_cli()
    out = tmp_path / "run"
    with FakeOllama() as srv:
        srv.add_model("m-one:cloud", _dispatch_responder)
        code = cli.main(["run", "--parts", "esi,tools", "--models", "m-one:cloud", "--base-url", srv.base_url,
                         "--recipe", str(llm_esi_env.recipe_dir), "--out", str(out), "--n-cases", str(N_CASES),
                         "--no-preflight", "--cache", str(tmp_path / "cache.jsonl")])
        n_requests = srv.chat_count()
    assert code == 0
    assert n_requests == N_CASES * 3                          # esi: 1 krok, tools: 2 kroki na przypadek
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Część ESI" in report and "## Część narzędziowa" in report
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["parts"]["esi"]["summary"]["m-one:cloud"]["level_match_rate"] == 1.0
    assert results["parts"]["tools"]["summary"]["m-one:cloud"]["completion_rate"] == 1.0
    raw_parts = {json.loads(line)["part"] for line in (out / "raw_responses.jsonl").read_text(encoding="utf-8")
                 .splitlines()}
    assert {"esi", "tools"} <= raw_parts                       # surowe odpowiedzi w pliku przebiegu
    assert (out / "ratings" / "ratings.html").exists() and (out / "key.json").exists()


def test_cli_all_three_parts_rank_models(llm_esi_env, env_paths, tmp_path):
    """Pełny przebieg CLI (exams + esi + tools) na triage_responder: raport ze wszystkimi sekcjami i wyborem."""
    from tests.fake_ollama import answer_key_from_frame, triage_responder
    from tests.test_llm_eval_exams import make_nask_frame

    cli = _load_cli()
    frame = make_nask_frame(n_per_edition=20)
    exam_csv = tmp_path / "nask.csv"
    frame.to_csv(exam_csv, index=False, encoding="utf-8")
    key = answer_key_from_frame(frame)
    strong, weak = "glm-5.3:cloud", "gpt-oss:120b-cloud"
    out = tmp_path / "run"
    with FakeOllama() as srv:
        srv.add_model(strong, triage_responder(key, skill=0.95, seed=0, thinking=True))
        srv.add_model(weak, triage_responder(key, skill=0.35, seed=1))
        code = cli.main(["run", "--models", f"{strong},{weak}", "--base-url", srv.base_url,
                         "--exam-file", str(exam_csv), "--n-exam", "40", "--n-pes", "20", "--n-boot", "300",
                         "--recipe", str(llm_esi_env.recipe_dir), "--n-cases", str(N_CASES),
                         "--out", str(out), "--cache", str(tmp_path / "cache.jsonl")])
    assert code == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    for heading in ("## Warunki pomiaru", "## Polskie egzaminy lekarskie", "## Część ESI", "## Część narzędziowa",
                    "## Koszty i zużycie tokenów", "## Wybór modelu", "## Ograniczenia", "## Pliki przebiegu"):
        assert heading in report, heading
    assert f"Wybrany model: `{strong}`" in report and "Czas na pozycję" in report
    assert "structured outputs" in report and "ratings.html" in report
    parts_md = report.split("## Część ESI")[1].split("## Koszty")[0]
    assert not re.search(r"\d,\d", parts_md)                  # bez --decimal-comma: kropka jak w reszcie raportu
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["selection"]["winner"] == strong
    assert {c["part"] for c in results["costs"]} == {"exams", "esi", "tools"}
    assert all(c["cost_usd"] is not None and c["cost_usd"] > 0 for c in results["costs"])
    esi = results["parts"]["esi"]["summary"]
    assert esi[strong]["valid_rate"] >= esi[weak]["valid_rate"]
    assert esi[strong]["hallucination_rate"] <= esi[weak]["hallucination_rate"]
    tools = results["parts"]["tools"]["summary"]
    assert tools[strong]["valid_tool_call_rate"] > tools[weak]["valid_tool_call_rate"]


def test_cli_dry_run_reports_esi_availability(llm_esi_env, env_paths, tmp_path, capsys):
    cli = _load_cli()
    with FakeOllama() as srv:
        srv.add_model("*", "ok")
        base = ["run", "--dry-run", "--parts", "esi,tools", "--models", "glm-5.3:cloud", "--base-url", srv.base_url,
                "--n-cases", "40", "--cache", str(tmp_path / "c.jsonl")]
        assert cli.main(base + ["--recipe", str(llm_esi_env.recipe_dir)]) == 0
        ready = capsys.readouterr().out
        assert cli.main(base + ["--recipe", str(tmp_path / "brak")]) == 0
        missing = capsys.readouterr().out
        assert srv.chat_count() == 0
    assert "Część esi: 40 przypadków (receptura" in ready and "logreg" in ready and "$" in ready
    assert "Część tools: 40 przypadków" in ready
    assert "Część esi: zostanie POMINIĘTA" in missing and "Brak pliku receptury" in missing


def test_cases_status(llm_esi_env, env_paths, tmp_path):
    from src.llm_eval.esi_cases import cases_status

    ok, detail = cases_status(llm_esi_env.recipe_dir)
    assert ok and "lightgbm" in detail
    ok, detail = cases_status(runs_dir=tmp_path / "pusty")
    assert not ok and "brak ukończonych runów" in detail
    ok, detail = cases_status(runs_dir=env_paths.RUNS_DIR)       # runy testowe są bez tuningu
    assert not ok
