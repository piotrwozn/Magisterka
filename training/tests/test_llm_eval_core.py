"""
Testy rdzenia porównania modeli LLM (src/llm_eval): klient Ollamy (na atrapie serwera
tests/fake_ollama.py — bez sieci), pamięć podręczna, cennik i wybór modelu w raporcie.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from src.llm_eval.cache import JsonlWriter, ResponseCache, prompt_hash
from src.llm_eval.client import (
    ChatJob,
    ChatResult,
    OllamaChat,
    canonical_model,
    is_cloud_url,
    normalize_base_url,
    parse_think,
    split_think_tags,
)
from src.llm_eval.pricing import (
    DEFAULT_PRICES,
    cost_usd,
    estimate_part,
    format_usd,
    load_prices,
    price_for,
)
from tests.fake_ollama import FakeOllama, Reply

MSG = [{"role": "user", "content": "Ile to 2+2?"}]


def make_client(srv: FakeOllama, **kw) -> OllamaChat:
    kw.setdefault("backoff_base_s", 0.0)
    kw.setdefault("backoff_max_s", 0.0)
    kw.setdefault("timeout_s", 5)
    return OllamaChat(base_url=srv.base_url, **kw)


@pytest.fixture
def srv():
    """Uruchomiona atrapa Ollamy (tests/fake_ollama.py) — bez modeli, dodawane w teście."""
    with FakeOllama() as server:
        yield server


# ─────────────────────────────────────────
# Nazwy, adresy, think
# ─────────────────────────────────────────
@pytest.mark.parametrize("tag,expected", [
    ("glm-5.3:cloud", "glm-5.3"),
    ("gpt-oss:120b-cloud", "gpt-oss:120b"),
    ("kimi-k3:cloud", "kimi-k3"),
    ("deepseek-v4.1-flash:cloud", "deepseek-v4.1-flash"),
    ("qwen3:8b", "qwen3:8b"),
    ("model-cloud", "model"),
])
def test_canonical_model_strips_cloud_suffix(tag, expected):
    assert canonical_model(tag) == expected


def test_cloud_url_detection_and_normalization():
    assert is_cloud_url("https://ollama.com")
    assert is_cloud_url("https://ollama.com/api/")
    assert not is_cloud_url("http://localhost:11434")
    assert normalize_base_url("https://ollama.com/api/") == "https://ollama.com"
    assert normalize_base_url("localhost:11434") == "http://localhost:11434"
    cloud = OllamaChat("https://ollama.com", api_key="k")
    local = OllamaChat("http://localhost:11434")
    assert cloud.mode == "cloud" and cloud.api_model("glm-5.3:cloud") == "glm-5.3"
    assert local.mode == "local" and local.api_model("glm-5.3:cloud") == "glm-5.3:cloud"


def test_parse_think_values():
    assert parse_think("low") == "low"
    assert parse_think("MAX") == "max"
    assert parse_think("default") is None
    assert parse_think("true") is True
    assert parse_think("off") is False
    with pytest.raises(ValueError):
        parse_think("bardzo dużo")


def test_split_think_tags():
    assert split_think_tags("<think>rozważam</think>\nOdpowiedź: B") == ("Odpowiedź: B", "rozważam")
    assert split_think_tags("Odpowiedź: B") == ("Odpowiedź: B", None)


# ─────────────────────────────────────────
# chat(): parsowanie odpowiedzi
# ─────────────────────────────────────────
def test_chat_parses_content_thinking_tokens_and_tool_calls(srv):
    srv.add_model("m", [
        Reply(content="Prawidłowa odpowiedź to B.", thinking="krótko", prompt_tokens=11, completion_tokens=7),
        Reply(tool_calls=[{"name": "submit_explanation", "arguments": {"poziom_esi": 2}},
                          {"name": "what_if", "arguments": '{"changes": {"sbp": 80}}'},
                          {"name": "zly", "arguments": "{nie json"}]),
    ])
    client = make_client(srv)
    res = client.chat("m", MSG, think="low", options={"temperature": 0})
    assert res.ok and res.content == "Prawidłowa odpowiedź to B." and res.thinking == "krótko"
    assert (res.prompt_tokens, res.completion_tokens) == (11, 7)
    assert res.latency_s >= 0 and res.attempts == 1 and res.raw["done"] is True
    body = srv.chat_requests("m")[0]
    assert body["think"] == "low" and body["stream"] is False and body["options"] == {"temperature": 0}

    res2 = client.chat("m", MSG, tools=[{"type": "function", "function": {"name": "submit_explanation"}}])
    assert [c["name"] for c in res2.tool_calls] == ["submit_explanation", "what_if", "zly"]
    assert res2.tool_calls[0]["arguments"] == {"poziom_esi": 2}
    assert res2.tool_calls[1]["arguments"] == {"changes": {"sbp": 80}}       # JSON w tekście → dict
    assert res2.tool_calls[2]["arguments"] == {} and "arguments_raw" in res2.tool_calls[2]
    assert "tools" in srv.chat_requests("m")[1]


def test_chat_splits_think_tags_in_content(srv):
    srv.add_model("m", "<think>tok rozumowania</think>Prawidłowa odpowiedź to C.")
    res = make_client(srv).chat("m", MSG)
    assert res.content == "Prawidłowa odpowiedź to C." and res.thinking == "tok rozumowania"


def test_chat_sends_bearer_key(srv):
    srv.add_model("m", "ok")
    make_client(srv, api_key="sekret").chat("m", MSG)
    headers = {k.lower(): v for k, v in srv.requests[-1]["headers"].items()}
    assert headers["authorization"] == "Bearer sekret"


def test_chat_handles_streaming_reply(srv):
    """Gdyby serwer zignorował stream=false — odpowiedź NDJSON jest sklejana."""
    import requests

    srv.add_model("m", Reply(content="Answer: D", thinking="x"))
    # Atrapa odpowiada NDJSON-em, gdy stream != false — sprawdzamy funkcję sklejania na prawdziwej odpowiedzi
    from src.llm_eval.client import _merge_ndjson

    text = requests.post(srv.base_url + "/api/chat",
                         json={"model": "m", "messages": MSG}).text
    merged = _merge_ndjson(text)
    assert merged["message"]["content"] == "Answer: D" and merged["eval_count"] > 0


# ─────────────────────────────────────────
# Ponowienia, błędy, bezpiecznik
# ─────────────────────────────────────────
def test_retries_on_429_and_503_then_succeeds(srv):
    srv.add_model("m", "ok")
    srv.fail_next(429, model="m", retry_after=0)
    srv.fail_next(503, times=2, model="m")
    res = make_client(srv, max_retries=5).chat("m", MSG)
    assert res.ok and res.content == "ok" and res.attempts == 4
    assert srv.chat_count("m") == 4


def test_gives_up_after_max_retries_without_raising(srv):
    srv.add_model("m", "ok")
    srv.fail_next(502, times=10, model="m")
    res = make_client(srv, max_retries=2).chat("m", MSG)
    assert not res.ok and "HTTP 502" in res.error and res.attempts == 3 and res.status == 502
    assert res.raw["status"] == 502


def test_no_retry_on_404_unknown_model(srv):
    res = make_client(srv).chat("brak", MSG)
    assert not res.ok and "404" in res.error and res.attempts == 1


def test_auth_error_has_hint(srv):
    srv.add_model("m", Reply(status=401, error="unauthorized"))
    res = make_client(srv).chat("m", MSG)
    assert not res.ok and "ollama signin" in res.error and res.attempts == 1


def test_timeout_is_retried_and_reported(srv):
    srv.add_model("m", Reply(content="za późno", delay_s=1.5))
    res = make_client(srv, timeout_s=0.3, max_retries=1).chat("m", MSG)
    assert not res.ok and "czas" in res.error and res.attempts == 2


def test_connection_refused_does_not_raise():
    client = OllamaChat("http://127.0.0.1:9", timeout_s=2, max_retries=1, backoff_base_s=0, backoff_max_s=0)
    res = client.chat("m", MSG)
    assert not res.ok and "połączenia" in res.error and res.attempts == 2
    info = client.server_info()
    assert info["version"] is None and info["error"]
    assert client.model_info("m")["error"]


def test_think_rejected_is_retried_without_think(srv):
    def respond(req):
        if "think" in req:
            return Reply(status=400, error='"m" does not support thinking')
        return Reply(content="bez rozumowania")

    srv.add_model("m", respond)
    res = make_client(srv).chat("m", MSG, think="low")
    assert res.ok and res.content == "bez rozumowania" and res.notes
    bodies = srv.chat_requests("m")
    assert "think" in bodies[0] and "think" not in bodies[1]


def test_breaker_stops_calling_a_dead_model(srv):
    srv.add_model("m", Reply(status=500, error="boom"))
    client = make_client(srv, max_retries=0, max_consecutive_failures=2)
    assert not client.chat("m", MSG).ok
    assert not client.chat("m", MSG).ok
    skipped = client.chat("m", MSG)
    assert "pominięto" in skipped.error and srv.chat_count("m") == 2
    client.reset_breaker("m")
    client.chat("m", MSG)
    assert srv.chat_count("m") == 3
    assert client.usage["m"]["calls"] == 4 and client.usage["m"]["errors"] == 4


# ─────────────────────────────────────────
# chat_many: kolejność i limit równoległości
# ─────────────────────────────────────────
def test_chat_many_keeps_order_and_bounds_concurrency(srv):
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    def respond(req):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.15)
        with lock:
            state["now"] -= 1
        return Reply(content=req["messages"][-1]["content"].upper())

    srv.add_model("m", respond)
    client = make_client(srv, concurrency=2)
    seen = []
    jobs = [ChatJob("m", [{"role": "user", "content": f"x{i}"}], key=i) for i in range(6)]
    results = client.chat_many(jobs, on_result=lambda i, job, res: seen.append((i, job.key, res.ok)))
    assert [r.content for r in results] == [f"X{i}" for i in range(6)]
    assert sorted(seen) == [(i, i, True) for i in range(6)]
    assert state["max"] == 2


def test_chat_many_accepts_dict_jobs(srv):
    srv.add_model("*", "ok")
    res = make_client(srv).chat_many([{"model": "a", "messages": MSG, "extra": 1}, {"model": "b", "messages": MSG}])
    assert [r.model for r in res] == ["a", "b"] and all(r.ok for r in res)
    assert make_client(srv).chat_many([]) == []


# ─────────────────────────────────────────
# server_info / model_info
# ─────────────────────────────────────────
def test_server_and_model_info(srv):
    srv.version = "0.31.2"
    srv.add_model("glm-5.3:cloud", "ok", details={"family": "glm", "parameter_size": "744B"}, digest="abc123def456789")
    client = make_client(srv)
    info = client.server_info()
    assert info["version"] == "0.31.2" and info["mode"] == "local" and info["error"] is None
    mi = client.model_info("glm-5.3:cloud")
    assert mi["error"] is None and mi["details"]["family"] == "glm" and mi["digest"] == "abc123def456789"
    assert "tools" in mi["capabilities"]
    assert client.model_info("brak")["error"].startswith("HTTP 404")


# ─────────────────────────────────────────
# Pamięć podręczna
# ─────────────────────────────────────────
def test_cache_roundtrip_skips_errors_and_survives_truncation(tmp_path):
    path = tmp_path / "c" / "responses.jsonl"
    cache = ResponseCache(path)
    h = prompt_hash(MSG, think="low")
    ok = ChatResult(content="B", prompt_tokens=3, completion_tokens=1, latency_s=0.5, model="m")
    assert cache.put("exams", "m", "standard:1", h, ok)
    assert not cache.put("exams", "m", "standard:2", h, ChatResult(error="HTTP 503", model="m"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"part": "exams", "model": "m", "item_id": "x", "prompt_ha')    # ucięta linia
    again = ResponseCache(path)
    hit = again.get("exams", "m", "standard:1", h)
    assert hit is not None and ChatResult.from_dict(hit).content == "B"
    assert again.get("exams", "m", "standard:2", h) is None
    assert again.get("exams", "m", "standard:1", prompt_hash(MSG, think="high")) is None
    assert ResponseCache(path, fresh=True).get("exams", "m", "standard:1", h) is None
    assert again.stats()["hits"] == 1


def test_prompt_hash_depends_on_every_input():
    base = prompt_hash(MSG, think="low", options={"temperature": 0})
    assert base == prompt_hash(MSG, think="low", options={"temperature": 0})
    assert base != prompt_hash(MSG, think="medium", options={"temperature": 0})
    assert base != prompt_hash(MSG, think="low", options={"temperature": 0.5})
    assert base != prompt_hash(MSG, tools=[{"x": 1}], think="low", options={"temperature": 0})
    assert len(base) == 16


def test_jsonl_writer_utf8(tmp_path):
    w = JsonlWriter(tmp_path / "raw.jsonl")
    w.write({"tekst": "żółć"})
    w.write({"n": 2})
    lines = (tmp_path / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["tekst"] == "żółć" and w.count == 2


# ─────────────────────────────────────────
# Cennik
# ─────────────────────────────────────────
def test_default_prices_and_costs():
    assert price_for("gpt-oss:120b-cloud") == {"input": 0.15, "output": 0.60}
    assert price_for("glm-5.3:cloud") == DEFAULT_PRICES["glm-5.3"]
    assert price_for("kimi-k3:cloud") == {"input": 3.00, "output": 15.00}
    assert price_for("deepseek-v4.1-flash:cloud") == {"input": 0.30, "output": 1.20}
    assert cost_usd("kimi-k3:cloud", 1_000_000, 100_000) == pytest.approx(3.0 + 1.5)
    assert cost_usd("nieznany", 10, 10) is None
    assert format_usd(None) == "—" and format_usd(1.234) == "$1.23"
    est = estimate_part("esi", "glm-5.3:cloud", 40)
    assert est["calls"] > 0 and est["cost_usd"] > 0


def test_price_overrides_from_json(tmp_path):
    path = tmp_path / "ceny.json"
    path.write_text(json.dumps({"glm-5.3:cloud": {"input": 1.0, "output": 2.0}, "nowy": {"input": 0, "output": 1}}),
                    encoding="utf-8")
    prices = load_prices(path)
    assert prices["glm-5.3"] == {"input": 1.0, "output": 2.0} and "nowy" in prices
    assert cost_usd("nowy", 5, 1_000_000, prices) == pytest.approx(1.0)
    path.write_text(json.dumps({"x": {"input": 1}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_prices(path)


# ─────────────────────────────────────────
# Wybór modelu (report.select_model)
# ─────────────────────────────────────────
def _fake_exams(acc: dict[str, tuple[float, float]], n: int = 400, seed: int = 0) -> dict:
    """Wyniki egzaminów w formacie run_exams_part z wektorami trafień o zadanych trafnościach."""
    from src.llm_eval.exams import evaluate

    import pandas as pd

    rng = np.random.default_rng(seed)
    items = pd.DataFrame({"item_id": [str(i) for i in range(n)], "split": ["r_lek"] * n})
    base = rng.random(n)
    rows = []
    for model, (a_std, a_abs) in acc.items():
        for protocol, a in (("standard", a_std), ("abstaining_substitution", a_abs)):
            correct = base < a          # wspólna „trudność” pytań → silnie sparowane modele
            for i in range(n):
                rows.append({"model": model, "protocol": protocol, "item_id": str(i), "split": "r_lek",
                             "correct": bool(correct[i]), "parse_ok": True, "error": None, "latency_s": 1.0,
                             "prompt_tokens": 100, "completion_tokens": 10, "cached": False})
    responses = pd.DataFrame(rows)
    res = evaluate(responses, items, list(acc), n_boot=300)
    res["models"] = list(acc)
    from src.llm_eval.exams import correctness_matrix

    res["_correctness"] = {p: correctness_matrix(responses, items, p, list(acc))
                           for p in ("standard", "abstaining_substitution")}
    return res


def test_select_model_primary_criterion_wins():
    from src.llm_eval.report import select_model

    exams = _fake_exams({"a": (0.80, 0.40), "b": (0.90, 0.70), "c": (0.85, 0.50)})
    sel = select_model(exams, n_boot=300)
    assert sel["winner"] == "b" and sel["decided_by"] == "primary"
    assert [r["model"] for r in sel["ranking"]] == ["b", "c", "a"]


def test_select_model_tie_broken_by_standard_then_esi():
    from src.llm_eval.report import select_model

    # identyczne wektory abstaining (remis), standard wyraźnie lepszy dla "y"
    exams = _fake_exams({"x": (0.60, 0.50), "y": (0.85, 0.50)})
    sel = select_model(exams, n_boot=300)
    assert sel["winner"] == "y" and sel["decided_by"] == "standard" and set(sel["primary_tie"]) == {"x", "y"}

    exams2 = _fake_exams({"x": (0.70, 0.50), "y": (0.70, 0.50)})
    sel2 = select_model(exams2, {"ranking_score": {"x": 0.9, "y": 0.4}}, n_boot=300)
    assert sel2["winner"] == "x" and sel2["decided_by"] == "esi"
    sel3 = select_model(exams2, None, n_boot=300)
    assert sel3["decided_by"] == "point_estimate" and "remis" in sel3["explanation"].lower()


def test_select_model_without_exams_uses_esi_or_reports_missing():
    from src.llm_eval.report import select_model

    assert select_model(None, {"ranking_score": {"a": 0.2, "b": 0.7}})["winner"] == "b"
    assert select_model(None, None)["winner"] is None


def test_deployability_notes_cover_default_models():
    from src.llm_eval.report import deployability_note

    for tag in ("glm-5.3:cloud", "gpt-oss:120b-cloud", "deepseek-v4.1-flash:cloud", "kimi-k3:cloud"):
        assert deployability_note(tag) != "brak notatki"
    assert deployability_note("inny:1b") == "brak notatki"


def test_esi_scores_from_part_summary():
    from src.llm_eval.report import esi_scores

    summary = {"a": {"valid_rate": 1.0, "level_match_rate": 1.0, "shap_coverage": 0.6, "hallucination_rate": 0.0,
                     "polish_rate": 1.0},
               "b": {"valid_rate": 0.5, "level_match_rate": None, "shap_coverage": 0.2, "hallucination_rate": 0.5,
                     "polish_rate": 1.0}}
    scores = esi_scores({"skipped": None, "summary": summary})
    assert scores["a"] == pytest.approx(0.92) and scores["b"] == pytest.approx((0.5 + 0.2 + 0.5 + 1.0) / 4)
    assert esi_scores({"ranking_score": {"a": 2}, "summary": summary}) == {"a": 2.0}
    assert esi_scores({"skipped": "brak runów", "summary": summary}) == {}
    assert esi_scores(None) == {}


def test_think_rejection_is_remembered_per_model(srv):
    def respond(req):
        return Reply(status=400, error="think value not supported") if "think" in req else Reply(content="ok")

    srv.add_model("m", respond)
    client = make_client(srv)
    assert client.chat("m", MSG, think="low").ok
    second = client.chat("m", MSG, think="low")
    assert second.ok and "pominięte" in second.notes[0]
    assert srv.chat_count("m") == 3                  # 400 + ponowienie, potem od razu bez think


def test_api_error_banner():
    from src.llm_eval.report import api_error_banner

    parts = {"exams": {"metrics": [{"model": "a", "split": "all", "api_errors": 3},
                                   {"model": "a", "split": "r_lek", "api_errors": 3},
                                   {"model": "b", "split": "all", "api_errors": 0}]},
             "esi": {"summary": {"b": {"api_errors": 1}}}}
    text = "\n".join(api_error_banner(parts))
    assert "`a`: 3" in text and "`b`: 1" in text
    assert api_error_banner({"exams": {"metrics": []}}) == []


def test_select_model_equal_esi_scores_fall_back_to_point_estimate():
    from src.llm_eval.report import select_model

    exams = _fake_exams({"x": (0.70, 0.50), "y": (0.70, 0.50)})
    sel = select_model(exams, {"summary": {"x": {"valid_rate": 0.0}, "y": {"valid_rate": 0.0}}}, n_boot=300)
    assert sel["decided_by"] == "point_estimate"
