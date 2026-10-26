"""
Atrapa serwera Ollama do testów — lokalny HTTP na 127.0.0.1 (bez sieci, bez modeli, bez GPU).

STABILNE API (używają go testy CORE i ESI — zmiany tylko zgodne wstecz)
=======================================================================

    from tests.fake_ollama import FakeOllama, Reply, fake_ollama   # fake_ollama = fixture pytest

    # 1) menedżer kontekstu (albo fixture `fake_ollama`, która zwraca uruchomiony serwer)
    with FakeOllama() as srv:
        srv.add_model("fake-a", "Prawidłowa odpowiedź to B.")             # stała odpowiedź (str)
        srv.add_model("fake-b", [Reply(thinking="hmm", content="A"),      # skrypt: kolejne odpowiedzi
                                 Reply(tool_calls=[{"name": "submit_explanation",
                                                    "arguments": {"poziom_esi": 2}}])])
                                                                           # (po wyczerpaniu: ostatnia w kółko)
        srv.add_model("fake-c", lambda req: Reply(content=req["messages"][-1]["content"]))
        srv.add_model("*", "domyślna")      # odpowiada za KAŻDY model bez własnej odpowiedzi

        client = OllamaChat(base_url=srv.base_url)
        srv.fail_next(503, times=2, model="fake-a")  # 2 kolejne /api/chat dla fake-a → HTTP 503
        srv.fail_next(429, retry_after=0)            # następne dowolne /api/chat → 429 (+ Retry-After: 0)
        srv.latency_s = 0.05                         # opóźnienie każdej odpowiedzi /api/chat
        srv.version = "0.30.0-fake"                  # GET /api/version

        srv.requests                 # lista wszystkich zapytań: {"method", "path", "headers", "body"}
        srv.chat_requests("fake-a")  # ciała JSON zapytań /api/chat (opcjonalnie tylko dla modelu)
        srv.chat_count("fake-a")     # ich liczba
        srv.reset_log()              # czyści log zapytań (np. przed drugim przebiegiem)

    Reply(content="", thinking=None, tool_calls=None, prompt_tokens=None, completion_tokens=None,
          status=200, error=None, delay_s=0.0, raw=None, done_reason="stop", retry_after=None)
        tool_calls   — [{"name": str, "arguments": dict | str}] (str = JSON w tekście, jak u
                       niektórych backendów); serwer zwraca je w formacie Ollamy
                       message.tool_calls[i].function.{index, name, arguments}
        status       — != 200 → odpowiedź {"error": error} z tym kodem (tylko ta jedna odpowiedź)
        raw          — dict zwracany DOKŁADNIE (np. zepsuta / nietypowa odpowiedź)
        delay_s      — opóźnienie tej odpowiedzi (do testu timeoutu klienta)
        prompt_tokens / completion_tokens — None → liczone ze słów (prompt_eval_count / eval_count)

    Odpowiedź (responder) modelu: str (= content), Reply, dict (= argumenty Reply), lista ich
    (skrypt) albo funkcja(req: dict) → str | Reply | dict, gdzie req = ciało JSON zapytania
    ({"model", "messages", "tools"?, "think"?, "options"?, "stream"?}).

    srv.add_model(name, responder, details=None, capabilities=None, digest=None, thinking=None) —
    details trafiają do /api/show i /api/tags (domyślnie family "fake", parameter_size "1B", …);
    thinking = obiekt "thinking" z /api/show (domyślnie {"values": ["low", "medium", "high"]}).

    exam_oracle(answer_key, p_correct=…) — gotowa odpowiedź dla testów egzaminów (zob. niżej).
    triage_responder(answer_key, skill=…) — „model” dla wszystkich części (egzaminy + esi + tools).

Endpointy
    POST /api/chat    — nieznany model → 404 {"error": "model '…' not found"}; stream=true →
                        NDJSON (dwie linie: treść + końcowa z done=true i licznikami tokenów)
    GET  /api/version — {"version": srv.version}
    POST /api/show    — {"model"} → {details, capabilities, model_info, modified_at, thinking}
    GET  /api/tags    — {"models": [{name, model, digest, details}]}

Uruchomienie samodzielne (ręczny test CLI): python -m tests.fake_ollama --models fake-a,fake-b
[--exam-csv plik.csv] [--port 11500]  → wypisuje adres i działa do Ctrl+C. Każdy model to
triage_responder (kolejne modele coraz słabsze), więc przebieg `12_llm_bakeoff.py run --base-url …`
przechodzi przez wszystkie trzy części bez prawdziwej Ollamy.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

# ─────────────────────────────────────────
# Odpowiedź skryptowa
# ─────────────────────────────────────────
@dataclass
class Reply:
    """Jedna odpowiedź atrapy na /api/chat (zob. docstring modułu)."""

    content: str = ""
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    status: int = 200
    error: str | None = None
    delay_s: float = 0.0
    raw: dict[str, Any] | None = None
    done_reason: str = "stop"
    retry_after: float | None = None


Responder = Callable[[dict[str, Any]], Any] | str | Reply | dict | list


@dataclass
class _Model:
    responder: Any
    details: dict[str, Any]
    capabilities: list[str]
    digest: str
    position: int = 0          # pozycja w skrypcie (lista odpowiedzi)
    thinking: dict[str, Any] | None = None     # /api/show "thinking" (None → low/medium/high)


@dataclass
class _Failure:
    status: int
    model: str | None
    error: str
    retry_after: float | None


def _as_reply(value: Any) -> Reply:
    if isinstance(value, Reply):
        return value
    if isinstance(value, str):
        return Reply(content=value)
    if isinstance(value, dict):
        return Reply(**value)
    if value is None:
        return Reply(content="")
    raise TypeError(f"Nieobsługiwany typ odpowiedzi atrapy: {type(value).__name__}")


def _words(text: Any) -> int:
    return len(str(text or "").split())


# ─────────────────────────────────────────
# Serwer
# ─────────────────────────────────────────
class FakeOllama:
    """Atrapa Ollamy w wątku w tle (ThreadingHTTPServer na 127.0.0.1, losowy wolny port)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.version: str = "0.30.0-fake"
        self.latency_s: float = 0.0
        self.requests: list[dict[str, Any]] = []
        self._models: dict[str, _Model] = {}
        self._failures: list[_Failure] = []
        self._lock = threading.Lock()
        self._host, self._port = host, port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── cykl życia ──
    def start(self) -> FakeOllama:
        if self._server is not None:
            return self
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="fake-ollama", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None

    def __enter__(self) -> FakeOllama:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Atrapa Ollamy nie jest uruchomiona (start())")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    # ── konfiguracja ──
    def add_model(self, name: str, responder: Responder, details: dict[str, Any] | None = None,
                  capabilities: list[str] | None = None, digest: str | None = None,
                  thinking: dict[str, Any] | None = None) -> FakeOllama:
        base = {"format": "gguf", "family": "fake", "families": ["fake"], "parameter_size": "1B",
                "quantization_level": "F16", "context_length": 8192, "parent_model": ""}
        base.update(details or {})
        with self._lock:
            self._models[name] = _Model(
                responder=responder,
                details=base,
                capabilities=list(capabilities or ["completion", "tools", "thinking"]),
                digest=digest or hashlib.sha256(name.encode("utf-8")).hexdigest(),
                thinking=thinking,
            )
        return self

    def fail_next(self, status: int = 503, times: int = 1, model: str | None = None,
                  error: str | None = None, retry_after: float | None = None) -> FakeOllama:
        """Kolejne `times` zapytań /api/chat (dla `model` albo dowolnego) dostanie HTTP `status`."""
        msg = error or {429: "too many requests", 502: "bad gateway", 503: "server overloaded"}.get(
            status, f"fake error {status}")
        with self._lock:
            self._failures.extend(_Failure(status, model, msg, retry_after) for _ in range(int(times)))
        return self

    def reset_log(self) -> None:
        with self._lock:
            self.requests.clear()

    def chat_requests(self, model: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            bodies = [r["body"] for r in self.requests if r["path"] == "/api/chat"]
        return [b for b in bodies if model is None or b.get("model") == model]

    def chat_count(self, model: str | None = None) -> int:
        return len(self.chat_requests(model))

    # ── logika odpowiedzi ──
    def _log(self, method: str, path: str, headers: dict[str, str], body: Any) -> None:
        with self._lock:
            self.requests.append({"method": method, "path": path, "headers": headers, "body": body})

    def _take_failure(self, model: str) -> _Failure | None:
        with self._lock:
            for i, f in enumerate(self._failures):
                if f.model is None or f.model == model:
                    return self._failures.pop(i)
        return None

    def _lookup(self, model: str) -> _Model | None:
        with self._lock:
            return self._models.get(model) or self._models.get("*")

    def _next_reply(self, entry: _Model, body: dict[str, Any]) -> Reply:
        responder = entry.responder
        if isinstance(responder, list):
            with self._lock:
                idx = min(entry.position, len(responder) - 1)
                entry.position += 1
            responder = responder[idx]
        if callable(responder) and not isinstance(responder, (Reply, str, dict)):
            responder = responder(body)
        return _as_reply(responder)

    def chat_payload(self, body: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str], float]:
        """(status, JSON, nagłówki, opóźnienie) odpowiedzi na /api/chat."""
        model = str(body.get("model", ""))
        failure = self._take_failure(model)
        if failure is not None:
            headers = {} if failure.retry_after is None else {"Retry-After": str(failure.retry_after)}
            return failure.status, {"error": failure.error}, headers, self.latency_s
        entry = self._lookup(model)
        if entry is None:
            return 404, {"error": f"model '{model}' not found"}, {}, self.latency_s
        reply = self._next_reply(entry, body)
        delay = self.latency_s + float(reply.delay_s or 0.0)
        headers = {} if reply.retry_after is None else {"Retry-After": str(reply.retry_after)}
        if reply.status != 200:
            return reply.status, {"error": reply.error or f"fake error {reply.status}"}, headers, delay
        if reply.raw is not None:
            return 200, reply.raw, headers, delay

        message: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
        if reply.thinking:
            message["thinking"] = reply.thinking
        if reply.tool_calls:
            message["tool_calls"] = [
                {"function": {"index": i, "name": tc.get("name"), "arguments": tc.get("arguments", {})}}
                for i, tc in enumerate(reply.tool_calls)
            ]
        prompt_tokens = reply.prompt_tokens
        if prompt_tokens is None:
            prompt_tokens = 10 + sum(_words(m.get("content")) for m in body.get("messages") or []
                                     if isinstance(m, dict))
        completion_tokens = reply.completion_tokens
        if completion_tokens is None:
            completion_tokens = _words(reply.content) + _words(reply.thinking) + 5 * len(reply.tool_calls or [])
        payload = {
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "done": True,
            "done_reason": reply.done_reason,
            "total_duration": int(delay * 1e9) + 1_000_000,
            "load_duration": 0,
            "prompt_eval_count": int(prompt_tokens),
            "eval_count": int(completion_tokens),
        }
        return 200, payload, headers, delay

    def show_payload(self, model: str) -> tuple[int, dict[str, Any]]:
        entry = self._lookup(model)
        if entry is None:
            return 404, {"error": f"model '{model}' not found"}
        return 200, {
            "details": entry.details,
            "capabilities": entry.capabilities,
            "model_info": {"general.architecture": entry.details.get("family", "fake")},
            "modified_at": "2026-10-01T00:00:00Z",
            "thinking": entry.thinking or {"values": ["low", "medium", "high"], "default": "medium"},
        }

    def tags_payload(self) -> dict[str, Any]:
        with self._lock:
            items = [(n, m) for n, m in self._models.items() if n != "*"]
        return {"models": [{"name": n, "model": n, "digest": m.digest, "details": m.details} for n, m in items]}


def _make_handler(srv: FakeOllama) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:     # cisza w testach
            return

        def _send(self, status: int, payload: Any, headers: dict[str, str] | None = None,
                  content_type: str = "application/json") -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):   # klient zrezygnował (timeout)
                pass

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {"_invalid_json": raw.decode("utf-8", errors="replace")}

        def do_GET(self) -> None:  # noqa: N802
            srv._log("GET", self.path, dict(self.headers.items()), None)
            if self.path == "/api/version":
                self._send(200, {"version": srv.version})
            elif self.path == "/api/tags":
                self._send(200, srv.tags_payload())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            body = self._body()
            srv._log("POST", self.path, dict(self.headers.items()), body)
            if not isinstance(body, dict) or "_invalid_json" in body:
                self._send(400, {"error": "invalid JSON"})
                return
            if self.path == "/api/chat":
                status, payload, headers, delay = srv.chat_payload(body)
                if delay > 0:
                    time.sleep(delay)
                if status == 200 and body.get("stream", True) is not False and "message" in payload:
                    self._send(200, _ndjson(payload), headers, content_type="application/x-ndjson")
                else:
                    self._send(status, payload, headers)
            elif self.path == "/api/show":
                status, payload = srv.show_payload(str(body.get("model") or body.get("name") or ""))
                self._send(status, payload)
            else:
                self._send(404, {"error": "not found"})

    return Handler


def _ndjson(payload: dict[str, Any]) -> bytes:
    """Odpowiedź strumieniowa: treść w pierwszej linii, liczniki w ostatniej (done=true)."""
    first = {"model": payload["model"], "created_at": payload["created_at"], "message": payload["message"],
             "done": False}
    last = dict(payload)
    last["message"] = {"role": "assistant", "content": ""}
    return (json.dumps(first, ensure_ascii=False) + "\n" + json.dumps(last, ensure_ascii=False) + "\n").encode("utf-8")


# ─────────────────────────────────────────
# Fixture pytest
# ─────────────────────────────────────────
@pytest.fixture
def fake_ollama() -> Iterator[FakeOllama]:
    """Uruchomiona atrapa Ollamy (bez modeli — dodaj je przez add_model)."""
    with FakeOllama() as srv:
        yield srv


# ─────────────────────────────────────────
# Gotowa odpowiedź dla pytań egzaminacyjnych (NASK)
# ─────────────────────────────────────────
ABSTAIN_TEXT = "żadna z odpowiedzi nie jest poprawna"


def _stem(question: str) -> str:
    return str(question).split("\n", 1)[0].strip()


def exam_oracle(answer_key: dict[str, str], p_correct: float = 0.8, p_correct_abstain: float | None = None,
                seed: int = 0, p_unparseable: float = 0.0, thinking: bool = False,
                style: str = "prev") -> Callable[[dict[str, Any]], Reply]:
    """
    Odpowiedź „modelu” na pytanie testowe: zna klucz (pierwsza linia pytania → litera) i trafia
    z prawdopodobieństwem p_correct (p_correct_abstain, gdy w opcjach jest „żadna z odpowiedzi
    nie jest poprawna”). Losowość deterministyczna: hash(seed, pytanie). p_unparseable — odsetek
    odpowiedzi bez litery. style: "prev" („Prawidłowa odpowiedź to X.”), "letter" („X”),
    "answer" („Answer: X”). Nieznane pytanie (np. test łączności) → „OK”.
    """
    p_abst = p_correct if p_correct_abstain is None else p_correct_abstain

    def respond(req: dict[str, Any]) -> Reply:
        messages = [m for m in req.get("messages") or [] if isinstance(m, dict)]
        user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        letter = answer_key.get(_stem(user))
        if letter is None:
            return Reply(content="OK")
        h = int(hashlib.sha256(f"{seed}|{user}".encode("utf-8")).hexdigest()[:12], 16)
        u1, u2 = (h % 10_000) / 10_000, ((h // 10_000) % 10_000) / 10_000
        abstain = ABSTAIN_TEXT in user
        if u2 < p_unparseable:
            return Reply(content="Trudno powiedzieć, to zależy od sytuacji klinicznej.",
                         thinking="rozważam" if thinking else None)
        options = "ABCDE"
        if u1 < (p_abst if abstain else p_correct):
            chosen = letter
        else:
            others = [c for c in options if c != letter]
            chosen = others[h % len(others)]
        content = {"prev": f"Prawidłowa odpowiedź to {chosen}.", "letter": chosen,
                   "answer": f"Answer: {chosen}"}[style]
        return Reply(content=content, thinking=f"Analiza pytania… odpowiedź to {options[h % 5]}." if thinking else None)

    return respond


_DECISION_RE = re.compile(r"Decyzja systemu: ESI (\d)")
_RULE_RE = re.compile(r"Reguły bezpieczeństwa ESI: (?:rozważ )?ESI (\d)")


def _tool_names(req: dict[str, Any]) -> list[str]:
    return [str((t.get("function") or {}).get("name")) for t in req.get("tools") or [] if isinstance(t, dict)]


def _users(req: dict[str, Any]) -> list[str]:
    return [str(m.get("content", "")) for m in req.get("messages") or []
            if isinstance(m, dict) and m.get("role") == "user"]


def triage_responder(answer_key: dict[str, str], skill: float = 0.8, seed: int = 0,
                     thinking: bool = False) -> Callable[[dict[str, Any]], Reply]:
    """
    „Model” do ręcznego przebiegu wszystkich części porównania (egzaminy + esi + tools).
    skill ∈ [0, 1] steruje jakością: trafność egzaminów (exam_oracle), a w częściach esi/tools —
    zgodność poziomu z decyzją, posłuszeństwo (narzędzie vs tekst), zmyślone pomiary i narzędzia,
    sugestie eskalacji. Losowość deterministyczna: hash(seed, pierwsza wiadomość użytkownika).
    """
    exam = exam_oracle(answer_key, p_correct=skill, p_correct_abstain=max(0.0, skill - 0.15), seed=seed,
                       thinking=thinking)

    def unit(text: str, salt: str) -> float:
        h = hashlib.sha256(f"{seed}|{salt}|{text}".encode("utf-8")).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def explanation(prompt: str, factor: str | None, extra: str = "") -> dict[str, Any]:
        level = int(_DECISION_RE.search(prompt).group(1))
        if unit(prompt, "level") > skill + 0.15:
            level = min(5, level + 1)                                  # przekręcony poziom
        hr = re.search(r"tętno (\d+)/min", prompt)
        hr_val = int(hr.group(1)) if hr else None
        if hr_val is not None and unit(prompt, "halu") > skill + 0.1:
            hr_val += 25                                               # zmyślona wartość
        hr_text = f"tętno {hr_val}/min [S1]" if hr_val is not None else "brak pomiaru tętna [S1]"
        rule = _RULE_RE.search(prompt)
        escalate = bool(rule and int(rule.group(1)) < level and unit(prompt, "esc") < skill)
        factor_text = f"Najważniejszym czynnikiem decyzji jest {factor.lower()} [S3]. " if factor else ""
        return {
            "poziom_esi": level,
            "uzasadnienie": (f"System wyznaczył poziom ESI {level} [S2]. U pacjenta stwierdzono {hr_text}. "
                             f"{factor_text}{extra}Decyzja jest zgodna z danymi z triażu."),
            "kluczowe_czynniki": [factor] if factor else ["parametry życiowe"],
            "sugestia_eskalacji": ({"czy": True, "poziom": int(rule.group(1)), "powod": "reguły bezpieczeństwa [S4]"}
                                   if escalate else {"czy": False, "poziom": None, "powod": None}),
        }

    def respond(req: dict[str, Any]) -> Reply:
        names = _tool_names(req)
        users = _users(req)
        prompt = users[0] if users else ""
        if "submit_explanation" not in names or not _DECISION_RE.search(prompt):
            return exam(req)
        think = "Analizuję dane pacjenta i decyzję systemu." if thinking else None
        if len(names) == 1:                                            # część esi: jedno narzędzie
            shap = re.search(r"\n1\. (.+?) = ", prompt)
            args = explanation(prompt, shap.group(1) if shap else None)
            if "Twoja odpowiedź nie była poprawna" in users[-1]:
                return Reply(content=json.dumps(args, ensure_ascii=False), thinking=think)
            if unit(prompt, "obey") > skill + 0.1:
                return Reply(content="Decyzja systemu wynika z parametrów życiowych pacjenta.", thinking=think)
            return Reply(tool_calls=[{"name": "submit_explanation", "arguments": args}], thinking=think)
        # część tools: agent
        results = [json.loads(m["content"]) for m in req.get("messages") or []
                   if isinstance(m, dict) and m.get("role") == "tool"]
        if not results:
            if unit(prompt, "agent") > skill:
                return Reply(tool_calls=[{"name": "get_patient_history", "arguments": {"id": 1}}], thinking=think)
            return Reply(tool_calls=[{"name": "explain_model", "arguments": {"top_n": 5}},
                                     {"name": "what_if", "arguments": {"changes": {"triage_vital_o2": 88}}}],
                         thinking=think)
        shap = next((r for r in results if r.get("czynniki")), None)
        wi = next((r for r in results if "poziom_esi_po" in r), None)
        factor = shap["czynniki"][0]["cecha"] if shap else None
        extra = (f"Gdyby SpO2 wynosiło 88%, system wyznaczyłby ESI {wi['poziom_esi_po']} [{wi['id_zrodla']}] "
                 f"— to scenariusz hipotetyczny. ") if wi and wi.get("ok") else ""
        args = explanation(prompt, factor, extra)
        if shap:
            args["uzasadnienie"] = args["uzasadnienie"].replace("[S3]", f"[{shap['id_zrodla']}]")
        return Reply(tool_calls=[{"name": "submit_explanation", "arguments": args}], thinking=think)

    return respond


def answer_key_from_frame(df: Any) -> dict[str, str]:
    """Klucz {pierwsza linia pytania: litera} z ramki w formacie NASK (kolumny question, answer)."""
    return {_stem(q): str(a).strip().upper() for q, a in zip(df["question"], df["answer"])}


# ─────────────────────────────────────────
# Uruchomienie samodzielne
# ─────────────────────────────────────────
def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Atrapa serwera Ollama (testy ręczne)")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--models", default="fake-a,fake-b", help="Modele po przecinku")
    parser.add_argument("--exam-csv", default=None, help="CSV w formacie NASK → odpowiedzi z kluczem")
    args = parser.parse_args(argv)

    key: dict[str, str] = {}
    if args.exam_csv:
        import pandas as pd

        key = answer_key_from_frame(pd.read_csv(args.exam_csv, encoding="utf-8"))
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    srv = FakeOllama(port=args.port).start()
    for i, name in enumerate(names):
        p = max(0.3, 0.85 - 0.2 * i)
        srv.add_model(name, triage_responder(key, skill=p, seed=i, thinking=True))
    print(f"Atrapa Ollamy: {srv.base_url}  (modele: {', '.join(names)}) — Ctrl+C kończy", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["ABSTAIN_TEXT", "FakeOllama", "Reply", "answer_key_from_frame", "exam_oracle", "fake_ollama",
           "triage_responder"]
