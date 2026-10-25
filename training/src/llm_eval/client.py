"""
Klient HTTP Ollamy dla porównania modeli językowych (bake-off) — lokalny demon albo chmura.

    from src.llm_eval.client import OllamaChat

    client = OllamaChat()                                   # lokalny demon (po `ollama signin`)
    client = OllamaChat("https://ollama.com", api_key=os.environ["OLLAMA_API_KEY"])  # bezpośrednio
    res = client.chat("gpt-oss:120b-cloud", [{"role": "user", "content": "…"}], think="low")
    res.content, res.thinking, res.tool_calls, res.prompt_tokens, res.latency_s, res.error

    results = client.chat_many([ChatJob(model, messages), …], on_result=callback)  # ≤ concurrency naraz

Tryby
    lokalny  — base_url http://localhost:11434; modele chmurowe mają tagi „…:cloud” / „…-cloud”,
               demon przekazuje je do ollama.com (wymaga `ollama signin`).
    chmura   — base_url https://ollama.com (+ nagłówek Authorization: Bearer <OLLAMA_API_KEY>);
               nazwy modeli BEZ przyrostka chmurowego — klient obcina go sam
               („glm-5.3:cloud” → „glm-5.3”, „gpt-oss:120b-cloud” → „gpt-oss:120b”).

Odporność (nocne przebiegi)
    - 429 / 5xx / 408 / przekroczony czas / zerwane połączenie → ponowienia z wykładniczym
      odstępem i losowym rozrzutem (nagłówek Retry-After ma pierwszeństwo), do `max_retries` razy;
    - `chat()` NIGDY nie rzuca wyjątku z powodu modelu ani sieci — wypełnia `error`;
    - 400 z komunikatem o „think” (model bez poziomów rozumowania) → jedno ponowienie bez `think`
      (odnotowane w `notes`);
    - `max_consecutive_failures` kolejnych błędów jednego modelu → kolejne zapytania do niego
      kończą się od razu błędem (bez czekania godzinami na padnięty model); błędy nie trafiają do
      pamięci podręcznej, więc ponowne uruchomienie spróbuje jeszcze raz;
    - liczba równoległych zapytań HTTP ograniczona semaforem `concurrency` (plan Pro: 3) —
      także gdy `chat()` wołają własne wątki części ESI / narzędzi.

Parametr `format` (structured outputs) jest IGNOROWANY przez modele chmurowe — klient go nie
używa; strukturę wymusza się narzędziem `submit_*` albo tolerancyjnym parsowaniem.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields
from typing import Any
from urllib.parse import urlparse

import requests

# ─────────────────────────────────────────
# Stałe
# ─────────────────────────────────────────
DEFAULT_BASE_URL: str = "http://localhost:11434"
CLOUD_BASE_URL: str = "https://ollama.com"
THINK_LEVELS: tuple[str, ...] = ("low", "medium", "high", "max")
RETRY_STATUSES: frozenset[int] = frozenset({408, 409, 425, 429})     # + każdy 5xx
_THINK_TAG = re.compile(r"^\s*<think(?:ing)?>(.*?)</think(?:ing)?>\s*", re.DOTALL | re.IGNORECASE)


# ─────────────────────────────────────────
# Nazwy modeli i adresy
# ─────────────────────────────────────────
def normalize_base_url(base_url: str | None) -> str:
    """'https://ollama.com/api/' → 'https://ollama.com'; brak schematu → http://."""
    url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    if url.endswith("/api"):
        url = url[: -len("/api")]
    if "://" not in url:
        url = "http://" + url
    return url


def is_cloud_url(base_url: str) -> bool:
    """Czy adres to bezpośrednie API chmury Ollama (ollama.com)."""
    host = (urlparse(normalize_base_url(base_url)).hostname or "").lower()
    return host == "ollama.com" or host.endswith(".ollama.com")


def canonical_model(model: str) -> str:
    """Nazwa modelu bez przyrostka chmurowego: 'glm-5.3:cloud' → 'glm-5.3', 'gpt-oss:120b-cloud' → 'gpt-oss:120b'."""
    name = str(model).strip()
    if ":" in name:
        base, tag = name.rsplit(":", 1)
        if tag == "cloud":
            return base
        if tag.endswith("-cloud"):
            return f"{base}:{tag[: -len('-cloud')]}"
    elif name.endswith("-cloud"):
        return name[: -len("-cloud")]
    return name


def parse_think(value: Any) -> bool | str | None:
    """
    Wartość pola `think` z CLI/konfiguracji: None/''/'default'/'auto' → None (domyślne modelu),
    'true'/'on' → True, 'false'/'off'/'none' → False, 'low'|'medium'|'high'|'max' → ten poziom.
    """
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("", "default", "auto"):
        return None
    if text in ("true", "on", "yes", "1"):
        return True
    if text in ("false", "off", "no", "0", "none"):
        return False
    if text in THINK_LEVELS:
        return text
    raise ValueError(f"Nieznana wartość think '{value}' (low | medium | high | max | true | false | default)")


def split_think_tags(content: str) -> tuple[str, str | None]:
    """Treść zaczynająca się od <think>…</think> (część modeli przez API zgodne) → (odpowiedź, rozumowanie)."""
    match = _THINK_TAG.match(content or "")
    if not match:
        return content or "", None
    return content[match.end():], match.group(1).strip() or None


# ─────────────────────────────────────────
# Wynik i zadanie
# ─────────────────────────────────────────
@dataclass
class ChatResult:
    """
    Wynik jednego zapytania /api/chat.

    content           — odpowiedź (bez rozumowania),
    tool_calls        — [{"name": str, "arguments": dict}] (+ "arguments_raw", gdy argumenty nie
                        były poprawnym JSON-em; wtedy arguments = {}),
    thinking          — ślad rozumowania (message.thinking) albo None,
    prompt_tokens / completion_tokens — prompt_eval_count / eval_count (None, gdy brak),
    latency_s         — czas ostatniej (udanej) próby HTTP,
    error             — None = sukces; tekst błędu po wyczerpaniu ponowień,
    raw               — pełna odpowiedź JSON (przy błędzie: {"status", "body"}).
    Pola dodatkowe: model (tag z zapytania), attempts, status, total_s (z ponowieniami), notes.
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    thinking: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    attempts: int = 0
    status: int | None = None
    total_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatResult:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names})


@dataclass
class ChatJob:
    """Jedno zapytanie dla `chat_many` (key — dowolny identyfikator wywołującego)."""

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    think: bool | str | None = None
    options: dict[str, Any] | None = None
    key: Any = None


def _as_job(job: ChatJob | dict[str, Any]) -> ChatJob:
    if isinstance(job, ChatJob):
        return job
    known = {f.name for f in fields(ChatJob)}
    return ChatJob(**{k: v for k, v in job.items() if k in known})


def _parse_tool_call(call: Any) -> dict[str, Any]:
    """Wywołanie narzędzia w formacie Ollamy/OpenAI → {"name", "arguments": dict}."""
    if not isinstance(call, dict):
        return {"name": None, "arguments": {}, "arguments_raw": str(call)}
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    out: dict[str, Any] = {"name": fn.get("name"), "arguments": {}}
    args = fn.get("arguments")
    if isinstance(args, dict):
        out["arguments"] = args
    elif isinstance(args, str):
        try:
            parsed = json.loads(args) if args.strip() else {}
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            out["arguments"] = parsed
        else:
            out["arguments_raw"] = args
    elif args is not None:
        out["arguments_raw"] = json.dumps(args, ensure_ascii=False)
    if call.get("id"):
        out["id"] = call["id"]
    return out


def _merge_ndjson(text: str) -> dict[str, Any] | None:
    """Odpowiedź strumieniowa (NDJSON) sklejona w jeden obiekt jak przy stream=false."""
    content, thinking, tool_calls, last = [], [], [], None
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except ValueError:
            return None
        if not isinstance(chunk, dict):
            return None
        if "error" in chunk:
            return chunk
        msg = chunk.get("message") or {}
        content.append(msg.get("content") or "")
        thinking.append(msg.get("thinking") or "")
        tool_calls.extend(msg.get("tool_calls") or [])
        last = chunk
    if last is None:
        return None
    merged = dict(last)
    merged["message"] = {"role": "assistant", "content": "".join(content), "thinking": "".join(thinking),
                         "tool_calls": tool_calls}
    return merged


# ─────────────────────────────────────────
# Klient
# ─────────────────────────────────────────
class OllamaChat:
    """Klient /api/chat z ponowieniami, limitem równoległości i licznikiem zużycia (zob. docstring modułu)."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout_s: float = 180,
        max_retries: int = 5,
        concurrency: int = 3,
        *,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        info_timeout_s: float = 15.0,
        max_consecutive_failures: int = 20,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self.cloud = is_cloud_url(self.base_url)
        self.api_key = api_key or None
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.concurrency = max(1, int(concurrency))
        self.backoff_base_s = max(0.0, float(backoff_base_s))
        self.backoff_max_s = max(0.0, float(backoff_max_s))
        self.info_timeout_s = float(info_timeout_s)
        self.max_consecutive_failures = max(0, int(max_consecutive_failures or 0))
        self._sem = threading.BoundedSemaphore(self.concurrency)
        self._lock = threading.Lock()
        self._fail_streak: dict[str, int] = {}
        self._last_error: dict[str, str] = {}
        self._tags: list[dict[str, Any]] | None = None
        self.usage: dict[str, dict[str, float]] = {}

    # ── pomocnicze ──
    @property
    def mode(self) -> str:
        return "cloud" if self.cloud else "local"

    def api_model(self, model: str) -> str:
        """Nazwa wysyłana do serwera (w trybie chmury bez przyrostka ':cloud' / '-cloud')."""
        return canonical_model(model) if self.cloud else str(model)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(self.backoff_max_s, max(0.0, float(retry_after)))
            except ValueError:
                pass
        base = min(self.backoff_max_s, self.backoff_base_s * (2 ** max(0, attempt - 1)))
        return base * (0.5 + random.random())

    def _record(self, model: str, res: ChatResult) -> None:
        with self._lock:
            u = self.usage.setdefault(model, {"calls": 0, "errors": 0, "prompt_tokens": 0,
                                              "completion_tokens": 0, "latency_s": 0.0})
            u["calls"] += 1
            u["errors"] += 0 if res.ok else 1
            u["prompt_tokens"] += res.prompt_tokens or 0
            u["completion_tokens"] += res.completion_tokens or 0
            u["latency_s"] += res.latency_s or 0.0
            if res.ok:
                self._fail_streak[model] = 0
            else:
                self._fail_streak[model] = self._fail_streak.get(model, 0) + 1
                self._last_error[model] = res.error or ""

    def _tripped(self, model: str) -> str | None:
        if not self.max_consecutive_failures:
            return None
        with self._lock:
            streak = self._fail_streak.get(model, 0)
            if streak >= self.max_consecutive_failures:
                return (f"pominięto: model wyłączony po {streak} kolejnych błędach "
                        f"(ostatni: {self._last_error.get(model, '?')[:200]})")
        return None

    def reset_breaker(self, model: str | None = None) -> None:
        """Zeruje licznik kolejnych błędów (jednego modelu albo wszystkich)."""
        with self._lock:
            if model is None:
                self._fail_streak.clear()
            else:
                self._fail_streak.pop(model, None)

    @staticmethod
    def _error_text(status: int, payload: Any, text: str) -> str:
        detail = payload.get("error") if isinstance(payload, dict) and payload.get("error") else text
        detail = " ".join(str(detail or "").split())[:300]
        return f"HTTP {status}: {detail}" if detail else f"HTTP {status}"

    def _auth_hint(self, status: int) -> str:
        if status not in (401, 403):
            return ""
        if self.cloud:
            return " — sprawdź klucz API (zmienna OLLAMA_API_KEY, https://ollama.com/settings/keys)"
        return " — zaloguj lokalny demon do chmury: `ollama signin`"

    # ── /api/chat ──
    def chat(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        think: bool | str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Jedno zapytanie (bez strumieniowania) z ponowieniami; nie rzuca wyjątków z powodu modelu/sieci."""
        tripped = self._tripped(model)
        if tripped:
            res = ChatResult(error=tripped, model=model)
            self._record(model, res)
            return res

        body: dict[str, Any] = {"model": self.api_model(model), "messages": list(messages), "stream": False}
        if tools:
            body["tools"] = tools
        if think is not None:
            body["think"] = think
        if options:
            body["options"] = options

        url = f"{self.base_url}/api/chat"
        notes: list[str] = []
        attempts = 0
        started = time.perf_counter()
        last_error, last_status, last_raw = "nieznany błąd", None, {}
        while True:
            attempts += 1
            retry_after: str | None = None
            retryable = False
            with self._sem:
                t0 = time.perf_counter()
                try:
                    resp = requests.post(url, json=body, headers=self._headers(),
                                         timeout=(min(15.0, self.timeout_s), self.timeout_s))
                    latency = time.perf_counter() - t0
                    exc_text = None
                except requests.Timeout:
                    exc_text, retryable = f"przekroczony czas odpowiedzi ({self.timeout_s:.0f} s)", True
                except requests.ConnectionError as exc:
                    exc_text, retryable = f"błąd połączenia z {self.base_url}: {exc.__class__.__name__}", True
                except requests.RequestException as exc:
                    exc_text = f"błąd zapytania: {exc}"
            if exc_text is not None:
                last_error, last_status, last_raw = exc_text, None, {}
            else:
                status = resp.status_code
                text = resp.text
                try:
                    payload = resp.json()
                except ValueError:
                    payload = _merge_ndjson(text) if status == 200 else None
                if status == 200 and isinstance(payload, dict) and "error" not in payload:
                    res = self._parse(payload, model, latency)
                    res.attempts, res.status, res.notes = attempts, status, notes
                    res.total_s = time.perf_counter() - started
                    self._record(model, res)
                    return res
                last_status = status
                last_error = self._error_text(status, payload, text) + self._auth_hint(status)
                last_raw = {"status": status, "body": text[:2000]}
                if status == 400 and "think" in body and "think" in last_error.lower():
                    notes.append(f"model odrzucił think={body['think']!r} — ponowienie bez think")
                    body.pop("think")
                    attempts -= 1
                    continue
                retryable = status in RETRY_STATUSES or status >= 500 or status == 200
                retry_after = resp.headers.get("Retry-After")
            if not retryable or attempts > self.max_retries:
                break
            time.sleep(self._backoff(attempts, retry_after))

        res = ChatResult(error=last_error, raw=last_raw, model=model, attempts=attempts, status=last_status,
                         total_s=time.perf_counter() - started, notes=notes)
        self._record(model, res)
        return res

    @staticmethod
    def _parse(payload: dict[str, Any], model: str, latency: float) -> ChatResult:
        msg = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        content = msg.get("content") if isinstance(msg.get("content"), str) else ""
        thinking = msg.get("thinking") if isinstance(msg.get("thinking"), str) else None
        if not thinking:
            content, thinking = split_think_tags(content)
        calls = [_parse_tool_call(tc) for tc in (msg.get("tool_calls") or [])]

        def _int(value: Any) -> int | None:
            return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

        return ChatResult(content=content, tool_calls=calls, thinking=thinking or None,
                          prompt_tokens=_int(payload.get("prompt_eval_count")),
                          completion_tokens=_int(payload.get("eval_count")),
                          latency_s=float(latency), raw=payload, model=model)

    def chat_many(
        self,
        jobs: Iterable[ChatJob | dict[str, Any]],
        *,
        on_result: Callable[[int, ChatJob, ChatResult], None] | None = None,
    ) -> list[ChatResult]:
        """
        Zapytania w puli `concurrency` wątków; wyniki w kolejności zadań. on_result(i, job, wynik)
        wołane w wątku wywołującym zaraz po ukończeniu każdego zadania (np. zapis do pamięci
        podręcznej — przerwanie Ctrl+C nie gubi gotowych odpowiedzi).
        """
        job_list = [_as_job(j) for j in jobs]
        results: list[ChatResult | None] = [None] * len(job_list)
        if not job_list:
            return []
        pool = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="ollama")
        futures: dict[Future, int] = {}
        try:
            for i, job in enumerate(job_list):
                futures[pool.submit(self.chat, job.model, job.messages, tools=job.tools, think=job.think,
                                    options=job.options)] = i
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:          # błąd programistyczny w wątku — nie przerywa całości
                    res = ChatResult(error=f"wyjątek klienta: {exc!r}", model=job_list[i].model)
                results[i] = res
                if on_result is not None:
                    on_result(i, job_list[i], res)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        return [r if r is not None else ChatResult(error="brak wyniku", model=j.model)
                for r, j in zip(results, job_list)]

    # ── informacje o serwerze i modelach ──
    def _get_json(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[Any, str | None]:
        try:
            resp = requests.request(method, f"{self.base_url}{path}", json=body, headers=self._headers(),
                                    timeout=self.info_timeout_s)
        except requests.RequestException as exc:
            return None, f"{exc.__class__.__name__}: {exc}"
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        if resp.status_code != 200:
            return payload, self._error_text(resp.status_code, payload, resp.text) + self._auth_hint(resp.status_code)
        return payload, None

    def server_info(self) -> dict[str, Any]:
        """{"base_url", "mode", "version", "auth", "error"} — GET /api/version (nie rzuca wyjątków)."""
        payload, err = self._get_json("GET", "/api/version")
        version = payload.get("version") if isinstance(payload, dict) else None
        return {"base_url": self.base_url, "mode": self.mode, "version": version,
                "auth": "bearer" if self.api_key else "none", "error": err}

    def _digest(self, api_model: str) -> str | None:
        if self._tags is None:
            payload, err = self._get_json("GET", "/api/tags")
            self._tags = payload.get("models", []) if isinstance(payload, dict) and not err else []
        wanted = {api_model, f"{api_model}:latest"}
        for item in self._tags or []:
            if isinstance(item, dict) and (item.get("name") in wanted or item.get("model") in wanted):
                return item.get("digest")
        return None

    def model_info(self, model: str) -> dict[str, Any]:
        """Podsumowanie /api/show (+ digest z /api/tags, jeśli jest); nie rzuca wyjątków."""
        api_model = self.api_model(model)
        payload, err = self._get_json("POST", "/api/show", {"model": api_model})
        out: dict[str, Any] = {"model": model, "api_model": api_model, "error": err}
        if not isinstance(payload, dict) or err:
            return out
        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        out["details"] = {k: details.get(k) for k in ("family", "parameter_size", "quantization_level", "format",
                                                       "context_length", "parent_model") if details.get(k)}
        info = payload.get("model_info") if isinstance(payload.get("model_info"), dict) else {}
        extra = {k: v for k, v in info.items()
                 if k in ("general.architecture", "general.parameter_count") or k.endswith(".context_length")}
        if extra:
            out["model_info"] = extra
        for key in ("capabilities", "remote_model", "remote_host", "modified_at", "thinking", "license"):
            value = payload.get(key)
            if value:
                out[key] = value if key != "license" else str(value).splitlines()[0][:120]
        out["digest"] = payload.get("digest") or self._digest(api_model)
        return out


__all__ = [
    "CLOUD_BASE_URL",
    "ChatJob",
    "ChatResult",
    "DEFAULT_BASE_URL",
    "OllamaChat",
    "THINK_LEVELS",
    "canonical_model",
    "is_cloud_url",
    "normalize_base_url",
    "parse_think",
    "split_think_tags",
]
