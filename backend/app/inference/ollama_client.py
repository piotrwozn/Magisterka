"""Async HTTP client for a local Ollama server.

Supports:
- `/api/chat` with structured output (JSON schema)
- `/api/tags` for health check
- Graceful fallback: if Ollama is unreachable, callers can use deterministic stubs.

Local Ollama install: https://ollama.com
Pull models: `ollama pull llama3`, `ollama pull medgemma:27b`, `ollama pull qwen3.6`
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised on any unrecoverable error talking to Ollama."""


class OllamaClient:
    """Thin async wrapper around the Ollama REST API."""

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_s, connect=3.0)
        self._client: httpx.AsyncClient | None = None
        self._healthy: bool | None = None  # tri-state: None=unknown, True/False once checked
        self._healthy_at: float = 0.0
        self._health_ttl_s: float = 60.0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, base_url=self.base_url)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Health ────────────────────────────────────────────────────────

    async def health(self, force: bool = False) -> bool:
        """Return True if Ollama responds. Cached for `_health_ttl_s` seconds."""
        now = asyncio.get_running_loop().time()
        if not force and self._healthy is not None and (now - self._healthy_at) < self._health_ttl_s:
            return self._healthy

        try:
            client = await self._get_client()
            resp = await client.get("/api/tags", timeout=httpx.Timeout(3.0))
            self._healthy = resp.status_code == 200
        except Exception as exc:
            log.debug("Ollama health check failed: %s", exc)
            self._healthy = False
        self._healthy_at = now
        return self._healthy

    async def list_models(self) -> list[str]:
        try:
            client = await self._get_client()
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    # ── Chat / structured output ──────────────────────────────────────

    async def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Call /api/chat and parse the assistant reply as JSON.

        When `schema` is provided we ask Ollama for structured output (format=schema)
        — this enforces a valid JSON return on supported models.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":  False,
            "options": {"temperature": temperature, "seed": seed},
        }
        if schema is not None:
            payload["format"] = schema

        client = await self._get_client()
        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama HTTP error: {exc}") from exc

        body = resp.json()
        content = body.get("message", {}).get("content", "")
        if not content:
            raise OllamaError("Empty Ollama response")

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            # Sometimes models wrap JSON in ```json fences; try to recover.
            stripped = content.strip().strip("`").strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                raise OllamaError(f"Could not parse JSON: {content[:200]}") from exc


# ── Module-level singleton (lazy) ────────────────────────────────────

_singleton: OllamaClient | None = None


def get_ollama_client(base_url: str, timeout_s: float = 30.0) -> OllamaClient:
    global _singleton
    if _singleton is None or _singleton.base_url != base_url.rstrip("/"):
        _singleton = OllamaClient(base_url=base_url, timeout_s=timeout_s)
    return _singleton
