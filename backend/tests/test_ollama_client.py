"""Tests for the OllamaClient — uses respx to mock HTTP calls."""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.inference.ollama_client import OllamaClient, OllamaError


BASE_URL = "http://localhost:11434"


@pytest.fixture
async def client():
    c = OllamaClient(BASE_URL, timeout_s=2.0)
    yield c
    await c.aclose()


class TestHealthCheck:

    @respx.mock
    async def test_health_returns_true_when_ollama_up(self, client):
        respx.get(f"{BASE_URL}/api/tags").respond(200, json={"models": []})
        assert await client.health(force=True) is True

    @respx.mock
    async def test_health_returns_false_on_timeout(self, client):
        respx.get(f"{BASE_URL}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
        assert await client.health(force=True) is False

    @respx.mock
    async def test_health_returns_false_on_500(self, client):
        respx.get(f"{BASE_URL}/api/tags").respond(500)
        assert await client.health(force=True) is False

    @respx.mock
    async def test_health_is_cached(self, client):
        route = respx.get(f"{BASE_URL}/api/tags").respond(200, json={"models": []})
        await client.health()
        await client.health()  # second call should be cached
        await client.health()
        assert route.call_count == 1   # only first call hits HTTP


class TestListModels:

    @respx.mock
    async def test_list_models(self, client):
        respx.get(f"{BASE_URL}/api/tags").respond(200, json={
            "models": [{"name": "llama3"}, {"name": "medgemma:27b"}],
        })
        models = await client.list_models()
        assert "llama3" in models
        assert "medgemma:27b" in models

    @respx.mock
    async def test_list_models_returns_empty_on_error(self, client):
        respx.get(f"{BASE_URL}/api/tags").respond(500)
        assert await client.list_models() == []


class TestChatJson:

    @respx.mock
    async def test_chat_json_returns_parsed_dict(self, client):
        respx.post(f"{BASE_URL}/api/chat").respond(200, json={
            "message": {
                "content": json.dumps({"chief_complaints": ["chestpain"], "altered_mental_status": False}),
            },
        })
        result = await client.chat_json(
            model="llama3", system="parse", user="ból w klatce",
            schema={"type": "object"},
        )
        assert result == {"chief_complaints": ["chestpain"], "altered_mental_status": False}

    @respx.mock
    async def test_chat_json_handles_code_fences(self, client):
        respx.post(f"{BASE_URL}/api/chat").respond(200, json={
            "message": {
                "content": '```json\n{"category": 1, "confidence": 0.8}\n```',
            },
        })
        result = await client.chat_json(
            model="medgemma:27b", system="assess", user="patient",
        )
        assert result == {"category": 1, "confidence": 0.8}

    @respx.mock
    async def test_chat_json_raises_on_empty(self, client):
        respx.post(f"{BASE_URL}/api/chat").respond(200, json={"message": {"content": ""}})
        with pytest.raises(OllamaError, match="Empty"):
            await client.chat_json(model="llama3", system="x", user="y")

    @respx.mock
    async def test_chat_json_raises_on_invalid_json(self, client):
        respx.post(f"{BASE_URL}/api/chat").respond(200, json={
            "message": {"content": "not even json"},
        })
        with pytest.raises(OllamaError, match="parse JSON"):
            await client.chat_json(model="llama3", system="x", user="y")

    @respx.mock
    async def test_chat_json_raises_on_http_error(self, client):
        respx.post(f"{BASE_URL}/api/chat").respond(500)
        with pytest.raises(OllamaError):
            await client.chat_json(model="llama3", system="x", user="y")
