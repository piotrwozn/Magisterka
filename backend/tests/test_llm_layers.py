"""Layer 1B (MedGemma) and Layer 2 (synthesis): ESI 1–5 prompts and answer parsing."""
from __future__ import annotations

import pytest

from app.models.schemas import MedGemmaAssessment, ModelPrediction, Vitals
from app.pipeline import layer1b_nlp, layer2_synthesis
from app.pipeline.layer1b_nlp import ESI_LABELS_PL, esi_to_category

VITALS = Vitals(age=67, temp=38.2, hr=118, sbp=95, dbp=62, rr=22, o2=94)


class FakeOllama:
    """Healthy Ollama returning a fixed JSON answer; records the prompts."""

    def __init__(self, answer: dict) -> None:
        self.answer = answer
        self.calls: list[dict] = []

    async def health(self) -> bool:
        return True

    async def chat_json(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return self.answer


def _mp(model: str, category: int) -> ModelPrediction:
    probs = [0.025] * 5
    probs[category] = 0.9
    return ModelPrediction(modelName=model, category=category, probabilities=probs, confidence=0.9)


def _mg(category: int) -> MedGemmaAssessment:
    return MedGemmaAssessment(category=category, confidence=0.9, reasoning="r", riskFlags=[], keyFindings=[])


class TestEsiLabels:
    def test_labels(self):
        assert ESI_LABELS_PL[0].startswith("ESI 1 · czerwony")
        assert ESI_LABELS_PL[1] == "ESI 2 · pomarańczowy (do 10 min)"
        assert ESI_LABELS_PL[4].startswith("ESI 5 · niebieski")

    @pytest.mark.parametrize("esi,expected", [(1, 0), (2, 1), (5, 4), ("3", 2), (0, 0), (9, 4), (None, 2), ("x", 2)])
    def test_esi_to_category(self, esi, expected):
        assert esi_to_category(esi, default=2) == expected


class TestMedGemma:
    async def test_llm_answer_in_esi_levels(self):
        client = FakeOllama({"esi": 2, "confidence": 0.8, "reasoning": "tachykardia",
                             "risk_flags": ["tachykardia"], "key_findings": ["HR=118"]})
        result = await layer1b_nlp.assess_async(2, VITALS, "ból w klatce", client=client, model="m")
        assert result.category == 1                        # ESI 2
        call = client.calls[0]
        assert call["schema"]["required"][0] == "esi"
        assert call["schema"]["properties"]["esi"] == {"type": "integer", "minimum": 1, "maximum": 5}
        assert "Emergency Severity Index" in call["system"]
        assert ESI_LABELS_PL[2] in call["user"]

    def test_stub_reasoning_mentions_esi(self):
        result = layer1b_nlp.stub_assessment(1, VITALS)
        assert result.category == 1
        assert result.reasoning.startswith("ESI 2")


class TestSynthesis:
    async def test_llm_recommendation_in_esi_levels(self):
        client = FakeOllama({"summary": "s", "recommended_esi": 1, "confidence": 0.7})
        result = await layer2_synthesis.synthesize(
            [_mp("a", 1), _mp("b", 3)], _mg(1), VITALS, client=client, model="q",
        )
        assert result is not None
        assert result.recommended_category == 0           # ESI 1
        call = client.calls[0]
        assert "recommended_esi" in call["schema"]["required"]
        assert ESI_LABELS_PL[3] in call["user"]
        assert "ESI 1 = natychmiastowa interwencja" in call["system"]

    async def test_offline_fallback_uses_esi_levels(self):
        result = await layer2_synthesis.synthesize([_mp("a", 1), _mp("b", 3)], _mg(2), VITALS,
                                                   client=None, model=None)
        assert result is not None
        assert result.recommended_category == 1
        assert "ESI 2–4" in result.summary
