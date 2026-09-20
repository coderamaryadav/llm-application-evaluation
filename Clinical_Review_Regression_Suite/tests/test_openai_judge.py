import json
from types import SimpleNamespace

from clinical_review_regression.config import Settings
from clinical_review_regression.evaluators import OpenAIJudge
from clinical_review_regression.models import GoldenCase


class FakeResponses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text=json.dumps({
            "relevance": 0.9,
            "faithfulness": 0.95,
            "precision": 0.88,
            "similarity": 0.91,
            "hallucination_rate": 0.05,
        }))


def test_openai_judge_uses_responses_api_and_requested_model():
    settings = Settings(evaluation_mode="openai", openai_api_key="test-key")
    # Construct without creating a real HTTP client; this unit test must never
    # make a network request or depend on workstation proxy configuration.
    judge = OpenAIJudge.__new__(OpenAIJudge)
    judge.model = settings.openai_model
    fake = FakeResponses()
    judge.client = SimpleNamespace(responses=fake)
    case = GoldenCase(case_id="Q001", category="Factual", question="Purpose?", expected_answer="Support reviews.", source_reference="Section 1")

    scores = judge.score(case, "Support reviews.", "Support reviews.")

    assert fake.kwargs["model"] == "gpt-5.4-mini"
    assert fake.kwargs["reasoning"] == {"effort": "low"}
    assert scores.faithfulness == 0.95
