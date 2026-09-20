import json
import re
from abc import ABC, abstractmethod

from openai import OpenAI
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Settings
from .models import GoldenCase, MetricScores


def _clip(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 2 and t not in ENGLISH_STOP_WORDS}


def _cosine(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform([a, b])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
    except ValueError:
        return 0.0


class Evaluator(ABC):
    @abstractmethod
    def score(self, case: GoldenCase, actual: str, knowledge_base: str) -> MetricScores: ...


class LexicalEvaluator(Evaluator):
    """Deterministic local scorer intended for stable CI regression signals."""

    def score(self, case: GoldenCase, actual: str, knowledge_base: str) -> MetricScores:
        expected_terms = _terms(case.expected_answer)
        actual_terms = _terms(actual)
        question_terms = _terms(case.question)

        precision = len(expected_terms & actual_terms) / max(1, len(expected_terms))
        question_coverage = len(question_terms & actual_terms) / max(1, len(question_terms))
        similarity = _cosine(case.expected_answer, actual)
        relevance = 0.65 * similarity + 0.35 * question_coverage

        kb_terms = _terms(knowledge_base)
        supported = len(actual_terms & kb_terms) / max(1, len(actual_terms))
        expected_support = len(actual_terms & expected_terms) / max(1, len(actual_terms))
        faithfulness = 0.75 * supported + 0.25 * expected_support
        hallucination = 1.0 - supported
        return MetricScores(
            relevance=_clip(relevance),
            faithfulness=_clip(faithfulness),
            precision=_clip(precision),
            similarity=_clip(similarity),
            hallucination_rate=_clip(hallucination),
        )


class OpenAIJudge(Evaluator):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EVALUATION_MODE=openai")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def score(self, case: GoldenCase, actual: str, knowledge_base: str) -> MetricScores:
        rubric = {
            "task": "Score a clinical RAG answer. Return JSON only with five numbers from 0 to 1.",
            "definitions": {
                "relevance": "Directly answers the question without irrelevant content.",
                "faithfulness": "Every material claim is supported by the supplied knowledge base.",
                "precision": "Clinical details agree with the expected answer.",
                "similarity": "Semantic similarity to the expected answer.",
                "hallucination_rate": "Fraction of material claims unsupported by the knowledge base.",
            },
            "question": case.question,
            "expected_answer": case.expected_answer,
            "actual_answer": actual,
            "knowledge_base": knowledge_base,
        }
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "low"},
            instructions=(
                "You are a strict clinical AI quality evaluator. Return only a valid JSON "
                "object containing relevance, faithfulness, precision, similarity, and "
                "hallucination_rate. Every value must be a number from 0 to 1."
            ),
            input=json.dumps(rubric),
        )
        raw = (response.output_text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"OpenAI judge returned invalid JSON: {raw[:200]!r}") from exc
        missing = set(MetricScores.model_fields) - set(payload)
        if missing:
            raise ValueError(f"OpenAI judge response is missing metrics: {sorted(missing)}")
        return MetricScores(**{key: _clip(payload[key]) for key in MetricScores.model_fields})


def build_evaluator(settings: Settings) -> Evaluator:
    return OpenAIJudge(settings) if settings.evaluation_mode == "openai" else LexicalEvaluator()
