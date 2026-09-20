from clinical_review_regression.evaluators import LexicalEvaluator
from clinical_review_regression.models import GoldenCase


def test_identical_grounded_answer_scores_high():
    expected = "State that medication duration is not documented and do not infer duration."
    case = GoldenCase(case_id="Q001", category="Medication Questions", question="What if duration is missing?", expected_answer=expected, source_reference="Medication Duration")
    scores = LexicalEvaluator().score(case, expected, expected)
    assert scores.precision == 1.0
    assert scores.faithfulness >= 0.95
    assert scores.similarity >= 0.99


def test_empty_answer_scores_zero():
    case = GoldenCase(case_id="Q001", category="Factual Clinical Questions", question="Purpose?", expected_answer="Support reviews", source_reference="Section 1")
    scores = LexicalEvaluator().score(case, "", "Support reviews")
    assert scores.relevance == scores.faithfulness == scores.precision == 0

