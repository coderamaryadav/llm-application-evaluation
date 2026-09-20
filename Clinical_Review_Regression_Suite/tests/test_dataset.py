from pathlib import Path

from clinical_review_regression.runner import load_cases


def test_supplied_dataset_is_valid():
    cases = load_cases(Path("data/clinical_review_golden_dataset.csv"))
    assert len(cases) == 39
    assert all(c.question and c.expected_answer and c.source_reference for c in cases)

