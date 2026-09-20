import os
from pathlib import Path

import pytest

from clinical_review_regression.app_client import ClinicalReviewApp
from clinical_review_regression.config import Settings


@pytest.mark.live
@pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="Set RUN_LIVE_TESTS=1 to call the deployed application")
def test_live_application_returns_an_answer():
    settings = Settings()
    with ClinicalReviewApp(settings) as client:
        client.upload_knowledge_base(Path("data/clinical_review_knowledge_base.md"))
        answer, latency = client.ask("What is the purpose of the Clinical Review Assistant?")
    assert answer
    assert latency < settings.question_timeout_seconds

