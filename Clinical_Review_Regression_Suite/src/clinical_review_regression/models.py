from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class GoldenCase(BaseModel):
    case_id: str
    category: str
    question: str
    expected_answer: str
    source_reference: str


class MetricScores(BaseModel):
    relevance: float = Field(ge=0, le=1)
    faithfulness: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    similarity: float = Field(ge=0, le=1)
    hallucination_rate: float = Field(ge=0, le=1)


class SafetyFinding(BaseModel):
    rule: str
    risk_level: Literal["Critical", "High", "Medium", "Low"]
    expected_result: str
    actual_result: str
    status: Literal["Pass", "Fail"]


class CaseResult(BaseModel):
    run_id: str
    case_id: str
    category: str
    question: str
    expected_answer: str
    source_reference: str
    actual_answer: str = ""
    relevance: float = 0.0
    faithfulness: float = 0.0
    precision: float = 0.0
    similarity: float = 0.0
    hallucination_rate: float = 1.0
    status: Literal["Pass", "Fail", "Error"] = "Error"
    regression_status: Literal["Stable", "Regressed", "No Baseline"] = "No Baseline"
    latency_seconds: float = 0.0
    error: str = ""
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    safety_findings: list[SafetyFinding] = Field(default_factory=list)

