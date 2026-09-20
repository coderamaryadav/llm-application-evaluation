from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_url: str = "https://llm-rag-testing-workshop.streamlit.app/Application"
    dataset_path: Path = Path("data/clinical_review_golden_dataset.csv")
    knowledge_base_path: Path = Path("data/clinical_review_knowledge_base.md")
    results_dir: Path = Path("results")
    headless: bool = True
    question_timeout_seconds: int = Field(default=90, ge=10)
    page_load_timeout_seconds: int = Field(default=120, ge=10)
    max_retries: int = Field(default=2, ge=0, le=5)
    relevance_threshold: float = Field(default=0.80, ge=0, le=1)
    faithfulness_threshold: float = Field(default=0.85, ge=0, le=1)
    precision_threshold: float = Field(default=0.80, ge=0, le=1)
    max_regression_drop: float = Field(default=0.05, ge=0, le=1)
    evaluation_mode: Literal["lexical", "openai"] = "lexical"
    openai_model: str = "gpt-5.4-mini"
    openai_api_key: str | None = None

    question_placeholder: str = "Ask a clinical question based on uploaded context..."
    upload_label: str = "Upload TXT or Markdown files"
