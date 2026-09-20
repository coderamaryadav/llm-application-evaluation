from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .app_client import ClinicalReviewApp
from .config import Settings
from .evaluators import build_evaluator
from .models import CaseResult, GoldenCase
from .reporting import write_reports
from .safety import assess_safety

app = typer.Typer(help="Clinical Review Assistant regression suite")
REQUIRED_COLUMNS = {"Category", "Question", "Expected Answer", "Source Reference"}


def load_cases(path: Path) -> list[GoldenCase]:
    df = pd.read_csv(path).fillna("")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset contains no cases")
    if df["Question"].duplicated().any():
        dupes = df.loc[df["Question"].duplicated(), "Question"].tolist()
        raise ValueError(f"Duplicate questions found: {dupes[:3]}")
    return [GoldenCase(
        case_id=f"Q{i:03d}", category=row["Category"].strip(), question=row["Question"].strip(),
        expected_answer=row["Expected Answer"].strip(), source_reference=row["Source Reference"].strip(),
    ) for i, row in df.iterrows()]


def _baseline(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return pd.read_csv(path).set_index("case_id").to_dict(orient="index")


@app.command("validate-data")
def validate_data() -> None:
    settings = Settings()
    cases = load_cases(settings.dataset_path)
    counts = pd.Series([c.category for c in cases]).value_counts().to_dict()
    typer.echo(f"Valid dataset: {len(cases)} cases")
    for category, count in counts.items():
        typer.echo(f"  {category}: {count}")


@app.command()
def run(
    limit: Optional[int] = typer.Option(None, min=1, help="Run only the first N matching cases"),
    category: Optional[str] = typer.Option(None, help="Run one exact category"),
    headed: bool = typer.Option(False, help="Show Chromium while executing"),
) -> None:
    settings = Settings(headless=not headed) if headed else Settings()
    cases = load_cases(settings.dataset_path)
    if category:
        cases = [c for c in cases if c.category.casefold() == category.casefold()]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise typer.BadParameter("No dataset cases match the requested filters")

    kb = settings.knowledge_base_path.read_text(encoding="utf-8")
    evaluator = build_evaluator(settings)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_dir = settings.results_dir / run_id
    baseline = _baseline(settings.results_dir / "baseline_results.csv")
    results: list[CaseResult] = []

    with ClinicalReviewApp(settings) as client:
        client.upload_knowledge_base(settings.knowledge_base_path)
        for case in cases:
            typer.echo(f"[{case.case_id}] {case.question}")
            result = CaseResult(run_id=run_id, case_id=case.case_id, category=case.category,
                                question=case.question, expected_answer=case.expected_answer,
                                source_reference=case.source_reference)
            try:
                answer, latency = client.ask(case.question)
                scores = evaluator.score(case, answer, kb)
                result.actual_answer, result.latency_seconds = answer, latency
                for key, value in scores.model_dump().items():
                    setattr(result, key, value)
                result.safety_findings = assess_safety(case, answer)
                metrics_pass = (result.relevance >= settings.relevance_threshold and
                                result.faithfulness >= settings.faithfulness_threshold and
                                result.precision >= settings.precision_threshold)
                safety_pass = all(f.status == "Pass" for f in result.safety_findings)
                result.status = "Pass" if metrics_pass and safety_pass else "Fail"
                prior = baseline.get(case.case_id)
                if prior:
                    drops = [float(prior.get(m, 0)) - getattr(result, m) for m in ("relevance", "faithfulness", "precision")]
                    result.regression_status = "Regressed" if max(drops) > settings.max_regression_drop else "Stable"
            except Exception as exc:  # preserve the rest of the run
                result.error = f"{type(exc).__name__}: {exc}"
                result.status = "Error"
            results.append(result)

    summary = write_reports(results, output_dir)
    typer.echo(f"Reports: {output_dir.resolve()}")
    typer.echo(f"Pass rate: {summary['pass_rate']:.1%}; recommendation: {summary['recommendation']}")
    if summary["failed"] or summary["errors"] or summary["regressions"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

