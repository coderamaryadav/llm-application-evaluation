# Clinical Review Assistant Regression Suite

Reusable, dataset-driven UI regression suite for:

`https://llm-rag-testing-workshop.streamlit.app/Application`

It uploads the supplied clinical knowledge base, submits every golden-dataset
question, captures the chatbot response, calculates evaluation metrics, applies
safety rules, and writes CSV, JSON, HTML, defect, and summary reports.

## Project layout

```text
clinical_review_regression_suite/
├── data/
│   ├── clinical_review_golden_dataset.csv
│   └── clinical_review_knowledge_base.md
├── results/                    # generated run artifacts
├── src/clinical_review_regression/
│   ├── app_client.py           # Streamlit UI automation
│   ├── config.py               # environment configuration
│   ├── evaluators.py           # lexical and optional LLM judging
│   ├── models.py               # result schemas
│   ├── reporting.py            # CSV/JSON/HTML/defect reports
│   ├── runner.py               # orchestration and CLI
│   └── safety.py               # safety/compliance checks
├── tests/
├── .env.example
├── pyproject.toml
├── pytest.ini
└── requirements.txt
```

## Quick start

Requires Python 3.11+ and Chromium.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
python -m clinical_review_regression.runner run
```

On Windows PowerShell, set the module path before invoking the CLI:

```powershell
$env:PYTHONPATH = "src"
python -m clinical_review_regression.runner run
```

Useful commands:

```bash
PYTHONPATH=src python -m clinical_review_regression.runner validate-data
PYTHONPATH=src python -m clinical_review_regression.runner run --limit 3 --headed
PYTHONPATH=src python -m clinical_review_regression.runner run --category "Medication Questions"
pytest
```

Run the complete dataset and generate one consolidated report:

```powershell
$env:PYTHONPATH = "src"
python -m clinical_review_regression.runner run
```

The default run evaluates all 39 dataset cases. The `--limit` option is only
for debugging and intentionally runs a subset. Each run writes the complete
set of CSV/JSON artifacts and a single consolidated HTML report to
`results/benchmark_report.html`, replacing the previous run's artifacts.

## Scoring modes

- `lexical` (default): deterministic, free, regression-stable TF-IDF/claim-support
  scoring. No external model key is required.
- `openai`: optional rubric-based JSON judge using the OpenAI Responses API and
  `gpt-5.4-mini` by default. Set `EVALUATION_MODE=openai`, `OPENAI_API_KEY`,
  and optionally `OPENAI_MODEL` in `.env`. The UI answer,
  expected answer, and relevant KB are sent to the configured API.

### Configure `gpt-5.4-mini`

Copy `.env.example` to `.env`, then edit the local `.env` file:

```dotenv
EVALUATION_MODE=openai
OPENAI_MODEL=gpt-5.4-mini
OPENAI_API_KEY=your-rotated-project-key
```

Alternatively, set the variables in the shell so the key is not written to a
file. Never commit `.env`, paste a real key into source code, or include it in a
ZIP. If a key has been shared in chat or another exposed location, revoke it in
the OpenAI dashboard and create a replacement before running the suite.

The default thresholds match the assignment: relevance `0.80`, faithfulness
`0.85`, and context precision `0.80`. A test passes only when every required
metric meets its threshold and no safety rule fails.

## Reports

Every run writes the following artifacts directly under `results/`:

- `execution_results.csv`
- `execution_results.json`
- `regression_summary.json`
- `safety_test_report.csv`
- `defect_log.csv`
- `benchmark_report.html`

`results/baseline_results.csv` is optional. If present, the suite compares each
case with the baseline and flags metric drops greater than
`MAX_REGRESSION_DROP` (default `0.05`).

## Notes

- The supplied dataset contains 39 cases, so this suite evaluates all 39. It
  does not silently manufacture the 90 cases suggested by the assignment's
  target category counts.
- Streamlit markup can change. Selectors prioritize accessible labels and use
  narrowly-scoped fallbacks. Override text labels in `.env` if the app changes.
- Use small `--limit` runs while debugging to avoid unnecessary model traffic.
