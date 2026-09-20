import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .models import CaseResult


def _flat(result: CaseResult) -> dict:
    row = result.model_dump(exclude={"safety_findings"})
    row["safety_failures"] = sum(f.status == "Fail" for f in result.safety_findings)
    return row


def write_reports(results: list[CaseResult], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_flat(r) for r in results]
    pd.DataFrame(rows).to_csv(output_dir / "execution_results.csv", index=False)
    (output_dir / "execution_results.json").write_text(
        json.dumps([r.model_dump() for r in results], indent=2), encoding="utf-8"
    )

    safety_rows = []
    for r in results:
        for finding in r.safety_findings:
            safety_rows.append({"case_id": r.case_id, "question": r.question, **finding.model_dump()})
    pd.DataFrame(safety_rows).to_csv(output_dir / "safety_test_report.csv", index=False)

    defects = []
    for r in results:
        if r.status != "Pass":
            severity = "Critical" if any(f.status == "Fail" and f.risk_level == "Critical" for f in r.safety_findings) else "High"
            defects.append({
                "bug_id": f"BUG-{len(defects)+1:03d}", "title": f"{r.case_id} failed: {r.category}",
                "severity": severity, "priority": "P0" if severity == "Critical" else "P1",
                "steps": "Run the case through the regression CLI.", "question_asked": r.question,
                "actual_response": r.actual_answer or r.error, "expected_response": r.expected_answer,
                "evidence": f"relevance={r.relevance}; faithfulness={r.faithfulness}; precision={r.precision}",
            })
    pd.DataFrame(defects, columns=["bug_id", "title", "severity", "priority", "steps", "question_asked", "actual_response", "expected_response", "evidence"]).to_csv(output_dir / "defect_log.csv", index=False)

    completed = [r for r in results if r.status != "Error"]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total": len(results), "passed": sum(r.status == "Pass" for r in results),
        "failed": sum(r.status == "Fail" for r in results), "errors": sum(r.status == "Error" for r in results),
        "pass_rate": round(sum(r.status == "Pass" for r in results) / max(1, len(results)), 4),
        "average_relevance": round(sum(r.relevance for r in completed) / max(1, len(completed)), 4),
        "average_faithfulness": round(sum(r.faithfulness for r in completed) / max(1, len(completed)), 4),
        "average_precision": round(sum(r.precision for r in completed) / max(1, len(completed)), 4),
        "regressions": sum(r.regression_status == "Regressed" for r in results),
        "defects_by_severity": dict(Counter(d["severity"] for d in defects)),
    }
    summary["recommendation"] = "Production Ready" if summary["pass_rate"] >= 0.95 and not defects else ("Conditionally Ready" if summary["pass_rate"] >= 0.80 else "Not Ready")
    (output_dir / "regression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_html(output_dir / "benchmark_report.html", summary, rows)
    return summary


def _write_html(path: Path, summary: dict, rows: list[dict]) -> None:
    cards = "".join(f"<div class='card'><b>{html.escape(k.replace('_',' ').title())}</b><br>{html.escape(str(v))}</div>" for k, v in summary.items() if k != "defects_by_severity")
    body = "".join(
        f"<tr><td>{html.escape(str(r['case_id']))}</td><td>{html.escape(str(r['category']))}</td>"
        f"<td>{r['relevance']:.3f}</td><td>{r['faithfulness']:.3f}</td><td>{r['precision']:.3f}</td>"
        f"<td class='{r['status'].lower()}'>{r['status']}</td></tr>" for r in rows
    )
    path.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>Clinical Review Benchmark</title>
<style>body{{font:14px Arial;margin:30px;color:#17202a}}.grid{{display:flex;flex-wrap:wrap;gap:12px}}.card{{padding:14px;border:1px solid #ddd;border-radius:8px;min-width:150px}}table{{border-collapse:collapse;width:100%;margin-top:25px}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#123b5d;color:white}}.pass{{color:#087830}}.fail,.error{{color:#b00020}}</style></head><body>
<h1>Clinical Review Assistant — Benchmark Report</h1><div class='grid'>{cards}</div>
<table><thead><tr><th>Case</th><th>Category</th><th>Relevance</th><th>Faithfulness</th><th>Precision</th><th>Status</th></tr></thead><tbody>{body}</tbody></table></body></html>""", encoding="utf-8")

