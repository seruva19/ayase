"""Serialize pipeline validation reports in the public export formats."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import Pipeline


def export_report(pipeline: "Pipeline", path: Path, format: str = "json") -> None:
    """Export a detailed validation report.

    Args:
        path: Output file path
        format: 'json', 'csv', or 'html'
    """
    if format not in {"json", "csv", "html"}:
        raise ValueError(f"Unsupported report format: {format!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        with open(path, "w", encoding="utf-8") as f:
            data = {
                "run_status": pipeline.get_run_status(),
                "stats": pipeline.stats.model_dump(),
                "samples": [pipeline.dump_sample(s) for s in pipeline.results.values()],
            }
            json.dump(data, f, indent=2, default=str)

    elif format == "csv":
        import csv

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["Path", "Valid", "Issues", "Recommendations", "Technical Score"])

            for s in pipeline.results.values():
                issues_str = "; ".join([i.message for i in s.validation_issues])
                recs_str = "; ".join(
                    [i.recommendation for i in s.validation_issues if i.recommendation]
                )
                score = (
                    s.quality_metrics.canonical_model_dump().get("technical_score")
                    if s.quality_metrics
                    else None
                ) or 0.0

                writer.writerow([str(s.path), s.is_valid, issues_str, recs_str, f"{score:.2f}"])

    elif format == "html":
        # Simple HTML report
        html = [
            "<html><head><title>Ayase Validation Report</title>",
            "<style>body{font-family:sans-serif} .error{color:red} .warn{color:orange}</style>",
            "</head><body>",
            f"<h1>Validation Report</h1>",
            f"<p>Total: {pipeline.stats.total_samples} | Valid: {pipeline.stats.valid_samples} | Invalid: {pipeline.stats.invalid_samples}</p>",
            "<table border='1'><tr><th>Path</th><th>Status</th><th>Issues</th><th>Recommendations</th></tr>",
        ]

        from html import escape as _esc

        for s in pipeline.results.values():
            status_color = "green" if s.is_valid else "red"
            issues_html = (
                "<ul>"
                + "".join([f"<li>{_esc(i.message)}</li>" for i in s.validation_issues])
                + "</ul>"
            )
            recs_html = (
                "<ul>"
                + "".join(
                    [
                        f"<li>{_esc(i.recommendation)}</li>"
                        for i in s.validation_issues
                        if i.recommendation
                    ]
                )
                + "</ul>"
            )

            html.append(
                f"<tr><td>{_esc(s.path.name)}</td><td style='color:{status_color}'>{s.is_valid}</td><td>{issues_html}</td><td>{recs_html}</td></tr>"
            )

        html.append("</table></body></html>")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
