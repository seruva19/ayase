from pathlib import Path

from ayase.models import QualityMetrics


def _load_metrics_rows():
    readme = Path("README.md").read_text(encoding="utf-8")
    rows = []
    in_metrics = False
    for line in readme.splitlines():
        if line.strip() == "## Metrics":
            in_metrics = True
            continue
        if in_metrics and line.startswith("## "):
            break
        if in_metrics and line.startswith("|"):
            cols = [col.strip() for col in line.strip().strip("|").split("|")]
            if not cols:
                continue
            if cols[0] == "#" or cols[0].startswith("---"):
                continue
            if len(cols) >= 5:
                rows.append(cols[:5])
    return rows


def test_metrics_table_matches_quality_metrics():
    """Verify the README metrics table has one row per QualityMetrics field,
    in the correct order, with the 5-column API-reference format:
    #, Metric, Module, Input, Description."""
    rows = _load_metrics_rows()
    fields = list(QualityMetrics.model_fields.keys())
    assert len(rows) == len(fields), f"Table has {len(rows)} rows but model has {len(fields)} fields"
    metrics = [row[1] for row in rows]
    assert metrics == fields
    for index, row in enumerate(rows, 1):
        number, metric, module, inp, desc = row
        assert number == str(index)
        assert metric == fields[index - 1]
        assert module  # Module column must not be empty
