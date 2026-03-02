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
            if len(cols) >= 4:
                rows.append(cols[:4])
    return rows


def test_metrics_table_matches_quality_metrics():
    rows = _load_metrics_rows()
    fields = list(QualityMetrics.model_fields.keys())
    assert len(rows) == len(fields)
    metrics = [row[1] for row in rows]
    assert metrics == fields
    for index, row in enumerate(rows, 1):
        number, metric, title, tested = row
        assert number == str(index)
        assert metric == fields[index - 1]
        assert title
        assert tested == "Yes"
