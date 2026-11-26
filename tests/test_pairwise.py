import json

import pytest
from typer.testing import CliRunner

from ayase.cli import app
from ayase.pairwise import compare


def row(prompt, a, b, outcome):
    return dict(prompt=prompt, model_a=a, model_b=b, outcome=outcome)


def test_two_model_strengths_match_smoothed_win_odds():
    from math import log

    result = compare(
        [
            row("a win", "alpha", "beta", "a"),
            row("another a win", "beta", "alpha", "b"),
            row("tie", "alpha", "beta", "tie"),
        ],
        draws=0,
    )
    assert result["connected"]
    assert result["tie_count"] == 1
    ratings = result["components"][0]["ratings"]
    assert [rating["model"] for rating in ratings] == ["alpha", "beta"]
    assert [rating["rank"] for rating in ratings] == [1, 2]
    assert ratings[0]["log_strength"] - ratings[1]["log_strength"] == pytest.approx(log(3))


def test_disconnected_ranks_and_prompt_bootstrap():
    rows = [
        row("bridge", "a", "b", "tie"),
        row("second", "b", "c", "a"),
        row("one-prompt component", "x", "y", "tie"),
    ]
    result = compare(rows, draws=100, seed=12)
    assert result == compare(reversed(rows), draws=100, seed=12)
    assert result["connected"] is False
    first, second = result["components"]
    assert 2 <= first["bootstrap_accepted"] < 100
    assert all(rating["ci95"] is not None for rating in first["ratings"])
    assert second["bootstrap_accepted"] == 0
    assert [rating["rank"] for rating in second["ratings"]] == [1, 1]
    assert all(rating["ci95"] is None for rating in second["ratings"])


def test_bootstrap_resamples_complete_prompts():
    rows = [row("many judgments", "a", "b", "a")] * 10 + [row("one judgment", "a", "b", "b")]
    bounds = compare(rows, draws=128, seed=3)["components"][0]["ratings"][0]["ci95"]
    assert bounds[0] < 0 < bounds[1]


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_compare_cli_reads_files_and_writes_json(tmp_path, suffix):
    source = tmp_path / f"judgments{suffix}"
    output = tmp_path / "report.json"
    data = [row("prompt", "a", "b", "b"), row("another prompt", "a", "b", "tie")]
    content = json.dumps(data) if suffix == ".json" else "\n".join(map(json.dumps, data))
    source.write_text(content, encoding="utf-8")
    result = CliRunner().invoke(app, ["compare", str(source), "--draws", "10", "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8")) == compare(data, draws=10)


def test_compare_rejects_invalid_inputs(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps([row("p", "a", "a", "a")]), encoding="utf-8")
    result = CliRunner().invoke(app, ["compare", str(source)])
    assert result.exit_code == 2
    assert "distinct models" in result.output
