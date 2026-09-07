"""Regression checks for final computation failures and report export."""

import json

import pytest

from ayase.base_modules import BatchMetricModule
from ayase.models import Sample
from ayase.pipeline import Pipeline


class _FailingBatch(BatchMetricModule):
    name = "audit_final_failure"

    def __init__(self):
        super().__init__()
        self.released = False

    def extract_features(self, sample):
        return [1.0, 2.0]

    def compute_distribution_metric(self, features, reference_features=None):
        raise RuntimeError("final computation failed")

    def teardown(self):
        self.released = True


def test_batch_failure_marks_run_and_samples_incomplete_and_releases_resources(tmp_path):
    module = _FailingBatch()
    pipeline = Pipeline([module])
    pipeline.start()
    for name in ("one.png", "two.png"):
        path = tmp_path / name
        path.touch()
        pipeline.process_sample(Sample(path=path, is_video=False))
    pipeline.stop()

    status = pipeline.get_run_status()
    assert status["complete"] is False
    assert "final computation failed" in status["module_failures"][module.name]
    assert all(module.name in s.failed_modules for s in pipeline.results.values())
    assert module.released
    assert module._feature_cache == []
    assert module._reference_cache == []
    report = tmp_path / "reports" / "failure.json"
    pipeline.export_report(report)
    assert json.loads(report.read_text())["run_status"]["complete"] is False


def test_direct_batch_failure_propagates_after_cleanup():
    module = _FailingBatch()
    module._feature_cache = [[1.0], [2.0]]
    with pytest.raises(RuntimeError, match="final computation failed"):
        module.on_dispose()
    assert module.released
    assert module._feature_cache == []


def test_unknown_export_format_has_no_filesystem_side_effects(tmp_path):
    path = tmp_path / "missing" / "report.yaml"
    with pytest.raises(ValueError, match="Unsupported report format"):
        Pipeline([]).export_report(path, format="yaml")
    assert not path.parent.exists()


@pytest.mark.parametrize("format", ["json", "csv", "html"])
def test_report_export_creates_parent_directories(tmp_path, format):
    path = tmp_path / "new" / "nested" / f"report.{format}"
    Pipeline([]).export_report(path, format=format)
    assert path.stat().st_size > 0
