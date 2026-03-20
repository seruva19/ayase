"""Tests for t2v_compbench module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_t2v_compbench_basics():
    from ayase.modules.t2v_compbench import T2VCompBenchModule
    _test_module_basics(T2VCompBenchModule, "t2v_compbench")

def test_t2v_compbench_video(video_sample):
    from ayase.modules.t2v_compbench import T2VCompBenchModule
    video_sample.quality_metrics = QualityMetrics()
    m = T2VCompBenchModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
