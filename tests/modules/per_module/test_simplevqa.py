"""Tests for simplevqa module."""

import os

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_simplevqa_basics():
    from ayase.modules.simplevqa import SimpleVQAModule
    _test_module_basics(SimpleVQAModule, "simplevqa")

def test_simplevqa_image(image_sample):
    from ayase.modules.simplevqa import SimpleVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = SimpleVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_simplevqa_video(video_sample):
    from ayase.modules.simplevqa import SimpleVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = SimpleVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    # Graceful-unavailable / test-mode path emits no fabricated score.
    assert result.quality_metrics.simplevqa_score is None


# Real-path test: builds the Swin+SlowFast network, downloads the LSVQ
# checkpoint (~360MB) plus SlowFast-R50 weights (~260MB) and runs a forward
# pass. Opt-in via AYASE_SIMPLEVQA_REAL=1 so normal/CI runs stay fast.
@pytest.mark.skipif(
    os.environ.get("AYASE_SIMPLEVQA_REAL") != "1",
    reason="set AYASE_SIMPLEVQA_REAL=1 to run the real SimpleVQA path (downloads ~600MB)",
)
def test_simplevqa_real_path(video_sample):
    pytest.importorskip("timm")
    pytest.importorskip("pytorchvideo")
    pytest.importorskip("torch")

    from ayase.pipeline import PipelineModule
    from ayase.modules.simplevqa import SimpleVQAModule

    prev = PipelineModule._global_test_mode
    PipelineModule.set_test_mode(False)
    try:
        m = SimpleVQAModule()
        m.setup()
        if m._backend == "unavailable":
            pytest.skip("SimpleVQA weights/deps unavailable")
        assert m._backend == "real"
        assert m._ml_available is True

        video_sample.quality_metrics = QualityMetrics()
        result = m.process(video_sample)
        score = result.quality_metrics.simplevqa_score
        assert score is not None
        assert isinstance(score, float)
        assert score == score  # not NaN
    finally:
        PipelineModule.set_test_mode(prev)
