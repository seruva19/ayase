"""Tests for mc360iqa module."""

import os

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_mc360iqa_basics():
    from ayase.modules.mc360iqa import MC360IQAModule
    _test_module_basics(MC360IQAModule, "mc360iqa")


def test_mc360iqa_image(image_sample):
    # Light/test mode: setup() is skipped, backend stays unavailable, and the
    # module must gracefully leave mc360iqa_score unset.
    from ayase.modules.mc360iqa import MC360IQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = MC360IQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.mc360iqa_score is None


def test_mc360iqa_video(video_sample):
    from ayase.modules.mc360iqa import MC360IQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = MC360IQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.mc360iqa_score is None


@pytest.mark.skipif(
    os.environ.get("AYASE_TEST_WEIGHTS") != "1",
    reason="real MC360IQA weights download disabled (set AYASE_TEST_WEIGHTS=1)",
)
def test_mc360iqa_real_backend(image_sample):
    """Weights-guarded real path: build arch, load OIQA state_dict, score a frame."""
    from ayase.pipeline import PipelineModule
    from ayase.modules.mc360iqa import MC360IQAModule

    PipelineModule.set_test_mode(False)
    try:
        m = MC360IQAModule({"device": "cpu"})
        m.setup()
        assert m._backend == "real", "expected real backend with weights present"
        image_sample.quality_metrics = QualityMetrics()
        result = m.process(image_sample)
        assert result.quality_metrics.mc360iqa_score is not None
        assert isinstance(result.quality_metrics.mc360iqa_score, float)
    finally:
        PipelineModule.set_test_mode(True)
