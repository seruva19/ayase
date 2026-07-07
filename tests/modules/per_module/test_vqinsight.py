"""Tests for vqinsight module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vqinsight_basics():
    from ayase.modules.vqinsight import VQInsightModule
    _test_module_basics(VQInsightModule, "vqinsight")

def test_vqinsight_image(image_sample):
    from ayase.modules.vqinsight import VQInsightModule
    image_sample.quality_metrics = QualityMetrics()
    m = VQInsightModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_vqinsight_video(video_sample):
    from ayase.modules.vqinsight import VQInsightModule
    video_sample.quality_metrics = QualityMetrics()
    m = VQInsightModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_vqinsight_parse_score():
    """Parsing logic works without loading any weights."""
    from ayase.modules.vqinsight import VQInsightModule

    aigc = VQInsightModule({"video_type": "aigc"})
    out = (
        "<think>reasoning here</think><answer>"
        '{"spatial": 60.0, "temporal": 80.0, "consistency": 40.0}'
        "</answer>"
    )
    # mean(60,80,40)=60 -> 0.60
    assert abs(aigc._parse_score(out) - 0.60) < 1e-6
    # No <answer> tag but JSON present -> still parses (tolerant fallback).
    assert aigc._parse_score('{"spatial": 50, "temporal": 50, "consistency": 50}') == 0.5
    # Garbage -> None, never a silent-wrong number.
    assert aigc._parse_score("no numbers at all here") is None
    assert aigc._parse_score("") is None

    natural = VQInsightModule({"video_type": "natural"})
    assert abs(natural._parse_score("<answer>72.5</answer>") - 0.725) < 1e-6
    assert natural._parse_score("<answer>no score</answer>") is None


def test_vqinsight_real_backend(video_sample):
    """Real-path: only runs when explicitly opted in on a GPU host with weights.

    This is gated behind ``AYASE_VQINSIGHT_REAL=1`` and skips BEFORE calling
    ``setup()`` so a normal test run never triggers the ~16 GB ByteDance/Q-Insight
    download. Run it on a machine that has the weights cached (e.g. the workbox
    GPU host) with ``AYASE_VQINSIGHT_REAL=1 pytest -k vqinsight_real_backend``.
    """
    import os

    import pytest

    if os.environ.get("AYASE_VQINSIGHT_REAL") != "1":
        pytest.skip("set AYASE_VQINSIGHT_REAL=1 to exercise the real weights")

    from ayase.pipeline import PipelineModule
    from ayase.modules.vqinsight import VQInsightModule

    prev = PipelineModule._global_test_mode
    PipelineModule.set_test_mode(False)
    try:
        m = VQInsightModule()
        m.setup()
        if m._backend != "real":
            pytest.skip("VQ-Insight weights/GPU not available")
        video_sample.quality_metrics = QualityMetrics()
        out = m.process(video_sample)
        score = out.quality_metrics.vqinsight_score
        assert score is not None
        assert 0.0 <= score <= 1.0
    finally:
        PipelineModule.set_test_mode(prev)
