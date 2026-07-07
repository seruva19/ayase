"""Tests for provqa module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_provqa_basics():
    from ayase.modules.provqa import ProVQAModule
    _test_module_basics(ProVQAModule, "provqa")

def test_provqa_image(image_sample):
    from ayase.modules.provqa import ProVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = ProVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_provqa_video(video_sample):
    from ayase.modules.provqa import ProVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = ProVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def _real_model_or_skip():
    """Build the real ProVQA model or skip (missing torch / hub / weights)."""
    pytest.importorskip("torch")
    pytest.importorskip("huggingface_hub")
    from ayase.modules.provqa import _build_model

    try:
        return _build_model("cpu")
    except Exception as e:  # no network, missing weights, or key mismatch
        pytest.skip(f"ProVQA real weights/deps unavailable: {e}")


def test_provqa_real_forward(video_sample):
    """Weights-guarded: strict-load the real net and score a synthetic video.

    Exercises the real inference path directly (bypassing the global
    ``test_mode`` gate) so it runs whenever the mirrored weights + torch are
    available, and is skipped otherwise.
    """
    from ayase.modules.provqa import ProVQAModule

    net = _real_model_or_skip()  # strict load is verified inside _build_model

    m = ProVQAModule()
    m._model = net
    m._backend = "real"
    m._ml_available = True
    m._device = "cpu"

    video_sample.quality_metrics = QualityMetrics()
    result = m.process(video_sample)
    assert result is video_sample
    score = result.quality_metrics.provqa_score
    assert score is not None
    assert 0.0 <= score <= 1.0
