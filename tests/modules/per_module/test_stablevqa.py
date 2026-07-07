"""Tests for stablevqa module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_stablevqa_basics():
    from ayase.modules.stablevqa import StableVQAModule
    _test_module_basics(StableVQAModule, "stablevqa")


def test_stablevqa_video(video_sample):
    """Graceful path: in test mode setup is skipped, no score is fabricated."""
    from ayase.modules.stablevqa import StableVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = StableVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    # No trained backend loaded in test mode -> metric stays unset (no heuristic).
    assert result.quality_metrics.stablevqa_score is None


def test_stablevqa_real_backend(video_sample):
    """Weights-guarded real-path: load the 861-tensor checkpoint and run a
    real forward.  Skips cleanly when torch/timm/weights are unavailable."""
    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        pytest.skip("torch/timm/huggingface_hub not installed")

    from ayase.modules.stablevqa import StableVQAModule, _load_model_definitions

    # Reconstructed architecture must expose exactly 861 tensors.
    model = _load_model_definitions()()
    assert len(model.state_dict()) == 861

    m = StableVQAModule({"test_mode": False, "device": "auto"})
    m.setup()
    if m._backend != "real":
        pytest.skip("StableVQA weights unavailable (offline or download failed)")

    video_sample.quality_metrics = QualityMetrics()
    result = m.process(video_sample)
    assert result is video_sample
    score = result.quality_metrics.stablevqa_score
    assert score is None or isinstance(score, float)
    # A loaded real backend must produce a finite score for a valid video.
    if score is not None:
        import math
        assert math.isfinite(score)
