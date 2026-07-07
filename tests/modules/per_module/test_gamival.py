"""Tests for gamival module."""

import importlib.util

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_gamival_basics():
    from ayase.modules.gamival import GAMIVALModule
    _test_module_basics(GAMIVALModule, "gamival")


def test_gamival_image(image_sample):
    from ayase.modules.gamival import GAMIVALModule
    image_sample.quality_metrics = QualityMetrics()
    m = GAMIVALModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample


def test_gamival_video(video_sample):
    from ayase.modules.gamival import GAMIVALModule
    video_sample.quality_metrics = QualityMetrics()
    m = GAMIVALModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_gamival_graceful_unavailable(video_sample):
    """Without the real two-branch pipeline the metric stays unset (no heuristic)."""
    from ayase.modules.gamival import GAMIVALModule
    from ayase.pipeline import PipelineModule

    video_sample.quality_metrics = QualityMetrics()
    m = GAMIVALModule({"test_mode": False})
    prev = PipelineModule._global_test_mode
    PipelineModule.set_test_mode(False)
    try:
        m.setup()
    finally:
        PipelineModule.set_test_mode(prev)
    assert m._backend == "unavailable"
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.gamival_score is None


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="NDNetGaming CNN branch requires TensorFlow/Keras",
)
def test_gamival_real_path(video_sample):
    """Dep-guarded real-path probe.

    Even when TensorFlow is importable, GAMIVAL still requires a Python port
    of the MATLAB NSS branch and a trained SVR predictor (neither ships
    upstream), so the honest outcome remains ``unavailable`` -- and the score
    must never be fabricated.
    """
    from ayase.modules.gamival import GAMIVALModule
    from ayase.pipeline import PipelineModule

    video_sample.quality_metrics = QualityMetrics()
    m = GAMIVALModule({"test_mode": False})
    prev = PipelineModule._global_test_mode
    PipelineModule.set_test_mode(False)
    try:
        m.setup()
    finally:
        PipelineModule.set_test_mode(prev)
    assert m._backend in ("real", "unavailable")
    result = m.process(video_sample)
    if m._backend != "real":
        assert result.quality_metrics.gamival_score is None
