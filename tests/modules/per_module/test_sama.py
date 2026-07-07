"""Tests for sama module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_sama_basics():
    from ayase.modules.sama import SAMAModule
    _test_module_basics(SAMAModule, "sama")


def test_sama_image(image_sample):
    # Images are not videos; SAMA leaves the sample untouched.
    from ayase.modules.sama import SAMAModule
    image_sample.quality_metrics = QualityMetrics()
    m = SAMAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample


def test_sama_video(video_sample):
    # Graceful-unavailable path: without setup() no backend is loaded, so
    # process() is a no-op and never fabricates a score (no-heuristic policy).
    from ayase.modules.sama import SAMAModule
    video_sample.quality_metrics = QualityMetrics()
    m = SAMAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.sama_score is None


def _sama_weights_available() -> bool:
    try:
        import torch  # noqa: F401
        import decord  # noqa: F401
        from huggingface_hub import hf_hub_download

        from ayase.modules.sama import SAMA_REPO_ID, SAMA_WEIGHTS_FILE

        hf_hub_download(repo_id=SAMA_REPO_ID, filename=SAMA_WEIGHTS_FILE)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _sama_weights_available(),
    reason="SAMA weights / deps (torch, decord, huggingface_hub) unavailable",
)
def test_sama_real_forward(video_sample, monkeypatch):
    """Weights-guarded real path: build arch, load checkpoint, score a video."""
    from ayase.pipeline import PipelineModule
    from ayase.modules.sama import SAMAModule

    # Force real (non-test) mode so setup() actually loads the model.
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    prev = PipelineModule._global_test_mode
    PipelineModule._global_test_mode = False
    try:
        video_sample.quality_metrics = QualityMetrics()
        m = SAMAModule()
        assert m.test_mode is False
        m.setup()
        assert m._backend == "real"
        assert m._ml_available is True

        result = m.process(video_sample)
        score = result.quality_metrics.sama_score
        assert score is not None
        assert isinstance(score, float)
    finally:
        PipelineModule._global_test_mode = prev
