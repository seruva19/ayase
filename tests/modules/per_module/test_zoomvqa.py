"""Tests for zoomvqa module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics
from ayase.pipeline import PipelineModule


def test_zoomvqa_basics():
    from ayase.modules.zoomvqa import ZoomVQAModule
    _test_module_basics(ZoomVQAModule, "zoomvqa")

def test_zoomvqa_image(image_sample):
    from ayase.modules.zoomvqa import ZoomVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = ZoomVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_zoomvqa_video(video_sample):
    from ayase.modules.zoomvqa import ZoomVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = ZoomVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def _real_backend_available() -> bool:
    """Real path requires torch/timm/einops/decord and the cached checkpoints."""
    try:
        from ayase.modules.zoomvqa import (
            _TORCH_OK, HF_REPO, IQA_WEIGHT, VQA_WEIGHT,
        )
        import decord  # noqa: F401
        from huggingface_hub import hf_hub_download

        if not _TORCH_OK:
            return False
        # Only proceed when weights are already in the local HF cache so a
        # normal light test run never triggers a large download.
        hf_hub_download(repo_id=HF_REPO, filename=IQA_WEIGHT, local_files_only=True)
        hf_hub_download(repo_id=HF_REPO, filename=VQA_WEIGHT, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _real_backend_available(),
                    reason="zoomvqa real checkpoints/deps not available")
def test_zoomvqa_real_backend(image_sample, video_sample):
    from ayase.modules.zoomvqa import ZoomVQAModule

    # Bypass the global light-mode test flag for this weights-guarded test.
    PipelineModule.set_test_mode(False)
    try:
        m = ZoomVQAModule({"vqa_num_clips": 1})  # single clip -> fast
        m.setup()
        assert m._backend == "real"

        image_sample.quality_metrics = QualityMetrics()
        m.process(image_sample)
        s_img = image_sample.quality_metrics.zoomvqa_score
        assert s_img is not None and isinstance(s_img, float)

        video_sample.quality_metrics = QualityMetrics()
        m.process(video_sample)
        s_vid = video_sample.quality_metrics.zoomvqa_score
        assert s_vid is not None and isinstance(s_vid, float)
    finally:
        PipelineModule.set_test_mode(True)
