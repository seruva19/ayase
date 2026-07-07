"""Tests for mdvqa module."""

import os

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_mdvqa_basics():
    from ayase.modules.mdvqa import MDVQAModule
    _test_module_basics(MDVQAModule, "mdvqa")

def test_mdvqa_image(image_sample):
    from ayase.modules.mdvqa import MDVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = MDVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_mdvqa_video(video_sample):
    from ayase.modules.mdvqa import MDVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = MDVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    # Graceful-unavailable / test-mode path emits no fabricated score.
    assert result.quality_metrics.mdvqa_score is None


# Real-path test: builds the EfficientNetV2-S + R(2+1)D-18 backbones (torchvision
# pretrained weights, ~200MB) plus the MD-VQA fusion/head checkpoint from the
# Hugging Face Hub, then runs a full forward pass. Opt-in via AYASE_MDVQA_REAL=1
# so normal/CI runs stay fast and offline.
@pytest.mark.skipif(
    os.environ.get("AYASE_MDVQA_REAL") != "1",
    reason="set AYASE_MDVQA_REAL=1 to run the real MD-VQA path (downloads backbones + head)",
)
def test_mdvqa_real_path(video_sample):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("huggingface_hub")

    from ayase.pipeline import PipelineModule
    from ayase.modules.mdvqa import MDVQAModule

    prev = PipelineModule._global_test_mode
    PipelineModule.set_test_mode(False)
    try:
        m = MDVQAModule()
        m.setup()
        if m._backend == "unavailable":
            pytest.skip("MD-VQA weights/deps unavailable")
        assert m._backend == "real"
        assert m._ml_available is True

        video_sample.quality_metrics = QualityMetrics()
        result = m.process(video_sample)
        score = result.quality_metrics.mdvqa_score
        assert score is not None
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    finally:
        PipelineModule.set_test_mode(prev)
