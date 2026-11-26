"""Tests for vqa_score module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vqa_score_basics():
    from ayase.modules.vqa_score import VQAScoreModule
    _test_module_basics(VQAScoreModule, "vqa_score")

def test_vqa_score_image(image_sample):
    from ayase.modules.vqa_score import VQAScoreModule
    image_sample.quality_metrics = QualityMetrics()
    m = VQAScoreModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_vqa_score_video(video_sample):
    from ayase.modules.vqa_score import VQAScoreModule
    video_sample.quality_metrics = QualityMetrics()
    m = VQAScoreModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_vqa_score_cleans_partial_jpeg_after_save_error(video_sample, monkeypatch):
    from pathlib import Path

    from PIL import Image

    from ayase.models import CaptionMetadata
    from ayase.modules.vqa_score import VQAScoreModule

    files = []

    def partial_save(_image, path, format, *, quality):
        assert format == "JPEG"
        assert quality == 95
        path = Path(path)
        path.write_bytes(b"partial JPEG")
        files.append(path)
        raise OSError("simulated encoder failure")

    monkeypatch.setattr(Image.Image, "save", partial_save)
    video_sample.caption = CaptionMetadata(text="A test clip", length=len("A test clip"))
    module = VQAScoreModule()
    module._ml_available = True
    module._model = object()

    assert module.process(video_sample) is video_sample
    assert files and not files[0].parent.exists()
