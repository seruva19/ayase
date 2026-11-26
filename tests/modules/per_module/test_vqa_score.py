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
def test_vqa_score_pil_video_frames_are_opt_in_and_avoid_jpeg(video_sample, monkeypatch):
    from types import SimpleNamespace

    from PIL import Image

    from ayase.models import CaptionMetadata
    from ayase.modules.vqa_score import VQAScoreModule

    seen = []

    class Scorer:
        model = SimpleNamespace(supports_pil_images=True)

        def __call__(self, *, images, texts):
            assert texts == ["clip"]
            assert len(images) == 1 and isinstance(images[0], Image.Image)
            seen.append(images[0].mode)
            return SimpleNamespace(item=lambda: 0.75)

    def reject_save(*args, **kwargs):
        raise AssertionError("PIL mode must not encode temporary JPEGs")

    monkeypatch.setattr(Image.Image, "save", reject_save)
    video_sample.caption = CaptionMetadata(text="clip", length=4)
    module = VQAScoreModule({"video_frame_input": "pil"})
    module._model = Scorer()
    module._ml_available = True

    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics.vqa_score_alignment == 0.75
    assert seen and set(seen) == {"RGB"}
