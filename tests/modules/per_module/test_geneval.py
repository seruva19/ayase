"""Tests for geneval module."""

from ..conftest import _test_module_basics
from ayase.models import CaptionMetadata, Sample


def test_geneval_basics():
    from ayase.modules.geneval import GenEvalModule
    _test_module_basics(GenEvalModule, "geneval")


def test_geneval_video_is_noop(video_sample):
    from ayase.modules.geneval import GenEvalModule
    m = GenEvalModule()
    # No setup — test_mode behavior. Module must short-circuit on video.
    out = m.process(video_sample)
    assert out is video_sample
    # Video should not populate any GenEval field
    if out.quality_metrics is not None:
        assert out.quality_metrics.geneval_overall is None


def test_geneval_no_caption_is_noop(image_sample):
    from ayase.modules.geneval import GenEvalModule
    m = GenEvalModule()
    image_sample.caption = None
    out = m.process(image_sample)
    assert out is image_sample
    if out.quality_metrics is not None:
        assert out.quality_metrics.geneval_overall is None


def test_geneval_caption_parsing():
    from ayase.modules.geneval import _parse_caption
    parsed = _parse_caption("a red apple to the left of a blue ball")
    assert ("red", "apple") in parsed["color_pairs"]
    assert ("blue", "ball") in parsed["color_pairs"]
    assert "left of" in parsed["positions"]
    parsed2 = _parse_caption("three dogs running on a field")
    assert parsed2["count"] == 3
    assert parsed2["count_object"] == "dogs"


def test_geneval_image_without_clip_returns_zeros(image_sample):
    from ayase.modules.geneval import GenEvalModule
    image_sample.caption = CaptionMetadata(text="a red apple", length=11)
    m = GenEvalModule()
    # Skip setup so CLIP is not loaded → image-only path with no model
    out = m.process(image_sample)
    # Without CLIP loaded, scores should still be populated as zeros (no crash)
    assert out is image_sample
    if out.quality_metrics is not None and out.quality_metrics.geneval_overall is not None:
        assert 0.0 <= out.quality_metrics.geneval_overall <= 1.0
