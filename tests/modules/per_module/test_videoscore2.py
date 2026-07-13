"""Tests for videoscore2 module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_videoscore2_basics():
    from ayase.modules.videoscore2 import VideoScore2Module

    _test_module_basics(VideoScore2Module, "videoscore2")


def test_videoscore2_image(image_sample):
    from ayase.modules.videoscore2 import VideoScore2Module

    image_sample.quality_metrics = QualityMetrics()
    module = VideoScore2Module()
    module.on_mount()
    result = module.process(image_sample)
    assert result is image_sample


def test_videoscore2_video(video_sample):
    from ayase.modules.videoscore2 import VideoScore2Module

    video_sample.quality_metrics = QualityMetrics()
    module = VideoScore2Module()
    module.on_mount()
    result = module.process(video_sample)
    assert result is video_sample


def test_videoscore2_preserves_sampling_default():
    from ayase.modules.videoscore2 import VideoScore2Module

    assert VideoScore2Module.default_config["do_sample"] is True


def test_videoscore2_unavailable_backend_is_noop(image_sample):
    from ayase.modules.videoscore2 import VideoScore2Module

    module = VideoScore2Module()
    module._backend = "unavailable"
    module._compute_scores = lambda sample: pytest.fail("unavailable backend executed")

    assert module.process(image_sample) is image_sample


def test_videoscore2_soft_score_is_probability_weighted_expectation():
    torch = pytest.importorskip("torch")
    from ayase.modules.videoscore2 import VideoScore2Module

    class Tokenizer:
        @staticmethod
        def encode(value, add_special_tokens=False):
            return [int(value)]

    probabilities = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.25, 0.15])
    logits = torch.log(probabilities.clamp_min(1e-12))
    module = VideoScore2Module()

    score = module._ll_based_soft_score_normed(3, 0, [logits.unsqueeze(0)], Tokenizer())

    assert score == pytest.approx(3.15, abs=1e-4)
    assert 1.0 <= score <= 5.0
