"""Tests for clip_temporal module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_clip_temporal_basics():
    from ayase.modules.clip_temporal import CLIPTemporalModule
    _test_module_basics(CLIPTemporalModule, "clip_temporal")

def test_clip_temporal_video(video_sample):
    from ayase.modules.clip_temporal import CLIPTemporalModule
    video_sample.quality_metrics = QualityMetrics()
    m = CLIPTemporalModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_clip_temporal_scores_two_frames():
    import torch

    from ayase.models import Sample
    from ayase.modules.clip_temporal import CLIPTemporalModule

    sample = Sample(path="two-frames.mp4", is_video=True)
    module = CLIPTemporalModule()
    module._apply_scores(sample, torch.tensor([[1.0, 0.0], [0.8, 0.6]]))

    assert sample.quality_metrics.clip_temp == pytest.approx(0.8)
    assert sample.quality_metrics.face_consistency == pytest.approx(0.8)
