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


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_vectorized_clip_scores_preserve_thresholds(dtype):
    import math

    import numpy as np
    import torch

    from ayase.models import Sample
    from ayase.modules.clip_temporal import CLIPTemporalModule

    generator = torch.Generator().manual_seed(7)
    embeddings = torch.randn(33, 512, generator=generator).to(getattr(torch, dtype))
    unit = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    temporal = float(np.mean([(a @ b).item() for a, b in zip(unit[:-1], unit[1:])]))
    face = float(np.mean([(unit[0] @ frame).item() for frame in unit[1:]]))

    sample = Sample(path="clip.mp4", is_video=True)
    CLIPTemporalModule()._apply_scores(sample, embeddings)
    tolerance = max(1e-6, float(torch.finfo(unit.dtype).eps))
    assert sample.quality_metrics.clip_temp == pytest.approx(temporal, abs=tolerance)
    assert sample.quality_metrics.face_consistency == pytest.approx(face, abs=tolerance)

    sample = Sample(path="clip.mp4", is_video=True)
    module = CLIPTemporalModule({
        "temp_threshold": temporal,
        "face_threshold": math.nextafter(face, math.inf),
    })
    module._apply_scores(sample, embeddings)
    assert sample.quality_metrics.clip_temp == temporal
    assert sample.quality_metrics.face_consistency == face
    assert len(sample.validation_issues) == 1
    assert "identity/face" in sample.validation_issues[0].message
