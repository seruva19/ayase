"""Tests for flolpips module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_flolpips_basics():
    from ayase.modules.flolpips import FloLPIPSModule
    _test_module_basics(FloLPIPSModule, "flolpips")

def test_flolpips_video(video_sample):
    from ayase.modules.flolpips import FloLPIPSModule
    video_sample.quality_metrics = QualityMetrics()
    m = FloLPIPSModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_flow_receives_one_batch_with_two_ordered_frames():
    import torch
    from ayase.modules.flolpips import FloLPIPSModule

    first = torch.zeros(1, 3, 8, 10)
    second = torch.ones(1, 3, 8, 10)
    expected_flow = torch.full((1, 1, 2, 8, 10), 3.0)

    def flownet(inputs):
        frames = inputs['images']
        assert frames.shape == (1, 2, 3, 8, 10)
        assert torch.equal(frames[:, 0], first)
        assert torch.equal(frames[:, 1], second)
        return {'flows': expected_flow}

    module = FloLPIPSModule()
    module._flownet = flownet
    assert torch.equal(module._flow(first, second), expected_flow[:, 0])
