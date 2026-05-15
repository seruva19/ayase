"""Tests for videophy module."""

from ..conftest import _test_module_basics


def test_videophy_basics():
    from ayase.modules.videophy import VideoPhyModule
    _test_module_basics(VideoPhyModule, "videophy")


def test_videophy_image_is_noop(image_sample):
    from ayase.modules.videophy import VideoPhyModule
    m = VideoPhyModule()
    out = m.process(image_sample)
    assert out is image_sample
    if out.quality_metrics is not None:
        assert out.quality_metrics.videophy_pc_score is None


def test_videophy_parse_likert():
    from ayase.modules.videophy import _parse_likert
    assert _parse_likert("3") == 0.5
    assert _parse_likert("The answer is 5.") == 1.0
    assert _parse_likert("score: 1, terrible") == 0.0
    assert _parse_likert("no digit here") == 0.5  # neutral when unparseable


def test_videophy_trajectory_fallback_populates_fields(video_sample):
    from ayase.modules.videophy import VideoPhyModule
    m = VideoPhyModule(config={"backend": "trajectory"})
    # Without setup, trajectory fallback path will hit the "no model" branch
    # and write neutral 0.5 values — main thing is it does not crash and
    # the fields exist on the model.
    out = m.process(video_sample)
    assert out.quality_metrics is not None
    assert out.quality_metrics.videophy_pc_score is not None
    assert out.quality_metrics.videophy_sa_score is not None
    assert 0.0 <= out.quality_metrics.videophy_pc_score <= 1.0
    assert 0.0 <= out.quality_metrics.videophy_sa_score <= 1.0
