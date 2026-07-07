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
    # No fabricated neutral: unparseable responses yield None
    assert _parse_likert("no digit here") is None


def test_videophy_no_backend_leaves_fields_unset(video_sample):
    """Without a loaded backend the module must not fabricate neutral scores."""
    from ayase.modules.videophy import VideoPhyModule
    m = VideoPhyModule(config={"backend": "trajectory"})
    # Without setup there is no VLM and no trajectory model — the module
    # must not crash and must leave both fields unset (no neutral 0.5).
    out = m.process(video_sample)
    assert out is video_sample
    if out.quality_metrics is not None:
        assert out.quality_metrics.videophy_pc_score is None
        assert out.quality_metrics.videophy_sa_score is None
