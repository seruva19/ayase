from ayase.models import QualityMetrics


def test_camera_jitter_basics():
    from ayase.modules.camera_jitter import CameraJitterModule
    from .conftest import _test_module_basics

    _test_module_basics(CameraJitterModule, "camera_jitter")


def test_camera_jitter_video(video_sample):
    from ayase.modules.camera_jitter import CameraJitterModule

    video_sample.quality_metrics = QualityMetrics()
    m = CameraJitterModule()
    result = m.process(video_sample)
    assert result.quality_metrics.camera_jitter_score is not None
    assert 0.0 <= result.quality_metrics.camera_jitter_score <= 1.0


def test_camera_jitter_image(image_sample):
    from ayase.modules.camera_jitter import CameraJitterModule

    image_sample.quality_metrics = QualityMetrics()
    m = CameraJitterModule()
    result = m.process(image_sample)
    assert result.quality_metrics.camera_jitter_score is None


def test_jump_cut_basics():
    from ayase.modules.jump_cut import JumpCutModule
    from .conftest import _test_module_basics

    _test_module_basics(JumpCutModule, "jump_cut")


def test_jump_cut_video(video_sample):
    from ayase.modules.jump_cut import JumpCutModule

    video_sample.quality_metrics = QualityMetrics()
    m = JumpCutModule()
    result = m.process(video_sample)
    assert result.quality_metrics.jump_cut_score is not None
    assert 0.0 <= result.quality_metrics.jump_cut_score <= 1.0


def test_playback_speed_basics():
    from ayase.modules.playback_speed import PlaybackSpeedModule
    from .conftest import _test_module_basics

    _test_module_basics(PlaybackSpeedModule, "playback_speed")


def test_playback_speed_video(video_sample):
    from ayase.modules.playback_speed import PlaybackSpeedModule

    video_sample.quality_metrics = QualityMetrics()
    m = PlaybackSpeedModule()
    result = m.process(video_sample)
    assert result.quality_metrics.playback_speed_score is not None


def test_flow_coherence_basics():
    from ayase.modules.flow_coherence import FlowCoherenceModule
    from .conftest import _test_module_basics

    _test_module_basics(FlowCoherenceModule, "flow_coherence")


def test_flow_coherence_video(video_sample):
    from ayase.modules.flow_coherence import FlowCoherenceModule

    video_sample.quality_metrics = QualityMetrics()
    m = FlowCoherenceModule()
    result = m.process(video_sample)
    assert result.quality_metrics.flow_coherence is not None
    assert 0.0 <= result.quality_metrics.flow_coherence <= 1.0


def test_letterbox_basics():
    from ayase.modules.letterbox import LetterboxModule
    from .conftest import _test_module_basics

    _test_module_basics(LetterboxModule, "letterbox")


def test_letterbox_video(video_sample):
    from ayase.modules.letterbox import LetterboxModule

    video_sample.quality_metrics = QualityMetrics()
    m = LetterboxModule()
    result = m.process(video_sample)
    assert result.quality_metrics.letterbox_ratio is not None
    assert 0.0 <= result.quality_metrics.letterbox_ratio <= 1.0


def test_letterbox_image(image_sample):
    from ayase.modules.letterbox import LetterboxModule

    image_sample.quality_metrics = QualityMetrics()
    m = LetterboxModule()
    result = m.process(image_sample)
    assert result.quality_metrics.letterbox_ratio is not None


def test_vtss_basics():
    from ayase.modules.vtss import VTSSModule
    from .conftest import _test_module_basics

    _test_module_basics(VTSSModule, "vtss")


def test_vtss_with_metrics(video_sample):
    from ayase.modules.vtss import VTSSModule

    video_sample.quality_metrics = QualityMetrics(
        aesthetic_score=7.0,
        technical_score=60.0,
        motion_score=8.0,
        temporal_consistency=0.85,
        blur_score=300.0,
        noise_score=10.0,
    )
    m = VTSSModule()
    result = m.process(video_sample)
    assert result.quality_metrics.vtss is not None
    assert 0.0 <= result.quality_metrics.vtss <= 1.0


def test_vtss_no_metrics(video_sample):
    from ayase.modules.vtss import VTSSModule

    video_sample.quality_metrics = QualityMetrics()
    m = VTSSModule()
    result = m.process(video_sample)
    assert result.quality_metrics.vtss is None
