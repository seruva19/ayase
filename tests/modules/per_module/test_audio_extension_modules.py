"""Tests for optional audio extension modules."""

from ayase.models import DatasetStats, QualityMetrics

from ..conftest import _test_module_basics


def test_audio_isc_basics(video_sample):
    from ayase.modules.audio_isc import AudioISCModule

    _test_module_basics(AudioISCModule, "audio_isc")
    module = AudioISCModule()
    assert module.process(video_sample) is video_sample
    assert module.extract_features(video_sample) is None
    assert "audio_isc_mean" in DatasetStats.model_fields
    assert "audio_isc_std" in DatasetStats.model_fields


def test_audio_kl_basics(video_sample):
    from ayase.modules.audio_kl import AudioKLModule

    _test_module_basics(AudioKLModule, "audio_kl")
    module = AudioKLModule()
    assert module.process(video_sample) is video_sample
    assert module.extract_features(video_sample) is None
    assert "audio_kl" in DatasetStats.model_fields


def test_clap_score_variants_basics(video_sample):
    from ayase.modules.clap_score import (
        GenericCLAPScoreModule,
        LAIONCLAPScoreModule,
        MSCLAPScoreModule,
    )

    _test_module_basics(LAIONCLAPScoreModule, "laion_clap_score")
    _test_module_basics(MSCLAPScoreModule, "ms_clap_score")
    _test_module_basics(GenericCLAPScoreModule, "clap_score")

    for module_cls in (LAIONCLAPScoreModule, MSCLAPScoreModule, GenericCLAPScoreModule):
        module = module_cls()
        assert module.process(video_sample) is video_sample

    qm = QualityMetrics(laion_clap_score=0.5, ms_clap_score=0.5, clap_score=0.5)
    assert qm.laion_clap_score == 0.5
    assert qm.ms_clap_score == 0.5
    assert qm.clap_score == 0.5


def test_imagebind_score_basics(video_sample):
    from ayase.modules.imagebind_score import ImageBindScoreModule

    _test_module_basics(ImageBindScoreModule, "imagebind_score")
    module = ImageBindScoreModule()
    assert module.process(video_sample) is video_sample
    qm = QualityMetrics(imagebind_score=0.5)
    assert qm.imagebind_score == 0.5


def test_nima_legacy_onnx_basics(image_sample):
    from ayase.modules.nima_legacy_onnx import NIMALegacyONNXModule

    _test_module_basics(NIMALegacyONNXModule, "nima_legacy_onnx")
    module = NIMALegacyONNXModule()
    assert module.process(image_sample) is image_sample
    assert image_sample.quality_metrics is not None
    qm = QualityMetrics(aesthetic_score_legacy=5.0)
    assert qm.aesthetic_score_legacy == 5.0
