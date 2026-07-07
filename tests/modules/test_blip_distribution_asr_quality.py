"""Focused tests for BLIP, distribution, ASR, and speech-quality metrics."""

from pathlib import Path

import numpy as np
import pytest

from ayase.models import CaptionMetadata, QualityMetrics, Sample
from tests.modules.conftest import _test_module_basics


@pytest.fixture
def synthetic_wav(tmp_path):
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.25 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "speech.wav"
    sf.write(str(path), audio, sr)
    return path


def test_blip_distribution_module_basics():
    from ayase.modules.blip_score import BLIPScoreModule
    from ayase.modules.cmmd import CMMDModule
    from ayase.modules.fid import FIDModule
    from ayase.modules.prdc_dinov2 import PRDCDINOv2Module

    _test_module_basics(BLIPScoreModule, "blip_score")
    _test_module_basics(CMMDModule, "cmmd")
    _test_module_basics(FIDModule, "fid")
    _test_module_basics(PRDCDINOv2Module, "prdc_dinov2")


def test_asr_speech_quality_relevance_module_basics():
    from ayase.modules.aqascore import AQAScoreModule
    from ayase.modules.asr_cer import ASRCERModule
    from ayase.modules.asr_transcribe import ASRTranscribeModule
    from ayase.modules.asr_wer import ASRWERModule
    from ayase.modules.audio_utmos_v2 import AudioUTMOSv2Module
    from ayase.modules.human_clap import HumanCLAPModule
    from ayase.modules.kad import KADModule
    from ayase.modules.pam import PAMModule
    from ayase.modules.scoreq import SCOREQModule
    from ayase.modules.ttsds2 import TTSDS2Module

    _test_module_basics(ASRTranscribeModule, "asr_transcribe")
    _test_module_basics(ASRCERModule, "asr_cer")
    _test_module_basics(ASRWERModule, "asr_wer")
    _test_module_basics(SCOREQModule, "scoreq")
    _test_module_basics(TTSDS2Module, "ttsds2")
    _test_module_basics(AudioUTMOSv2Module, "audio_utmos_v2")
    _test_module_basics(KADModule, "kad")
    _test_module_basics(HumanCLAPModule, "human_clap")
    _test_module_basics(PAMModule, "pam")
    _test_module_basics(AQAScoreModule, "aqascore")


def test_quality_metrics_fields_for_metric_modules():
    fields = QualityMetrics.model_fields
    for field in [
        "blip_score",
        "asr_cer",
        "asr_wer",
        "scoreq_score",
        "ttsds2_score",
        "utmos_v2_score",
        "human_clap_score",
        "pam_score",
        "aqascore_score",
    ]:
        assert field in fields


def test_image_distribution_features_real_backend_or_none(image_sample):
    from ayase.modules.cmmd import CMMDModule
    from ayase.modules.fid import FIDModule
    from ayase.modules.prdc_dinov2 import PRDCDINOv2Module

    # CMMD/FID no longer have proxy feature extractors: without the real
    # CLIP/InceptionV3 backend loaded, extract_features must return None.
    for cls, real_backend in ((CMMDModule, "clip"), (FIDModule, "inception_v3")):
        mod = cls({"use_torchvision_backend": False} if cls is FIDModule else {})
        feat = mod.extract_features(image_sample)
        if mod._backend == real_backend:
            assert feat is not None
            assert np.asarray(feat).ndim == 1
        else:
            assert mod._backend == "unavailable"
            assert feat is None

    # PRDC's image_stats fallback is a real (handcrafted-statistics) feature
    # extractor, not a fabricated proxy — it must keep producing features.
    prdc = PRDCDINOv2Module({})
    feat = prdc.extract_features(image_sample)
    assert feat is not None
    assert np.asarray(feat).ndim == 1


def test_asr_cer_wer_use_shared_transcript_config(synthetic_wav):
    from ayase.modules.asr_cer import ASRCERModule
    from ayase.modules.asr_wer import ASRWERModule

    sample = Sample(path=synthetic_wav, is_video=False)
    cfg = {"expected_text": "hello world", "transcript": "hello word"}

    result = ASRCERModule(cfg).process(sample)
    result = ASRWERModule(cfg).process(result)

    assert result.quality_metrics is not None
    assert result.quality_metrics.asr_cer > 0
    assert result.quality_metrics.asr_wer == pytest.approx(0.5)


def test_audio_modules_real_backend_or_none(synthetic_wav):
    """Signal-proxy tiers were removed: each module scores only via its real
    backend and otherwise leaves its field unset with _backend 'unavailable'."""
    from ayase.modules.audio_utmos_v2 import AudioUTMOSv2Module
    from ayase.modules.pam import PAMModule
    from ayase.modules.scoreq import SCOREQModule
    from ayase.modules.ttsds2 import TTSDS2Module

    sample = Sample(path=synthetic_wav, is_video=False)
    checks = [
        (SCOREQModule(), "scoreq_score", ("scoreq",), (0.0, 1.0)),
        (AudioUTMOSv2Module(), "utmos_v2_score",
         ("utmosv2_package", "torch_hub"), (1.0, 5.0)),
        (PAMModule(), "pam_score", ("clap",), (0.0, 1.0)),
        (TTSDS2Module({"enabled": True}), "ttsds2_score", ("ttsds2",), (0.0, 1.0)),
    ]
    for module, field, real_backends, (lo, hi) in checks:
        module.on_mount()
        sample = module.process(sample)
        value = getattr(sample.quality_metrics, field) if sample.quality_metrics else None
        if module._backend in real_backends:
            assert value is not None, f"{field} should be set by real backend"
            assert lo <= value <= hi
        else:
            assert module._backend in (None, "unavailable")
            assert value is None, f"{field} must stay unset without a real backend"


def test_aqascore_is_opt_in(synthetic_wav):
    from ayase.modules.aqascore import AQAScoreModule

    sample = Sample(
        path=synthetic_wav,
        is_video=False,
        caption=CaptionMetadata(text="clear speech audio", length=18),
    )
    assert AQAScoreModule().process(sample).quality_metrics is None

    enabled = AQAScoreModule({"enabled": True})
    enabled.on_mount()
    result = enabled.process(sample)
    if enabled._backend == "qwen_omni":
        assert result.quality_metrics is not None
        assert 0.0 <= result.quality_metrics.aqascore_score <= 1.0
    else:
        # Honest state: opt-in but real Qwen2.5-Omni backend unavailable
        assert enabled._backend == "unavailable"
        assert (result.quality_metrics is None
                or result.quality_metrics.aqascore_score is None)


def test_kad_and_fad_infinity_math():
    from ayase.modules.fad import FADModule
    from ayase.modules.kad import KADModule

    rng = np.random.default_rng(123)
    features = [rng.normal(size=8) for _ in range(8)]
    refs = [rng.normal(loc=0.1, size=8) for _ in range(8)]

    kad = KADModule().compute_distribution_metric(features, refs)
    fad_inf = FADModule({"infinity": True}).compute_distribution_metric(features, refs)

    assert np.isfinite(kad)
    assert np.isfinite(fad_inf)
