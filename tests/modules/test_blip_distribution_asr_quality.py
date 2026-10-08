"""Focused tests for BLIP, distribution, ASR, and speech-quality metrics."""

import os
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
    from ayase.modules.tts_system_dist import TTSDS2Module

    _test_module_basics(ASRTranscribeModule, "asr_transcribe")
    _test_module_basics(ASRCERModule, "asr_cer")
    _test_module_basics(ASRWERModule, "asr_wer")
    _test_module_basics(SCOREQModule, "scoreq")
    _test_module_basics(TTSDS2Module, "tts_system_dist")
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
        "tts_system_dist_score",
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

    # CMMD/FID/PRDC no longer have proxy feature extractors: without the real
    # CLIP/InceptionV3/DINOv2 backend loaded, extract_features must return None.
    # (PRDC computed on handcrafted image statistics is a different quantity
    # than PRDC on DINOv2 features — the feature space is part of the metric.)
    for cls, real_backend in (
        (CMMDModule, "clip"),
        (FIDModule, "inception_v3"),
        (PRDCDINOv2Module, "dinov2"),
    ):
        mod = cls({"use_torchvision_backend": False} if cls is FIDModule else {})
        feat = mod.extract_features(image_sample)
        if mod._backend == real_backend:
            assert feat is not None
            assert np.asarray(feat).ndim == 1
        else:
            assert mod._backend == "unavailable"
            assert feat is None


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
    from ayase.modules.tts_system_dist import TTSDS2Module

    sample = Sample(path=synthetic_wav, is_video=False)
    checks = [
        (SCOREQModule(), "scoreq_score", ("scoreq",), (0.0, 1.0)),
        (AudioUTMOSv2Module(), "utmos_v2_score",
         ("utmosv2_package", "torch_hub"), (1.0, 5.0)),
        (PAMModule(), "pam_score", ("clap",), (0.0, 1.0)),
        (TTSDS2Module({"enabled": True}), "tts_system_dist_score", (), (0.0, 100.0)),
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


def test_ttsds2_declares_dataset_protocol_and_never_scores_one_file(synthetic_wav):
    from ayase.modules.tts_system_dist import TTSDS2Module

    sample = Sample(path=synthetic_wav, is_video=False)
    module = TTSDS2Module({"enabled": True})
    assert module.requires_external_backend is True
    assert module.provenance == "published"
    module.setup()
    result = module.process(sample)
    assert result is sample
    assert result.quality_metrics is None
    assert module._backend is None


def test_ttsds2_opt_in_reports_external_backend_unavailable(synthetic_wav):
    from ayase.modules.tts_system_dist import TTSDS2Module
    from ayase.pipeline import Pipeline

    pipeline = Pipeline([TTSDS2Module({"enabled": True})])
    pipeline.start()
    result = pipeline.process_sample(Sample(path=synthetic_wav, is_video=False))
    status = pipeline.get_run_status()
    assert result.quality_metrics is None
    assert status["complete"] is False
    assert status["availability_excluded"] == {
        "tts_system_dist": "external_backend_unavailable"
    }


def test_scoreq_redirects_native_download_to_models_dir(monkeypatch, tmp_path):
    import ayase.modules.scoreq as scoreq_module

    calls = []
    package_dir = tmp_path / "installed" / "scoreq"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "raise AssertionError('scoreq package initializer must not execute')\n",
        encoding="utf-8",
    )
    (package_dir / "scoreq.py").write_text(
        """
class Scoreq:
    def __init__(self, data_domain='natural', mode='nr'):
        self.data_domain = data_domain
        self.mode = mode
        self.model_path = self._download_model(
            'adapt_nr_telephone.onnx', 'https://example.invalid/model', 'onnx-models'
        )
""",
        encoding="utf-8",
    )

    class Distribution:
        version = "1.0.1"
        files = [Path("scoreq") / "scoreq.py"]

        def locate_file(self, relative_path):
            return tmp_path / "installed" / relative_path

    def fake_download(relative_path, url, models_dir):
        calls.append((relative_path, url, models_dir))
        return Path(models_dir) / relative_path

    monkeypatch.setattr(scoreq_module.metadata, "distribution", lambda name: Distribution())
    monkeypatch.setattr(scoreq_module, "download_model_file", fake_download)
    monkeypatch.setattr(
        os.path,
        "expanduser",
        lambda path: (_ for _ in ()).throw(AssertionError(f"HOME access attempted: {path}")),
    )

    models_dir = tmp_path / "models"
    module = scoreq_module.SCOREQModule({"models_dir": str(models_dir)})
    module.setup()

    assert module._backend == "scoreq"
    assert module._backend_version == "1.0.1"
    assert module._backend_source == str((package_dir / "scoreq.py").resolve())
    assert type(module._model).__mro__[1].__module__ == "_ayase_native_scoreq_1_0_1"
    assert module._model.data_domain == "natural"
    assert module._model.mode == "nr"
    assert calls == [
        (
            "scoreq/onnx-models/adapt_nr_telephone.onnx",
            "https://example.invalid/model",
            str(models_dir),
        )
    ]
    assert not (tmp_path / "untouched-home").exists()


def test_scoreq_cache_rejects_path_traversal(monkeypatch, tmp_path):
    import ayase.modules.scoreq as scoreq_module

    source_path = tmp_path / "installed" / "scoreq" / "scoreq.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        """
class Scoreq:
    def __init__(self, data_domain='natural', mode='nr'):
        self._download_model(
            '../../../escape.onnx', 'https://example.invalid/model', 'onnx-models'
        )
""",
        encoding="utf-8",
    )

    class Distribution:
        version = "1.0.0"
        files = [Path("scoreq") / "scoreq.py"]

        def locate_file(self, relative_path):
            return tmp_path / "installed" / relative_path

    monkeypatch.setattr(scoreq_module.metadata, "distribution", lambda name: Distribution())

    models_dir = tmp_path / "models"
    module = scoreq_module.SCOREQModule({"models_dir": str(models_dir)})
    module.setup()

    assert module._backend == "unavailable"
    assert not (tmp_path / "escape.onnx").exists()


def test_scoreq_process_preserves_native_output(tmp_path):
    from ayase.modules.scoreq import SCOREQModule

    class NativePredictor:
        def predict(self, test_path, ref_path=None):
            assert test_path == str(tmp_path / "speech.wav")
            assert ref_path is None
            return 1.23456789

    module = SCOREQModule()
    module._backend = "scoreq"
    module._model = NativePredictor()
    sample = Sample(path=tmp_path / "speech.wav", is_video=False)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics.scoreq_score == 1.23456789


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


def test_kad_reference_math_and_fad_infinity():
    from ayase.modules.fad import FADModule
    from ayase.modules.kad import KADModule

    rng = np.random.default_rng(123)
    features = [rng.normal(size=(2, 8)).astype(np.float32) for _ in range(8)]
    refs = [rng.normal(loc=0.1, size=(2, 8)).astype(np.float32) for _ in range(8)]

    import torch

    def official_test_fn(x, y, cache_dirs, device, bandwidth=None, kernel="gaussian"):
        del cache_dirs, device, bandwidth, kernel
        # A compact unbiased linear-kernel MMD stand-in for API-contract testing.
        kxx = x @ x.T
        kyy = y @ y.T
        kxy = x @ y.T
        nx, ny = len(x), len(y)
        value = (
            (kxx.sum() - kxx.diag().sum()) / (nx * (nx - 1))
            + (kyy.sum() - kyy.diag().sum()) / (ny * (ny - 1))
            - 2 * kxy.mean()
        )
        return value * 100

    module = KADModule()
    module._torch = torch
    module._kad_fn = official_test_fn
    kad = module.compute_distribution_metric(features, refs)
    fad_features = [feature.mean(axis=0) for feature in features]
    fad_refs = [feature.mean(axis=0) for feature in refs]
    fad_inf = FADModule({"infinity": True}).compute_distribution_metric(
        fad_features, fad_refs
    )

    assert np.isfinite(kad)
    assert np.isfinite(fad_inf)


def test_kad_requires_real_backend_and_references(synthetic_wav):
    from ayase.modules.kad import KADModule

    sample = Sample(path=synthetic_wav, is_video=False)
    module = KADModule()
    assert module.extract_features(sample) is None
    with pytest.raises(RuntimeError, match="upstream KADTK"):
        module.compute_distribution_metric([np.zeros((2, 8), dtype=np.float32)])
