"""Tests for optional audio extension modules."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ayase.models import DatasetStats, QualityMetrics, Sample

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
    qm = QualityMetrics(imagebind_score=0.5, imagebind_av_score=-0.25)
    assert qm.imagebind_score == 0.5
    assert qm.imagebind_av_score == -0.25
    assert ImageBindScoreModule.metric_groups == {
        "imagebind_score": "audio",
        "imagebind_av_score": "alignment",
    }
    metadata = ImageBindScoreModule.get_metadata()
    assert set(metadata["output_fields"]) == {
        "imagebind_score",
        "imagebind_av_score",
    }
    assert set(metadata["metric_info"]) == {
        "imagebind_score",
        "imagebind_av_score",
    }


def test_imagebind_av_uses_official_file_preprocessing_and_raw_cosine(
    tmp_path, monkeypatch
):
    import torch

    from ayase.modules.imagebind_score import ImageBindScoreModule

    modality = SimpleNamespace(TEXT="text", AUDIO="audio", VISION="vision")
    calls = {}

    class FakeData:
        @staticmethod
        def load_and_transform_text(texts, device):
            calls["text"] = (texts, device)
            return torch.zeros((1, 77), dtype=torch.long)

    class FakeModel:
        def __call__(self, inputs):
            assert set(inputs) == {"audio", "text", "vision"}
            return {
                "audio": torch.tensor([[1.0, 0.0]]),
                "text": torch.tensor([[-1.0, 0.0]]),
                "vision": torch.tensor([[0.6, 0.8]]),
            }

    module = ImageBindScoreModule({"device": "cpu"})
    module._model = FakeModel()
    module._modality_type = modality
    module._data_module = FakeData()
    module._device = "cpu"
    monkeypatch.setattr(
        module,
        "_transform_audio_array",
        lambda audio: calls.setdefault(
            "audio", torch.zeros((1, 3, 1, 128, 204))
        ),
    )
    monkeypatch.setattr(
        module,
        "_transform_video_path",
        lambda path: calls.setdefault(
            "video", (path, torch.zeros((1, 15, 3, 2, 224, 224)))
        )[1],
    )

    video_path = tmp_path / "clip.mp4"
    scores = module._score(
        np.zeros(16000, dtype=np.float32),
        caption="a bell rings",
        video_path=video_path,
    )

    assert scores["imagebind_score"] == pytest.approx(0.0)
    # Official JavisBench sim_av is raw cosine, not shifted into [0, 1].
    assert scores["imagebind_av_score"] == pytest.approx(0.6)
    assert calls["text"] == (["a bell rings"], "cpu")
    assert calls["video"][0] == video_path
    assert calls["audio"].shape == (1, 3, 1, 128, 204)


def test_imagebind_process_writes_av_output_and_backend_without_caption(monkeypatch):
    import ayase.modules.imagebind_score as imagebind_module
    from ayase.modules.imagebind_score import ImageBindScoreModule

    sample = Sample(path=Path("clip.mp4"), is_video=True)
    module = ImageBindScoreModule()
    module._ml_available = True
    module._backend = "imagebind"
    monkeypatch.setattr(
        imagebind_module,
        "load_audio",
        lambda *args, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    monkeypatch.setattr(
        module,
        "_score",
        lambda audio, caption, video_path: {"imagebind_av_score": -0.25},
    )

    result = module.process(sample)

    assert result is sample
    assert sample.quality_metrics is not None
    assert sample.quality_metrics.imagebind_score is None
    assert sample.quality_metrics.imagebind_av_score == -0.25
    assert sample.quality_metrics.metric_backends["imagebind_av_score"] == "imagebind"
    assert sample.validation_issues == []


def test_imagebind_av_gracefully_skips_missing_audio(monkeypatch):
    import ayase.modules.imagebind_score as imagebind_module
    from ayase.modules.imagebind_score import ImageBindScoreModule

    sample = Sample(path=Path("silent.mp4"), is_video=True)
    module = ImageBindScoreModule()
    module._ml_available = True
    module._backend = "imagebind"
    monkeypatch.setattr(imagebind_module, "load_audio", lambda *args, **kwargs: None)

    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_nima_onnx_basics(image_sample):
    from ayase.modules.nima_onnx import NIMAONNXModule

    _test_module_basics(NIMAONNXModule, "nima_onnx")
    module = NIMAONNXModule()
    assert module.process(image_sample) is image_sample
    assert image_sample.quality_metrics is not None
    qm = QualityMetrics(nima_onnx_score=5.0)
    assert qm.nima_onnx_score == 5.0
