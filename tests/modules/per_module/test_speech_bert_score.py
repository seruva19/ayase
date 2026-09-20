"""Focused tests for the pinned SpeechBERTScore implementation."""

import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ayase.models import Sample

from ..conftest import _test_module_basics


def test_speech_bert_score_basics_and_provenance():
    from ayase.modules.speech_bert_score import SpeechBERTScoreModule

    _test_module_basics(SpeechBERTScoreModule, "speech_bert_score")
    assert SpeechBERTScoreModule.metric_groups == {"speech_bert_score": "audio"}
    assert SpeechBERTScoreModule.hidden_state_index == 14
    assert SpeechBERTScoreModule.sample_rate == 16000
    assert SpeechBERTScoreModule.model_revision == (
        "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
    )
    assert SpeechBERTScoreModule.checkpoint_size == 1_261_990_257
    assert SpeechBERTScoreModule.checkpoint_sha256 == (
        "fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d"
    )
    assert "CC BY-SA 3.0" in SpeechBERTScoreModule.models[0]["notes"]
    assert "8f8cbd22d352fc59dfd5bf19de979b05bb5c7938/LICENSE" in (
        SpeechBERTScoreModule.models[0]["notes"]
    )
    assert SpeechBERTScoreModule.default_config["max_duration_seconds"] == 30.0


def test_models_doc_keeps_explicit_wavlm_provenance():
    from ayase.models_doc import generate_models_doc

    document = generate_models_doc(fetch_licenses=False)

    assert "CC BY-SA 3.0" in document
    assert "1,261,990,257 bytes" in document
    assert "fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d" in document
    assert "https://arxiv.org/abs/2110.13900" in document


def test_setup_verifies_snapshot_before_local_only_load(tmp_path, monkeypatch):
    from ayase.modules.speech_bert_score import SpeechBERTScoreModule

    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"verified checkpoint")
    calls = []

    class FakeModel:
        def to(self, device):
            calls.append(("to", str(device)))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    class FakeWavLMModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("load", path, kwargs))
            return FakeModel()

    def snapshot_download(**kwargs):
        calls.append(("snapshot", kwargs))
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(WavLMModel=FakeWavLMModel),
    )
    monkeypatch.setattr(SpeechBERTScoreModule, "checkpoint_size", checkpoint.stat().st_size)
    monkeypatch.setattr(
        SpeechBERTScoreModule,
        "checkpoint_sha256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )

    module = SpeechBERTScoreModule({"device": "cpu", "models_dir": str(tmp_path)})
    module.setup()

    snapshot_call = calls[0]
    load_call = next(call for call in calls if call[0] == "load")
    assert snapshot_call[0] == "snapshot"
    assert snapshot_call[1]["revision"] == SpeechBERTScoreModule.model_revision
    assert snapshot_call[1]["allow_patterns"] == ["config.json", "pytorch_model.bin"]
    assert load_call[2] == {"local_files_only": True, "use_safetensors": False}
    assert calls.index(snapshot_call) < calls.index(load_call)
    assert module._backend.endswith(":hidden_states[14]")


@pytest.mark.parametrize("failure", ["size", "sha256"])
def test_checkpoint_verification_rejects_mismatch(tmp_path, monkeypatch, failure):
    from ayase.modules.speech_bert_score import SpeechBERTScoreModule

    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(SpeechBERTScoreModule, "checkpoint_size", checkpoint.stat().st_size)
    monkeypatch.setattr(
        SpeechBERTScoreModule,
        "checkpoint_sha256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    if failure == "size":
        monkeypatch.setattr(SpeechBERTScoreModule, "checkpoint_size", 1)
    else:
        monkeypatch.setattr(SpeechBERTScoreModule, "checkpoint_sha256", "0" * 64)

    expected_message = "size" if failure == "size" else "SHA-256"
    with pytest.raises(RuntimeError, match=expected_message):
        SpeechBERTScoreModule._verify_checkpoint(checkpoint)


def test_blockwise_precision_matches_dense_and_is_asymmetric():
    import torch

    from ayase.modules.speech_bert_score import SpeechBERTScoreModule

    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    generated = torch.tensor([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
    dense = (
        (generated @ reference.T)
        / (
            torch.linalg.vector_norm(generated, dim=1)[:, None]
            * torch.linalg.vector_norm(reference, dim=1)[None, :]
        )
    ).max(dim=1).values.mean()

    blockwise = SpeechBERTScoreModule._blockwise_precision(generated, reference, 1)
    reverse = SpeechBERTScoreModule._blockwise_precision(reference, generated, 1)

    assert torch.equal(blockwise, dense)
    assert -1.0 <= blockwise.item() <= 1.0
    assert blockwise.item() != reverse.item()


def test_encode_uses_hidden_states_index_14():
    import torch

    from ayase.modules.speech_bert_score import SpeechBERTScoreModule

    sentinel = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    hidden_states = tuple(torch.zeros_like(sentinel) for _ in range(15))
    hidden_states = hidden_states[:14] + (sentinel,)

    class FakeModel:
        def __call__(self, waveform, **kwargs):
            assert waveform.shape == (1, 1600)
            assert kwargs == {"output_hidden_states": True}
            return SimpleNamespace(hidden_states=hidden_states)

    module = SpeechBERTScoreModule()
    module._model = FakeModel()

    assert torch.equal(module._encode(torch.ones(1, 1600)), sentinel.squeeze(0))


def test_prepare_waveform_resamples_without_normalization(monkeypatch):
    import torch

    from ayase.modules.speech_bert_score import SpeechBERTScoreModule

    source = np.linspace(-0.25, 0.5, 800, dtype=np.float32)
    calls = []

    def resample(waveform, orig_freq, new_freq):
        calls.append((waveform.clone(), orig_freq, new_freq))
        return waveform.repeat_interleave(2, dim=1)

    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        SimpleNamespace(functional=SimpleNamespace(resample=resample)),
    )
    module = SpeechBERTScoreModule({"device": "cpu"})
    module._device = torch.device("cpu")
    monkeypatch.setattr(module, "_decode_native", lambda _path: (source, 8000))

    waveform = module._prepare_waveform(SimpleNamespace())

    assert calls[0][1:] == (8000, 16000)
    assert torch.equal(calls[0][0], torch.from_numpy(source).unsqueeze(0))
    assert waveform.shape == (1, 1600)
    assert waveform.max().item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "audio,sample_rate",
    [
        (np.zeros(16000, dtype=np.float32), 16000),
        (np.full(16000, np.nan, dtype=np.float32), 16000),
        (np.ones(1599, dtype=np.float32), 16000),
        (np.ones(30 * 16000 + 1, dtype=np.float32), 16000),
    ],
)
def test_prepare_waveform_rejects_silent_nonfinite_short_and_overlong(
    monkeypatch, audio, sample_rate
):
    import torch

    from ayase.modules.speech_bert_score import SpeechBERTScoreModule

    module = SpeechBERTScoreModule({"device": "cpu"})
    module._device = torch.device("cpu")
    monkeypatch.setattr(
        module, "_decode_native", lambda _path: (audio, sample_rate)
    )

    assert module._prepare_waveform(SimpleNamespace()) is None


def test_decode_native_averages_channels_without_normalization(monkeypatch):
    from ayase.modules.speech_bert_score import SpeechBERTScoreModule

    stereo = np.array([[0.25, 0.75], [-0.5, 0.0]], dtype=np.float32)
    fake_soundfile = SimpleNamespace(
        read=lambda *_args, **_kwargs: (stereo, 22050)
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    audio, sample_rate = SpeechBERTScoreModule._decode_native(SimpleNamespace())

    np.testing.assert_array_equal(audio, np.array([0.5, -0.25], dtype=np.float32))
    assert sample_rate == 22050


def test_process_writes_raw_precision_and_backend(tmp_path, monkeypatch):
    import torch

    from ayase.modules.speech_bert_score import SpeechBERTScoreModule

    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference_path.touch()
    candidate_path.touch()
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidate = torch.tensor([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
    features = iter([reference, candidate])

    module = SpeechBERTScoreModule({"device": "cpu", "similarity_block_frames": 1})
    module._model = object()
    module._device = torch.device("cpu")
    module._backend = "test-backend"
    monkeypatch.setattr(module, "_prepare_waveform", lambda _path: torch.ones(1, 1600))
    monkeypatch.setattr(module, "_encode", lambda _waveform: next(features))
    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics is not None
    expected = (1.0 + 2 ** -0.5 + 0.0) / 3.0
    assert result.quality_metrics.speech_bert_score == pytest.approx(expected)
    assert result.quality_metrics.metric_backends["speech_bert_score"] == "test-backend"
    assert result.validation_issues == []


def test_missing_reference_unavailable_backend_and_runtime_failure_are_no_ops(
    tmp_path, monkeypatch
):
    import torch

    from ayase.modules.speech_bert_score import SpeechBERTScoreModule

    candidate_path = tmp_path / "candidate.wav"
    reference_path = tmp_path / "reference.wav"
    candidate_path.touch()
    reference_path.touch()

    unavailable = SpeechBERTScoreModule()
    no_reference = Sample(path=candidate_path, is_video=False)
    assert unavailable.process(no_reference) is no_reference
    assert no_reference.quality_metrics is None

    module = SpeechBERTScoreModule({"device": "cpu"})
    module._model = object()
    module._device = torch.device("cpu")
    monkeypatch.setattr(
        module,
        "_prepare_waveform",
        lambda _path: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )
    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )
    assert module.process(sample) is sample
    assert sample.quality_metrics is None
