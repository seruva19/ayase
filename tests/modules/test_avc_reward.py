import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ayase.models import Sample


def test_basics_and_research_only_model_metadata():
    from ayase.modules.avc_reward import AVCRewardModule, MODEL_REVISION, SOURCE_REVISION
    from tests.modules.conftest import _test_module_basics

    _test_module_basics(AVCRewardModule, "avc_reward")
    assert len(MODEL_REVISION) == len(SOURCE_REVISION) == 40
    assert AVCRewardModule.models[0]["auto_download"] is True
    assert "Research use only" in AVCRewardModule.models[0]["notes"]


def test_parser_uses_tail_first_complete_object_aliases_and_clamps():
    from ayase.modules.avc_reward import parse_avc_reward_output

    text = '<think>{"macro_semantic_alignment": 0}</think> preface '
    text += '{"macro semantic alignment": 1.2, "temporal_sync": .25, '
    text += '"fine-grained event correspondence": -0.1}'
    scores = parse_avc_reward_output(text)
    assert scores["macro_semantic_alignment"] == 1.0
    assert scores["temporal_synchronicity"] == 0.25
    assert scores["fine_grained_event_correspondence"] == 0.0
    assert math.isclose(scores["score"], 1.25 / 3)


def test_parser_uses_last_repeated_plain_label():
    from ayase.modules.avc_reward import parse_avc_reward_output

    scores = parse_avc_reward_output(
        "macro_semantic_alignment: 0.1 macro_semantic_alignment: 0.9 "
        "temporal_synchronicity: 0.7 fine_grained_event_correspondence: 0.9"
    )
    assert math.isclose(scores["score"], 2.5 / 3)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_parser_rejects_nonfinite_json(value):
    from ayase.modules.avc_reward import parse_avc_reward_output

    text = (
        '{"macro_semantic_alignment": '
        + value
        + ', "temporal_synchronicity": 0.5, "fine_grained_event_correspondence": 0.5}'
    )
    with pytest.raises(ValueError, match="non-finite"):
        parse_avc_reward_output(text)


def test_explicit_missing_audio_rejects_without_sidecar_fallback(tmp_path):
    from ayase.modules.avc_reward import AVCRewardModule

    video = tmp_path / "clip.mp4"
    video.touch()
    video.with_suffix(".wav").touch()
    module = AVCRewardModule({"audio_path": tmp_path / "missing.wav"})
    with pytest.raises(FileNotFoundError, match="Explicit"):
        module._audio_input(Sample(path=video, is_video=True), tmp_path)


def test_audio_precedence_metadata_then_sidecar(tmp_path):
    from ayase.modules.avc_reward import AVCRewardModule

    video = tmp_path / "clip.mp4"
    configured = tmp_path / "chosen.flac"
    sidecar = video.with_suffix(".wav")
    for path in (video, configured, sidecar):
        path.touch()
    sample = Sample(path=video, is_video=True, metadata={"avc_audio_path": str(configured)})
    assert AVCRewardModule()._audio_input(sample, tmp_path) == (
        configured.resolve(),
        "sample_metadata",
    )


def test_test_mode_never_acquires_protocol_or_model(monkeypatch):
    from ayase.modules.avc_reward import AVCRewardModule

    module = AVCRewardModule({"test_mode": True})
    monkeypatch.setattr(
        module,
        "_prepare_backend",
        lambda: (_ for _ in ()).throw(AssertionError("acquisition attempted")),
    )
    module.setup()
    assert module._backend is None


def test_strict_checkpoint_loading_rejects_every_partial_state_category():
    from ayase.modules.avc_reward import _validate_loading_info

    _validate_loading_info({})
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        with pytest.raises(ValueError, match=key):
            _validate_loading_info({key: ["broken.entry"]})


def test_native_generate_separates_audio_and_unwraps_thinker(tmp_path):
    from ayase.modules.avc_reward import AVCRewardModule

    observed = {}

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return "templated"

        def batch_decode(self, tokens, **kwargs):
            return ["decoded"]

        def __call__(self, **kwargs):
            observed["processor"] = kwargs
            return {"input_ids": Tensor()}

    processor = Processor()
    thinker = SimpleNamespace(sequences=SimpleNamespace(__getitem__=lambda self, key: self))

    class Tensor:
        shape = (1, 3)

        def to(self, device):
            return self

        def __getitem__(self, key):
            return self

    thinker.sequences = Tensor()

    def generate(**kwargs):
        observed["generate"] = kwargs
        return thinker, None

    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device="cpu", dtype="float32")]),
        generate=generate,
    )
    module = AVCRewardModule()
    module._model = model
    module._processor = processor
    module._system_prompt = "system"
    module._user_prompt = "user"
    module._video_frames = lambda path: ("raw-video", {"selected_frames": 16})

    def process_mm_info(messages, **kwargs):
        observed["helper"] = kwargs
        return [np.zeros(161, dtype=np.float32)], None, None, {}

    module._process_mm_info = process_mm_info
    assert module._generate(tmp_path / "v.mp4", tmp_path / "a.wav") == "decoded"
    assert observed["helper"]["use_audio_in_video"] is False
    assert observed["helper"]["return_video_kwargs"] is True
    assert observed["processor"]["use_audio_in_video"] is False
    assert observed["processor"]["videos"] == ["raw-video"]
    assert len(observed["processor"]["audio"][0]) == 320
    assert "fps" not in observed["processor"]
    assert observed["generate"]["use_audio_in_video"] is False
    assert observed["generate"]["return_audio"] is False
    assert observed["generate"]["thinker_return_dict_in_generate"] is True


def test_process_is_atomic_on_invalid_response(tmp_path):
    from ayase.modules.avc_reward import AVCRewardModule

    video = tmp_path / "clip.mp4"
    audio = video.with_suffix(".wav")
    video.touch()
    audio.touch()
    module = AVCRewardModule({"models_dir": str(tmp_path)})
    module._backend = "mock"
    module._generate = lambda *args: '{"macro_semantic_alignment": 0.5}'
    sample = Sample(path=video, is_video=True)
    assert module.process(sample) is sample
    assert sample.quality_metrics is None
    assert "avc_reward" not in sample.metadata
