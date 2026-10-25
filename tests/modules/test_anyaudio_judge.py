import math
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ayase.models import CaptionMetadata, Sample


class _Item:
    def __init__(self, item_id, question, p_yes):
        self.payload = {
            "id": item_id, "dimension": "sound", "question": question,
            "answer": "yes" if p_yes >= 0.5 else "no", "p_yes": p_yes, "evidence": "",
        }

    def to_dict(self):
        return dict(self.payload)


class _Judge:
    def __init__(self, probabilities=(0.25, 0.75)):
        self.probabilities = probabilities
        self.calls = []

    def judge(self, path, rubric, mode):
        self.calls.append((path, rubric, mode))
        items = [_Item(i, item["question"], p) for i, (item, p) in enumerate(
            zip(rubric, self.probabilities)
        )]
        return SimpleNamespace(score=sum(self.probabilities[:len(items)]) / len(items), items=items)


def _ready(module, judge=None):
    module._backend = "anyaudio-author-logits-7b"
    module._judge = judge or _Judge()
    module._load_source_audio = lambda path: np.zeros(160, dtype=np.float32)
    return module


def test_basics_and_pinned_model_identity():
    from ayase.modules.anyaudio_judge import (
        DECOMPOSER_REVISION, JUDGE_MODELS, SOURCE_REVISION, AnyAudioJudgeModule,
    )
    from tests.modules.conftest import _test_module_basics

    _test_module_basics(AnyAudioJudgeModule, "anyaudio_judge")
    assert len(SOURCE_REVISION) == len(DECOMPOSER_REVISION) == 40
    assert JUDGE_MODELS["7b"][0] == "cucl2/AnyAudio-Judge-7B"
    assert AnyAudioJudgeModule.default_config["model_size"] == "7b"
    assert AnyAudioJudgeModule.provenance == {"anyaudio_judge_score": "published"}


def test_test_mode_never_acquires_source_or_models(monkeypatch):
    from ayase.modules.anyaudio_judge import AnyAudioJudgeModule

    module = AnyAudioJudgeModule({"test_mode": True})
    monkeypatch.setattr(module, "_prepare_backend", lambda: (_ for _ in ()).throw(
        AssertionError("provisioning attempted")
    ))
    module.setup()
    assert module._backend is None


def test_invalid_dtype_fails_before_source_or_model_acquisition(monkeypatch):
    from ayase.modules.anyaudio_judge import AnyAudioJudgeModule
    import ayase.modules.anyaudio_judge as implementation

    monkeypatch.setattr(implementation, "_ensure_source", lambda path: (_ for _ in ()).throw(
        AssertionError("source acquisition attempted")
    ))
    module = AnyAudioJudgeModule({"dtype": "auto"})
    with pytest.raises(ValueError, match="dtype"):
        module._prepare_backend()


def test_sample_rubric_precedes_config_and_preserves_input(tmp_path):
    from ayase.modules.anyaudio_judge import AnyAudioJudgeModule

    supplied = [{"维度": "环境声", "题目": "是否有雨声？", "依据": "雨声"}]
    sample = Sample(path=tmp_path / "audio.wav", is_video=False)
    sample.metadata["anyaudio_judge_rubric"] = supplied
    module = _ready(AnyAudioJudgeModule({
        "rubric": [{"dimension": "wrong", "question": "wrong", "basis": "wrong"}]
    }), _Judge((0.8,)))

    assert module.process(sample) is sample
    assert sample.quality_metrics.anyaudio_judge_score == 0.8
    assert sample.metadata["anyaudio_judge_rubric"] == supplied
    trace = sample.metadata["anyaudio_judge"]
    assert trace["rubric_source"] == "sample_metadata"
    assert trace["rubric_mode"] == "supplied"
    assert trace["coverage"] == 1.0
    assert trace["model_revision"] == module.models[0]["revision"]
    assert module._judge.calls[0][2] == "logits"
    assert module._judge.calls[0][1][0]["question"] == "是否有雨声？"


def test_malformed_explicit_rubric_does_not_fall_back_to_caption(tmp_path, monkeypatch):
    from ayase.modules.anyaudio_judge import AnyAudioJudgeModule

    sample = Sample(
        path=tmp_path / "audio.wav", is_video=False,
        caption=CaptionMetadata(text="rain falling", length=12),
        metadata={"anyaudio_judge_rubric": [{"question": ""}]},
    )
    module = _ready(AnyAudioJudgeModule())
    monkeypatch.setattr(module, "_decompose", lambda *args: (_ for _ in ()).throw(
        AssertionError("caption fallback attempted")
    ))
    module.process(sample)
    assert sample.quality_metrics is None
    assert "anyaudio_judge" not in sample.metadata


def test_caption_uses_author_decomposition_alias_cache_and_holistic_trace(tmp_path, monkeypatch):
    from ayase.modules.anyaudio_judge import AnyAudioJudgeModule

    sample = Sample(
        path=tmp_path / "audio.wav", is_video=False,
        caption=CaptionMetadata(text="soft rain", length=9),
        metadata={"anyaudio_judge_subset": "sound_gen"},
    )
    module = _ready(AnyAudioJudgeModule(), _Judge((0.6,)))
    calls = []

    def decompose(caption, subset, clean):
        calls.append((caption, subset, clean))
        return ([{"dimension": "整体", "question": caption, "basis": caption}],
                "holistic_fallback")

    monkeypatch.setattr(module, "_decompose", decompose)
    module.process(sample)
    module.process(sample)
    assert calls == [("soft rain", "sound", True)]
    trace = sample.metadata["anyaudio_judge"]
    assert trace["rubric_mode"] == "holistic_fallback"
    assert trace["decomposer_revision"] is not None


def test_caption_falls_back_to_utf8_sidecar(tmp_path):
    from ayase.modules.anyaudio_judge import AnyAudioJudgeModule

    audio = tmp_path / "clip.wav"
    audio.with_suffix(".txt").write_text("  steady rainfall  ", encoding="utf-8")
    assert AnyAudioJudgeModule()._caption(Sample(path=audio, is_video=False)) == "steady rainfall"


def test_process_keeps_only_finite_item_coverage(tmp_path):
    from ayase.modules.anyaudio_judge import AnyAudioJudgeModule

    rubric = [
        {"dimension": "a", "question": "one?", "basis": "one"},
        {"dimension": "b", "question": "two?", "basis": "two"},
    ]
    module = _ready(AnyAudioJudgeModule({"rubric": rubric}), _Judge((0.3, float("nan"))))
    sample = Sample(path=tmp_path / "audio.wav", is_video=False)
    # Author aggregation ignores invalid items. Mirror that independently in this fixture.
    module._judge.judge = lambda path, rubric, mode: SimpleNamespace(
        score=0.3, items=[_Item(0, "one?", 0.3), _Item(1, "two?", float("nan"))]
    )
    module.process(sample)
    assert math.isclose(sample.quality_metrics.anyaudio_judge_score, 0.3)
    assert sample.metadata["anyaudio_judge"]["valid_items"] == 1
    assert sample.metadata["anyaudio_judge"]["coverage"] == 0.5
    assert sample.metadata["anyaudio_judge"]["items"][1]["p_yes"] is None
    json.dumps(sample.metadata, allow_nan=False)


def test_video_extract_preserves_rate_and_channels_for_author_loader(tmp_path, monkeypatch):
    from ayase.modules.anyaudio_judge import AnyAudioJudgeModule
    import ayase.managed_ffmpeg
    import ayase.modules.anyaudio_judge as implementation

    observed = {}
    monkeypatch.setattr(ayase.managed_ffmpeg, "resolve_media_tool", lambda *a, **k: Path("ffmpeg"))

    def run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(implementation.subprocess, "run", run)
    module = AnyAudioJudgeModule()
    sample = Sample(path=tmp_path / "video.mp4", is_video=True)
    output = module._audio_path(sample, tmp_path)
    command = observed["command"]
    assert output == tmp_path / "audio.wav"
    assert command[command.index("-c:a") + 1] == "pcm_f32le"
    assert "-ar" not in command and "-ac" not in command
    assert command[command.index("-map") + 1] == "0:a:0"


def test_invalid_float_wav_skips_before_caption_decomposition(tmp_path, monkeypatch):
    import soundfile as sf
    from ayase.modules.anyaudio_judge import AnyAudioJudgeModule

    for name, values in {
        "empty.wav": np.empty(0, dtype=np.float32),
        "nonfinite.wav": np.array([0.0, np.nan, 0.25], dtype=np.float32),
    }.items():
        path = tmp_path / name
        sf.write(path, values, 16000, subtype="FLOAT")
        sample = Sample(
            path=path, is_video=False,
            caption=CaptionMetadata(text="rain falling", length=12),
        )
        module = _ready(AnyAudioJudgeModule())

        def source_loader(audio_path):
            audio, _ = sf.read(audio_path, dtype="float32", always_2d=True)
            return audio.mean(axis=1)

        monkeypatch.setattr(module, "_load_source_audio", source_loader)
        monkeypatch.setattr(module, "_decompose", lambda *args: (_ for _ in ()).throw(
            AssertionError("expensive decomposition attempted for invalid audio")
        ))
        assert module.process(sample) is sample
        assert sample.quality_metrics is None
        assert module._judge.calls == []


def test_full_omni_adapter_limits_thinker_disables_audio_and_unwraps_qwen3():
    from ayase.modules.anyaudio_judge import _install_generate_adapter

    observed = {}
    thinker_result = SimpleNamespace(scores=["first-step"])

    def generate(**kwargs):
        observed.update(kwargs)
        return thinker_result, None

    judge = SimpleNamespace(model=SimpleNamespace(generate=generate))
    _install_generate_adapter(judge)
    result = judge.model.generate(max_new_tokens=1, do_sample=False)
    assert result is thinker_result
    assert observed["thinker_max_new_tokens"] == 1
    assert observed["return_audio"] is False
    assert observed["max_new_tokens"] == 1


def test_dispose_clears_owned_and_author_decomposer_caches(monkeypatch):
    from ayase.modules.anyaudio_judge import AnyAudioJudgeModule

    cleared = []
    cached = SimpleNamespace(cache_clear=lambda: cleared.append(True))
    monkeypatch.setitem(sys.modules, "anyaudio_judge.decompose", types.SimpleNamespace(
        _cached_decomposer=cached
    ))
    module = AnyAudioJudgeModule()
    module._decomposition_cache[("caption", "mix", True)] = [{"question": "caption"}]
    module._decomposition_modes[("caption", "mix", True)] = "holistic_fallback"
    module._decomposer_path = Path("model")
    module.on_dispose()
    assert module._decomposition_cache == {}
    assert module._decomposition_modes == {}
    assert module._decomposer_path is None
    assert cleared == [True]


def test_omni_adapter_matches_audio_precision_and_preserves_integer_inputs():
    import torch
    from ayase.modules.anyaudio_judge import _install_generate_adapter

    observed = {}
    model = SimpleNamespace(
        thinker=SimpleNamespace(audio_tower=torch.nn.Linear(2, 2).bfloat16()),
        generate=lambda **kwargs: observed.update(kwargs),
    )
    judge = SimpleNamespace(model=model)
    features = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    mask = torch.tensor([[1, 1]], dtype=torch.int64)
    _install_generate_adapter(judge)
    judge.model.generate(input_features=features, input_ids=ids, feature_attention_mask=mask)
    assert observed["input_features"].dtype == torch.bfloat16
    assert torch.equal(observed["input_features"].float(), features)
    assert observed["input_ids"] is ids
    assert observed["feature_attention_mask"] is mask
