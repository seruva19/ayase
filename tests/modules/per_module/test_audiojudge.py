import base64
import hashlib
import json
import sys
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ayase.models import Sample
from ayase.pipeline import PipelineModule
from ayase.third_party.audiojudge import protocol


@pytest.fixture(autouse=True)
def _allow_mocked_transport(monkeypatch):
    previous = PipelineModule._global_test_mode
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    PipelineModule.set_test_mode(False)
    try:
        yield
    finally:
        PipelineModule.set_test_mode(previous)


def _wav(path: Path, frames: bytes, rate: int = 24000, channels: int = 1, width: int = 2):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        stream.writeframes(frames)
    return path


def _ready_module(config, response):
    from ayase.modules.audiojudge import AudioJudgeModule

    module = AudioJudgeModule(config)
    create = _Create(response)
    module._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    module._backend = "openai-chat-completions-adapted"
    module._mounted = True
    return module, create


class _Create:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))],
            model="gpt-audio-1.5-actual",
            id="response-1",
            _request_id="request-1",
            usage=None,
        )


def test_audiojudge_basics_and_test_mode_skip(tmp_path):
    from tests.modules.conftest import _test_module_basics
    from ayase.modules.audiojudge import AudioJudgeModule

    _test_module_basics(AudioJudgeModule, "audiojudge")
    sample = Sample(path=_wav(tmp_path / "sample.wav", b"\x00\x00" * 8), is_video=False)
    module = AudioJudgeModule({"test_mode": True, "api_key": "must-not-be-used"})
    module.on_mount()
    assert module.process(sample) is sample
    assert module._client is None
    assert "audiojudge" not in sample.metadata


def test_all_canonical_tasks_variants_and_prompt_bytes():
    pair = {
        "pronunciation": 4,
        "speaker": 4,
        "speed": 4,
        "tmhintq": 4,
        "somos": 4,
        "thaimos": 4,
        "speakbench": 7,
        "speakbench508": 7,
        "chatbotarena": 1,
        "chatbotarena_BA": 1,
    }
    point = {"tmhintq": 2, "somos": 2, "thaimos": 2, "speakbench": 2, "speakbench508": 2}
    assert {key: len(value) for key, value in protocol.PAIRWISE_PROMPTS.items()} == pair
    assert {key: len(value) for key, value in protocol.POINTWISE_PROMPTS.items()} == point
    assert sum(pair.values()) + sum(point.values()) == 50
    assert protocol.canonical_contract("pairwise", "speed", "no_cot").semantic == "choice"
    assert protocol.canonical_contract("pointwise", "somos", "standard_cot").semantic == "score"
    root = Path(protocol.__file__).parent
    assert hashlib.sha256((root / "system_prompts.py").read_bytes()).hexdigest() == (
        "b8a60dee542fc08ce677e0fe46cdf1ecdbb20a8152e3cf9f6f13fcf911b044c2"
    )
    assert hashlib.sha256((root / "system_prompts_pointwise.py").read_bytes()).hexdigest() == (
        "394497c2b50900d5cf4b089480d60c8a3d03dd047ccab3c2717902ed03353040"
    )
    assert hashlib.sha256((root / "LICENSE").read_bytes()).hexdigest() == (
        "d6d3e8a7c97bec99a7cc2b55e319fcb872b0d4e7a1ba24e09ab3af5798211913"
    )
    assert set(protocol._ASSET_SHA256) == {
        path.name for path in (root / "signal_audios").glob("*.wav")
    }
    for name, digest in protocol._ASSET_SHA256.items():
        assert hashlib.sha256((root / "signal_audios" / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "contract,text,mapped",
    [
        (("pointwise", "somos", "no_cot"), '{"score":5}', 5.0),
        (("pairwise", "speaker", "no_cot"), '{"match":false}', 0.0),
        (("pairwise", "speed", "no_cot"), '{"label":"2"}', 2.0),
        (
            ("pairwise", "speakbench", "standard_cot"),
            '{"reasoning":"same", "label":"tie"}',
            0.0,
        ),
    ],
)
def test_strict_parser_valid(contract, text, mapped):
    resolved = protocol.canonical_contract(*contract)
    assert protocol.parse_response(text, resolved)[0] == mapped


@pytest.mark.parametrize(
    "text",
    [
        '{"score":true}',
        '{"score":"5"}',
        '{"score":6}',
        '{"score":5,"extra":0}',
        '{"score":1,"score":5}',
        "```json\n{\"score\":5}\n```",
    ],
)
def test_strict_parser_rejects_malformed_or_ambiguous(text):
    contract = protocol.canonical_contract("pointwise", "somos", "no_cot")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        protocol.parse_response(text, contract)


def test_custom_prompt_requires_exact_schema_and_semantic():
    schema = {
        "result_key": "winner",
        "result_type": "string",
        "allowed": ["1", "2", "tie"],
        "reasoning_key": None,
    }
    contract = protocol.custom_contract("pairwise", "custom", "preference", schema)
    assert protocol.parse_response('{"winner":"1"}', contract)[0] == 1.0
    with pytest.raises(ValueError):
        protocol.custom_contract("pairwise", "custom", "choice", schema)
    with pytest.raises(ValueError):
        protocol.custom_contract(
            "pairwise", "custom", "preference", {**schema, "reasoning_key": "winner"}
        )


def test_pair_and_pointwise_pcm_order_duration_and_repair(tmp_path):
    first = _wav(tmp_path / "first.wav", b"\xe8\x03" * 17)
    second = _wav(tmp_path / "second.wav", b"\xd0\x07" * 23)
    joined = protocol.concatenate_pair_audio(
        [[first, second]], tmp_path / "pair.wav", is_test=True
    )
    with wave.open(str(joined), "rb") as stream:
        assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) == (
            24000,
            1,
            2,
        )
        expected = sum(
            wave.open(str(path), "rb").getnframes()
            for path in (
                protocol._asset("test.wav"),
                protocol._asset("audio_1.wav"),
                first,
                protocol._asset("audio_2.wav"),
                second,
            )
        ) + 5 * 12000
        assert stream.getnframes() == expected
    point = protocol.concatenate_pointwise_audio([first], tmp_path / "point.wav")
    with wave.open(str(point), "rb") as stream:
        expected = sum(
            wave.open(str(path), "rb").getnframes()
            for path in (protocol._asset("example_1.wav"), protocol._asset("audio.wav"), first)
        ) + 2 * 12000 + 16800
        assert stream.getnframes() == expected

    stereo = _wav(tmp_path / "stereo.wav", b"\xe8\x03\xb8\x0b" * 3, channels=2)
    normalized = protocol.normalize_wav(stereo, tmp_path / "mono.wav")
    with wave.open(str(normalized), "rb") as stream:
        assert stream.getnframes() == 3
        assert stream.readframes(3) == b"\xd0\x07" * 3
    unsigned = _wav(tmp_path / "u8.wav", bytes((0, 128, 255)), width=1)
    normalized = protocol.normalize_wav(unsigned, tmp_path / "s16.wav")
    with wave.open(str(normalized), "rb") as stream:
        assert stream.readframes(3) == b"\x00\x80\x00\x00\x00\x7f"


@pytest.mark.parametrize("method", protocol.PAIRWISE_METHODS)
def test_openai_pairwise_message_methods(tmp_path, method):
    first = _wav(tmp_path / "first.wav", b"\x00\x00" * 5)
    second = _wav(tmp_path / "second.wav", b"\x01\x00" * 7)
    messages = protocol.build_openai_messages(
        mode="pairwise",
        candidate_path=first,
        comparator_path=second,
        system_prompt="SYSTEM",
        user_prompt="USER",
        examples=(),
        concatenation_method=method,
        work_dir=tmp_path,
    )
    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    assert messages[-1]["role"] == "user"
    audio_parts = [
        part
        for part in messages[-1]["content"]
        if isinstance(part, dict) and part.get("type") == "input_audio"
    ]
    assert len(audio_parts) == (1 if "test_concatenation" in method else 2)
    assert all(base64.b64decode(part["input_audio"]["data"])[:4] == b"RIFF" for part in audio_parts)


def test_mocked_openai_transport_and_pointwise_metadata(tmp_path):
    source = _wav(tmp_path / "candidate.wav", b"\x00\x00" * 100)
    module, create = _ready_module(
        {"mode": "pointwise", "task": "somos", "variant": "no_cot"},
        '{"score":4}',
    )
    sample = Sample(path=source, is_video=False)
    assert module.process(sample) is sample
    assert sample.quality_metrics.audiojudge_score == 4.0
    assert sample.quality_metrics.audiojudge_preference is None
    assert sample.metadata["audiojudge"]["raw_response"] == '{"score":4}'
    assert sample.metadata["audiojudge"]["request"]["requested_model"] == "gpt-audio-1.5"
    payload = create.calls[0]
    assert payload["modalities"] == ["text"]
    assert payload["max_completion_tokens"] == 800
    assert payload["temperature"] == 1e-8


@pytest.mark.parametrize(
    "task,response,field,value",
    [
        ("speed", '{"label":"2"}', "audiojudge_choice", 2.0),
        ("speaker", '{"match":true}', "audiojudge_match", 1.0),
        ("somos", '{"label":"2"}', "audiojudge_preference", -1.0),
    ],
)
def test_pairwise_semantics_never_cross_populate_quality(tmp_path, task, response, field, value):
    candidate = _wav(tmp_path / "candidate.wav", b"\x00\x00" * 10)
    comparator = _wav(tmp_path / "comparator.wav", b"\x01\x00" * 10)
    module, _ = _ready_module(
        {
            "mode": "pairwise",
            "task": task,
            "variant": "no_cot",
            "comparator_path": str(comparator),
            "concatenation_method": "no_concatenation",
        },
        response,
    )
    sample = Sample(path=candidate, is_video=False)
    module.process(sample)
    assert getattr(sample.quality_metrics, field) == value
    assert sample.quality_metrics.audiojudge_score is None
    if task != "somos":
        assert sample.quality_metrics.audiojudge_preference is None


def test_instruction_contract_and_max_examples_fail_before_api(tmp_path):
    candidate = _wav(tmp_path / "candidate.wav", b"\x00\x00" * 10)
    comparator = _wav(tmp_path / "comparator.wav", b"\x01\x00" * 10)
    module, create = _ready_module(
        {
            "task": "speakbench",
            "variant": "standard_cot",
            "comparator_path": str(comparator),
            "concatenation_method": "no_concatenation",
        },
        '{"reasoning":"x","label":"1"}',
    )
    sample = Sample(path=candidate, is_video=False)
    module.process(sample)
    assert not create.calls
    assert "requires the original instruction" in sample.metadata["audiojudge"]["error"]

    module.config["task"] = "somos"
    module.config["variant"] = "no_cot"
    module.config["examples"] = [{}] * 11
    sample = Sample(path=candidate, is_video=False)
    module.process(sample)
    assert not create.calls
    assert "at most 10" in sample.metadata["audiojudge"]["error"]


def test_test_mode_blocks_previously_ready_transport(tmp_path):
    source = _wav(tmp_path / "candidate.wav", b"\x00\x00" * 10)
    module, create = _ready_module(
        {"mode": "pointwise", "task": "somos", "variant": "no_cot"},
        '{"score":4}',
    )
    module.config["test_mode"] = True
    sample = Sample(path=source, is_video=False)
    assert module.process(sample) is sample
    assert not create.calls
    assert sample.quality_metrics is None


def test_parse_failure_preserves_raw_and_provider_response_metadata(tmp_path):
    source = _wav(tmp_path / "candidate.wav", b"\x00\x00" * 10)
    module, create = _ready_module(
        {"mode": "pointwise", "task": "somos", "variant": "no_cot"},
        '{"score":4,"extra":true}',
    )
    sample = Sample(path=source, is_video=False)
    module.process(sample)
    detail = sample.metadata["audiojudge"]
    assert len(create.calls) == 1
    assert detail["success"] is False
    assert detail["raw_response"] == '{"score":4,"extra":true}'
    assert detail["task"] == "somos"
    assert detail["request"]["requested_model"] == "gpt-audio-1.5"
    assert detail["response"]["actual_model"] == "gpt-audio-1.5-actual"
    assert detail["response"]["request_id"] == "request-1"


def test_dispose_closes_sdk_client_and_calls_base_cleanup():
    from ayase.modules.audiojudge import AudioJudgeModule

    closed = []
    module = AudioJudgeModule()
    module._client = SimpleNamespace(close=lambda: closed.append(True))
    module._backend = "openai-chat-completions-adapted"
    module._mounted = True
    module.on_dispose()
    assert closed == [True]
    assert module._client is None
    assert module._backend is None
    assert module._mounted is False


@pytest.mark.parametrize(
    "mode,task,variant",
    [
        ("pairwise", "speakbench", "standard_cot"),
        ("pairwise", "speakbench508", "lexical_cot"),
        ("pairwise", "chatbotarena", "standard_cot"),
        ("pairwise", "chatbotarena_BA", "standard_cot"),
        ("pointwise", "speakbench", "no_cot"),
        ("pointwise", "speakbench508", "standard_cot"),
    ],
)
def test_instruction_conditioned_profiles_require_explicit_context(mode, task, variant):
    from ayase.modules.audiojudge import AudioJudgeModule

    module = AudioJudgeModule()
    contract = protocol.canonical_contract(mode, task, variant)
    with pytest.raises(ValueError, match="requires|instruction_text"):
        module._validate_instruction_contract(contract, None, "none", 0)
    module._validate_instruction_contract(contract, None, "text", 0)


def test_mocked_gemini_transport_converts_audio_parts_and_controls(monkeypatch):
    from ayase.modules.audiojudge import AudioJudgeModule

    calls = []

    class Part:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"part_data": data, "part_mime": mime_type}

    fake_types = SimpleNamespace(
        Part=Part,
        GenerateContentConfig=lambda **kwargs: {"config": kwargs},
    )
    fake_genai = ModuleType("google.genai")
    fake_genai.types = fake_types
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            text='{"score":3}',
            model_version="gemini-explicit-actual",
            response_id="gemini-response-1",
            usage_metadata=None,
        )

    module = AudioJudgeModule({"provider": "gemini", "model": "gemini-explicit"})
    module._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    text, metadata = module._call_gemini(
        ["SYSTEM", {"mime_type": "audio/wav", "data": b"RIFFpayload"}],
        "gemini-explicit",
    )
    assert text == '{"score":3}'
    assert calls[0]["contents"][1] == {
        "part_data": b"RIFFpayload",
        "part_mime": "audio/wav",
    }
    assert calls[0]["config"]["config"] == {
        "temperature": 1e-8,
        "max_output_tokens": 800,
        "response_mime_type": "application/json",
    }
    assert metadata["actual_model"] == "gemini-explicit-actual"
    assert metadata["response_id"] == "gemini-response-1"
