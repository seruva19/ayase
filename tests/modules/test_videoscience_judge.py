import json
import math

import cv2
import numpy as np
import pytest

from ayase.models import CaptionMetadata, Sample
from ayase.third_party.videoscience.protocol import (
    CHECKLIST_KEYS,
    build_checklist_prompt,
    build_judge_prompt,
    compute_overall,
    extract_frames,
    parse_judgment,
)


@pytest.fixture(autouse=True)
def _local_test_mode(monkeypatch):
    from ayase.pipeline import PipelineModule

    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    previous = PipelineModule._global_test_mode
    PipelineModule._global_test_mode = False
    yield
    PipelineModule._global_test_mode = previous


def _response(scores=None):
    values = scores or {
        "prompt_consistency": 1,
        "expected_phenomenon": 2,
        "coherence": 3,
        "immutability": 4,
        "dynamism": 2,
    }
    return json.dumps({"scores": values, "explanations": {"summary": "ok", "issues": []}})


class _Response:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {"x-request-id": "req-test"}
        self.content = b""

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self.payload)


def _mounted_module(config=None):
    from ayase.modules.videoscience_judge import VideoScienceJudgeModule

    module = VideoScienceJudgeModule({"api_key": "secret-test-key", **(config or {})})
    module._mounted = True
    module._backend = str(module.config["provider"]) + "-released-protocol"
    return module


def test_module_basics():
    from tests.modules.conftest import _test_module_basics
    from ayase.modules.videoscience_judge import VideoScienceJudgeModule

    _test_module_basics(VideoScienceJudgeModule, "videoscience_judge")


def test_parser_accepts_direct_and_fenced_json_and_rejects_bad_scores():
    direct = parse_judgment(_response())
    fenced = parse_judgment("prose\n```json\n" + _response() + "\n```\nmore")
    assert direct["scores"] == fenced["scores"]
    for invalid in (None, float("nan"), 0, 5, "bad"):
        scores = {key: 2 for key in direct["scores"]}
        scores["coherence"] = invalid
        with pytest.raises(ValueError):
            parse_judgment(_response(scores))


def test_source_weights_are_raw_one_to_four():
    scores = {
        "prompt_consistency": 1,
        "expected_phenomenon": 2,
        "coherence": 3,
        "immutability": 4,
        "dynamism": 2,
    }
    assert compute_overall(scores) == pytest.approx(2.3)
    assert compute_overall({key: 1 for key in scores}) == 1
    assert compute_overall({key: 4 for key in scores}) == 4


def test_prompt_uses_authoritative_truth_checklist_and_caller_cv_evidence():
    checklist = {key: [f"{key} visible"] for key in CHECKLIST_KEYS}
    prompt = build_judge_prompt(
        "authoritative truth",
        "experiment prompt",
        checklist,
        {"results": {"raft": {"mean_flow_magnitude": 1}}},
    )
    assert "Ground-truth phenomenon: authoritative truth" in prompt
    assert "Ground-truth description (authoritative):\nexperiment prompt" in prompt
    assert "phenomenon_congruency visible" in prompt
    assert "RAFT optical flow: mean flow magnitude≈1.000" in prompt


def test_checklist_prompt_does_not_include_candidate_content():
    prompt = build_checklist_prompt("experiment", "truth", "trusted source", "https://source")
    assert "experiment" in prompt and "truth" in prompt and "trusted source" in prompt
    assert "candidate" not in prompt.lower()


def test_missing_ground_truth_skips_without_transport(video_sample):
    module = _mounted_module()
    module._session = pytest.fail
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_process_requires_mount(video_sample):
    module = _mounted_module({"experiment_prompt": "setup", "expected_phenomenon": "truth"})
    module._mounted = False
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_openai_payload_frame_counts_weights_and_metadata(video_sample):
    video_sample.reference_path = video_sample.path
    module = _mounted_module({"experiment_prompt": "setup", "expected_phenomenon": "truth"})
    session = _Session({"output_text": _response(), "model": "gpt-5-pro-2026-test"})
    module._session = session
    result = module.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.videoscience_score == pytest.approx(2.3)
    assert result.quality_metrics.videoscience_prompt_score == 1
    assert result.quality_metrics.videoscience_phenomenon_score == 2
    _, call = session.calls[0]
    payload = call["json"]
    assert payload["max_output_tokens"] == 20480
    assert "temperature" not in payload and "seed" not in payload
    content = payload["input"][0]["content"]
    assert sum(item["type"] == "input_image" for item in content) == 30
    metadata = result.metadata["videoscience"]
    assert len(metadata["candidate_evidence"]) == 20
    assert len(metadata["reference_evidence"]) == 10
    assert metadata["actual_model"] == "gpt-5-pro-2026-test"
    assert metadata["request_id"] == "req-test"
    assert "secret-test-key" not in json.dumps(metadata)


def test_transport_failure_is_atomic_and_redacted(video_sample, caplog):
    module = _mounted_module({"experiment_prompt": "setup", "expected_phenomenon": "truth"})

    class Broken:
        def post(self, *args, **kwargs):
            raise RuntimeError("secret-test-key")

    module._session = Broken()
    with caplog.at_level("WARNING"):
        assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None
    assert "secret-test-key" not in caplog.text


def test_extracts_source_24_then_first_20_and_reference_12_then_first_10(tmp_path):
    path = tmp_path / "long.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (32, 32))
    for index in range(240):
        writer.write(np.full((32, 32, 3), index % 255, np.uint8))
    writer.release()
    candidate = extract_frames(path, 24)
    reference = extract_frames(path, 12)
    assert len(candidate) == 24 and len(reference) == 12
    assert candidate[0][0] == 0
    assert candidate[19][0] == pytest.approx(19 * 10 / 24, rel=0.01)
    assert reference[9][0] == pytest.approx(9 * 10 / 12, rel=0.01)
    assert candidate[19][0] < 10 and reference[9][0] < 10


def test_canonical_scenario_override_and_mismatch(tmp_path):
    module = _mounted_module({"models_dir": str(tmp_path), "scenario_id": "7"})
    path = module._canonical_file()
    path.parent.mkdir(parents=True)
    rows = [
        {"vid": str(index), "prompt": f"canonical {index}", "expected phenomenon": f"truth {index}"}
        for index in range(160)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    sample = Sample(
        path=tmp_path / "candidate.mp4",
        is_video=True,
        caption=CaptionMetadata(text="different", length=9),
    )
    prompt, truth, provenance = module._inputs(sample)
    assert (prompt, truth) == ("canonical 7", "truth 7")
    assert provenance["mode"] == "canonical-160"
    assert provenance["scenario_id"] == "7"
    assert provenance["caption_mismatch"] is True


def test_canonical_missing_or_invalid_file_skips(tmp_path, video_sample):
    module = _mounted_module({"models_dir": str(tmp_path), "scenario_id": "nope"})
    path = module._canonical_file()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"vid": "1", "prompt": "p"}), encoding="utf-8")
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_checklist_generation_requires_authoritative_source(video_sample):
    module = _mounted_module(
        {
            "provider": "gemini",
            "experiment_prompt": "setup",
            "expected_phenomenon": "truth",
            "generate_checklist": True,
        }
    )
    module._gemini_client = object()
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


@pytest.mark.parametrize("reference_kind", ["missing", "empty"])
def test_invalid_media_skips_before_paid_checklist_call(tmp_path, reference_kind):
    candidate = tmp_path / "candidate.mp4"
    candidate.write_bytes(b"video")
    reference = tmp_path / "reference.mp4"
    if reference_kind == "empty":
        reference.write_bytes(b"")
    module = _mounted_module(
        {
            "provider": "gemini",
            "experiment_prompt": "setup",
            "expected_phenomenon": "truth",
            "generate_checklist": True,
            "source_url": "https://example.invalid/source",
            "source_text": "trusted",
            "reference_video_path": str(reference),
        }
    )
    module._gemini_client = pytest.fail
    sample = Sample(path=candidate, is_video=True)
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_empty_candidate_skips_before_paid_checklist_call(tmp_path):
    candidate = tmp_path / "candidate.mp4"
    candidate.write_bytes(b"")
    module = _mounted_module(
        {
            "provider": "gemini",
            "experiment_prompt": "setup",
            "expected_phenomenon": "truth",
            "generate_checklist": True,
            "source_url": "https://example.invalid/source",
            "source_text": "trusted",
        }
    )
    module._gemini_client = pytest.fail
    sample = Sample(path=candidate, is_video=True)
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_test_mode_never_provisions_or_calls(video_sample):
    module = _mounted_module(
        {
            "test_mode": True,
            "scenario_id": "7",
            "generate_checklist": True,
            "experiment_prompt": "setup",
            "expected_phenomenon": "truth",
        }
    )
    module._session = pytest.fail
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_setup_and_dispose_lifecycle(monkeypatch):
    module = _mounted_module({"api_key": None})
    module._mounted = False
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    module.setup()
    assert module._backend == "unavailable"
    module.on_dispose()
    assert module._backend is None and module._mounted is False


def test_provider_scoped_credentials_never_reuse_openai_key_for_gemini(monkeypatch):
    module = _mounted_module({"provider": "openai", "api_key": "openai-secret"})
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert module._api_key("openai") == "openai-secret"
    assert module._api_key("gemini") is None


def test_gemini_credential_precedence_and_google_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-env")
    explicit = _mounted_module(
        {
            "provider": "openai",
            "api_key": "openai-secret",
            "gemini_api_key": "gemini-explicit",
        }
    )
    assert explicit._api_key("gemini") == "gemini-explicit"

    gemini_env = _mounted_module({"provider": "openai", "api_key": "openai-secret"})
    assert gemini_env._api_key("gemini") == "gemini-env"
    monkeypatch.delenv("GEMINI_API_KEY")
    assert gemini_env._api_key("gemini") == "google-env"


def test_setup_never_transmits_openai_key_to_gemini_client(monkeypatch):
    from google import genai

    captured = []
    monkeypatch.setattr(genai, "Client", lambda **kwargs: captured.append(kwargs) or object())
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    module = _mounted_module(
        {"provider": "openai", "api_key": "openai-secret", "generate_checklist": True}
    )
    module._mounted = False
    module.setup()
    assert module._backend == "unavailable"
    assert captured == []

    module = _mounted_module(
        {
            "provider": "openai",
            "api_key": "openai-secret",
            "generate_checklist": True,
            "gemini_api_key": "gemini-secret",
        }
    )
    module._mounted = False
    module.setup()
    assert captured == [{"api_key": "gemini-secret"}]
    assert "openai-secret" not in json.dumps(captured)
