import json

import pytest

from ayase.models import Sample
from ayase.third_party.vgif_score.protocol import evaluate_predictions, validate_annotations


def _annotations():
    dimensions = {}
    for key in ("cinematography", "purity", "motion_smoothness", "physics_adherence"):
        dimensions[key] = {
            "dimension_goal": f"Judge {key}",
            "vlm_scoring_prompt": f"Score visible {key}",
            "focus_points": ["visible evidence"],
            "score_anchors": {str(i): f"level {i}" for i in range(1, 6)},
        }
    return {
        "prompt": "A red ball rolls into a box, which then falls.",
        "original_qa_pairs": [
            {"id": "q1", "question": "Does the ball roll?", "expected_answer": "Yes", "type": "action", "dependency": "None"},
            {"id": "q2", "question": "Does the box fall?", "expected_answer": "Yes", "type": "causal", "dependency": "q1"},
            {"id": "q3", "question": "Does the box remain blue?", "expected_answer": "No", "type": "attribute", "dependency": "q1 OR q2"},
        ],
        "dimensions": dimensions,
        "provenance": {"kind": "user_supplied"},
    }


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.requests = []

    def post(self, url, json, timeout):
        self.requests.append((url, json, timeout))
        payload = next(self.payloads)
        return _Response(payload)


def _envelope(data, version, response_id):
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(data)}]}}],
        "modelVersion": version,
        "responseId": response_id,
    }


def test_dependency_gating_matches_source_oracle():
    qas = _annotations()["original_qa_pairs"][:2]
    rows, raw, gated, blocked = evaluate_predictions(
        qas,
        {
            "q1": {"answer": "No", "reason": "not visible"},
            "q2": {"answer": "Yes", "reason": "visible"},
        },
    )
    assert (raw, gated, blocked) == (1, 0, 1)
    assert rows[1]["dependency_passed"] is False


def test_validation_rejects_missing_dependency_cycle_and_bad_rubric():
    missing = _annotations()
    missing["original_qa_pairs"][1]["dependency"] = "q99"
    with pytest.raises(ValueError, match="missing dependency"):
        validate_annotations(missing)

    cycle = _annotations()
    cycle["original_qa_pairs"][0]["dependency"] = "q2"
    with pytest.raises(ValueError, match="cycle"):
        validate_annotations(cycle)

    forward = _annotations()
    forward["original_qa_pairs"][0]["dependency"] = "q2"
    forward["original_qa_pairs"][1]["dependency"] = "None"
    validate_annotations(forward)

    malformed = _annotations()
    del malformed["dimensions"]["physics_adherence"]["score_anchors"]["5"]
    with pytest.raises(ValueError, match="anchors 1..5"):
        validate_annotations(malformed)


def test_test_mode_setup_does_not_create_session(monkeypatch):
    from ayase.modules.vgif_score import VGIFScoreModule

    monkeypatch.setenv("VGIF_API_KEY", "must-not-be-used")
    module = VGIFScoreModule({"test_mode": True})
    module.setup()
    assert module._backend is None
    assert module._session is None
    assert module._api_key is None


def test_setup_without_credentials_is_unavailable(monkeypatch):
    from ayase.modules.vgif_score import VGIFScoreModule

    monkeypatch.setattr(VGIFScoreModule, "_global_test_mode", False)
    monkeypatch.delenv("VGIF_API_KEY", raising=False)
    module = VGIFScoreModule({"base_url": "https://example.invalid"})
    module.setup()
    assert module._backend == "unavailable"
    assert module._session is None


def test_full_dependency_round_workflow_and_source_math(tmp_path):
    from ayase.modules.vgif_score import VGIFScoreModule

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-bytes")
    sample = Sample(path=video, is_video=True, metadata={"vgif_annotations": _annotations()})
    payloads = [
        _envelope(
            {"answers": []},
            "objective-v1", "r1",
        ),
        _envelope(
            {"answers": [{"id": "q2", "answer": "Yes", "reason": "box falls"}]},
            "objective-v1", "r2",
        ),
        _envelope(
            {"answers": [{"id": "q3", "answer": "No", "reason": "not blue"}]},
            "objective-v1", "r3",
        ),
        _envelope(
            {"scores": {"Cin": {"score": 9}}},
            "rubric-v1", "bad-rubric",
        ),
        _envelope(
            {"scores": {
                "Cin": {"score": "score 1", "reason": "poor"},
                "Pur": {"score": 1.0, "reason": "poor"},
                "Mot": {"score": 1, "reason": "poor"},
                "Phy": {"score": 1, "reason": "poor"},
            }},
            "rubric-v1", "r4",
        ),
    ]
    module = VGIFScoreModule()
    module._backend = "gemini_api"
    module._session = _Session(payloads)
    result = module.process(sample)

    assert result is sample
    assert sample.quality_metrics.vgif_objective_score == 0.0
    assert sample.quality_metrics.vgif_subjective_score == pytest.approx(0.2)
    assert sample.quality_metrics.vgif_score == pytest.approx(0.1)
    assert sample.quality_metrics.vgif_cinematography_score == 1.0
    trace = sample.metadata["vgif_score"]
    assert trace["raw_correct_count"] == 2
    assert trace["dependency_blocked_count"] == 2
    assert trace["answer_coverage"] == pytest.approx(2 / 3)
    assert [call["response_id"] for call in trace["calls"]] == ["r1", "r2", "r3", "r4"]
    assert "Previous locked answers:\nNone" in module._session.requests[1][1]["contents"][0]["parts"][0]["text"]
    assert len(module._session.requests) == 5
    assert module._session.requests[0][1]["model"] == "gemini-3.1-pro-preview"
    assert "role" not in module._session.requests[0][1]["contents"][0]
    assert all(request[1]["contents"][0]["parts"][1]["inline_data"]["data"] for request in module._session.requests)


def test_invalid_input_is_rejected_before_video_read_or_upload(tmp_path, monkeypatch):
    from ayase.modules.vgif_score import VGIFScoreModule

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    annotations = _annotations()
    annotations["original_qa_pairs"] = []
    sample = Sample(path=path, is_video=True, metadata={"vgif_annotations": annotations})
    module = VGIFScoreModule({"annotations": _annotations()})
    module._backend = "gemini_api"
    monkeypatch.setattr(module, "_post", lambda *args: pytest.fail("must not upload"))
    monkeypatch.setattr(type(path), "read_bytes", lambda self: pytest.fail("must not read video"))
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_malformed_response_leaves_all_fields_unset(tmp_path):
    from ayase.modules.vgif_score import VGIFScoreModule

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    sample = Sample(path=path, is_video=True, metadata={"vgif_annotations": _annotations()})
    module = VGIFScoreModule({"format_retries": 0})
    module._backend = "gemini_api"
    module._session = _Session([_envelope({"answers": []}, "v", "bad")])
    assert module.process(sample) is sample
    assert sample.quality_metrics is None
    assert "vgif_score" not in sample.metadata


@pytest.mark.parametrize("invalid", ["10", "25", True])
def test_rubric_does_not_turn_invalid_values_into_valid_ratings(invalid):
    from ayase.modules.vgif_score import VGIFScoreModule

    payload = {"scores": {key: {"score": invalid} for key in ("Cin", "Pur", "Mot", "Phy")}}
    with pytest.raises(ValueError):
        VGIFScoreModule._normalize_rubric(payload)


def test_text_extraction_skips_empty_parts():
    from ayase.modules.vgif_score import VGIFScoreModule

    envelope = {"candidates": [{"content": {"parts": [{"text": "  "}, {"text": "answer"}]}}]}
    assert VGIFScoreModule._extract_text(envelope) == "answer"
