"""Behavioral tests for the PQSG adapter (no network or paid API calls)."""

from types import SimpleNamespace

import pytest

from ayase.models import CaptionMetadata, QualityMetrics, Sample
from ayase.modules.pqsg import PQSGModule
from ayase.third_party.pqsg import protocol


def _graph():
    return {
        "nodes": {
            "object_existence": {"O1": "Is there a ball?"},
            "action_verification": {"A1": "Does it fall?", "A2": "Does it land?"},
            "physics": {"P1": "Does it accelerate?", "P2": "Does it stop?"},
        },
        "edges": [
            {"from": ["O1"], "to": "A1"},
            {"from": ["A1"], "to": "A2"},
            {"from": ["O1"], "to": "P1"},
            {"from": ["P1"], "to": "P2"},
        ],
    }


def _answers(values):
    result = {}
    for category, questions in _graph()["nodes"].items():
        result[category] = {
            qid: {"is_correct": values[qid], "question": text} for qid, text in questions.items()
        }
    return result


def test_graph_validation_accepts_released_forward_extensions_and_rejects_bad_graphs():
    protocol.validate_graph(_graph())
    backward = _graph()
    backward["edges"].append({"from": ["P2"], "to": "A2"})
    with pytest.raises(ValueError, match="Unsupported"):
        protocol.validate_graph(backward)
    dangling = _graph()
    dangling["edges"].append({"from": ["O9"], "to": "A1"})
    with pytest.raises(ValueError, match="Dangling"):
        protocol.validate_graph(dangling)
    empty = {"nodes": {name: {} for name in protocol.CATEGORIES}, "edges": []}
    with pytest.raises(ValueError, match="no questions"):
        protocol.validate_graph(empty)


def test_dependency_propagation_and_conditional_category_scores():
    answers = _answers({"O1": True, "A1": False, "A2": True, "P1": True, "P2": False})
    assert protocol.tree_score(answers, _graph()["edges"]) == pytest.approx(0.4)
    assert protocol.category_scores(answers, _graph()["edges"]) == {
        "object": 1.0,
        "action": 0.0,
        "physics": 0.5,
    }
    blocked = _answers({"O1": False, "A1": True, "A2": True, "P1": True, "P2": True})
    assert protocol.category_scores(blocked, _graph()["edges"]) == {
        "object": 0.0,
        "action": None,
        "physics": None,
    }
    assert protocol.eligibility_mask(blocked, _graph()["edges"]) == {
        "O1": True,
        "A1": False,
        "A2": False,
        "P1": False,
        "P2": False,
    }


def test_openai_two_call_payload_and_missing_final_answers_default_no():
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            text = (
                "[O1]: visible\n[A1]: falling\n[A2]: landed\n[P1]: accelerating\n[P2]: stopped"
                if len(calls) == 1
                else "[O1]: yes\n[A1]: yes"
            )
            return SimpleNamespace(
                id=f"r{len(calls)}",
                model="gpt-returned",
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    answers, trace = protocol.answer_openai(
        "unused.mp4",
        _graph(),
        "gpt-5",
        2,
        client=client,
        frame_extractor=lambda *args, **kwargs: ["jpeg-a", "jpeg-b"],
    )
    assert len(calls) == 2
    assert calls[0]["max_completion_tokens"] == 16384
    assert calls[1]["max_completion_tokens"] == 8192
    assert calls[0]["messages"][0]["content"][1]["image_url"]["detail"] == "low"
    assert answers["action_verification"]["A2"]["final_answer"] == "no"
    assert trace["request_ids"] == ["r1", "r2"]


def test_module_scores_atomically_and_records_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(PQSGModule, "_global_test_mode", False)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-read-by-mock")
    sample = Sample(
        path=video, is_video=True, caption=CaptionMetadata(text="A ball falls", length=12)
    )
    answers = _answers({"O1": True, "A1": True, "A2": True, "P1": False, "P2": True})
    monkeypatch.setattr(
        protocol,
        "generate_psg",
        lambda *a, **k: (_graph(), {"request_id": "qg1", "raw_response": "{}"}),
    )
    monkeypatch.setattr(
        protocol,
        "answer_gemini",
        lambda **k: (
            answers,
            {"request_ids": ["qa1", "qa2"], "raw_responses": ["reason", "final"]},
        ),
    )
    module = PQSGModule({"models_dir": str(tmp_path), "qg_client": object(), "qa_client": object()})
    module.setup()
    result = module.process(sample)
    assert result is sample
    assert result.quality_metrics.pqsg_raw_score == pytest.approx(0.8)
    assert result.quality_metrics.pqsg_score == pytest.approx(0.6)
    assert result.metadata["pqsg"]["question_count"] == 5


def test_malformed_graph_fails_before_video_qa_and_writes_no_partial_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(PQSGModule, "_global_test_mode", False)
    (tmp_path / "v.mp4").write_bytes(b"video")
    sample = Sample(
        path=tmp_path / "v.mp4", is_video=True, caption=CaptionMetadata(text="x", length=1)
    )
    bad = {"nodes": {name: {} for name in protocol.CATEGORIES}, "edges": []}
    monkeypatch.setattr(protocol, "generate_psg", lambda *a, **k: (bad, {}))
    monkeypatch.setattr(protocol, "answer_gemini", lambda **k: pytest.fail("video QA must not run"))
    module = PQSGModule({"qg_client": object(), "qa_client": object()})
    module.setup()
    assert module.process(sample) is sample
    assert sample.quality_metrics is None
    assert "pqsg" not in sample.metadata


def test_test_mode_skips_setup_and_all_api_work(monkeypatch, tmp_path):
    module = PQSGModule({"test_mode": True, "models_dir": str(tmp_path)})
    monkeypatch.setattr(protocol, "generate_psg", lambda *a, **k: pytest.fail("API must not run"))
    module.on_mount()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    assert module.process(sample) is sample
    assert module._backend is None


def test_unmounted_and_missing_credentials_never_request(monkeypatch, tmp_path):
    monkeypatch.setattr(PQSGModule, "_global_test_mode", False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(protocol, "generate_psg", lambda *a, **k: pytest.fail("no request"))
    sample = Sample(
        path=tmp_path / "v.mp4",
        is_video=True,
        caption=CaptionMetadata(text="x", length=1),
    )
    module = PQSGModule({})
    assert module.process(sample) is sample
    module.setup()
    assert module._backend == "unavailable"
    assert module.process(sample) is sample


def test_missing_or_empty_video_never_spends_question_generation_request(monkeypatch, tmp_path):
    monkeypatch.setattr(PQSGModule, "_global_test_mode", False)
    monkeypatch.setattr(protocol, "generate_psg", lambda *a, **k: pytest.fail("no request"))
    module = PQSGModule({"qg_client": object(), "qa_client": object()})
    module.setup()
    for path in (tmp_path / "missing.mp4", tmp_path / "empty.mp4"):
        if path.name == "empty.mp4":
            path.touch()
        sample = Sample(
            path=path,
            is_video=True,
            caption=CaptionMetadata(text="x", length=1),
        )
        assert module.process(sample) is sample


def test_provider_defaults_survive_managed_worker_config_roundtrip():
    host = PQSGModule({"qg_backend": "openai", "qa_backend": "openai"})
    worker = PQSGModule(dict(host.config))
    assert worker._model("qg", "openai") == "gpt-5"
    assert worker._model("qa", "openai") == "gpt-5"
    assert worker._qa_fps("openai") == 2

    overridden = PQSGModule({"qa_backend": "openai", "qa_model": "gpt-custom", "qa_fps": 8})
    worker = PQSGModule(dict(overridden.config))
    assert worker._model("qa", "openai") == "gpt-custom"
    assert worker._qa_fps("openai") == 8


def test_api_keys_are_redacted_from_failures(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(PQSGModule, "_global_test_mode", False)
    secret = "top-secret-pqsg-key"
    module = PQSGModule(
        {
            "qg_api_key": secret,
            "qa_api_key": secret,
            "qg_client": object(),
            "qa_client": object(),
        }
    )
    module.setup()
    monkeypatch.setattr(
        protocol,
        "generate_psg",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"provider echoed {secret}")),
    )
    (tmp_path / "v.mp4").write_bytes(b"video")
    sample = Sample(
        path=tmp_path / "v.mp4",
        is_video=True,
        caption=CaptionMetadata(text="x", length=1),
    )
    module.process(sample)
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text
