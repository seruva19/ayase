"""Focused tests for the pinned EditScore adapter."""

import ast
import math
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from ayase.models import CaptionMetadata, Sample
from tests.modules.conftest import _test_module_basics


@pytest.fixture(autouse=True)
def _json_repair_test_double(monkeypatch):
    """Supply the managed-runtime dependency without installing it in the host."""

    replacement = SimpleNamespace(loads=lambda value: ast.literal_eval(value))
    monkeypatch.setitem(
        sys.modules,
        "json_repair",
        replacement,
    )
    from ayase.third_party.edit_score import _upstream_utils

    monkeypatch.setattr(_upstream_utils, "json_repair", replacement)


def test_edit_score_basics_and_pins():
    from ayase.modules.edit_score import (
        ADAPTER_REVISION,
        BASE_MODEL_REVISION,
        EditScoreModule,
        SOURCE_REVISION,
    )

    _test_module_basics(EditScoreModule, "edit_score")
    module = EditScoreModule()
    assert module.runtime_trace["source_revision"] == SOURCE_REVISION
    assert BASE_MODEL_REVISION in module.runtime_trace["base_model"]
    assert ADAPTER_REVISION in module.runtime_trace["adapter"]
    assert module.runtime_trace["protocol"] == "two-images SC; edited-only PQ"
    assert module.runtime_trace["reduction"] == "average_last"


def test_source_evaluate_oracle_and_image_routing(monkeypatch):
    import ayase.third_party.edit_score.qwen3vl as qwen
    from ayase.third_party.edit_score import EditScore

    calls = []
    outputs = iter(
        [
            '{"reasoning":"sc1","score":[20,5]}',
            '{"reasoning":"pq1","score":[15,10]}',
            '{"reasoning":"sc2","score":[5,25]}',
            '{"reasoning":"pq2","score":[25,25]}',
        ]
    )

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def prepare_input(self, images, prompt):
            calls.append(("prepare", images, prompt))
            return len(calls)

        def inference(self, prepared, seed):
            calls.append(("infer", prepared, seed))
            return next(outputs)

    monkeypatch.setattr(qwen, "Qwen3VL", FakeModel)
    source, edited = object(), object()
    scorer = EditScore("qwen3vl", num_pass=2, seed=42)
    result = scorer.evaluate([source, edited], "make it blue")

    assert calls[0][1] == [source, edited]
    assert "make it blue" in calls[0][2]
    assert calls[1][1] is edited
    assert [entry[2] for entry in calls if entry[0] == "infer"] == [42, 42, 43, 43]
    assert result["prompt_following"] == pytest.approx(5.0)
    assert result["consistency"] == pytest.approx(6.0)
    assert result["perceptual_quality"] == pytest.approx(7.0)
    assert result["overall"] == pytest.approx(3.6502815398728847)
    assert result["SC_reasoning"] == "sc2"
    assert result["PQ_reasoning"] == "pq2"


@pytest.mark.parametrize(
    "raw,raises",
    [
        ("I'm sorry, but I can't assist with that request.", True),
        ('{"reasoning":"guess_if_cannot_parse | bad","score":[5,5]}', True),
        ('{"reasoning":"bad","score":[1]}', False),
        ('{"reasoning":"bad","score":[1,26]}', False),
        ('{"reasoning":"bad","score":[1,NaN]}', False),
        ("not json", False),
    ],
)
def test_strict_parser_rejects_non_scientific_outputs(raw, raises):
    from ayase.third_party.edit_score.utils import strict_mllm_output_to_dict

    if raises:
        with pytest.raises(ValueError):
            strict_mllm_output_to_dict(raw, score_range=25)
    else:
        assert strict_mllm_output_to_dict(raw, score_range=25) is False


def test_strict_parser_repairs_valid_json_only(monkeypatch):
    from ayase.third_party.edit_score import _upstream_utils
    from ayase.third_party.edit_score.utils import strict_mllm_output_to_dict

    monkeypatch.setattr(
        _upstream_utils,
        "json_repair",
        SimpleNamespace(
            loads=lambda _value: {"reasoning": "ok", "score": [12.5, 25]}
        ),
    )
    parsed = strict_mllm_output_to_dict(
        "prefix {'reasoning':'ok', 'score':[12.5, 25]} suffix", score_range=25
    )
    assert parsed == {"reasoning": "ok", "score": [12.5, 25.0]}


@pytest.mark.parametrize(
    "raw,expected_reasoning",
    [
        ("[20, 5]", "System: output is simply a list of scores"),
        (
            'prefix ||V^=^V||{"reasoning":"ok","score":[20,5]}||V^=^V|| suffix',
            "ok",
        ),
        ("{'reasoning':'repaired', 'score':[20,5]}", "repaired"),
    ],
)
def test_source_parser_success_paths(raw, expected_reasoning):
    from ayase.third_party.edit_score.utils import mllm_output_to_dict

    parsed = mllm_output_to_dict(raw, score_range=25)
    assert parsed == {"reasoning": expected_reasoning, "score": [20.0, 5.0]}


def test_valid_third_response_is_rejected_like_source_give_up(monkeypatch):
    import ayase.third_party.edit_score.qwen3vl as qwen
    from ayase.third_party.edit_score import EditScore

    outputs = iter(
        [
            "bad", "bad",
            "still bad", "still bad",
            '{"reasoning":"valid but too late","score":[20,5]}',
            '{"reasoning":"valid but too late","score":[20,5]}',
        ]
    )

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def prepare_input(self, images, prompt):
            return (images, prompt)

        def inference(self, _prepared, seed):
            return next(outputs)

    monkeypatch.setattr(qwen, "Qwen3VL", FakeModel)
    scorer = EditScore("qwen3vl", num_pass=1, seed=42)
    with pytest.raises(ValueError, match="exhausted parsing retries"):
        scorer.evaluate([object(), object()], "edit instruction")


def test_json_repair_failure_never_reaches_zero_padding(monkeypatch):
    import ayase.third_party.edit_score.qwen3vl as qwen
    from ayase.third_party.edit_score import EditScore
    from ayase.third_party.edit_score import _upstream_utils

    inference_calls = []

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def prepare_input(self, images, prompt):
            return (images, prompt)

        def inference(self, _prepared, seed):
            inference_calls.append(seed)
            # Upstream's fallback_repair_json would pad this absent score to
            # [0.0, 0.0]. The strict hook must fail instead.
            return "{'reasoning': 'model emitted no score'}"

    def repair_failure(_value):
        raise ValueError("cannot repair")

    monkeypatch.setattr(qwen, "Qwen3VL", FakeModel)
    monkeypatch.setattr(_upstream_utils.json_repair, "loads", repair_failure)
    scorer = EditScore("qwen3vl", num_pass=1, seed=42)
    with pytest.raises(ValueError, match="exhausted parsing retries"):
        scorer.evaluate([object(), object()], "edit instruction")
    # Two calls per attempt (SC then PQ), with the source seed reset each time.
    assert inference_calls == [42, 42, 42, 42, 42, 42]


def test_test_mode_and_cpu_gate_never_download(tmp_path, monkeypatch):
    import ayase.modules.edit_score as implementation

    def download(*_args, **_kwargs):
        pytest.fail("unexpected asset download")

    monkeypatch.setattr(implementation, "download_hf_snapshot", download)
    module = implementation.EditScoreModule(
        {"test_mode": True, "models_dir": str(tmp_path)}
    )
    module.setup()
    assert module._backend is None
    assert not module._ml_available

    monkeypatch.setattr(implementation.EditScoreModule, "_global_test_mode", False)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    module = implementation.EditScoreModule({"models_dir": str(tmp_path)})
    module.setup()
    assert module._backend == "unavailable"
    assert not module._ml_available


def test_process_uses_reference_edited_caption_and_publishes_all_scores(tmp_path):
    from ayase.modules.edit_score import EditScoreModule

    source_path, edited_path = tmp_path / "source.png", tmp_path / "edited.png"
    Image.new("RGB", (3, 2), "red").save(source_path)
    Image.new("RGB", (4, 5), "blue").save(edited_path)
    calls = []

    def evaluate(images, instruction):
        calls.append((images, instruction))
        return {
            "overall": math.sqrt(24),
            "prompt_following": 8.0,
            "consistency": 6.0,
            "perceptual_quality": 4.0,
            "SC_reasoning": "followed",
            "PQ_reasoning": "clean",
        }

    module = EditScoreModule()
    module._scorer = SimpleNamespace(evaluate=evaluate)
    module._backend = "test:pinned"
    module._ml_available = True
    sample = Sample(
        path=edited_path,
        reference_path=source_path,
        is_video=False,
        caption=CaptionMetadata(text="turn it blue", length=12),
    )
    assert module.process(sample) is sample
    images, instruction = calls[0]
    assert [image.size for image in images] == [(3, 2), (4, 5)]
    assert instruction == "turn it blue"
    assert sample.quality_metrics.edit_score_overall == pytest.approx(math.sqrt(24))
    assert sample.quality_metrics.edit_score_prompt_following == 8.0
    assert sample.quality_metrics.edit_score_consistency == 6.0
    assert sample.quality_metrics.edit_score_perceptual_quality == 4.0
    assert sample.metadata["edit_score"]["reasoning"] == {
        "SC": "followed",
        "PQ": "clean",
    }


@pytest.mark.parametrize("bad", [float("nan"), -0.01, 10.01])
def test_process_rejects_invalid_public_score(tmp_path, bad):
    from ayase.modules.edit_score import EditScoreModule

    source_path, edited_path = tmp_path / "source.png", tmp_path / "edited.png"
    Image.new("RGB", (2, 2)).save(source_path)
    Image.new("RGB", (2, 2)).save(edited_path)
    module = EditScoreModule()
    module._scorer = SimpleNamespace(
        evaluate=lambda *_args: {
            "overall": bad,
            "prompt_following": 5.0,
            "consistency": 5.0,
            "perceptual_quality": 5.0,
            "SC_reasoning": "x",
            "PQ_reasoning": "y",
        }
    )
    module._backend = "test"
    module._ml_available = True
    sample = Sample(path=edited_path, reference_path=source_path, is_video=False)
    edited_path.with_suffix(".txt").write_text("edit", encoding="utf-8")
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_invalid_utf8_sidecar_returns_sample_without_scoring(tmp_path):
    from ayase.modules.edit_score import EditScoreModule

    edited = tmp_path / "edited.png"
    edited.with_suffix(".txt").write_bytes(b"\xff")
    module = EditScoreModule()
    module._ml_available = True
    module._scorer = SimpleNamespace(evaluate=lambda *_args: pytest.fail("invalid caption scored"))
    sample = Sample(path=edited, reference_path=tmp_path / "source.png", is_video=False)
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_setup_routes_pinned_local_base_and_adapter_into_native_scorer(tmp_path, monkeypatch):
    import ayase.modules.edit_score as implementation
    import ayase.third_party.edit_score as vendor

    calls, loaded = [], {}

    def download(repo, models_dir, **kwargs):
        calls.append((repo, models_dir, kwargs))
        return tmp_path / ("adapter" if "EditScore/" in repo else "base")

    def construct(**kwargs):
        loaded.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(implementation.EditScoreModule, "_global_test_mode", False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(implementation, "download_hf_snapshot", download)
    monkeypatch.setattr(vendor, "EditScore", construct)
    module = implementation.EditScoreModule({"models_dir": str(tmp_path)})
    module.setup()
    assert module._ml_available
    assert [call[2]["revision"] for call in calls] == [
        "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "87dcfd0eba27335c036ef9616fa0e725141ba571",
    ]
    assert "*.safetensors" in calls[0][2]["allow_patterns"]
    assert "adapter_model.safetensors" in calls[1][2]["allow_patterns"]
    assert loaded["model_name_or_path"] == str(tmp_path / "base")
    assert loaded["lora_path"] == str(tmp_path / "adapter")
    assert loaded["backbone"] == "qwen3vl"
    assert loaded["num_pass"] == 1 and loaded["reduction"] == "average_last"
