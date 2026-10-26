"""Focused tests for the approximate MIEScore hosted-VLM adapter."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (3, 2), color).save(path)
    return path


def _sample(path: Path, instruction: str = "Put the red object behind the blue object"):
    from ayase.models import CaptionMetadata, Sample

    return Sample(
        path=path,
        is_video=False,
        caption=CaptionMetadata(text=instruction, length=len(instruction)),
    )


class _Completions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _Client:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=_Completions(content))


@pytest.fixture(autouse=True)
def _disable_global_test_mode(monkeypatch):
    from ayase.modules.miescore import MIEScoreModule

    monkeypatch.setattr(MIEScoreModule, "_global_test_mode", False)


def test_metadata_marks_approximation_without_aggregate():
    from ayase.modules.miescore import MIEScoreModule

    assert MIEScoreModule.name == "miescore"
    assert MIEScoreModule.input_type == "img"
    assert MIEScoreModule.implementation_status == "approximate"
    assert "trained LoRA regression head" in MIEScoreModule.implementation_notes
    assert set(MIEScoreModule.metric_info) == {
        "miescore_approx_visual_quality",
        "miescore_approx_instruction_following",
        "miescore_approx_attribute_preservation",
    }
    assert not any("overall" in name for name in MIEScoreModule.metric_info)
    assert MIEScoreModule.models[0]["auto_download"] is False


def test_setup_uses_lazy_openai_sdk_without_api_call(monkeypatch):
    from ayase.modules.miescore import MIEScoreModule

    constructed = []

    class OpenAI:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    module = MIEScoreModule(
        {"api_key": "secret", "base_url": "https://judge.invalid/v1", "timeout": 9}
    )
    module.setup()

    assert constructed == [
        {"api_key": "secret", "base_url": "https://judge.invalid/v1", "timeout": 9.0}
    ]
    assert module._backend == "openai_compatible"


def test_all_source_images_keep_order_and_caption_instruction(tmp_path):
    from ayase.modules.miescore import MIEScoreModule

    source_a = _image(tmp_path / "a.png", (255, 0, 0))
    source_b = _image(tmp_path / "b.png", (0, 0, 255))
    target = _image(tmp_path / "target.png", (128, 0, 128))
    sample = _sample(target, "caption instruction wins")
    sample.metadata["source_paths"] = [str(source_a), str(source_b)]
    client = _Client(
        json.dumps(
            {
                "visual_quality": 81,
                "instruction_following": 72.5,
                "attribute_preservation": 63,
            }
        )
    )
    module = MIEScoreModule({"instruction": "fallback only"})
    module._client = client
    module._backend = "openai_compatible"

    assert module.process(sample) is sample

    call = client.chat.completions.calls[0]
    content = call["messages"][0]["content"]
    assert [part["text"] for part in content if part["type"] == "text"][1:] == [
        "SOURCE IMAGE 1",
        "SOURCE IMAGE 2",
        "EDITED TARGET IMAGE",
    ]
    assert "caption instruction wins" in content[0]["text"]
    image_urls = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    assert len(image_urls) == 3
    assert len(set(image_urls)) == 3
    assert sample.quality_metrics.miescore_approx_visual_quality == 81.0
    assert sample.quality_metrics.miescore_approx_instruction_following == 72.5
    assert sample.quality_metrics.miescore_approx_attribute_preservation == 63.0
    assert sample.metadata["miescore"]["source_paths"] == [str(source_a), str(source_b)]
    assert not hasattr(sample.quality_metrics, "miescore_overall")


def test_config_sources_and_instruction_are_supported(tmp_path):
    from ayase.modules.miescore import MIEScoreModule

    configured = _image(tmp_path / "configured.png", (1, 2, 3))
    ignored = _image(tmp_path / "ignored.png", (4, 5, 6))
    target = _image(tmp_path / "target.png", (7, 8, 9))
    sample = _sample(target, "")
    sample.metadata["source_paths"] = [str(ignored)]
    client = _Client(
        '{"visual_quality": 1, "instruction_following": 2, '
        '"attribute_preservation": 3}'
    )
    module = MIEScoreModule(
        {"source_paths": [configured], "instruction": "configured instruction"}
    )
    module._client = client
    module._backend = "openai_compatible"

    module.process(sample)

    content = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "configured instruction" in content[0]["text"]
    assert sample.metadata["miescore"]["source_paths"] == [str(configured)]


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "{}",
        '{"visual_quality": true, "instruction_following": 2, "attribute_preservation": 3}',
        '{"visual_quality": "1", "instruction_following": 2, "attribute_preservation": 3}',
        '{"visual_quality": -1, "instruction_following": 2, "attribute_preservation": 3}',
        '{"visual_quality": 1, "instruction_following": 101, "attribute_preservation": 3}',
        '{"visual_quality": 1e999, "instruction_following": 2, "attribute_preservation": 3}',
        '{"visual_quality": 1, "instruction_following": 2, "attribute_preservation": 3, "overall": 2}',
    ],
)
def test_bad_responses_do_not_write_metrics(tmp_path, content):
    from ayase.modules.miescore import MIEScoreModule

    source = _image(tmp_path / "source.png", (1, 1, 1))
    target = _image(tmp_path / "target.png", (2, 2, 2))
    sample = _sample(target)
    sample.metadata["source_paths"] = [str(source)]
    module = MIEScoreModule()
    module._client = _Client(content)

    assert module.process(sample) is sample
    assert sample.quality_metrics is None
    assert "miescore" not in sample.metadata


@pytest.mark.parametrize("case", ["no_sources", "missing_source", "no_instruction", "video"])
def test_invalid_inputs_never_call_api(tmp_path, case):
    from ayase.modules.miescore import MIEScoreModule

    target = _image(tmp_path / "target.png", (2, 2, 2))
    sample = _sample(target, "" if case == "no_instruction" else "edit it")
    if case == "missing_source":
        sample.metadata["source_paths"] = [str(tmp_path / "missing.png")]
    elif case not in {"no_sources", "no_instruction"}:
        sample.metadata["source_paths"] = [str(target)]
    if case == "video":
        sample.is_video = True
    client = _Client("{}")
    module = MIEScoreModule()
    module._client = client

    assert module.process(sample) is sample
    assert client.chat.completions.calls == []
    assert sample.quality_metrics is None


def test_test_mode_skips_sdk_and_preexisting_transport(monkeypatch, tmp_path):
    from ayase.modules.miescore import MIEScoreModule

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=lambda **kwargs: pytest.fail("SDK construction")),
    )
    target = _image(tmp_path / "target.png", (2, 2, 2))
    sample = _sample(target)
    sample.metadata["source_paths"] = [str(target)]
    client = _Client("{}")
    module = MIEScoreModule({"test_mode": True})
    module._client = client

    module.setup()
    assert module.process(sample) is sample
    assert client.chat.completions.calls == []


def _stale_metrics():
    from ayase.models import QualityMetrics

    return QualityMetrics(
        miescore_approx_visual_quality=91.0,
        miescore_approx_instruction_following=92.0,
        miescore_approx_attribute_preservation=93.0,
        vmaf=44.0,
        metric_provenance={
            "miescore_approx_visual_quality": "adapted",
            "miescore_approx_instruction_following": "adapted",
            "miescore_approx_attribute_preservation": "adapted",
            "vmaf": "published",
        },
        metric_backends={"miescore": "openai_compatible", "vmaf": "libvmaf"},
    )


def _assert_stale_miescore_cleared(sample):
    metrics = sample.quality_metrics
    assert metrics.miescore_approx_visual_quality is None
    assert metrics.miescore_approx_instruction_following is None
    assert metrics.miescore_approx_attribute_preservation is None
    assert metrics.vmaf == 44.0
    assert metrics.metric_provenance == {"vmaf": "published"}
    assert metrics.metric_backends == {"vmaf": "libvmaf"}
    assert "miescore" not in sample.metadata


def test_fresh_success_then_failed_request_clears_owned_state_only(tmp_path):
    from ayase.models import QualityMetrics
    from ayase.modules.miescore import MIEScoreModule

    source = _image(tmp_path / "source.png", (1, 1, 1))
    target = _image(tmp_path / "target.png", (2, 2, 2))
    sample = _sample(target)
    sample.metadata["source_paths"] = [str(source)]
    sample.quality_metrics = QualityMetrics(
        vmaf=44.0,
        metric_provenance={"vmaf": "published"},
        metric_backends={"vmaf": "libvmaf"},
    )
    module = MIEScoreModule()
    module._client = _Client(
        '{"visual_quality": 91, "instruction_following": 92, '
        '"attribute_preservation": 93}'
    )
    module._backend = "openai_compatible"

    assert module.process(sample) is sample
    assert sample.quality_metrics.miescore_approx_visual_quality == 91.0
    assert sample.metadata["miescore"]["implementation_status"] == "approximate"

    sample.quality_metrics.metric_provenance.update(
        {field: "adapted" for field in MIEScoreModule.metric_info}
    )
    sample.quality_metrics.metric_backends["miescore"] = "openai_compatible"
    module._client = _Client("invalid response")

    assert module.process(sample) is sample
    assert len(module._client.chat.completions.calls) == 1
    _assert_stale_miescore_cleared(sample)


def test_invalid_next_input_clears_prior_scores_without_api_call(tmp_path):
    from ayase.modules.miescore import MIEScoreModule

    target = _image(tmp_path / "target.png", (2, 2, 2))
    sample = _sample(target)
    sample.metadata["miescore"] = {"implementation_status": "approximate"}
    sample.quality_metrics = _stale_metrics()
    client = _Client("{}")
    module = MIEScoreModule()
    module._client = client

    assert module.process(sample) is sample
    assert client.chat.completions.calls == []
    _assert_stale_miescore_cleared(sample)


def test_real_pipeline_stamps_provenance_backend_status_and_serializes(monkeypatch, tmp_path):
    from ayase.models import QualityMetrics
    from ayase.modules.miescore import MIEScoreModule
    from ayase.pipeline import Pipeline

    client = _Client(
        '{"visual_quality": 80, "instruction_following": 70, '
        '"attribute_preservation": 60}'
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=lambda **kwargs: client),
    )
    source = _image(tmp_path / "source.png", (1, 1, 1))
    target = _image(tmp_path / "target.png", (2, 2, 2))
    sample = _sample(target)
    sample.metadata["source_paths"] = [str(source)]
    sample.quality_metrics = QualityMetrics(
        vmaf=55.0,
        metric_provenance={"vmaf": "published"},
        metric_backends={"vmaf": "libvmaf"},
    )
    pipeline = Pipeline(
        [MIEScoreModule({"api_key": "fake", "allow_provenance": ["adapted"]})]
    )

    pipeline.start()
    try:
        result = pipeline.process_sample(sample)
    finally:
        pipeline.stop()

    fields = set(MIEScoreModule.metric_info)
    assert result.quality_metrics.metric_provenance == {
        "vmaf": "published",
        **{field: "adapted" for field in fields},
    }
    assert result.quality_metrics.metric_backends == {
        "vmaf": "libvmaf",
        "miescore": "openai_compatible",
    }
    assert result.metadata["miescore"]["implementation_status"] == "approximate"
    dumped = json.loads(result.model_dump_json())
    assert dumped["quality_metrics"]["miescore_approx_visual_quality"] == 80.0
    assert dumped["quality_metrics"]["miescore_approx_instruction_following"] == 70.0
    assert dumped["quality_metrics"]["miescore_approx_attribute_preservation"] == 60.0
    assert dumped["metadata"]["miescore"]["implementation_status"] == "approximate"
    assert len(client.chat.completions.calls) == 1
