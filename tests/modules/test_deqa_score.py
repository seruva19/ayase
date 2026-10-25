"""Focused contract tests for the pinned DeQA-Score Mix3 adapter."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ayase.models import Sample

from .conftest import _test_module_basics


def _complete_snapshot(root: Path) -> Path:
    from ayase.modules.deqa_score import SNAPSHOT_FILES

    for name in SNAPSHOT_FILES:
        (root / name).write_bytes(b"fixture")
    return root


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def numel(self):
        return 1

    def float(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *_shape):
        return self

    def __getitem__(self, _index):
        return self

    def item(self):
        return self.value


def test_deqa_score_basics_and_trace_metadata():
    from ayase.modules.deqa_score import DeQAScoreModule, MODEL_REVISION, SOURCE_REVISION

    _test_module_basics(DeQAScoreModule, "deqa_score")
    assert DeQAScoreModule.input_type == "image"
    assert DeQAScoreModule.provenance == "published"
    assert DeQAScoreModule.metric_groups == {"deqa_score": "nr_quality"}
    module = DeQAScoreModule()
    assert module.runtime_trace == {
        "source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "input": "image-only RGB",
        "precision": "float16",
        "attention": "eager",
    }


def test_test_mode_setup_never_provisions(tmp_path, monkeypatch):
    import ayase.modules.deqa_score as implementation

    monkeypatch.setattr(
        implementation,
        "download_hf_snapshot",
        lambda *_args, **_kwargs: pytest.fail("test mode attempted provisioning"),
    )
    module = implementation.DeQAScoreModule(
        {"test_mode": True, "models_dir": str(tmp_path)}
    )
    module.setup()
    assert module._model is None
    assert module._backend is None
    assert not module._ml_available


def test_setup_downloads_exact_snapshot_then_loads_offline(tmp_path, monkeypatch):
    import ayase.modules.deqa_score as implementation

    monkeypatch.setattr(implementation.DeQAScoreModule, "_global_test_mode", False)
    snapshot = _complete_snapshot(tmp_path)
    calls = []

    class FakeModel:
        config = SimpleNamespace(level_ids=list(implementation.LEVEL_IDS))
        preferential_ids_ = list(implementation.LEVEL_IDS)

        def eval(self):
            calls.append(("eval",))
            return self

    def download(*args, **kwargs):
        calls.append(("download", args, kwargs))
        return snapshot

    def load(path, **kwargs):
        calls.append(("load", path, kwargs, os.environ.get("HF_HUB_OFFLINE")))
        return FakeModel()

    def load_config(path, **kwargs):
        calls.append(
            (
                "config",
                path,
                kwargs,
                os.environ.get("HF_HUB_OFFLINE"),
                os.environ.get("TRANSFORMERS_OFFLINE"),
            )
        )
        return SimpleNamespace(_name_or_path="remote-model-id")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float16="float16-sentinel",
    )
    fake_transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=load_config),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load)
    )
    monkeypatch.setattr(implementation, "download_hf_snapshot", download)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "previous-value")

    module = implementation.DeQAScoreModule(
        {"models_dir": str(tmp_path / "models"), "device": "cuda:1"}
    )
    module.setup()

    download_call = calls[0]
    config_call = calls[1]
    load_call = calls[2]
    assert download_call[1] == (implementation.MODEL_ID, str(tmp_path / "models"))
    assert download_call[2]["revision"] == implementation.MODEL_REVISION
    assert download_call[2]["allow_patterns"] == list(implementation.SNAPSHOT_FILES)
    assert download_call[2]["ignore_patterns"] == ["*.bin", "trainer_state.json"]
    assert config_call[1] == str(snapshot)
    assert config_call[2] == {"trust_remote_code": True, "local_files_only": True}
    assert config_call[3:] == ("1", "1")
    assert load_call[1] == str(snapshot)
    model_config = load_call[2].pop("config")
    assert model_config._name_or_path == str(snapshot)
    assert load_call[2] == {
        "trust_remote_code": True,
        "local_files_only": True,
        "use_safetensors": True,
        "attn_implementation": "eager",
        "torch_dtype": "float16-sentinel",
        "device_map": {"": "cuda:1"},
    }
    assert load_call[3] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ
    assert os.environ["TRANSFORMERS_OFFLINE"] == "previous-value"
    assert module._ml_available
    assert module._backend.endswith(":eager-float16-image")


def test_explicit_snapshot_is_local_and_requires_all_four_shards(tmp_path, monkeypatch):
    import ayase.modules.deqa_score as implementation

    _complete_snapshot(tmp_path)
    (tmp_path / "model-00004-of-00004.safetensors").unlink()
    monkeypatch.setattr(
        implementation,
        "download_hf_snapshot",
        lambda *_args, **_kwargs: pytest.fail("explicit model_path attempted a download"),
    )
    module = implementation.DeQAScoreModule({"model_path": str(tmp_path)})
    with pytest.raises(FileNotFoundError, match="model-00004-of-00004.safetensors"):
        module._resolve_snapshot()


def test_cpu_gate_precedes_snapshot_provisioning(tmp_path, monkeypatch):
    import ayase.modules.deqa_score as implementation

    monkeypatch.setattr(implementation.DeQAScoreModule, "_global_test_mode", False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), float16=object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=object(), AutoModelForCausalLM=object()),
    )
    monkeypatch.setattr(
        implementation,
        "download_hf_snapshot",
        lambda *_args, **_kwargs: pytest.fail("CPU host attempted a 16 GB download"),
    )
    module = implementation.DeQAScoreModule({"models_dir": str(tmp_path)})
    module.setup()
    assert not module._ml_available
    assert module._backend == "unavailable"


def test_process_uses_direct_image_score_rgb_and_native_arguments(tmp_path):
    from ayase.modules.deqa_score import DeQAScoreModule

    image_path = tmp_path / "rgba.png"
    Image.new("RGBA", (7, 4), (10, 20, 30, 100)).save(image_path)
    calls = []

    class FakeModel:
        device = "cuda:1"

        def score(self, images, **kwargs):
            calls.append((images, kwargs))
            return _ScalarResult(2.85)

    module = DeQAScoreModule()
    module._model = FakeModel()
    module._ml_available = True
    module._backend = "test:pinned-source"
    sample = Sample(path=image_path, is_video=False)

    result = module.process(sample)

    assert result is sample
    assert len(calls) == 1
    assert calls[0][0][0].mode == "RGB"
    assert calls[0][0][0].size == (7, 4)
    assert calls[0][1] == {"task_": "quality", "input_": "image"}
    assert result.quality_metrics.deqa_score == pytest.approx(2.85)
    assert result.quality_metrics.metric_backends["deqa_score"] == "test:pinned-source"
    assert result.metadata["deqa_score"] == {
        "source_revision": module.runtime_trace["source_revision"],
        "model_revision": module.runtime_trace["model_revision"],
        "input": "image-only RGB",
        "precision": "float16",
        "attention": "eager",
        "device": "cuda:1",
        "requested_device": "auto",
        "dtype": "float16",
        "backend": "test:pinned-source",
        "expert_override": False,
    }


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 0.999, 5.001])
def test_process_rejects_nonfinite_or_out_of_range_without_clamping(
    tmp_path, invalid
):
    from ayase.modules.deqa_score import DeQAScoreModule

    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image_path)
    module = DeQAScoreModule()
    module._model = SimpleNamespace(
        score=lambda *_args, **_kwargs: _ScalarResult(invalid)
    )
    module._ml_available = True
    sample = Sample(path=image_path, is_video=False)

    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_video_and_unavailable_backend_are_no_ops(tmp_path):
    from ayase.modules.deqa_score import DeQAScoreModule

    model = SimpleNamespace(
        score=lambda *_args, **_kwargs: pytest.fail("unsupported input was scored")
    )
    module = DeQAScoreModule()
    module._model = model
    module._ml_available = True
    video = Sample(path=tmp_path / "video.mp4", is_video=True)
    assert module.process(video) is video
    assert video.quality_metrics is None

    unavailable = DeQAScoreModule()
    image = Sample(path=tmp_path / "missing.png", is_video=False)
    assert unavailable.process(image) is image
    assert image.quality_metrics is None


def test_process_rejects_multiple_scores_and_dispose_releases_cuda(tmp_path, monkeypatch):
    import ayase.modules.deqa_score as implementation

    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image_path)

    class MultipleResults(_ScalarResult):
        def numel(self):
            return 2

    module = implementation.DeQAScoreModule()
    module._model = SimpleNamespace(
        score=lambda *_args, **_kwargs: MultipleResults(3.0)
    )
    module._ml_available = True
    module._backend = "test"
    sample = Sample(path=image_path, is_video=False)
    assert module.process(sample) is sample
    assert sample.quality_metrics is None

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: calls.append("empty_cache"),
            )
        ),
    )
    module.on_dispose()
    assert calls == ["empty_cache"]
    assert module._model is None
    assert module._backend is None
    assert not module._ml_available
