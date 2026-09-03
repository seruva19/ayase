"""Contract tests for native MJ-Video inference."""

import sys
from types import ModuleType, SimpleNamespace

from ..conftest import _test_module_basics


def test_mj_video_basics():
    from ayase.modules.mj_video import MJVideoModule

    _test_module_basics(MJVideoModule, "mj_video")


def test_mj_video_unmounted_process_is_noop(video_sample):
    from ayase.modules.mj_video import MJVideoModule

    assert MJVideoModule().process(video_sample) is video_sample


def test_mj_video_uses_native_autodownload_defaults():
    from ayase.modules.mj_video import MJVideoModule

    assert MJVideoModule.default_config["model_name"] == "MJ-Bench/MJ-VIDEO-2B"
    assert MJVideoModule.default_config["models_dir"] == "models"
    assert "results_path" not in MJVideoModule.default_config
    assert "runner_command" not in MJVideoModule.default_config
    # The reward architecture is vendored, not downloaded: no source URL exists.
    assert "source_url" not in MJVideoModule.default_config
    assert MJVideoModule._vendored_source().is_dir()
    assert MJVideoModule.default_config["tokenizer_base_url"] == (
        "https://huggingface.co/internlm/internlm2-chat-1_8b/resolve"
    )
    assert MJVideoModule.models[0]["auto_download"] is True


def test_mj_video_setup_downloads_reference_checkpoint(monkeypatch, tmp_path):
    import torch

    import ayase.config
    from ayase.modules.mj_video import MJVideoModule

    captured = {}
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)

    def fake_snapshot(repo_id, models_dir, **kwargs):
        captured["repo_id"] = repo_id
        captured["checkpoint"] = checkpoint
        return checkpoint

    monkeypatch.setattr(ayase.config, "download_hf_snapshot", fake_snapshot)
    monkeypatch.setattr(
        ayase.config, "download_model_file", lambda *args, **kwargs: tmp_path / "source.zip"
    )
    monkeypatch.setattr(MJVideoModule, "_vendored_source", staticmethod(lambda: source))
    monkeypatch.setattr(MJVideoModule, "_prepare_tokenizer_files", lambda *args: None)
    monkeypatch.setattr(
        MJVideoModule, "_ensure_transformers_doc_compatibility", staticmethod(lambda: None)
    )
    monkeypatch.setattr(
        MJVideoModule, "_configure_internvl_single_process", staticmethod(lambda: None)
    )

    class FakeTokenizer:
        pad_token_id = 0

        @staticmethod
        def convert_tokens_to_ids(value):
            return 1

    class FakeConfig:
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured["config_path"] = path
            return cls()

    class FakeModel:
        def __init__(self, name, config):
            self.config = config
            self.model = SimpleNamespace(img_context_token_id=None)

        def load_state_dict(self, weights, strict):
            captured["strict"] = strict

        def to(self, **kwargs):
            return self

        def eval(self):
            return self

    data_processor = ModuleType("data_processor")
    data_processor.load_video = lambda *args, **kwargs: None
    model_module = ModuleType("model")
    model_module.InternVLChatRewardModeling = FakeModel
    model_module.InternVLChatRewardModelingConfig = FakeConfig
    model_module.prepare_chat_input = lambda *args, **kwargs: None
    safetensors_module = ModuleType("safetensors")
    safetensors_torch = ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda *args, **kwargs: {}
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer()
    )
    internvl2_module = ModuleType("internvl2")
    internvl2_config = ModuleType("internvl2.configuration_internvl_chat")
    internvl2_config.InternVLChatConfig = FakeConfig
    monkeypatch.setitem(sys.modules, "data_processor", data_processor)
    monkeypatch.setitem(sys.modules, "model", model_module)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_module)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "internvl2", internvl2_module)
    monkeypatch.setitem(sys.modules, "internvl2.configuration_internvl_chat", internvl2_config)

    module = MJVideoModule({"models_dir": tmp_path, "device": "cpu"})
    module.setup()

    assert captured["checkpoint"] == checkpoint
    assert captured["repo_id"] == "MJ-Bench/MJ-VIDEO-2B"
    assert captured["config_path"] == str(checkpoint)
    assert captured["strict"] is True
    assert module._dtype == torch.float32
    assert module._backend == "mj_video"


def test_mj_video_attaches_reference_output_shape(video_sample):
    from ayase.modules.mj_video import MJVideoModule

    output = SimpleNamespace(
        score=[0.75],
        aspect_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        rewards=[float(index) for index in range(28)],
    )
    MJVideoModule()._attach_output(video_sample, output)

    assert video_sample.quality_metrics.mj_video_overall_score == 0.75
    assert video_sample.quality_metrics.mj_video_coherence_score == 0.4
    assert len(video_sample.metadata["mj_video_criteria_scores"]) == 28


def test_mj_video_rejects_wrong_output_shape(video_sample):
    import pytest

    from ayase.modules.mj_video import MJVideoModule

    with pytest.raises(ValueError, match="unexpected reward shape"):
        MJVideoModule()._attach_output(
            video_sample,
            SimpleNamespace(score=[1.0], aspect_scores=[1.0, 2.0], rewards=list(range(28))),
        )


def test_mj_video_metadata_exposes_fields():
    from ayase.modules.mj_video import MJVideoModule

    metadata = MJVideoModule.get_metadata()
    assert "mj_video_overall_score" in metadata["output_fields"]
    assert "mj_video_fairness_score" in metadata["output_fields"]


def test_mj_video_downloads_no_source_code():
    """The reward architecture is vendored; only weights may be fetched."""
    from ayase.modules.mj_video import MJVideoModule

    for entry in MJVideoModule.models:
        assert not str(entry.get("url", "")).endswith(".zip"), entry["id"]
