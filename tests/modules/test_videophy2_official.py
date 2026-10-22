"""CPU-light contract tests for the official VideoPhy-2 AutoEval adapter."""

def test_native_rating_parser_accepts_only_rating_tokens():
    from ayase.modules.videophy2 import parse_native_rating

    assert parse_native_rating("5") == 5
    assert parse_native_rating("The rating is four.") == 4
    assert parse_native_rating("rating: 3/5") == 3
    assert parse_native_rating("version 2025") is None
    assert parse_native_rating("unparseable") is None


def test_official_joint_threshold_protocol():
    from ayase.modules.videophy2 import joint_pass

    assert joint_pass(4, 4) == 1.0
    assert joint_pass(5, 4) == 1.0
    assert joint_pass(3, 5) == 0.0
    assert joint_pass(None, 5) is None


def test_test_mode_does_not_load_backend():
    from ayase.modules.videophy2 import VideoPhy2AutoEvalModule

    module = VideoPhy2AutoEvalModule({"test_mode": True, "models_dir": "G:/missing"})
    module.setup()
    assert module.active_backend == "unavailable"
    assert module._backend is None


def test_default_uses_namespaced_vendored_runtime():
    from ayase.modules.videophy2 import VideoPhy2AutoEvalModule

    module = VideoPhy2AutoEvalModule({"test_mode": True})
    modeling, processing = module._official_modules()
    assert modeling.__name__.startswith("ayase.third_party.videophy2.")
    assert processing.__name__.startswith("ayase.third_party.videophy2.")


def test_checkpoint_download_is_pinned_and_uses_models_dir(monkeypatch, tmp_path):
    import ayase.modules.videophy2 as videophy2

    calls = {}

    def fake_download(repo_id, models_dir, revision):
        calls.update(repo_id=repo_id, models_dir=models_dir, revision=revision)
        return tmp_path / "snapshot"

    monkeypatch.setattr(videophy2, "resolve_model_path", lambda model, models_dir: model)
    monkeypatch.setattr(videophy2, "download_hf_snapshot", fake_download)
    module = videophy2.VideoPhy2AutoEvalModule({"models_dir": str(tmp_path)})

    assert module._resolve_checkpoint() == str(tmp_path / "snapshot")
    assert calls == {
        "repo_id": videophy2.MODEL_ID,
        "models_dir": str(tmp_path),
        "revision": videophy2.MODEL_REVISION,
    }


def test_checkpoint_auto_download_can_be_disabled(monkeypatch, tmp_path):
    import pytest
    import ayase.modules.videophy2 as videophy2

    monkeypatch.setattr(videophy2, "resolve_model_path", lambda model, models_dir: model)
    module = videophy2.VideoPhy2AutoEvalModule(
        {"models_dir": str(tmp_path), "auto_download": False}
    )
    with pytest.raises(FileNotFoundError):
        module._resolve_checkpoint()


def test_published_protocol_rejects_frame_override(monkeypatch):
    from ayase.modules.videophy2 import VideoPhy2AutoEvalModule

    module = VideoPhy2AutoEvalModule(
        {"num_frames": 16, "official_source_dir": "unused", "verify_source_revision": False}
    )
    try:
        module._load_official_backend()
    except ValueError as exc:
        assert "num_frames=32" in str(exc)
    else:
        raise AssertionError("non-official frame count was accepted")


def test_no_backend_leaves_sample_unchanged(video_sample):
    from ayase.modules.videophy2 import VideoPhy2AutoEvalModule

    module = VideoPhy2AutoEvalModule({"test_mode": True})
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_mocked_official_inference_preserves_native_scale(video_sample):
    from ayase.modules.videophy2 import VideoPhy2AutoEvalModule
    from ayase.models import CaptionMetadata

    caption = "A ball bounces down stairs"
    video_sample.caption = CaptionMetadata(text=caption, length=len(caption))
    module = VideoPhy2AutoEvalModule({"test_mode": True})
    module.active_backend = "official"
    module._backend = "official"
    responses = iter((5, 4))
    module._infer = lambda path, prompt: next(responses)

    result = module.process(video_sample)
    assert result.quality_metrics.videophy2_pc_rating == 5.0
    assert result.quality_metrics.videophy2_sa_rating == 4.0
    assert result.quality_metrics.videophy2_joint_pass == 1.0


def test_missing_caption_scores_pc_only(video_sample):
    from ayase.modules.videophy2 import VideoPhy2AutoEvalModule

    module = VideoPhy2AutoEvalModule({"test_mode": True})
    module.active_backend = "official"
    module._backend = "official"
    module._infer = lambda path, prompt: 4

    result = module.process(video_sample)
    assert result.quality_metrics.videophy2_pc_rating == 4.0
    assert result.quality_metrics.videophy2_sa_rating is None
    assert result.quality_metrics.videophy2_joint_pass is None


def test_infer_uses_official_processor_and_generation_contract(video_sample):
    import torch

    from ayase.modules.videophy2 import VideoPhy2AutoEvalModule

    calls = {}

    class Processor:
        def __call__(self, **kwargs):
            calls["processor"] = kwargs
            return {"input_ids": torch.tensor([[1]])}

    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return torch.tensor([[5]])

    class Tokenizer:
        def decode(self, tokens, skip_special_tokens):
            calls["decode"] = (tokens, skip_special_tokens)
            return "five"

    module = VideoPhy2AutoEvalModule({"test_mode": True})
    module._processor = Processor()
    module._model = Model()
    module._tokenizer = Tokenizer()

    assert module._infer(video_sample.path, "official prompt") == 5
    assert calls["processor"] == {
        "text": ["official prompt"],
        "videos": [str(video_sample.path)],
        "num_frames": 32,
        "return_tensors": "pt",
    }
    assert calls["generate"]["do_sample"] is False
    assert calls["generate"]["top_k"] == 1
    assert calls["generate"]["temperature"] == 0.001
    assert calls["generate"]["max_length"] == 256
