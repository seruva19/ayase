"""Focused tests for the approximate Long-CODE adapter."""

import json
import sys
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _long_code():
    import ayase.modules.long_code as module

    return module


@pytest.fixture(autouse=True)
def _allow_injected_backends(monkeypatch):
    from ayase.pipeline import PipelineModule

    previous = PipelineModule._global_test_mode
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    PipelineModule.set_test_mode(False)
    try:
        yield
    finally:
        PipelineModule.set_test_mode(previous)


def _sample(path: Path, prompts=None):
    from ayase.models import Sample

    path.write_bytes(b"video-placeholder")
    return Sample(
        path=path,
        is_video=True,
        metadata={"shot_prompts": prompts or ["arrival", "discovery", "resolution"]},
    )


class _Tokenizer:
    model_max_length = 77

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()) + 2))}


class _Processor:
    def __init__(self):
        self.calls = []
        self.tokenizer = _Tokenizer()
        self.image_processor = self._images

    @staticmethod
    def _images(images, **kwargs):
        import torch

        return {"pixel_values": torch.ones(1, 8, 3, 2, 2)}

    def __call__(self, **kwargs):
        import torch

        self.calls.append(kwargs)
        return {"input_ids": torch.ones(1, 1)}


class _Encoder:
    config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=77))

    def get_text_features(self, **kwargs):
        import torch

        return torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    def get_video_features(self, **kwargs):
        import torch

        return torch.tensor([[0.0, 1.0]])

    def __call__(self, **kwargs):
        return SimpleNamespace(video_embeds=self.get_video_features())


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _Client:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_Completions(responses))


def _ready_module(monkeypatch, config=None):
    module = _long_code().LongCodeModule(config)
    module._processor = _Processor()
    module._encoder = _Encoder()
    module._encoder_backend = "fake-order-sensitive-video"
    frames = [np.zeros((3, 4, 3), dtype=np.uint8) for _ in range(8)]
    timestamps = [float(index) for index in range(8)]
    monkeypatch.setattr(module, "_read_frames", lambda path, count: (frames, timestamps))
    return module


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 2, 3, 4], [1, 2, 3, 4], 1.0),
        ([1, 2, 3, 4], [4, 3, 2, 1], -1.0),
        ([1, 1, 2], [3, 3, 1], -1.0),
    ],
)
def test_spearman_order_ties_and_shuffle(left, right, expected):
    assert _long_code()._spearman(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1, 1, 1], [1, 2, 3]),
        ([1, 2, 3], [4, 4, 4]),
        ([1, float("nan"), 3], [1, 2, 3]),
        ([1, float("inf"), 3], [1, 2, 3]),
        ([1, 2], [1]),
    ],
)
def test_spearman_rejects_degenerate_nonfinite_and_mismatched(left, right):
    assert _long_code()._spearman(left, right) is None


def test_metadata_declares_approximation_and_unreleased_reference_bank():
    cls = _long_code().LongCodeModule
    assert cls.name == "long_code"
    assert cls.input_type == "video"
    assert cls.implementation_status == "approximate"
    assert "reference bank" in cls.implementation_notes
    assert cls.default_config["alpha"] == 0.5
    assert cls.default_config["judge_model"] is None
    assert cls.models[0]["id"] == "microsoft/xclip-base-patch32"
    assert cls.models[1]["auto_download"] is False
    assert set(cls.metric_info) == {
        "long_code_approx_dsa",
        "long_code_approx_narrative_score",
        "long_code_approx_causality_score",
        "long_code_approx_identity_score",
        "long_code_approx_semantic_score",
        "long_code_approx_score",
    }


def test_dsa_process_works_without_semantic_credentials(monkeypatch, tmp_path):
    sample = _sample(tmp_path / "clip.mp4")
    module = _ready_module(monkeypatch)

    assert module.process(sample) is sample
    assert sample.quality_metrics.long_code_approx_dsa == pytest.approx(1.0)
    assert sample.quality_metrics.long_code_approx_semantic_score is None
    assert sample.quality_metrics.long_code_approx_score is None
    assert sample.metadata["long_code"]["availability"] == "dsa_only"
    assert sample.metadata["long_code"]["reference_bank"] == "omitted-unreleased"
    assert sample.metadata["long_code"]["official_parity"] is False
    assert sample.metadata["long_code"]["sampled_frame_count"] == 8
    assert sample.metadata["long_code"]["text_truncation"]["all_shots_represented"] is True


def test_global_story_mean_includes_long_final_shot(monkeypatch, tmp_path):
    prompts = ["opening", "middle", "final " + "word " * 100]
    sample = _sample(tmp_path / "clip.mp4", prompts)
    module = _ready_module(monkeypatch)

    assert module.process(sample) is sample
    text_call = module._processor.calls[0]
    assert text_call["text"] == [prompt.strip() for prompt in prompts]
    assert text_call["truncation"] is True
    assert text_call["max_length"] == 77
    truncation = sample.metadata["long_code"]["text_truncation"]
    assert truncation["shot_prompts_truncated"] == [3]
    assert truncation["global_story_strategy"] == (
        "normalized_mean_of_all_per_shot_text_embeddings"
    )
    assert truncation["all_shots_represented"] is True


def test_truncation_detection_accepts_batchencoding_style_mapping():
    class BatchEncodingLikeTokenizer:
        model_max_length = 77

        def __call__(self, text, **kwargs):
            size = 90 if text.startswith("long") else 4
            return UserDict({"input_ids": list(range(size))})

    module = _long_code().LongCodeModule()
    module._processor = SimpleNamespace(tokenizer=BatchEncodingLikeTokenizer())

    result = module._text_truncation(["short shot", "long final shot"], 77)

    assert result["shot_prompts_truncated"] == [2]
    assert result["length_detection_available"] is True
    assert result["all_shots_represented"] is True


def test_real_attempt_clears_stale_owned_metrics_lineage_and_success_metadata(
    monkeypatch, tmp_path
):
    sample = _sample(tmp_path / "clip.mp4")
    from ayase.models import QualityMetrics

    fields = _long_code().OUTPUT_FIELDS
    sample.quality_metrics = QualityMetrics(
        blur_score=55.0,
        long_code_approx_dsa=0.25,
        long_code_approx_narrative_score=99.0,
        long_code_approx_causality_score=98.0,
        long_code_approx_identity_score=97.0,
        long_code_approx_semantic_score=98.0,
        long_code_approx_score=75.0,
        metric_provenance={**{field: "old" for field in fields}, "blur_score": "published"},
        metric_backends={"long_code": "old-backend", "blur": "laplacian"},
    )
    sample.metadata["long_code"] = {"availability": "full_approximate"}
    module = _ready_module(monkeypatch, {"judge_model": "fixture-vision"})
    module._client = _Client(["malformed"])

    assert module.process(sample) is sample
    assert sample.quality_metrics.long_code_approx_dsa == 1.0
    for field in fields[1:]:
        assert getattr(sample.quality_metrics, field) is None
    assert sample.quality_metrics.blur_score == 55.0
    assert sample.quality_metrics.metric_provenance == {"blur_score": "published"}
    assert sample.quality_metrics.metric_backends == {"blur": "laplacian"}
    assert sample.metadata["long_code"]["availability"] == "semantic_failed"


def test_semantic_transport_exception_preserves_fresh_dsa_and_failure_metadata(
    monkeypatch, tmp_path
):
    sample = _sample(tmp_path / "clip.mp4")
    module = _ready_module(monkeypatch, {"judge_model": "fixture-vision"})

    class ThrowingCompletions:
        def create(self, **kwargs):
            raise TimeoutError("fixture timeout")

    module._client = SimpleNamespace(
        chat=SimpleNamespace(completions=ThrowingCompletions())
    )
    module._semantic_backend = "mock-three-stage"

    assert module.process(sample) is sample
    assert sample.quality_metrics.long_code_approx_dsa == 1.0
    assert sample.quality_metrics.long_code_approx_score is None
    assert sample.metadata["long_code"]["availability"] == "semantic_failed"
    assert sample.metadata["long_code"]["semantic_error_type"] == "TimeoutError"


def test_test_mode_preserves_existing_state_and_never_computes_or_calls_api(
    monkeypatch, tmp_path
):
    sample = _sample(tmp_path / "clip.mp4")
    from ayase.models import QualityMetrics

    sample.quality_metrics = QualityMetrics(long_code_approx_score=88.0)
    sample.metadata["long_code"] = {"availability": "old"}
    module = _long_code().LongCodeModule({"test_mode": True, "judge_model": "fixture"})
    module._processor = _Processor()
    module._encoder = _Encoder()
    module._client = _Client([])
    monkeypatch.setattr(
        module, "_read_frames", lambda *args: pytest.fail("test mode decoded video")
    )

    assert module.process(sample) is sample
    assert sample.quality_metrics.long_code_approx_score == 88.0
    assert sample.metadata["long_code"] == {"availability": "old"}
    assert module._client.chat.completions.calls == []


def test_short_video_is_resampled_to_exactly_eight_chronological_frames(tmp_path):
    import cv2

    path = tmp_path / "two-frame.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (8, 8)
    )
    assert writer.isOpened()
    writer.write(np.zeros((8, 8, 3), dtype=np.uint8))
    writer.write(np.full((8, 8, 3), 255, dtype=np.uint8))
    writer.release()

    frames, timestamps = _long_code().LongCodeModule._read_frames(path, 8)
    assert len(frames) == len(timestamps) == 8
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0.0
    assert timestamps[-1] == 0.5


def test_three_stage_order_multiround_average_and_combination(monkeypatch, tmp_path):
    sample = _sample(tmp_path / "clip.mp4")
    responses = [
        json.dumps({"captions": [str(index) for index in range(8)]}),
        json.dumps({"analysis": "The sequence progresses and preserves its subject."}),
        json.dumps({"narrative": 80, "causality": 60, "identity": 40}),
        json.dumps({"narrative": 100, "causality": 80, "identity": 60}),
    ]
    module = _ready_module(
        monkeypatch,
        {"judge_model": "fixture-vision", "semantic_rounds": 2, "alpha": 0.25},
    )
    module._client = _Client(responses)
    module._semantic_backend = "mock-three-stage"

    assert module.process(sample) is sample

    calls = module._client.chat.completions.calls
    assert [call["messages"][0]["content"].rsplit(": ", 1)[-1].rstrip(".") for call in calls] == [
        "caption",
        "reasoning",
        "scoring",
        "scoring",
    ]
    assert sample.quality_metrics.long_code_approx_narrative_score == 90.0
    assert sample.quality_metrics.long_code_approx_causality_score == 70.0
    assert sample.quality_metrics.long_code_approx_identity_score == 50.0
    assert sample.quality_metrics.long_code_approx_semantic_score == 70.0
    assert sample.quality_metrics.long_code_approx_score == pytest.approx(77.5)
    assert sample.metadata["long_code"]["availability"] == "full_approximate"
    caption_content = calls[0]["messages"][1]["content"]
    assert [item["text"] for item in caption_content if item["type"] == "text"][1:] == [
        f"FRAME {index + 1} @ {index:.3f}s" for index in range(8)
    ]


@pytest.mark.parametrize(
    "bad_response",
    [
        "not json",
        json.dumps({"captions": ["one", "two"]}),
        json.dumps({"captions": [str(index) for index in range(8)], "extra": 1}),
    ],
)
def test_malformed_caption_stage_keeps_only_dsa(monkeypatch, tmp_path, bad_response):
    sample = _sample(tmp_path / "clip.mp4")
    module = _ready_module(monkeypatch, {"judge_model": "fixture-vision"})
    module._client = _Client([bad_response])

    assert module.process(sample) is sample
    assert sample.quality_metrics.long_code_approx_dsa == 1.0
    assert sample.quality_metrics.long_code_approx_semantic_score is None
    assert sample.metadata["long_code"]["availability"] == "semantic_failed"


@pytest.mark.parametrize(
    "score_payload",
    [
        {"narrative": True, "causality": 60, "identity": 70},
        {"narrative": -1, "causality": 60, "identity": 70},
        {"narrative": 101, "causality": 60, "identity": 70},
        {"narrative": float("inf"), "causality": 60, "identity": 70},
        {"narrative": 80, "causality": 60, "identity": 70, "overall": 70},
    ],
)
def test_malformed_score_never_writes_partial_semantics(monkeypatch, tmp_path, score_payload):
    sample = _sample(tmp_path / "clip.mp4")
    module = _ready_module(monkeypatch, {"judge_model": "fixture-vision"})
    module._client = _Client(
        [
            json.dumps({"captions": [str(index) for index in range(8)]}),
            json.dumps({"analysis": "analysis"}),
            json.dumps(score_payload),
        ]
    )

    assert module.process(sample) is sample
    assert sample.quality_metrics.long_code_approx_dsa == 1.0
    assert sample.quality_metrics.long_code_approx_narrative_score is None
    assert sample.metadata["long_code"]["availability"] == "semantic_failed"


@pytest.mark.parametrize(
    "config",
    [
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"alpha": True},
        {"alpha": float("nan")},
        {"semantic_rounds": 0},
        {"semantic_rounds": True},
        {"num_frames": 1},
        {"num_frames": 7},
        {"num_frames": 9},
        {"timeout": 0},
        {"max_tokens": 0},
    ],
)
def test_invalid_config_makes_no_api_call_or_output(monkeypatch, tmp_path, config):
    sample = _sample(tmp_path / "clip.mp4")
    module = _ready_module(monkeypatch, {"judge_model": "fixture-vision", **config})
    client = _Client([])
    module._client = client

    assert module.process(sample) is sample
    assert client.chat.completions.calls == []
    assert sample.quality_metrics is None
    assert "long_code" not in sample.metadata


@pytest.mark.parametrize(
    "prompts",
    [None, [], ["one", "two"], ["one", "", "three"], ["one", 2, "three"]],
)
def test_invalid_shot_prompts_do_not_fabricate_or_call_api(monkeypatch, tmp_path, prompts):
    sample = _sample(tmp_path / "clip.mp4", ["placeholder", "placeholder", "placeholder"])
    if prompts is None:
        sample.metadata.pop("shot_prompts")
    else:
        sample.metadata["shot_prompts"] = prompts
    module = _ready_module(monkeypatch, {"judge_model": "fixture-vision"})
    client = _Client([])
    module._client = client

    assert module.process(sample) is sample
    assert client.chat.completions.calls == []
    assert sample.quality_metrics is None


def test_test_mode_setup_does_not_load_models_or_construct_client(monkeypatch):
    module = _long_code().LongCodeModule(
        {"test_mode": True, "judge_model": "fixture-vision", "api_key": "secret"}
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(XCLIPModel=pytest.fail, XCLIPProcessor=pytest.fail),
    )

    module.setup()
    assert module._encoder is None
    assert module._client is None


def test_invalid_config_is_rejected_before_snapshot_or_client_setup(monkeypatch):
    module = _long_code().LongCodeModule(
        {"num_frames": 7, "judge_model": "fixture", "api_key": "secret"}
    )
    monkeypatch.setattr(
        _long_code(),
        "download_hf_snapshot",
        lambda *args, **kwargs: pytest.fail("invalid config downloaded a model"),
    )

    module.setup()
    assert module._encoder is None
    assert module._client is None


def test_setup_uses_configured_managed_snapshot_and_checks_eight_frame_contract(
    monkeypatch, tmp_path
):
    calls = []
    cached = tmp_path / "models" / "microsoft--xclip-base-patch32"
    cached.mkdir(parents=True)
    monkeypatch.setattr(
        _long_code(),
        "download_hf_snapshot",
        lambda repo, models_dir, revision=None, allow_patterns=None: calls.append(
            (repo, models_dir, revision, allow_patterns)
        )
        or cached,
    )

    class LoadedEncoder:
        config = SimpleNamespace(
            vision_config=SimpleNamespace(num_frames=8),
            text_config=SimpleNamespace(max_position_embeddings=77),
        )

        def to(self, device):
            calls.append(("device", device))
            return self

        def eval(self):
            return self

    processor = _Processor()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            XCLIPProcessor=SimpleNamespace(
                from_pretrained=lambda path: calls.append(("processor", path)) or processor
            ),
            XCLIPModel=SimpleNamespace(
                from_pretrained=lambda path: calls.append(("encoder", path)) or LoadedEncoder()
            ),
        ),
    )
    module = _long_code().LongCodeModule({"models_dir": str(tmp_path / "models")})

    module.setup()

    assert calls[0] == (
        "microsoft/xclip-base-patch32",
        str(tmp_path / "models"),
        _long_code().ENCODER_REVISION,
        _long_code().ENCODER_ALLOW_PATTERNS,
    )
    assert ("processor", str(cached)) in calls
    assert ("encoder", str(cached)) in calls
    assert ("device", "cpu") in calls
    assert module._encoder_backend == "xclip-full-video"


def test_actual_xclip_processor_and_tiny_model_produce_finite_dsa_without_downloads(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    vocab = tmp_path / "vocab.json"
    merges = tmp_path / "merges.txt"
    vocab.write_text(
        json.dumps(
            {
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "b</w>": 3,
            }
        ),
        encoding="utf-8",
    )
    merges.write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = transformers.CLIPTokenizer(str(vocab), str(merges))
    image_processor = transformers.VideoMAEImageProcessor(
        size={"shortest_edge": 16},
        crop_size={"height": 16, "width": 16},
    )
    processor = transformers.XCLIPProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    config = transformers.XCLIPConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 77,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 16,
            "patch_size": 8,
            "num_channels": 3,
            "num_frames": 8,
            "mit_hidden_size": 16,
            "mit_intermediate_size": 32,
            "mit_num_hidden_layers": 1,
            "mit_num_attention_heads": 2,
        },
        projection_dim=16,
        prompt_layers=1,
        prompt_num_attention_heads=2,
    )
    torch.manual_seed(0)
    encoder = transformers.XCLIPModel(config).eval()

    module = _long_code().LongCodeModule()
    module._encoder = encoder
    module._processor = processor
    module._device = "cpu"
    frames = [np.full((20, 20, 3), index * 20, dtype=np.uint8) for index in range(8)]

    dsa, truncation = module._compute_dsa(["a", "b", "a b"], frames)

    assert dsa is not None and np.isfinite(dsa)
    assert -1.0 <= dsa <= 1.0
    assert truncation["all_shots_represented"] is True


@pytest.mark.parametrize(
    "video_batch",
    [
        {},
        pytest.param("wrong-frame-count", id="wrong-frame-count"),
    ],
)
def test_dsa_rejects_missing_or_wrong_shaped_native_pixel_values(video_batch):
    import torch

    if video_batch == "wrong-frame-count":
        video_batch = {"pixel_values": torch.zeros(1, 7, 3, 2, 2)}
    processor = _Processor()
    processor.image_processor = lambda images, **kwargs: video_batch
    module = _long_code().LongCodeModule()
    module._encoder = _Encoder()
    module._processor = processor
    module._device = "cpu"
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(8)]

    with pytest.raises(ValueError, match="pixel_values shaped"):
        module._compute_dsa(["a", "b", "a b"], frames)


def test_dispose_closes_client_and_releases_resources():
    closed = []
    module = _long_code().LongCodeModule()
    module._client = SimpleNamespace(close=lambda: closed.append(True))
    module._processor = object()
    module._encoder = object()
    module._mounted = True

    module.on_dispose()

    assert closed == [True]
    assert module._client is module._processor is module._encoder is None
    assert module._mounted is False


def test_pipeline_provenance_backend_and_serialization_for_success_then_failure(
    monkeypatch, tmp_path
):
    from ayase.pipeline import Pipeline

    responses = [
        json.dumps({"captions": [str(index) for index in range(8)]}),
        json.dumps({"analysis": "The ordered story remains coherent."}),
        json.dumps({"narrative": 90, "causality": 80, "identity": 70}),
    ]
    module = _ready_module(monkeypatch, {"judge_model": "fixture-vision"})
    module._client = _Client(responses)
    module._semantic_backend = "mock-three-stage"
    module._mounted = True
    pipeline = Pipeline([module], allow_provenance=["adapted"])
    pipeline.start()

    success = pipeline.process_sample(_sample(tmp_path / "success.mp4"))

    assert success.quality_metrics.long_code_approx_dsa == 1.0
    assert success.quality_metrics.long_code_approx_score == 90.0
    assert success.quality_metrics.metric_backends["long_code"] == (
        "fake-order-sensitive-video+mock-three-stage"
    )
    for field in _long_code().OUTPUT_FIELDS:
        assert success.quality_metrics.metric_provenance[field] == "adapted"
    restored_success = type(success).model_validate_json(success.model_dump_json())
    assert restored_success.quality_metrics.long_code_approx_score == 90.0
    assert restored_success.metadata["long_code"]["availability"] == "full_approximate"

    class ThrowingCompletions:
        def create(self, **kwargs):
            raise ConnectionError("fixture transport failure")

    module._client = SimpleNamespace(
        chat=SimpleNamespace(completions=ThrowingCompletions())
    )
    failed_semantic = pipeline.process_sample(_sample(tmp_path / "failure.mp4"))

    assert failed_semantic.quality_metrics.long_code_approx_dsa == 1.0
    assert failed_semantic.quality_metrics.long_code_approx_score is None
    assert failed_semantic.quality_metrics.metric_backends["long_code"] == (
        "fake-order-sensitive-video"
    )
    assert failed_semantic.quality_metrics.metric_provenance == {
        "long_code_approx_dsa": "adapted"
    }
    assert failed_semantic.metadata["long_code"]["availability"] == "semantic_failed"
    assert failed_semantic.metadata["long_code"]["semantic_error_type"] == "ConnectionError"
    restored_failure = type(failed_semantic).model_validate_json(
        failed_semantic.model_dump_json()
    )
    assert restored_failure.quality_metrics.long_code_approx_score is None
    assert restored_failure.quality_metrics.metric_backends["long_code"] == (
        "fake-order-sensitive-video"
    )
