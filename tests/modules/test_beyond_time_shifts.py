"""Focused tests for the pinned Beyond Time Shifts AV-sync evaluator."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ayase.models import Sample
from tests.modules.conftest import _test_module_basics


# Immutable MIT-licensed source fixture from SOURCE_REVISION; no external checkout.
SOURCE_FILE = Path(__file__).resolve().parents[1] / "fixtures/beyond_time_shifts/evaluator.py"


def test_basics_pins_and_native_metadata():
    from ayase.modules.beyond_time_shifts import (
        BASE_MODEL_REVISION,
        CHECKPOINT_REVISION,
        CHECKPOINT_SHA256,
        SOURCE_REVISION,
        BeyondTimeShiftsModule,
    )

    _test_module_basics(BeyondTimeShiftsModule, "beyond_time_shifts")
    module = BeyondTimeShiftsModule()
    assert SOURCE_REVISION == "f93e5ffaf8af144d37c39784426a8e0029734026"
    assert BASE_MODEL_REVISION == "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
    assert CHECKPOINT_REVISION == "5b88707fd6facf12fe627c5ed44196c0e689e467"
    assert CHECKPOINT_SHA256 == "ba36a405792367a15ee1ff69ac167ec4644bd2f09d87ed3286c1e0aecf10c0e6"
    assert "first 60 qwen-utils reader-sampled frames" in module.runtime_trace["protocol"]
    assert "768-frame reader cap" in module.runtime_trace["protocol"]
    assert module.runtime_trace["precision"] == "bf16"
    assert module.runtime_trace["attention"] == "sdpa"
    assert module.runtime_trace["parameters"] == 4_703_464_448
    assert module.runtime_trace["load_flow"] == "CPU FP32 -> CUDA FP32 -> CUDA BF16 (author SDPA/eager path)"
    assert "unbounded" in module.metric_info["beyond_time_shifts_score"].lower()
    assert "non-commercial" in module.models[0]["notes"].lower()


def test_author_score_batch_uses_final_token_and_linear_head(monkeypatch):
    """Execute the pinned author's real score_batch and regression-head methods."""
    torch = pytest.importorskip("torch")
    source = SOURCE_FILE

    processing = ModuleType("qwen2_5_omni.processing_qwen2_5_omni")
    processing.Qwen2_5OmniProcessor = object
    hacked = ModuleType("bts_test_hacked_qwen")
    hacked.hacked_Qwen_Thinker = object
    package = ModuleType("bts_test_models")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "qwen2_5_omni.processing_qwen2_5_omni", processing)
    monkeypatch.setitem(sys.modules, "bts_test_models", package)
    monkeypatch.setitem(sys.modules, "bts_test_models.hacked_qwen", hacked)
    spec = importlib.util.spec_from_file_location(
        "bts_test_models.evaluator", source, submodule_search_locations=None
    )
    assert spec is not None and spec.loader is not None
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)

    class Transformer(torch.nn.Module):
        def forward(self, **_kwargs):
            # Earlier token is deliberately very different from final [2, 3].
            hidden = torch.tensor([[[100.0, 100.0], [2.0, 3.0]]])
            return SimpleNamespace(hidden_states=(hidden,))

    evaluator = author.AVSyncEvaluator.__new__(author.AVSyncEvaluator)
    torch.nn.Module.__init__(evaluator)
    evaluator.transformer = Transformer()
    evaluator.regression_head = author.MLP_RegressionHead(input_dim=2, output_dim=1)
    evaluator.regression_head.fc_1.weight.data.copy_(torch.tensor([[4.0, 5.0]]))
    evaluator.regression_head.fc_1.bias.data.copy_(torch.tensor([1.0]))
    evaluator.precision_mode = "bf16"
    evaluator.device_str = "cpu"
    evaluator.dtype = torch.bfloat16
    evaluator._prepare_inputs = lambda _videos, _audios: {}

    score = evaluator.score_batch([torch.zeros(1)], [np.zeros(1, dtype=np.float32)])
    assert score == [24.0]


def test_author_prepare_inputs_keeps_score_conversation_and_drops_one_token(monkeypatch):
    torch = pytest.importorskip("torch")
    source = SOURCE_FILE

    processing = ModuleType("qwen2_5_omni.processing_qwen2_5_omni")
    processing.Qwen2_5OmniProcessor = object
    hacked = ModuleType("bts_prepare_hacked_qwen")
    hacked.hacked_Qwen_Thinker = object
    package = ModuleType("bts_prepare_models")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "qwen2_5_omni.processing_qwen2_5_omni", processing)
    monkeypatch.setitem(sys.modules, "bts_prepare_models", package)
    monkeypatch.setitem(sys.modules, "bts_prepare_models.hacked_qwen", hacked)
    spec = importlib.util.spec_from_file_location("bts_prepare_models.evaluator", source)
    assert spec is not None and spec.loader is not None
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    captured = {}

    class Batch(dict):
        def to(self, device):
            captured["device"] = device
            return self

    class Processor:
        def apply_chat_template(self, conversations, **kwargs):
            captured["conversations"] = conversations
            captured["template"] = kwargs
            return ["rendered"]

        def __call__(self, **kwargs):
            captured["processor"] = kwargs
            return Batch(
                input_ids=torch.tensor([[10, 11, 12]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
                input_features=torch.ones(1, dtype=torch.float32),
            )

    evaluator = author.AVSyncEvaluator.__new__(author.AVSyncEvaluator)
    torch.nn.Module.__init__(evaluator)
    evaluator.preprocessor = Processor()
    evaluator.device_str = "cpu"
    evaluator.dtype = torch.bfloat16
    evaluator.v_fps = 12
    evaluator.v_size = 140
    inputs = evaluator._prepare_inputs([torch.zeros(1)], [np.zeros(1)])

    assert captured["conversations"][0][-1]["content"][0]["text"] == "SCORE"
    assert captured["template"] == {
        "add_generation_prompt": True,
        "tokenize": False,
        "use_audio_in_video": True,
    }
    assert captured["processor"]["videos_kwargs"] == {
        "insert_timestamps": True,
        "do_resize": False,
        "use_audio_in_video": True,
        "fps": 12,
    }
    assert inputs["input_ids"].tolist() == [[10, 11]]
    assert inputs["attention_mask"].tolist() == [[1, 1]]
    assert inputs["input_features"].dtype == torch.bfloat16


@pytest.mark.parametrize("long", [False, True])
def test_load_clip_exact_first_five_seconds_and_zero_padding(monkeypatch, long):
    torch = pytest.importorskip("torch")
    from ayase.modules.beyond_time_shifts import load_clip

    frames = 65 if long else 2
    samples = 80_005 if long else 10
    video = torch.arange(frames * 3 * 2 * 2, dtype=torch.float32).reshape(frames, 3, 2, 2)
    audio = torch.arange(samples, dtype=torch.float32)
    utility = SimpleNamespace(
        process_mm_info=lambda conversations, use_audio_in_video: (
            [audio], None, [video]
        )
    )
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", utility)

    clipped_video, clipped_audio = load_clip("clip.mp4", fps=12, size=2)
    assert clipped_video.shape == (60, 3, 2, 2)
    assert clipped_audio.shape == (80_000,)
    if long:
        assert torch.equal(clipped_video, video[:60])
        assert torch.equal(clipped_audio, audio[:80_000])
    else:
        # This deliberately matches demo_single_video.load_clip: its default
        # NumPy zero dtype promotes short float32 video arrays to float64.
        raw_video = video.numpy()
        source_padding = np.concatenate(
            [np.zeros((1, 3, 2, 2)) for _ in range(58)], axis=0
        )
        source_video = torch.from_numpy(np.concatenate([raw_video, source_padding], axis=0))
        assert clipped_video.dtype == torch.float64
        assert torch.equal(clipped_video, source_video)
        assert not bool(clipped_video[2:].any())
        assert torch.equal(clipped_audio[:10], audio)
        assert not bool(clipped_audio[10:].any())


def test_long_video_keeps_first_60_of_qwen_utils_capped_reader_samples(monkeypatch):
    torch = pytest.importorskip("torch")
    from ayase.modules.beyond_time_shifts import load_clip

    # qwen-omni-utils smart_nframes globally caps long-video decoding at 768.
    # The author helper then slices that reader output, rather than performing
    # an independent five-wall-second streaming crop.
    video = torch.arange(768, dtype=torch.float32).reshape(768, 1, 1, 1).expand(
        768, 3, 2, 2
    )
    audio = torch.ones(100_000)
    monkeypatch.setitem(
        sys.modules,
        "qwen_omni_utils",
        SimpleNamespace(
            process_mm_info=lambda *_args, **_kwargs: ([audio], None, [video])
        ),
    )

    clipped, _ = load_clip("120-seconds.mp4", fps=12, size=2)
    assert clipped[:, 0, 0, 0].tolist() == list(range(60))


def test_load_clip_requires_both_audio_and_video(monkeypatch):
    pytest.importorskip("torch")
    from ayase.modules.beyond_time_shifts import load_clip

    monkeypatch.setitem(
        sys.modules,
        "qwen_omni_utils",
        SimpleNamespace(process_mm_info=lambda *_args, **_kwargs: ([], None, [])),
    )
    with pytest.raises(ValueError, match="paired embedded audio"):
        load_clip("silent.mp4")


def test_decode_forces_torchvision_and_private_ffmpeg_then_restores(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from ayase.modules.beyond_time_shifts import load_clip

    ffmpeg = tmp_path / "private" / "ffmpeg.exe"
    ffmpeg.parent.mkdir()
    ffmpeg.touch()
    observed = {}

    def process(*_args, **_kwargs):
        import os

        observed["reader"] = os.environ.get("FORCE_QWENVL_VIDEO_READER")
        observed["path_head"] = os.environ.get("PATH", "").split(os.pathsep)[0]
        return (
            [torch.ones(80_000)],
            None,
            [torch.ones(60, 3, 2, 2)],
        )

    vision = SimpleNamespace(
        FORCE_QWENVL_VIDEO_READER="decord",
        get_video_reader_backend=SimpleNamespace(cache_clear=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=process))
    monkeypatch.setitem(sys.modules, "qwen_omni_utils.v2_5.vision_process", vision)
    monkeypatch.setenv("FORCE_QWENVL_VIDEO_READER", "decord")
    old_path = __import__("os").environ.get("PATH")

    load_clip("clip.mp4", fps=12, size=2, ffmpeg_path=ffmpeg)

    assert observed == {"reader": "torchvision", "path_head": str(ffmpeg.parent)}
    assert vision.FORCE_QWENVL_VIDEO_READER == "decord"
    assert __import__("os").environ["FORCE_QWENVL_VIDEO_READER"] == "decord"
    assert __import__("os").environ.get("PATH") == old_path


def test_decode_restores_reader_global_created_during_first_import(monkeypatch):
    torch = pytest.importorskip("torch")
    from ayase.modules.beyond_time_shifts import load_clip

    monkeypatch.delenv("FORCE_QWENVL_VIDEO_READER", raising=False)
    monkeypatch.delitem(sys.modules, "qwen_omni_utils.v2_5.vision_process", raising=False)
    vision = SimpleNamespace(
        FORCE_QWENVL_VIDEO_READER="torchvision",
        get_video_reader_backend=SimpleNamespace(cache_clear=lambda: None),
    )

    def process(*_args, **_kwargs):
        # Real qwen-omni-utils imports this module while observing the temporary
        # environment variable. Reproduce that first-import timing here.
        sys.modules["qwen_omni_utils.v2_5.vision_process"] = vision
        return [torch.ones(80_000)], None, [torch.ones(60, 3, 2, 2)]

    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=process))
    load_clip("clip.mp4", fps=12, size=2)

    assert vision.FORCE_QWENVL_VIDEO_READER is None
    assert "FORCE_QWENVL_VIDEO_READER" not in __import__("os").environ


@pytest.mark.parametrize("kind", ["video_nan", "audio_inf"])
def test_load_clip_rejects_nonfinite_values_in_assessed_region(monkeypatch, kind):
    torch = pytest.importorskip("torch")
    from ayase.modules.beyond_time_shifts import load_clip

    video = torch.ones(2, 3, 2, 2)
    audio = torch.ones(10)
    if kind == "video_nan":
        video[0, 0, 0, 0] = float("nan")
    else:
        audio[0] = float("inf")
    monkeypatch.setitem(
        sys.modules,
        "qwen_omni_utils",
        SimpleNamespace(
            process_mm_info=lambda *_args, **_kwargs: ([audio], None, [video])
        ),
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_clip("invalid.mp4", fps=12, size=2)


def test_checkpoint_digest_and_trained_state_guards(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    import ayase.modules.beyond_time_shifts as implementation

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"not the checkpoint")
    monkeypatch.setattr(implementation, "_sha256", lambda _path: "0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        implementation._validate_checkpoint_file(checkpoint)

    evaluator = SimpleNamespace(
        transformer=SimpleNamespace(
            state_dict=lambda: {
                "layer.weight": torch.zeros(2, 2),
                "layer.bias": torch.zeros(2),
            },
            named_parameters=lambda: iter(
                [
                    ("layer.weight", torch.zeros(2, 2)),
                    ("layer.bias", torch.zeros(2)),
                ]
            ),
        )
    )
    valid = {
        "regression_head.fc_1.weight": torch.ones(1, 2048),
        "regression_head.fc_1.bias": torch.ones(1),
        "transformer.layer.weight": torch.ones(2, 2),
        "transformer.layer.bias": torch.ones(2),
    }
    implementation._validate_checkpoint_state(valid, evaluator)
    for missing in ("regression_head.fc_1.weight", "regression_head.fc_1.bias"):
        state = dict(valid)
        state.pop(missing)
        with pytest.raises(ValueError, match="trained tensor"):
            implementation._validate_checkpoint_state(state, evaluator)
    partial = {key: value for key, value in valid.items() if key != "transformer.layer.weight"}
    with pytest.raises(ValueError, match="1 missing"):
        implementation._validate_checkpoint_state(partial, evaluator)

    extra_head = dict(valid)
    extra_head["regression_head.fc_2.weight"] = torch.ones(1, 1)
    with pytest.raises(ValueError, match="unexpected regression-head"):
        implementation._validate_checkpoint_state(extra_head, evaluator)

    many_expected = {f"layer.{index}": torch.zeros(1) for index in range(11)}
    high_coverage_evaluator = SimpleNamespace(
        transformer=SimpleNamespace(state_dict=lambda: many_expected)
    )
    over_ninety_percent = {
        "regression_head.fc_1.weight": torch.ones(1, 2048),
        "regression_head.fc_1.bias": torch.ones(1),
        **{
            f"transformer.layer.{index}": torch.ones(1)
            for index in range(10)
        },
    }
    with pytest.raises(ValueError, match="1 missing"):
        implementation._validate_checkpoint_state(
            over_ninety_percent, high_coverage_evaluator
        )


def test_test_mode_and_cuda_gate_do_not_acquire_assets(monkeypatch, tmp_path):
    import ayase.modules.beyond_time_shifts as implementation

    def fail():
        pytest.fail("asset acquisition must not run")

    module = implementation.BeyondTimeShiftsModule(
        {"test_mode": True, "models_dir": str(tmp_path)}
    )
    monkeypatch.setattr(module, "_resolve_assets", fail)
    module.setup()
    assert module._backend is None
    assert module._evaluator is None

    monkeypatch.setattr(implementation.BeyondTimeShiftsModule, "_global_test_mode", False)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    module = implementation.BeyondTimeShiftsModule({"models_dir": str(tmp_path)})
    monkeypatch.setattr(module, "_resolve_assets", fail)
    module.setup()
    assert module._backend == "unavailable"
    assert not module._ml_available


def test_process_publishes_finite_unbounded_native_score(monkeypatch, tmp_path):
    import ayase.modules.beyond_time_shifts as implementation

    module = implementation.BeyondTimeShiftsModule()
    module._evaluator = SimpleNamespace(score_batch=lambda videos, audios: [-123.5])
    module._backend = "native:test"
    module._ml_available = True
    monkeypatch.setattr(
        implementation, "load_clip", lambda *_args, **_kwargs: (object(), object())
    )
    sample = Sample(path=tmp_path / "video.mp4", is_video=True)

    assert module.process(sample) is sample
    assert sample.quality_metrics.beyond_time_shifts_score == -123.5
    assert sample.quality_metrics.metric_backends["beyond_time_shifts"] == "native:test"


def test_process_rejects_nonfinite_and_dispose_resets_state(monkeypatch, tmp_path):
    import ayase.modules.beyond_time_shifts as implementation

    module = implementation.BeyondTimeShiftsModule()
    module._evaluator = SimpleNamespace(score_batch=lambda videos, audios: [float("nan")])
    module._backend = "native:test"
    module._ml_available = True
    monkeypatch.setattr(
        implementation, "load_clip", lambda *_args, **_kwargs: (object(), object())
    )
    sample = Sample(path=tmp_path / "video.mp4", is_video=True)
    assert module.process(sample) is sample
    assert sample.quality_metrics is None
    module.on_dispose()
    assert module._backend is None
    assert module._evaluator is None
    assert not module._ml_available
