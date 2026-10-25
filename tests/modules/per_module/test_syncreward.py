"""Tests for the SyncReward module."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from ..conftest import _test_module_basics


class _Output:
    def __init__(self, value):
        self.value = value

    def float(self):
        return self

    def __getitem__(self, index):
        assert index == 0
        return self.value


class _Model:
    def __init__(self, value=1.25):
        self.value = value
        self.calls = []

    def __call__(self, inputs, *, encode_batch_size):
        self.calls.append((inputs, encode_batch_size))
        return _Output(self.value)


class _Torch:
    bfloat16 = object()

    @staticmethod
    def no_grad():
        return nullcontext()

    @staticmethod
    def autocast(*args, **kwargs):
        return nullcontext()


def _ready_module(value=1.25):
    from ayase.modules.syncreward import SyncRewardModule

    module = SyncRewardModule()
    module._backend = "syncreward"
    module._processor = object()
    module._processor_sample_rate = lambda processor: 48000
    module._device = SimpleNamespace(type="cpu")
    module._torch = _Torch()
    module._model = _Model(value)
    module._load_av = lambda path, sr: (
        np.zeros((120, 8, 8, 3), dtype=np.uint8),
        25.0,
        np.zeros(240000, dtype=np.float32),
    )
    return module


def test_syncreward_basics():
    from ayase.modules.syncreward import SyncRewardModule

    _test_module_basics(SyncRewardModule, "syncreward")


def test_syncreward_canonical_inference_protocol(video_sample):
    module = _ready_module()
    captured = {}

    def prepare(processor, batch, args, crop, device):
        captured.update(batch=batch, args=args, crop=crop, device=device)
        return {"prepared": True}

    module._prepare_inputs = prepare
    result = module.process(video_sample)

    assert result is video_sample
    assert result.quality_metrics.syncreward_score == 1.25
    assert captured["crop"] == "center"
    assert captured["args"].num_segments == 18
    assert captured["args"].segment_duration == 0.5
    assert captured["args"].segment_stride == 0.25
    assert captured["batch"]["fps"] == [25.0]
    assert module._model.calls == [({"prepared": True}, 18)]


def test_syncreward_skips_non_video(image_sample):
    module = _ready_module()
    module._load_av = lambda *args: pytest.fail("decoder must not run")
    assert module.process(image_sample) is image_sample
    assert image_sample.quality_metrics is None


@pytest.mark.parametrize("video_frames,audio_samples", [(1, 4800), (2, 4799)])
def test_syncreward_skips_short_media(video_sample, video_frames, audio_samples):
    module = _ready_module()
    module._load_av = lambda path, sr: (
        np.zeros((video_frames, 2, 2, 3), dtype=np.uint8),
        25.0,
        np.zeros(audio_samples, dtype=np.float32),
    )
    module._prepare_inputs = lambda *args: pytest.fail("short media must not be prepared")
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_syncreward_decode_failure_leaves_no_score(video_sample):
    module = _ready_module()
    module._load_av = lambda *args: (_ for _ in ()).throw(RuntimeError("decode failed"))
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 2.01])
def test_syncreward_rejects_invalid_raw_outputs(video_sample, value):
    module = _ready_module(value)
    module._prepare_inputs = lambda *args: {}
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_syncreward_setup_test_mode_never_downloads(monkeypatch):
    from ayase.modules.syncreward import SyncRewardModule
    import ayase.config

    monkeypatch.setattr(
        ayase.config,
        "download_hf_snapshot",
        lambda *args, **kwargs: pytest.fail("test mode must not download"),
    )
    module = SyncRewardModule({"test_mode": True})
    module.setup()
    assert module._backend is None


def test_syncreward_setup_downloads_pinned_assets_and_wires_upstream(monkeypatch, tmp_path):
    from ayase.modules.syncreward import SyncRewardModule
    import ayase.config
    import ayase.third_party.syncreward.model as vendor_model

    downloads = []
    builds = []
    reward_root = tmp_path / "reward"
    encoder_root = tmp_path / "encoder"

    def download(repo_id, models_dir="models", **kwargs):
        downloads.append((repo_id, models_dir, kwargs))
        return reward_root if repo_id == "Guan123/SyncReward" else encoder_root

    class SetupModel:
        num_segments = 18

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    processor = object()
    monkeypatch.setattr(ayase.config, "download_hf_snapshot", download)
    monkeypatch.setattr(vendor_model, "load_processor", lambda path: (builds.append(("processor", path)), processor)[1])
    monkeypatch.setattr(
        vendor_model,
        "build_reward_model",
        lambda encoder, **kwargs: (builds.append(("model", encoder, kwargs)), SetupModel())[1],
    )
    monkeypatch.setattr(vendor_model, "processor_sample_rate", lambda value: 48000)
    monkeypatch.setattr(SyncRewardModule, "_global_test_mode", False)

    module = SyncRewardModule({"device": "cpu", "models_dir": str(tmp_path / "models")})
    module.setup()

    assert module._backend == "syncreward"
    assert downloads == [
        (
            "Guan123/SyncReward",
            str(tmp_path / "models"),
            {
                "revision": "04bc7701da8d8466c7b3aa2dec22a2a5051e6f0a",
                "allow_patterns": ["syncreward-stage2/*"],
            },
        ),
        (
            "facebook/pe-av-large",
            str(tmp_path / "models"),
            {"revision": "0d24878d4107d64bef49e53602fc34ce6f94f6d8"},
        ),
    ]
    assert builds == [
        ("processor", str(encoder_root)),
        (
            "model",
            str(encoder_root),
            {"checkpoint": str(reward_root / "syncreward-stage2"), "num_segments": 18},
        ),
    ]


def test_syncreward_expert_local_paths_skip_download(monkeypatch, tmp_path):
    from ayase.modules.syncreward import SyncRewardModule
    import ayase.config
    import ayase.third_party.syncreward.model as vendor_model

    checkpoint = tmp_path / "checkpoint"
    encoder = tmp_path / "encoder"
    seen = []

    class SetupModel:
        num_segments = 18

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        ayase.config,
        "download_hf_snapshot",
        lambda *args, **kwargs: pytest.fail("local overrides must skip downloads"),
    )
    monkeypatch.setattr(vendor_model, "load_processor", lambda path: seen.append(path) or object())
    monkeypatch.setattr(
        vendor_model,
        "build_reward_model",
        lambda path, **kwargs: seen.append((path, kwargs["checkpoint"])) or SetupModel(),
    )
    monkeypatch.setattr(SyncRewardModule, "_global_test_mode", False)

    module = SyncRewardModule(
        {"device": "cpu", "checkpoint": str(checkpoint), "encoder_path": str(encoder)}
    )
    module.setup()
    assert seen == [str(encoder), (str(encoder), str(checkpoint))]


def test_syncreward_runtime_cleanup_closes_dll_and_disables_processing(video_sample):
    module = _ready_module()

    class Handle:
        closed = False

        def close(self):
            self.closed = True

    handle = Handle()
    module._dll_directory = handle
    module.on_dispose()

    assert handle.closed
    assert module._dll_directory is None
    assert module._backend is None
    assert module._model is None
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_syncreward_setup_failure_releases_partial_runtime(monkeypatch):
    from ayase.modules.syncreward import SyncRewardModule
    import ayase.config

    class Handle:
        closed = False

        def close(self):
            self.closed = True

    handle = Handle()
    monkeypatch.setattr(
        ayase.config,
        "download_hf_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(SyncRewardModule, "_global_test_mode", False)
    module = SyncRewardModule({"device": "cpu"})
    module._dll_directory = handle
    module._model = object()
    module._processor = object()
    module.setup()

    assert handle.closed
    assert module._dll_directory is None
    assert module._backend == "unavailable"
    assert module._model is None
    assert module._processor is None


def test_syncreward_rejects_checkpoint_segment_mismatch():
    from ayase.modules.syncreward import SyncRewardModule

    module = SyncRewardModule({"num_segments": 12})
    with pytest.raises(ValueError, match=r"checkpoint num_segments \(18\).+configured.+\(12\)"):
        module._validate_model_num_segments(SimpleNamespace(num_segments=18))


def test_syncreward_vendor_center_segmentation_is_canonical():
    from ayase.third_party.syncreward.data import window_segments

    video = np.arange(250, dtype=np.uint8).reshape(250, 1, 1, 1)
    audio = np.arange(480000, dtype=np.float32)
    clips_v, clips_a = window_segments(video, audio, 25.0, 48000, 18, 0.5, 0.25, "center")

    assert len(clips_v) == len(clips_a) == 18
    assert clips_v[0].shape[0] == 12
    assert clips_a[0].shape[0] == 24000
    assert int(clips_v[0][0, 0, 0, 0]) == 66
    assert float(clips_a[0][0]) == 126000.0
    assert int(clips_v[-1][0, 0, 0, 0]) == 172
    assert float(clips_a[-1][0]) == 330000.0


def test_syncreward_vendor_segments_pad_short_inputs():
    from ayase.third_party.syncreward.data import window_segments

    video = np.array([[[[7]]], [[[9]]]], dtype=np.uint8)
    audio = np.array([1.0, 2.0], dtype=np.float32)
    clips_v, clips_a = window_segments(video, audio, 2.0, 4, 2, 1.0, 0.5, "center")

    assert [clip[:, 0, 0, 0].tolist() for clip in clips_v] == [[7, 9], [9, 9]]
    assert [clip.tolist() for clip in clips_a] == [[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]


def test_syncreward_wrapper_preserves_real_model_native_reward(video_sample):
    import math

    import torch
    from torch import nn

    from ayase.modules.syncreward import SyncRewardModule
    from ayase.third_party.syncreward.model import SyncRewardModel

    class TinyEncoder(nn.Module):
        embed_dim = 8

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward(self, inputs, num_segments, chunk_size):
            batch_size = 1
            shape = (batch_size, num_segments, self.embed_dim)
            return torch.zeros(shape), torch.ones(shape)

    model = SyncRewardModel(
        TinyEncoder(),
        num_segments=18,
        sync_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_reward=2.0,
    ).eval()
    with torch.no_grad():
        model.reward_head[-1].weight.zero_()
        model.reward_head[-1].bias.fill_(math.log(3.0))

    module = SyncRewardModule({"device": "cpu"})
    module._backend = "syncreward"
    module._torch = torch
    module._device = torch.device("cpu")
    module._processor = object()
    module._processor_sample_rate = lambda processor: 48000
    module._load_av = lambda path, sample_rate: (
        np.zeros((120, 2, 2, 3), dtype=np.uint8),
        25.0,
        np.zeros(240000, dtype=np.float32),
    )
    module._prepare_inputs = lambda *args: {"stub": True}
    module._model = model

    result = module.process(video_sample)

    assert result is video_sample
    assert result.quality_metrics.syncreward_score == pytest.approx(1.5, abs=1e-6)
