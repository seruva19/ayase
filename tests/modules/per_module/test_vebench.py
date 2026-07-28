"""Tests for the upstream VE-Bench video-edit evaluator."""

import hashlib
from types import SimpleNamespace

import pytest

from tests.modules.conftest import _test_module_basics


def test_vebench_basics():
    from ayase.modules.vebench import VEBenchModule

    _test_module_basics(VEBenchModule, "vebench")
    assert VEBenchModule.metric_groups == {"vebench_score": "alignment"}


def test_vebench_requires_video_reference_and_instruction(video_sample):
    from ayase.modules.vebench import VEBenchModule

    module = VEBenchModule()
    module._backend = "vebench"
    module._model = SimpleNamespace(evaluate=lambda *_args: 1.0)

    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_vebench_process_stores_reference_scalar(video_sample, synthetic_video):
    from ayase.modules.vebench import VEBenchModule

    calls = []
    video_sample.reference_path = synthetic_video
    module = VEBenchModule({"instruction": "Turn the subject's head"})
    module._backend = "vebench"
    module._model = SimpleNamespace(
        evaluate=lambda *args: calls.append(args) or 1.3104406595230103
    )

    result = module.process(video_sample)

    assert result is video_sample
    assert result.quality_metrics.vebench_score == pytest.approx(1.3104406595230103)
    assert calls[0][0] == "Turn the subject's head"
    assert calls[0][1] == str(synthetic_video.resolve())
    assert calls[0][2] == str(video_sample.path.resolve())


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
def test_vebench_rejects_non_finite_scores(video_sample, synthetic_video, bad_score):
    from ayase.modules.vebench import VEBenchModule

    video_sample.reference_path = synthetic_video
    module = VEBenchModule({"instruction": "Edit the video"})
    module._backend = "vebench"
    module._model = SimpleNamespace(evaluate=lambda *_args: bad_score)

    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_vebench_weight_snapshot_uses_exact_sizes_and_hashes(tmp_path, monkeypatch):
    import ayase.modules.vebench as vebench_module
    from ayase.modules.vebench import VEBenchModule

    weights = tuple(
        (
            name,
            size,
            hashlib.sha256(b"\0" * size).hexdigest(),
        )
        for name, size in (
            ("first.pth", 17),
            ("second.pth", 23),
        )
    )
    monkeypatch.setattr(vebench_module, "_WEIGHTS", weights)

    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    for filename, size, _sha256 in weights:
        (checkpoint_dir / filename).write_bytes(b"\0" * size)
    VEBenchModule._ensure_weights(checkpoint_dir)
    VEBenchModule._ensure_weights(checkpoint_dir)

    for filename, size, sha256 in weights:
        assert (checkpoint_dir / filename).stat().st_size == size
        assert len(sha256) == 64

    (checkpoint_dir / "first.pth").write_bytes(b"x" * 17)
    with pytest.raises(RuntimeError, match="SHA-256"):
        VEBenchModule._ensure_weights(checkpoint_dir)


def test_vebench_declares_huggingface_snapshot():
    from ayase.modules.vebench import VEBenchModule, _MODEL_REVISION

    assert VEBenchModule.models[0]["id"] == "vebench==1.0.0"
    assert len(VEBenchModule.models) == 2
    assert VEBenchModule.models[1]["type"] == "huggingface"
    assert VEBenchModule.models[1]["id"] == "AkaneTendo25/ayase-runtime-assets"
    assert _MODEL_REVISION in VEBenchModule.models[1]["notes"]
    assert VEBenchModule.models[1]["auto_download"] is True
