import math
from pathlib import Path

import pytest

from ayase.models import Sample


class _Pipeline:
    def __init__(self):
        self.metrics = {}

    def add_dataset_metric(self, name, value):
        self.metrics[name] = value


def _image(path: Path, color=(10, 20, 30)) -> Path:
    from PIL import Image

    Image.new("RGB", (9, 7), color).save(path)
    return path


def test_chd_basics_and_canonical_math():
    import numpy as np

    from ayase.modules.chd import CHDModule, _hellinger_distance

    module = CHDModule({"test_mode": True})
    assert module.name == "chd"
    assert module.input_type == "image"
    assert module.metric_info["chd_1d"].startswith("Released unigram")
    assert _hellinger_distance([1, 0], [1, 0]) == pytest.approx(0.0)
    assert _hellinger_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    expected = math.sqrt(
        (math.sqrt(0.25) - math.sqrt(0.5)) ** 2 + (math.sqrt(0.75) - math.sqrt(0.5)) ** 2
    ) / math.sqrt(2)
    assert _hellinger_distance([1, 3], [2, 2]) == pytest.approx(expected)
    with pytest.raises(ValueError):
        _hellinger_distance([0, 0], [1, 0])
    with pytest.raises(ValueError):
        _hellinger_distance(np.array([1.0, math.nan]), np.array([1.0, 0.0]))


def test_chd_process_is_image_only_and_rejects_invalid(tmp_path):
    from ayase.modules.chd import CHDModule

    valid = _image(tmp_path / "valid.png")
    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image", encoding="utf-8")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    module = CHDModule({"test_mode": True})
    image_sample = Sample(path=valid, is_video=False)
    bad_sample = Sample(path=invalid, is_video=False)
    video_sample = Sample(path=video, is_video=True)
    assert module.process(image_sample) is image_sample
    assert module.process(bad_sample) is bad_sample
    assert module.process(video_sample) is video_sample
    assert module._generated_paths == [valid.resolve()]
    assert module._invalid_generated is True


def test_chd_reference_sets_are_independent_and_strict(tmp_path):
    from ayase.modules.chd import CHDModule

    reference = _image(tmp_path / "reference.png")
    generated = _image(tmp_path / "generated.png", (30, 20, 10))
    module = CHDModule({"reference_paths": [reference], "test_mode": True})
    sample = Sample(path=generated, is_video=False, reference_path=generated)
    module.process(sample)
    assert module._references() == [reference.resolve()]
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"bad")
    module.config["reference_paths"] = [reference, bad]
    assert module._references() is None
    module.config.update({"reference_paths": [reference], "reference_dir": str(tmp_path)})
    assert module._references() is None


def test_chd_post_process_routes_dataset_metric_and_resets(tmp_path, monkeypatch):
    from ayase.modules.chd import CHDModule

    reference = _image(tmp_path / "reference.png")
    generated = _image(tmp_path / "generated.png")
    module = CHDModule({"reference_paths": [reference], "test_mode": True})
    module._ml_available = True
    module._backend = "mocked-real-backend"
    module._source_root = tmp_path
    module._model_root = tmp_path
    module.pipeline = _Pipeline()
    module.process(Sample(path=generated, is_video=False))
    monkeypatch.setattr(
        module,
        "_invoke",
        lambda generated, references: {
            "ok": True,
            "real": [1, 0],
            "generated": [0, 1],
            "device": "cpu",
        },
    )
    module.post_process([])
    assert module.pipeline.metrics["chd_1d"] == pytest.approx(1.0)
    assert module.pipeline.metrics["chd_metadata"]["generated_count"] == 1
    assert module.pipeline.metrics["chd_metadata"]["reference_count"] == 1
    assert module.pipeline.metrics["chd_metadata"]["protocol"].startswith("direct RGB")
    assert module.last_metadata["reference_count"] == 1
    module.on_execute()
    assert module._generated_paths == []
    assert module.last_metadata == {}


def test_chd_bad_generated_image_prevents_partial_score(tmp_path, monkeypatch):
    from ayase.modules.chd import CHDModule

    reference = _image(tmp_path / "reference.png")
    generated = _image(tmp_path / "generated.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"bad")
    module = CHDModule({"reference_paths": [reference], "test_mode": True})
    module._ml_available = True
    module.pipeline = _Pipeline()
    module.process(Sample(path=generated, is_video=False))
    module.process(Sample(path=bad, is_video=False))
    monkeypatch.setattr(module, "_invoke", lambda *_: pytest.fail("must not score partial set"))
    module.post_process([])
    assert module.pipeline.metrics == {}


def test_chd_test_mode_never_provisions(monkeypatch):
    from ayase.modules.chd import CHDModule

    module = CHDModule({"test_mode": True})
    monkeypatch.setattr(module, "_resolve_python", lambda: pytest.fail("must not provision"))
    monkeypatch.setattr(module, "_resolve_assets", lambda: pytest.fail("must not download"))
    module.setup()
    assert module._ml_available is False


def test_chd_missing_reference_skips_asset_download(monkeypatch):
    from ayase.modules.chd import CHDModule

    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    CHDModule.set_test_mode(False)
    module = CHDModule()
    monkeypatch.setattr(module, "_resolve_assets", lambda: pytest.fail("must not download"))
    module.setup()
    assert module._backend == "unavailable"
    assert module._ml_available is False


def test_chd_setup_has_no_fake_or_proxy_fallback(monkeypatch, tmp_path):
    from ayase.modules.chd import CHDModule

    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    CHDModule.set_test_mode(False)
    module = CHDModule({"reference_paths": [_image(tmp_path / "reference.png")]})
    monkeypatch.setattr(module, "_resolve_python", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    module.setup()
    assert module._ml_available is False
    assert module._backend == "unavailable"


def test_chd_asset_routing_uses_pinned_snapshot(tmp_path, monkeypatch):
    import ayase.modules.chd as chd

    source = tmp_path / "source"
    for relative in chd._REQUIRED_SOURCE:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (source / ".ayase-source-revision").write_text(chd.SOURCE_REVISION, encoding="ascii")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    weights = model / "model.safetensors"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(chd, "_ensure_source", lambda destination: source)
    monkeypatch.setattr(chd, "download_hf_snapshot", lambda repo_id, models_dir, **kwargs: model)
    monkeypatch.setattr(chd, "MODEL_SHA256", __import__("hashlib").sha256(b"weights").hexdigest())
    module = chd.CHDModule({"models_dir": str(tmp_path)})
    resolved_source, resolved_model = module._resolve_assets()
    assert (resolved_source, resolved_model) == (source, model)


def test_chd_auto_download_false_never_calls_network(tmp_path, monkeypatch):
    import ayase.modules.chd as chd

    monkeypatch.setattr(chd, "_ensure_source", lambda *_: pytest.fail("must not download source"))
    monkeypatch.setattr(
        chd,
        "download_hf_snapshot",
        lambda *_args, **_kwargs: pytest.fail("must not download model"),
    )
    module = chd.CHDModule({"models_dir": str(tmp_path), "auto_download": False})
    with pytest.raises(FileNotFoundError, match="source is not provisioned"):
        module._resolve_assets()
