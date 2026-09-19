"""Tests for the upstream ColorVideoVDP module."""

import hashlib
import sys
import types
from types import SimpleNamespace

import pytest

from tests.modules.conftest import _test_module_basics


def test_cvvdp_basics():
    from ayase.modules.cvvdp import ColorVideoVDPModule

    _test_module_basics(ColorVideoVDPModule, "cvvdp")
    assert ColorVideoVDPModule.metric_field == "cvvdp_score"


def test_cvvdp_ml_transformer_basics():
    from ayase.modules.cvvdp import ColorVideoVDPMLTransformerModule

    _test_module_basics(ColorVideoVDPMLTransformerModule, "cvvdp_ml_transformer")
    assert ColorVideoVDPMLTransformerModule.metric_field == "cvvdp_ml_transformer_score"


def test_cvvdp_ml_transformer_stores_separate_score(image_sample, synthetic_image):
    from ayase.modules.cvvdp import ColorVideoVDPMLTransformerModule

    image_sample.reference_path = synthetic_image
    module = ColorVideoVDPMLTransformerModule()
    module._backend = "cvvdp"
    module._metric = object()
    module._pycvvdp = object()
    module.compute_reference_score = lambda _test, _reference: 8.5

    result = module.process(image_sample)

    assert result.quality_metrics.cvvdp_ml_transformer_score == pytest.approx(8.5)
    assert result.quality_metrics.cvvdp_score is None


def _install_fake_cvvdp_ml_runtime(monkeypatch, checkpoint_path):
    calls = {}
    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda value: f"device:{value}"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)

    fake_metric_module = types.ModuleType("pycvvdp.cvvdp_ml_metric")
    fake_metric_module.hf_hub_download = lambda **_kwargs: "moving-main"
    fake_pycvvdp = types.ModuleType("pycvvdp")
    fake_pycvvdp.__path__ = []
    fake_pycvvdp.cvvdp_ml_metric = fake_metric_module

    def metric_class(**kwargs):
        calls["constructor"] = kwargs
        calls["redirected_path"] = fake_metric_module.hf_hub_download()
        return object()

    fake_pycvvdp.cvvdp_ml_transformer = metric_class
    fake_hub = types.ModuleType("huggingface_hub")

    def download(**kwargs):
        calls["download"] = kwargs
        return str(checkpoint_path)

    fake_hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pycvvdp", fake_pycvvdp)
    monkeypatch.setitem(sys.modules, "pycvvdp.cvvdp_ml_metric", fake_metric_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    return calls


def test_cvvdp_ml_transformer_setup_uses_verified_pin(monkeypatch, tmp_path):
    from ayase.modules.cvvdp import ColorVideoVDPMLTransformerModule

    checkpoint = tmp_path / "cvvdp.ckpt"
    checkpoint.write_bytes(b"verified checkpoint")
    calls = _install_fake_cvvdp_ml_runtime(monkeypatch, checkpoint)
    monkeypatch.setattr(ColorVideoVDPMLTransformerModule, "_global_test_mode", False)
    monkeypatch.setattr(ColorVideoVDPMLTransformerModule, "checkpoint_size", checkpoint.stat().st_size)
    monkeypatch.setattr(
        ColorVideoVDPMLTransformerModule,
        "checkpoint_sha256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )

    module = ColorVideoVDPMLTransformerModule({"device": "cpu"})
    module.setup()

    assert module._backend == "cvvdp"
    assert calls["download"] == {
        "repo_id": "gfxdisp/cvvdp_ml",
        "filename": "cvvdp_ml_transformer/cvvdp.ckpt",
        "revision": "b202a7893f6663a6a46f76f7b06c62d1235bc3ab",
    }
    assert calls["redirected_path"] == str(checkpoint)
    assert calls["constructor"] == {
        "display_name": "standard_fhd",
        "device": "device:cpu",
        "heatmap": None,
        "quiet": True,
        "gpu_mem": None,
    }


@pytest.mark.parametrize("failure", ["size", "sha256"])
def test_cvvdp_ml_transformer_setup_rejects_bad_checkpoint(
    monkeypatch, tmp_path, failure
):
    from ayase.modules.cvvdp import ColorVideoVDPMLTransformerModule

    checkpoint = tmp_path / "cvvdp.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    _install_fake_cvvdp_ml_runtime(monkeypatch, checkpoint)
    monkeypatch.setattr(ColorVideoVDPMLTransformerModule, "_global_test_mode", False)
    if failure == "size":
        monkeypatch.setattr(ColorVideoVDPMLTransformerModule, "checkpoint_size", 1)
    else:
        monkeypatch.setattr(ColorVideoVDPMLTransformerModule, "checkpoint_size", checkpoint.stat().st_size)
        monkeypatch.setattr(ColorVideoVDPMLTransformerModule, "checkpoint_sha256", "0" * 64)

    module = ColorVideoVDPMLTransformerModule({"device": "cpu"})
    module.setup()

    assert module._backend is None
    assert module._metric is None


def test_cvvdp_without_reference_is_graceful(image_sample):
    from ayase.modules.cvvdp import ColorVideoVDPModule

    result = ColorVideoVDPModule().process(image_sample)

    assert result is image_sample
    assert result.quality_metrics is None


def test_cvvdp_reference_file_source_and_score(tmp_path):
    from ayase.modules.cvvdp import ColorVideoVDPModule

    test_path = tmp_path / "test.png"
    reference_path = tmp_path / "reference.png"
    test_path.touch()
    reference_path.touch()
    calls = {}

    def predict(test, reference, **kwargs):
        calls.update(test=test, reference=reference, kwargs=kwargs)
        return 8.25, {}

    module = ColorVideoVDPModule({"display_name": "standard_fhd"})
    module._backend = "cvvdp"
    module._pycvvdp = object()
    module._load_pair = lambda _test, _reference: (
        "test-array",
        "reference-array",
        "HWC",
        0,
    )
    module._metric = SimpleNamespace(predict=predict)

    score = module.compute_reference_score(test_path, reference_path)

    assert score == pytest.approx(8.25)
    assert calls["test"] == "test-array"
    assert calls["reference"] == "reference-array"
    assert calls["kwargs"] == {
        "dim_order": "HWC",
        "frames_per_second": 0,
    }


def test_cvvdp_process_stores_jod_score(image_sample, synthetic_image):
    from ayase.modules.cvvdp import ColorVideoVDPModule

    image_sample.reference_path = synthetic_image
    module = ColorVideoVDPModule()
    module._backend = "cvvdp"
    module._metric = object()
    module._pycvvdp = object()
    module.compute_reference_score = lambda _test, _reference: 7.75

    result = module.process(image_sample)

    assert result is image_sample
    assert result.quality_metrics.cvvdp_score == pytest.approx(7.75)


def test_cvvdp_loads_image_pair(synthetic_image):
    from ayase.modules.cvvdp import ColorVideoVDPModule

    test, reference, dim_order, fps = ColorVideoVDPModule()._load_pair(
        synthetic_image, synthetic_image
    )

    assert test.shape == reference.shape
    assert test.ndim == 3
    assert dim_order == "HWC"
    assert fps == 0


def test_cvvdp_loads_complete_video_pair(synthetic_video):
    from ayase.modules.cvvdp import ColorVideoVDPModule

    test, reference, dim_order, fps = ColorVideoVDPModule()._load_pair(
        synthetic_video, synthetic_video
    )

    assert test.shape == reference.shape
    assert test.shape[0] == 64
    assert dim_order == "FHWC"
    assert fps == pytest.approx(30.0)


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), 10.1])
def test_cvvdp_rejects_invalid_scores(tmp_path, bad_score):
    from ayase.modules.cvvdp import ColorVideoVDPModule

    test_path = tmp_path / "test.png"
    reference_path = tmp_path / "reference.png"
    module = ColorVideoVDPModule()
    module._backend = "cvvdp"
    module._pycvvdp = object()
    module._load_pair = lambda *_args: (object(), object(), "HWC", 0)
    module._metric = SimpleNamespace(
        predict=lambda *_args, **_kwargs: (bad_score, {})
    )

    assert module.compute_reference_score(test_path, reference_path) is None


def test_cvvdp_declares_reference_package():
    from ayase.modules.cvvdp import ColorVideoVDPModule

    assert ColorVideoVDPModule.models == [
        {
            "id": "cvvdp",
            "type": "pip_package",
            "install": "pip install 'cvvdp>=0.5.6,<0.6'",
            "task": "ColorVideoVDP image/video perceptual metric",
            "auto_download": False,
            "notes": "MIT; calibration and display-model data ship in the package",
        }
    ]


def test_cvvdp_reference_package_is_rendered_in_model_catalog():
    from ayase.models_doc import generate_models_doc

    document = generate_models_doc(fetch_licenses=False)

    assert "## pip Packages" in document
    assert "### `cvvdp`" in document
    assert "**Used by**: `cvvdp`" in document
