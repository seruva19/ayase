"""Tests for the upstream ColorVideoVDP module."""

from types import SimpleNamespace

import pytest

from tests.modules.conftest import _test_module_basics


def test_cvvdp_basics():
    from ayase.modules.cvvdp import ColorVideoVDPModule

    _test_module_basics(ColorVideoVDPModule, "cvvdp")
    assert ColorVideoVDPModule.metric_field == "cvvdp_score"


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
