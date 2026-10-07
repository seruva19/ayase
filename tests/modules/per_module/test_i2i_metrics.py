"""Tests for deterministic and learned image-to-image metric modules."""

import cv2
import numpy as np
import pytest

from ayase.models import Sample
from ayase.pipeline import Pipeline
from tests.modules.conftest import _test_module_basics


def _write_image(path, offset=0):
    yy, xx = np.mgrid[:96, :96]
    image = np.stack(
        (
            (xx * 2 + offset) % 256,
            (yy * 2 + offset) % 256,
            ((xx + yy) * 2 + offset) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    cv2.rectangle(image, (24, 24), (72, 72), (20 + offset, 180, 240), 3)
    assert cv2.imwrite(str(path), image)
    return path


def test_i2i_module_basics():
    from ayase.modules.i2i_fidelity import I2IFidelityModule
    from ayase.modules.i2i_learned import I2ILearnedModule

    _test_module_basics(I2IFidelityModule, "i2i_fidelity")
    _test_module_basics(I2ILearnedModule, "i2i_learned")


def test_i2i_fidelity_identical_pair_populates_compact_fields(tmp_path):
    from ayase.modules.i2i_fidelity import ALL_FIELDS, I2IFidelityModule

    image = _write_image(tmp_path / "image.png")
    sample = Sample(path=image, is_video=False, reference_path=image)
    result = I2IFidelityModule().process(sample)
    metrics = result.quality_metrics

    assert result is sample
    assert metrics is not None
    assert len(ALL_FIELDS) == 3
    values = {field: getattr(metrics, field) for field in ALL_FIELDS}
    assert all(value is not None and np.isfinite(value) for value in values.values())
    assert values["i2i_mse"] == pytest.approx(0.0)
    assert values["i2i_mae"] == pytest.approx(0.0)
    assert values["i2i_gradient_similarity_mean"] == pytest.approx(1.0)


def test_i2i_fidelity_detects_changed_pair(tmp_path):
    from ayase.modules.i2i_fidelity import I2IFidelityModule

    reference = _write_image(tmp_path / "reference.png")
    generated = _write_image(tmp_path / "generated.png", offset=30)
    sample = Sample(path=generated, is_video=False, reference_path=reference)
    metrics = I2IFidelityModule().process(sample).quality_metrics

    assert metrics.i2i_mse > 0
    assert metrics.i2i_mae > 0
    assert metrics.i2i_gradient_similarity_mean < 1
    assert metrics.i2i_dinov2_cls_similarity is None


def test_i2i_fidelity_without_reference_is_noop(tmp_path):
    from ayase.modules.i2i_fidelity import I2IFidelityModule

    image = _write_image(tmp_path / "image.png")
    sample = Sample(path=image, is_video=False)
    assert I2IFidelityModule().process(sample) is sample
    assert sample.quality_metrics is None


def test_i2i_learned_without_setup_degrades_gracefully(tmp_path):
    from ayase.modules.i2i_learned import I2ILearnedModule

    image = _write_image(tmp_path / "image.png")
    sample = Sample(path=image, is_video=False, reference_path=image)
    assert I2ILearnedModule().process(sample) is sample
    assert sample.quality_metrics is None


def test_i2i_configurable_and_preprocessed_fields_are_adapted():
    from ayase.modules.i2i_learned import I2ILearnedModule

    provenance = I2ILearnedModule.field_provenance()
    assert provenance["i2i_dinov2_cls_similarity"] == "adapted"
    assert provenance["i2i_clip_similarity"] == "adapted"
    assert provenance["i2i_lpips_alex"] == "adapted"
    assert "configurable" in I2ILearnedModule.deviations["i2i_clip_similarity"]
    assert "paired-image" in I2ILearnedModule.deviations["i2i_clip_similarity"]
    assert "256×256" in I2ILearnedModule.deviations["i2i_lpips_alex"]
    assert "does not prescribe" in I2ILearnedModule.deviations["i2i_lpips_alex"]


def test_i2i_learned_requires_adapted_opt_in():
    from ayase.modules.i2i_learned import I2ILearnedModule

    default_module = I2ILearnedModule()
    default_pipeline = Pipeline([default_module])
    assert "adapted" not in default_pipeline._module_allowed_provenance[id(default_module)]
    assert default_pipeline._provenance_excluded[default_module.name] == "adapted+own"

    opted_in_module = I2ILearnedModule()
    opted_in_pipeline = Pipeline([opted_in_module], allow_provenance=["adapted"])
    assert "adapted" in opted_in_pipeline._module_allowed_provenance[id(opted_in_module)]
    assert opted_in_module.name not in opted_in_pipeline._provenance_excluded


def test_i2i_learned_metadata_declares_image_reference_input():
    from ayase.modules.i2i_learned import I2ILearnedModule

    assert I2ILearnedModule.get_metadata()["input_type"] == "img +ref"
